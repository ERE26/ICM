"""Reservas de balance del sistema espaniol, desde la plataforma de ENTSO-E.

Por que este modulo existe y por que no lo hay para ESIOS
---------------------------------------------------------
El modulo de almacenamiento llego a una conclusion que depende de un dato que
el proyecto no tiene: **el arbitraje paga la operacion pero la banda de
regulacion secundaria paga la inversion**. Con la banda cubriendo el 25 % de
las horas el capex de equilibrio pasa de 71 a 232 EUR/kWh, que es la
diferencia entre un activo inviable y uno financiable. Sin datos de asignacion
de banda, esa afirmacion es una sensibilidad y no una valoracion.

La fuente natural en Espania es ESIOS, de Red Electrica. **No es alcanzable
desde este entorno**: todo el dominio de REE --``esios.ree.es``,
``apidatos.ree.es``, ``www.ree.es``, ``sistemaelectrico-ree.es``-- responde
403 de Incapsula, comprobado host por host, con y sin ``User-Agent``
descriptivo. No es la politica de red del contenedor, que esta abierta, ni la
falta de credencial: es proteccion anti-bot del origen. Por eso no hay un
``icm.datos.esios``: escribirlo seria escribir codigo que no puede funcionar.

ENTSO-E si es alcanzable. Publica las mismas magnitudes para toda Europa bajo
los articulos 17.1.B, C y D del reglamento de transparencia, y responde con el
protocolo correcto --un ``Acknowledgement_MarketDocument`` con
``<code>999</code> Authentication failed``-- en vez de un muro anti-bot. Solo
falta el token.

Como conseguir el token
-----------------------
Es gratuito y no lo puede pedir este codigo, porque va ligado a una persona:

1. Crear una cuenta en https://transparency.entsoe.eu.
2. Escribir a ``transparency@entsoe.eu`` con el asunto *Restful API access*
   desde la direccion de la cuenta, pidiendo acceso a la API.
3. Cuando lo activen, el token sale en *My Account Settings*.

Despues, ponerlo en la variable de entorno ``ENTSOE_TOKEN`` y todo lo de aqui
funciona sin tocar nada.

Sobre que esta probado y que no
-------------------------------
Esta distincion importa mas de lo normal, porque sin token no se ha podido
ejecutar una descarga real ni una sola vez:

* **``leer_documento`` esta probado.** El formato de ENTSO-E es un esquema
  publicado y estable (``iec62325.351``), la estructura
  ``TimeSeries -> Period -> Point`` no depende del indicador, y las pruebas lo
  ejercitan contra documentos construidos segun el esquema, incluidos los
  casos que rompen a un lector ingenuo: varias series en un documento, huecos
  en las posiciones y resoluciones distintas de la horaria.
* **Los parametros de consulta no estan verificados.** Que
  ``documentType="A81"`` con ``processType="A51"`` sea exactamente la reserva
  secundaria contratada sale de la documentacion del protocolo y **no de una
  respuesta real**. Por eso :func:`consultar` acepta cualquier combinacion y
  las funciones de conveniencia dicen en su nombre y en su docstring que son
  una conjetura razonada. La primera descarga con token puede tener que
  ajustarlas, y ajustarlas es una linea.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd

from icm.comun import ErrorDatos, ErrorDescarga


#: Punto de entrada de la API REST de la plataforma de transparencia.
URL_API = "https://web-api.tp.entsoe.eu/api"

#: Variable de entorno donde se busca el token si no se pasa a mano.
VARIABLE_TOKEN = "ENTSOE_TOKEN"

#: Codigo EIC de la zona de oferta espaniola.
ZONA_ESPANA = "10YES-REE------0"

#: Codigos EIC de las zonas vecinas, que hacen falta para los intercambios.
ZONAS = {
    "espania": ZONA_ESPANA,
    "portugal": "10YPT-REN------W",
    "francia": "10YFR-RTE------C",
}

#: Tipos de proceso del protocolo. El que interesa aqui es el de reserva
#: secundaria, que en la nomenclatura europea es aFRR --reserva de
#: restauracion de la frecuencia con activacion automatica--. Lo que en
#: Espania se llama "banda de regulacion secundaria" es la capacidad de aFRR
#: contratada, no su activacion.
PROCESOS = {
    "afrr": "A51",
    "mfrr": "A56",
    "reemplazo": "A46",
}

#: Duracion en minutos de cada resolucion del esquema, para reconstruir las
#: marcas de tiempo a partir de la posicion de cada punto.
RESOLUCIONES = {
    "PT15M": 15,
    "PT30M": 30,
    "PT60M": 60,
    "PT1H": 60,
    "P1D": 1440,
}

_ESPACIO = "{*}"


def token_disponible(token: str | None = None) -> bool:
    """Si hay token, sin revelarlo ni intentar la descarga.

    Sirve para que un cuaderno decida si ensiania la seccion de banda o un
    aviso, sin provocar un error de red ni dejar el token en la salida.
    """
    return bool(token or os.environ.get(VARIABLE_TOKEN, "").strip())


def consultar(
    parametros: dict[str, str],
    token: str | None = None,
    tiempo_maximo: float = 60.0,
) -> str:
    """Lanza una consulta a la API y devuelve el XML en crudo.

    Es deliberadamente generica: recibe los parametros del protocolo tal cual
    y no los interpreta. La razon es que sin token no se ha podido comprobar
    ni una sola combinacion contra una respuesta real, asi que imponer aqui un
    vocabulario cerrado seria fijar en el codigo una conjetura.

    Traduce los dos modos de fallo que el protocolo distingue y que conviene
    no confundir: un ``Acknowledgement_MarketDocument`` con razon 999 es que
    falta o no vale el token, y con otra razon es que la consulta es valida
    pero no hay datos para ella.
    """
    if not token:
        token = os.environ.get(VARIABLE_TOKEN, "").strip()
    if not token:
        raise ErrorDescarga(
            f"No hay token de ENTSO-E. Se busca en la variable de entorno "
            f"'{VARIABLE_TOKEN}' o se pasa en el argumento 'token'. Es gratuito: "
            f"cuenta en https://transparency.entsoe.eu y correo a "
            f"transparency@entsoe.eu pidiendo acceso a la API REST."
        )

    import requests  # import diferido: no hace falta para leer un XML de disco

    consulta = dict(parametros)
    consulta["securityToken"] = token
    try:
        respuesta = requests.get(URL_API, params=consulta, timeout=tiempo_maximo)
    except Exception as error:  # noqa: BLE001 -- se reetiqueta y se relanza
        raise ErrorDescarga(f"No se pudo consultar a ENTSO-E: {error}") from error

    texto = respuesta.text
    if respuesta.status_code == 401 or "Authentication failed" in texto:
        raise ErrorDescarga(
            "ENTSO-E rechaza el token (razon 999, 'Authentication failed'). El "
            "token existe pero puede no tener activado el acceso a la API REST, "
            "que se pide por correo aparte de crear la cuenta."
        )
    if respuesta.status_code != 200:
        raise ErrorDescarga(
            f"ENTSO-E respondio {respuesta.status_code} a la consulta "
            f"{ {k: v for k, v in parametros.items()} }."
        )
    if "Acknowledgement_MarketDocument" in texto:
        raise ErrorDescarga(
            f"ENTSO-E acepto la consulta pero no devuelve datos: "
            f"{_razon_del_acuse(texto)}. Suele ser que la combinacion de tipo de "
            f"documento, proceso y zona no se publica para ese periodo."
        )
    return texto


def _razon_del_acuse(xml: str) -> str:
    """Saca el codigo y el texto de un acuse de recibo negativo."""
    try:
        raiz = ET.fromstring(xml)
    except ET.ParseError:
        return "respuesta ilegible"
    codigo = raiz.find(f".//{_ESPACIO}code")
    texto = raiz.find(f".//{_ESPACIO}text")
    partes = [p.text for p in (codigo, texto) if p is not None and p.text]
    return ", ".join(partes) if partes else "sin razon indicada"


def leer_documento(xml: str, valor: str = "quantity") -> pd.DataFrame:
    """Convierte un documento de mercado de ENTSO-E en una tabla.

    La estructura es siempre la misma, sea cual sea el indicador: el documento
    lleva varias ``TimeSeries``, cada una con un ``Period`` que tiene su
    intervalo y su resolucion, y dentro los ``Point`` numerados por posicion.

    Dos cosas que un lector ingenuo hace mal y que aqui estan cuidadas:

    * **La posicion no es el indice de la fila.** Empieza en 1 y puede saltar:
      el esquema permite omitir un punto cuando repite el valor del anterior.
      Tomar el orden de aparicion desplazaria toda la serie a partir del
      primer hueco. Aqui la marca de tiempo se calcula del inicio del periodo
      mas ``(posicion - 1)`` veces la resolucion.
    * **La resolucion no siempre es horaria.** Las reservas se publican en
      bloques de 15 minutos, de una hora o de un dia segun el pais y el
      producto, y a veces cambian de un anio a otro.

    Parametros
    ----------
    xml:
        El documento, tal como lo devuelve :func:`consultar`.
    valor:
        ``"quantity"`` para volumenes o ``"price.amount"`` para precios, que
        son los dos nombres que usa el esquema segun el indicador.

    Devuelve
    --------
    Un DataFrame con ``momento``, ``valor``, ``serie`` --el identificador de
    la ``TimeSeries``, que distingue subir de bajar-- y ``direccion`` cuando
    el documento la trae.
    """
    try:
        raiz = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ErrorDatos(f"El documento de ENTSO-E no es XML valido: {error}") from error
    if "Acknowledgement" in raiz.tag:
        raise ErrorDatos(
            f"Esto es un acuse de recibo y no un documento con datos: "
            f"{_razon_del_acuse(xml)}."
        )

    filas: list[dict] = []
    for serie in raiz.findall(f".//{_ESPACIO}TimeSeries"):
        etiqueta = _texto(serie, "mRID") or str(len(filas))
        direccion = _texto(serie, "flowDirection.direction")
        for periodo in serie.findall(f"{_ESPACIO}Period"):
            filas.extend(_puntos_del_periodo(periodo, etiqueta, direccion, valor))

    if not filas:
        raise ErrorDatos(
            f"El documento no tiene ningun punto con '{valor}'. Si se pidieron "
            f"precios, el nombre en el esquema es 'price.amount'; si volumenes, "
            f"'quantity'."
        )
    tabla = pd.DataFrame(filas).sort_values(["serie", "momento"]).reset_index(drop=True)
    return tabla


def _puntos_del_periodo(periodo, etiqueta: str, direccion: str | None, valor: str) -> list[dict]:
    """Desarrolla un ``Period`` en una fila por punto, con su marca de tiempo."""
    resolucion = _texto(periodo, "resolution")
    minutos = RESOLUCIONES.get((resolucion or "").strip().upper())
    if minutos is None:
        raise ErrorDatos(
            f"Resolucion '{resolucion}' no reconocida. Se entienden "
            f"{sorted(RESOLUCIONES)}."
        )
    inicio_texto = _texto(periodo, "start")
    if not inicio_texto:
        raise ErrorDatos("Un periodo del documento no declara su instante de inicio.")
    inicio = pd.Timestamp(inicio_texto)

    filas = []
    for punto in periodo.findall(f"{_ESPACIO}Point"):
        posicion = _texto(punto, "position")
        crudo = _texto(punto, valor)
        if posicion is None or crudo is None:
            continue
        # La posicion empieza en 1 y puede saltar; nunca es el numero de fila.
        momento = inicio + pd.Timedelta(minutes=minutos * (int(posicion) - 1))
        fila = {"momento": momento, "valor": float(crudo), "serie": etiqueta}
        if direccion:
            fila["direccion"] = direccion
        filas.append(fila)
    return filas


def _texto(nodo, etiqueta: str) -> str | None:
    """Texto de un hijo directo o descendiente, sin depender del espacio de nombres.

    El esquema de ENTSO-E declara un espacio de nombres distinto por tipo de
    documento y por version, asi que buscar por nombre cualificado obligaria a
    mantener una lista que caduca. El comodin la evita.
    """
    hijo = nodo.find(f"{_ESPACIO}{etiqueta}")
    if hijo is None:
        hijo = nodo.find(f".//{_ESPACIO}{etiqueta}")
    return hijo.text.strip() if hijo is not None and hijo.text else None


def _marca(momento) -> str:
    """Pasa una fecha al formato ``aaaammddhhmm`` que exige el protocolo."""
    instante = pd.Timestamp(momento)
    if instante.tzinfo is not None:
        instante = instante.tz_convert("UTC").tz_localize(None)
    return instante.strftime("%Y%m%d%H%M")


def parametros_banda_secundaria(
    desde,
    hasta,
    zona: str = ZONA_ESPANA,
    proceso: str = "afrr",
    precios: bool = False,
) -> dict[str, str]:
    """Parametros de consulta para la banda de regulacion secundaria.

    **Conjetura razonada, no verificada.** Sale de la documentacion del
    protocolo y no de una respuesta real, porque sin token no se ha podido
    ejecutar ninguna. Se expone como funcion aparte, en vez de esconderla
    dentro de la descarga, justamente para que sea facil de corregir y de
    inspeccionar cuando alguien tenga token: basta con imprimirla y probarla
    contra la API.

    ``A81`` es el volumen de reservas contratadas del articulo 17.1.B y C, y
    ``A89`` el precio de las reservas adquiridas del 17.1.D. ``A51`` es aFRR,
    que es el nombre europeo de lo que en Espania se llama regulacion
    secundaria.
    """
    if proceso not in PROCESOS:
        raise ErrorDatos(f"Proceso '{proceso}' desconocido; hay {sorted(PROCESOS)}.")
    return {
        "documentType": "A89" if precios else "A81",
        "processType": PROCESOS[proceso],
        "controlArea_Domain": zona,
        "periodStart": _marca(desde),
        "periodEnd": _marca(hasta),
    }


def descargar_banda_secundaria(
    desde,
    hasta,
    zona: str = ZONA_ESPANA,
    precios: bool = False,
    token: str | None = None,
) -> pd.DataFrame:
    """Descarga y lee la banda secundaria contratada de un periodo.

    Es la union de :func:`parametros_banda_secundaria`, :func:`consultar` y
    :func:`leer_documento`, con la salvedad de los parametros que explica la
    primera. La API limita cada consulta a un anio, asi que para series largas
    hay que trocear.
    """
    xml = consultar(
        parametros_banda_secundaria(desde, hasta, zona, precios=precios),
        token=token,
    )
    return leer_documento(xml, valor="price.amount" if precios else "quantity")


def horas_con_banda(asignacion: pd.DataFrame, potencia: float) -> float:
    """Fraccion de horas en que un activo de ``potencia`` MW tendria banda.

    Es la pieza que le falta al VAN. ``van_almacenamiento`` pide el ingreso
    adicional por MW y anio, y para calcularlo hace falta saber en cuantas
    horas se asigna banda y a que precio; con la asignacion del sistema y una
    hipotesis de cuota se obtiene lo primero.

    La hipotesis que hace esta funcion es la mas simple defendible: el activo
    entra en la banda siempre que el sistema contrate al menos su potencia, lo
    que es una **cota superior**, porque compite con el resto del parque. Se
    devuelve la fraccion y no el ingreso para que la cuota de mercado la ponga
    quien valore y quede a la vista.
    """
    if "valor" not in asignacion.columns:
        raise ErrorDatos("Se espera la salida de 'leer_documento', con columna 'valor'.")
    if float(potencia) <= 0.0:
        raise ErrorDatos("La potencia tiene que ser positiva.")
    if asignacion.empty:
        raise ErrorDatos("La asignacion esta vacia: no hay horas sobre las que contar.")
    return float((asignacion["valor"] >= float(potencia)).mean())
