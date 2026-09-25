"""Simulador de precios de energia con estructura predecible.

Este entorno no tiene salida a internet, asi que los modelos de
:mod:`icm.ml` necesitan un banco de pruebas propio. Un paseo aleatorio no
sirve: si el proceso generador no tiene senal, ningun modelo puede batir a la
referencia ingenua y lo unico que se demuestra es que nada funciona. Aqui el
precio depende de variables fisicas observables (el frio que hace, el viento
que sopla, el gas que queda almacenado) con sensibilidades conocidas, de modo
que se puede contrastar lo que el modelo aprende contra la verdad con la que
se genero la serie.

Que reproduce, y por que importa
--------------------------------
* **Estacionalidad anual**: la demanda de gas y electricidad sigue al
  termometro. Amplitud alta en gas, casi nula en crudo (que es un mercado
  global y almacenable) y nula en las acciones.
* **Reversion a la media** tipo Ornstein-Uhlenbeck: un shock de oferta se
  corrige cuando entra produccion o se sustituye combustible. Una accion, en
  cambio, es un paseo aleatorio con deriva y no revierte.
* **Grados dia**: el sector no mira la temperatura, mira los grados dia de
  calefaccion (HDD) y refrigeracion (CDD). Es la variable con la que se
  contrata y se cubre la demanda, y aqui es la senal principal.
* **Saltos asimetricos al alza**: cuando el sistema se tensiona el precio se
  dispara, no se hunde. Las colas de la energia son derechas, al reves que en
  la renta variable.
* **Volatilidad agrupada** (GARCH) y un **factor comun** entre activos: las
  crisis energeticas mueven todo el complejo a la vez.

Cobertura de datos reales
-------------------------
Los activos con clave ``yahoo`` se pueden descargar cuando hay red. **TTF
(Paises Bajos), MIBGAS (Espana) y OMIE (pool electrico espanol) no tienen
feed gratuito en Yahoo Finance**: sus curvas se publican en las webs de ICE
Endex, MIBGAS y OMIE, y hay que descargarlas a mano y cargarlas con
:func:`icm.datos.cargar_precios_fichero`. :func:`activos_sin_yahoo` devuelve
justo esa lista.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos, como_serie
from icm.datos.proveedor import ProveedorDatos, normalizar_tickers, validar_marco_precios

# La dinamica de la volatilidad es la misma que la del simulador general: se
# reutiliza en lugar de duplicarla para que ambos mundos compartan un unico
# GARCH(1,1) de varianza incondicional unitaria.
from icm.datos.simulacion import _factor_volatilidad_garch as factor_volatilidad_garch

#: Duracion del ciclo estacional en dias naturales (no habiles: el frio no
#: sabe de fines de semana).
DIAS_ANIO = 365.25

#: Dia del anio en que el invierno pesa mas: maximo de HDD y de precio.
_PICO_INVIERNO = 15.0

#: Dia del anio en que el almacenamiento de gas esta mas lleno, al terminar la
#: campania de inyeccion y antes de empezar el invierno.
_PICO_ALMACENAMIENTO = 300.0

_GARCH_ALFA = 0.12
_GARCH_BETA = 0.85

#: Habilidad por defecto de las previsiones a 24 horas: correlacion entre lo
#: previsto y lo que acaba ocurriendo. Vease :func:`prevision_exogena`.
HABILIDAD_PREVISION = 0.90

#: Exogenas de las que se publica prevision. El almacenamiento no esta: se
#: conoce con rezago semanal, no por adelantado.
EXOGENAS_PREVISTAS = ("hdd", "cdd", "viento")

CATALOGO: dict[str, dict] = {
    "WTI": {
        "nombre": "Crudo West Texas Intermediate",
        "tipo": "crudo",
        "mercado": "NYMEX",
        "yahoo": "CL=F",
        "unidad": "USD/barril",
        "estacionalidad": 0.03,
    },
    "BRENT": {
        "nombre": "Crudo Brent del Mar del Norte",
        "tipo": "crudo",
        "mercado": "ICE Futures Europe",
        "yahoo": "BZ=F",
        "unidad": "USD/barril",
        "estacionalidad": 0.03,
    },
    "HENRY_HUB": {
        "nombre": "Gas natural Henry Hub",
        "tipo": "gas",
        "mercado": "NYMEX",
        "yahoo": "NG=F",
        "unidad": "USD/MMBtu",
        "estacionalidad": 0.22,
    },
    "TTF": {
        "nombre": "Gas natural Title Transfer Facility",
        "tipo": "gas",
        "mercado": "ICE Endex (Paises Bajos)",
        "yahoo": None,
        "unidad": "EUR/MWh",
        "estacionalidad": 0.26,
    },
    "MIBGAS": {
        "nombre": "Gas natural MIBGAS diario",
        "tipo": "gas",
        "mercado": "MIBGAS (Espana)",
        "yahoo": None,
        "unidad": "EUR/MWh",
        "estacionalidad": 0.22,
    },
    "OMIE": {
        "nombre": "Electricidad mercado diario peninsular",
        "tipo": "electricidad",
        "mercado": "OMIE (Espana)",
        "yahoo": None,
        "unidad": "EUR/MWh",
        "estacionalidad": 0.15,
    },
    "XLE": {
        "nombre": "Energy Select Sector SPDR",
        "tipo": "accion",
        "mercado": "NYSE Arca",
        "yahoo": "XLE",
        "unidad": "USD/participacion",
        "estacionalidad": 0.0,
    },
    "XLU": {
        "nombre": "Utilities Select Sector SPDR",
        "tipo": "accion",
        "mercado": "NYSE Arca",
        "yahoo": "XLU",
        "unidad": "USD/participacion",
        "estacionalidad": 0.0,
    },
}

# Perfil dinamico por tipo de activo. Las sensibilidades son elasticidades del
# log-precio al nivel de la exogena, asi que coinciden con el efecto de una
# variacion unitaria de esa exogena sobre el rendimiento del dia: es lo que
# una regresion de rendimientos sobre diferencias deberia recuperar.
_PERFIL_TIPO: dict[str, dict] = {
    "crudo": {
        "reversion": 0.010,
        "volatilidad": 0.34,
        "sensibilidad_hdd": 0.0005,
        "sensibilidad_cdd": 0.0006,
        "sensibilidad_viento": 0.0,
        "sensibilidad_almacenamiento": 0.0,
        "carga_factor": 0.78,
        "propension_saltos": 1.0,
        "tamanio_salto": 0.04,
        "deriva_anual": 0.0,
    },
    "gas": {
        "reversion": 0.030,
        "volatilidad": 0.55,
        "sensibilidad_hdd": 0.0100,
        "sensibilidad_cdd": 0.0025,
        "sensibilidad_viento": -0.06,
        "sensibilidad_almacenamiento": -0.0018,
        "carga_factor": 0.45,
        "propension_saltos": 2.0,
        "tamanio_salto": 0.07,
        "deriva_anual": 0.0,
    },
    "electricidad": {
        "reversion": 0.100,
        "volatilidad": 0.75,
        "sensibilidad_hdd": 0.0090,
        "sensibilidad_cdd": 0.0070,
        "sensibilidad_viento": -0.25,
        "sensibilidad_almacenamiento": -0.0004,
        "carga_factor": 0.30,
        "propension_saltos": 3.0,
        "tamanio_salto": 0.11,
        "deriva_anual": 0.0,
    },
    "accion": {
        "reversion": 0.0,
        "volatilidad": 0.24,
        "sensibilidad_hdd": 0.0002,
        "sensibilidad_cdd": 0.0001,
        "sensibilidad_viento": 0.0,
        "sensibilidad_almacenamiento": 0.0,
        "carga_factor": 0.70,
        "propension_saltos": 0.3,
        "tamanio_salto": 0.04,
        "deriva_anual": 0.07,
    },
}

# Lo que distingue a cada activo dentro de su tipo. "nivel" es el precio medio
# en la unidad de cotizacion del catalogo.
_AJUSTE_ACTIVO: dict[str, dict] = {
    "WTI": {"nivel": 72.0},
    "BRENT": {"nivel": 77.0, "carga_factor": 0.82},
    "HENRY_HUB": {"nivel": 3.2},
    "TTF": {
        "nivel": 32.0,
        "volatilidad": 0.70,
        "sensibilidad_hdd": 0.0125,
        "sensibilidad_almacenamiento": -0.0022,
    },
    "MIBGAS": {"nivel": 30.0, "volatilidad": 0.65, "sensibilidad_hdd": 0.0110},
    "OMIE": {"nivel": 85.0},
    "XLE": {"nivel": 85.0, "volatilidad": 0.28},
    "XLU": {"nivel": 68.0, "volatilidad": 0.17, "deriva_anual": 0.05},
}


def activos_con_yahoo() -> list[str]:
    """Activos del catalogo descargables de Yahoo Finance."""
    return [clave for clave, ficha in CATALOGO.items() if ficha["yahoo"] is not None]


def activos_sin_yahoo() -> list[str]:
    """Activos que hay que cargar de fichero: no hay feed gratuito de ellos.

    Son las referencias europeas (TTF, MIBGAS, OMIE). Se descargan de la web
    del mercado y se leen con :func:`icm.datos.cargar_precios_fichero`.
    """
    return [clave for clave, ficha in CATALOGO.items() if ficha["yahoo"] is None]


def _ciclo_anual(dia_del_anio: np.ndarray, pico: float) -> np.ndarray:
    """Coseno anual que vale 1 en el dia ``pico`` y -1 medio anio despues."""
    return np.cos(2.0 * np.pi * (dia_del_anio - pico) / DIAS_ANIO)


def _ar1(
    n_periodos: int,
    persistencia: float,
    desviacion: float,
    generador: np.random.Generator,
) -> np.ndarray:
    """Proceso AR(1) de media cero cuya desviacion tipica *incondicional* es ``desviacion``.

    La innovacion se escala por ``sqrt(1 - persistencia**2)`` y el primer valor
    se extrae ya de la distribucion estacionaria. Asi el parametro significa lo
    que su nombre dice, la serie no arrastra un transitorio inicial y cambiar
    la persistencia no cambia de paso la escala del proceso.
    """
    valores = np.empty(n_periodos)
    choques = generador.standard_normal(n_periodos) * desviacion
    valores[0] = choques[0]
    escala = np.sqrt(1.0 - persistencia**2)
    for t in range(1, n_periodos):
        valores[t] = persistencia * valores[t - 1] + escala * choques[t]
    return valores


def _temperatura(
    dia_del_anio: np.ndarray,
    generador: np.random.Generator,
    media_anual: float,
    amplitud: float,
    persistencia: float,
    ruido: float,
) -> np.ndarray:
    """Nucleo de :func:`simular_temperatura`, con el generador ya construido.

    Se separa de la funcion publica para que :func:`simular_energia` pueda
    extraer las exogenas y los choques de precio del *mismo* generador. Con dos
    generadores distintos sembrados con la misma semilla, ambos consumirian la
    misma secuencia de normales y el tiempo quedaria correlacionado con los
    choques de mercado, que es justo la trampa que arruinaria el ejercicio.
    """
    estacional = media_anual - amplitud * _ciclo_anual(dia_del_anio, _PICO_INVIERNO)
    return estacional + _ar1(len(dia_del_anio), persistencia, ruido, generador)


def simular_temperatura(
    n_periodos: int,
    inicio: str | pd.Timestamp = "2018-01-01",
    semilla: int | None = 42,
    media_anual: float = 14.0,
    amplitud: float = 9.0,
    persistencia: float = 0.85,
    ruido: float = 2.5,
) -> pd.Series:
    """Temperatura diaria en grados Celsius: ciclo anual mas un AR(1) alrededor.

    El ciclo recoge las estaciones (minimo a mediados de enero, maximo a
    mediados de julio, hemisferio norte) y el AR(1) recoge que el tiempo es
    persistente: a un dia frio le sigue otro dia frio. Esa persistencia es la
    que hace que la temperatura sea *predecible a corto plazo* y, por tanto,
    utilizable como variable explicativa.

    Parametros
    ----------
    amplitud:
        Semiamplitud del ciclo anual: la diferencia entre el verano y el
        invierno medios es el doble.
    persistencia:
        Coeficiente autorregresivo de la desviacion respecto al ciclo.
    ruido:
        Desviacion tipica *incondicional* de esa desviacion, en grados.
    """
    if n_periodos < 2:
        raise ErrorDatos("Hacen falta al menos 2 periodos para simular una temperatura.")
    if not 0.0 <= persistencia < 1.0:
        raise ErrorDatos(
            f"La persistencia debe estar en [0, 1) para que la temperatura sea "
            f"estacionaria; se ha recibido {persistencia}."
        )
    if amplitud < 0.0 or ruido < 0.0:
        raise ErrorDatos("La amplitud estacional y el ruido no pueden ser negativos.")

    fechas = pd.bdate_range(start=inicio, periods=n_periodos, name="fecha")
    valores = _temperatura(
        fechas.dayofyear.to_numpy(dtype=float),
        np.random.default_rng(semilla),
        media_anual,
        amplitud,
        persistencia,
        ruido,
    )
    return pd.Series(valores, index=fechas, name="temperatura")


def grados_dia(
    temperatura,
    base_calefaccion: float = 15.5,
    base_refrigeracion: float = 22.0,
) -> pd.DataFrame:
    """Convierte temperatura en grados dia de calefaccion (HDD) y refrigeracion (CDD).

    Es la transformacion con la que el sector traduce meteorologia en demanda:
    por debajo de la base de calefaccion se enciende la caldera y por encima de
    la de refrigeracion el aire acondicionado, mientras que en la banda
    intermedia la temperatura no mueve el consumo. La no linealidad importa:
    pasar de 20 a 18 grados no cambia nada y pasar de 2 a 0 dispara la demanda,
    algo que una regresion sobre la temperatura en bruto no puede captar.

    Las bases son las convencionales del sector (15,5 C y 22 C), heredadas de
    los 65 F y 72 F estadounidenses.
    """
    serie = como_serie(temperatura, "temperatura")
    if base_calefaccion > base_refrigeracion:
        raise ErrorDatos(
            f"La base de calefaccion ({base_calefaccion}) no puede superar a la de "
            f"refrigeracion ({base_refrigeracion}): dejaria dias contando a la vez "
            "como frios y como calurosos."
        )
    return pd.DataFrame(
        {
            "hdd": np.maximum(base_calefaccion - serie, 0.0),
            "cdd": np.maximum(serie - base_refrigeracion, 0.0),
        },
        index=serie.index,
    )


@dataclass
class EscenarioEnergia:
    """Una realizacion completa del mercado simulado.

    Atributos
    ---------
    precios:
        Precios diarios, indice de fechas habiles y una columna por activo.
    exogenas:
        Variables observables que mueven el precio: ``temperatura``, ``hdd``,
        ``cdd``, ``viento`` y ``almacenamiento``, mas las columnas
        ``prevision_*`` descritas en :func:`prevision_exogena`. Comparten
        indice con los precios, de modo que se pueden concatenar sin
        reindexar.
    parametros:
        La verdad con la que se genero el escenario. Sirve para lo unico que
        no permite un mercado real: comparar lo que el modelo estima con lo
        que de verdad hay detras.
    """

    precios: pd.DataFrame
    exogenas: pd.DataFrame
    parametros: dict

    def __post_init__(self) -> None:
        if not self.precios.index.equals(self.exogenas.index):
            raise ErrorDatos(
                "Los precios y las exogenas de un escenario deben compartir indice; "
                "reindexe antes de construir el EscenarioEnergia."
            )


def prevision_exogena(
    serie: pd.Series,
    habilidad: float = HABILIDAD_PREVISION,
    horizonte: int = 1,
    semilla: int | None = None,
) -> pd.Series:
    """Prevision publicada en ``t`` para el valor de ``t + horizonte``.

    Esta es la pieza que distingue al trading de energia del de renta variable,
    y conviene entender bien por que no es mirar al futuro.

    Nadie puede predecir el precio del gas de manana. Pero el frio que hara
    manana **si** se sabe hoy: el operador del sistema y los servicios
    meteorologicos publican previsiones de temperatura, demanda y produccion
    renovable antes de que cierre el mercado diario. Esa prevision es un dato
    disponible en ``t``, y una mesa real la usa. Excluirla no haria el analisis
    mas honesto, lo haria irreal.

    La frontera esta en la habilidad. Usar el valor futuro exacto si seria
    trampa, porque ninguna prevision acierta del todo. La prevision se
    construye como el valor futuro mas un error:

        prevision(t) = x(t + horizonte) + error

    y la habilidad se mide **sobre el cambio**, no sobre el nivel:

        corr(prevision(t) - x(t),  x(t + horizonte) - x(t)) = habilidad

    Esa eleccion no es un detalle tecnico. Medir la habilidad sobre el nivel
    daria una cifra enganiosamente alta: con una variable persistente como la
    temperatura, decir "manana hara como hoy" ya correlaciona 0,85 con lo que
    ocurre, sin aportar nada. Lo unico que aniade valor es acertar el CAMBIO, y
    es ahi donde hay que exigir la habilidad. Es el mismo principio que el
    error escalado de :mod:`icm.ml.evaluacion`: una medida solo informa si esta
    anclada a la referencia ingenua que hay que batir.

    Parametros
    ----------
    habilidad:
        Correlacion entre el cambio previsto y el cambio realizado, en (0, 1].
        ``0`` seria no saber nada mas que la persistencia y ``1``, acertar el
        cambio exactamente. Las previsiones meteorologicas a 24 horas se mueven
        en 0,85-0,95.
        Con ``habilidad=1`` no hay error y el resultado **si** constituye fuga
        de informacion: se permite solo para poder medir cuanto valdria una
        prevision perfecta, nunca para presentar resultados.
    semilla:
        Fija el error de prevision, para que el escenario sea reproducible.

    Devuelve
    --------
    Serie indexada por ``t`` cuyo valor se refiere a ``t + horizonte``. Las
    ultimas ``horizonte`` posiciones quedan a ``NaN``: la prevision de un dia
    posterior al final de la muestra no existe.

    Notas
    -----
    El nombre de la serie devuelta lleva el prefijo ``prevision_``, que es como
    :func:`icm.ml.caracteristicas.construir_matriz` la reconoce como variable
    conocida por adelantado y la usa con retardo cero.
    """
    if not isinstance(serie, pd.Series):
        raise ErrorDatos("prevision_exogena espera una Series indexada por fecha.")
    habilidad = float(habilidad)
    if not 0.0 < habilidad <= 1.0:
        raise ErrorDatos(
            f"La habilidad de la prevision debe estar en (0, 1], se ha recibido {habilidad}. "
            "Use 0,95 para temperatura a 24 horas y 0,85 para viento."
        )
    if horizonte < 1:
        raise ErrorDatos("El horizonte de la prevision debe ser de al menos un periodo.")

    futuro = serie.shift(-horizonte)
    if habilidad == 1.0:
        salida = futuro.copy()
    else:
        cambio = futuro - serie
        desviacion = float(cambio.std(ddof=1))
        constante = serie.nunique(dropna=True) <= 1
        if constante:
            # Una serie que no cambia es perfectamente previsible, y eso no es
            # un caso patologico: los grados-dia de refrigeracion valen cero
            # todo un invierno. Prever exactamente un cero no aporta ninguna
            # informacion, asi que aqui la prevision puede ser el valor futuro.
            salida = futuro.copy()
        elif not np.isfinite(desviacion) or desviacion == 0.0:
            # Con NIVEL variable, en cambio, devolver el futuro es una fuga
            # perfecta. La condicion anterior era 'desviacion == 0', que se
            # cumple siempre que el CAMBIO es constante: cualquier rampa
            # lineal, cualquier tramo interpolado. Medido: sobre la rampa
            # arange(0, 60, 0.5), con habilidad 0,99, 0,50 y hasta 0,01, el
            # error maximo de la prevision era 0 en los tres casos. Y con dos
            # observaciones, ddof=1 deja una sola diferencia valida, std
            # devuelve NaN y se tomaba tambien esta rama.
            raise ErrorDatos(
                "No se puede calibrar una habilidad de prevision sobre esta serie: su "
                f"nivel varia ({serie.nunique(dropna=True)} valores distintos) pero el "
                "cambio a un horizonte de "
                f"{horizonte} es constante o no calculable (desviacion={desviacion}). "
                "Devolver el valor futuro seria una prevision perfecta, no una con la "
                "habilidad pedida. Si la serie es una rampa o un tramo interpolado, "
                "previea la serie de la que salio; si tiene menos de tres "
                "observaciones, alarguela."
            )
        else:
            # corr(d + e, d) = sd(d) / sqrt(sd(d)^2 + sd(e)^2) = habilidad
            escala_error = desviacion * np.sqrt(1.0 / habilidad**2 - 1.0)
            generador = np.random.default_rng(semilla)
            error = generador.standard_normal(len(serie)) * escala_error
            salida = futuro + error
            salida[futuro.isna()] = np.nan

    nombre = serie.name if serie.name else "exogena"
    salida.name = f"prevision_{nombre}"
    return salida


def _validar_activos(activos: Sequence[str] | None) -> list[str]:
    if activos is None:
        return list(CATALOGO)
    claves = normalizar_tickers(activos)
    desconocidos = [c for c in claves if c not in CATALOGO]
    if desconocidos:
        raise ErrorDatos(
            f"Activos fuera del catalogo de energia: {desconocidos}. "
            f"Use alguno de: {', '.join(CATALOGO)}."
        )
    return claves


def _perfil_activo(clave: str) -> dict:
    """Parametros dinamicos de un activo: perfil de su tipo mas sus ajustes."""
    ficha = CATALOGO[clave]
    perfil = dict(_PERFIL_TIPO[ficha["tipo"]])
    perfil.update(_AJUSTE_ACTIVO.get(clave, {}))
    perfil["amplitud_estacional"] = float(ficha["estacionalidad"])
    return perfil


def _por_activo(claves: Sequence[str], valores: np.ndarray) -> dict[str, float]:
    """Empareja un vector de parametros con sus activos, en tipos de Python."""
    return {clave: float(valor) for clave, valor in zip(claves, valores)}


def _viento(dia_del_anio: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Factor de carga eolica en (0, 1), mas alto en invierno y persistente.

    Se modela como una logistica de un AR(1) para que quede acotado entre 0 y 1
    sin recortes bruscos: un factor de carga no puede ser negativo ni superar
    la potencia instalada.
    """
    latente = _ar1(len(dia_del_anio), 0.60, 1.0, generador)
    senal = -0.9 + 0.45 * _ciclo_anual(dia_del_anio, _PICO_INVIERNO) + 0.8 * latente
    return 1.0 / (1.0 + np.exp(-senal))


def _almacenamiento(dia_del_anio: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Nivel de llenado de los almacenamientos de gas, en porcentaje.

    Se inyecta en verano y se extrae en invierno, asi que el ciclo va medio
    anio desfasado respecto al de la demanda. Es la variable que explica por
    que un invierno frio con los depositos llenos no mueve el precio y con los
    depositos vacios lo dispara.
    """
    nivel = 55.0 + 30.0 * _ciclo_anual(dia_del_anio, _PICO_ALMACENAMIENTO)
    nivel = nivel + _ar1(len(dia_del_anio), 0.95, 6.0, generador)
    return np.clip(nivel, 5.0, 100.0)


def simular_energia(
    activos: Sequence[str] | None = None,
    n_periodos: int = 1500,
    inicio: str | pd.Timestamp = "2018-01-01",
    semilla: int | None = 42,
    intensidad_saltos: float = 0.02,
    habilidad_prevision: float = HABILIDAD_PREVISION,
) -> EscenarioEnergia:
    """Genera un mercado de energia completo: precios, exogenas y la verdad que los liga.

    El log-precio de cada activo es

    ``log P = log(nivel) + estacional(t) + sensibilidades @ exogenas_centradas(t) + x(t)``

    donde ``x`` es un Ornstein-Uhlenbeck discreto con volatilidad GARCH comun,
    un factor comun entre activos y saltos de Poisson al alza. Como las
    exogenas entran en el nivel del log-precio, la sensibilidad de cada activo
    es exactamente el coeficiente que una regresion del rendimiento diario
    sobre la *variacion* de la exogena deberia recuperar: esa es la senal que
    el ML tiene que encontrar, y ``parametros`` guarda la respuesta.

    Parametros
    ----------
    activos:
        Claves del :data:`CATALOGO`. Por omision, todas.
    intensidad_saltos:
        Numero medio de saltos por dia, antes de escalar por la propension de
        cada tipo de activo (la electricidad salta mucho mas que una accion).
        Con 0 no hay saltos y las colas quedan solo con las del GARCH.

    Notas
    -----
    Los saltos se compensan restando su media: asi la asimetria al alza se
    traduce en colas derechas y no en una deriva espuria que un modelo podria
    confundir con tendencia.
    """
    claves = _validar_activos(activos)
    if n_periodos < 2:
        raise ErrorDatos("Hacen falta al menos 2 periodos para simular un escenario.")
    if intensidad_saltos < 0.0:
        raise ErrorDatos(
            f"La intensidad de saltos es un numero medio de saltos por dia y no puede "
            f"ser negativa; se ha recibido {intensidad_saltos}."
        )

    fechas = pd.bdate_range(start=inicio, periods=n_periodos, name="fecha")
    dia_del_anio = fechas.dayofyear.to_numpy(dtype=float)
    generador = np.random.default_rng(semilla)

    # Mismo clima templado que el que trae por defecto simular_temperatura.
    temperatura = _temperatura(
        dia_del_anio,
        generador,
        media_anual=14.0,
        amplitud=9.0,
        persistencia=0.85,
        ruido=2.5,
    )
    grados = grados_dia(pd.Series(temperatura, index=fechas))
    exogenas = pd.DataFrame(
        {
            "temperatura": temperatura,
            "hdd": grados["hdd"].to_numpy(),
            "cdd": grados["cdd"].to_numpy(),
            "viento": _viento(dia_del_anio, generador),
            "almacenamiento": _almacenamiento(dia_del_anio, generador),
        },
        index=fechas,
    )

    # Las previsiones son el unico canal por el que un modelo puede anticipar
    # el precio de manana: el efecto de las exogenas es contemporaneo, asi que
    # conocer el frio de hoy no dice nada del rendimiento de manana, y conocer
    # el frio PREVISTO para manana lo dice casi todo. Sin esta pieza el
    # ejercicio no tiene senal explotable, que es exactamente lo que le pasa a
    # una mesa que no mira el parte meteorologico.
    for nombre in EXOGENAS_PREVISTAS:
        prevision = prevision_exogena(
            exogenas[nombre],
            habilidad=habilidad_prevision,
            semilla=generador.integers(0, 2**32 - 1),
        )
        exogenas[prevision.name] = prevision

    perfiles = [_perfil_activo(clave) for clave in claves]

    def campo(nombre: str) -> np.ndarray:
        """Un parametro de todos los activos como vector alineado con ``claves``."""
        return np.array([perfil[nombre] for perfil in perfiles], dtype=float)

    reversion = campo("reversion")
    volatilidad = campo("volatilidad") / np.sqrt(DIAS_HABILES_ANIO)
    carga = campo("carga_factor")
    tamanio_salto = campo("tamanio_salto")
    lambda_salto = intensidad_saltos * campo("propension_saltos")
    deriva = campo("deriva_anual") / DIAS_HABILES_ANIO
    sensibilidades = np.vstack(
        [
            campo("sensibilidad_hdd"),
            campo("sensibilidad_cdd"),
            campo("sensibilidad_viento"),
            campo("sensibilidad_almacenamiento"),
        ]
    )

    conductoras = exogenas[["hdd", "cdd", "viento", "almacenamiento"]].to_numpy()
    efecto_exogeno = (conductoras - conductoras.mean(axis=0)) @ sensibilidades

    # Un unico factor comun mas una parte idiosincratica por activo: las crisis
    # energeticas mueven todo el complejo, pero cada mercado tiene lo suyo.
    factor_comun = generador.standard_normal(n_periodos)
    propio = generador.standard_normal((n_periodos, len(claves)))
    innovaciones = carga * factor_comun[:, None] + np.sqrt(1.0 - carga**2) * propio
    garch = factor_volatilidad_garch(n_periodos, _GARCH_ALFA, _GARCH_BETA, generador)
    choques = volatilidad * garch[:, None] * innovaciones

    # Suma de k exponenciales = Gamma(k, escala). Siempre positivos: los picos
    # de energia son al alza.
    cuenta = generador.poisson(lambda_salto, size=(n_periodos, len(claves)))
    magnitudes = generador.gamma(np.maximum(cuenta, 1), tamanio_salto)
    saltos = np.where(cuenta > 0, magnitudes, 0.0) - lambda_salto * tamanio_salto

    # Ornstein-Uhlenbeck discreto: cada activo arranca en su nivel medio y lo
    # que se aleja de el vuelve a razon de "reversion" por dia. Con reversion
    # nula (las acciones) queda un paseo aleatorio con deriva, que es lo suyo.
    desviacion = np.empty((n_periodos, len(claves)))
    desviacion[0] = 0.0
    persistencia = 1.0 - reversion
    for t in range(1, n_periodos):
        desviacion[t] = persistencia * desviacion[t - 1] + deriva + choques[t] + saltos[t]

    estacional = campo("amplitud_estacional") * _ciclo_anual(dia_del_anio, _PICO_INVIERNO)[:, None]
    log_precio = np.log(campo("nivel")) + estacional + efecto_exogeno + desviacion
    precios = pd.DataFrame(np.exp(log_precio), index=fechas, columns=claves)

    # Las sensibilidades van por unidad de la exogena (grado dia, punto de
    # factor de carga, punto porcentual de llenado); la volatilidad objetivo es
    # anualizada y solo cubre la parte estocastica, asi que la realizada sale
    # algo mayor: las exogenas y los saltos aniaden varianza por encima.
    parametros = {
        "activos": list(claves),
        "inicio": str(fechas[0].date()),
        "n_periodos": int(n_periodos),
        "semilla": semilla,
        "garch": {"alfa": _GARCH_ALFA, "beta": _GARCH_BETA},
        "habilidad_prevision": float(habilidad_prevision),
        "exogenas_previstas": list(EXOGENAS_PREVISTAS),
        "sensibilidad_hdd": _por_activo(claves, sensibilidades[0]),
        "sensibilidad_cdd": _por_activo(claves, sensibilidades[1]),
        "sensibilidad_viento": _por_activo(claves, sensibilidades[2]),
        "sensibilidad_almacenamiento": _por_activo(claves, sensibilidades[3]),
        "velocidad_reversion": _por_activo(claves, reversion),
        "volatilidad_objetivo": _por_activo(claves, campo("volatilidad")),
        "intensidad_saltos": _por_activo(claves, lambda_salto),
        "tamanio_medio_salto": _por_activo(claves, tamanio_salto),
        "carga_factor_comun": _por_activo(claves, carga),
        "amplitud_estacional": _por_activo(claves, campo("amplitud_estacional")),
        "deriva_anual": _por_activo(claves, campo("deriva_anual")),
        "nivel_medio": _por_activo(claves, campo("nivel")),
    }
    return EscenarioEnergia(
        precios=validar_marco_precios(precios, claves),
        exogenas=exogenas,
        parametros=parametros,
    )


class ProveedorEnergiaSimulado(ProveedorDatos):
    """Proveedor que genera precios de energia en lugar de descargarlos.

    Encaja con :func:`icm.datos.cargar_precios` como cualquier otra fuente, de
    modo que un cuaderno se puede escribir contra la interfaz real y ejecutarse
    sin red. El escenario completo de la ultima llamada queda en
    :attr:`ultimo_escenario`, que es como se recuperan las exogenas y los
    parametros verdaderos despues de haber pedido solo los precios.
    """

    nombre = "energia_simulada"

    def __init__(self, semilla: int | None = 42, **parametros) -> None:
        self.semilla = semilla
        self.parametros = parametros
        self.ultimo_escenario: EscenarioEnergia | None = None

    def descargar(
        self,
        tickers: Sequence[str],
        inicio: str | pd.Timestamp | None = None,
        fin: str | pd.Timestamp | None = None,
        intervalo: str = "1d",
    ) -> pd.DataFrame:
        if intervalo != "1d":
            raise ErrorDatos("El proveedor de energia simulada solo genera series diarias.")
        parametros = dict(self.parametros)
        if inicio is not None:
            parametros.setdefault("inicio", inicio)
        if inicio is not None and fin is not None:
            habiles = len(pd.bdate_range(start=inicio, end=fin))
            parametros.setdefault("n_periodos", max(habiles, 2))

        escenario = simular_energia(activos=tickers, semilla=self.semilla, **parametros)
        self.ultimo_escenario = escenario
        precios = escenario.precios
        if inicio is not None:
            precios = precios.loc[precios.index >= pd.Timestamp(inicio)]
        if fin is not None:
            precios = precios.loc[precios.index <= pd.Timestamp(fin)]
        if precios.empty:
            raise ErrorDatos("La ventana solicitada no contiene sesiones simuladas.")
        return precios
