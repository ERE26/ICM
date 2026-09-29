"""Curvas agregadas de oferta y demanda de OMIE, y el impacto de operar en ellas.

Por que este modulo existe
--------------------------
Todo lo que hay en ``almacenamiento`` supone que el activo es **precio
aceptante**: compra y vende al precio publicado sin moverlo. Esa hipotesis es
razonable para 1 MW y es falsa para 500. Y no se puede comprobar con series de
precios, porque un precio es el resultado de la casacion, no la curva que la
produjo. Hace falta la pila de ofertas.

OMIE publica la curva agregada de cada sesion del diario: para cada periodo,
todas las ofertas de compra y de venta con su precio, su cantidad y si
quedaron casadas. Con eso se puede responder la pregunta que de verdad importa
para dimensionar un activo: **cuanto mueve el precio meter 100, 500 o 1 000 MW
mas de demanda o de oferta en una hora concreta**.

Lo que sale al medirlo (21 sesiones entre 2023 y 2026, 864 periodos)
--------------------------------------------------------------------
El resultado no es el que cabria esperar y es el mas relevante de todo el
modulo: **el impacto no es simetrico**. Anadir 500 MW de demanda sube el
precio 1,93 EUR/MWh en la mediana; anadir 500 MW de oferta lo baja 10,57.
Un factor de 5,5.

La razon es economica y no un artefacto: la demanda electrica es mucho mas
inelastica que la oferta. Buena parte se oferta al limite del mercado porque
tiene que consumir pase lo que pase, asi que en cuanto se agota el tramo
elastico el precio se desploma. La oferta, en cambio, tiene un tramo plano
largo de renovables a coste cero.

Lo que eso le cuesta a un activo real, medido sobre los 67 871 periodos
horarios de OMIE con una bateria de cuatro horas, es que el arbitraje deja de
escalar: la merma sobre el margen bruto es del 0,1 % a 1 MW, 1,2 % a 10, 6,0 %
a 50, 10,3 % a 100, 23,4 % a 250 y **41,5 % a 500**. Y el 70 % de ese coste lo
pone la descarga. Un activo el doble de grande no renta el doble.

Y la asimetria va **justo en contra** del almacenamiento, porque se ordena por
nivel de precio exactamente al reves de como opera una bateria:

    tramo        subir al comprar 500 MW    bajar al vender 500 MW
    < 5                      0,48                       2,29
    5-25                     7,06                       7,51
    25-60                    5,71                       6,89
    60-120                   3,83                       9,68
    > 120                    1,44                      20,01

La bateria **carga** en el tramo barato, donde comprar casi no mueve el precio
(0,48). Y **descarga** en el caro, donde vender lo hunde 20,01. De modo que el
diferencial que captura se estrecha por los dos extremos pero sobre todo por
arriba: el propio activo destruye el pico del que vive. Un parque de 1 000 MW
de baterias descargando a la vez en las horas de mas de 120 EUR/MWh se come
32 EUR/MWh de su propio margen.

Sobre la fiabilidad de la lectura
---------------------------------
Se puede contrastar, y conviene, porque una pila de ofertas mal leida daria
numeros plausibles y falsos. El contraste es directo: el precio publicado
tiene que salir de la propia curva. Sale exacto --a menos de 0,011 EUR/MWh del
maximo de venta casada o del minimo de compra casada-- en 709 de 864 periodos,
el 82,1 %.

Del 17,9 % restante, el 86,5 % tiene el precio publicado **dentro** del cruce,
entre el minimo de compra casada y el maximo de venta casada, con un error de
0,24 EUR/MWh en la mediana y de 2,00 en el percentil 99. Eso no es un fallo de
lectura: la curva agregada viene redondeada a MWh enteros y la unidad marginal
se casa parcialmente, asi que el precio de equilibrio cae *entre* los dos
extremos y no sobre uno de ellos.

El resto se explica por la separacion de mercados. Cuando la interconexion con
Portugal se satura, Espania y Portugal casan por separado y el precio espaniol
**no es** el de la casacion iberica unica que describe el fichero. Se ve en la
propia etiqueta de pais de las ofertas casadas: en los 43 periodos marcados
``ES,PT`` el acierto baja al 53,5 %, frente al 83,6 % de los 821 marcados
``MI``.

Nada de esto afecta a la medida de impacto, y la razon es importante: **el
impacto es una diferencia de dos puntos leidos de la misma curva**, asi que un
sesgo de nivel constante se cancela. Lo que haria falso el impacto seria leer
mal la *pendiente*, y la pendiente es lo que el contraste del cruce confirma.

Las tres variantes de formato
-----------------------------
El fichero ha cambiado dos veces y las tres versiones conviven en el historico,
asi que el lector tiene que reconocer las tres o sumara mal:

1. ``Hora`` + ``Energia`` en MWh, con horas 1..24. Hasta 2025-04-17.
2. ``Periodo`` + ``Potencia`` en MW, con periodos 1..24. De 2025-04-18 al
   2025-09-30.
3. ``Periodo`` + ``Potencia`` en MW, con etiquetas ``H1Q1``..``H24Q4``. Desde
   2025-10-01, cuando el mercado paso a periodos de quince minutos.

El detalle que hay que acertar es que en las variantes 2 y 3 la cantidad es
**potencia y no energia**. Con periodos horarios coincide numericamente, asi
que un error ahi pasa desapercibido durante cinco meses de datos; con periodos
cuarto-horarios sobrestima la energia por cuatro. La duracion se deduce de
cuantos periodos tiene el dia, con la misma regla que ya usa
``icm.datos.ficheros``, que ademas cubre los dias de cambio de hora.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos, como_serie
from icm.datos.ficheros import DURACION_PERIODO


COLUMNAS_CURVA = (
    "periodo",
    "fecha",
    "pais",
    "unidad",
    "tipo",
    "cantidad",
    "precio",
    "estado",
)

#: Tolerancia al contrastar el precio publicado contra la pila de ofertas. Es
#: el redondeo del propio fichero, un centimo, con margen para el error de
#: coma flotante.
TOLERANCIA_CASACION = 0.011

#: Tramos de precio de la tabla de elasticidad, en EUR/MWh.
TRAMOS_PRECIO = (-np.inf, 5.0, 25.0, 60.0, 120.0, np.inf)

ETIQUETAS_TRAMO = ("<5", "5-25", "25-60", "60-120", ">120")

#: Potencias, en MW, a las que esta medida la elasticidad.
POTENCIAS_MEDIDAS = (50.0, 100.0, 250.0, 500.0, 1000.0)

#: Impacto en el precio marginal, en EUR/MWh, de anadir potencia al mercado.
#: Mediana sobre 864 periodos de 21 sesiones del diario de OMIE repartidas
#: entre enero de 2023 y julio de 2026, medidas con :func:`impacto_precio`.
#: ``compra`` es cuanto **sube** el precio al anadir demanda --lo que hace una
#: bateria al cargar-- y ``venta`` cuanto **baja** al anadir oferta, al
#: descargar. Va embebida y no en un fichero para que importar el modulo no
#: dependa del disco; el detalle por periodo esta en
#: ``datos/reales/omie_elasticidad_oferta.csv``.
ELASTICIDAD_OMIE: dict[str, dict[str, tuple[float, ...]]] = {
    "<5": {
        "compra": (0.085, 0.109, 0.253, 0.479, 0.748),
        "venta": (0.450, 0.709, 1.489, 2.292, 4.771),
    },
    "5-25": {
        "compra": (0.981, 1.728, 3.038, 7.058, 17.745),
        "venta": (0.834, 1.630, 4.561, 7.506, 12.312),
    },
    "25-60": {
        "compra": (1.565, 1.980, 4.451, 5.705, 11.377),
        "venta": (0.300, 1.311, 3.924, 6.890, 15.288),
    },
    "60-120": {
        "compra": (0.490, 0.943, 1.815, 3.834, 7.104),
        "venta": (1.126, 2.020, 5.150, 9.677, 18.572),
    },
    ">120": {
        "compra": (0.228, 0.400, 0.778, 1.438, 2.841),
        "venta": (2.920, 4.980, 10.928, 20.010, 31.980),
    },
}


def leer_curva_agregada(ruta: str | Path) -> pd.DataFrame:
    """Lee un fichero de curva agregada de OMIE en cualquiera de sus formatos.

    Reconoce las tres variantes descritas en la cabecera del modulo y deja la
    cantidad convertida a energia en MWh en la columna ``energia``,
    independientemente de si el fichero la daba en MWh o en MW.

    Devuelve
    --------
    Un DataFrame con una fila por oferta y las columnas ``hora``, ``cuarto``,
    ``tipo`` (``C`` compra, ``V`` venta), ``precio``, ``cantidad``, ``energia``
    y ``estado`` (``C`` casada, ``O`` no casada). En ``attrs`` quedan
    ``n_periodos``, ``horas_por_periodo`` y ``fecha``.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise ErrorDatos(f"No existe el fichero de curva '{ruta}'.")

    marco = pd.read_csv(
        ruta,
        sep=";",
        skiprows=3,
        encoding="latin-1",
        decimal=",",
        thousands=".",
        header=None,
        usecols=range(len(COLUMNAS_CURVA)),
        dtype={0: str},
        names=list(COLUMNAS_CURVA),
    )
    marco = marco.dropna(subset=["periodo", "cantidad", "precio"]).copy()
    if marco.empty:
        raise ErrorDatos(f"'{ruta}' no tiene ninguna oferta legible.")

    hora, cuarto = _periodo_a_hora_y_cuarto(marco["periodo"])
    valido = (hora.notna() & cuarto.notna()).to_numpy()
    if not valido.any():
        raise ErrorDatos(
            f"'{ruta}': ninguna etiqueta de periodo es reconocible. Se esperan "
            f"enteros 1..24 o etiquetas HxQy."
        )
    marco = marco[valido].copy()
    marco["hora"] = hora.to_numpy()[valido].astype(int)
    marco["cuarto"] = cuarto.to_numpy()[valido].astype(int)

    n_periodos = len(marco.groupby(["hora", "cuarto"]))
    minutos = DURACION_PERIODO.get(n_periodos)
    if minutos is None:
        raise ErrorDatos(
            f"'{ruta}' tiene {n_periodos} periodos y no se puede deducir su "
            f"duracion. Se reconocen {sorted(DURACION_PERIODO)}."
        )
    horas_por_periodo = minutos / 60.0

    # En las variantes 2 y 3 la cantidad es potencia, no energia. Con periodos
    # horarios el factor es 1 y da igual; con cuarto-horarios, no corregirlo
    # sobrestimaria la energia por cuatro.
    marco["energia"] = marco["cantidad"].to_numpy(dtype=float) * horas_por_periodo
    marco.attrs["n_periodos"] = n_periodos
    marco.attrs["horas_por_periodo"] = horas_por_periodo
    marco.attrs["fecha"] = _fecha_iso(marco["fecha"].iloc[0])
    return marco


def _periodo_a_hora_y_cuarto(periodo: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Separa la etiqueta de periodo en hora y cuarto de hora.

    Las etiquetas ``HxQy`` traen las dos cosas; los enteros 1..24 son horas
    completas y se les asigna el cuarto 1 para que el resto del codigo pueda
    tratar los dos formatos igual.
    """
    etiqueta = periodo.astype(str).str.strip()
    partes = etiqueta.str.extract(r"^H(\d+)Q(\d+)$")
    if partes[0].notna().any():
        return (
            pd.to_numeric(partes[0], errors="coerce"),
            pd.to_numeric(partes[1], errors="coerce"),
        )
    horas = pd.to_numeric(etiqueta, errors="coerce")
    return horas, pd.Series(1.0, index=periodo.index).where(horas.notna())


def _fecha_iso(valor) -> str:
    """Pasa la fecha del fichero, en dd/mm/aaaa, a ISO."""
    partes = str(valor).strip().split("/")
    if len(partes) != 3:
        return str(valor).strip()
    return "-".join(partes[::-1])


def casacion(curva: pd.DataFrame) -> pd.DataFrame:
    """Precio y energia de casacion que se deducen de la propia pila de ofertas.

    Sirve para contrastar la lectura: el precio que sale de aqui tiene que
    parecerse al que OMIE publica. Cuanto se parece, y por que no siempre,
    esta medido en la cabecera del modulo.

    Devuelve
    --------
    Un DataFrame indexado por ``(hora, cuarto)`` con el maximo de venta casada,
    el minimo de compra casada, su punto medio como estimacion del precio y la
    energia casada en MWh.
    """
    _exigir_curva(curva)
    filas = []
    for (hora, cuarto), bloque in curva.groupby(["hora", "cuarto"]):
        casadas = bloque[bloque["estado"].astype(str).str.strip() == "C"]
        venta = casadas[casadas["tipo"].astype(str).str.strip() == "V"]["precio"]
        compra = casadas[casadas["tipo"].astype(str).str.strip() == "C"]["precio"]
        if venta.empty or compra.empty:
            continue
        filas.append(
            {
                "hora": int(hora),
                "cuarto": int(cuarto),
                "venta_maxima": float(venta.max()),
                "compra_minima": float(compra.min()),
                "precio": (float(venta.max()) + float(compra.min())) / 2.0,
                "energia": float(casadas[casadas["tipo"].astype(str).str.strip() == "V"]["energia"].sum()),
            }
        )
    if not filas:
        raise ErrorDatos(
            "La curva no tiene ningun periodo con ofertas casadas por los dos "
            "lados, asi que no se puede deducir ningun precio de ella."
        )
    return pd.DataFrame(filas).set_index(["hora", "cuarto"])


def impacto_precio(
    curva: pd.DataFrame,
    potencias: tuple[float, ...] = POTENCIAS_MEDIDAS,
    precios_publicados=None,
) -> pd.DataFrame:
    """Cuanto mueve el precio marginal anadir potencia, periodo a periodo.

    Se camina la pila de ofertas desde el punto de casacion. Anadir demanda
    obliga a aceptar ofertas de venta mas caras, y anadir oferta obliga a
    aceptar ofertas de compra mas baratas. Las dos direcciones se miden porque
    **no son simetricas**, con un factor de 5,5 entre ellas segun los datos.

    La cantidad se expresa en MW de potencia sostenida durante todo el periodo,
    que es como se dimensiona un activo, y no en MWh, que cambiaria de
    significado entre los ficheros horarios y los cuarto-horarios.

    Cuando la pila se agota
    -----------------------
    Si la potencia anadida supera lo que queda de pila, el impacto se trunca
    en el ultimo precio ofertado, asi que el valor devuelto es una **cota
    inferior** del impacto real. Pasa en el 3,4 % de los periodos medidos, y
    solo en el lado de la compra a 1 000 MW: la pila de venta no se agota
    nunca a esas potencias. Se deja truncado y no extrapolado porque lo que
    hay mas alla del ultimo tramo no esta en el fichero, y una extrapolacion
    seria inventarselo. El sesgo va en la direccion segura: la asimetria real
    entre comprar y vender es al menos la que se mide aqui, no menor.

    Parametros
    ----------
    curva:
        Salida de :func:`leer_curva_agregada`.
    potencias:
        MW adicionales a los que medir el impacto.
    precios_publicados:
        Precio de cada periodo, en el orden del fichero. Si no se da, se usa el
        que se deduce de la propia curva con :func:`casacion`. Darlo es mejor
        cuando se tiene, porque evita arrastrar el error del cruce.

    Devuelve
    --------
    Un DataFrame con una fila por periodo y las columnas ``precio``,
    ``oferta_mw``, ``demanda_mw`` y, por cada potencia, ``sube_al_comprar_N`` y
    ``baja_al_vender_N``, en EUR/MWh.
    """
    _exigir_curva(curva)
    potencias = tuple(float(p) for p in potencias)
    if any(p <= 0.0 for p in potencias):
        raise ErrorDatos("Las potencias a las que medir el impacto tienen que ser positivas.")

    horas_por_periodo = float(curva.attrs.get("horas_por_periodo", 1.0))
    deducidos = casacion(curva)
    if precios_publicados is None:
        referencia = deducidos["precio"].to_numpy(dtype=float)
    else:
        referencia = como_serie(precios_publicados, "precios_publicados").to_numpy(dtype=float)
        if len(referencia) < len(deducidos):
            raise ErrorDatos(
                f"Se han dado {len(referencia)} precios publicados para "
                f"{len(deducidos)} periodos con casacion en la curva."
            )
        referencia = referencia[: len(deducidos)]

    filas = []
    for posicion, ((hora, cuarto), _) in enumerate(deducidos.iterrows()):
        bloque = curva[(curva["hora"] == hora) & (curva["cuarto"] == cuarto)]
        precio = float(referencia[posicion])
        fila = {"hora": int(hora), "cuarto": int(cuarto), "precio": precio}

        precios_venta, volumen_venta = _pila(bloque, "V", horas_por_periodo)
        precios_compra, volumen_compra = _pila(bloque, "C", horas_por_periodo)
        if precios_venta is None or precios_compra is None:
            continue

        # Punto de partida: cuanta potencia hay ofertada a un precio que el de
        # casacion acepta, por cada lado.
        oferta = float(np.interp(precio, precios_venta, volumen_venta))
        demanda = float(np.interp(precio, precios_compra[::-1], volumen_compra[::-1]))
        fila["oferta_mw"] = oferta
        fila["demanda_mw"] = demanda

        for potencia in potencias:
            etiqueta = f"{potencia:g}"
            fila[f"sube_al_comprar_{etiqueta}"] = (
                float(np.interp(oferta + potencia, volumen_venta, precios_venta)) - precio
            )
            fila[f"baja_al_vender_{etiqueta}"] = (
                precio - float(np.interp(demanda + potencia, volumen_compra, precios_compra))
            )
        filas.append(fila)

    if not filas:
        raise ErrorDatos("Ningun periodo de la curva tiene pila suficiente para medir el impacto.")
    return pd.DataFrame(filas).set_index(["hora", "cuarto"])


def _pila(bloque: pd.DataFrame, tipo: str, horas_por_periodo: float):
    """Pila acumulada de un lado del mercado, en MW de potencia sostenida.

    La de venta se ordena por precio creciente --es la curva de oferta-- y la
    de compra por precio decreciente, que es el orden en que se casa. Devuelve
    ``(None, None)`` si no hay al menos dos puntos, porque con uno solo no hay
    pendiente que leer.
    """
    lado = bloque[bloque["tipo"].astype(str).str.strip() == tipo]
    if len(lado) < 2:
        return None, None
    lado = lado.sort_values("precio", ascending=(tipo == "V"))
    precios = lado["precio"].to_numpy(dtype=float)
    volumen = np.cumsum(lado["energia"].to_numpy(dtype=float)) / horas_por_periodo
    return precios, volumen


def _exigir_curva(curva) -> None:
    """Comprueba que lo recibido venga de :func:`leer_curva_agregada`."""
    if not isinstance(curva, pd.DataFrame):
        raise ErrorDatos("La curva tiene que ser un DataFrame de 'leer_curva_agregada'.")
    faltan = {"hora", "cuarto", "tipo", "precio", "energia", "estado"} - set(curva.columns)
    if faltan:
        raise ErrorDatos(
            f"A la curva le faltan las columnas {sorted(faltan)}. Se espera la "
            f"salida de 'leer_curva_agregada'."
        )


def contrastar_casacion(curva: pd.DataFrame, precios_publicados) -> pd.Series:
    """Comprueba que el precio publicado se reproduce desde la pila de ofertas.

    Es el contraste que decide si la lectura del fichero vale para algo. Sin
    el, una curva mal leida --las cantidades sin convertir de potencia a
    energia, por ejemplo-- daria impactos plausibles y falsos, y no habria
    forma de notarlo.

    Cuenta tres categorias, y la distincion importa: ``exactos`` son los
    periodos en que el publicado coincide con el maximo de venta casada o el
    minimo de compra casada; ``en_el_cruce`` los que caen entre ambos, que es
    lo que produce el redondeo del fichero y la casacion parcial de la unidad
    marginal; y el resto, que en la muestra medida se concentra en los dias
    con separacion del mercado iberico.

    Devuelve
    --------
    Una Series con los recuentos, sus proporciones y el error absoluto mediano
    y maximo en EUR/MWh.
    """
    deducidos = casacion(curva)
    publicados = como_serie(precios_publicados, "precios_publicados").to_numpy(dtype=float)
    if len(publicados) < len(deducidos):
        raise ErrorDatos(
            f"Se han dado {len(publicados)} precios publicados para "
            f"{len(deducidos)} periodos con casacion."
        )
    publicados = publicados[: len(deducidos)]

    venta = deducidos["venta_maxima"].to_numpy(dtype=float)
    compra = deducidos["compra_minima"].to_numpy(dtype=float)
    error = np.minimum(np.abs(publicados - venta), np.abs(publicados - compra))
    exactos = error < TOLERANCIA_CASACION
    bajo = np.minimum(venta, compra) - TOLERANCIA_CASACION
    alto = np.maximum(venta, compra) + TOLERANCIA_CASACION
    en_cruce = (~exactos) & (publicados >= bajo) & (publicados <= alto)

    n = len(publicados)
    return pd.Series(
        {
            "periodos": float(n),
            "exactos": float(exactos.sum()),
            "proporcion_exactos": float(exactos.mean()),
            "en_el_cruce": float(en_cruce.sum()),
            "fuera": float(n - exactos.sum() - en_cruce.sum()),
            "error_mediano": float(np.median(error[~exactos])) if (~exactos).any() else 0.0,
            "error_maximo": float(error.max()),
        },
        name=str(curva.attrs.get("fecha", "curva")),
    )


def tramo_de_precio(precios) -> pd.Series:
    """Etiqueta de tramo de precio de cada observacion.

    Los tramos no son cuantiles sino cortes fijos en 5, 25, 60 y 120 EUR/MWh,
    porque lo que separa regimenes de la pila de ofertas no es la frecuencia
    sino el nivel: por debajo de 5 manda el excedente renovable, por encima de
    120 el ciclo combinado en el limite.
    """
    serie = como_serie(precios, "precios")
    return pd.cut(serie, list(TRAMOS_PRECIO), labels=list(ETIQUETAS_TRAMO))


def impacto_estimado(precios, potencia: float, direccion: str = "compra") -> pd.Series:
    """Impacto esperado de operar ``potencia`` MW, segun la tabla medida.

    Es la via practica cuando se tiene una serie larga de precios pero no las
    curvas de oferta de cada dia, que es la situacion normal: las curvas pesan
    del orden de 7 MB por sesion y solo se han medido 21. Asigna a cada precio
    el impacto mediano de su tramo, interpolando en potencia.

    Lo que **no** hace es fingir precision: dentro de un tramo todas las horas
    reciben el mismo impacto, y la dispersion real es grande --el percentil 90
    dobla a la mediana--. Vale para dimensionar y para decidir si el impacto
    importa; no para liquidar una operacion concreta.

    Parametros
    ----------
    precios:
        Serie de precios en EUR/MWh.
    potencia:
        MW que se quieren mover. Por debajo de 50 se extrapola linealmente
        desde el origen, porque el primer MW mueve el precio muy poco y la
        curva es aproximadamente lineal en ese arranque.
    direccion:
        ``"compra"`` para anadir demanda --el precio sube-- o ``"venta"`` para
        anadir oferta --el precio baja--. En los dos casos el valor devuelto
        es positivo y representa un empeoramiento del precio para quien opera.
    """
    serie = como_serie(precios, "precios")
    if direccion not in ("compra", "venta"):
        raise ErrorDatos(f"La direccion es 'compra' o 'venta', no '{direccion}'.")
    potencia = float(potencia)
    if potencia < 0.0:
        raise ErrorDatos("La potencia no puede ser negativa.")

    tramos = tramo_de_precio(serie)
    impacto = pd.Series(np.nan, index=serie.index, dtype=float)
    malla = np.asarray((0.0,) + POTENCIAS_MEDIDAS, dtype=float)
    for etiqueta in ETIQUETAS_TRAMO:
        medidos = np.asarray((0.0,) + ELASTICIDAD_OMIE[etiqueta][direccion], dtype=float)
        impacto[tramos == etiqueta] = float(np.interp(potencia, malla, medidos))
    return impacto


def margen_con_impacto(despacho, precios, potencia: float | None = None) -> pd.Series:
    """Rehace el margen de un despacho cobrando el impacto del propio activo.

    Por que es necesario y que es lo que no arregla
    ----------------------------------------------
    El despacho de ``almacenamiento`` se resuelve como precio aceptante, asi
    que su margen es una **cota superior**: programa como si vender 500 MW no
    hundiera el precio. Esta funcion no rehace la optimizacion --eso seria un
    problema cuadratico, porque el precio pasaria a depender de la cantidad--,
    sino que liquida el mismo plan a los precios que el activo se encontraria
    de verdad.

    Eso deja un plan algo subooptimo, asi que el resultado tambien es una
    cota, pero ahora **por debajo** del optimo con impacto. El error va en la
    direccion segura, y eso permite concluir: si el margen con impacto sale
    negativo, el activo no es viable a esa escala ni reoptimizando, porque
    reoptimizar solo puede mejorar hasta el limite de no operar, que da cero.

    Parametros
    ----------
    despacho:
        Un :class:`~icm.energia.Despacho` de
        :func:`~icm.energia.despacho_optimo`.
    precios:
        Los mismos precios con los que se resolvio el despacho. Hacen falta
        aparte porque el ``Despacho`` guarda el plan pero no la serie de
        precios que lo genero.
    potencia:
        MW del activo. Si no se da, se toma la de su ficha.

    Devuelve
    --------
    Una Series con el margen bruto sin impacto, el coste del impacto separado
    por carga y descarga, el margen que queda y la merma en tanto por ciento.
    """
    for atributo in ("carga", "descarga", "parametros"):
        if not hasattr(despacho, atributo):
            raise ErrorDatos("Se espera un 'Despacho' de 'despacho_optimo'.")
    serie = como_serie(precios, "precios")
    if len(serie) != len(despacho.carga):
        raise ErrorDatos(
            f"El despacho tiene {len(despacho.carga)} periodos y se han dado "
            f"{len(serie)} precios. Tienen que ser los mismos con los que se "
            f"resolvio."
        )

    if potencia is None:
        potencia = float(despacho.parametros.potencia)
    potencia = float(potencia)
    if potencia <= 0.0:
        raise ErrorDatos("La potencia del activo tiene que ser positiva.")

    carga = despacho.carga.to_numpy(dtype=float)
    descarga = despacho.descarga.to_numpy(dtype=float)

    # Cada periodo paga el impacto de la potencia que mueve **en ese periodo**,
    # no de la nominal: media hora a media potencia mueve el precio menos. La
    # referencia es la energia maxima del periodo y no la potencia, porque con
    # periodos cuarto-horarios no coinciden y la proporcion saldria cuatro
    # veces menor de lo que es.
    horas = float(getattr(despacho.parametros, "horas_por_periodo", 1.0))
    energia_maxima = potencia * horas
    sube = impacto_estimado(serie, potencia, "compra").to_numpy(dtype=float)
    baja = impacto_estimado(serie, potencia, "venta").to_numpy(dtype=float)

    coste_carga = float(np.sum(carga * sube * carga / energia_maxima))
    coste_descarga = float(np.sum(descarga * baja * descarga / energia_maxima))
    bruto = float(despacho.margen_bruto_total)
    coste = coste_carga + coste_descarga

    return pd.Series(
        {
            "margen_bruto": bruto,
            "coste_impacto_carga": coste_carga,
            "coste_impacto_descarga": coste_descarga,
            "coste_impacto": coste,
            "margen_con_impacto": bruto - coste,
            "merma_por_ciento": 100.0 * coste / bruto if bruto != 0.0 else float("nan"),
        },
        name=despacho.parametros.nombre,
    )
