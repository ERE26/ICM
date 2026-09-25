"""Estructura temporal de los futuros de energia.

Una accion cotiza para siempre; un futuro no. La energia se negocia en
contratos con vencimiento, y cada pocas semanas el contrato que se estaba
siguiendo deja de existir. Para tener una serie larga hay que empalmar
contratos, y ahi esta la trampa: el precio del contrato que vence y el del
siguiente **no coinciden**, porque incorporan costes de almacenamiento,
estacionalidad y expectativas distintas. Ese salto no es dinero ganado ni
perdido por nadie: es un artefacto del cambio de contrato. Un backtest que
encadene precios sin corregirlo inventa rendimientos, y en gas o electricidad,
donde el escalon entre meses ronda el 5-10 %, los inventa a lo grande.

Que hace este modulo
--------------------
* :func:`serie_continua` empalma los contratos y corrige el escalon, con el
  ajuste multiplicativo del sector (``"ratio"``), el aditivo para series que
  pueden ser negativas (``"diferencia"``) o sin corregir (``"ninguno"``, que
  sirve para VER el problema antes de taparlo).
* :func:`rendimiento_roll` y :func:`estado_curva` miden el carry, que en
  materias primas almacenables pesa mas a largo plazo que el propio precio.
* :func:`pendiente_curva` e :func:`interpolar_curva` describen la curva
  completa a plazo constante, que es la unica forma de comparar dos dias.
* :func:`descomponer_rendimiento` separa lo que se gana por el precio de lo
  que se pierde (o se gana) por rodar de un contrato al siguiente.

Convencion de fechas
--------------------
Todos los plazos se cuentan en dias **naturales**, no en sesiones: el
almacenamiento y el coste financiero de una posicion corren tambien en fin de
semana, y los vencimientos estan fijados en el calendario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import interpolate

from icm.comun import ErrorDatos, como_serie

#: Dias naturales por anio, para anualizar carry y pendientes.
DIAS_ANIO_NATURAL = 365.25

#: Ajustes disponibles al empalmar contratos.
METODOS_AJUSTE = ("ratio", "diferencia", "ninguno")

#: Metodos de interpolacion de la curva a plazo constante.
METODOS_INTERPOLACION = ("lineal", "log_lineal", "cubica")

#: Estados posibles de la curva.
ESTADOS_CURVA = ("contango", "backwardation", "plana")


# --- utilidades internas -------------------------------------------------------


def _como_indice_fechas(indice, nombre: str = "indice") -> pd.DatetimeIndex:
    """Convierte la entrada en un DatetimeIndex ordenado en unidades de ns."""
    try:
        fechas = pd.DatetimeIndex(pd.to_datetime(indice))
    except (ValueError, TypeError) as exc:
        raise ErrorDatos(
            f"'{nombre}' no se ha podido interpretar como fechas ({exc}): "
            "conviertalo antes con pd.to_datetime."
        ) from exc
    if len(fechas) == 0:
        raise ErrorDatos(f"'{nombre}' esta vacio: no hay ninguna fecha que procesar.")
    return fechas.as_unit("ns")


def _vencimientos_como_serie(vencimientos) -> pd.Series:
    """Normaliza los vencimientos a una Series de fechas indexada por contrato."""
    if isinstance(vencimientos, pd.Series):
        serie = vencimientos.copy()
    elif isinstance(vencimientos, dict):
        serie = pd.Series(vencimientos)
    else:
        raise ErrorDatos(
            "'vencimientos' debe ser un dict {contrato: fecha} o una Series "
            "indexada por contrato."
        )
    if serie.empty:
        raise ErrorDatos("'vencimientos' esta vacio: indique al menos una fecha de vencimiento.")
    if serie.index.has_duplicates:
        repetidos = sorted({str(c) for c in serie.index[serie.index.duplicated()]})
        raise ErrorDatos(
            f"Hay contratos repetidos en 'vencimientos': {repetidos}. "
            "Deje una sola fecha por contrato."
        )
    try:
        fechas = pd.to_datetime(serie)
    except (ValueError, TypeError) as exc:
        raise ErrorDatos(
            f"Los vencimientos no se han podido interpretar como fechas ({exc}): "
            "use fechas del tipo '2024-01-31'."
        ) from exc
    if fechas.isna().any():
        sin_fecha = sorted({str(c) for c in fechas.index[fechas.isna()]})
        raise ErrorDatos(
            f"Estos contratos no tienen fecha de vencimiento: {sin_fecha}. "
            "Complete la fecha o quite el contrato."
        )
    return pd.Series(pd.DatetimeIndex(fechas.to_numpy()).as_unit("ns"), index=fechas.index)


def _marco_contratos(contratos) -> pd.DataFrame:
    """Valida la matriz de precios por contrato y la devuelve ordenada por fecha."""
    if not isinstance(contratos, pd.DataFrame):
        raise ErrorDatos(
            "'contratos' debe ser un DataFrame con una columna por contrato y el "
            "indice de fechas."
        )
    if contratos.shape[1] == 0 or len(contratos) == 0:
        raise ErrorDatos(
            "'contratos' no tiene datos: hace falta al menos un contrato con precios."
        )
    try:
        marco = contratos.astype(float)
    except (ValueError, TypeError) as exc:
        raise ErrorDatos(
            f"Los precios de 'contratos' no son numericos ({exc}): "
            "convierta las columnas con pd.to_numeric."
        ) from exc
    marco.index = _como_indice_fechas(contratos.index, "el indice de 'contratos'")
    if marco.index.has_duplicates:
        raise ErrorDatos(
            "El indice de 'contratos' tiene fechas repetidas: agregue o elimine los "
            "duplicados antes de construir la serie continua."
        )
    return marco.sort_index()


def _par_de_series(cercano, lejano) -> tuple[pd.Series, pd.Series]:
    """Alinea los precios del contrato cercano y del lejano sobre fechas comunes."""
    primera = como_serie(cercano, "cercano")
    segunda = como_serie(lejano, "lejano")
    if primera.index.equals(segunda.index):
        return primera, segunda
    comun = primera.index.intersection(segunda.index)
    if len(comun) == 0:
        raise ErrorDatos(
            "El contrato cercano y el lejano no comparten ninguna fecha: "
            "alinee ambas series (por ejemplo con .reindex) antes de compararlas."
        )
    return primera.loc[comun], segunda.loc[comun]


def _plazos_por_columna(vencimientos_meses, columnas) -> np.ndarray:
    """Devuelve el plazo en meses de cada columna, en el orden de las columnas."""
    if isinstance(vencimientos_meses, dict):
        vencimientos_meses = pd.Series(vencimientos_meses)
    if isinstance(vencimientos_meses, pd.Series):
        faltan = [c for c in columnas if c not in vencimientos_meses.index]
        if faltan:
            raise ErrorDatos(
                f"Faltan plazos para las columnas: {faltan}. "
                "Indique el plazo en meses de cada contrato de la curva."
            )
        valores = vencimientos_meses.reindex(list(columnas)).to_numpy()
    else:
        valores = np.asarray(vencimientos_meses)
        if valores.shape != (len(columnas),):
            raise ErrorDatos(
                f"Se esperaban {len(columnas)} plazos, uno por columna de la curva, "
                f"y se han recibido {valores.size}."
            )
    try:
        plazos = valores.astype(float)
    except (ValueError, TypeError) as exc:
        raise ErrorDatos(
            f"Los plazos no son numericos ({exc}): expreselos en meses hasta el vencimiento."
        ) from exc
    if not np.isfinite(plazos).all():
        raise ErrorDatos("Hay plazos ausentes o infinitos: indique los meses de cada contrato.")
    if len(np.unique(plazos)) < 2:
        raise ErrorDatos(
            "Hacen falta al menos dos plazos distintos para medir la pendiente de la curva."
        )
    return plazos


# --- calendario de vencimientos ------------------------------------------------


def dias_a_vencimiento(indice: pd.DatetimeIndex, vencimientos: pd.Series | dict) -> pd.Series:
    """Dias naturales que faltan, en cada fecha, hasta el proximo vencimiento.

    Es la variable que gobierna casi todo lo demas: la liquidez se concentra en
    el contrato mas cercano hasta pocos dias antes de su vencimiento, momento en
    que se seca de golpe y hay que rodar. Tenerla explicita permite excluir las
    ultimas sesiones de un contrato, fechar los rolls o usarla como variable
    explicativa (la volatilidad de un futuro crece al acercarse el vencimiento,
    el llamado efecto Samuelson).

    Mide el vencimiento, no el roll: el desplazamiento de ``dias_antes`` lo
    aplica :func:`serie_continua`.

    Devuelve
    --------
    Series indexada por fecha. Vale ``NaN`` en las fechas posteriores al ultimo
    vencimiento conocido, porque ahi no hay contrato al que mirar; si aparecen
    muchos ``NaN``, falta calendario, no datos.
    """
    fechas = _como_indice_fechas(indice)
    venc = _vencimientos_como_serie(vencimientos)
    orden = pd.DatetimeIndex(venc.sort_values().to_numpy()).as_unit("ns")

    posicion = orden.searchsorted(fechas, side="left")
    dias = np.full(len(fechas), np.nan)
    vigentes = posicion < len(orden)
    if vigentes.any():
        dias[vigentes] = (orden[posicion[vigentes]] - fechas[vigentes]) / pd.Timedelta(days=1)
    return pd.Series(dias, index=fechas, name="dias_a_vencimiento")


# --- serie continua ------------------------------------------------------------


def serie_continua(
    contratos: pd.DataFrame,
    vencimientos,
    metodo: str = "ratio",
    dias_antes: int = 5,
) -> pd.DataFrame:
    """Empalma los contratos en una unica serie de precios utilizable.

    Parametros
    ----------
    contratos:
        Una columna por contrato y el indice de fechas. Las columnas pueden
        tener huecos (un contrato lejano casi nunca cotiza desde el principio);
        solo se exige precio en las fechas en que el contrato esta activo.
    vencimientos:
        Fecha de vencimiento de cada contrato. Deben estar **todos** los que
        aparecen como columna, y el ultimo tiene que llegar al final del indice:
        un calendario sin los contratos nuevos dejaria activo uno ya vencido.
    metodo:
        ``"ratio"`` aplica un ajuste multiplicativo (el estandar del sector:
        conserva los rendimientos porcentuales, que es lo que se modeliza, y
        exige precios positivos en toda la serie activa, no solo en los
        empalmes);
        ``"diferencia"`` aplica uno aditivo, obligatorio cuando la serie puede
        cruzar el cero, como un spark spread o un precio electrico negativo;
        ``"ninguno"`` encadena sin corregir, util para ver el tamanio real del
        artefacto antes de taparlo.
    dias_antes:
        Dias naturales antes del vencimiento en que se rueda. No se espera al
        ultimo dia porque la liquidez del contrato que vence se seca y el precio
        deja de ser operable; el sector rueda entre 3 y 10 dias antes.

    Devuelve
    --------
    DataFrame con el mismo indice y cuatro columnas: ``"precio"`` (serie continua
    ya ajustada), ``"contrato"`` (cual esta activo en cada fecha), ``"ajuste"``
    (el factor multiplicativo aplicado, o el sumando si ``metodo="diferencia"``)
    y ``"metodo"`` (el ajuste con el que se construyo). La ultima existe para
    que :func:`descomponer_rendimiento` no tenga que adivinar el ajuste: es una
    columna y no solo una marca en ``.attrs`` porque los metadatos de pandas no
    sobreviven a un ``to_csv`` ni a un parquet, y guardar la serie para
    descomponerla otro dia es lo normal.

    Por que el ajuste va hacia atras
    --------------------------------
    El escalon del roll se corrige moviendo **el pasado**, no el futuro: el
    ultimo tramo se deja intacto y cada tramo anterior se multiplica (o se
    desplaza) acumulativamente. Asi el ultimo valor de la serie coincide
    exactamente con el precio de mercado de hoy, y la serie se puede usar para
    operar: los niveles, los soportes y las senales estan en la escala que
    aparece en la pantalla. Ajustar hacia delante daria la misma secuencia de
    rendimientos, pero dejaria el nivel actual desplazado respecto al mercado,
    y toda regla que mire niveles (o toda valoracion de una posicion) quedaria
    descuadrada. El precio pasado es, en ambos casos, un precio sintetico: no
    es lo que cotizo aquel dia, es lo que habria cotizado una posicion mantenida
    hasta hoy. Es la eleccion correcta para medir rentabilidad, y la incorrecta
    para responder "a cuanto se pago el gas aquel invierno".
    """
    marco = _marco_contratos(contratos)
    venc = _vencimientos_como_serie(vencimientos)
    if metodo not in METODOS_AJUSTE:
        raise ErrorDatos(
            f"Metodo de ajuste '{metodo}' desconocido. Disponibles: {', '.join(METODOS_AJUSTE)}."
        )
    dias_antes = int(dias_antes)
    if dias_antes < 0:
        raise ErrorDatos("'dias_antes' no puede ser negativo: se rueda antes del vencimiento.")

    faltan = [str(c) for c in marco.columns if c not in venc.index]
    if faltan:
        raise ErrorDatos(
            f"Faltan vencimientos para los contratos: {faltan}. "
            "Anada su fecha a 'vencimientos' o quite esas columnas de 'contratos'."
        )
    venc = venc.loc[list(marco.columns)].sort_values()
    if venc.duplicated().any():
        raise ErrorDatos(
            "Hay contratos con la misma fecha de vencimiento: no se puede decidir el "
            "orden de la cadena. Revise 'vencimientos'."
        )

    orden = list(venc.index)
    fechas_roll = pd.DatetimeIndex((venc - pd.Timedelta(days=dias_antes)).to_numpy()).as_unit("ns")
    # Un contrato deja de ser el activo en cuanto se alcanza su fecha de roll;
    # el ultimo de la cadena no tiene sucesor, asi que aguanta hasta el final.
    posicion = fechas_roll[:-1].searchsorted(marco.index, side="right")

    # 'posicion' indexa la cadena por orden de vencimiento, que no tiene por que
    # ser el orden de las columnas del DataFrame.
    columna_de_tramo = np.array([marco.columns.get_loc(c) for c in orden])
    activo = np.asarray(orden, dtype=object)[posicion]

    # El ultimo contrato de la cadena no tiene sucesor, asi que la cadena lo
    # mantendria activo indefinidamente aunque el calendario se haya quedado
    # corto. Cotizar un contrato ya vencido no es un dato: es el ultimo precio
    # arrastrado por el proveedor, y la serie entera saldria de ahi sin aviso.
    vencimiento_activo = pd.DatetimeIndex(venc.to_numpy()).as_unit("ns")[posicion]
    caducado = marco.index > vencimiento_activo
    if caducado.any():
        fecha = marco.index[caducado][0]
        raise ErrorDatos(
            f"El {fecha.date()} el contrato activo '{activo[caducado][0]}' ya habia "
            f"vencido (el {vencimiento_activo[caducado][0].date()}) y la cadena no "
            "tiene sucesor. Anada a 'vencimientos' los contratos posteriores o "
            "recorte el indice a las fechas que cubre el calendario."
        )

    crudo = marco.to_numpy()[np.arange(len(marco)), columna_de_tramo[posicion]]
    sin_precio = ~np.isfinite(crudo)
    if sin_precio.any():
        fecha = marco.index[sin_precio][0]
        raise ErrorDatos(
            f"El contrato activo '{activo[sin_precio][0]}' no tiene precio el "
            f"{fecha.date()}. Rellene el hueco o recorte el indice "
            "(por ejemplo con contratos.dropna(how='all'))."
        )

    # _ajustes_hacia_atras solo puede comprobar el signo en las fechas de empalme,
    # pero el ajuste multiplicativo escala la serie entera: un precio negativo en
    # medio de un tramo llega igualmente al divisor de cualquier rendimiento.
    if metodo == "ratio":
        no_positivo = crudo <= 0.0
        if no_positivo.any():
            fecha = marco.index[no_positivo][0]
            raise ErrorDatos(
                f"El contrato activo '{activo[no_positivo][0]}' cotiza a "
                f"{crudo[no_positivo][0]} el {fecha.date()} y el ajuste por ratio "
                "reescala precios. Use metodo='diferencia' para series que pueden "
                "ser negativas o cruzar el cero."
            )

    ajuste_por_tramo = _ajustes_hacia_atras(marco, orden, posicion, metodo)
    ajuste = np.array([ajuste_por_tramo[p] for p in posicion])
    precio = crudo + ajuste if metodo == "diferencia" else crudo * ajuste

    salida = pd.DataFrame(
        {"precio": precio, "contrato": activo, "ajuste": ajuste, "metodo": metodo},
        index=marco.index,
    )
    # La marca en attrs se mantiene por compatibilidad con quien ya la lea, pero
    # el dato que cuenta es la columna: attrs se pierde al guardar en disco.
    salida.attrs["metodo"] = metodo
    salida.attrs["dias_antes"] = dias_antes
    return salida


def _ajustes_hacia_atras(
    marco: pd.DataFrame,
    orden: list,
    posicion: np.ndarray,
    metodo: str,
) -> dict[int, float]:
    """Ajuste acumulado de cada tramo, empezando por el ultimo (que no se toca).

    El ancla de cada empalme es la ultima fecha en que el contrato que vence
    sigue activo: ese dia cotizan los dos contratos y su diferencia (o su
    cociente) es justo el escalon que hay que eliminar.
    """
    tramos = list(dict.fromkeys(posicion.tolist()))
    neutro = 0.0 if metodo == "diferencia" else 1.0
    ajustes = {tramos[-1]: neutro}
    if metodo == "ninguno":
        return dict.fromkeys(tramos, 1.0)

    for tramo, siguiente in reversed(list(zip(tramos[:-1], tramos[1:]))):
        ancla = marco.index[posicion == tramo][-1]
        cercano = float(marco.at[ancla, orden[tramo]])
        lejano = float(marco.at[ancla, orden[siguiente]])
        if not np.isfinite(lejano):
            raise ErrorDatos(
                f"El contrato '{orden[siguiente]}' no cotiza el {ancla.date()}, que es la "
                f"ultima sesion de '{orden[tramo]}' y donde se calcula el empalme. "
                "Rellene ese precio o adelante el roll con un 'dias_antes' mayor."
            )
        if metodo == "ratio":
            if cercano <= 0.0 or lejano <= 0.0:
                raise ErrorDatos(
                    f"El ajuste por ratio necesita precios positivos y el {ancla.date()} "
                    f"no lo son ({orden[tramo]}={cercano}, {orden[siguiente]}={lejano}). "
                    "Use metodo='diferencia' para series que pueden ser negativas."
                )
            ajustes[tramo] = (lejano / cercano) * ajustes[siguiente]
        else:
            ajustes[tramo] = (lejano - cercano) + ajustes[siguiente]
    return ajustes


# --- carry y forma de la curva -------------------------------------------------


def rendimiento_roll(cercano, lejano, dias_entre_vencimientos: float = 30.0) -> pd.Series:
    """Carry anualizado de mantener la posicion: ``(cercano - lejano) / |lejano|``.

    Mide lo que gana o pierde la posicion solo por el paso del tiempo, sin que
    el precio se mueva: al acercarse el vencimiento, el contrato que se tiene
    converge al precio del que era el cercano. Es positivo en backwardation
    (el cercano cotiza mas caro, y rodar hacia delante compra mas barato) y
    negativo en contango (cada roll compra mas caro de lo que se vende).

    En materias primas almacenables este termino domina la rentabilidad a largo
    plazo por encima del propio movimiento del precio: un indice de gas puede
    perder la mitad de su valor en una decada con el precio spot plano, solo por
    rodar en contango mes tras mes. Cualquier estrategia de energia que ignore
    el carry esta midiendo mal su rentabilidad.

    Parametros
    ----------
    dias_entre_vencimientos:
        Dias naturales que separan los dos contratos. Es lo que convierte el
        diferencial en una tasa comparable: un 2 % entre contratos mensuales y
        un 2 % entre contratos trimestrales no son el mismo carry.

    Notas
    -----
    El denominador es el **valor absoluto** del contrato lejano, igual que en
    :func:`estado_curva`. Con electricidad en horas de exceso solar los dos
    contratos pueden cotizar en negativo, y dividir por un lejano con signo
    invertiria el resultado: una curva en backwardation daria carry negativo,
    justo lo contrario de lo que este signo significa. Con el valor absoluto el
    signo del carry es siempre el signo del diferencial ``cercano - lejano``; la
    magnitud, en cambio, se vuelve poco informativa cerca del cero, asi que ahi
    conviene mirar directamente el diferencial absoluto.
    """
    precio_cercano, precio_lejano = _par_de_series(cercano, lejano)
    dias = float(dias_entre_vencimientos)
    if dias <= 0.0:
        raise ErrorDatos(
            "'dias_entre_vencimientos' debe ser positivo: es la distancia en dias "
            "naturales entre los dos vencimientos."
        )
    if (precio_lejano == 0.0).any():
        raise ErrorDatos(
            "Hay precios nulos en el contrato lejano y el carry relativo se divide "
            "por ellos: elimine esas fechas o use el diferencial absoluto."
        )
    serie = (precio_cercano - precio_lejano) / precio_lejano.abs() * (DIAS_ANIO_NATURAL / dias)
    serie.name = "rendimiento_roll"
    return serie


def estado_curva(cercano, lejano, tolerancia: float = 0.0) -> pd.Series:
    """Clasifica la curva en ``"contango"``, ``"backwardation"`` o ``"plana"``.

    Contango (lejano por encima del cercano) es el estado normal de un mercado
    con almacenamiento holgado: el precio a plazo incorpora el coste de guardar
    la mercancia. Backwardation aparece cuando el mercado esta tenso y tener el
    producto hoy vale mas que tenerlo manana; es la senal clasica de escasez
    fisica, y ademas el estado en que una posicion larga cobra carry.

    Parametros
    ----------
    tolerancia:
        Diferencia relativa (respecto al valor absoluto del contrato lejano) por
        debajo de la cual la curva se considera plana. Con el valor por defecto,
        cero, cualquier diferencia minima ya clasifica; conviene subirla para no
        leer ruido de cotizacion como una senal de mercado.
    """
    precio_cercano, precio_lejano = _par_de_series(cercano, lejano)
    tolerancia = float(tolerancia)
    if tolerancia < 0.0:
        raise ErrorDatos("La tolerancia debe ser cero o positiva: es una anchura, no un umbral.")
    magnitud = precio_lejano.abs()
    if (magnitud == 0.0).any():
        raise ErrorDatos(
            "Hay precios nulos en el contrato lejano: no se puede medir la pendiente "
            "relativa. Elimine esas fechas."
        )
    relativa = (precio_cercano - precio_lejano) / magnitud
    estados = np.where(
        relativa > tolerancia,
        "backwardation",
        np.where(relativa < -tolerancia, "contango", "plana"),
    )
    return pd.Series(estados, index=precio_cercano.index, name="estado_curva")


def pendiente_curva(curva: pd.DataFrame, vencimientos_meses) -> pd.Series:
    """Pendiente de la curva por regresion del log-precio contra el plazo.

    Con dos contratos basta un cociente, pero con la curva entera lo razonable
    es una regresion: usa todos los puntos, es robusta a que uno cotice raro y
    da una cifra comparable entre fechas y entre mercados. Al ir en logaritmos,
    la pendiente es la tasa de contango por mes de plazo (0.01 significa que
    cada mes adicional encarece el contrato un 1 % aproximadamente), positiva en
    contango y negativa en backwardation.

    Parametros
    ----------
    curva:
        Una fila por fecha y una columna por contrato de la curva.
    vencimientos_meses:
        Meses hasta el vencimiento de cada columna, como dict, Series o
        secuencia en el orden de las columnas.

    Devuelve
    --------
    Series con una pendiente por fecha. Vale ``NaN`` en las fechas con menos de
    dos contratos cotizados, donde la pendiente no esta definida.
    """
    if not isinstance(curva, pd.DataFrame):
        raise ErrorDatos(
            "'curva' debe ser un DataFrame con una fila por fecha y una columna por contrato."
        )
    if curva.shape[1] < 2:
        raise ErrorDatos("Hacen falta al menos dos contratos para medir la pendiente de la curva.")
    try:
        precios = curva.astype(float).to_numpy()
    except (ValueError, TypeError) as exc:
        raise ErrorDatos(f"Los precios de la curva no son numericos ({exc}).") from exc
    plazos = _plazos_por_columna(vencimientos_meses, curva.columns)
    cotizados = precios[np.isfinite(precios)]
    if cotizados.size and cotizados.min() <= 0.0:
        raise ErrorDatos(
            "Hay precios nulos o negativos en la curva y la pendiente se mide sobre el "
            "log-precio. Use precios positivos o trabaje con el diferencial absoluto."
        )

    logaritmos = np.log(precios)
    presente = np.isfinite(logaritmos)
    x = np.where(presente, plazos, 0.0)
    y = np.where(presente, np.nan_to_num(logaritmos), 0.0)
    n = presente.sum(axis=1)
    numerador = n * (x * y).sum(axis=1) - x.sum(axis=1) * y.sum(axis=1)
    denominador = n * (x**2).sum(axis=1) - x.sum(axis=1) ** 2

    pendiente = np.full(len(curva), np.nan)
    calculable = (n >= 2) & (denominador > 0.0)
    pendiente[calculable] = numerador[calculable] / denominador[calculable]
    return pd.Series(pendiente, index=curva.index, name="pendiente_curva")


def interpolar_curva(
    curva_puntual: pd.Series,
    plazos_objetivo,
    metodo: str = "lineal",
) -> pd.Series:
    """Lleva la curva cotizada a plazos constantes.

    Por que hace falta: los contratos cotizan a fechas fijas del calendario, asi
    que su plazo se acorta un dia cada dia. El "primer vencimiento" de hoy tiene
    cuarenta dias por delante y el de la semana que viene tendra treinta y tres;
    comparar sus precios mezcla el movimiento del mercado con el mero paso del
    tiempo sobre la curva. Para comparar dos dias, o para construir una serie
    homogenea de "precio a tres meses", hay que interpolar la curva al mismo
    plazo relativo, no a la misma fecha. Es la misma razon por la que en renta
    fija se trabaja con tipos a plazo constante y no con bonos concretos.

    Parametros
    ----------
    curva_puntual:
        Precios de un dia, indexados por el plazo (en meses o dias, pero la
        misma unidad que ``plazos_objetivo``).
    metodo:
        ``"lineal"`` interpola el precio; ``"log_lineal"`` interpola el
        logaritmo, lo que respeta el crecimiento geometrico de una curva en
        contango constante y nunca devuelve precios negativos; ``"cubica"``
        suaviza con un spline, util para derivar la curva pero capaz de
        sobreoscilar si hay pocos puntos.

    No extrapola: pedir un plazo fuera del rango cotizado devuelve
    ``ErrorDatos``, porque prolongar una curva de futuros mas alla del ultimo
    contrato liquido es inventarse un precio que nadie esta dispuesto a cruzar.
    """
    if not isinstance(curva_puntual, pd.Series):
        raise ErrorDatos("'curva_puntual' debe ser una Series de precios indexada por plazo.")
    if metodo not in METODOS_INTERPOLACION:
        raise ErrorDatos(
            f"Metodo de interpolacion '{metodo}' desconocido. "
            f"Disponibles: {', '.join(METODOS_INTERPOLACION)}."
        )
    serie = curva_puntual.dropna().astype(float)
    try:
        plazos = np.asarray(serie.index, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ErrorDatos(
            f"El indice de 'curva_puntual' debe ser el plazo en numeros ({exc}): "
            "convierta las fechas a meses hasta el vencimiento."
        ) from exc
    if len(np.unique(plazos)) != len(plazos):
        raise ErrorDatos("Hay plazos repetidos en la curva: deje un solo precio por plazo.")

    ordenados = np.argsort(plazos)
    plazos, precios = plazos[ordenados], serie.to_numpy()[ordenados]
    minimo = 4 if metodo == "cubica" else 2
    if len(plazos) < minimo:
        raise ErrorDatos(
            f"El metodo '{metodo}' necesita al menos {minimo} plazos cotizados y hay "
            f"{len(plazos)}. Use metodo='lineal' con curvas tan cortas."
        )

    objetivo = np.atleast_1d(np.asarray(plazos_objetivo, dtype=float))
    fuera = (objetivo < plazos[0]) | (objetivo > plazos[-1])
    if fuera.any():
        raise ErrorDatos(
            f"Los plazos {objetivo[fuera].tolist()} quedan fuera del rango cotizado "
            f"[{plazos[0]}, {plazos[-1]}] y la curva no se extrapola. Pida plazos "
            "dentro del rango o aniada contratos mas largos."
        )

    if metodo == "lineal":
        valores = np.interp(objetivo, plazos, precios)
    elif metodo == "log_lineal":
        if (precios <= 0.0).any():
            raise ErrorDatos(
                "La interpolacion log-lineal necesita precios positivos. "
                "Use metodo='lineal' si la curva puede ser negativa."
            )
        valores = np.exp(np.interp(objetivo, plazos, np.log(precios)))
    else:
        valores = interpolate.CubicSpline(plazos, precios)(objetivo)

    indice = pd.Index(objetivo, name=serie.index.name or "plazo")
    return pd.Series(valores, index=indice, name=serie.name or "precio")


# --- descomposicion de la rentabilidad -----------------------------------------


def _metodo_de_la_serie(marco: pd.DataFrame) -> str:
    """Ajuste con el que se construyo la serie, sin adivinarlo nunca.

    Suponer ``"ratio"`` cuando falta la marca no deja ningun rastro: el total
    acumulado no cambia (solo depende de ``"precio"``), pero el reparto entre
    precio y roll -- que es para lo que sirve la descomposicion -- se calcula
    con la formula equivocada y puede salir casi veinte veces mayor sin un solo
    error. La columna ``"metodo"`` viaja con los datos; ``.attrs`` se pierde al
    guardar en csv o en parquet, y se lee solo como respaldo para las series
    guardadas antes de que la columna existiera.
    """
    if "metodo" in marco.columns:
        marcas = pd.unique(marco["metodo"].dropna())
        if len(marcas) > 1:
            raise ErrorDatos(
                f"La columna 'metodo' trae varios ajustes ({sorted(map(str, marcas))}) "
                "y la salida de serie_continua se construye con uno solo. Descomponga "
                "cada serie por separado o pase 'metodo' explicitamente."
            )
        if len(marcas) == 1:
            return str(marcas[0])
    marca = marco.attrs.get("metodo")
    if marca is not None:
        return str(marca)
    raise ErrorDatos(
        "No se sabe con que ajuste se construyo la serie: no trae la columna 'metodo' "
        "y '.attrs' viene vacio (los metadatos de pandas se pierden al guardar en csv "
        "o en parquet). Pase el argumento 'metodo' con el mismo valor que uso en "
        "serie_continua: 'ratio' o 'diferencia'."
    )


def descomponer_rendimiento(
    serie_continua: pd.DataFrame,
    metodo: str | None = None,
) -> pd.DataFrame:
    """Separa el rendimiento total en movimiento del precio y roll.

    Es la descomposicion clasica de una materia prima: lo que gana una posicion
    es lo que sube el precio cotizado **mas** lo que cuesta rodar de un contrato
    al siguiente. El segundo termino es invisible en la pantalla y explica la
    paradoja de que una materia prima suba de precio y aun asi haga perder
    dinero a quien la mantiene: si la curva esta en contango, cada roll vende
    barato el contrato que vence y compra caro el siguiente, y ese goteo puede
    comerse una subida del precio de portada.

    * ``rendimiento_precio``: variacion del precio cotizado encadenado sin
      ajustar, es decir, lo que parece que hace el mercado.
    * ``rendimiento_roll``: cero salvo los dias de roll, donde recoge el escalon
      entre contratos con su signo (negativo en contango, positivo en
      backwardation).
    * ``rendimiento_total``: lo que realmente gana la posicion, y que cumple por
      construccion ``total = precio + roll``.

    Las tres partes se dividen por el precio **sin ajustar** que tenia la vispera
    el contrato que hoy esta activo, no por el nivel ajustado de la serie. Con el
    ajuste multiplicativo da lo mismo, porque el factor se cancela en el cociente;
    con el aditivo no, y dividir por el nivel ajustado meteria en el denominador
    la suma de todos los rolls posteriores: el rendimiento de una fecha pasada
    cambiaria cada vez que llegase un dato nuevo, que es fuga de informacion del
    futuro, y la magnitud quedaria sesgada.

    Un rendimiento relativo exige que ese precio de partida sea positivo. En una
    serie que cruza el cero (un spark spread, un precio electrico negativo) no
    existe el rendimiento porcentual, asi que la funcion devuelve ``ErrorDatos``
    en lugar de una cifra con el signo invertido: ahi hay que trabajar con la
    variacion absoluta de la columna ``"precio"``.

    Parametros
    ----------
    serie_continua:
        La salida de :func:`serie_continua`, con sus columnas ``"precio"``,
        ``"contrato"``, ``"ajuste"`` y ``"metodo"``.
    metodo:
        Ajuste con el que se construyo la serie. Si se omite se lee de la columna
        ``"metodo"`` y, en las series guardadas antes de que existiera, de la
        marca en ``.attrs``; si no hay ninguna de las dos se exige el argumento
        en lugar de suponer un ajuste, porque acertar el reparto entre precio y
        roll depende de el.
    """
    marco = serie_continua
    if not isinstance(marco, pd.DataFrame) or not {"precio", "ajuste"}.issubset(marco.columns):
        raise ErrorDatos(
            "Se esperaba la salida de serie_continua, con las columnas 'precio', "
            "'contrato' y 'ajuste'."
        )
    if len(marco) < 2:
        raise ErrorDatos("Hacen falta al menos dos fechas para calcular un rendimiento.")
    if metodo is None:
        metodo = _metodo_de_la_serie(marco)
    if metodo == "ninguno":
        raise ErrorDatos(
            "Una serie sin ajustar no permite separar precio y roll, porque el escalon "
            "del roll esta mezclado con el precio. Repita serie_continua con "
            "metodo='ratio' o metodo='diferencia'."
        )
    if metodo not in ("ratio", "diferencia"):
        raise ErrorDatos(
            f"Metodo de ajuste '{metodo}' desconocido. Disponibles: 'ratio', 'diferencia'."
        )

    precio = marco["precio"].astype(float)
    ajuste = marco["ajuste"].astype(float)

    anterior = precio.shift(1)

    # 'ingenuo' es el precio de hoy valorado con el ajuste de ayer: reproduce lo
    # que veria quien encadena cotizaciones sin corregir el salto del roll.
    # 'base' es el divisor comun de las tres partes, y tiene que ser el precio
    # CRUDO que tenia ayer el contrato que hoy esta activo. Con el ajuste
    # multiplicativo el factor es comun al numerador y al denominador y se
    # cancela, asi que el nivel ajustado de ayer ya sirve de divisor. Con el
    # aditivo NO se cancela: el sumando sobrevive en el denominador, y como
    # _ajustes_hacia_atras acumula todos los escalones posteriores, el
    # rendimiento de una fecha pasada cambiaria al llegar dato futuro. El crudo
    # se reconstruye restando el ajuste de HOY al nivel de ayer, porque el ajuste
    # de hoy es el del contrato que hoy se tiene.
    if metodo == "ratio":
        if (ajuste <= 0.0).any():
            raise ErrorDatos(
                "El ajuste multiplicativo debe ser positivo: la serie no procede de "
                "serie_continua con metodo='ratio'."
            )
        base = anterior
        ingenuo = precio * ajuste.shift(1) / ajuste
    else:
        base = anterior - ajuste
        ingenuo = precio - ajuste + ajuste.shift(1)

    divisor = base.iloc[1:]
    if not bool((divisor > 0.0).all()):
        flojas = divisor.index[~(divisor > 0.0)]
        raise ErrorDatos(
            f"El precio del que se parte no es positivo el {flojas[0].date()} "
            f"({divisor.loc[flojas[0]]}), y un rendimiento relativo sobre un nivel "
            "nulo o negativo sale con el signo invertido. Una serie que cruza el "
            "cero no tiene rendimiento porcentual: mida la variacion absoluta con "
            "la columna 'precio' y .diff()."
        )

    total = (precio - anterior) / base
    parte_precio = (ingenuo - anterior) / base
    salida = pd.DataFrame(
        {
            "rendimiento_total": total,
            "rendimiento_precio": parte_precio,
            "rendimiento_roll": total - parte_precio,
        }
    ).iloc[1:]
    if "contrato" in marco.columns:
        salida["contrato"] = marco["contrato"].iloc[1:]
    salida.attrs["metodo"] = metodo
    return salida
