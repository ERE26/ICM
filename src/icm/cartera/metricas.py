"""Medidas de rentabilidad y de rentabilidad ajustada al riesgo.

Salvo que se indique lo contrario, las funciones aceptan rendimientos simples
o logaritmicos y trabajan sobre la serie tal cual se les entrega. Las que
necesitan componer (la rentabilidad acumulada, por ejemplo) miran la marca
``tipo_rendimiento`` que deja :func:`icm.datos.rendimientos`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos, como_serie, periodos_por_anio


def _es_logaritmico(serie: pd.Series, tipo: str | None) -> bool:
    if tipo is not None:
        if tipo not in {"log", "simple"}:
            raise ErrorDatos(f"Tipo de rendimiento '{tipo}' desconocido: use 'log' o 'simple'.")
        return tipo == "log"
    return serie.attrs.get("tipo_rendimiento") == "log"


def _a_simples(serie: pd.Series, tipo: str | None) -> pd.Series:
    return np.expm1(serie) if _es_logaritmico(serie, tipo) else serie


def rentabilidad_acumulada(rendimientos, tipo: str | None = None) -> float:
    """Rentabilidad total del periodo, como fraccion (0.25 = +25 %)."""
    serie = _a_simples(como_serie(rendimientos, "rendimientos"), tipo)
    return float((1.0 + serie).prod() - 1.0)


def rentabilidad_anualizada(
    rendimientos,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
    tipo: str | None = None,
) -> float:
    """Tasa geometrica anual equivalente (CAGR) de la serie."""
    serie = _a_simples(como_serie(rendimientos, "rendimientos"), tipo)
    n = periodos_por_anio(periodos_anio)
    crecimiento = float((1.0 + serie).prod())
    if crecimiento <= 0:
        # La cartera se ha ido a cero: la tasa anual no esta definida.
        return -1.0
    anios = len(serie) / n
    return float(crecimiento ** (1.0 / anios) - 1.0)


def volatilidad_anualizada(
    rendimientos,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
) -> float:
    """Desviacion tipica anualizada por la raiz del tiempo."""
    serie = como_serie(rendimientos, "rendimientos")
    return float(serie.std(ddof=1) * np.sqrt(periodos_por_anio(periodos_anio)))


def semidesviacion_anualizada(
    rendimientos,
    objetivo: float = 0.0,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
) -> float:
    """Volatilidad calculada solo con los rendimientos por debajo del objetivo.

    Es el denominador del ratio de Sortino: penaliza unicamente las caidas, no
    las subidas, que es lo que de verdad preocupa al inversor.
    """
    serie = como_serie(rendimientos, "rendimientos")
    desviaciones = np.minimum(serie - objetivo, 0.0)
    semivarianza = float((desviaciones**2).sum() / max(len(serie) - 1, 1))
    return float(np.sqrt(semivarianza * periodos_por_anio(periodos_anio)))


def ratio_sharpe(
    rendimientos,
    tasa_libre_riesgo: float = 0.0,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
    tipo: str | None = None,
    media: str = "geometrica",
) -> float:
    """Exceso de rentabilidad anual por unidad de volatilidad.

    ``tasa_libre_riesgo`` se expresa en terminos anuales.

    Cual de las dos medias, y por que importa
    -----------------------------------------
    Con ``media="geometrica"`` (lo que hay por omision) el numerador es la
    tasa anual compuesta, que es lo que de verdad gano el inversor. Con
    ``media="aritmetica"`` es la media de los rendimientos multiplicada por
    los periodos del anio, que es la definicion de libro de Sharpe y la unica
    cuya distribucion muestral se conoce.

    **No son la misma cifra, y la diferencia no es pequenia.** El numerador
    geometrico es aproximadamente el aritmetico menos ``sigma^2 / 2``, asi que
    con volatilidad alta el Sharpe geometrico sale sistematicamente por
    debajo. Con rendimientos de media 0,05 % y volatilidad 2 % diarias, esa
    resta vale ``0,02^2 * 252 / 2 = 5,04`` puntos porcentuales de rentabilidad
    anual sobre los 12,6 que da la media aritmetica: medido con 200 000
    observaciones, **+0,2488 geometrico frente a +0,3985 aritmetico**, un
    37,6 % de hueco contra el 40,0 % que predice la formula.

    En muestras cortas conviene mirar la diferencia absoluta y no el
    porcentaje: cuando el Sharpe ronda cero, el cociente entre los dos se
    dispara sin que pase nada (sobre ocho semillas de 2 000 observaciones el
    "hueco" va del -87 % al +317 %), mientras que la resta en el numerador
    sigue siendo esos 5,04 puntos.

    Esto importa porque el paquete usa las dos. ``icm.ml.estrategia`` calcula
    el Sharpe **aritmetico** en ``resumen_estrategia`` y en
    ``coste_de_equilibrio``, porque ahi se comparan estrategias entre si y hace
    falta la definicion con distribucion conocida. Esta funcion usa la
    geometrica porque describe una serie ya realizada. Comparar un numero de
    un modulo con uno del otro **no es una comparacion valida**, y antes nada
    lo decia.
    """
    if media not in {"geometrica", "aritmetica"}:
        raise ErrorDatos(f"Media '{media}' desconocida: use 'geometrica' o 'aritmetica'.")
    serie = como_serie(rendimientos, "rendimientos")
    n = periodos_por_anio(periodos_anio)
    volatilidad = volatilidad_anualizada(serie, n)
    if volatilidad == 0:
        raise ErrorDatos("La volatilidad es cero: el ratio de Sharpe no esta definido.")
    if media == "aritmetica":
        anual = float(_a_simples(serie, tipo).mean()) * n
    else:
        anual = rentabilidad_anualizada(serie, n, tipo)
    return float((anual - tasa_libre_riesgo) / volatilidad)


def ratio_sortino(
    rendimientos,
    tasa_libre_riesgo: float = 0.0,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
    tipo: str | None = None,
) -> float:
    """Como el Sharpe, pero dividiendo solo por el riesgo de caida."""
    serie = como_serie(rendimientos, "rendimientos")
    objetivo_periodo = tasa_libre_riesgo / periodos_por_anio(periodos_anio)
    semidesviacion = semidesviacion_anualizada(serie, objetivo_periodo, periodos_anio)
    if semidesviacion == 0:
        raise ErrorDatos("No hay rendimientos por debajo del objetivo: Sortino no definido.")
    exceso = rentabilidad_anualizada(serie, periodos_anio, tipo) - tasa_libre_riesgo
    return float(exceso / semidesviacion)


def ratio_calmar(
    rendimientos,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
    tipo: str | None = None,
) -> float:
    """Rentabilidad anual dividida por la maxima caida sufrida."""
    from icm.riesgo.caidas import maxima_caida  # import local: evita un ciclo

    serie = como_serie(rendimientos, "rendimientos")
    caida = abs(maxima_caida(serie, tipo=tipo))
    if caida == 0:
        raise ErrorDatos("La maxima caida es cero: el ratio de Calmar no esta definido.")
    return float(rentabilidad_anualizada(serie, periodos_anio, tipo) / caida)


def beta(rendimientos_activo, rendimientos_mercado) -> float:
    """Sensibilidad del activo a los movimientos del mercado."""
    activo = como_serie(rendimientos_activo, "activo")
    mercado = como_serie(rendimientos_mercado, "mercado")
    comun = activo.index.intersection(mercado.index)
    if len(comun) < 3:
        raise ErrorDatos("Hacen falta al menos 3 observaciones comunes para estimar la beta.")
    activo, mercado = activo.loc[comun], mercado.loc[comun]
    varianza = float(mercado.var(ddof=1))
    if varianza == 0:
        raise ErrorDatos("El mercado no tiene varianza: la beta no esta definida.")
    return float(activo.cov(mercado) / varianza)


def alfa_jensen(
    rendimientos_activo,
    rendimientos_mercado,
    tasa_libre_riesgo: float = 0.0,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
    tipo: str | None = None,
) -> float:
    """Rentabilidad anual no explicada por la exposicion al mercado (CAPM)."""
    n = periodos_por_anio(periodos_anio)
    b = beta(rendimientos_activo, rendimientos_mercado)
    r_activo = rentabilidad_anualizada(rendimientos_activo, n, tipo)
    r_mercado = rentabilidad_anualizada(rendimientos_mercado, n, tipo)
    return float(r_activo - (tasa_libre_riesgo + b * (r_mercado - tasa_libre_riesgo)))


def error_seguimiento(
    rendimientos_cartera,
    rendimientos_referencia,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
) -> float:
    """Volatilidad anualizada de la diferencia frente al indice de referencia."""
    cartera = como_serie(rendimientos_cartera, "cartera")
    referencia = como_serie(rendimientos_referencia, "referencia")
    comun = cartera.index.intersection(referencia.index)
    diferencia = cartera.loc[comun] - referencia.loc[comun]
    return volatilidad_anualizada(diferencia, periodos_anio)


def ratio_informacion(
    rendimientos_cartera,
    rendimientos_referencia,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
    tipo: str | None = None,
) -> float:
    """Exceso sobre el indice por unidad de error de seguimiento."""
    seguimiento = error_seguimiento(rendimientos_cartera, rendimientos_referencia, periodos_anio)
    if seguimiento == 0:
        raise ErrorDatos("El error de seguimiento es cero: el ratio de informacion no existe.")
    exceso = rentabilidad_anualizada(rendimientos_cartera, periodos_anio, tipo) - (
        rentabilidad_anualizada(rendimientos_referencia, periodos_anio, tipo)
    )
    return float(exceso / seguimiento)


def asimetria(rendimientos) -> float:
    """Tercer momento estandarizado: mide la falta de simetria de la distribucion."""
    return float(stats.skew(como_serie(rendimientos, "rendimientos"), bias=False))


def curtosis(rendimientos, exceso: bool = True) -> float:
    """Cuarto momento estandarizado: mide el grosor de las colas.

    Con ``exceso=True`` se resta 3, de modo que la normal marca 0 y cualquier
    valor positivo indica colas mas gruesas que la normal.
    """
    serie = como_serie(rendimientos, "rendimientos")
    return float(stats.kurtosis(serie, fisher=exceso, bias=False))


def contraste_jarque_bera(rendimientos) -> dict[str, float]:
    """Contraste de normalidad basado en la asimetria y la curtosis.

    Un p-valor pequenio rechaza la normalidad, lo que en la practica significa
    que el VaR parametrico normal infravalorara el riesgo de cola.
    """
    serie = como_serie(rendimientos, "rendimientos")
    estadistico, p_valor = stats.jarque_bera(serie)
    return {
        "estadistico": float(estadistico),
        "p_valor": float(p_valor),
        "normal_al_5pct": bool(p_valor > 0.05),
    }


def resumen_metricas(
    rendimientos,
    tasa_libre_riesgo: float = 0.0,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
    tipo: str | None = None,
) -> pd.Series:
    """Tabla resumen con las medidas habituales de una serie de rendimientos."""
    from icm.riesgo.caidas import maxima_caida
    from icm.riesgo.var import es_historico, var_historico

    serie = como_serie(rendimientos, "rendimientos")
    n = periodos_por_anio(periodos_anio)

    def _seguro(funcion, *args, **kwargs):
        """Algunas medidas no existen en casos degenerados; se devuelve NaN."""
        try:
            return funcion(*args, **kwargs)
        except ErrorDatos:
            return float("nan")

    return pd.Series(
        {
            "observaciones": float(len(serie)),
            "rentabilidad_acumulada": rentabilidad_acumulada(serie, tipo),
            "rentabilidad_anualizada": rentabilidad_anualizada(serie, n, tipo),
            "volatilidad_anualizada": volatilidad_anualizada(serie, n),
            "ratio_sharpe": _seguro(ratio_sharpe, serie, tasa_libre_riesgo, n, tipo),
            "ratio_sortino": _seguro(ratio_sortino, serie, tasa_libre_riesgo, n, tipo),
            "ratio_calmar": _seguro(ratio_calmar, serie, n, tipo),
            "maxima_caida": maxima_caida(serie, tipo=tipo),
            "asimetria": asimetria(serie),
            "curtosis_exceso": curtosis(serie),
            "var_historico_95": var_historico(serie, 0.95),
            "es_historico_95": es_historico(serie, 0.95),
        },
        name=getattr(serie, "name", "serie") or "serie",
    )
