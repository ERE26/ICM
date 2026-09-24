"""Valor en Riesgo (VaR) y Perdida Esperada (Expected Shortfall).

Convencion de signos
--------------------
Todas las funciones devuelven la perdida como **numero positivo**. Un VaR al
99 % de 0.031 significa "se espera perder mas del 3,1 % solo el 1 % de los
dias". Es la convencion de los comites de riesgos y evita el baile de signos
que tanto confunde al comparar modelos.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos, como_serie, validar_nivel

METODOS = ("historico", "parametrico_normal", "parametrico_t", "cornish_fisher")


def var_historico(rendimientos, nivel: float = 0.99) -> float:
    """VaR por simulacion historica: el cuantil empirico de las perdidas.

    No supone ninguna distribucion, asi que recoge de forma natural la
    asimetria y las colas gruesas. A cambio, no puede decir nada de perdidas
    mayores que la peor observada en la muestra.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    return float(-np.quantile(serie, 1.0 - nivel))


def es_historico(rendimientos, nivel: float = 0.99) -> float:
    """Perdida media condicionada a superar el VaR historico.

    A diferencia del VaR, el ES si mira dentro de la cola, y es una medida
    coherente (subaditiva): diversificar nunca puede empeorarlo. Es la medida
    que Basilea III adopto en sustitucion del VaR.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    umbral = np.quantile(serie, 1.0 - nivel)
    cola = serie[serie <= umbral]
    if cola.empty:  # pragma: no cover - solo con muestras minusculas
        return float(-umbral)
    return float(-cola.mean())


def var_parametrico(
    rendimientos,
    nivel: float = 0.99,
    distribucion: str = "normal",
) -> float:
    """VaR suponiendo una distribucion parametrica de los rendimientos.

    Parametros
    ----------
    distribucion:
        ``"normal"`` es el modelo clasico de varianza-covarianza; ``"t"`` ajusta
        una t de Student por maxima verosimilitud, lo que da colas mas gruesas
        y un VaR mas prudente cuando los datos no son normales.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    alfa = 1.0 - nivel

    if distribucion == "normal":
        media, desviacion = float(serie.mean()), float(serie.std(ddof=1))
        return float(-(media + desviacion * stats.norm.ppf(alfa)))
    if distribucion == "t":
        gl, loc, escala = stats.t.fit(serie)
        return float(-(loc + escala * stats.t.ppf(alfa, gl)))
    raise ErrorDatos(f"Distribucion '{distribucion}' desconocida: use 'normal' o 't'.")


def es_parametrico(
    rendimientos,
    nivel: float = 0.99,
    distribucion: str = "normal",
) -> float:
    """Perdida esperada bajo la distribucion parametrica ajustada."""
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    alfa = 1.0 - nivel

    if distribucion == "normal":
        media, desviacion = float(serie.mean()), float(serie.std(ddof=1))
        densidad = stats.norm.pdf(stats.norm.ppf(alfa))
        return float(-(media - desviacion * densidad / alfa))
    if distribucion == "t":
        gl, loc, escala = stats.t.fit(serie)
        cuantil = stats.t.ppf(alfa, gl)
        # Formula cerrada del ES de la t de Student estandarizada.
        factor = (gl + cuantil**2) / (gl - 1.0) * stats.t.pdf(cuantil, gl) / alfa
        return float(-(loc - escala * factor))
    raise ErrorDatos(f"Distribucion '{distribucion}' desconocida: use 'normal' o 't'.")


def _momentos(serie: pd.Series) -> tuple[float, float, float, float]:
    """Media, desviacion tipica, asimetria y exceso de curtosis de la serie."""
    return (
        float(serie.mean()),
        float(serie.std(ddof=1)),
        float(stats.skew(serie, bias=False)),
        float(stats.kurtosis(serie, fisher=True, bias=False)),
    )


def _cuantil_cornish_fisher(probabilidad, asimetria: float, curtosis: float):
    """Cuantil normal corregido por los momentos de orden tres y cuatro."""
    z = stats.norm.ppf(probabilidad)
    return (
        z
        + (z**2 - 1.0) * asimetria / 6.0
        + (z**3 - 3.0 * z) * curtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * asimetria**2 / 36.0
    )


def dominio_valido_cornish_fisher(asimetria: float, curtosis: float) -> bool:
    """Indica si la expansion de Cornish-Fisher sigue siendo una funcion de cuantiles.

    La expansion es asintotica: solo vale para asimetria y curtosis moderadas.
    Pasado cierto punto el polinomio deja de ser creciente, y una funcion de
    cuantiles que no crece no es una funcion de cuantiles: el "VaR" resultante
    puede ser menor al 99 % que al 95 %, o dispararse a decenas de sigmas.

    La derivada del polinomio respecto a ``z`` es una cuadratica; el criterio
    (Maillard, 2012) es que no tenga raices reales y se abra hacia arriba.

    >>> dominio_valido_cornish_fisher(0.0, 0.0)     # la normal
    True
    >>> dominio_valido_cornish_fisher(-1.2, 29.0)   # colas muy gruesas
    False
    """
    s, k = float(asimetria), float(curtosis)
    a = k / 8.0 - s**2 / 6.0
    b = s / 3.0
    c = 1.0 - k / 8.0 + 5.0 * s**2 / 36.0

    if abs(a) < 1e-12:
        return abs(b) < 1e-12 and c >= 0.0
    if a < 0.0:
        return False
    return (b**2 - 4.0 * a * c) <= 0.0


def _avisar_si_fuera_de_dominio(asimetria: float, curtosis: float) -> None:
    if dominio_valido_cornish_fisher(asimetria, curtosis):
        return
    warnings.warn(
        f"Cornish-Fisher fuera de su dominio de validez (asimetria {asimetria:.2f}, "
        f"exceso de curtosis {curtosis:.2f}): la expansion deja de ser monotona y el "
        "valor devuelto no es un cuantil fiable. Use el metodo historico o la t de "
        "Student con colas tan gruesas.",
        UserWarning,
        stacklevel=3,
    )


def var_cornish_fisher(rendimientos, nivel: float = 0.99) -> float:
    """VaR normal corregido por asimetria y curtosis.

    La expansion de Cornish-Fisher desplaza el cuantil normal en funcion de los
    momentos de tercer y cuarto orden. Es un termino medio barato entre el
    parametrico normal y el historico: mantiene la formula cerrada pero deja de
    suponer simetria y colas normales.

    Ojo: es una expansion asintotica. Con colas muy gruesas se sale de su
    dominio de validez y el resultado deja de ser un cuantil; en ese caso se
    emite un ``UserWarning``. Vease :func:`dominio_valido_cornish_fisher`.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    media, desviacion, s, k = _momentos(serie)
    _avisar_si_fuera_de_dominio(s, k)
    return float(-(media + desviacion * _cuantil_cornish_fisher(1.0 - nivel, s, k)))


def es_cornish_fisher(rendimientos, nivel: float = 0.99, n_tramos: int = 1000) -> float:
    """ES obtenido integrando numericamente el VaR de Cornish-Fisher en la cola.

    No existe una formula cerrada, asi que se promedia el VaR de Cornish-Fisher
    sobre los niveles de la cola, que es la definicion misma del ES. Los
    momentos se calculan una sola vez: solo cambia el cuantil.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    alfa = 1.0 - nivel
    media, desviacion, s, k = _momentos(serie)
    _avisar_si_fuera_de_dominio(s, k)
    # Se evita el extremo alfa=0, donde el cuantil diverge.
    probabilidades = np.linspace(alfa / n_tramos, alfa, n_tramos)
    cuantiles = _cuantil_cornish_fisher(probabilidades, s, k)
    return float(-(media + desviacion * cuantiles.mean()))


def var(
    rendimientos,
    nivel: float = 0.99,
    metodo: str = "historico",
) -> float:
    """Punto de entrada unico a los distintos metodos de VaR."""
    if metodo == "historico":
        return var_historico(rendimientos, nivel)
    if metodo == "parametrico_normal":
        return var_parametrico(rendimientos, nivel, "normal")
    if metodo == "parametrico_t":
        return var_parametrico(rendimientos, nivel, "t")
    if metodo == "cornish_fisher":
        return var_cornish_fisher(rendimientos, nivel)
    raise ErrorDatos(f"Metodo '{metodo}' desconocido. Disponibles: {', '.join(METODOS)}.")


def expected_shortfall(
    rendimientos,
    nivel: float = 0.99,
    metodo: str = "historico",
) -> float:
    """Punto de entrada unico a los distintos metodos de ES."""
    if metodo == "historico":
        return es_historico(rendimientos, nivel)
    if metodo == "parametrico_normal":
        return es_parametrico(rendimientos, nivel, "normal")
    if metodo == "parametrico_t":
        return es_parametrico(rendimientos, nivel, "t")
    if metodo == "cornish_fisher":
        return es_cornish_fisher(rendimientos, nivel)
    raise ErrorDatos(f"Metodo '{metodo}' desconocido. Disponibles: {', '.join(METODOS)}.")


def escalar_horizonte(medida: float, dias: int, metodo: str = "raiz_tiempo") -> float:
    """Lleva una medida de riesgo diaria a un horizonte de varios dias.

    La regla de la raiz del tiempo supone rendimientos independientes y sin
    autocorrelacion. Con volatilidad agrupada (que es lo normal) infravalora el
    riesgo a horizontes largos, asi que conviene usarla solo para 1-10 dias,
    que es justo el rango que exige la normativa.
    """
    if dias < 1:
        raise ErrorDatos("El horizonte debe ser de al menos un dia.")
    if metodo != "raiz_tiempo":
        raise ErrorDatos(f"Metodo de escalado '{metodo}' desconocido.")
    return float(medida * np.sqrt(dias))


def tabla_medidas(
    rendimientos,
    niveles=(0.95, 0.99),
    metodos=METODOS,
    valor_cartera: float | None = None,
) -> pd.DataFrame:
    """Compara VaR y ES de todos los metodos y niveles en una sola tabla.

    Si se indica ``valor_cartera``, se aniaden las cifras en unidades
    monetarias, que es como se presentan en un comite.
    """
    serie = como_serie(rendimientos, "rendimientos")
    filas = []
    for metodo in metodos:
        for nivel in niveles:
            fila = {
                "metodo": metodo,
                "nivel": nivel,
                "var": var(serie, nivel, metodo),
                "es": expected_shortfall(serie, nivel, metodo),
            }
            if valor_cartera is not None:
                fila["var_monetario"] = fila["var"] * valor_cartera
                fila["es_monetario"] = fila["es"] * valor_cartera
            filas.append(fila)
    return pd.DataFrame(filas).set_index(["metodo", "nivel"])


def var_incremental(
    rendimientos_activos: pd.DataFrame,
    pesos,
    nivel: float = 0.99,
    metodo: str = "historico",
) -> pd.Series:
    """Cuanto cambia el VaR de la cartera al eliminar cada activo.

    Un valor positivo significa que el activo aumenta el VaR total; uno
    negativo, que lo reduce (diversifica).
    """
    from icm.comun import normalizar_pesos
    from icm.datos.precios import rendimientos_cartera

    marco = rendimientos_activos
    serie_pesos = normalizar_pesos(pesos, marco.columns)
    var_total = var(rendimientos_cartera(marco, serie_pesos), nivel, metodo)

    incrementos = {}
    for activo in marco.columns:
        restantes = [c for c in marco.columns if c != activo]
        if not restantes:
            incrementos[activo] = var_total
            continue
        pesos_sin = normalizar_pesos(serie_pesos.loc[restantes], restantes)
        var_sin = var(rendimientos_cartera(marco[restantes], pesos_sin), nivel, metodo)
        incrementos[activo] = var_total - var_sin
    return pd.Series(incrementos, name=f"var_incremental_{nivel:.0%}")


def var_anualizado(rendimientos, nivel: float = 0.99, metodo: str = "historico") -> float:
    """VaR diario escalado a un anio por la raiz del tiempo.

    Se ofrece por comodidad, pero con la advertencia de :func:`escalar_horizonte`:
    a un anio vista la regla es poco fiable.
    """
    return escalar_horizonte(var(rendimientos, nivel, metodo), DIAS_HABILES_ANIO)
