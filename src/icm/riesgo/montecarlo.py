"""Simulacion de Monte Carlo y bootstrap historico.

Los metodos parametricos dan una cifra; la simulacion da la distribucion
entera de resultados posibles, que es lo que permite responder preguntas del
tipo "que probabilidad hay de perder mas de un 10 % en un mes".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos, como_matriz, normalizar_pesos, validar_nivel


def _factor_cholesky(cov: np.ndarray) -> np.ndarray:
    """Descomposicion de Cholesky, con un ajuste si la matriz esta al limite.

    Una matriz de covarianzas estimada puede salir con autovalores minimamente
    negativos por error numerico. En ese caso se proyecta sobre el cono de las
    semidefinidas positivas en lugar de fallar.
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        autovalores, autovectores = np.linalg.eigh(cov)
        autovalores = np.clip(autovalores, 1e-12, None)
        reparada = autovectores @ np.diag(autovalores) @ autovectores.T
        return np.linalg.cholesky(reparada)


def simular_rendimientos(
    mu,
    cov,
    n_simulaciones: int = 10_000,
    horizonte: int = 1,
    distribucion: str = "normal",
    grados_libertad: float = 5.0,
    semilla: int | None = 42,
) -> np.ndarray:
    """Genera rendimientos multivariantes acumulados al horizonte pedido.

    Parametros
    ----------
    distribucion:
        ``"normal"`` o ``"t"``. La t multivariante reproduce el hecho de que
        los mercados caen juntos en los dias malos, algo que la normal
        subestima gravemente.

    Devuelve
    --------
    Matriz ``(n_simulaciones, n_activos)`` con el rendimiento simple acumulado
    de cada activo a lo largo del horizonte.
    """
    marco_cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(np.asarray(cov, float))
    matriz = marco_cov.to_numpy(float)
    n_activos = matriz.shape[0]
    vector_mu = (
        mu.reindex(marco_cov.columns).to_numpy(float)
        if isinstance(mu, pd.Series)
        else np.asarray(mu, float)
    )
    if vector_mu.shape != (n_activos,):
        raise ErrorDatos("Las rentabilidades esperadas no encajan con la matriz de covarianzas.")
    if horizonte < 1:
        raise ErrorDatos("El horizonte debe ser de al menos un periodo.")

    generador = np.random.default_rng(semilla)
    factor = _factor_cholesky(matriz)

    if distribucion == "normal":
        crudos = generador.standard_normal((n_simulaciones, horizonte, n_activos))
    elif distribucion == "t":
        if grados_libertad <= 2:
            raise ErrorDatos("Los grados de libertad deben ser mayores que 2.")
        normales = generador.standard_normal((n_simulaciones, horizonte, n_activos))
        mezcla = generador.chisquare(grados_libertad, size=(n_simulaciones, horizonte, 1))
        crudos = normales / np.sqrt(mezcla / grados_libertad)
        crudos *= np.sqrt((grados_libertad - 2.0) / grados_libertad)
    else:
        raise ErrorDatos(f"Distribucion '{distribucion}' desconocida: use 'normal' o 't'.")

    choques = crudos @ factor.T
    logaritmicos = vector_mu + choques
    # Se acumulan en logaritmos y se devuelve el rendimiento simple del tramo.
    return np.expm1(logaritmicos.sum(axis=1))


def simular_cartera(
    mu,
    cov,
    pesos,
    n_simulaciones: int = 10_000,
    horizonte: int = 1,
    distribucion: str = "normal",
    grados_libertad: float = 5.0,
    semilla: int | None = 42,
) -> pd.Series:
    """Distribucion simulada del rendimiento de la cartera al horizonte."""
    marco_cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(np.asarray(cov, float))
    vector_pesos = normalizar_pesos(pesos, marco_cov.columns).to_numpy(float)
    simulaciones = simular_rendimientos(
        mu, cov, n_simulaciones, horizonte, distribucion, grados_libertad, semilla
    )
    return pd.Series(simulaciones @ vector_pesos, name="rendimiento_simulado")


def bootstrap_historico(
    rendimientos_activos,
    pesos,
    n_simulaciones: int = 10_000,
    horizonte: int = 1,
    tamano_bloque: int | None = None,
    semilla: int | None = 42,
) -> pd.Series:
    """Remuestrea la historia real en lugar de suponer una distribucion.

    Con ``tamano_bloque`` se remuestrean tramos consecutivos en vez de dias
    sueltos (bootstrap por bloques), lo que conserva el agrupamiento de
    volatilidad: en la realidad los dias malos vienen seguidos, y muestrear al
    azar destruiria justo esa dependencia.

    El bootstrap por bloques es **circular**: el bloque que empieza cerca del
    final continua por el principio de la muestra. Sin eso, las observaciones
    de los extremos entran muchas menos veces que las del centro -- con 60
    observaciones y bloques de 10, diez veces menos -- y como las series
    financieras van en orden cronologico, eso penaliza sistematicamente las
    sesiones **mas recientes**. Medido: la misma crisis de diez sesiones,
    movida del centro al final de la muestra, perdia el 41 % de su
    probabilidad de cola.
    """
    marco = como_matriz(rendimientos_activos, "rendimientos")
    vector_pesos = normalizar_pesos(pesos, marco.columns).to_numpy(float)
    historico = marco.to_numpy(float)
    n_observaciones = len(historico)
    if horizonte < 1:
        raise ErrorDatos("El horizonte debe ser de al menos un periodo.")
    if n_observaciones < horizonte:
        raise ErrorDatos("No hay suficientes observaciones para ese horizonte.")

    generador = np.random.default_rng(semilla)

    if tamano_bloque is None:
        indices = generador.integers(0, n_observaciones, size=(n_simulaciones, horizonte))
    else:
        if tamano_bloque < 1:
            raise ErrorDatos("El tamanio de bloque debe ser positivo.")
        if tamano_bloque > n_observaciones:
            raise ErrorDatos(
                f"El bloque ({tamano_bloque}) no cabe en la muestra "
                f"({n_observaciones} observaciones): todas las trayectorias serian "
                "la misma. Reduzca 'tamano_bloque' o alargue la historia."
            )
        n_bloques = int(np.ceil(horizonte / tamano_bloque))
        # Circular: cualquier observacion puede iniciar un bloque, de modo que
        # todas entran exactamente 'tamano_bloque' veces.
        inicios = generador.integers(0, n_observaciones, size=(n_simulaciones, n_bloques))
        desplazamientos = np.arange(tamano_bloque)
        indices = (inicios[:, :, None] + desplazamientos[None, None, :]) % n_observaciones
        indices = indices.reshape(n_simulaciones, -1)[:, :horizonte]

    trayectorias = historico[indices]  # (n_simulaciones, horizonte, n_activos)
    # La suma ponderada de rendimientos solo es exacta en convencion SIMPLE:
    # el logaritmo de una suma no es la suma de logaritmos. La marca que deja
    # 'icm.datos.precios.rendimientos' dice en cual estamos, y antes no se
    # miraba: con rendimientos logaritmicos la media a 21 dias se desviaba
    # entre un 118 % y un 262 %.
    if _es_logaritmico(rendimientos_activos, marco):
        trayectorias = np.expm1(trayectorias)
    rendimiento_cartera = trayectorias @ vector_pesos
    acumulado = np.prod(1.0 + rendimiento_cartera, axis=1) - 1.0
    return pd.Series(acumulado, name="rendimiento_bootstrap")


def _es_logaritmico(original, marco: pd.DataFrame) -> bool:
    """Lee la convencion de la marca que deja ``rendimientos``."""
    for fuente in (original, marco):
        marca = getattr(fuente, "attrs", {}).get("tipo_rendimiento")
        if marca is not None:
            return str(marca) == "log"
    return False


def var_montecarlo(simulaciones, nivel: float = 0.99) -> float:
    """Cuantil de perdida de una distribucion simulada, como numero positivo."""
    nivel = validar_nivel(nivel)
    valores = np.asarray(simulaciones, dtype=float)
    return float(-np.quantile(valores, 1.0 - nivel))


def es_montecarlo(simulaciones, nivel: float = 0.99) -> float:
    """Perdida media mas alla del VaR en una distribucion simulada."""
    nivel = validar_nivel(nivel)
    valores = np.asarray(simulaciones, dtype=float)
    umbral = np.quantile(valores, 1.0 - nivel)
    cola = valores[valores <= umbral]
    return float(-cola.mean()) if cola.size else float(-umbral)


def resumen_simulacion(
    simulaciones,
    niveles=(0.95, 0.99),
    valor_cartera: float | None = None,
    umbrales_perdida=(0.05, 0.10, 0.20),
) -> pd.Series:
    """Resume una distribucion simulada: cuantiles, VaR, ES y probabilidades de perdida."""
    valores = np.asarray(simulaciones, dtype=float)
    if valores.size < 2:
        raise ErrorDatos(
            f"Hacen falta al menos 2 simulaciones para resumir una distribucion; "
            f"se han recibido {valores.size}. Con una, la desviacion sale NaN y el "
            "VaR es el propio valor."
        )
    if not np.isfinite(valores).any():
        raise ErrorDatos("Todas las simulaciones son NaN o infinitas.")
    resumen = {
        "simulaciones": float(valores.size),
        "media": float(valores.mean()),
        "mediana": float(np.median(valores)),
        "desviacion": float(valores.std(ddof=1)),
        "minimo": float(valores.min()),
        "maximo": float(valores.max()),
        "prob_perdida": float((valores < 0).mean()),
    }
    # Las etiquetas llevan los decimales que hagan falta: con '{:.0%}', el
    # 99,5 % y el 99,9 % daban los dos "100%", el segundo pisaba al primero
    # dentro del diccionario y el que quedaba salia con una etiqueta que no
    # era la suya.
    for nivel in niveles:
        resumen[f"var_{_etiqueta_nivel(nivel)}"] = var_montecarlo(valores, nivel)
        resumen[f"es_{_etiqueta_nivel(nivel)}"] = es_montecarlo(valores, nivel)
    for umbral in umbrales_perdida:
        resumen[f"prob_perder_mas_{_etiqueta_nivel(umbral)}"] = float((valores < -umbral).mean())
    if valor_cartera is not None:
        for nivel in niveles:
            etiqueta = _etiqueta_nivel(nivel)
            resumen[f"var_{etiqueta}_monetario"] = resumen[f"var_{etiqueta}"] * valor_cartera
            resumen[f"es_{etiqueta}_monetario"] = resumen[f"es_{etiqueta}"] * valor_cartera
    return pd.Series(resumen, name="simulacion")


def _etiqueta_nivel(nivel: float) -> str:
    """Porcentaje con los decimales justos para no colisionar."""
    texto = f"{nivel * 100:.4f}".rstrip("0").rstrip(".")
    return f"{texto}%"


def trayectorias_simuladas(
    mu,
    cov,
    pesos,
    horizonte: int = 252,
    n_trayectorias: int = 200,
    valor_inicial: float = 100.0,
    distribucion: str = "normal",
    grados_libertad: float = 5.0,
    semilla: int | None = 42,
) -> pd.DataFrame:
    """Genera trayectorias completas del valor de la cartera, para representarlas.

    A diferencia de :func:`simular_cartera`, que solo devuelve el resultado
    final, aqui se conserva el camino, que es lo que permite estudiar caidas
    intermedias y no solo el punto de llegada.
    """
    marco_cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(np.asarray(cov, float))
    vector_pesos = normalizar_pesos(pesos, marco_cov.columns).to_numpy(float)
    matriz = marco_cov.to_numpy(float)
    vector_mu = (
        mu.reindex(marco_cov.columns).to_numpy(float)
        if isinstance(mu, pd.Series)
        else np.asarray(mu, float)
    )

    generador = np.random.default_rng(semilla)
    factor = _factor_cholesky(matriz)
    n_activos = matriz.shape[0]

    if distribucion == "normal":
        crudos = generador.standard_normal((n_trayectorias, horizonte, n_activos))
    elif distribucion == "t":
        normales = generador.standard_normal((n_trayectorias, horizonte, n_activos))
        mezcla = generador.chisquare(grados_libertad, size=(n_trayectorias, horizonte, 1))
        crudos = normales / np.sqrt(mezcla / grados_libertad)
        crudos *= np.sqrt((grados_libertad - 2.0) / grados_libertad)
    else:
        raise ErrorDatos(f"Distribucion '{distribucion}' desconocida: use 'normal' o 't'.")

    logaritmicos = vector_mu + crudos @ factor.T
    por_periodo = np.expm1(logaritmicos) @ vector_pesos
    valor = valor_inicial * np.cumprod(1.0 + por_periodo, axis=1)
    valor = np.hstack([np.full((n_trayectorias, 1), valor_inicial), valor])

    return pd.DataFrame(
        valor.T,
        index=pd.RangeIndex(0, horizonte + 1, name="periodo"),
        columns=[f"trayectoria_{i}" for i in range(n_trayectorias)],
    )
