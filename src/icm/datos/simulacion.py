"""Proveedor de precios simulados, para trabajar sin conexion.

Sirve para tres cosas: que los tests sean deterministas y no toquen la red,
que los notebooks se puedan ejecutar en un entorno cerrado, y que los modelos
se puedan contrastar contra un proceso generador conocido.

Los rendimientos se simulan con innovaciones t de Student correlacionadas, de
modo que las series tienen colas mas gruesas que la normal, como las reales.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos
from icm.datos.proveedor import ProveedorDatos, normalizar_tickers, validar_marco_precios


def matriz_correlacion_uniforme(n_activos: int, correlacion: float) -> np.ndarray:
    """Matriz de correlacion con un unico valor fuera de la diagonal."""
    if not -1.0 / max(n_activos - 1, 1) < correlacion < 1.0:
        raise ErrorDatos(
            f"Una correlacion comun de {correlacion} no da una matriz definida positiva "
            f"con {n_activos} activos."
        )
    matriz = np.full((n_activos, n_activos), float(correlacion))
    np.fill_diagonal(matriz, 1.0)
    return matriz


def _factor_volatilidad_garch(
    n_periodos: int,
    alfa: float,
    beta: float,
    generador: np.random.Generator,
) -> np.ndarray:
    """Factor de volatilidad comun que sigue un GARCH(1,1) de varianza unitaria.

    Multiplica por igual a todos los activos, de modo que introduce
    agrupamiento de volatilidad sin alterar la correlacion entre ellos. El
    termino independiente se fija en ``1 - alfa - beta`` para que la varianza
    incondicional del factor sea exactamente 1 y la volatilidad objetivo del
    simulador se siga respetando.
    """
    omega = 1.0 - alfa - beta
    choques = generador.standard_normal(n_periodos)
    varianzas = np.empty(n_periodos)
    varianzas[0] = 1.0
    innovacion = np.sqrt(varianzas[0]) * choques[0]
    for t in range(1, n_periodos):
        varianzas[t] = omega + alfa * innovacion**2 + beta * varianzas[t - 1]
        innovacion = np.sqrt(varianzas[t]) * choques[t]
    return np.sqrt(varianzas)


def simular_precios(
    tickers: Sequence[str],
    n_periodos: int = 1000,
    inicio: str | pd.Timestamp = "2015-01-01",
    deriva_anual: float | Sequence[float] = 0.07,
    volatilidad_anual: float | Sequence[float] = 0.20,
    correlacion: float | np.ndarray = 0.35,
    precio_inicial: float = 100.0,
    grados_libertad: float = 5.0,
    garch_alfa: float = 0.08,
    garch_beta: float = 0.90,
    semilla: int | None = 42,
) -> pd.DataFrame:
    """Genera precios diarios correlacionados, con colas gruesas y volatilidad agrupada.

    Parametros
    ----------
    deriva_anual, volatilidad_anual:
        Un valor comun a todos los activos o uno por activo.
    correlacion:
        Correlacion comun entre activos, o directamente la matriz.
    grados_libertad:
        Grados de libertad de la t de Student. Cuanto menor, mas gruesas las
        colas. Con un valor alto (>50) el resultado es practicamente normal.
    garch_alfa, garch_beta:
        Parametros de un GARCH(1,1) sobre un factor de volatilidad comun a
        todos los activos, que reproduce el agrupamiento tipico de los
        mercados: las rachas turbulentas se suceden y las tranquilas tambien.
        Con ambos a cero la volatilidad es constante y los rendimientos
        quedan independientes. Su suma debe ser menor que 1.
    semilla:
        Fija el generador para que la simulacion sea reproducible.
    """
    tickers = normalizar_tickers(tickers)
    n_activos = len(tickers)
    if n_periodos < 2:
        raise ErrorDatos("Hacen falta al menos 2 periodos para simular precios.")
    if grados_libertad <= 2:
        raise ErrorDatos("Los grados de libertad deben ser mayores que 2 para que exista varianza.")
    if garch_alfa < 0 or garch_beta < 0:
        raise ErrorDatos("Los parametros del GARCH no pueden ser negativos.")
    if garch_alfa + garch_beta >= 1.0:
        raise ErrorDatos(
            "La suma de garch_alfa y garch_beta debe ser menor que 1 para que la "
            "varianza sea estacionaria."
        )

    deriva = np.broadcast_to(np.asarray(deriva_anual, dtype=float), (n_activos,)).copy()
    volatilidad = np.broadcast_to(np.asarray(volatilidad_anual, dtype=float), (n_activos,)).copy()
    if np.any(volatilidad <= 0):
        raise ErrorDatos("La volatilidad anual debe ser positiva.")

    if isinstance(correlacion, (int, float)):
        matriz_corr = matriz_correlacion_uniforme(n_activos, float(correlacion))
    else:
        matriz_corr = np.asarray(correlacion, dtype=float)
        if matriz_corr.shape != (n_activos, n_activos):
            raise ErrorDatos("La matriz de correlacion no encaja con el numero de activos.")

    generador = np.random.default_rng(semilla)
    factor = np.linalg.cholesky(matriz_corr)

    # t multivariante = normal correlacionada escalada por una chi-cuadrado
    # comun. Se reescala para que la varianza quede en 1 y la volatilidad
    # objetivo se respete.
    normales = generador.standard_normal((n_periodos, n_activos)) @ factor.T
    mezcla = generador.chisquare(grados_libertad, size=(n_periodos, 1)) / grados_libertad
    innovaciones = normales / np.sqrt(mezcla)
    innovaciones *= np.sqrt((grados_libertad - 2.0) / grados_libertad)

    if garch_alfa + garch_beta > 0:
        innovaciones *= _factor_volatilidad_garch(
            n_periodos, garch_alfa, garch_beta, generador
        )[:, None]

    paso = 1.0 / DIAS_HABILES_ANIO
    vol_periodo = volatilidad * np.sqrt(paso)
    deriva_periodo = (deriva - 0.5 * volatilidad**2) * paso
    rendimientos_log = deriva_periodo + vol_periodo * innovaciones

    fechas = pd.bdate_range(start=inicio, periods=n_periodos + 1, name="fecha")
    recorrido = np.vstack([np.zeros(n_activos), np.cumsum(rendimientos_log, axis=0)])
    precios = pd.DataFrame(
        precio_inicial * np.exp(recorrido), index=fechas, columns=tickers
    )
    return validar_marco_precios(precios, tickers)


class ProveedorSimulado(ProveedorDatos):
    """Proveedor que genera precios sinteticos en lugar de descargarlos.

    Respeta la ventana ``inicio``/``fin`` que se le pida recortando la serie
    simulada, de modo que es intercambiable con :class:`ProveedorYahoo`.
    """

    nombre = "simulado"

    def __init__(self, semilla: int | None = 42, **parametros) -> None:
        self.semilla = semilla
        self.parametros = parametros

    def descargar(
        self,
        tickers: Sequence[str],
        inicio: str | pd.Timestamp | None = None,
        fin: str | pd.Timestamp | None = None,
        intervalo: str = "1d",
    ) -> pd.DataFrame:
        if intervalo != "1d":
            raise ErrorDatos("El proveedor simulado solo genera series diarias.")
        parametros = dict(self.parametros)
        if inicio is not None:
            parametros.setdefault("inicio", inicio)
        if inicio is not None and fin is not None:
            habiles = len(pd.bdate_range(start=inicio, end=fin))
            parametros.setdefault("n_periodos", max(habiles - 1, 2))
        precios = simular_precios(tickers, semilla=self.semilla, **parametros)
        if inicio is not None:
            precios = precios.loc[precios.index >= pd.Timestamp(inicio)]
        if fin is not None:
            precios = precios.loc[precios.index <= pd.Timestamp(fin)]
        if precios.empty:
            raise ErrorDatos("La ventana solicitada no contiene sesiones simuladas.")
        return precios
