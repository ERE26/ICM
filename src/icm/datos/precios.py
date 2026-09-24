"""Carga de precios y transformacion en rendimientos."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos, ErrorDescarga, como_matriz, normalizar_pesos
from icm.datos.cache import CacheParquet, ProveedorConCache
from icm.datos.proveedor import ProveedorDatos
from icm.datos.simulacion import ProveedorSimulado
from icm.datos.yahoo import ProveedorYahoo


def proveedor_por_defecto(
    usar_cache: bool = True,
    directorio_cache: str | None = None,
) -> ProveedorDatos:
    """Yahoo Finance, con cache en disco salvo que se pida lo contrario."""
    proveedor: ProveedorDatos = ProveedorYahoo()
    if usar_cache:
        cache = CacheParquet(directorio_cache) if directorio_cache else CacheParquet()
        proveedor = ProveedorConCache(proveedor, cache)
    return proveedor


def cargar_precios(
    tickers: str | Sequence[str],
    inicio: str | pd.Timestamp | None = None,
    fin: str | pd.Timestamp | None = None,
    intervalo: str = "1d",
    proveedor: ProveedorDatos | None = None,
    respaldo_simulado: bool = False,
    semilla_respaldo: int = 42,
) -> pd.DataFrame:
    """Descarga precios de cierre ajustados.

    Parametros
    ----------
    proveedor:
        Fuente a utilizar. Si se omite, Yahoo Finance con cache.
    respaldo_simulado:
        Si la descarga falla (sin red, proveedor caido, dominio bloqueado) y
        esta opcion esta activa, se generan datos sinteticos y se avisa con un
        ``UserWarning``. Pensado para que un notebook siga siendo ejecutable en
        un entorno sin salida a internet; **nunca** para conclusiones reales.

    Devuelve
    --------
    DataFrame con el indice de fechas y una columna por ticker.
    """
    proveedor = proveedor if proveedor is not None else proveedor_por_defecto()
    try:
        return proveedor.descargar(tickers, inicio, fin, intervalo)
    except (ErrorDescarga, ErrorDatos, OSError) as exc:
        if not respaldo_simulado:
            raise
        warnings.warn(
            f"No se ha podido descargar de '{proveedor.nombre}' ({exc}). "
            "Se usan DATOS SIMULADOS: los resultados no son reales.",
            UserWarning,
            stacklevel=2,
        )
        return ProveedorSimulado(semilla=semilla_respaldo).descargar(tickers, inicio, fin)


def rendimientos(
    precios: pd.DataFrame | pd.Series,
    tipo: str = "log",
) -> pd.DataFrame | pd.Series:
    """Convierte precios en rendimientos.

    Parametros
    ----------
    tipo:
        ``"log"`` para rendimientos logaritmicos (aditivos en el tiempo, que es
        lo comodo para agregar horizontes y para los modelos de volatilidad) o
        ``"simple"`` para variaciones porcentuales (aditivas entre activos, que
        es lo correcto para agregar una cartera).
    """
    if tipo not in {"log", "simple"}:
        raise ErrorDatos(f"Tipo de rendimiento '{tipo}' desconocido: use 'log' o 'simple'.")
    if (precios <= 0).any().any() if isinstance(precios, pd.DataFrame) else (precios <= 0).any():
        raise ErrorDatos("Hay precios nulos o negativos: no se pueden calcular rendimientos.")

    if tipo == "log":
        salida = np.log(precios).diff()
    else:
        salida = precios.pct_change()
    salida = salida.dropna(how="all")
    # Se deja constancia de la convencion para que las funciones que agregan
    # carteras no tengan que adivinarla.
    salida.attrs["tipo_rendimiento"] = tipo
    return salida


def rendimientos_cartera(
    rendimientos_activos: pd.DataFrame,
    pesos,
    reequilibrio: str = "periodico",
    tipo: str | None = None,
) -> pd.Series:
    """Agrega los rendimientos de los activos en el de la cartera.

    Parametros
    ----------
    reequilibrio:
        ``"periodico"`` mantiene los pesos constantes en cada periodo (supone
        reequilibrar al cierre de cada sesion); ``"comprar_y_mantener"`` deja
        que los pesos deriven con la evolucion de cada activo desde el inicio.
    tipo:
        Convencion de los rendimientos de entrada, ``"log"`` o ``"simple"``.
        Si se omite se lee la marca que deja :func:`rendimientos` y, a falta de
        ella, se supone ``"simple"``.

    Notas
    -----
    La suma ponderada de rendimientos solo es exacta en la convencion simple:
    el logaritmo de una suma no es la suma de logaritmos. Por eso, si la
    entrada es logaritmica se convierte a simple para agregar y el resultado se
    devuelve de nuevo en logaritmos, sin mezclar convenciones por el camino.
    """
    marco = como_matriz(rendimientos_activos, "rendimientos")
    marco.attrs.update(getattr(rendimientos_activos, "attrs", {}))
    pesos = normalizar_pesos(pesos, marco.columns)

    if reequilibrio not in {"periodico", "comprar_y_mantener"}:
        raise ErrorDatos(
            f"Modo de reequilibrio '{reequilibrio}' desconocido: "
            "use 'periodico' o 'comprar_y_mantener'."
        )
    if tipo is None:
        tipo = marco.attrs.get("tipo_rendimiento", "simple")
    if tipo not in {"log", "simple"}:
        raise ErrorDatos(f"Tipo de rendimiento '{tipo}' desconocido: use 'log' o 'simple'.")

    es_log = tipo == "log"
    simples = np.expm1(marco) if es_log else marco

    if reequilibrio == "periodico":
        serie = simples.mul(pesos, axis=1).sum(axis=1)
    else:
        valor_activos = (1.0 + simples).cumprod().mul(pesos, axis=1)
        valor_cartera = valor_activos.sum(axis=1)
        serie = valor_cartera.pct_change()
        serie.iloc[0] = valor_cartera.iloc[0] - 1.0

    if es_log:
        serie = np.log1p(serie)
    serie.name = "cartera"
    serie.attrs["tipo_rendimiento"] = tipo
    return serie


def matriz_covarianzas(
    rendimientos_activos: pd.DataFrame,
    metodo: str = "muestral",
    factor_decaimiento: float = 0.94,
    anualizar: bool = False,
    periodos_anio: int = 252,
) -> pd.DataFrame:
    """Matriz de covarianzas de los rendimientos.

    Parametros
    ----------
    metodo:
        ``"muestral"`` es la covarianza clasica; ``"ewma"`` pondera mas las
        observaciones recientes, util cuando la volatilidad cambia de regimen;
        ``"ledoit_wolf"`` encoge la muestral hacia una diagonal, lo que da
        matrices mejor condicionadas cuando hay muchos activos y pocos datos.
    """
    marco = como_matriz(rendimientos_activos, "rendimientos")
    if metodo == "muestral":
        cov = marco.cov()
    elif metodo == "ewma":
        cov = _covarianza_ewma(marco, factor_decaimiento)
    elif metodo == "ledoit_wolf":
        cov = _covarianza_ledoit_wolf(marco)
    else:
        raise ErrorDatos(
            f"Metodo de covarianza '{metodo}' desconocido: "
            "use 'muestral', 'ewma' o 'ledoit_wolf'."
        )
    if anualizar:
        cov = cov * periodos_anio
    return cov


def _covarianza_ewma(marco: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Covarianza con ponderacion exponencial, al estilo RiskMetrics."""
    if not 0.0 < factor < 1.0:
        raise ErrorDatos("El factor de decaimiento debe estar entre 0 y 1.")
    centrado = marco - marco.mean()
    n = len(centrado)
    pesos = factor ** np.arange(n - 1, -1, -1)
    pesos = pesos / pesos.sum()
    valores = centrado.to_numpy()
    cov = (valores * pesos[:, None]).T @ valores
    return pd.DataFrame(cov, index=marco.columns, columns=marco.columns)


def _covarianza_ledoit_wolf(marco: pd.DataFrame) -> pd.DataFrame:
    """Encogimiento de Ledoit-Wolf hacia una matriz diagonal de varianzas.

    Se estima la intensidad del encogimiento comparando la varianza de los
    errores de estimacion con la distancia al objetivo, siguiendo la version
    de Ledoit y Wolf (2004) con objetivo de correlacion nula.
    """
    valores = marco.to_numpy()
    n, k = valores.shape
    if n < 2:
        raise ErrorDatos("Hacen falta al menos 2 observaciones para estimar covarianzas.")
    centrado = valores - valores.mean(axis=0)
    muestral = centrado.T @ centrado / n
    objetivo = np.diag(np.diag(muestral))

    # Varianza de las entradas de la matriz muestral.
    pi = 0.0
    for t in range(n):
        producto = np.outer(centrado[t], centrado[t])
        pi += np.sum((producto - muestral) ** 2)
    pi /= n

    distancia = np.sum((muestral - objetivo) ** 2)
    intensidad = 0.0 if distancia <= 0 else max(0.0, min(1.0, (pi / n) / distancia))
    encogida = intensidad * objetivo + (1.0 - intensidad) * muestral
    salida = pd.DataFrame(encogida, index=marco.columns, columns=marco.columns)
    salida.attrs["intensidad_encogimiento"] = intensidad
    return salida
