"""Constantes y utilidades compartidas por todos los modulos."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Numero de sesiones bursatiles en un anio natural.
DIAS_HABILES_ANIO = 252

#: Periodos por anio segun la frecuencia de la serie.
PERIODOS_POR_ANIO = {
    "diaria": DIAS_HABILES_ANIO,
    "semanal": 52,
    "mensual": 12,
    "trimestral": 4,
    "anual": 1,
}


class ErrorDatos(Exception):
    """Los datos de entrada no tienen la forma o el contenido esperados."""


class ErrorDescarga(Exception):
    """No se han podido obtener los precios del proveedor externo."""


class ErrorOptimizacion(Exception):
    """El optimizador no ha encontrado una solucion valida."""


def periodos_por_anio(frecuencia: str | int) -> int:
    """Traduce una frecuencia a numero de periodos por anio.

    Acepta tanto un nombre ("diaria", "mensual", ...) como un entero, que se
    devuelve tal cual para permitir frecuencias no contempladas.
    """
    if isinstance(frecuencia, (int, np.integer)):
        if frecuencia <= 0:
            raise ErrorDatos("El numero de periodos por anio debe ser positivo.")
        return int(frecuencia)
    clave = frecuencia.lower()
    if clave not in PERIODOS_POR_ANIO:
        opciones = ", ".join(sorted(PERIODOS_POR_ANIO))
        raise ErrorDatos(f"Frecuencia '{frecuencia}' desconocida. Opciones: {opciones}.")
    return PERIODOS_POR_ANIO[clave]


def como_serie(datos, nombre: str = "serie") -> pd.Series:
    """Convierte la entrada en una Series de float, descartando los ausentes."""
    if isinstance(datos, pd.Series):
        serie = datos.astype(float)
    elif isinstance(datos, pd.DataFrame):
        if datos.shape[1] != 1:
            raise ErrorDatos(
                f"Se esperaba una unica columna en '{nombre}', se han recibido {datos.shape[1]}."
            )
        serie = datos.iloc[:, 0].astype(float)
    else:
        serie = pd.Series(np.asarray(datos, dtype=float), name=nombre)
    serie = serie.dropna()
    if serie.empty:
        raise ErrorDatos(f"La serie '{nombre}' no contiene observaciones validas.")
    return serie


def como_matriz(datos, nombre: str = "matriz") -> pd.DataFrame:
    """Convierte la entrada en un DataFrame de float sin filas incompletas."""
    if isinstance(datos, pd.DataFrame):
        marco = datos.astype(float)
    elif isinstance(datos, pd.Series):
        marco = datos.to_frame().astype(float)
    else:
        marco = pd.DataFrame(np.asarray(datos, dtype=float))
    marco = marco.dropna(how="any")
    if marco.empty:
        raise ErrorDatos(f"'{nombre}' no contiene filas completas.")
    return marco


def validar_nivel(nivel: float) -> float:
    """Comprueba que un nivel de confianza este en el intervalo (0, 1)."""
    nivel = float(nivel)
    if not 0.0 < nivel < 1.0:
        raise ErrorDatos(f"El nivel de confianza debe estar entre 0 y 1, se ha recibido {nivel}.")
    return nivel


def normalizar_pesos(pesos, activos=None) -> pd.Series:
    """Devuelve los pesos como Series indexada por activo y con suma unitaria."""
    if isinstance(pesos, pd.Series):
        serie = pesos.astype(float)
    elif isinstance(pesos, dict):
        serie = pd.Series(pesos, dtype=float)
    else:
        valores = np.asarray(pesos, dtype=float)
        indice = list(activos) if activos is not None else range(len(valores))
        serie = pd.Series(valores, index=indice)
    if activos is not None:
        faltan = [a for a in activos if a not in serie.index]
        if faltan:
            raise ErrorDatos(f"Faltan pesos para los activos: {faltan}.")
        serie = serie.reindex(list(activos))
    suma = serie.sum()
    if np.isclose(suma, 0.0):
        raise ErrorDatos("Los pesos suman cero y no se pueden normalizar.")
    return serie / suma
