"""Fixtures compartidas por toda la bateria de pruebas.

Ninguna prueba toca la red: todos los datos salen del simulador con semilla
fija, de modo que los resultados son identicos en cualquier maquina.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.datos import rendimientos, simular_precios

ACTIVOS = ["ALFA", "BETA", "GAMA"]


@pytest.fixture(scope="session")
def precios() -> pd.DataFrame:
    """Precios diarios de tres activos con colas gruesas y volatilidad agrupada."""
    return simular_precios(ACTIVOS, n_periodos=1500, semilla=7)


@pytest.fixture(scope="session")
def precios_sin_agrupamiento() -> pd.DataFrame:
    """Precios con volatilidad constante e innovaciones independientes.

    Util para contrastar los modelos contra un proceso generador conocido:
    aqui el VaR parametrico normal deberia funcionar bien.
    """
    return simular_precios(
        ACTIVOS,
        n_periodos=2000,
        volatilidad_anual=0.20,
        deriva_anual=0.0,
        grados_libertad=100.0,
        garch_alfa=0.0,
        garch_beta=0.0,
        semilla=11,
    )


@pytest.fixture(scope="session")
def rend(precios: pd.DataFrame) -> pd.DataFrame:
    """Rendimientos logaritmicos de los precios de la fixture principal."""
    return rendimientos(precios, "log")


@pytest.fixture(scope="session")
def rend_simples(precios: pd.DataFrame) -> pd.DataFrame:
    return rendimientos(precios, "simple")


@pytest.fixture(scope="session")
def serie(rend: pd.DataFrame) -> pd.Series:
    """Una unica serie de rendimientos."""
    return rend[ACTIVOS[0]]


@pytest.fixture(scope="session")
def normal_estandar() -> pd.Series:
    """Muestra normal grande de media 0 y desviacion 0,01.

    Permite contrastar las formulas parametricas contra sus valores teoricos.
    El tamanio es un compromiso: con 100.000 observaciones el error del cuantil
    empirico al 99 % ronda el 0,5 %, suficiente para las tolerancias que usan
    las pruebas, y el ajuste de la t de Student sigue siendo rapido.
    """
    generador = np.random.default_rng(123)
    valores = generador.standard_normal(100_000) * 0.01
    indice = pd.bdate_range("2000-01-03", periods=len(valores))
    return pd.Series(valores, index=indice, name="normal")


@pytest.fixture
def pesos_iguales() -> pd.Series:
    return pd.Series(1.0 / len(ACTIVOS), index=ACTIVOS)
