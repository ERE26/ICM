"""Pruebas de humo del modulo de graficos.

No se comprueba el aspecto de las figuras, solo que se construyan sin errores y
con los elementos que deben llevar: leyenda cuando hay varias series, etiquetas
de ejes y el numero de marcas esperado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from icm import graficos  # noqa: E402  (debe ir tras fijar el backend)
from icm.cartera import cartera_pesos_iguales, contribuciones_riesgo, frontera_eficiente
from icm.datos import matriz_covarianzas, rendimientos_cartera
from icm.riesgo import serie_caidas, serie_var_movil, simular_cartera, var_montecarlo


@pytest.fixture(autouse=True)
def _estilo_y_cierre():
    import matplotlib.pyplot as plt

    graficos.aplicar_estilo()
    yield
    plt.close("all")


def test_la_paleta_no_repite_colores():
    assert len(set(graficos.PALETA)) == len(graficos.PALETA)


def test_los_colores_de_estado_no_estan_en_la_paleta():
    # Un color de estado nunca debe poder confundirse con una serie mas.
    assert not set(graficos.ESTADO.values()) & set(graficos.PALETA)


def test_la_frontera_se_dibuja_con_sus_elementos(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    mu = rend.mean() * 252
    frontera = frontera_eficiente(mu, cov, n_puntos=10, incluir_pesos=False)

    figura, ax = graficos.frontera_eficiente(
        frontera,
        carteras={"Mínima varianza": (0.15, 0.05)},
        activos={activo: (0.2, 0.06) for activo in rend.columns},
    )
    assert ax.get_xlabel() and ax.get_ylabel()
    assert ax.get_legend() is not None
    assert figura is ax.figure


def test_las_contribuciones_dibujan_dos_barras_por_activo(rend, pesos_iguales):
    tabla = contribuciones_riesgo(pesos_iguales, matriz_covarianzas(rend))
    figura, ax = graficos.contribuciones_riesgo(tabla)
    assert len(ax.patches) == 2 * len(tabla)
    assert ax.get_legend() is not None


def test_una_sola_serie_no_lleva_leyenda(serie):
    figura, ax = graficos.series_temporales({"Única": serie})
    assert ax.get_legend() is None


def test_varias_series_llevan_leyenda(rend):
    figura, ax = graficos.series_temporales({c: rend[c] for c in rend.columns})
    assert ax.get_legend() is not None
    assert len(ax.lines) == len(rend.columns)


def test_las_series_conservan_el_color_por_posicion(rend):
    _, ax = graficos.series_temporales({c: rend[c] for c in rend.columns})
    colores = [linea.get_color() for linea in ax.lines]
    assert colores == graficos.PALETA[: len(rend.columns)]


def test_el_grafico_de_caidas_anota_la_peor(serie):
    caidas = serie_caidas(serie)
    figura, ax = graficos.caidas(caidas)
    textos = [t.get_text() for t in ax.texts]
    assert any("Maxima caida" in texto for texto in textos)


def test_la_distribucion_simulada_marca_var_y_es(rend, pesos_iguales):
    simulaciones = simular_cartera(
        rend.mean(), matriz_covarianzas(rend), pesos_iguales, 5000, semilla=1
    )
    figura, ax = graficos.distribucion_simulada(
        simulaciones, var=var_montecarlo(simulaciones, 0.99), es=0.05
    )
    # Dos lineas verticales: el VaR y el ES.
    assert len(ax.lines) == 2
    assert any("VaR" in t.get_text() for t in ax.texts)


def test_el_backtest_marca_las_excepciones(rend, pesos_iguales):
    cartera = rendimientos_cartera(rend, pesos_iguales)
    var_movil = serie_var_movil(cartera, 0.95, 250, "historico")
    figura, ax = graficos.backtest(cartera, var_movil)

    etiquetas = [texto.get_text() for texto in ax.get_legend().get_texts()]
    assert any("Excepciones" in etiqueta for etiqueta in etiquetas)
    # Las excepciones se dibujan como una nube de puntos aparte de las lineas.
    assert len(ax.collections) == 1


def test_el_backtest_acepta_excepciones_calculadas_aparte(rend, pesos_iguales):
    from icm.riesgo import excepciones

    cartera = rendimientos_cartera(rend, pesos_iguales)
    var_movil = serie_var_movil(cartera, 0.95, 250, "historico")
    fallos = excepciones(cartera, var_movil)
    figura, ax = graficos.backtest(cartera, var_movil, fallos)
    assert ax.collections[0].get_offsets().shape[0] == int(fallos.sum())


def test_los_ejes_de_porcentaje_se_formatean(serie):
    figura, ax = graficos.series_temporales({"Serie": serie}, porcentaje=True)
    etiqueta = ax.yaxis.get_major_formatter()(0.05)
    assert etiqueta.endswith("%")


def test_se_puede_dibujar_sobre_unos_ejes_existentes(rend):
    import matplotlib.pyplot as plt

    figura_previa, ejes = plt.subplots(1, 2, figsize=(10, 4))
    graficos.series_temporales({"A": rend.iloc[:, 0]}, ax=ejes[0])
    graficos.series_temporales({"B": rend.iloc[:, 1]}, ax=ejes[1])
    assert len(ejes[0].lines) == 1
    assert len(ejes[1].lines) == 1
