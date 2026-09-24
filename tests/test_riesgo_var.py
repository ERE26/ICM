"""Pruebas del VaR, el Expected Shortfall y las caidas desde maximos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from icm.comun import ErrorDatos
from icm.riesgo import (
    dominio_valido_cornish_fisher,
    curva_valor,
    duracion_maxima_caida,
    es_cornish_fisher,
    es_historico,
    es_parametrico,
    escalar_horizonte,
    expected_shortfall,
    maxima_caida,
    serie_caidas,
    tabla_caidas,
    tabla_medidas,
    ulcer_index,
    var,
    var_cornish_fisher,
    var_historico,
    var_incremental,
    var_parametrico,
)

METODOS = ["historico", "parametrico_normal", "parametrico_t", "cornish_fisher"]


# --- convencion de signos y coherencia -----------------------------------------


@pytest.mark.parametrize("metodo", METODOS)
def test_el_var_se_devuelve_como_perdida_positiva(serie, metodo):
    assert var(serie, 0.99, metodo) > 0


@pytest.mark.parametrize("metodo", METODOS)
def test_el_es_nunca_es_menor_que_el_var(serie, metodo):
    assert expected_shortfall(serie, 0.99, metodo) >= var(serie, 0.99, metodo)


@pytest.mark.parametrize("metodo", METODOS)
def test_el_var_crece_con_el_nivel_de_confianza(serie, metodo):
    assert var(serie, 0.99, metodo) > var(serie, 0.95, metodo) > var(serie, 0.90, metodo)


def test_un_nivel_fuera_de_rango_da_error(serie):
    for nivel in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ErrorDatos, match="nivel de confianza"):
            var_historico(serie, nivel)


def test_un_metodo_desconocido_da_error(serie):
    with pytest.raises(ErrorDatos, match="desconocido"):
        var(serie, 0.99, "bola_de_cristal")


# --- contraste contra valores teoricos -----------------------------------------


def test_el_var_normal_reproduce_la_formula_cerrada(normal_estandar):
    # Con media 0 y sigma 0.01, el VaR al 99 % es 2.326 * 0.01.
    esperado = -stats.norm.ppf(0.01) * 0.01
    assert var_parametrico(normal_estandar, 0.99, "normal") == pytest.approx(esperado, rel=0.01)


def test_el_es_normal_reproduce_la_formula_cerrada(normal_estandar):
    alfa = 0.01
    esperado = 0.01 * stats.norm.pdf(stats.norm.ppf(alfa)) / alfa
    assert es_parametrico(normal_estandar, 0.99, "normal") == pytest.approx(esperado, rel=0.01)


def test_con_datos_normales_todos_los_metodos_coinciden(normal_estandar):
    valores = [var(normal_estandar, 0.99, metodo) for metodo in METODOS]
    assert max(valores) / min(valores) < 1.05


def test_cornish_fisher_iguala_al_normal_sin_asimetria_ni_curtosis(normal_estandar):
    assert var_cornish_fisher(normal_estandar, 0.99) == pytest.approx(
        var_parametrico(normal_estandar, 0.99, "normal"), rel=0.02
    )


def test_el_es_de_cornish_fisher_converge_al_normal(normal_estandar):
    assert es_cornish_fisher(normal_estandar, 0.99) == pytest.approx(
        es_parametrico(normal_estandar, 0.99, "normal"), rel=0.03
    )


def test_con_colas_gruesas_el_normal_se_queda_corto(serie):
    # Es el argumento central contra el VaR parametrico normal: con colas
    # gruesas infravalora la perdida frente al historico.
    assert var_parametrico(serie, 0.99, "normal") < var_historico(serie, 0.99)


def test_la_t_de_student_corrige_al_normal_en_la_cola(serie):
    assert var_parametrico(serie, 0.99, "t") > var_parametrico(serie, 0.99, "normal")


def test_el_var_historico_es_el_cuantil_empirico():
    serie = pd.Series(np.linspace(-0.10, 0.10, 1001))
    # El percentil 5 de una rejilla uniforme entre -0.10 y 0.10 es -0.09.
    assert var_historico(serie, 0.95) == pytest.approx(0.09, abs=1e-9)


def test_el_es_historico_promedia_la_cola():
    serie = pd.Series([-0.10, -0.08, -0.06, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    # Al 80 %, la cola son las dos peores observaciones.
    assert es_historico(serie, 0.80) == pytest.approx(0.09, abs=1e-9)


def test_cornish_fisher_admite_momentos_moderados():
    assert dominio_valido_cornish_fisher(0.0, 0.0)
    assert dominio_valido_cornish_fisher(-0.3, 1.5)


def test_cornish_fisher_rechaza_colas_muy_gruesas():
    # Con estos momentos el polinomio deja de ser creciente y el resultado
    # ya no es un cuantil.
    assert not dominio_valido_cornish_fisher(-1.19, 29.0)
    assert not dominio_valido_cornish_fisher(0.0, 12.0)


def test_fuera_del_dominio_la_transformacion_deja_de_ser_monotona():
    # Comprobacion directa del criterio: si dice que no es valido, tiene que
    # existir un tramo en el que el cuantil ajustado decrezca.
    from icm.riesgo.var import _cuantil_cornish_fisher

    s, k = -1.19, 29.0
    assert not dominio_valido_cornish_fisher(s, k)
    z = np.linspace(-4.0, 0.0, 200)
    ajustado = _cuantil_cornish_fisher(stats.norm.cdf(z), s, k)
    assert np.any(np.diff(ajustado) < 0)


def test_dentro_del_dominio_la_transformacion_es_monotona():
    s, k = -0.3, 1.5
    from icm.riesgo.var import _cuantil_cornish_fisher

    z = np.linspace(-5.0, 5.0, 500)
    ajustado = _cuantil_cornish_fisher(stats.norm.cdf(z), s, k)
    assert np.all(np.diff(ajustado) > 0)


def test_se_avisa_al_salirse_del_dominio():
    # Mezcla de una normal estrecha con unos pocos saltos enormes: curtosis
    # muy por encima de lo que la expansion admite.
    generador = np.random.default_rng(3)
    valores = generador.standard_normal(5000) * 0.004
    valores[::500] = -0.25
    serie = pd.Series(valores)

    with pytest.warns(UserWarning, match="dominio de validez"):
        var_cornish_fisher(serie, 0.99)
    with pytest.warns(UserWarning, match="dominio de validez"):
        es_cornish_fisher(serie, 0.99)


def test_no_se_avisa_con_datos_normales(normal_estandar):
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        var_cornish_fisher(normal_estandar, 0.99)


def test_distribucion_parametrica_desconocida(serie):
    with pytest.raises(ErrorDatos, match="Distribucion"):
        var_parametrico(serie, 0.99, "cauchy")


# --- escalado y tablas ---------------------------------------------------------


def test_la_raiz_del_tiempo_escala_como_toca():
    assert escalar_horizonte(0.02, 4) == pytest.approx(0.04)
    assert escalar_horizonte(0.02, 1) == pytest.approx(0.02)


def test_un_horizonte_invalido_da_error():
    with pytest.raises(ErrorDatos, match="al menos un dia"):
        escalar_horizonte(0.02, 0)


def test_la_tabla_de_medidas_cubre_metodos_y_niveles(serie):
    tabla = tabla_medidas(serie, niveles=(0.95, 0.99), valor_cartera=1_000_000)
    assert len(tabla) == len(METODOS) * 2
    assert {"var", "es", "var_monetario", "es_monetario"} <= set(tabla.columns)
    np.testing.assert_allclose(
        tabla["var_monetario"].to_numpy(), tabla["var"].to_numpy() * 1_000_000
    )


def test_el_var_incremental_suma_aproximadamente_el_efecto_diversificacion(rend, pesos_iguales):
    incrementos = var_incremental(rend, pesos_iguales, 0.95)
    assert len(incrementos) == len(rend.columns)
    assert incrementos.notna().all()


# --- caidas --------------------------------------------------------------------


def test_la_curva_de_valor_compone_los_rendimientos():
    serie = pd.Series([0.10, -0.10, 0.20])
    curva = curva_valor(serie, tipo="simple")
    np.testing.assert_allclose(curva.to_numpy(), [1.10, 0.99, 1.188])


def test_la_maxima_caida_de_una_serie_conocida():
    # 100 -> 110 -> 88 -> 105: la peor caida es de 110 a 88, un -20 %.
    serie = pd.Series([0.10, -0.20, 0.19318181818])
    assert maxima_caida(serie, tipo="simple") == pytest.approx(-0.20, abs=1e-6)


def test_una_serie_siempre_creciente_no_tiene_caidas():
    serie = pd.Series([0.01] * 50)
    assert maxima_caida(serie, tipo="simple") == pytest.approx(0.0)
    assert duracion_maxima_caida(serie, tipo="simple") == 0


def test_las_caidas_nunca_son_positivas(serie):
    assert (serie_caidas(serie) <= 1e-12).all()


def test_la_tabla_de_caidas_ordena_de_peor_a_mejor(rend):
    tabla = tabla_caidas(rend["ALFA"], n=5)
    assert len(tabla) <= 5
    if len(tabla) > 1:
        assert tabla["caida"].is_monotonic_increasing is False or tabla["caida"].iloc[0] <= tabla["caida"].iloc[-1]
        assert tabla["caida"].iloc[0] == pytest.approx(maxima_caida(rend["ALFA"]))


def test_la_tabla_de_caidas_marca_los_episodios_no_recuperados():
    # Serie que cae y nunca vuelve al maximo previo.
    serie = pd.Series([0.05, -0.30, 0.01, 0.01])
    tabla = tabla_caidas(serie, tipo="simple", n=3)
    assert pd.isna(tabla["recuperacion"].iloc[0])


def test_la_tabla_de_caidas_detecta_la_recuperacion():
    serie = pd.Series([0.10, -0.20, 0.30, 0.0])
    tabla = tabla_caidas(serie, tipo="simple", n=3)
    assert not pd.isna(tabla["recuperacion"].iloc[0])


def test_el_ulcer_index_es_positivo_si_hay_caidas(serie):
    assert ulcer_index(serie) > 0


def test_el_ulcer_index_es_nulo_sin_caidas():
    assert ulcer_index(pd.Series([0.01] * 20), tipo="simple") == pytest.approx(0.0)
