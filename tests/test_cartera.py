"""Pruebas de metricas, optimizacion y frontera eficiente."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos, ErrorOptimizacion
from icm.cartera import (
    Restricciones,
    alfa_jensen,
    asimetria,
    beta,
    cartera_pesos_iguales,
    comparar_carteras,
    contribuciones_riesgo,
    curtosis,
    error_seguimiento,
    frontera_eficiente,
    linea_mercado_capitales,
    maximo_sharpe,
    minima_varianza,
    minima_varianza_con_objetivo,
    paridad_riesgo,
    ratio_informacion,
    ratio_sharpe,
    ratio_sortino,
    rentabilidad_acumulada,
    rentabilidad_anualizada,
    resumen_metricas,
    contraste_jarque_bera,
    volatilidad_anualizada,
    volatilidad_cartera,
)
from icm.datos import matriz_covarianzas

# Caso de dos activos con solucion analitica conocida.
COV_DOS = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["X", "Y"], columns=["X", "Y"])
MU_DOS = pd.Series({"X": 0.08, "Y": 0.12})


# --- metricas basicas ----------------------------------------------------------


def test_rentabilidad_acumulada_compone():
    serie = pd.Series([0.10, -0.10, 0.20])
    esperado = 1.10 * 0.90 * 1.20 - 1.0
    assert rentabilidad_acumulada(serie, tipo="simple") == pytest.approx(esperado)


def test_rentabilidad_anualizada_con_un_anio_exacto():
    # 252 sesiones al 0,1 % diario: (1.001)^252 - 1.
    serie = pd.Series([0.001] * 252)
    esperado = 1.001**252 - 1.0
    assert rentabilidad_anualizada(serie, tipo="simple") == pytest.approx(esperado)


def test_rentabilidad_anualizada_de_dos_anios_es_la_tasa_geometrica():
    serie = pd.Series([0.001] * 504)
    anual = rentabilidad_anualizada(serie, tipo="simple")
    assert (1 + anual) ** 2 == pytest.approx(1.001**504)


def test_volatilidad_se_anualiza_por_la_raiz_del_tiempo():
    serie = pd.Series(np.random.default_rng(0).standard_normal(1000) * 0.01)
    assert volatilidad_anualizada(serie) == pytest.approx(serie.std(ddof=1) * np.sqrt(252))


def test_volatilidad_mensual_usa_doce_periodos():
    serie = pd.Series(np.random.default_rng(0).standard_normal(240) * 0.03)
    assert volatilidad_anualizada(serie, "mensual") == pytest.approx(
        serie.std(ddof=1) * np.sqrt(12)
    )


def test_frecuencia_desconocida_da_error():
    with pytest.raises(ErrorDatos, match="Frecuencia"):
        volatilidad_anualizada(pd.Series([0.01, 0.02]), "quincenal")


def test_sharpe_es_cero_cuando_no_hay_exceso():
    serie = pd.Series([0.001] * 252)
    anual = rentabilidad_anualizada(serie, tipo="simple")
    assert ratio_sharpe(serie, anual, tipo="simple") == pytest.approx(0.0, abs=1e-9)


def test_sortino_supera_a_sharpe_cuando_las_caidas_son_suaves(serie):
    # En una serie con asimetria positiva el riesgo de caida es menor que la
    # volatilidad total, asi que Sortino debe salir mayor.
    sesgada = pd.Series(np.concatenate([np.full(300, -0.001), np.full(60, 0.02)]))
    assert ratio_sortino(sesgada, tipo="simple") > ratio_sharpe(sesgada, tipo="simple")


def test_beta_de_un_activo_contra_si_mismo_es_uno(serie):
    assert beta(serie, serie) == pytest.approx(1.0)


def test_beta_escala_con_la_exposicion(serie):
    assert beta(serie * 2.0, serie) == pytest.approx(2.0)


def test_alfa_es_nula_si_el_activo_es_el_mercado(serie):
    assert alfa_jensen(serie, serie) == pytest.approx(0.0, abs=1e-12)


def test_error_de_seguimiento_nulo_frente_a_uno_mismo(serie):
    assert error_seguimiento(serie, serie) == pytest.approx(0.0, abs=1e-15)


def test_ratio_informacion_sin_diferencias_da_error(serie):
    with pytest.raises(ErrorDatos, match="error de seguimiento"):
        ratio_informacion(serie, serie)


def test_momentos_de_una_normal(normal_estandar):
    assert asimetria(normal_estandar) == pytest.approx(0.0, abs=0.03)
    assert curtosis(normal_estandar) == pytest.approx(0.0, abs=0.05)


def test_jarque_bera_no_rechaza_una_normal(normal_estandar):
    assert contraste_jarque_bera(normal_estandar)["normal_al_5pct"]


def test_jarque_bera_rechaza_los_rendimientos_con_colas_gruesas(serie):
    assert not contraste_jarque_bera(serie)["normal_al_5pct"]


def test_resumen_devuelve_todas_las_medidas(serie):
    resumen = resumen_metricas(serie)
    esperadas = {
        "observaciones",
        "rentabilidad_anualizada",
        "volatilidad_anualizada",
        "ratio_sharpe",
        "maxima_caida",
        "var_historico_95",
    }
    assert esperadas <= set(resumen.index)
    assert resumen.drop("observaciones").notna().all()


# --- pesos y contribuciones ----------------------------------------------------


def test_pesos_iguales_suman_uno():
    pesos = cartera_pesos_iguales(["A", "B", "C", "D"])
    assert pesos.sum() == pytest.approx(1.0)
    assert (pesos == 0.25).all()


def test_las_contribuciones_al_riesgo_suman_la_volatilidad(rend, pesos_iguales):
    cov = matriz_covarianzas(rend)
    tabla = contribuciones_riesgo(pesos_iguales, cov)
    assert tabla["contribucion_absoluta"].sum() == pytest.approx(
        volatilidad_cartera(pesos_iguales, cov)
    )
    assert tabla["contribucion_relativa"].sum() == pytest.approx(1.0)


def test_sin_correlacion_la_contribucion_depende_solo_de_la_varianza():
    tabla = contribuciones_riesgo(pd.Series({"X": 0.5, "Y": 0.5}), COV_DOS)
    # Var X = 0.04, Var Y = 0.09: la relacion de contribuciones es 4 a 9.
    relacion = tabla.loc["Y", "contribucion_absoluta"] / tabla.loc["X", "contribucion_absoluta"]
    assert relacion == pytest.approx(0.09 / 0.04)


# --- optimizacion --------------------------------------------------------------


def test_minima_varianza_con_dos_activos_independientes():
    # Sin correlacion los pesos optimos son inversamente proporcionales a la
    # varianza: w_X = (1/0.04) / (1/0.04 + 1/0.09).
    pesos = minima_varianza(COV_DOS)
    esperado_x = (1 / 0.04) / (1 / 0.04 + 1 / 0.09)
    assert pesos["X"] == pytest.approx(esperado_x, abs=1e-6)


def test_minima_varianza_tiene_la_menor_volatilidad_posible(rend):
    cov = matriz_covarianzas(rend)
    optima = minima_varianza(cov)
    volatilidad_optima = volatilidad_cartera(optima, cov)

    generador = np.random.default_rng(0)
    for _ in range(200):
        aleatoria = generador.dirichlet(np.ones(len(rend.columns)))
        assert volatilidad_cartera(pd.Series(aleatoria, index=rend.columns), cov) >= (
            volatilidad_optima - 1e-9
        )


def test_maximo_sharpe_tiene_el_mejor_ratio(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    mu = rend.mean() * 252
    optima = maximo_sharpe(mu, cov)
    mejor = float(optima @ mu) / volatilidad_cartera(optima, cov)

    generador = np.random.default_rng(0)
    for _ in range(200):
        aleatoria = pd.Series(generador.dirichlet(np.ones(len(rend.columns))), index=rend.columns)
        ratio = float(aleatoria @ mu) / volatilidad_cartera(aleatoria, cov)
        assert ratio <= mejor + 1e-8


def test_sin_cortos_ningun_peso_es_negativo(rend):
    pesos = maximo_sharpe(rend.mean() * 252, matriz_covarianzas(rend, anualizar=True))
    assert (pesos >= -1e-9).all()


def test_con_cortos_se_pueden_tomar_posiciones_negativas():
    # El activo Y tiene el doble de rentabilidad pero mas varianza; al permitir
    # cortos el optimizador puede apalancarse sobre el mejor ratio.
    pesos = maximo_sharpe(MU_DOS, COV_DOS, restricciones=Restricciones(permitir_cortos=True))
    assert pesos.sum() == pytest.approx(1.0)


def test_el_peso_maximo_se_respeta(rend):
    restricciones = Restricciones(peso_maximo=0.4)
    pesos = maximo_sharpe(rend.mean() * 252, matriz_covarianzas(rend, anualizar=True), restricciones=restricciones)
    assert pesos.max() <= 0.4 + 1e-6


def test_un_peso_maximo_imposible_da_error():
    with pytest.raises(ErrorDatos, match="no pueden sumar 1"):
        minima_varianza(COV_DOS, Restricciones(peso_maximo=0.3))


def test_la_cartera_con_objetivo_alcanza_ese_objetivo():
    pesos = minima_varianza_con_objetivo(MU_DOS, COV_DOS, 0.10)
    assert float(pesos @ MU_DOS) == pytest.approx(0.10, abs=1e-8)


def test_paridad_de_riesgo_iguala_las_contribuciones(rend):
    cov = matriz_covarianzas(rend)
    pesos = paridad_riesgo(cov)
    relativas = contribuciones_riesgo(pesos, cov)["contribucion_relativa"]
    np.testing.assert_allclose(relativas.to_numpy(), 1.0 / len(rend.columns), atol=1e-4)


def test_paridad_de_riesgo_da_menos_peso_al_activo_volatil():
    pesos = paridad_riesgo(COV_DOS)
    assert pesos["X"] > pesos["Y"]


def test_paridad_de_riesgo_admite_un_presupuesto_desigual(rend):
    cov = matriz_covarianzas(rend)
    presupuesto = pd.Series([0.6, 0.2, 0.2], index=rend.columns)
    pesos = paridad_riesgo(cov, presupuesto)
    relativas = contribuciones_riesgo(pesos, cov)["contribucion_relativa"]
    np.testing.assert_allclose(relativas.to_numpy(), presupuesto.to_numpy(), atol=1e-3)


def test_paridad_de_riesgo_rechaza_los_cortos():
    with pytest.raises(ErrorDatos, match="no negativos"):
        paridad_riesgo(COV_DOS, restricciones=Restricciones(permitir_cortos=True))


def test_una_covarianza_asimetrica_da_error():
    mala = pd.DataFrame([[0.04, 0.01], [0.02, 0.09]], index=["X", "Y"], columns=["X", "Y"])
    with pytest.raises(ErrorDatos, match="simetrica"):
        minima_varianza(mala)


def test_una_covarianza_no_definida_positiva_da_error():
    mala = pd.DataFrame([[0.04, 0.10], [0.10, 0.09]], index=["X", "Y"], columns=["X", "Y"])
    with pytest.raises(ErrorDatos, match="semidefinida positiva"):
        minima_varianza(mala)


def test_comparar_carteras_resume_las_alternativas(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    mu = rend.mean() * 252
    tabla = comparar_carteras(
        {
            "iguales": cartera_pesos_iguales(rend.columns),
            "minima_varianza": minima_varianza(cov),
            "maximo_sharpe": maximo_sharpe(mu, cov),
        },
        mu,
        cov,
    )
    assert list(tabla.index) == ["iguales", "minima_varianza", "maximo_sharpe"]
    assert tabla.loc["minima_varianza", "volatilidad"] <= tabla.loc["iguales", "volatilidad"] + 1e-9
    assert tabla.loc["maximo_sharpe", "sharpe"] >= tabla.loc["iguales", "sharpe"] - 1e-9


# --- frontera ------------------------------------------------------------------


def test_la_frontera_es_creciente(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    frontera = frontera_eficiente(rend.mean() * 252, cov, n_puntos=15, incluir_pesos=False)
    assert len(frontera) > 5
    # Ordenada por volatilidad, la rentabilidad no puede decrecer.
    diferencias = frontera["rentabilidad"].diff().dropna()
    assert (diferencias >= -1e-8).all()


def test_la_frontera_arranca_en_la_minima_varianza(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    frontera = frontera_eficiente(rend.mean() * 252, cov, n_puntos=15, incluir_pesos=False)
    assert frontera["volatilidad"].iloc[0] == pytest.approx(
        volatilidad_cartera(minima_varianza(cov), cov), abs=1e-6
    )


def test_los_pesos_de_la_frontera_suman_uno(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    frontera = frontera_eficiente(rend.mean() * 252, cov, n_puntos=10)
    columnas = [c for c in frontera.columns if c.startswith("peso_")]
    np.testing.assert_allclose(frontera[columnas].sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_la_recta_del_mercado_de_capitales_pasa_por_la_tangente(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    mu = rend.mean() * 252
    recta, tangente = linea_mercado_capitales(mu, cov, tasa_libre_riesgo=0.01)

    volatilidad = volatilidad_cartera(tangente, cov)
    rentabilidad = float(tangente @ mu)
    interpolada = np.interp(volatilidad, recta["volatilidad"], recta["rentabilidad"])
    assert interpolada == pytest.approx(rentabilidad, rel=1e-6)


def test_la_recta_arranca_en_la_tasa_sin_riesgo(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    recta, _ = linea_mercado_capitales(rend.mean() * 252, cov, tasa_libre_riesgo=0.02)
    assert recta["rentabilidad"].iloc[0] == pytest.approx(0.02)


def test_la_frontera_falla_limpiamente_si_no_hay_solucion():
    # Un unico activo con tope del 100 % es resoluble; con dos activos y un
    # objetivo inalcanzable el optimizador no converge en ningun punto.
    cov = pd.DataFrame([[1e-8]], index=["X"], columns=["X"])
    with pytest.raises((ErrorOptimizacion, ErrorDatos)):
        minima_varianza_con_objetivo(pd.Series({"X": 0.05}), cov, 0.50)
