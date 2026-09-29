"""Pruebas de los modelos de volatilidad condicional."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.datos import rendimientos, simular_precios
from icm.riesgo import (
    ajustar_garch,
    comparar_modelos_volatilidad,
    es_garch,
    prevision_volatilidad,
    var_garch,
    volatilidad_ewma,
    volatilidad_movil,
)


@pytest.fixture(scope="module")
def serie_con_agrupamiento() -> pd.Series:
    """Serie larga con volatilidad muy persistente, donde el GARCH tiene algo que decir."""
    precios = simular_precios(
        ["A"], n_periodos=3000, garch_alfa=0.10, garch_beta=0.88, semilla=21
    )
    return rendimientos(precios, "log")["A"]


@pytest.fixture(scope="module")
def ajuste(serie_con_agrupamiento):
    return ajustar_garch(serie_con_agrupamiento)


# --- media movil y EWMA --------------------------------------------------------


def test_la_volatilidad_movil_coincide_con_pandas(serie):
    obtenida = volatilidad_movil(serie, 21)
    esperada = serie.rolling(21).std(ddof=1).dropna()
    pd.testing.assert_series_equal(obtenida, esperada, check_names=False)


def test_la_volatilidad_movil_se_anualiza(serie):
    sin_anualizar = volatilidad_movil(serie, 21)
    anualizada = volatilidad_movil(serie, 21, anualizar=True)
    np.testing.assert_allclose(anualizada.to_numpy(), sin_anualizar.to_numpy() * np.sqrt(252))


def test_una_ventana_minuscula_da_error(serie):
    with pytest.raises(ErrorDatos, match="al menos 2"):
        volatilidad_movil(serie, 1)


def test_la_ewma_devuelve_una_prediccion_por_fecha(serie):
    # La primera fecha sale NaN a proposito: antes de la primera observacion
    # no hay informacion con la que predecir nada, y el valor que se daba ahi
    # venia de promediar las 30 primeras, es decir, del futuro.
    ewma = volatilidad_ewma(serie)
    assert len(ewma) == len(serie)
    assert np.isnan(ewma.iloc[0])
    assert (ewma.iloc[1:] > 0).all()


def test_el_arranque_de_la_ewma_no_mira_al_futuro():
    """Un choque posterior no puede mover una estimacion ya publicada.

    El arranque era la varianza de las 30 primeras observaciones, asi que el
    valor de t=0 se calculaba con datos de t=0 a t=29 y esa semilla sobrevivia
    con peso lambda**t. Medido con un choque en t=25: la volatilidad estimada
    en t=5 se multiplicaba por 90,7.
    """
    plana = pd.Series(np.full(200, 0.001))
    con_choque = plana.copy()
    con_choque.iloc[25] = 0.30

    sin, con = volatilidad_ewma(plana), volatilidad_ewma(con_choque)
    # Todo lo anterior al choque, y el propio dia del choque, intacto.
    pd.testing.assert_series_equal(sin.iloc[:26], con.iloc[:26])
    # Y el dia siguiente si reacciona, que es la informacion que toca.
    assert con.iloc[26] > 10 * sin.iloc[26]


def test_con_menos_de_treinta_observaciones_tampoco_hay_fuga():
    # Era el caso peor: con la ventana de arranque abarcando la serie entera,
    # la contaminacion era total. En t=5 la volatilidad se multiplicaba por
    # 82,7 por un choque en la ULTIMA fecha.
    corta = pd.Series(np.full(25, 0.001))
    con_choque = corta.copy()
    con_choque.iloc[24] = 0.30
    pd.testing.assert_series_equal(volatilidad_ewma(corta), volatilidad_ewma(con_choque))


def test_el_calentamiento_descarta_fechas_en_vez_de_inventarlas():
    # La semilla estable sigue disponible, pero pagando lo que cuesta: las
    # fechas cuya estimacion la usaria salen NaN, no con un numero.
    serie_larga = pd.Series(np.full(200, 0.001))
    ewma = volatilidad_ewma(serie_larga, calentamiento=30)
    assert int(ewma.isna().sum()) == 30
    assert ewma.notna().to_numpy().argmax() == 30


def test_un_calentamiento_que_se_come_la_serie_se_rechaza():
    with pytest.raises(ErrorDatos, match="serie entera"):
        volatilidad_ewma(pd.Series(np.full(10, 0.001)), calentamiento=10)


def test_la_ewma_no_usa_el_dato_del_propio_dia():
    # Se construye una serie plana con un unico dia extremo: la volatilidad
    # debe saltar al dia SIGUIENTE, no el mismo dia.
    valores = np.full(100, 0.001)
    valores[50] = 0.20
    serie_test = pd.Series(valores)
    ewma = volatilidad_ewma(serie_test)
    assert ewma.iloc[51] > 5 * ewma.iloc[50]


def test_la_ewma_reacciona_mas_rapido_con_un_factor_bajo():
    valores = np.full(200, 0.001)
    valores[100:] = 0.05
    serie_test = pd.Series(valores)
    rapida = volatilidad_ewma(serie_test, 0.80)
    lenta = volatilidad_ewma(serie_test, 0.99)
    assert rapida.iloc[120] > lenta.iloc[120]


def test_un_factor_de_decaimiento_invalido_da_error(serie):
    for factor in (0.0, 1.0, 1.5):
        with pytest.raises(ErrorDatos, match="entre 0 y 1"):
            volatilidad_ewma(serie, factor)


# --- GARCH ---------------------------------------------------------------------


def test_el_garch_recupera_la_persistencia_simulada(ajuste):
    # La serie se simulo con alfa 0.10 y beta 0.88: persistencia 0.98.
    assert ajuste.persistencia == pytest.approx(0.98, abs=0.04)


def test_la_volatilidad_condicional_varia_en_el_tiempo(ajuste):
    condicional = ajuste.volatilidad_condicional()
    assert condicional.max() / condicional.min() > 2.0


def test_la_volatilidad_condicional_media_se_parece_a_la_incondicional(
    ajuste, serie_con_agrupamiento
):
    condicional = ajuste.volatilidad_condicional().mean()
    assert condicional == pytest.approx(serie_con_agrupamiento.std(ddof=1), rel=0.15)


def test_la_prevision_converge_al_nivel_de_largo_plazo(ajuste, serie_con_agrupamiento):
    prevision = prevision_volatilidad(ajuste, 250)
    assert len(prevision) == 250
    # A largo plazo la prevision tiende a la volatilidad incondicional.
    assert prevision.iloc[-1] == pytest.approx(serie_con_agrupamiento.std(ddof=1), rel=0.25)


def test_la_prevision_a_un_dia_es_positiva(ajuste):
    assert prevision_volatilidad(ajuste, 1).iloc[0] > 0


def test_un_horizonte_invalido_da_error(ajuste):
    with pytest.raises(ErrorDatos, match="al menos un periodo"):
        prevision_volatilidad(ajuste, 0)


def test_el_var_garch_reacciona_a_la_turbulencia_reciente(serie_con_agrupamiento):
    # Se comparan dos ventanas que acaban en momentos de calma y de tension.
    condicional = ajustar_garch(serie_con_agrupamiento).volatilidad_condicional()
    fecha_calma = condicional.idxmin()
    fecha_tension = condicional.idxmax()

    tramo_calma = serie_con_agrupamiento.loc[:fecha_calma].tail(750)
    tramo_tension = serie_con_agrupamiento.loc[:fecha_tension].tail(750)
    if len(tramo_calma) < 500 or len(tramo_tension) < 500:
        pytest.skip("Los extremos de volatilidad caen demasiado al principio de la muestra.")

    assert var_garch(tramo_tension, 0.99) > var_garch(tramo_calma, 0.99)


def test_el_es_garch_supera_al_var_garch(serie_con_agrupamiento, ajuste):
    assert es_garch(serie_con_agrupamiento, 0.99, ajuste=ajuste) > var_garch(
        serie_con_agrupamiento, 0.99, ajuste=ajuste
    )


def test_el_var_garch_crece_con_el_horizonte(serie_con_agrupamiento, ajuste):
    a_un_dia = var_garch(serie_con_agrupamiento, 0.99, 1, ajuste=ajuste)
    a_diez_dias = var_garch(serie_con_agrupamiento, 0.99, 10, ajuste=ajuste)
    assert a_diez_dias > a_un_dia


def test_una_muestra_corta_no_se_estima():
    with pytest.raises(ErrorDatos, match="no es fiable"):
        ajustar_garch(pd.Series(np.random.default_rng(0).standard_normal(50) * 0.01))


@pytest.mark.parametrize("especificacion", ["garch", "gjr", "egarch"])
def test_todas_las_especificaciones_se_estiman(serie_con_agrupamiento, especificacion):
    resultado = ajustar_garch(serie_con_agrupamiento, especificacion)
    assert np.isfinite(resultado.resultado.loglikelihood)
    assert 0.0 <= resultado.persistencia < 1.05


def test_una_especificacion_desconocida_da_error(serie_con_agrupamiento):
    with pytest.raises(ErrorDatos, match="Especificacion"):
        ajustar_garch(serie_con_agrupamiento, "arch_infinito")


def test_una_distribucion_desconocida_da_error(serie_con_agrupamiento):
    with pytest.raises(ErrorDatos, match="Distribucion"):
        ajustar_garch(serie_con_agrupamiento, distribucion="uniforme")


def test_la_t_de_student_da_un_var_mas_prudente_que_la_normal(serie_con_agrupamiento):
    con_t = var_garch(serie_con_agrupamiento, 0.99, ajuste=ajustar_garch(serie_con_agrupamiento, distribucion="t"))
    con_normal = var_garch(
        serie_con_agrupamiento, 0.99, ajuste=ajustar_garch(serie_con_agrupamiento, distribucion="normal")
    )
    assert con_t > con_normal


def test_la_comparacion_ordena_por_bic(serie_con_agrupamiento):
    tabla = comparar_modelos_volatilidad(serie_con_agrupamiento)
    assert tabla["bic"].is_monotonic_increasing
    assert {"aic", "bic", "log_verosimilitud", "persistencia"} <= set(tabla.columns)


def test_la_escala_interna_no_afecta_al_resultado(serie_con_agrupamiento):
    # El modelo se estima sobre los rendimientos multiplicados por 100; la
    # volatilidad condicional debe volver a la escala original.
    ajuste_local = ajustar_garch(serie_con_agrupamiento)
    condicional = ajuste_local.volatilidad_condicional()
    assert 0.2 < condicional.mean() / serie_con_agrupamiento.std(ddof=1) < 5.0


def test_la_volatilidad_movil_sin_desplazar_mira_el_dia_que_estima():
    # No es un fallo: es la convencion de la volatilidad realizada, y hay que
    # poder comprobarla para saber que la otra rama hace falta. Metiendo un
    # rendimiento del 50 % en un solo dia, la ventana de 21 se multiplica por
    # 11,2 en la fecha del propio choque.
    generador = np.random.default_rng(3)
    indice = pd.bdate_range("2024-01-01", periods=300)
    limpia = pd.Series(generador.standard_normal(300) * 0.01, index=indice)
    con_choque = limpia.copy()
    con_choque.iloc[150] = 0.50

    fecha = indice[150]
    antes = volatilidad_movil(limpia, 21).loc[fecha]
    despues = volatilidad_movil(con_choque, 21).loc[fecha]
    assert despues > 10.0 * antes


def test_la_volatilidad_movil_como_prevision_no_ve_el_dia_que_estima():
    generador = np.random.default_rng(3)
    indice = pd.bdate_range("2024-01-01", periods=300)
    limpia = pd.Series(generador.standard_normal(300) * 0.01, index=indice)
    con_choque = limpia.copy()
    con_choque.iloc[150] = 0.50

    fecha = indice[150]
    antes = volatilidad_movil(limpia, 21, prevision=True).loc[fecha]
    despues = volatilidad_movil(con_choque, 21, prevision=True).loc[fecha]
    assert despues == pytest.approx(antes)
    # Y en la fecha siguiente si tiene que haberlo incorporado.
    assert volatilidad_movil(con_choque, 21, prevision=True).loc[indice[151]] > 10.0 * antes


def test_la_volatilidad_movil_como_prevision_solo_desplaza():
    serie = pd.Series(
        np.random.default_rng(4).standard_normal(200) * 0.01,
        index=pd.bdate_range("2024-01-01", periods=200),
    )
    directa = volatilidad_movil(serie, 21)
    desplazada = volatilidad_movil(serie, 21, prevision=True)
    pd.testing.assert_series_equal(
        desplazada, directa.shift(1).dropna().rename("volatilidad_movil_21")
    )


def test_el_es_del_garch_integra_la_cola_por_el_punto_medio(serie_con_agrupamiento, ajuste):
    # Con los extremos derechos de cada tramo la integral salia corta de forma
    # sistematica, siempre hacia el lado inseguro. La prueba es que el
    # resultado ya no dependa apenas del numero de tramos: con la regla del
    # punto medio el error cae con el cuadrado del paso.
    serie = serie_con_agrupamiento
    referencia = es_garch(serie, 0.99, ajuste=ajuste, n_tramos=200_000)
    obtenido = es_garch(serie, 0.99, ajuste=ajuste, n_tramos=1_000)

    # La misma integral con la regla antigua, los extremos derechos de cada
    # tramo, reproducida aqui para poder compararlas.
    prevision = ajuste.resultado.forecast(horizon=1, reindex=False)
    media = float(np.sum(prevision.mean.iloc[-1])) / ajuste.escala
    desviacion = float(np.sqrt(np.sum(prevision.variance.iloc[-1]))) / ajuste.escala
    derechos = np.linspace(0.01 / 1_000, 0.01, 1_000)
    cuantiles = ajuste.cuantiles_estandarizados(derechos)
    con_regla_antigua = -(media + desviacion * cuantiles.mean())

    error_nuevo = abs(obtenido - referencia)
    error_viejo = abs(con_regla_antigua - referencia)
    assert error_nuevo < error_viejo / 3.0
    # Y el sesgo iba siempre hacia el lado inseguro: el ES salia corto.
    assert con_regla_antigua < referencia
    assert es_garch(serie, 0.99, ajuste=ajuste) > var_garch(serie, 0.99, ajuste=ajuste)


def test_el_es_del_garch_necesita_al_menos_dos_tramos(serie_con_agrupamiento, ajuste):
    with pytest.raises(ErrorDatos, match="dos tramos"):
        es_garch(serie_con_agrupamiento, 0.99, ajuste=ajuste, n_tramos=1)
