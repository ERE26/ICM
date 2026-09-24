"""Pruebas de la evaluacion de predicciones y del contraste de precision.

El hilo conductor es que una metrica sin referencia no dice nada, asi que la
mayoria de las pruebas no comprueban un numero suelto sino una **propiedad**:
que el MASE de la referencia contra si misma sea exactamente 1, que invertir el
signo de una prediccion de exactamente el acierto direccional complementario,
que el contraste de Diebold-Mariano no distinga lo que es identico y que
rechace alrededor del 5 % de las veces cuando los dos modelos son igual de
buenos. Un contraste que siempre encuentra diferencias no es un contraste.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.ml.evaluacion import (
    NOMBRE_INGENUA,
    acierto_direccional,
    contraste_diebold_mariano,
    correlacion_informacion,
    error_absoluto_medio,
    error_escalado_medio,
    prediccion_ingenua,
    prediccion_media_movil,
    raiz_error_cuadratico_medio,
    tabla_evaluacion,
)

N = 500


@pytest.fixture(scope="module")
def rendimientos_ruido() -> pd.Series:
    """Rendimientos diarios sin ninguna estructura predecible."""
    generador = np.random.default_rng(2024)
    indice = pd.bdate_range("2019-01-01", periods=N, name="fecha")
    return pd.Series(generador.normal(0.0002, 0.012, N), index=indice, name="rend")


@pytest.fixture(scope="module")
def precios() -> pd.Series:
    """Precios que siguen un paseo aleatorio con deriva."""
    generador = np.random.default_rng(99)
    indice = pd.bdate_range("2019-01-01", periods=N, name="fecha")
    saltos = generador.normal(0.0004, 0.015, N)
    return pd.Series(60.0 * np.exp(np.cumsum(saltos)), index=indice, name="precio")


@pytest.fixture(scope="module")
def caso_con_senal() -> tuple[pd.Series, pd.Series]:
    """Rendimientos con una senal real y el modelo que la conoce.

    La senal depende de una variable observable el dia anterior, asi que el
    modelo no hace trampa: es el caso en el que un modelo debe ganar, y sirve
    para comprobar que las metricas lo detectan.
    """
    generador = np.random.default_rng(31415)
    indice = pd.bdate_range("2019-01-01", periods=N, name="fecha")
    senal = 0.004 * generador.normal(size=N)
    y = pd.Series(senal + generador.normal(0.0, 0.004, N), index=indice, name="rend")
    return y, pd.Series(senal, index=indice, name="modelo")


# --- predicciones de referencia ------------------------------------------------


def test_sobre_precios_la_ingenua_repite_el_ultimo_precio_observado(precios):
    prediccion = prediccion_ingenua(precios)

    assert np.isnan(prediccion.iloc[0])
    np.testing.assert_allclose(prediccion.iloc[1:], precios.iloc[:-1])


def test_sobre_rendimientos_la_ingenua_predice_cero(rendimientos_ruido):
    prediccion = prediccion_ingenua(rendimientos_ruido)

    assert np.isnan(prediccion.iloc[0])
    assert (prediccion.iloc[1:] == 0.0).all()


def test_la_marca_explicita_manda_sobre_la_regla_de_escala(rendimientos_ruido, precios):
    rendimientos_como_nivel = rendimientos_ruido.copy()
    rendimientos_como_nivel.attrs["tipo_serie"] = "precio"
    precios_como_rendimiento = precios.copy()
    precios_como_rendimiento.attrs["tipo_serie"] = "rendimiento"

    np.testing.assert_allclose(
        prediccion_ingenua(rendimientos_como_nivel).iloc[1:],
        rendimientos_ruido.iloc[:-1],
    )
    assert (prediccion_ingenua(precios_como_rendimiento).iloc[1:] == 0.0).all()


def test_la_marca_que_deja_el_modulo_de_precios_indica_rendimientos(precios):
    # 'tipo_rendimiento' es la marca que pone icm.datos.precios.rendimientos.
    disfrazada = precios.copy()
    disfrazada.attrs["tipo_rendimiento"] = "log"

    assert (prediccion_ingenua(disfrazada).iloc[1:] == 0.0).all()


def test_una_marca_de_tipo_desconocida_se_rechaza(rendimientos_ruido):
    serie = rendimientos_ruido.copy()
    serie.attrs["tipo_serie"] = "retorno"

    with pytest.raises(ErrorDatos, match="tipo_serie"):
        prediccion_ingenua(serie)


def test_un_spread_centrado_en_cero_se_trata_como_nivel_y_no_como_rendimiento():
    """Un spread vale cero de media pero se mueve en euros, no en tantos por uno.

    Es el caso limite de la regla de escala: sin el segundo criterio, la
    referencia de un spark spread pasaria a ser "cero" en lugar de "lo mismo
    que ayer", que es lo correcto para un nivel.
    """
    generador = np.random.default_rng(5)
    indice = pd.bdate_range("2020-01-01", periods=200, name="fecha")
    spread = pd.Series(generador.normal(0.0, 8.0, 200), index=indice, name="spark")

    np.testing.assert_allclose(prediccion_ingenua(spread).iloc[1:], spread.iloc[:-1])


def test_la_ingenua_a_varios_dias_desplaza_el_horizonte_completo(precios):
    prediccion = prediccion_ingenua(precios, horizonte=5)

    assert prediccion.iloc[:5].isna().all()
    np.testing.assert_allclose(prediccion.iloc[5:], precios.iloc[:-5])


@pytest.mark.parametrize("horizonte", [0, -1, 2.5, "1"])
def test_un_horizonte_invalido_se_rechaza(precios, horizonte):
    with pytest.raises(ErrorDatos, match="horizonte"):
        prediccion_ingenua(precios, horizonte=horizonte)


def test_un_horizonte_mas_largo_que_la_serie_se_rechaza(precios):
    with pytest.raises(ErrorDatos, match="longitud de la serie"):
        prediccion_ingenua(precios.iloc[:4], horizonte=4)


def test_la_media_movil_solo_usa_observaciones_anteriores():
    serie = pd.Series(
        np.arange(1.0, 11.0), index=pd.bdate_range("2022-01-03", periods=10, name="fecha")
    )
    prediccion = prediccion_media_movil(serie, ventana=3)

    assert prediccion.iloc[:3].isna().all()
    # La prediccion del cuarto dia es la media de los tres primeros, no de los
    # tres que terminan en el: si mirase al presente valdria 3.
    assert prediccion.iloc[3] == pytest.approx(2.0)
    assert prediccion.iloc[9] == pytest.approx(8.0)


def test_la_media_movil_de_una_serie_constante_es_esa_constante():
    serie = pd.Series(np.full(50, 7.0), index=pd.bdate_range("2022-01-03", periods=50))
    prediccion = prediccion_media_movil(serie, ventana=10)

    assert prediccion.dropna().eq(7.0).all()


@pytest.mark.parametrize("ventana", [0, -3, 1.5])
def test_una_ventana_invalida_se_rechaza(rendimientos_ruido, ventana):
    with pytest.raises(ErrorDatos, match="ventana"):
        prediccion_media_movil(rendimientos_ruido, ventana=ventana)


def test_una_ventana_mas_larga_que_la_serie_se_rechaza(rendimientos_ruido):
    with pytest.raises(ErrorDatos, match="longitud de la serie"):
        prediccion_media_movil(rendimientos_ruido.iloc[:20], ventana=20)


# --- metricas ------------------------------------------------------------------


def test_la_prediccion_perfecta_no_comete_ningun_error(rendimientos_ruido):
    y = rendimientos_ruido

    assert error_absoluto_medio(y, y) == 0.0
    assert raiz_error_cuadratico_medio(y, y) == 0.0
    assert error_escalado_medio(y, y) == 0.0
    assert acierto_direccional(y, y) == 1.0
    assert correlacion_informacion(y, y) == pytest.approx(1.0)


def test_el_mase_de_la_referencia_contra_si_misma_es_exactamente_uno(rendimientos_ruido, precios):
    ingenua_rendimientos = prediccion_ingenua(rendimientos_ruido)
    ingenua_precios = prediccion_ingenua(precios)

    # Sin referencia explicita (se calcula por dentro) y con ella: los dos
    # caminos deben dar el mismo 1 exacto, no un 0,999999.
    assert error_escalado_medio(precios, ingenua_precios) == 1.0
    assert error_escalado_medio(precios, ingenua_precios, ingenua_precios) == 1.0
    assert error_escalado_medio(rendimientos_ruido, ingenua_rendimientos) == 1.0


def test_el_mase_no_depende_de_las_unidades(caso_con_senal):
    y, modelo = caso_con_senal
    referencia = prediccion_ingenua(y)
    factor = 1000.0

    original = error_escalado_medio(y, modelo, referencia)
    reescalado = error_escalado_medio(y * factor, modelo * factor, referencia * factor)

    assert reescalado == pytest.approx(original, rel=1e-12)
    # El MAE, en cambio, se multiplica por el factor: por eso no se puede usar
    # para comparar mercados distintos.
    assert error_absoluto_medio(y * factor, modelo * factor) == pytest.approx(
        factor * error_absoluto_medio(y, modelo), rel=1e-12
    )


def test_un_modelo_con_senal_real_baja_del_uno_y_la_media_movil_no(caso_con_senal):
    y, modelo = caso_con_senal

    assert error_escalado_medio(y, modelo) < 0.95
    # La media movil de los rendimientos no aporta nada sobre predecir cero: su
    # MASE se queda pegado a 1 (por arriba o por abajo segun el tramo, nunca
    # claramente por debajo), que es justo lo que se espera de una referencia
    # que solo estima una deriva inexistente.
    assert 0.98 < error_escalado_medio(y, prediccion_media_movil(y, 21)) < 1.10


def test_la_raiz_del_error_cuadratico_nunca_es_menor_que_el_error_absoluto(caso_con_senal):
    y, modelo = caso_con_senal

    assert raiz_error_cuadratico_medio(y, modelo) >= error_absoluto_medio(y, modelo)


def test_con_errores_del_mismo_tamanio_las_dos_metricas_coinciden():
    indice = pd.bdate_range("2022-01-03", periods=6, name="fecha")
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=indice)
    pred = pd.Series([1.5, 1.5, 3.5, 3.5, 5.5, 5.5], index=indice)

    assert error_absoluto_medio(y, pred) == pytest.approx(0.5)
    assert raiz_error_cuadratico_medio(y, pred) == pytest.approx(0.5)


def test_una_referencia_sin_error_deja_el_mase_indefinido():
    indice = pd.bdate_range("2022-01-03", periods=10, name="fecha")
    constante = pd.Series(np.full(10, 5.0), index=indice)

    with pytest.raises(ErrorDatos, match="MASE"):
        error_escalado_medio(constante, constante + 1.0, constante)


def test_las_fechas_sin_dato_se_descartan_por_parejas():
    indice = pd.bdate_range("2022-01-03", periods=4, name="fecha")
    y = pd.Series([1.0, 2.0, 3.0, 4.0], index=indice)
    pred = pd.Series([2.0, np.nan, 5.0, 4.0], index=indice)

    # Se comparan solo las tres fechas completas: |1-2|, |3-5| y |4-4|.
    assert error_absoluto_medio(y, pred) == pytest.approx(1.0)


def test_sin_ninguna_fecha_comparable_se_avisa():
    indice = pd.bdate_range("2022-01-03", periods=3, name="fecha")
    y = pd.Series([1.0, 2.0, 3.0], index=indice)
    pred = pd.Series([np.nan, np.nan, np.nan], index=indice)

    with pytest.raises(ErrorDatos, match="ninguna fecha"):
        error_absoluto_medio(y, pred)


# --- acierto direccional -------------------------------------------------------


def test_invertir_el_signo_da_el_acierto_direccional_complementario(caso_con_senal):
    y, modelo = caso_con_senal

    original = acierto_direccional(y, modelo)

    assert acierto_direccional(y, -modelo) == pytest.approx(1.0 - original)
    assert 0.0 < original < 1.0


def test_el_acierto_direccional_ignora_los_dias_sin_direccion_y_sin_apuesta():
    indice = pd.bdate_range("2022-01-03", periods=5, name="fecha")
    y = pd.Series([1.0, -1.0, 2.0, 0.0, 3.0], index=indice)
    pred = pd.Series([2.0, 1.0, 1.0, 5.0, 0.0], index=indice)

    # Quedan tres fechas: se acierta el signo en dos. El dia de rendimiento
    # nulo no tiene direccion que acertar y el dia sin apuesta no es una
    # apuesta fallada, porque si lo fuese invertir el signo no daria el
    # complementario.
    assert acierto_direccional(y, pred) == pytest.approx(2.0 / 3.0)
    assert acierto_direccional(y, -pred) == pytest.approx(1.0 / 3.0)


def test_quien_predice_siempre_cero_no_tiene_acierto_direccional(rendimientos_ruido):
    ingenua = prediccion_ingenua(rendimientos_ruido)

    assert np.isnan(acierto_direccional(rendimientos_ruido, ingenua))


# --- coeficiente de informacion ------------------------------------------------


def test_spearman_no_cambia_con_una_transformacion_monotona(caso_con_senal):
    y, modelo = caso_con_senal
    deformada = np.exp(300.0 * modelo)

    assert correlacion_informacion(y, deformada) == pytest.approx(
        correlacion_informacion(y, modelo)
    )
    # Pearson si se resiente: la deformacion cambia las magnitudes y basta con
    # eso para que la correlacion lineal caiga.
    assert correlacion_informacion(y, deformada, "pearson") < correlacion_informacion(
        y, modelo, "pearson"
    )


def test_una_prediccion_constante_no_tiene_coeficiente_de_informacion(rendimientos_ruido):
    constante = pd.Series(0.01, index=rendimientos_ruido.index)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado = correlacion_informacion(rendimientos_ruido, constante)

    assert np.isnan(resultado)


def test_una_prediccion_sin_relacion_tiene_coeficiente_de_informacion_casi_nulo(
    rendimientos_ruido,
):
    generador = np.random.default_rng(777)
    valores = generador.normal(size=len(rendimientos_ruido))
    ajena = pd.Series(valores, index=rendimientos_ruido.index)

    # Con 500 fechas la desviacion tipica del IC bajo independencia ronda 0,045.
    assert abs(correlacion_informacion(rendimientos_ruido, ajena)) < 0.2


@pytest.mark.parametrize("metodo", ["spearman", "pearson", "kendall"])
def test_los_tres_metodos_detectan_una_relacion_perfecta_creciente(metodo, caso_con_senal):
    y, _ = caso_con_senal

    assert correlacion_informacion(y, y, metodo) == pytest.approx(1.0)


def test_un_metodo_de_correlacion_desconocido_se_rechaza(caso_con_senal):
    y, modelo = caso_con_senal

    with pytest.raises(ErrorDatos, match="spearman"):
        correlacion_informacion(y, modelo, metodo="informacion_mutua")


# --- Diebold-Mariano -----------------------------------------------------------


def test_dos_predicciones_identicas_no_se_distinguen():
    generador = np.random.default_rng(3)
    errores = generador.normal(size=300)

    resultado = contraste_diebold_mariano(errores, errores.copy())

    assert resultado["diferencia_media"] == 0.0
    assert resultado["estadistico"] == 0.0
    assert resultado["p_valor"] == 1.0
    assert resultado["mejor"] == "empate"
    assert resultado["rechaza_al_5pct"] is False


def test_una_prediccion_claramente_mejor_da_un_p_valor_bajo():
    generador = np.random.default_rng(11)
    errores_buenos = generador.normal(0.0, 0.5, 300)
    errores_malos = generador.normal(0.0, 2.0, 300)

    resultado = contraste_diebold_mariano(errores_buenos, errores_malos)

    assert resultado["mejor"] == "a"
    assert resultado["diferencia_media"] < 0.0
    assert resultado["p_valor"] < 0.01
    assert resultado["rechaza_al_5pct"] is True


def test_intercambiar_las_dos_predicciones_solo_cambia_el_signo():
    generador = np.random.default_rng(17)
    errores_a = generador.normal(0.0, 1.0, 250)
    errores_b = generador.normal(0.0, 1.4, 250)

    directo = contraste_diebold_mariano(errores_a, errores_b)
    inverso = contraste_diebold_mariano(errores_b, errores_a)

    assert inverso["estadistico"] == pytest.approx(-directo["estadistico"])
    assert inverso["p_valor"] == pytest.approx(directo["p_valor"])
    assert inverso["diferencia_media"] == pytest.approx(-directo["diferencia_media"])
    assert {directo["mejor"], inverso["mejor"]} == {"a", "b"}


def test_con_dos_modelos_igual_de_buenos_se_rechaza_en_torno_al_cinco_por_ciento():
    """El contraste debe equivocarse poco, no nunca: esa es su calibracion.

    Bajo la hipotesis nula (dos modelos con la misma precision) un contraste al
    5 % debe rechazar alrededor del 5 % de las veces. Si rechazase mucho mas,
    cada comparacion de modelos produciria hallazgos inventados; si no
    rechazase casi nunca, no detectaria las mejoras de verdad. Se repite el
    experimento con muestras independientes en lugar de fiarlo a una semilla.
    """
    generador = np.random.default_rng(20240924)
    rechazos = sum(
        contraste_diebold_mariano(
            generador.normal(0.0, 1.0, 200), generador.normal(0.0, 1.0, 200)
        )["rechaza_al_5pct"]
        for _ in range(200)
    )

    assert 0.0 < rechazos / 200 < 0.15


def test_la_correccion_de_muestras_pequenias_encoge_el_estadistico():
    generador = np.random.default_rng(23)
    errores_a = generador.normal(0.0, 1.0, 40)
    errores_b = generador.normal(0.0, 1.6, 40)

    resultado = contraste_diebold_mariano(errores_a, errores_b)

    diferencia = errores_a**2 - errores_b**2
    sin_corregir = float(
        diferencia.mean() / np.sqrt(diferencia.var(ddof=0) / diferencia.size)
    )
    # Harvey-Leybourne-Newbold multiplica por sqrt((n - 1) / n) a horizonte 1.
    assert resultado["estadistico"] == pytest.approx(sin_corregir * np.sqrt(39 / 40))
    assert abs(resultado["estadistico"]) < abs(sin_corregir)


def test_a_horizonte_largo_el_solapamiento_ensancha_el_intervalo():
    """Con errores autocorrelacionados, ignorar el solapamiento infla la evidencia.

    Se contrastan dos modelos identicos en distribucion pero con errores
    persistentes: al tener en cuenta las autocovarianzas hasta el retardo
    ``horizonte - 1``, la varianza de largo plazo crece y el estadistico se
    modera.
    """
    generador = np.random.default_rng(29)
    ruido_a = generador.normal(size=400)
    ruido_b = generador.normal(size=400)
    persistente_a = pd.Series(ruido_a).rolling(5).mean().dropna().to_numpy()
    persistente_b = pd.Series(ruido_b * 1.2).rolling(5).mean().dropna().to_numpy()

    corto = contraste_diebold_mariano(persistente_a, persistente_b, horizonte=1)
    largo = contraste_diebold_mariano(persistente_a, persistente_b, horizonte=5)

    assert abs(largo["estadistico"]) < abs(corto["estadistico"])
    assert largo["p_valor"] > corto["p_valor"]


def test_la_potencia_uno_compara_errores_absolutos():
    indice = pd.bdate_range("2022-01-03", periods=4, name="fecha")
    errores_a = pd.Series([1.0, -1.0, 1.0, -1.0], index=indice)
    errores_b = pd.Series([2.0, -2.0, 2.0, -2.0], index=indice)

    con_absolutos = contraste_diebold_mariano(errores_a, errores_b, potencia=1)
    con_cuadrados = contraste_diebold_mariano(errores_a, errores_b, potencia=2)

    assert con_absolutos["diferencia_media"] == pytest.approx(-1.0)
    assert con_cuadrados["diferencia_media"] == pytest.approx(-3.0)
    # La diferencia es constante: no hay variabilidad muestral que contrastar y
    # la respuesta prudente es no rechazar, nunca un estadistico infinito.
    assert con_absolutos["mejor"] == "a"
    assert np.isfinite(con_absolutos["estadistico"])
    assert con_absolutos["p_valor"] == 1.0


@pytest.mark.parametrize(
    "escala_a, escala_b, n, horizonte, potencia",
    [
        (1.0, 1.0, 300, 1, 2.0),
        (0.2, 3.0, 300, 1, 2.0),
        (1.0, 1.0, 5, 1, 1.0),
        (1.0, 2.0, 50, 10, 2.0),
        (3.0, 0.1, 120, 4, 0.5),
        (1.0, 1.0, 2, 1, 2.0),
    ],
)
def test_el_p_valor_siempre_esta_en_el_intervalo_unidad(escala_a, escala_b, n, horizonte, potencia):
    generador = np.random.default_rng(101)
    resultado = contraste_diebold_mariano(
        generador.normal(0.0, escala_a, n),
        generador.normal(0.0, escala_b, n),
        horizonte=horizonte,
        potencia=potencia,
    )

    assert 0.0 <= resultado["p_valor"] <= 1.0
    assert np.isfinite(resultado["estadistico"])
    assert np.isfinite(resultado["diferencia_media"])
    assert resultado["mejor"] in {"a", "b", "empate"}
    assert set(resultado) == {
        "estadistico",
        "p_valor",
        "mejor",
        "diferencia_media",
        "rechaza_al_5pct",
    }


def test_una_muestra_demasiado_corta_para_el_horizonte_se_rechaza():
    generador = np.random.default_rng(5)

    with pytest.raises(ErrorDatos, match="demasiado corta"):
        contraste_diebold_mariano(
            generador.normal(size=10), generador.normal(size=10), horizonte=10
        )


def test_una_potencia_no_positiva_se_rechaza():
    with pytest.raises(ErrorDatos, match="potencia"):
        contraste_diebold_mariano(np.ones(10), np.zeros(10), potencia=0)


# --- entradas incompatibles ----------------------------------------------------


def test_dos_series_de_distinta_longitud_se_rechazan(rendimientos_ruido):
    with pytest.raises(ErrorDatos, match="observaciones"):
        error_absoluto_medio(rendimientos_ruido, rendimientos_ruido.iloc[:-5])


def test_dos_series_con_indices_incompatibles_se_rechazan(rendimientos_ruido):
    desplazada = rendimientos_ruido.copy()
    desplazada.index = pd.bdate_range("2021-01-01", periods=len(desplazada), name="fecha")

    with pytest.raises(ErrorDatos, match="indices distintos"):
        raiz_error_cuadratico_medio(rendimientos_ruido, desplazada)


def test_el_contraste_tambien_exige_series_compatibles():
    with pytest.raises(ErrorDatos, match="observaciones"):
        contraste_diebold_mariano(np.ones(20), np.ones(19))


def test_la_tabla_exige_indices_compatibles(rendimientos_ruido):
    desplazada = rendimientos_ruido.copy()
    desplazada.index = pd.bdate_range("2021-01-01", periods=len(desplazada), name="fecha")

    with pytest.raises(ErrorDatos, match="indices distintos"):
        tabla_evaluacion(rendimientos_ruido, {"desplazada": desplazada})


# --- tabla de evaluacion -------------------------------------------------------


def test_la_tabla_anade_la_ingenua_cuando_no_esta(caso_con_senal):
    y, modelo = caso_con_senal

    tabla = tabla_evaluacion(y, {"modelo": modelo})

    assert NOMBRE_INGENUA in tabla.index
    assert list(tabla.columns) == [
        "mae",
        "rmse",
        "mase",
        "acierto_direccional",
        "ic",
        "p_valor_dm",
        "n_observaciones",
    ]


def test_en_la_tabla_la_referencia_vale_exactamente_uno_y_no_se_distingue_de_si_misma(
    caso_con_senal,
):
    y, modelo = caso_con_senal

    tabla = tabla_evaluacion(y, {"modelo": modelo, "media_movil": prediccion_media_movil(y)})

    assert tabla.loc[NOMBRE_INGENUA, "mase"] == 1.0
    assert tabla.loc[NOMBRE_INGENUA, "p_valor_dm"] == 1.0


def test_la_tabla_esta_ordenada_por_mase_ascendente(caso_con_senal):
    y, modelo = caso_con_senal
    generador = np.random.default_rng(64)
    malo = pd.Series(generador.normal(0.0, 0.05, len(y)), index=y.index)

    tabla = tabla_evaluacion(
        y, {"malo": malo, "modelo": modelo, "media_movil": prediccion_media_movil(y)}
    )

    assert tabla["mase"].is_monotonic_increasing
    assert tabla.index[0] == "modelo"
    assert tabla.index[-1] == "malo"


def test_la_tabla_senala_al_modelo_bueno_como_significativamente_mejor(caso_con_senal):
    y, modelo = caso_con_senal

    tabla = tabla_evaluacion(y, {"modelo": modelo})

    assert tabla.loc["modelo", "mase"] < 1.0
    assert tabla.loc["modelo", "p_valor_dm"] < 0.05
    assert tabla.loc["modelo", "ic"] > 0.2


def test_todos_los_modelos_se_miden_sobre_las_mismas_fechas(caso_con_senal):
    y, modelo = caso_con_senal

    sin_calentamiento = tabla_evaluacion(y, {"modelo": modelo})
    con_media_movil = tabla_evaluacion(
        y, {"modelo": modelo, "media_movil": prediccion_media_movil(y, 21)}
    )

    assert sin_calentamiento["n_observaciones"].nunique() == 1
    assert con_media_movil["n_observaciones"].nunique() == 1
    # Con la ingenua (un dia de arranque) se evaluan 499 de las 500 fechas; al
    # entrar la media movil de 21 sesiones el periodo se acorta para todos los
    # modelos por igual, porque comparar cada uno en su propio tramo no seria
    # comparar.
    assert sin_calentamiento["n_observaciones"].iloc[0] == len(y) - 1
    assert con_media_movil["n_observaciones"].iloc[0] == len(y) - 21


def test_se_puede_tomar_como_referencia_cualquier_modelo_de_la_tabla(caso_con_senal):
    y, modelo = caso_con_senal
    media_movil = prediccion_media_movil(y, 21)

    tabla = tabla_evaluacion(
        y, {"modelo": modelo, "media_movil": media_movil}, referencia="media_movil"
    )

    assert tabla.loc["media_movil", "mase"] == 1.0
    assert NOMBRE_INGENUA not in tabla.index
    assert tabla.attrs["referencia"] == "media_movil"


def test_una_referencia_que_no_esta_en_la_tabla_se_rechaza(caso_con_senal):
    y, modelo = caso_con_senal

    with pytest.raises(ErrorDatos, match="no esta entre las predicciones"):
        tabla_evaluacion(y, {"modelo": modelo}, referencia="arima")


def test_una_tabla_sin_predicciones_se_rechaza(caso_con_senal):
    y, _ = caso_con_senal

    with pytest.raises(ErrorDatos, match="ninguna prediccion"):
        tabla_evaluacion(y, {})
