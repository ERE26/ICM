"""Pruebas del contraste de los modelos de VaR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.datos import rendimientos, simular_precios
from icm.riesgo import (
    backtest_var,
    comparar_metodos_var,
    contraste_christoffersen_condicional,
    contraste_christoffersen_independencia,
    contraste_kupiec,
    diagnostico_es,
    excepciones,
    serie_var_movil,
    zona_basilea,
)


@pytest.fixture(scope="module")
def serie_larga() -> pd.Series:
    """Serie de 2500 sesiones: suficiente para que el backtest tenga potencia."""
    precios = simular_precios(["A"], n_periodos=2500, semilla=31)
    return rendimientos(precios, "log")["A"]


# --- estimacion movil ----------------------------------------------------------


def test_el_var_movil_usa_exactamente_la_ventana_anterior(serie_larga):
    # Comprobacion directa de que no hay fuga de informacion: la estimacion de
    # cada fecha debe coincidir con calcular el VaR a mano sobre las 250
    # sesiones estrictamente anteriores.
    from icm.riesgo import var_historico

    var_movil = serie_var_movil(serie_larga, 0.99, 250, "historico")
    for posicion in (250, 400, len(serie_larga) - 1):
        ventana = serie_larga.iloc[posicion - 250 : posicion]
        assert var_movil.loc[serie_larga.index[posicion]] == pytest.approx(
            var_historico(ventana, 0.99)
        )


def test_el_var_movil_reacciona_al_cambio_de_regimen():
    # Trescientas sesiones tranquilas seguidas de trescientas turbulentas.
    generador = np.random.default_rng(41)
    serie = pd.Series(
        np.concatenate(
            [generador.standard_normal(300) * 0.002, generador.standard_normal(300) * 0.02]
        )
    )
    var_movil = serie_var_movil(serie, 0.99, ventana=250, metodo="historico")
    # Al entrar en el nuevo regimen la ventana aun es toda tranquila.
    assert var_movil.iloc[0] < 0.01
    # Cien sesiones despues ya ha absorbido la turbulencia.
    assert var_movil.iloc[150] > 3 * var_movil.iloc[0]


def test_el_var_movil_deja_fuera_la_ventana_inicial(serie_larga):
    var_movil = serie_var_movil(serie_larga, 0.99, 250, "historico")
    assert len(var_movil) == len(serie_larga) - 250
    assert var_movil.index[0] == serie_larga.index[250]


@pytest.mark.parametrize(
    "metodo", ["historico", "parametrico_normal", "parametrico_t", "cornish_fisher", "ewma"]
)
def test_todos_los_metodos_producen_un_var_positivo(serie_larga, metodo):
    # Basta una ventana de evaluacion corta: se comprueba la forma del
    # resultado, no su potencia estadistica, y el ajuste de la t en cada
    # ventana movil es caro.
    var_movil = serie_var_movil(serie_larga.tail(600), 0.99, 250, metodo)
    assert (var_movil > 0).all()
    assert len(var_movil) == 350


def test_cornish_fisher_avisa_una_sola_vez_en_el_backtest(serie_larga):
    # Un aviso por ventana llenaria la pantalla; el resumen debe decir en
    # cuantas ventanas ocurre.
    with pytest.warns(UserWarning, match=r"ventanas \(\d+%\)") as avisos:
        serie_var_movil(serie_larga.tail(600), 0.99, 250, "cornish_fisher")
    del_dominio = [a for a in avisos if "dominio de validez" in str(a.message)]
    assert len(del_dominio) == 1


def test_el_var_movil_garch_funciona(serie_larga):
    var_movil = serie_var_movil(
        serie_larga.tail(700), 0.99, 250, "garch", frecuencia_reajuste=100
    )
    assert (var_movil > 0).all()
    # Un VaR condicional tiene que moverse mucho mas que uno historico.
    historico = serie_var_movil(serie_larga.tail(700), 0.99, 250, "historico")
    assert var_movil.std() > historico.std()


def test_una_ventana_demasiado_corta_da_error(serie_larga):
    with pytest.raises(ErrorDatos, match="al menos 30"):
        serie_var_movil(serie_larga, 0.99, 10)


def test_una_serie_mas_corta_que_la_ventana_da_error():
    with pytest.raises(ErrorDatos, match="Hacen falta mas de"):
        serie_var_movil(pd.Series(np.zeros(100) + 0.001), 0.99, 250)


def test_un_metodo_de_backtest_desconocido(serie_larga):
    with pytest.raises(ErrorDatos, match="desconocido"):
        serie_var_movil(serie_larga, 0.99, 250, "adivinacion")


# --- excepciones ---------------------------------------------------------------


def test_las_excepciones_se_marcan_por_debajo_del_var():
    rend = pd.Series([-0.05, 0.01, -0.02, -0.10], index=pd.RangeIndex(4))
    var_estimado = pd.Series([0.03, 0.03, 0.03, 0.03], index=pd.RangeIndex(4))
    np.testing.assert_array_equal(excepciones(rend, var_estimado).to_numpy(), [1, 0, 0, 1])


def test_las_series_se_alinean_por_fecha():
    rend = pd.Series([-0.05, 0.01, -0.10], index=[1, 2, 3])
    var_estimado = pd.Series([0.03, 0.03], index=[2, 3])
    resultado = excepciones(rend, var_estimado)
    assert list(resultado.index) == [2, 3]


def test_sin_fechas_comunes_da_error():
    rend = pd.Series([0.01], index=[1])
    var_estimado = pd.Series([0.03], index=[99])
    with pytest.raises(ErrorDatos, match="no comparten fechas"):
        excepciones(rend, var_estimado)


# --- Kupiec --------------------------------------------------------------------


def test_kupiec_no_rechaza_la_proporcion_correcta():
    # Exactamente el 1 % de excepciones en 1000 dias.
    fallos = np.zeros(1000, dtype=int)
    fallos[:10] = 1
    resultado = contraste_kupiec(fallos, 0.99)
    assert resultado["estadistico_lr"] == pytest.approx(0.0, abs=1e-9)
    assert not resultado["rechaza_al_5pct"]


def test_kupiec_rechaza_un_exceso_flagrante():
    fallos = np.zeros(1000, dtype=int)
    fallos[:60] = 1  # 6 % frente al 1 % esperado
    resultado = contraste_kupiec(fallos, 0.99)
    assert resultado["rechaza_al_5pct"]
    assert resultado["proporcion_observada"] == pytest.approx(0.06)


def test_kupiec_rechaza_un_modelo_demasiado_prudente():
    # Cero excepciones en 2000 dias al 95 %: el VaR es absurdamente alto.
    resultado = contraste_kupiec(np.zeros(2000, dtype=int), 0.95)
    assert resultado["rechaza_al_5pct"]
    assert resultado["excepciones"] == 0


def test_kupiec_cuenta_bien_lo_esperado():
    resultado = contraste_kupiec(np.zeros(500, dtype=int), 0.99)
    assert resultado["excepciones_esperadas"] == pytest.approx(5.0)


def test_kupiec_sin_observaciones_da_error():
    with pytest.raises(ErrorDatos, match="No hay observaciones"):
        contraste_kupiec(np.array([], dtype=int), 0.99)


# --- Christoffersen ------------------------------------------------------------


def test_la_independencia_no_se_rechaza_con_excepciones_dispersas():
    # Una excepcion cada 100 dias: perfectamente repartidas.
    fallos = np.zeros(1000, dtype=int)
    fallos[::100] = 1
    assert not contraste_christoffersen_independencia(fallos)["rechaza_al_5pct"]


def test_la_independencia_se_rechaza_con_excepciones_agrupadas():
    # Las mismas diez excepciones, pero todas seguidas.
    fallos = np.zeros(1000, dtype=int)
    fallos[500:510] = 1
    resultado = contraste_christoffersen_independencia(fallos)
    assert resultado["rechaza_al_5pct"]
    assert resultado["prob_excepcion_tras_excepcion"] > resultado["prob_excepcion_tras_normal"]


def test_la_matriz_de_transiciones_cuadra():
    fallos = np.array([0, 1, 1, 0, 0, 1, 0])
    resultado = contraste_christoffersen_independencia(fallos)
    total = resultado["n00"] + resultado["n01"] + resultado["n10"] + resultado["n11"]
    assert total == len(fallos) - 1


def test_el_contraste_condicional_suma_los_dos_estadisticos():
    fallos = np.zeros(1000, dtype=int)
    fallos[500:510] = 1
    kupiec = contraste_kupiec(fallos, 0.99)
    independencia = contraste_christoffersen_independencia(fallos)
    condicional = contraste_christoffersen_condicional(fallos, 0.99)
    assert condicional["estadistico_lr"] == pytest.approx(
        kupiec["estadistico_lr"] + independencia["estadistico_lr"]
    )


def test_el_contraste_condicional_detecta_el_agrupamiento_aunque_el_numero_sea_correcto():
    # Diez excepciones en 1000 dias es exactamente lo esperado al 99 %, pero
    # todas seguidas: Kupiec pasa y el condicional no.
    fallos = np.zeros(1000, dtype=int)
    fallos[500:510] = 1
    assert not contraste_kupiec(fallos, 0.99)["rechaza_al_5pct"]
    assert contraste_christoffersen_condicional(fallos, 0.99)["rechaza_al_5pct"]


def test_la_independencia_necesita_al_menos_dos_observaciones():
    with pytest.raises(ErrorDatos, match="al menos 2"):
        contraste_christoffersen_independencia(np.array([1]))


# --- semaforo de Basilea -------------------------------------------------------


@pytest.mark.parametrize(
    ("excepciones_observadas", "zona_esperada"),
    [(0, "verde"), (4, "verde"), (5, "amarilla"), (9, "amarilla"), (10, "roja"), (15, "roja")],
)
def test_el_semaforo_reproduce_los_umbrales_de_basilea(excepciones_observadas, zona_esperada):
    # Con la ventana supervisora estandar de 250 dias al 99 %.
    assert zona_basilea(excepciones_observadas, 250, 0.99)["zona"] == zona_esperada


def test_el_recargo_crece_dentro_de_la_zona_amarilla():
    recargos = [zona_basilea(n, 250, 0.99)["recargo_multiplicador"] for n in range(5, 10)]
    assert recargos == sorted(recargos)
    assert zona_basilea(4, 250, 0.99)["recargo_multiplicador"] == 0.0
    assert zona_basilea(10, 250, 0.99)["recargo_multiplicador"] == 1.0


def test_el_semaforo_se_adapta_a_otras_ventanas():
    # En 1000 dias al 99 % se esperan 10 excepciones, no 2.5: cinco ya no
    # deberian encender el ambar.
    assert zona_basilea(5, 1000, 0.99)["zona"] == "verde"


# --- backtest completo ---------------------------------------------------------


def test_el_backtest_devuelve_un_resumen_coherente(serie_larga):
    resultado = backtest_var(serie_larga, 0.99, 250, "historico")
    resumen = resultado.resumen
    assert resumen["observaciones"] == len(serie_larga) - 250
    assert resumen["excepciones"] == resultado.excepciones.sum()
    assert resumen["zona_basilea"] in {"verde", "amarilla", "roja"}
    assert isinstance(resumen["modelo_valido"], (bool, np.bool_))


def test_un_modelo_correcto_pasa_el_backtest(precios_sin_agrupamiento):
    # Datos casi normales y sin agrupamiento: el VaR parametrico normal es el
    # modelo correcto y debe superar el contraste.
    serie = rendimientos(precios_sin_agrupamiento, "log")["ALFA"]
    resultado = backtest_var(serie, 0.99, 250, "parametrico_normal")
    assert not resultado.condicional["rechaza_al_5pct"]
    assert resultado.basilea["zona"] == "verde"


def test_un_var_deliberadamente_bajo_falla_el_backtest(serie_larga):
    # Se toma el VaR historico y se reduce a la mitad: tiene que saltar.
    var_correcto = serie_var_movil(serie_larga, 0.99, 250, "historico")
    fallos = excepciones(serie_larga, var_correcto * 0.5)
    assert contraste_kupiec(fallos, 0.99)["rechaza_al_5pct"]


def test_la_comparacion_de_metodos_devuelve_una_fila_por_metodo(serie_larga):
    metodos = ("historico", "parametrico_normal", "ewma")
    tabla = comparar_metodos_var(serie_larga, 0.99, 250, metodos)
    assert list(tabla["metodo"]) == list(metodos)
    assert (tabla["observaciones"] == len(serie_larga) - 250).all()


def test_el_diagnostico_del_es_compara_con_lo_observado(serie_larga):
    var_movil = serie_var_movil(serie_larga, 0.95, 250, "historico")
    resultado = diagnostico_es(serie_larga, var_movil, es_estimado=var_movil * 1.3)
    assert resultado["excepciones"] > 0
    assert resultado["ic_inferior_95"] <= resultado["perdida_media_observada"]
    assert resultado["perdida_media_observada"] <= resultado["ic_superior_95"]


def test_el_diagnostico_del_es_falla_sin_excepciones(serie_larga):
    enorme = serie_var_movil(serie_larga, 0.99, 250, "historico") * 100
    with pytest.raises(ErrorDatos, match="ninguna excepcion"):
        diagnostico_es(serie_larga, enorme, es_estimado=enorme)


# --- lo que la segunda revision adversarial encontro ---------------------------


def test_los_p_valores_no_colapsan_a_cero():
    # '1 - chi2.cdf' agota la precision del flotante y devuelve 0,0 exacto a
    # partir de un estadistico de unos 75. Un p-valor de cero exacto no es un
    # p-valor, y ademas impide distinguir un rechazo claro de uno aplastante.
    fallos = np.zeros(400, dtype=int)
    fallos[100:140] = 1  # racha larga: los dos contrastes se van muy arriba
    serie = pd.Series(fallos, index=pd.bdate_range("2024-01-01", periods=400))

    for resultado in (
        contraste_kupiec(serie, 0.99),
        contraste_christoffersen_independencia(serie),
        contraste_christoffersen_condicional(serie, 0.99),
    ):
        assert resultado["estadistico_lr"] > 75.0
        assert resultado["p_valor"] > 0.0
        assert resultado["p_valor"] < 1e-15
        assert resultado["rechaza_al_5pct"]


def test_la_independencia_exige_orden_cronologico():
    # El contraste cuenta transiciones de un dia al siguiente sobre la
    # secuencia que recibe, asi que el orden ES el dato: con las mismas 12
    # excepciones agrupadas, barajar el indice llevaba el estadistico de 76,38
    # (rechaza) a 0,37 (no rechaza).
    fallos = np.zeros(250, dtype=int)
    fallos[100:112] = 1
    serie = pd.Series(fallos, index=pd.bdate_range("2024-01-01", periods=250))

    ordenado = contraste_christoffersen_independencia(serie)
    assert ordenado["estadistico_lr"] > 70.0
    assert ordenado["rechaza_al_5pct"]

    with pytest.raises(ErrorDatos, match="no esta ordenada"):
        contraste_christoffersen_independencia(serie.sample(frac=1.0, random_state=3))


def test_kupiec_no_acepta_una_serie_que_no_sea_de_excepciones():
    # 'asarray(dtype=int)' truncaba hacia cero, asi que una serie de
    # rendimientos pasada por error se convertia en ceros y colaba como
    # "ninguna excepcion", con su Kupiec y todo.
    with pytest.raises(ErrorDatos, match="solo pueden valer 0 o 1"):
        contraste_kupiec(pd.Series([0.01, -0.02, 0.005]), 0.99)
    # A Kupiec el orden le da igual porque solo cuenta.
    fallos = pd.Series([0, 1, 0, 0, 1], index=pd.bdate_range("2024-01-01", periods=5))
    assert contraste_kupiec(fallos.sample(frac=1.0, random_state=1), 0.99)["excepciones"] == 2.0


def test_el_recargo_de_basilea_solo_se_da_en_su_ventana_de_250_dias():
    # La zona la calcula la binomial y vale para cualquier ventana, pero el
    # recargo es la tabla del Comite para 250 sesiones. Antes se leia esa
    # tabla pasase lo que pasase, con un get(..., 0.85) que devolvia el tope
    # para cualquier recuento no tabulado: con 500 observaciones y 12
    # excepciones daba 0,85 con aspecto de cifra regulatoria.
    normativa = zona_basilea(9, 250, 0.99)
    assert normativa["zona"] == "amarilla"
    assert normativa["recargo_multiplicador"] == pytest.approx(0.85)
    assert normativa["recargo_aplicable"]

    fuera = zona_basilea(12, 500, 0.99)
    assert fuera["zona"] == "amarilla"  # la binomial sigue diciendo amarilla
    assert np.isnan(fuera["recargo_multiplicador"])
    assert not fuera["recargo_aplicable"]

    # Verde y roja si tienen recargo definido para cualquier ventana.
    assert zona_basilea(5, 1000, 0.99)["recargo_aplicable"]
    assert zona_basilea(40, 500, 0.99)["recargo_multiplicador"] == pytest.approx(1.00)


def test_basilea_rechaza_recuentos_imposibles():
    with pytest.raises(ErrorDatos, match="No puede haber"):
        zona_basilea(300, 250, 0.99)
    with pytest.raises(ErrorDatos, match="imposibles"):
        zona_basilea(-1, 250, 0.99)


def test_cornish_fisher_ya_no_avisa_en_la_mitad_de_las_ventanas():
    # Con el criterio global, cualquier serie -- incluida una normal -- salia
    # fuera de dominio en mas de la mitad de las ventanas, porque basta una
    # curtosis de exceso negativa por minima que sea.
    import warnings as _warnings

    serie = pd.Series(
        np.random.default_rng(4).standard_normal(700) * 0.01,
        index=pd.bdate_range("2022-01-03", periods=700),
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        var_movil = serie_var_movil(serie, 0.99, 250, "cornish_fisher")
    assert len(var_movil) == 450
    assert (var_movil > 0).all()
