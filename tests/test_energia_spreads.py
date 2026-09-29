"""Pruebas de los spreads de generacion y de refino.

Casi todo se contrasta contra numeros calculados a mano o contra propiedades
que deben cumplirse siempre (monotonia en la eficiencia, dominancia de la
opcionalidad sobre el spread, efecto del CO2). Las series aleatorias salen de
``np.random.default_rng`` con semilla fija, asi que ninguna prueba depende de
una semilla afortunada: las que usan aleatoriedad comprueban desigualdades que
se cumplen observacion a observacion.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.energia.spreads import (
    GALONES_POR_BARRIL,
    PLANTAS,
    ParametrosPlanta,
    crack_spread,
    dark_spread,
    dark_spread_limpio,
    horas_en_dinero,
    margen_planta,
    resumen_spread,
    spark_spread,
    spark_spread_limpio,
    spread_calendario,
    spread_cambio_combustible,
    spread_locacional,
    valor_opcionalidad,
)

N_PERIODOS = 400


@pytest.fixture(scope="module")
def indice() -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", periods=N_PERIODOS)


@pytest.fixture(scope="module")
def precios(indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Precios diarios plausibles de pool, gas, carbon y CO2.

    Los niveles imitan al mercado europeo de 2022 (electricidad en el entorno
    de 150 EUR/MWh, gas en 80, carbon en 25 y derechos en 80), con dispersion
    suficiente para que el spark spread cambie de signo varias veces.
    """
    generador = np.random.default_rng(2024)
    n = len(indice)
    return pd.DataFrame(
        {
            "pool": 150.0 + 40.0 * generador.standard_normal(n),
            "gas": 80.0 + 15.0 * generador.standard_normal(n),
            "carbon": 25.0 + 5.0 * generador.standard_normal(n),
            "co2": 80.0 + 10.0 * generador.standard_normal(n),
        },
        index=indice,
    ).clip(lower=1.0)


# --- valores calculados a mano -------------------------------------------------


def test_el_spark_spread_con_precios_constantes_da_el_numero_exacto():
    # 1 MWh electrico al 50 % exige 2 MWh termicos de gas: 100 - 30/0.5 = 40.
    spread = spark_spread(100.0, 30.0, eficiencia=0.50)
    assert spread.iloc[0] == pytest.approx(40.0)

    con_costes = spark_spread(100.0, 30.0, eficiencia=0.50, costes_variables=2.0)
    assert con_costes.iloc[0] == pytest.approx(38.0)


def test_el_dark_spread_con_precios_constantes_da_el_numero_exacto():
    # 80 - 12/0.38 = 80 - 31.578947... = 48.421052...
    assert dark_spread(80.0, 12.0).iloc[0] == pytest.approx(80.0 - 12.0 / 0.38)


def test_dividir_por_la_eficiencia_no_es_multiplicar_por_ella():
    """El error clasico da un margen absurdamente alto; se comprueba la brecha."""
    correcto = float(spark_spread(100.0, 30.0, eficiencia=0.40).iloc[0])
    error_clasico = 100.0 - 30.0 * 0.40
    assert correcto == pytest.approx(25.0)
    assert error_clasico == pytest.approx(88.0)
    assert correcto < error_clasico


def test_el_spread_limpio_resta_las_emisiones_por_mwh_electrico():
    # 0.20 tCO2/MWh termico / 0.50 = 0.40 t por MWh electrico; a 110 EUR/t son 44.
    limpio = spark_spread_limpio(100.0, 40.0, 110.0, eficiencia=0.50, factor_emision=0.20)
    assert limpio.iloc[0] == pytest.approx(-24.0)


def test_con_co2_caro_el_spread_limpio_es_negativo_y_el_sucio_positivo():
    """El caso que decide si la planta arranca: sucio +20, limpio -24."""
    sucio = spark_spread(100.0, 40.0, eficiencia=0.50)
    limpio = spark_spread_limpio(100.0, 40.0, 110.0, eficiencia=0.50)
    assert float(sucio.iloc[0]) > 0.0
    assert float(limpio.iloc[0]) < 0.0
    assert float(sucio.iloc[0]) == pytest.approx(20.0)


def test_el_dark_spread_limpio_penaliza_mas_que_el_spark_limpio():
    """0.34/0.38 = 0.89 tCO2 por MWh electrico frente a 0.20/0.50 = 0.40."""
    coste_gas = float(
        spark_spread(100.0, 40.0).iloc[0] - spark_spread_limpio(100.0, 40.0, 100.0).iloc[0]
    )
    coste_carbon = float(
        dark_spread(100.0, 40.0).iloc[0] - dark_spread_limpio(100.0, 40.0, 100.0).iloc[0]
    )
    assert coste_gas == pytest.approx(100.0 * 0.20 / 0.50)
    assert coste_carbon == pytest.approx(100.0 * 0.34 / 0.38)
    assert coste_carbon > 2.0 * coste_gas


def test_el_crack_spread_321_da_el_numero_calculado_a_mano():
    """(2*42*2.50 + 1*42*2.80 - 3*80) / 3 = (210 + 117.6 - 240) / 3 = 29.2."""
    assert crack_spread(80.0, 2.50, 2.80).iloc[0] == pytest.approx(29.2)


def test_el_crack_spread_admite_otro_ratio():
    """5-3-2: (3*42*2.50 + 2*42*2.80 - 5*80) / 5 = (315 + 235.2 - 400) / 5 = 30.04."""
    assert crack_spread(80.0, 2.50, 2.80, ratio=(5, 3, 2)).iloc[0] == pytest.approx(30.04)


def test_olvidar_los_galones_por_barril_cambia_el_resultado():
    """Sin la conversion el margen seria negativo: es un error que salta a la vista."""
    sin_conversion = (2 * 2.50 + 1 * 2.80 - 3 * 80.0) / 3.0
    assert GALONES_POR_BARRIL == 42.0
    assert sin_conversion < 0.0 < float(crack_spread(80.0, 2.50, 2.80).iloc[0])


def test_los_spreads_de_estructura_dan_los_valores_exactos():
    assert spread_calendario(52.0, 47.0).iloc[0] == pytest.approx(5.0)
    assert spread_locacional(90.0, 80.0, coste_transporte=3.0).iloc[0] == pytest.approx(7.0)
    assert spread_locacional(90.0, 80.0).iloc[0] == pytest.approx(10.0)


# --- propiedades ---------------------------------------------------------------


@pytest.mark.parametrize("eficiencia", [0.30, 0.38, 0.45, 0.50, 0.60, 1.0])
def test_una_eficiencia_mayor_da_siempre_un_spread_mayor(precios, eficiencia):
    """Con el mismo gas se produce mas electricidad: el margen solo puede subir."""
    peor = spark_spread(precios["pool"], precios["gas"], eficiencia=eficiencia - 0.05)
    mejor = spark_spread(precios["pool"], precios["gas"], eficiencia=eficiencia)
    assert (mejor > peor).all()


def test_el_spread_limpio_nunca_supera_al_sucio(precios):
    sucio = spark_spread(precios["pool"], precios["gas"])
    limpio = spark_spread_limpio(precios["pool"], precios["gas"], precios["co2"])
    assert (limpio <= sucio).all()
    # Con precio del CO2 estrictamente positivo la desigualdad es estricta.
    assert (limpio < sucio).all()


def test_con_co2_nulo_el_spread_limpio_coincide_con_el_sucio(precios):
    sucio = spark_spread(precios["pool"], precios["gas"])
    limpio = spark_spread_limpio(precios["pool"], precios["gas"], 0.0)
    assert np.allclose(sucio.to_numpy(), limpio.to_numpy())


def test_los_costes_variables_restan_euro_a_euro(precios):
    sin_costes = spark_spread(precios["pool"], precios["gas"])
    con_costes = spark_spread(precios["pool"], precios["gas"], costes_variables=4.0)
    assert np.allclose((sin_costes - con_costes).to_numpy(), 4.0)


def test_el_spread_de_cambio_de_combustible_es_la_diferencia_de_los_limpios(precios):
    spark = spark_spread_limpio(precios["pool"], precios["gas"], precios["co2"])
    dark = dark_spread_limpio(precios["pool"], precios["carbon"], precios["co2"])
    climate = spread_cambio_combustible(spark, dark)
    assert np.allclose(climate.to_numpy(), (spark - dark).to_numpy())


def test_un_co2_mas_caro_favorece_al_gas_frente_al_carbon(precios):
    """Subir el precio del derecho empuja el climate spread hacia el gas."""

    def climate(precio_co2: float) -> float:
        spark = spark_spread_limpio(precios["pool"], precios["gas"], precio_co2)
        dark = dark_spread_limpio(precios["pool"], precios["carbon"], precio_co2)
        return float(spread_cambio_combustible(spark, dark).mean())

    assert climate(200.0) > climate(100.0) > climate(0.0)


def test_el_precio_de_cambio_de_combustible_es_un_punto_de_corte(precios):
    """Existe un precio del CO2 donde el climate spread cambia de signo."""

    def climate(precio_co2: float) -> float:
        spark = spark_spread_limpio(precios["pool"], precios["gas"], precio_co2)
        dark = dark_spread_limpio(precios["pool"], precios["carbon"], precio_co2)
        return float(spread_cambio_combustible(spark, dark).mean())

    assert climate(0.0) < 0.0 < climate(300.0)


def test_el_spread_locacional_es_antisimetrico(precios):
    ida = spread_locacional(precios["pool"], precios["gas"])
    vuelta = spread_locacional(precios["gas"], precios["pool"])
    assert np.allclose(ida.to_numpy(), -vuelta.to_numpy())


# --- opcionalidad y resumen ----------------------------------------------------


def test_la_opcionalidad_nunca_es_negativa_y_domina_al_spread(precios):
    spread = spark_spread_limpio(precios["pool"], precios["gas"], precios["co2"])
    pago = valor_opcionalidad(spread)
    assert (pago >= 0.0).all()
    assert (pago >= spread).all()


def test_valorar_como_si_produjese_siempre_infravalora_la_planta():
    """Desigualdad de Jensen: media de max(S,0) >= max(media de S, 0)."""
    generador = np.random.default_rng(7)
    spread = pd.Series(generador.normal(loc=-5.0, scale=25.0, size=5000))
    ingenuo = max(float(spread.mean()), 0.0)
    con_opcion = float(valor_opcionalidad(spread).mean())
    assert ingenuo == 0.0
    assert con_opcion > 5.0


def test_el_coste_de_arranque_descuenta_el_pago():
    spread = pd.Series([-10.0, 0.0, 3.0, 20.0])
    pago = valor_opcionalidad(spread, coste_arranque=5.0)
    assert list(pago) == [0.0, 0.0, 0.0, 15.0]


def test_horas_en_dinero_de_un_spread_siempre_positivo_es_uno():
    assert horas_en_dinero(pd.Series([1.0, 2.0, 3.0])) == 1.0


def test_horas_en_dinero_de_un_spread_siempre_negativo_es_cero():
    assert horas_en_dinero(pd.Series([-1.0, -2.0, -3.0])) == 0.0


def test_horas_en_dinero_no_cuenta_los_ceros():
    """Nadie arranca una turbina para cubrir exactamente el combustible."""
    assert horas_en_dinero(pd.Series([0.0, 0.0, 5.0])) == pytest.approx(1.0 / 3.0)


def test_resumen_spread_da_las_medidas_calculadas_a_mano():
    resumen = resumen_spread(pd.Series([-1.0, 0.0, 1.0, 2.0, 3.0]), nombre="ejemplo")
    assert resumen.name == "ejemplo"
    assert resumen["observaciones"] == 5.0
    assert resumen["media"] == pytest.approx(1.0)
    assert resumen["mediana"] == pytest.approx(1.0)
    assert resumen["horas_en_dinero"] == pytest.approx(0.6)
    assert resumen["opcionalidad_acumulada"] == pytest.approx(6.0)
    assert resumen["percentil_5"] <= resumen["percentil_95"]


def test_resumen_spread_hereda_el_nombre_de_la_serie(precios):
    resumen = resumen_spread(spark_spread(precios["pool"], precios["gas"]))
    assert resumen.name == "spark_spread"


# --- catalogo de plantas -------------------------------------------------------


def test_el_catalogo_trae_las_tecnologias_de_referencia():
    assert {"ciclo_combinado", "carbon", "ciclo_combinado_antiguo"} <= set(PLANTAS)
    assert PLANTAS["ciclo_combinado"].eficiencia == pytest.approx(0.50)
    assert PLANTAS["ciclo_combinado"].factor_emision == pytest.approx(0.20)
    assert PLANTAS["carbon"].eficiencia == pytest.approx(0.38)
    assert PLANTAS["carbon"].factor_emision == pytest.approx(0.34)
    assert PLANTAS["ciclo_combinado_antiguo"].eficiencia == pytest.approx(0.42)


def test_los_parametros_del_catalogo_son_fisicamente_plausibles():
    for clave, ficha in PLANTAS.items():
        assert 0.0 < ficha.eficiencia <= 1.0, clave
        assert 0.0 <= ficha.factor_emision <= 0.5, clave
        assert ficha.costes_variables >= 0.0, clave


def test_la_ficha_de_planta_es_inmutable():
    ficha = PLANTAS["ciclo_combinado"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ficha.eficiencia = 0.9


def test_una_ficha_de_planta_invalida_da_error():
    with pytest.raises(ErrorDatos, match="eficiencia"):
        ParametrosPlanta(nombre="imposible", eficiencia=1.4, factor_emision=0.2)
    with pytest.raises(ErrorDatos, match="factor_emision"):
        ParametrosPlanta(nombre="imposible", eficiencia=0.5, factor_emision=-0.1)


# --- margen de una planta ------------------------------------------------------


def test_margen_planta_reune_spreads_opcionalidad_y_acumulado(precios):
    marco = margen_planta(precios, "ciclo_combinado", "pool", "gas", "co2")
    esperadas = {
        "spread_sucio",
        "coste_co2",
        "spread_limpio",
        "valor_opcionalidad",
        "margen_acumulado",
    }
    assert esperadas == set(marco.columns)
    assert (marco["spread_limpio"] <= marco["spread_sucio"]).all()
    assert (marco["valor_opcionalidad"] >= 0.0).all()
    assert marco["margen_acumulado"].iloc[-1] == pytest.approx(marco["valor_opcionalidad"].sum())
    assert marco.attrs["eficiencia"] == pytest.approx(0.50)


def test_margen_planta_acumula_solo_lo_que_gana_cuando_produce(precios):
    """El acumulado suma la opcionalidad, no el spread: parar no resta."""
    marco = margen_planta(precios, "ciclo_combinado", "pool", "gas", "co2")
    assert (marco["spread_limpio"] < 0.0).any()
    assert marco["margen_acumulado"].is_monotonic_increasing
    assert marco["margen_acumulado"].iloc[-1] > marco["spread_limpio"].sum()


def test_margen_planta_sin_columna_de_co2_solo_da_el_sucio(precios):
    marco = margen_planta(precios, "carbon", "pool", "carbon")
    assert "spread_limpio" not in marco.columns
    assert "coste_co2" not in marco.columns
    esperado = dark_spread(
        precios["pool"],
        precios["carbon"],
        eficiencia=PLANTAS["carbon"].eficiencia,
        costes_variables=PLANTAS["carbon"].costes_variables,
    )
    assert np.allclose(marco["spread_sucio"].to_numpy(), esperado.to_numpy())


def test_margen_planta_acepta_una_ficha_propia(precios):
    ficha = ParametrosPlanta(nombre="unidad propia", eficiencia=0.58, factor_emision=0.20)
    propia = margen_planta(precios, ficha, "pool", "gas", "co2")
    catalogo = margen_planta(precios, "ciclo_combinado", "pool", "gas", "co2")
    assert (propia["spread_limpio"] > catalogo["spread_limpio"]).all()
    assert propia.attrs["planta"] == "unidad propia"


def test_margen_planta_con_planta_desconocida_da_error(precios):
    with pytest.raises(ErrorDatos, match="desconocida"):
        margen_planta(precios, "fusion_fria", "pool", "gas")


def test_margen_planta_con_columna_inexistente_da_error(precios):
    with pytest.raises(ErrorDatos, match="Faltan las columnas"):
        margen_planta(precios, "ciclo_combinado", "pool", "gasoil")


# --- errores de entrada --------------------------------------------------------


@pytest.mark.parametrize("eficiencia", [0.0, -0.1, 1.5, float("nan")])
def test_una_eficiencia_fuera_de_rango_da_error(eficiencia):
    with pytest.raises(ErrorDatos, match="eficiencia"):
        spark_spread(100.0, 30.0, eficiencia=eficiencia)
    with pytest.raises(ErrorDatos, match="eficiencia"):
        dark_spread(100.0, 30.0, eficiencia=eficiencia)
    with pytest.raises(ErrorDatos, match="eficiencia"):
        spark_spread_limpio(100.0, 30.0, 80.0, eficiencia=eficiencia)
    with pytest.raises(ErrorDatos, match="eficiencia"):
        dark_spread_limpio(100.0, 30.0, 80.0, eficiencia=eficiencia)


def test_una_eficiencia_del_cien_por_cien_es_valida():
    """El limite superior entra: es el maximo termodinamico, no un error."""
    assert spark_spread(100.0, 30.0, eficiencia=1.0).iloc[0] == pytest.approx(70.0)


@pytest.mark.parametrize("negativo", [-0.01, -5.0])
def test_los_costes_negativos_dan_error(negativo):
    with pytest.raises(ErrorDatos, match="costes_variables"):
        spark_spread(100.0, 30.0, costes_variables=negativo)
    with pytest.raises(ErrorDatos, match="factor_emision"):
        spark_spread_limpio(100.0, 30.0, 80.0, factor_emision=negativo)
    with pytest.raises(ErrorDatos, match="coste_arranque"):
        valor_opcionalidad(pd.Series([1.0, 2.0]), coste_arranque=negativo)
    with pytest.raises(ErrorDatos, match="coste_transporte"):
        spread_locacional(90.0, 80.0, coste_transporte=negativo)


@pytest.mark.parametrize("ratio", [(3, 2), (3, 2, 1, 1), (3, 0, 1), (3, 2, -1)])
def test_un_ratio_de_refino_invalido_da_error(ratio):
    with pytest.raises(ErrorDatos, match="ratio"):
        crack_spread(80.0, 2.5, 2.8, ratio=ratio)


def test_una_serie_sin_observaciones_validas_da_error(indice):
    vacia = pd.Series([np.nan] * len(indice), index=indice)
    with pytest.raises(ErrorDatos):
        spark_spread(pd.Series(100.0, index=indice), vacia)


# --- alineacion de indices -----------------------------------------------------


def test_las_series_con_indices_distintos_se_alinean_por_fecha():
    electricidad = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=10))
    gas = pd.Series(30.0, index=pd.bdate_range("2024-01-08", periods=10))
    spread = spark_spread(electricidad, gas)
    comunes = electricidad.index.intersection(gas.index)
    assert list(spread.index) == list(comunes)
    assert len(spread) == 5
    assert spread.index[0] == pd.Timestamp("2024-01-08")
    assert np.allclose(spread.to_numpy(), 40.0)


def test_sin_fechas_comunes_da_error():
    electricidad = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=10))
    gas = pd.Series(30.0, index=pd.bdate_range("2025-01-01", periods=10))
    with pytest.raises(ErrorDatos, match="no comparten ninguna fecha"):
        spark_spread(electricidad, gas)


def test_los_valores_ausentes_se_descartan_de_las_dos_patas():
    indice = pd.bdate_range("2024-01-01", periods=5)
    electricidad = pd.Series([100.0, np.nan, 100.0, 100.0, 100.0], index=indice)
    gas = pd.Series([30.0, 30.0, np.nan, 30.0, 30.0], index=indice)
    spread = spark_spread(electricidad, gas)
    assert len(spread) == 3
    assert np.allclose(spread.to_numpy(), 40.0)


def test_un_escalar_se_difunde_sobre_la_serie(precios):
    """Un precio de CO2 fijo es un escenario habitual de analisis."""
    limpio = spark_spread_limpio(precios["pool"], precios["gas"], 100.0)
    sucio = spark_spread(precios["pool"], precios["gas"])
    assert len(limpio) == len(precios)
    assert np.allclose((sucio - limpio).to_numpy(), 100.0 * 0.20 / 0.50)


def test_una_secuencia_sin_indice_se_alinea_por_posicion():
    indice = pd.bdate_range("2024-01-01", periods=3)
    electricidad = pd.Series([100.0, 110.0, 120.0], index=indice)
    spread = spark_spread(electricidad, [30.0, 30.0, 30.0])
    assert list(spread.index) == list(indice)
    assert list(spread) == [40.0, 50.0, 60.0]


def test_una_secuencia_de_longitud_distinta_da_error():
    electricidad = pd.Series([100.0, 110.0, 120.0], index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ErrorDatos, match="pd.Series"):
        spark_spread(electricidad, [30.0, 30.0])


def test_dos_secuencias_sin_indice_deben_medir_lo_mismo():
    with pytest.raises(ErrorDatos, match="longitudes distintas"):
        spark_spread([100.0, 110.0], [30.0, 30.0, 30.0])


# --- lo que la segunda revision adversarial verifico como correcto -------------


def test_la_aritmetica_de_los_spreads_contra_una_reimplementacion_independiente():
    """Lo que el revisor comprobo y salio bien: se deja fijado como prueba.

    Error 0,000e+00 en 3 000 casos aleatorios, con la eficiencia dividiendo al
    combustible y el factor de emision por MWh termico, que es lo que promete
    el docstring. Se incluyen precios de electricidad negativos, que en el pool
    iberico son rutina.
    """
    generador = np.random.default_rng(11)
    peor_spark = peor_dark_limpio = peor_crack = 0.0

    for _ in range(3000):
        electricidad = generador.uniform(-50.0, 700.0)
        gas = generador.uniform(5.0, 250.0)
        eficiencia = generador.uniform(0.25, 0.65)
        costes = generador.uniform(0.0, 10.0)
        obtenido = float(
            spark_spread(pd.Series([electricidad]), pd.Series([gas]), eficiencia, costes).iloc[0]
        )
        peor_spark = max(peor_spark, abs(obtenido - (electricidad - gas / eficiencia - costes)))

        carbon = generador.uniform(5.0, 200.0)
        co2 = generador.uniform(0.0, 120.0)
        emision = generador.uniform(0.2, 0.45)
        obtenido = float(
            dark_spread_limpio(
                pd.Series([electricidad]), pd.Series([carbon]), pd.Series([co2]),
                eficiencia, emision, costes,
            ).iloc[0]
        )
        esperado = electricidad - carbon / eficiencia - co2 * emision / eficiencia - costes
        peor_dark_limpio = max(peor_dark_limpio, abs(obtenido - esperado))

        crudo = generador.uniform(20.0, 140.0)
        gasolina, destilado = generador.uniform(0.5, 4.0), generador.uniform(0.5, 4.0)
        obtenido = float(
            crack_spread(pd.Series([crudo]), pd.Series([gasolina]), pd.Series([destilado])).iloc[0]
        )
        esperado = (42.0 * (2.0 * gasolina + 1.0 * destilado) - 3.0 * crudo) / 3.0
        peor_crack = max(peor_crack, abs(obtenido - esperado))

    assert peor_spark == 0.0
    assert peor_dark_limpio < 1e-12
    assert peor_crack == 0.0


def test_la_opcionalidad_cumple_la_desigualdad_de_jensen_y_crece_con_la_volatilidad():
    """La afirmacion central del docstring de 'valor_opcionalidad'.

    Valorar la planta como si produjese siempre la infravalora, y la diferencia
    crece con la volatilidad. Una planta con spread medio NEGATIVO puede valer
    mucho: es el caso del ciclo abierto de punta.
    """
    generador = np.random.default_rng(3)
    pagos = []
    for volatilidad in (5.0, 20.0, 60.0):
        spread = pd.Series(generador.normal(-5.0, volatilidad, 8760))
        media_del_maximo = float(valor_opcionalidad(spread).mean())
        maximo_de_la_media = max(float(spread.mean()), 0.0)
        assert media_del_maximo >= maximo_de_la_media
        assert maximo_de_la_media == 0.0  # spread medio negativo
        assert media_del_maximo > 0.0  # y aun asi la planta vale
        pagos.append(media_del_maximo)

    assert pagos == sorted(pagos), "el valor de la opcionalidad crece con la volatilidad"
