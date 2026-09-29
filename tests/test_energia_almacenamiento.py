"""Pruebas del despacho optimo de almacenamiento.

La propiedad que mas importa aqui no es que el margen salga "alto", sino que
el optimizador resuelva **de verdad** el programa lineal: que respete el
estado de carga, que pierda exactamente el rendimiento de ciclo y que no se
pueda batir a mano. Por eso casi todas las pruebas usan casos con optimo
conocido.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.energia.almacenamiento import (
    ALMACENAMIENTOS,
    ParametrosAlmacenamiento,
    despacho_optimo,
    liquidar_plan,
    valor_de_la_prevision,
)

SIN_PERDIDAS = ParametrosAlmacenamiento(
    nombre="prueba 1 MW / 2 h sin perdidas",
    capacidad=2.0,
    potencia=1.0,
    rendimiento_ciclo=1.0,
    coste_degradacion=0.0,
)


def _horas(valores, inicio="2024-01-01"):
    return pd.Series(
        np.asarray(valores, dtype=float),
        index=pd.date_range(inicio, periods=len(valores), freq="h"),
    )


# --- el optimo, contra el calculo a mano ---------------------------------------


def test_el_despacho_coincide_con_el_optimo_calculado_a_mano():
    # Dos horas baratas a 10 y dos caras a 100, con 2 MWh de capacidad y 1 MW
    # de potencia: lo unico que se puede hacer es llenar en las dos primeras y
    # vaciar en las dos ultimas. Margen = 2 * (100 - 10) = 180.
    precios = _horas([10.0, 10.0, 100.0, 100.0])
    plan = despacho_optimo(precios, SIN_PERDIDAS)

    np.testing.assert_allclose(plan.carga.to_numpy(), [1.0, 1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(plan.descarga.to_numpy(), [0.0, 0.0, 1.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(plan.estado_carga.to_numpy(), [1.0, 2.0, 1.0, 0.0], atol=1e-9)
    assert plan.margen_total == pytest.approx(180.0)


def test_el_rendimiento_de_ciclo_se_pierde_exactamente():
    precios = _horas([10.0, 10.0, 100.0, 100.0])
    ficha = ParametrosAlmacenamiento("con perdidas", 2.0, 1.0, rendimiento_ciclo=0.85,
                                     coste_degradacion=0.0)
    plan = despacho_optimo(precios, ficha)

    descargado, cargado = plan.descarga.sum(), plan.carga.sum()
    assert descargado / cargado == pytest.approx(0.85)
    # Se compran 2 MWh a 10 y se venden 1,7 a 100.
    assert plan.margen_total == pytest.approx(2.0 * -10.0 + 1.7 * 100.0)


def test_el_lp_encuentra_dos_ciclos_donde_la_heuristica_ve_uno():
    # Perfil con DOS valles, que es lo que produce la fotovoltaica en el pool:
    # madrugada barata, punta de manana, mediodia solar barato, punta de
    # tarde. "Comprar en el minimo y vender en el maximo" hace un solo ciclo.
    precios = _horas([20.0, 10.0, 80.0, 90.0, 15.0, 12.0, 95.0, 100.0])
    plan = despacho_optimo(precios, SIN_PERDIDAS)

    assert plan.ciclos_equivalentes == pytest.approx(2.0)
    un_solo_ciclo = (precios.max() - precios.min()) * SIN_PERDIDAS.capacidad
    assert plan.margen_total > 1.5 * un_solo_ciclo  # medido: +71 %


def test_con_precios_negativos_el_activo_cobra_por_cargar():
    # En el pool iberico los precios negativos son rutina: 3.553 periodos de
    # 102.143. Para un activo de almacenamiento no son un problema, son una
    # oportunidad, porque le pagan por llenarse.
    precios = _horas([-20.0, -20.0, 50.0, 50.0])
    plan = despacho_optimo(precios, SIN_PERDIDAS)

    assert plan.carga.iloc[:2].sum() == pytest.approx(2.0)
    assert plan.margen_total == pytest.approx(2.0 * 20.0 + 2.0 * 50.0)


def test_sin_diferencia_de_precio_no_se_opera():
    precios = _horas([50.0] * 8)
    plan = despacho_optimo(precios, ALMACENAMIENTOS["bateria_4h"])
    assert plan.energia_movida == pytest.approx(0.0)
    assert plan.margen_total == pytest.approx(0.0)


def test_el_coste_de_degradacion_frena_los_ciclos_irrelevantes():
    # Diferencia de 4 EUR/MWh: con degradacion de 0,5 compensa, con 3 no.
    precios = _horas([48.0, 48.0, 52.0, 52.0])
    barato = ParametrosAlmacenamiento("barato", 2.0, 1.0, 1.0, coste_degradacion=0.5)
    caro = ParametrosAlmacenamiento("caro", 2.0, 1.0, 1.0, coste_degradacion=3.0)

    assert despacho_optimo(precios, barato).energia_movida > 0.0
    assert despacho_optimo(precios, caro).energia_movida == pytest.approx(0.0)


# --- las restricciones -----------------------------------------------------------


def test_el_estado_de_carga_nunca_se_sale_de_los_limites():
    generador = np.random.default_rng(7)
    precios = _horas(generador.normal(60.0, 40.0, 240))
    ficha = ALMACENAMIENTOS["bateria_4h"]
    plan = despacho_optimo(precios, ficha)

    assert plan.estado_carga.min() >= -1e-9
    assert plan.estado_carga.max() <= ficha.capacidad + 1e-9
    assert plan.carga.max() <= ficha.energia_por_periodo + 1e-9
    assert plan.descarga.max() <= ficha.energia_por_periodo + 1e-9


def test_cerrar_el_ciclo_evita_vender_energia_gratis():
    # Sin la restriccion de terminar como se empezo, un activo que arranca
    # lleno vende esa energia sin haberla pagado y el margen sale inflado.
    precios = _horas([10.0, 10.0, 100.0, 100.0])
    lleno = 2.0
    cerrado = despacho_optimo(precios, SIN_PERDIDAS, carga_inicial=lleno, cerrar_ciclo=True)
    abierto = despacho_optimo(precios, SIN_PERDIDAS, carga_inicial=lleno, cerrar_ciclo=False)

    assert cerrado.estado_carga.iloc[-1] == pytest.approx(lleno)
    assert abierto.margen_total > cerrado.margen_total
    assert abierto.estado_carga.iloc[-1] == pytest.approx(0.0)


def test_el_despacho_por_dia_no_arbitra_entre_dias():
    # Dia 1 barato entero, dia 2 caro entero. Por dia no hay nada que hacer,
    # porque dentro de cada dia el precio es plano; de una vez se compra el
    # dia 1 y se vende el dia 2. (Al reves no valdria: una bateria no puede
    # vender lo que todavia no ha comprado.)
    indice = pd.date_range("2024-01-01", periods=48, freq="h")
    precios = pd.Series(np.r_[np.full(24, 10.0), np.full(24, 100.0)], index=indice)

    por_dia = despacho_optimo(precios, SIN_PERDIDAS, por_dia=True)
    de_una_vez = despacho_optimo(precios, SIN_PERDIDAS, por_dia=False)

    assert por_dia.margen_total == pytest.approx(0.0)
    assert de_una_vez.margen_total > 0.0


def test_un_indice_desordenado_da_error():
    # El estado de carga es una suma acumulada: con las filas barajadas el
    # plan no es peor, es otro.
    precios = _horas([10.0, 90.0, 20.0, 80.0])
    with pytest.raises(ErrorDatos, match="no esta ordenado"):
        despacho_optimo(precios.iloc[[2, 0, 3, 1]], SIN_PERDIDAS)
    with pytest.raises(ErrorDatos, match="repetidos"):
        despacho_optimo(
            pd.Series([10.0, 90.0], index=[precios.index[0]] * 2), SIN_PERDIDAS
        )


@pytest.mark.parametrize(
    "campo, valor",
    [("capacidad", 0.0), ("potencia", -1.0), ("rendimiento_ciclo", 1.5),
     ("rendimiento_ciclo", 0.0), ("coste_degradacion", -1.0), ("horas_por_periodo", 0.0)],
)
def test_una_ficha_imposible_se_rechaza(campo, valor):
    parametros = {"nombre": "x", "capacidad": 4.0, "potencia": 1.0}
    parametros[campo] = valor
    with pytest.raises(ErrorDatos):
        ParametrosAlmacenamiento(**parametros)


def test_un_activo_desconocido_da_error():
    with pytest.raises(ErrorDatos, match="desconocido"):
        despacho_optimo(_horas([10.0, 90.0]), "volante_de_inercia")


# --- lo que hace honesto el ejercicio --------------------------------------------


def test_liquidar_a_precios_reales_castiga_una_prevision_mala():
    """El plan se decide con la prevision y se cobra a los precios reales.

    Esta separacion es lo unico que impide que la evaluacion mire al futuro,
    y es el equivalente de lo que 'prediccion_ingenua' hace en la evaluacion
    de modelos.
    """
    # La realidad es cara primero y barata despues, asi que no hay nada que
    # hacer. La prevision dice lo contrario y manda llenar en las dos primeras
    # horas para vender en las dos ultimas.
    reales = _horas([100.0, 100.0, 10.0, 10.0])
    al_reves = _horas([10.0, 10.0, 100.0, 100.0])

    plan_malo = despacho_optimo(al_reves, SIN_PERDIDAS)
    assert plan_malo.margen_total == pytest.approx(180.0)  # creia que ganaba 180

    liquidado = liquidar_plan(plan_malo, reales)
    # A precios reales eso es comprar a 100 y vender a 10.
    assert liquidado.margen_total == pytest.approx(-180.0)

    # Y con la realidad de frente no se opera: no se puede vender antes de
    # comprar, asi que el oraculo de esa serie es cero.
    assert despacho_optimo(reales, SIN_PERDIDAS).margen_total == pytest.approx(0.0)


def test_liquidar_exige_el_mismo_indice():
    plan = despacho_optimo(_horas([10.0, 10.0, 100.0, 100.0]), SIN_PERDIDAS)
    with pytest.raises(ErrorDatos, match="mismo indice"):
        liquidar_plan(plan, _horas([10.0, 100.0]))


def test_ninguna_prevision_supera_al_oraculo():
    """La propiedad que hace util esta metrica: esta acotada por arriba.

    Un MASE se puede hacer bajar eligiendo mal la referencia -- paso en este
    mismo proyecto. Un porcentaje del oraculo no se puede pasar de 100 sin que
    el fallo sea evidente.
    """
    generador = np.random.default_rng(11)
    indice = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    forma = 30.0 * np.sin(2.0 * np.pi * (indice.hour.to_numpy() - 4) / 24.0)
    precios = pd.Series(70.0 + forma + generador.normal(0.0, 12.0, len(indice)), index=indice)

    tabla = valor_de_la_prevision(
        precios,
        {
            "perfecta": precios,
            "ruido puro": pd.Series(generador.normal(70.0, 30.0, len(indice)), index=indice),
        },
        "bateria_4h",
    )

    assert tabla.loc["oraculo", "pct_oraculo"] == pytest.approx(100.0)
    assert tabla.loc["perfecta", "pct_oraculo"] == pytest.approx(100.0)
    assert (tabla["pct_oraculo"] <= 100.0 + 1e-6).all()
    # Una prevision aleatoria destruye valor; la ingenua, no tanto.
    assert tabla.loc["ruido puro", "pct_oraculo"] < tabla.loc["ingenua", "pct_oraculo"]


def test_la_referencia_ingenua_repite_el_perfil_del_dia_anterior():
    indice = pd.date_range("2024-01-01", periods=48, freq="h")
    generador = np.random.default_rng(3)
    precios = pd.Series(generador.normal(60.0, 25.0, 48), index=indice)

    tabla = valor_de_la_prevision(precios, {}, "bateria_4h")
    assert "ingenua" in tabla.index
    # Acotada por arriba y NO por abajo: con precios sin estructura, repetir
    # el perfil de ayer pierde dinero, y aqui sale -1,66 %. Que pueda ser
    # negativa es correcto y es la mitad informativa de la metrica.
    assert tabla.loc["ingenua", "pct_oraculo"] <= 100.0
    assert tabla.loc["ingenua", "pct_oraculo"] < 20.0  # sin patron horario no hay nada que repetir


def test_el_resumen_cuenta_bien_los_ciclos():
    precios = _horas([10.0, 10.0, 100.0, 100.0])
    resumen = despacho_optimo(precios, SIN_PERDIDAS).resumen()
    assert resumen["ciclos_equivalentes"] == pytest.approx(1.0)
    assert resumen["energia_movida"] == pytest.approx(4.0)  # 2 cargados + 2 descargados
    assert resumen["margen_por_ciclo"] == pytest.approx(180.0)


# --- valoracion: VAN, capex y degradacion ----------------------------------------


def _perfil_diario(dias=120, amplitud=40.0, nivel=70.0):
    """Precios con un ciclo diario limpio: el VAN sale determinista.

    120 dias y no mas: el despacho resuelve un programa lineal por dia y la
    curva de capacidad lo repite por cada factor, asi que alargar la muestra
    multiplica el coste sin cambiar ninguna de las propiedades que se
    comprueban. Con 400 dias este modulo doblaba el tiempo de toda la suite.
    """
    indice = pd.date_range("2020-01-01", periods=24 * dias, freq="h")
    forma = amplitud * np.sin(2.0 * np.pi * (indice.hour.to_numpy() - 4) / 24.0)
    return pd.Series(nivel + forma, index=indice)


SIN_DEGRADAR = ParametrosAlmacenamiento(
    nombre="prueba 1 MW / 4 h", capacidad=4.0, potencia=1.0,
    rendimiento_ciclo=1.0, coste_degradacion=0.0,
)


def test_el_van_coincide_con_el_flujo_descontado_a_mano():
    from icm.energia.almacenamiento import ParametrosInversion, van_almacenamiento

    precios = _perfil_diario()
    inversion = ParametrosInversion(
        capex_por_kwh=100.0, coste_fijo_anual_por_kw=0.0, vida_anios=3,
        tasa_descuento=0.10, retencion_anual=1.0,
    )
    resultado = van_almacenamiento(precios, SIN_DEGRADAR, inversion)

    capex = 4.0 * 1000.0 * 100.0
    margen = resultado["margen_bruto_anual"]
    esperado = -capex + sum(margen / 1.10**k for k in (1, 2, 3))
    assert resultado["van"] == pytest.approx(esperado)
    assert resultado["capex"] == pytest.approx(capex)


def test_la_tir_anula_el_van():
    from icm.energia.almacenamiento import _tir

    flujos = np.array([-1000.0, 400.0, 400.0, 400.0])
    tir = _tir(flujos)
    van_a_la_tir = float(np.sum(flujos / (1.0 + tir) ** np.arange(len(flujos))))
    assert van_a_la_tir == pytest.approx(0.0, abs=1e-9)
    # Un activo que no recupera el capex no tiene TIR, y se dice con NaN.
    assert np.isnan(_tir(np.array([-1000.0, 100.0, 100.0])))
    assert np.isnan(_tir(np.array([1000.0, 100.0])))  # sin inversion inicial


def test_el_periodo_de_retorno_interpola_dentro_del_anio():
    from icm.energia.almacenamiento import _periodo_de_retorno

    assert _periodo_de_retorno(np.array([-1000.0, 400.0, 400.0, 400.0])) == pytest.approx(2.5)
    assert _periodo_de_retorno(np.array([-1000.0, 1000.0])) == pytest.approx(1.0)
    # Si no se recupera, NaN y no la vida entera.
    assert np.isnan(_periodo_de_retorno(np.array([-1000.0, 100.0, 100.0])))


def test_el_margen_no_es_proporcional_a_la_capacidad():
    """Por eso la degradacion no se puede aplicar como un factor sobre el margen.

    Las horas que se aniaden al ampliar la capacidad son cada vez menos
    extremas, asi que perder el 20 % de capacidad cuesta MENOS del 20 % del
    ingreso. Un VAN que lo trate como proporcional castiga de mas.
    """
    from icm.energia.almacenamiento import curva_margen_por_capacidad

    curva = curva_margen_por_capacidad(_perfil_diario(), SIN_DEGRADAR, factores=(0.5, 0.75))

    assert curva.is_monotonic_increasing  # mas capacidad, mas margen
    relativo = curva / curva.loc[1.0]
    assert relativo.loc[0.5] > 0.5, "al 50 % de capacidad se gana MAS del 50 %"
    assert relativo.loc[0.75] > 0.75


def test_la_degradacion_no_se_cuenta_dos_veces():
    """El punto delicado de toda la valoracion.

    El coste por MWh cumple dos papeles: senial de despacho y amortizacion de
    la vida util. En un VAN que ya descuenta el capex, restarlo tambien seria
    pagar la bateria dos veces. El flujo usa el margen BRUTO y la degradacion
    entra como perdida de capacidad.
    """
    from icm.energia.almacenamiento import ParametrosInversion, van_almacenamiento

    precios = _perfil_diario()
    con_coste = ParametrosAlmacenamiento("con coste", 4.0, 1.0, 1.0, coste_degradacion=5.0)
    inversion = ParametrosInversion(capex_por_kwh=100.0, vida_anios=5, retencion_anual=1.0)
    resultado = van_almacenamiento(precios, con_coste, inversion)

    # El bruto es mayor que el neto: el coste existe y se informa.
    assert resultado["margen_bruto_anual"] > resultado["margen_neto_anual"]
    # Y el flujo del primer anio se construye con el BRUTO, no con el neto.
    esperado = (
        resultado["margen_bruto_anual"]
        + resultado["ingreso_adicional_anual"]
        - resultado["coste_fijo_anual"]
    )
    assert resultado["flujo_anio_1"] == pytest.approx(esperado)


def test_la_retencion_hace_decrecer_los_flujos():
    from icm.energia.almacenamiento import ParametrosInversion, van_almacenamiento

    precios = _perfil_diario()
    sin_fade = ParametrosInversion(capex_por_kwh=100.0, vida_anios=10, retencion_anual=1.0,
                                   coste_fijo_anual_por_kw=0.0)
    con_fade = ParametrosInversion(capex_por_kwh=100.0, vida_anios=10, retencion_anual=0.95,
                                   coste_fijo_anual_por_kw=0.0)

    a = van_almacenamiento(precios, SIN_DEGRADAR, sin_fade)
    b = van_almacenamiento(precios, SIN_DEGRADAR, con_fade)

    assert b["van"] < a["van"]
    assert b["flujo_ultimo_anio"] < b["flujo_anio_1"]
    assert a["flujo_ultimo_anio"] == pytest.approx(a["flujo_anio_1"])


def test_el_capex_de_equilibrio_deja_el_van_en_cero():
    from icm.energia.almacenamiento import ParametrosInversion, van_almacenamiento

    precios = _perfil_diario()
    inversion = ParametrosInversion(capex_por_kwh=300.0, vida_anios=10)
    resultado = van_almacenamiento(precios, SIN_DEGRADAR, inversion)

    equilibrio = resultado["capex_equilibrio_por_kwh"]
    comprobacion = van_almacenamiento(
        precios,
        SIN_DEGRADAR,
        ParametrosInversion(
            capex_por_kwh=equilibrio,
            vida_anios=inversion.vida_anios,
            tasa_descuento=inversion.tasa_descuento,
            retencion_anual=inversion.retencion_anual,
            coste_fijo_anual_por_kw=inversion.coste_fijo_anual_por_kw,
        ),
    )
    assert comprobacion["van"] == pytest.approx(0.0, abs=1.0)


def test_los_ingresos_adicionales_entran_por_potencia_no_por_capacidad():
    """Capacidad y servicios de ajuste se remuneran por MW, no por MWh."""
    from icm.energia.almacenamiento import ParametrosInversion, van_almacenamiento

    precios = _perfil_diario()
    inversion = ParametrosInversion(capex_por_kwh=100.0, ingreso_adicional_anual_por_mw=30_000.0,
                                    vida_anios=5, retencion_anual=1.0)
    resultado = van_almacenamiento(precios, SIN_DEGRADAR, inversion)
    assert resultado["ingreso_adicional_anual"] == pytest.approx(30_000.0)  # 1 MW


@pytest.mark.parametrize(
    "campo, valor",
    [("capex_por_kwh", -1.0), ("vida_anios", 0), ("tasa_descuento", 8.0),
     ("retencion_anual", 1.5), ("retencion_anual", 0.0)],
)
def test_una_ficha_de_inversion_imposible_se_rechaza(campo, valor):
    from icm.energia.almacenamiento import ParametrosInversion

    with pytest.raises(ErrorDatos):
        ParametrosInversion(**{campo: valor})


def test_anualizar_con_menos_de_un_mes_da_error():
    from icm.energia.almacenamiento import ParametrosInversion, van_almacenamiento

    corta = _perfil_diario(dias=10)
    with pytest.raises(ErrorDatos, match="dias"):
        van_almacenamiento(corta, SIN_DEGRADAR, ParametrosInversion())


def test_el_valor_van_de_la_prevision_ordena_oraculo_ingenua():
    """La tabla final: cuanto vale la prevision en euros de VAN."""
    from icm.energia.almacenamiento import ParametrosInversion, valor_van_de_la_prevision

    generador = np.random.default_rng(5)
    precios = _perfil_diario(dias=120)
    precios = precios + generador.normal(0.0, 10.0, len(precios))

    tabla = valor_van_de_la_prevision(
        precios, {}, SIN_DEGRADAR,
        ParametrosInversion(capex_por_kwh=120.0, vida_anios=10),
    )
    assert tabla.loc["oraculo", "pct_oraculo"] == pytest.approx(100.0)
    assert tabla.loc["ingenua", "pct_oraculo"] < 100.0
    assert tabla.loc["ingenua", "van"] < tabla.loc["oraculo", "van"]
    # La diferencia de VAN es el valor maximo que puede tener un modelo.
    assert tabla.loc["oraculo", "van"] - tabla.loc["ingenua", "van"] > 0.0


# --- La cadencia del indice frente a 'horas_por_periodo' -----------------


def test_una_serie_cuartohoraria_con_ficha_horaria_avisa():
    """El error que 'omie_intradiario.csv' provoca con los parametros de serie.

    El fichero real del proyecto es horario hasta el 2025-09-30 y
    cuarto-horario desde el 2025-10-01. Despacharlo con el valor por omision
    de una hora multiplica por cuatro la energia del ultimo tercio, y el plan
    resultante sigue siendo factible, asi que nada lo delata.
    """
    indice = pd.date_range("2024-01-01", periods=96, freq="15min")
    precios = pd.Series(np.tile([10.0, 20.0, 80.0, 90.0], 24), index=indice)
    with pytest.warns(RuntimeWarning, match="avanza 0.25 h"):
        despacho_optimo(precios, "bateria_4h")


def test_una_serie_horaria_limpia_no_avisa():
    indice = pd.date_range("2024-01-01", periods=96, freq="h")
    precios = pd.Series(np.tile([10.0, 20.0, 80.0, 90.0], 24), index=indice)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        despacho_optimo(precios, "bateria_4h")


def test_el_cambio_de_hora_no_cuenta_como_cadencia_mixta():
    """Un salto raro al anio es el horario de verano, no un fichero mixto.

    El caso se construye en hora local sin huso, que es como llegan los
    ficheros de OMIE: en el cambio de marzo falta la hora de las dos, asi que
    el indice tiene un salto de 2 h de verdad. Si el aviso saltara con esto
    seria ruido en toda serie horaria europea, y un aviso que siempre salta no
    se lee.
    """
    horas = pd.date_range("2024-03-30", periods=120, freq="h")
    indice = horas[(horas.month != 3) | (horas.day != 31) | (horas.hour != 2)]
    precios = pd.Series(np.arange(len(indice), dtype=float) % 4 * 30.0, index=indice)
    saltos = indice.to_series().diff().dropna().unique()
    assert len(saltos) == 2, "el caso tiene que tener un salto de 2 h de verdad"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        despacho_optimo(precios, "bateria_4h")


def test_una_ficha_cuartohoraria_con_serie_cuartohoraria_no_avisa():
    indice = pd.date_range("2024-01-01", periods=96, freq="15min")
    precios = pd.Series(np.tile([10.0, 20.0, 80.0, 90.0], 24), index=indice)
    ficha = ParametrosAlmacenamiento(
        nombre="cuarto-horaria", capacidad=4.0, potencia=1.0, horas_por_periodo=0.25
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        despacho_optimo(precios, ficha)
