"""Pruebas del despacho optimo de almacenamiento.

La propiedad que mas importa aqui no es que el margen salga "alto", sino que
el optimizador resuelva **de verdad** el programa lineal: que respete el
estado de carga, que pierda exactamente el rendimiento de ciclo y que no se
pueda batir a mano. Por eso casi todas las pruebas usan casos con optimo
conocido.
"""

from __future__ import annotations

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
