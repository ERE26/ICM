"""Pruebas del simulador de energia.

El simulador solo sirve si tiene senal: se contrasta que los precios reaccionan
a las variables fisicas con las sensibilidades con las que se generaron, ademas
de los hechos estilizados del sector (estacionalidad, reversion, saltos al alza
y volatilidad agrupada).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from icm.comun import ErrorDatos
from icm.datos import cargar_precios
from icm.datos.proveedor import validar_marco_precios
from icm.energia.simulacion import (
    CATALOGO,
    EscenarioEnergia,
    ProveedorEnergiaSimulado,
    activos_con_yahoo,
    activos_sin_yahoo,
    grados_dia,
    simular_energia,
    simular_temperatura,
)

EXOGENAS = ["hdd", "cdd", "viento", "almacenamiento"]


def rendimientos_log(precios: pd.DataFrame) -> pd.DataFrame:
    return np.log(precios).diff().dropna()


def ciclo_anual(indice: pd.DatetimeIndex, pico: float = 15.0) -> np.ndarray:
    return np.cos(2.0 * np.pi * (indice.dayofyear.to_numpy(dtype=float) - pico) / 365.25)


def coeficientes_exogenas(escenario: EscenarioEnergia, activo: str) -> dict[str, float]:
    """MCO del rendimiento diario sobre las variaciones de las exogenas.

    Es la regresion que recupera las sensibilidades con las que se genero el
    escenario. Se estiman todas a la vez porque las exogenas estan
    correlacionadas entre si (el frio vacia los almacenamientos), y estimarlas
    de una en una dejaria variable omitida.
    """
    rendimientos = rendimientos_log(escenario.precios)[activo].to_numpy()
    variaciones = escenario.exogenas[EXOGENAS].diff().dropna().to_numpy()
    matriz = np.column_stack([np.ones(len(variaciones)), variaciones])
    beta, *_ = np.linalg.lstsq(matriz, rendimientos, rcond=None)
    return dict(zip(EXOGENAS, beta[1:]))


def razon_varianzas(precios: pd.Series, horizonte: int = 20) -> float:
    """Varianza del rendimiento a ``horizonte`` dias dividida por ``horizonte`` veces la diaria.

    Vale 1 en un paseo aleatorio, menos de 1 si la serie revierte a su media y
    mas de 1 si tiene tendencia. Es el contraste clasico de Lo y MacKinlay.
    """
    diarios = np.log(precios).diff().dropna()
    agregados = diarios.rolling(horizonte).sum().dropna()
    return float((agregados.var(ddof=1) / horizonte) / diarios.var(ddof=1))


@pytest.fixture(scope="module")
def escenario() -> EscenarioEnergia:
    """Escenario de referencia con todo el catalogo y seis anios de sesiones."""
    return simular_energia(n_periodos=1500, semilla=42)


# --- catalogo ------------------------------------------------------------------


@pytest.mark.parametrize("clave", list(CATALOGO))
def test_cada_ficha_del_catalogo_esta_completa(clave):
    ficha = CATALOGO[clave]
    assert set(ficha) == {"nombre", "tipo", "mercado", "yahoo", "unidad", "estacionalidad"}
    assert ficha["tipo"] in {"crudo", "gas", "electricidad", "accion"}
    assert ficha["nombre"] and ficha["mercado"] and ficha["unidad"]
    assert ficha["estacionalidad"] >= 0.0


def test_los_activos_se_reparten_segun_tengan_feed_en_yahoo():
    con, sin = activos_con_yahoo(), activos_sin_yahoo()
    assert set(con) | set(sin) == set(CATALOGO)
    assert not set(con) & set(sin)
    # Las referencias europeas no cotizan en Yahoo: van por fichero.
    assert {"TTF", "MIBGAS"} <= set(sin)
    assert {"WTI", "BRENT", "HENRY_HUB", "XLE", "XLU"} <= set(con)
    assert CATALOGO["WTI"]["yahoo"] == "CL=F"
    assert CATALOGO["HENRY_HUB"]["yahoo"] == "NG=F"


def test_el_gas_tiene_mucha_mas_estacionalidad_que_el_crudo():
    gas = CATALOGO["HENRY_HUB"]["estacionalidad"]
    crudo = CATALOGO["WTI"]["estacionalidad"]
    assert gas > 5 * crudo
    assert CATALOGO["XLE"]["estacionalidad"] == 0.0


# --- temperatura ---------------------------------------------------------------


def test_la_temperatura_tiene_estacionalidad_anual_detectable():
    temperatura = simular_temperatura(2000, semilla=11, media_anual=14.0, amplitud=9.0)
    fase = 2.0 * np.pi * temperatura.index.dayofyear.to_numpy(dtype=float) / 365.25
    matriz = np.column_stack([np.ones(len(fase)), np.cos(fase), np.sin(fase)])
    beta, *_ = np.linalg.lstsq(matriz, temperatura.to_numpy(), rcond=None)

    ajuste = matriz @ beta
    residuo = temperatura.to_numpy() - ajuste
    r2 = 1.0 - residuo.var() / temperatura.to_numpy().var()

    assert beta[0] == pytest.approx(14.0, abs=0.9)
    assert np.hypot(beta[1], beta[2]) == pytest.approx(9.0, rel=0.15)
    assert r2 > 0.6


def test_el_ciclo_pone_el_frio_en_invierno_y_el_calor_en_verano():
    temperatura = simular_temperatura(2000, semilla=4)
    por_mes = temperatura.groupby(temperatura.index.month).mean()
    mes = temperatura.index.month
    verano = temperatura[np.isin(mes, [6, 7, 8])].mean()
    invierno = temperatura[np.isin(mes, [12, 1, 2])].mean()

    assert por_mes.idxmin() in (12, 1, 2)
    assert por_mes.idxmax() in (6, 7, 8)
    # Dos veces la amplitud, algo menos porque un trimestre no es un solo dia.
    assert verano - invierno == pytest.approx(16.0, rel=0.20)


def test_la_desviacion_respecto_al_ciclo_conserva_la_persistencia_pedida():
    temperatura = simular_temperatura(3000, semilla=6, persistencia=0.85, ruido=2.5)
    ciclo = 14.0 - 9.0 * ciclo_anual(temperatura.index)
    desviacion = temperatura - ciclo
    assert desviacion.autocorr(1) == pytest.approx(0.85, abs=0.05)
    # "ruido" es la desviacion tipica incondicional, no la de la innovacion.
    assert desviacion.std(ddof=1) == pytest.approx(2.5, rel=0.15)


def test_la_temperatura_es_reproducible_y_cambia_con_la_semilla():
    pd.testing.assert_series_equal(
        simular_temperatura(300, semilla=5), simular_temperatura(300, semilla=5)
    )
    otra = simular_temperatura(300, semilla=6)
    assert not np.allclose(simular_temperatura(300, semilla=5).to_numpy(), otra.to_numpy())


@pytest.mark.parametrize("persistencia", [1.0, 1.2, -0.1])
def test_la_temperatura_rechaza_persistencias_no_estacionarias(persistencia):
    with pytest.raises(ErrorDatos, match="persistencia"):
        simular_temperatura(100, persistencia=persistencia)


def test_la_temperatura_necesita_al_menos_dos_periodos():
    with pytest.raises(ErrorDatos, match="al menos 2 periodos"):
        simular_temperatura(1)


# --- grados dia ----------------------------------------------------------------


def test_los_grados_dia_son_la_distancia_a_la_base():
    temperatura = pd.Series([2.0, 15.5, 18.0, 22.0, 28.0])
    grados = grados_dia(temperatura, base_calefaccion=15.5, base_refrigeracion=22.0)
    np.testing.assert_allclose(grados["hdd"], [13.5, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(grados["cdd"], [0.0, 0.0, 0.0, 0.0, 6.0])


def test_ningun_dia_cuenta_a_la_vez_como_frio_y_como_caluroso():
    grados = grados_dia(simular_temperatura(1000, semilla=8))
    assert (grados >= 0.0).all().all()
    assert (grados["hdd"] * grados["cdd"] == 0.0).all()


def test_hdd_se_apaga_en_verano_y_cdd_en_invierno():
    grados = grados_dia(simular_temperatura(2000, semilla=9))
    por_mes = grados.groupby(grados.index.month).mean()

    assert por_mes.loc[7, "hdd"] < 0.1 < por_mes.loc[7, "cdd"]
    assert por_mes.loc[1, "cdd"] == 0.0
    assert por_mes.loc[1, "hdd"] > 5.0


def test_subir_las_bases_aumenta_la_calefaccion_y_reduce_la_refrigeracion():
    temperatura = simular_temperatura(500, semilla=12)
    base = grados_dia(temperatura)
    alta = grados_dia(temperatura, base_calefaccion=18.0, base_refrigeracion=25.0)
    assert alta["hdd"].sum() > base["hdd"].sum()
    assert alta["cdd"].sum() < base["cdd"].sum()


def test_grados_dia_rechaza_bases_cruzadas():
    with pytest.raises(ErrorDatos, match="base de calefaccion"):
        grados_dia(pd.Series([10.0, 20.0]), base_calefaccion=24.0, base_refrigeracion=22.0)


# --- escenario: forma y reproducibilidad ---------------------------------------


def test_el_escenario_es_reproducible_con_la_misma_semilla():
    primero = simular_energia(["WTI", "HENRY_HUB"], n_periodos=300, semilla=3)
    segundo = simular_energia(["WTI", "HENRY_HUB"], n_periodos=300, semilla=3)
    pd.testing.assert_frame_equal(primero.precios, segundo.precios)
    pd.testing.assert_frame_equal(primero.exogenas, segundo.exogenas)
    assert primero.parametros == segundo.parametros


def test_el_escenario_cambia_con_la_semilla():
    primero = simular_energia(["HENRY_HUB"], n_periodos=300, semilla=3)
    segundo = simular_energia(["HENRY_HUB"], n_periodos=300, semilla=4)
    assert not np.allclose(primero.precios.to_numpy(), segundo.precios.to_numpy())
    assert not np.allclose(primero.exogenas.to_numpy(), segundo.exogenas.to_numpy())


def test_los_precios_son_positivos_y_pasan_la_validacion(escenario):
    assert (escenario.precios > 0.0).all().all()
    assert np.isfinite(escenario.precios.to_numpy()).all()
    validado = validar_marco_precios(escenario.precios, list(escenario.precios.columns))
    pd.testing.assert_frame_equal(validado, escenario.precios)


def test_precios_y_exogenas_comparten_indice_de_fechas_habiles(escenario):
    assert escenario.precios.index.equals(escenario.exogenas.index)
    assert escenario.precios.index.is_monotonic_increasing
    assert list(escenario.exogenas.columns) == ["temperatura", *EXOGENAS]
    assert (escenario.precios.index.dayofweek < 5).all()


def test_los_grados_dia_del_escenario_salen_de_su_temperatura(escenario):
    esperado = grados_dia(escenario.exogenas["temperatura"])
    pd.testing.assert_frame_equal(escenario.exogenas[["hdd", "cdd"]], esperado)


def test_el_escenario_exige_que_precios_y_exogenas_esten_alineados(escenario):
    with pytest.raises(ErrorDatos, match="compartir indice"):
        EscenarioEnergia(
            precios=escenario.precios,
            exogenas=escenario.exogenas.iloc[:-1],
            parametros=escenario.parametros,
        )


def test_los_parametros_recogen_la_verdad_de_cada_activo(escenario):
    claves = {
        "sensibilidad_hdd",
        "sensibilidad_cdd",
        "velocidad_reversion",
        "volatilidad_objetivo",
        "intensidad_saltos",
    }
    assert claves <= set(escenario.parametros)
    for clave in claves:
        assert set(escenario.parametros[clave]) == set(escenario.precios.columns)

    # La energia revierte y la accion no: es la diferencia de fondo entre ambas.
    assert escenario.parametros["velocidad_reversion"]["HENRY_HUB"] > 0.0
    assert escenario.parametros["velocidad_reversion"]["XLE"] == 0.0


@pytest.mark.parametrize(
    ("argumentos", "mensaje"),
    [
        ({"activos": ["WTI", "ORO"]}, "fuera del catalogo"),
        ({"n_periodos": 1}, "al menos 2 periodos"),
        ({"intensidad_saltos": -0.1}, "no puede ser negativa"),
    ],
)
def test_simular_energia_rechaza_entradas_invalidas(argumentos, mensaje):
    with pytest.raises(ErrorDatos, match=mensaje):
        simular_energia(**{"n_periodos": 100, **argumentos})


def test_los_activos_se_admiten_en_minuscula():
    escenario = simular_energia(["wti", "henry_hub"], n_periodos=100, semilla=1)
    assert list(escenario.precios.columns) == ["WTI", "HENRY_HUB"]


# --- escenario: la senal que el ML debe encontrar ------------------------------


def test_hay_senal_el_rendimiento_del_gas_sigue_al_cambio_en_hdd():
    """El contraste que justifica todo el modulo.

    Si el rendimiento del gas no reacciona a la variacion de los grados dia de
    calefaccion, el simulador es un paseo aleatorio disfrazado y ningun modelo
    podra batir a la referencia ingenua. El crudo sirve de control: es la misma
    exogena, pero apenas le afecta.
    """
    escenario = simular_energia(n_periodos=1500, semilla=42)
    rendimientos = rendimientos_log(escenario.precios)
    variacion_hdd = escenario.exogenas["hdd"].diff().dropna()

    correlacion_gas, p_gas = stats.pearsonr(rendimientos["HENRY_HUB"], variacion_hdd)
    correlacion_crudo, _ = stats.pearsonr(rendimientos["WTI"], variacion_hdd)

    assert correlacion_gas > 0.12
    assert p_gas < 1e-6
    assert abs(correlacion_crudo) < 0.10
    assert correlacion_gas > 2 * abs(correlacion_crudo)


def test_la_sensibilidad_estimada_recupera_la_verdad_del_simulador():
    escenario = simular_energia(n_periodos=2000, semilla=17)
    for activo in ("HENRY_HUB", "TTF", "OMIE"):
        estimada = coeficientes_exogenas(escenario, activo)["hdd"]
        verdadera = escenario.parametros["sensibilidad_hdd"][activo]
        assert estimada == pytest.approx(verdadera, rel=0.40)


def test_el_viento_abarata_la_electricidad_y_el_almacenamiento_el_gas():
    escenario = simular_energia(n_periodos=2000, semilla=23)
    electricidad = coeficientes_exogenas(escenario, "OMIE")
    gas = coeficientes_exogenas(escenario, "TTF")

    assert electricidad["viento"] == pytest.approx(
        escenario.parametros["sensibilidad_viento"]["OMIE"], rel=0.30
    )
    assert electricidad["viento"] < 0.0
    assert gas["almacenamiento"] < 0.0
    # El crudo no tiene nada que ver con el viento del norte de Europa.
    assert escenario.parametros["sensibilidad_viento"]["WTI"] == 0.0


def test_el_invierno_encarece_el_gas_y_no_mueve_a_la_accion():
    escenario = simular_energia(n_periodos=1500, semilla=31)
    log_precio = np.log(escenario.precios)
    mes = log_precio.index.month
    salto_estacional = (
        log_precio[np.isin(mes, [12, 1, 2])].mean() - log_precio[np.isin(mes, [6, 7, 8])].mean()
    )
    assert salto_estacional["HENRY_HUB"] > 0.25
    assert salto_estacional["TTF"] > 0.25
    assert abs(salto_estacional["XLE"]) < 0.20


# --- escenario: hechos estilizados ---------------------------------------------


def test_los_saltos_al_alza_producen_exceso_de_curtosis_y_asimetria_positiva():
    con = simular_energia(["HENRY_HUB"], n_periodos=3000, semilla=5, intensidad_saltos=0.08)
    sin = simular_energia(["HENRY_HUB"], n_periodos=3000, semilla=5, intensidad_saltos=0.0)
    con_saltos = rendimientos_log(con.precios)["HENRY_HUB"]
    sin_saltos = rendimientos_log(sin.precios)["HENRY_HUB"]

    assert stats.kurtosis(con_saltos, fisher=True, bias=False) > 3.0

    # Que las colas sean gruesas lo puede lograr tambien el GARCH; lo que solo
    # pueden hacer los saltos es que sean colas DERECHAS. Ambas series comparten
    # semilla y, por tanto, las mismas innovaciones gaussianas: la unica
    # diferencia entre ellas son los saltos.
    asimetria_con = stats.skew(con_saltos, bias=False)
    asimetria_sin = stats.skew(sin_saltos, bias=False)
    assert asimetria_con > asimetria_sin + 0.5
    assert abs(asimetria_sin) < 0.5


def test_el_escenario_por_defecto_ya_tiene_colas_gruesas(escenario):
    curtosis = stats.kurtosis(
        rendimientos_log(escenario.precios)["HENRY_HUB"], fisher=True, bias=False
    )
    assert curtosis > 0.5


@pytest.mark.parametrize("activo", ["WTI", "BRENT", "XLE"])
def test_la_volatilidad_se_agrupa(activo):
    escenario = simular_energia([activo], n_periodos=3000, semilla=13)
    rendimientos = rendimientos_log(escenario.precios)[activo]
    # El signo del rendimiento es impredecible, su tamanio no: esa asimetria
    # entre ambas autocorrelaciones es la firma del agrupamiento.
    assert rendimientos.abs().autocorr(1) > 0.02
    assert rendimientos.abs().autocorr(1) > abs(rendimientos.autocorr(1))


def test_la_energia_revierte_a_su_media_y_la_accion_no():
    escenario = simular_energia(["OMIE", "XLE"], n_periodos=2000, semilla=19)
    assert razon_varianzas(escenario.precios["OMIE"]) < 0.6
    assert razon_varianzas(escenario.precios["XLE"]) > 0.7
    # La reversion se ve tambien dia a dia: tras una subida viene una correccion.
    assert rendimientos_log(escenario.precios)["OMIE"].autocorr(1) < -0.01


def test_el_factor_comun_correlaciona_los_dos_crudos(escenario):
    rendimientos = rendimientos_log(escenario.precios)
    correlacion = rendimientos[["WTI", "BRENT"]].corr().iloc[0, 1]
    assert correlacion > 0.25
    # Pero cada mercado conserva lo suyo: no son el mismo activo.
    assert correlacion < 0.95


def test_la_volatilidad_ordena_los_activos_como_su_tipo(escenario):
    volatilidad = rendimientos_log(escenario.precios).std(ddof=1) * np.sqrt(252)
    assert volatilidad["OMIE"] > volatilidad["HENRY_HUB"] > volatilidad["WTI"] > volatilidad["XLU"]


def test_la_senal_no_depende_de_los_saltos():
    """Sin saltos la serie es mucho mas tranquila, y aun asi la senal sigue ahi.

    Importa porque un modelo que solo acertase los dias de pico no seria un
    modelo: seria un detector de valores atipicos.
    """
    escenario = simular_energia(n_periodos=1500, semilla=42, intensidad_saltos=0.0)
    rendimientos = rendimientos_log(escenario.precios)["HENRY_HUB"]
    variacion_hdd = escenario.exogenas["hdd"].diff().dropna()
    assert stats.pearsonr(rendimientos, variacion_hdd)[0] > 0.15


# --- proveedor -----------------------------------------------------------------


def test_el_proveedor_encaja_con_cargar_precios():
    proveedor = ProveedorEnergiaSimulado(semilla=2)
    precios = cargar_precios(["WTI", "TTF"], "2020-01-01", "2020-12-31", proveedor=proveedor)

    assert list(precios.columns) == ["WTI", "TTF"]
    assert precios.index.min() >= pd.Timestamp("2020-01-01")
    assert precios.index.max() <= pd.Timestamp("2020-12-31")
    assert len(precios) > 200
    assert (precios > 0.0).all().all()


def test_el_proveedor_guarda_el_escenario_completo():
    proveedor = ProveedorEnergiaSimulado(semilla=2, n_periodos=400)
    assert proveedor.ultimo_escenario is None

    precios = proveedor.descargar(["HENRY_HUB"])
    escenario = proveedor.ultimo_escenario

    assert isinstance(escenario, EscenarioEnergia)
    assert escenario.parametros["sensibilidad_hdd"]["HENRY_HUB"] > 0.0
    assert escenario.exogenas.index.equals(precios.index)


def test_el_proveedor_es_reproducible_y_respeta_su_nombre():
    primero = ProveedorEnergiaSimulado(semilla=7, n_periodos=200).descargar(["WTI"])
    segundo = ProveedorEnergiaSimulado(semilla=7, n_periodos=200).descargar(["WTI"])
    pd.testing.assert_frame_equal(primero, segundo)
    assert ProveedorEnergiaSimulado.nombre == "energia_simulada"


def test_el_proveedor_solo_genera_series_diarias():
    with pytest.raises(ErrorDatos, match="diarias"):
        ProveedorEnergiaSimulado().descargar(["WTI"], intervalo="1mo")


def test_el_proveedor_avisa_si_la_ventana_queda_vacia():
    proveedor = ProveedorEnergiaSimulado(semilla=1, n_periodos=200, inicio="2018-01-01")
    with pytest.raises(ErrorDatos, match="ventana"):
        proveedor.descargar(["WTI"], fin="2017-01-01")


def test_el_proveedor_rechaza_activos_fuera_del_catalogo():
    with pytest.raises(ErrorDatos, match="fuera del catalogo"):
        ProveedorEnergiaSimulado().descargar(["AAPL"])
