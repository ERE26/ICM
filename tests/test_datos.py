"""Pruebas de la capa de datos: proveedores, cache y transformaciones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos, ErrorDescarga
from icm.datos import (
    CacheParquet,
    ProveedorConCache,
    ProveedorSimulado,
    cargar_precios,
    matriz_covarianzas,
    rendimientos,
    rendimientos_cartera,
    simular_precios,
)
from icm.datos.proveedor import ProveedorDatos, normalizar_tickers, validar_marco_precios


class ProveedorQueFalla(ProveedorDatos):
    """Proveedor de prueba que siempre falla, para ejercitar el respaldo."""

    nombre = "roto"

    def __init__(self) -> None:
        self.llamadas = 0

    def descargar(self, tickers, inicio=None, fin=None, intervalo="1d"):
        self.llamadas += 1
        raise ErrorDescarga("sin conexion")


class ProveedorContador(ProveedorDatos):
    """Cuenta cuantas veces se le pide una descarga de verdad."""

    nombre = "contador"

    def __init__(self) -> None:
        self.llamadas = 0

    def descargar(self, tickers, inicio=None, fin=None, intervalo="1d"):
        self.llamadas += 1
        return simular_precios(tickers, n_periodos=100, semilla=1)


# --- normalizacion de entradas -------------------------------------------------


def test_normalizar_tickers_acepta_cadena_suelta():
    assert normalizar_tickers("aapl") == ["AAPL"]


def test_normalizar_tickers_elimina_duplicados_conservando_el_orden():
    assert normalizar_tickers(["msft", "AAPL", "MSFT"]) == ["MSFT", "AAPL"]


@pytest.mark.parametrize("entrada", [[], [""], ["  "]])
def test_normalizar_tickers_rechaza_entradas_vacias(entrada):
    with pytest.raises(ErrorDatos):
        normalizar_tickers(entrada)


def test_validar_marco_admite_precios_no_positivos():
    # Un precio negativo no es un error del fichero: es lo que hace un mercado
    # electrico cuando sobra renovable. Rechazarlo aqui dejaba fuera 390 dias
    # del pool espaniol y el dia mas interesante de la serie del WTI.
    marco = pd.DataFrame({"A": [1.0, -2.0, 0.0]}, index=pd.date_range("2024-01-01", periods=3))
    validado = validar_marco_precios(marco, ["A"])
    np.testing.assert_allclose(validado["A"].to_numpy(), [1.0, -2.0, 0.0])


def test_validar_marco_rechaza_una_columna_entera_sin_datos():
    # Es lo que devuelve Yahoo para un ticker que no existe, y antes pasaba:
    # la unica comprobacion de valores era '<= 0', y NaN <= 0 es False.
    marco = pd.DataFrame(
        {"A": [1.0, 2.0], "ZZZZ": [np.nan, np.nan]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    with pytest.raises(ErrorDatos, match="sin un solo dato"):
        validar_marco_precios(marco, ["A", "ZZZZ"])


def test_validar_marco_rechaza_fechas_repetidas():
    marco = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-01"]))
    with pytest.raises(ErrorDatos, match="fechas repetidas"):
        validar_marco_precios(marco, ["A"])


def test_validar_marco_no_lee_un_indice_entero_como_nanosegundos():
    # pd.to_datetime lee un entero como nanosegundos desde 1970, asi que un
    # indice AAAAMMDD se convertia en 1970 sin protestar. Es el mismo fallo
    # que ya se habia corregido en convertir_fechas y aqui seguia vivo.
    marco = pd.DataFrame({"A": [100.0, 101.0]}, index=[20240102, 20240103])
    validado = validar_marco_precios(marco, ["A"])
    assert validado.index[0] == pd.Timestamp("2024-01-02")
    assert validado.index[-1] == pd.Timestamp("2024-01-03")


def test_validar_marco_detecta_tickers_ausentes():
    marco = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ErrorDatos, match="no ha devuelto datos"):
        validar_marco_precios(marco, ["A", "B"])


# --- simulador -----------------------------------------------------------------


def test_simulador_es_reproducible():
    primera = simular_precios(["A", "B"], n_periodos=200, semilla=5)
    segunda = simular_precios(["A", "B"], n_periodos=200, semilla=5)
    pd.testing.assert_frame_equal(primera, segunda)


def test_simulador_cambia_con_la_semilla():
    primera = simular_precios(["A"], n_periodos=200, semilla=5)
    segunda = simular_precios(["A"], n_periodos=200, semilla=6)
    assert not np.allclose(primera.to_numpy(), segunda.to_numpy())


def test_simulador_respeta_la_volatilidad_objetivo():
    precios = simular_precios(
        ["A"],
        n_periodos=50_000,
        volatilidad_anual=0.25,
        grados_libertad=100.0,
        garch_alfa=0.0,
        garch_beta=0.0,
        semilla=3,
    )
    observada = rendimientos(precios, "log")["A"].std(ddof=1) * np.sqrt(252)
    assert observada == pytest.approx(0.25, rel=0.03)


def test_simulador_respeta_la_correlacion_objetivo():
    precios = simular_precios(
        ["A", "B"], n_periodos=30_000, correlacion=0.6, garch_alfa=0.0, garch_beta=0.0, semilla=3
    )
    observada = rendimientos(precios, "log").corr().iloc[0, 1]
    assert observada == pytest.approx(0.6, abs=0.03)


def test_el_agrupamiento_de_volatilidad_genera_autocorrelacion_en_los_modulos():
    con = rendimientos(simular_precios(["A"], n_periodos=4000, semilla=2), "log")["A"]
    sin = rendimientos(
        simular_precios(["A"], n_periodos=4000, garch_alfa=0.0, garch_beta=0.0, semilla=2), "log"
    )["A"]
    assert con.abs().autocorr(1) > sin.abs().autocorr(1) + 0.03


def test_simulador_rechaza_parametros_garch_no_estacionarios():
    with pytest.raises(ErrorDatos, match="menor que 1"):
        simular_precios(["A"], n_periodos=100, garch_alfa=0.3, garch_beta=0.8)


def test_el_silenciador_de_yfinance_restaura_los_niveles():
    import logging

    from icm.datos.yahoo import _yfinance_callado

    registro = logging.getLogger("yfinance")
    registro.setLevel(logging.INFO)

    with _yfinance_callado(True):
        assert registro.level == logging.CRITICAL
    assert registro.level == logging.INFO

    with _yfinance_callado(False):
        assert registro.level == logging.INFO


def test_proveedor_simulado_respeta_la_ventana_pedida():
    precios = ProveedorSimulado(semilla=1).descargar(["A"], "2020-01-01", "2020-06-30")
    assert precios.index.min() >= pd.Timestamp("2020-01-01")
    assert precios.index.max() <= pd.Timestamp("2020-06-30")


# --- cache ---------------------------------------------------------------------


def test_el_cache_evita_la_segunda_descarga(tmp_path):
    contador = ProveedorContador()
    con_cache = ProveedorConCache(contador, CacheParquet(tmp_path))

    primera = con_cache.descargar(["A", "B"], "2020-01-01", "2020-12-31")
    segunda = con_cache.descargar(["A", "B"], "2020-01-01", "2020-12-31")

    assert contador.llamadas == 1
    pd.testing.assert_frame_equal(primera, segunda)


def test_el_cache_distingue_consultas_distintas(tmp_path):
    contador = ProveedorContador()
    con_cache = ProveedorConCache(contador, CacheParquet(tmp_path))
    con_cache.descargar(["A"], "2020-01-01", "2020-12-31")
    con_cache.descargar(["A"], "2021-01-01", "2021-12-31")
    assert contador.llamadas == 2


def test_el_cache_caduca(tmp_path):
    contador = ProveedorContador()
    con_cache = ProveedorConCache(contador, CacheParquet(tmp_path), caducidad_horas=0.0)
    con_cache.descargar(["A"])
    con_cache.descargar(["A"])
    assert contador.llamadas == 2


def test_vaciar_el_cache_borra_los_ficheros(tmp_path):
    cache = CacheParquet(tmp_path)
    con_cache = ProveedorConCache(ProveedorContador(), cache)
    con_cache.descargar(["A"])
    assert cache.vaciar() == 1
    assert cache.vaciar() == 0


# --- carga con respaldo --------------------------------------------------------


def test_cargar_precios_propaga_el_error_si_no_hay_respaldo():
    with pytest.raises(ErrorDescarga):
        cargar_precios(["A"], proveedor=ProveedorQueFalla(), respaldo_simulado=False)


def test_cargar_precios_avisa_al_usar_el_respaldo():
    with pytest.warns(UserWarning, match="DATOS SIMULADOS"):
        precios = cargar_precios(
            ["A", "B"], "2020-01-01", "2020-12-31",
            proveedor=ProveedorQueFalla(), respaldo_simulado=True,
        )
    assert list(precios.columns) == ["A", "B"]
    assert not precios.empty


# --- rendimientos --------------------------------------------------------------


def test_rendimiento_logaritmico_y_simple_coinciden_en_primer_orden(precios):
    log = rendimientos(precios, "log")
    simple = rendimientos(precios, "simple")
    np.testing.assert_allclose(np.expm1(log.to_numpy()), simple.to_numpy(), atol=1e-12)


def test_los_rendimientos_marcan_su_convencion(precios):
    assert rendimientos(precios, "log").attrs["tipo_rendimiento"] == "log"
    assert rendimientos(precios, "simple").attrs["tipo_rendimiento"] == "simple"


def test_rendimientos_rechaza_un_tipo_desconocido(precios):
    with pytest.raises(ErrorDatos, match="desconocido"):
        rendimientos(precios, "geometrico")


def test_la_cartera_reequilibrada_es_la_media_ponderada_en_convencion_simple(
    rend_simples, pesos_iguales
):
    esperado = rend_simples.mul(pesos_iguales, axis=1).sum(axis=1)
    obtenido = rendimientos_cartera(rend_simples, pesos_iguales)
    pd.testing.assert_series_equal(obtenido, esperado.rename("cartera"), check_names=False)


def test_comprar_y_mantener_difiere_de_reequilibrar(rend_simples, pesos_iguales):
    periodica = rendimientos_cartera(rend_simples, pesos_iguales, "periodico")
    estatica = rendimientos_cartera(rend_simples, pesos_iguales, "comprar_y_mantener")
    assert not np.allclose(periodica.to_numpy(), estatica.to_numpy())
    # Ambas parten del mismo primer periodo: los pesos aun no han derivado.
    assert periodica.iloc[0] == pytest.approx(estatica.iloc[0], abs=1e-12)


def test_una_cartera_con_un_solo_activo_reproduce_ese_activo(rend_simples):
    solo = rendimientos_cartera(rend_simples[["ALFA"]], {"ALFA": 1.0})
    np.testing.assert_allclose(solo.to_numpy(), rend_simples["ALFA"].to_numpy(), atol=1e-12)


def test_la_convencion_logaritmica_se_conserva(rend, pesos_iguales):
    cartera = rendimientos_cartera(rend, pesos_iguales)
    assert cartera.attrs["tipo_rendimiento"] == "log"
    # El resultado en logaritmos debe coincidir con convertir la version simple.
    simples = np.expm1(rend)
    equivalente = np.log1p(simples.mul(pesos_iguales, axis=1).sum(axis=1))
    np.testing.assert_allclose(cartera.to_numpy(), equivalente.to_numpy(), atol=1e-12)


def test_los_pesos_se_normalizan_solos(rend_simples):
    duplicados = rendimientos_cartera(rend_simples, pd.Series(2.0, index=rend_simples.columns))
    iguales = rendimientos_cartera(rend_simples, pd.Series(1.0, index=rend_simples.columns))
    np.testing.assert_allclose(duplicados.to_numpy(), iguales.to_numpy(), atol=1e-14)


def test_pesos_incompletos_dan_error(rend_simples):
    with pytest.raises(ErrorDatos, match="Faltan pesos"):
        rendimientos_cartera(rend_simples, {"ALFA": 1.0})


# --- covarianzas ---------------------------------------------------------------


def test_la_covarianza_muestral_coincide_con_pandas(rend):
    pd.testing.assert_frame_equal(matriz_covarianzas(rend), rend.cov())


def test_anualizar_escala_por_el_numero_de_periodos(rend):
    anual = matriz_covarianzas(rend, anualizar=True)
    np.testing.assert_allclose(anual.to_numpy(), rend.cov().to_numpy() * 252)


@pytest.mark.parametrize("metodo", ["muestral", "ewma", "ledoit_wolf"])
def test_todas_las_covarianzas_son_simetricas_y_semidefinidas_positivas(rend, metodo):
    cov = matriz_covarianzas(rend, metodo).to_numpy()
    np.testing.assert_allclose(cov, cov.T, atol=1e-15)
    assert np.linalg.eigvalsh(cov).min() > -1e-12


def test_el_encogimiento_reduce_las_correlaciones(rend):
    muestral = matriz_covarianzas(rend, "muestral").to_numpy()
    encogida = matriz_covarianzas(rend, "ledoit_wolf").to_numpy()
    fuera = ~np.eye(len(muestral), dtype=bool)
    assert np.all(np.abs(encogida[fuera]) <= np.abs(muestral[fuera]) + 1e-12)


def test_la_ewma_pondera_mas_lo_reciente():
    # Serie tranquila durante mucho tiempo y turbulenta al final: la EWMA debe
    # recoger el cambio de regimen, la covarianza muestral apenas lo nota.
    generador = np.random.default_rng(0)
    tranquila = generador.standard_normal((1000, 2)) * 0.005
    turbulenta = generador.standard_normal((50, 2)) * 0.05
    marco = pd.DataFrame(np.vstack([tranquila, turbulenta]), columns=["A", "B"])

    varianza_ewma = matriz_covarianzas(marco, "ewma", factor_decaimiento=0.94).iloc[0, 0]
    varianza_muestral = matriz_covarianzas(marco, "muestral").iloc[0, 0]
    assert varianza_ewma > 3 * varianza_muestral


def test_la_ewma_se_acerca_a_la_muestral_con_factor_alto():
    generador = np.random.default_rng(1)
    marco = pd.DataFrame(generador.standard_normal((2000, 2)) * 0.01, columns=["A", "B"])
    ewma = matriz_covarianzas(marco, "ewma", factor_decaimiento=0.999).to_numpy()
    muestral = matriz_covarianzas(marco, "muestral").to_numpy()
    np.testing.assert_allclose(ewma, muestral, rtol=0.15)


def test_metodo_de_covarianza_desconocido(rend):
    with pytest.raises(ErrorDatos, match="desconocido"):
        matriz_covarianzas(rend, "inventado")
