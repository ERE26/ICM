"""El modulo del que depende toda la libreria, que no tenia pruebas.

`comun.py` lo importan `cartera`, `riesgo`, `datos`, `energia` y `ml`. No
existia `tests/test_comun.py`, y de ahi que su peor defecto -- invertir el
signo de toda cartera con exposicion neta negativa -- sobreviviese a 944
pruebas en verde: ninguna miraba aqui.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import (
    DIAS_HABILES_ANIO,
    ErrorDatos,
    como_matriz,
    como_serie,
    normalizar_pesos,
    periodos_por_anio,
    validar_nivel,
)


# --------------------------------------------------------------------------
# normalizar_pesos
# --------------------------------------------------------------------------


def test_los_pesos_normales_se_normalizan():
    pesos = normalizar_pesos({"A": 2.0, "B": 2.0})
    np.testing.assert_allclose(pesos.to_numpy(), [0.5, 0.5])
    assert pesos.sum() == pytest.approx(1.0)


def test_una_cartera_con_cortos_pero_neta_larga_conserva_los_signos():
    # 130/30: larga neta, con una pierna corta. Es legitima y debe pasar.
    pesos = normalizar_pesos({"A": 1.3, "B": -0.3})
    assert pesos["A"] > 0 and pesos["B"] < 0
    assert pesos.sum() == pytest.approx(1.0)


def test_una_cartera_con_exposicion_neta_negativa_se_rechaza():
    """El peor hallazgo de la revision: dividir por una suma negativa.

    ``{A: +0,3, B: -1,3}`` salia como ``{A: -0,3, B: +1,3}``: la larga
    convertida en corta y al reves, sumando 1,0 y sin nada que lo delatase.
    """
    with pytest.raises(ErrorDatos, match="invertiria el signo"):
        normalizar_pesos({"A": 0.3, "B": -1.3})


def test_el_signo_invertido_falseaba_el_resultado_de_la_cartera():
    # La prueba de que no era un detalle formal: con los pesos invertidos la
    # serie de resultados sale con el signo cambiado en todas las fechas.
    from icm.datos.precios import rendimientos_cartera

    fechas = pd.bdate_range("2020-01-01", periods=5)
    r = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.00, 0.03], "B": [0.00, 0.01, 0.02, -0.02, 0.01]},
        index=fechas,
    )
    with pytest.raises(ErrorDatos, match="invertiria el signo"):
        rendimientos_cartera(r, pd.Series({"A": 0.3, "B": -1.3}))


def test_unos_pesos_que_suman_cero_se_rechazan():
    with pytest.raises(ErrorDatos, match="despreciable frente a la bruta"):
        normalizar_pesos({"A": 1.0, "B": -1.0})


@pytest.mark.parametrize("descuadre", [1e-7, 1e-9, 1e-12])
def test_una_cartera_neutral_casi_cuadrada_tampoco_se_normaliza(descuadre):
    """Lo que rompe la normalizacion no es el cero exacto.

    Con ``np.isclose(suma, 0.0)`` y su ``atol=1e-8``, una cartera neutral
    descuadrada por 1e-7 pasaba el filtro y salia con pesos del orden de 1e7.
    El criterio correcto es el cociente entre la exposicion neta y la bruta.
    """
    with pytest.raises(ErrorDatos, match="despreciable frente a la bruta"):
        normalizar_pesos({"A": 1.0, "B": -1.0 + descuadre})


def test_una_exposicion_neta_pequenia_pero_real_si_se_normaliza():
    # 5 % de exposicion neta sobre una bruta de 1,95 es una cartera perfectamente
    # normal: el filtro no debe confundir "pequenia" con "despreciable".
    pesos = normalizar_pesos({"A": 1.0, "B": -0.95})
    assert pesos.sum() == pytest.approx(1.0)
    assert pesos["A"] > 0 and pesos["B"] < 0


def test_un_peso_ausente_no_se_ignora():
    # 'sum' descarta los NaN, asi que el resultado sumaba 1,0 y parecia
    # normalizado aunque llevase un NaN dentro.
    with pytest.raises(ErrorDatos, match="ausentes o infinitos"):
        normalizar_pesos({"A": 1.0, "B": np.nan, "C": 1.0})


def test_un_peso_infinito_no_se_ignora():
    with pytest.raises(ErrorDatos, match="ausentes o infinitos"):
        normalizar_pesos({"A": 1.0, "B": np.inf})


def test_los_pesos_se_alinean_por_nombre_cuando_se_dan_los_activos():
    pesos = normalizar_pesos({"B": 3.0, "A": 1.0}, activos=["A", "B"])
    assert list(pesos.index) == ["A", "B"]
    np.testing.assert_allclose(pesos.to_numpy(), [0.25, 0.75])


def test_faltan_pesos_para_un_activo_pedido():
    with pytest.raises(ErrorDatos, match="Faltan pesos"):
        normalizar_pesos({"A": 1.0}, activos=["A", "B"])


# --------------------------------------------------------------------------
# como_serie y como_matriz
# --------------------------------------------------------------------------


def test_como_serie_descarta_los_ausentes():
    serie = como_serie(pd.Series([1.0, np.nan, 3.0]))
    np.testing.assert_allclose(serie.to_numpy(), [1.0, 3.0])


def test_como_serie_denuncia_los_infinitos_en_vez_de_colarlos():
    """Un infinito no es un hueco y ``dropna`` no lo ve.

    Con **un** ``+inf`` en 500 rendimientos, el VaR historico devolvia
    exactamente la misma cifra que con la serie limpia: el numero seguia
    teniendo buen aspecto y la serie estaba corrompida.
    """
    with pytest.raises(ErrorDatos, match="infinitos"):
        como_serie(pd.Series([0.01, np.inf, 0.02]))


def test_como_matriz_denuncia_los_infinitos():
    with pytest.raises(ErrorDatos, match="infinitos"):
        como_matriz(pd.DataFrame({"A": [0.01, -np.inf], "B": [0.0, 0.1]}))


def test_un_precio_cero_produce_un_infinito_que_ahora_se_ve():
    # El caso real: en el pool electrico hay horas casadas a cero, y la
    # variacion porcentual sobre un cero es infinita.
    precios = pd.Series([50.0, 40.0, 0.0, 35.0])
    with pytest.raises(ErrorDatos, match="infinitos"):
        como_serie(precios.pct_change().dropna())


def test_como_serie_rechaza_texto_con_error_de_la_libreria():
    # Antes escapaba el ValueError crudo de numpy, que ningun except de la
    # libreria recoge.
    with pytest.raises(ErrorDatos, match="no son numeros"):
        como_serie(["a", "b"])


def test_como_serie_rechaza_dos_dimensiones():
    with pytest.raises(ErrorDatos, match="una dimension"):
        como_serie([[1.0, 2.0], [3.0, 4.0]])


def test_como_serie_rechaza_un_marco_de_varias_columnas():
    with pytest.raises(ErrorDatos, match="unica columna"):
        como_serie(pd.DataFrame({"A": [1.0], "B": [2.0]}))


def test_como_serie_acepta_un_marco_de_una_columna():
    serie = como_serie(pd.DataFrame({"A": [1.0, 2.0]}))
    np.testing.assert_allclose(serie.to_numpy(), [1.0, 2.0])


def test_como_matriz_descarta_las_filas_incompletas():
    marco = como_matriz(pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [1.0, 2.0, 3.0]}))
    assert len(marco) == 2


def test_una_serie_vacia_tras_limpiar_se_denuncia():
    with pytest.raises(ErrorDatos, match="observaciones validas"):
        como_serie(pd.Series([np.nan, np.nan]))


# --------------------------------------------------------------------------
# validar_nivel y periodos_por_anio
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nivel", [0.90, 0.95, 0.99, 0.999])
def test_los_niveles_validos_pasan(nivel):
    assert validar_nivel(nivel) == pytest.approx(nivel)


@pytest.mark.parametrize("nivel", [0.0, 1.0, -0.1, 1.5, np.nan])
def test_los_niveles_fuera_del_intervalo_abierto_se_rechazan(nivel):
    with pytest.raises(ErrorDatos):
        validar_nivel(nivel)


@pytest.mark.parametrize(
    ("frecuencia", "esperado"),
    [("diaria", DIAS_HABILES_ANIO), ("mensual", 12), ("ANUAL", 1), (26, 26)],
)
def test_las_frecuencias_conocidas_se_traducen(frecuencia, esperado):
    assert periodos_por_anio(frecuencia) == esperado


def test_un_booleano_no_es_una_frecuencia():
    # bool es subclase de int, asi que True colaba como "un periodo al anio".
    with pytest.raises(ErrorDatos, match="booleano"):
        periodos_por_anio(True)


@pytest.mark.parametrize("frecuencia", [0, -12])
def test_una_frecuencia_no_positiva_se_rechaza(frecuencia):
    with pytest.raises(ErrorDatos, match="positivo"):
        periodos_por_anio(frecuencia)


def test_una_frecuencia_desconocida_se_rechaza():
    with pytest.raises(ErrorDatos, match="desconocida"):
        periodos_por_anio("quincenal")
