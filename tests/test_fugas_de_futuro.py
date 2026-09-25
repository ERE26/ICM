"""Las fugas de informacion futura, cada una con su reproduccion.

Una fuga de futuro no se parece a un fallo: los numeros salen **mejor**. Por
eso todas estas sobrevivieron a una suite en verde, y por eso sus pruebas
viven juntas y comparten una misma forma de contrastarlas: se modifica el
futuro de una serie y se comprueba que nada de lo ya publicado se mueve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.energia.estacionalidad import desestacionalizar, desestacionalizar_recursiva
from icm.energia.simulacion import prevision_exogena
from icm.riesgo.backtesting import excepciones, serie_var_movil
from icm.riesgo.volatilidad import volatilidad_ewma


# --------------------------------------------------------------------------
# 1. El arranque de la EWMA
# --------------------------------------------------------------------------


def test_la_ewma_no_se_revisa_al_anadir_futuro():
    """Alargar la serie no puede cambiar una estimacion ya publicada."""
    g = np.random.default_rng(3)
    completa = pd.Series(g.standard_normal(300) * 0.01)
    corta = completa.iloc[:150]
    pd.testing.assert_series_equal(
        volatilidad_ewma(completa).iloc[:150], volatilidad_ewma(corta)
    )


# --------------------------------------------------------------------------
# 2. La desestacionalizacion
# --------------------------------------------------------------------------


@pytest.fixture
def serie_estacional() -> pd.Series:
    g = np.random.default_rng(0)
    idx = pd.date_range("2005-01-01", periods=240, freq="ME")
    ciclo = 10.0 * np.cos(2.0 * np.pi * (idx.month - 1) / 12.0)
    return pd.Series(50.0 + ciclo + g.standard_normal(240) * 1.5, index=idx)


def test_la_desestacionalizacion_clasica_si_se_revisa(serie_estacional):
    """La version clasica mira al futuro, y la prueba lo deja por escrito.

    No es un fallo que se arregle aqui: la descomposicion clasica es
    inherentemente bilateral, y su sitio es la lectura ex post. Lo que era un
    defecto es que el docstring la recomendase para alimentar modelos.
    """
    antes = desestacionalizar(serie_estacional, periodo=12)
    tocada = serie_estacional.copy()
    tocada.iloc[-36:] = tocada.iloc[-36:] * 3.0
    despues = desestacionalizar(tocada, periodo=12)
    # El valor de enero de 2005 cambia por modificar 2022-2024.
    assert abs(antes.iloc[0] - despues.iloc[0]) > 1.0


def test_la_desestacionalizacion_recursiva_no_se_revisa(serie_estacional):
    recursiva = desestacionalizar_recursiva(serie_estacional, por="mes")
    tocada = serie_estacional.copy()
    tocada.iloc[-36:] = tocada.iloc[-36:] * 3.0
    despues = desestacionalizar_recursiva(tocada, por="mes")
    pd.testing.assert_series_equal(recursiva.iloc[:-36], despues.iloc[:-36])


def test_la_recursiva_no_depende_de_donde_acabe_la_muestra(serie_estacional):
    # El mismo analista en 2014 y en 2024 debe obtener la misma respuesta para
    # una fecha de 2010.
    hasta_2014 = desestacionalizar_recursiva(serie_estacional.iloc[:120], por="mes")
    hasta_2024 = desestacionalizar_recursiva(serie_estacional, por="mes")
    pd.testing.assert_series_equal(hasta_2014, hasta_2024.iloc[:120])


def test_la_recursiva_si_quita_la_estacionalidad(serie_estacional):
    # Que no mire al futuro no sirve de nada si no hace su trabajo.
    limpia = desestacionalizar_recursiva(serie_estacional, por="mes").dropna()
    perfil_antes = serie_estacional.groupby(serie_estacional.index.month).mean()
    perfil_despues = limpia.groupby(limpia.index.month).mean()
    rango_antes = perfil_antes.max() - perfil_antes.min()
    rango_despues = perfil_despues.max() - perfil_despues.min()
    assert rango_despues < 0.2 * rango_antes


def test_la_recursiva_deja_NaN_al_principio(serie_estacional):
    # Hasta que no hay repeticiones de una posicion del ciclo no hay nada que
    # promediar, y decir un numero seria inventarlo.
    recursiva = desestacionalizar_recursiva(serie_estacional, por="mes")
    assert recursiva.iloc[:12].isna().all()
    assert recursiva.iloc[-1:].notna().all()


# --------------------------------------------------------------------------
# 3. El indice desordenado del backtest
# --------------------------------------------------------------------------


def test_un_indice_desordenado_no_pasa_por_el_var_movil():
    """Las ventanas se toman por posicion, asi que el orden ES el calendario.

    Sobre 1.200 rendimientos permutados, 947 de las 950 ventanas contenian
    observaciones posteriores a la fecha que estimaban, con una media de 126,7
    futuras de 250, y el semaforo de Basilea cambiaba de verde a amarilla
    segun la permutacion.
    """
    g = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=400)
    serie = pd.Series(g.standard_normal(400) * 0.01, index=idx)
    revuelta = serie.sample(frac=1.0, random_state=1)
    with pytest.raises(ErrorDatos, match="no esta ordenado"):
        serie_var_movil(revuelta, 0.99, 250, "historico")
    # Ordenada funciona.
    assert len(serie_var_movil(serie, 0.99, 250, "historico")) > 0


def test_las_excepciones_salen_en_orden_cronologico():
    """Christoffersen cuenta transiciones, asi que el orden decide el veredicto.

    Con las mismas 12 excepciones agrupadas en una racha, el estadistico de
    independencia pasaba de 86,93 (rechaza) a 0,74 (no rechaza) solo porque el
    marco venia sin ordenar.
    """
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(0.005, index=idx)
    r.iloc[20:32] = -0.10  # una racha de doce perdidas seguidas
    var = pd.Series(0.03, index=idx)

    ordenadas = excepciones(r, var)
    revueltas = excepciones(r.sample(frac=1.0, random_state=2), var)
    pd.testing.assert_series_equal(ordenadas, revueltas, check_freq=False)
    assert revueltas.index.is_monotonic_increasing

    # Y el contraste que depende del orden da el mismo veredicto en los dos.
    from icm.riesgo.backtesting import contraste_christoffersen_independencia

    assert contraste_christoffersen_independencia(
        ordenadas
    ) == pytest.approx(contraste_christoffersen_independencia(revueltas))


def test_las_excepciones_rechazan_fechas_repetidas():
    idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
    with pytest.raises(ErrorDatos, match="fechas repetidas"):
        excepciones(pd.Series([0.01, 0.02, 0.03], index=idx), pd.Series(0.03, index=idx))


# --------------------------------------------------------------------------
# 4. La rama degenerada de la prevision
# --------------------------------------------------------------------------


def test_una_serie_constante_si_se_preve_sin_error():
    # El caso que justificaba la rama, y que es inocuo: prever exactamente un
    # cero no aporta ninguna informacion.
    idx = pd.bdate_range("2024-01-01", periods=40)
    ceros = pd.Series(np.zeros(40), index=idx, name="cdd")
    prevision = prevision_exogena(ceros, habilidad=0.90, semilla=1)
    assert prevision.dropna().eq(0.0).all()


def test_una_rampa_no_se_preve_perfectamente():
    """La condicion era 'el cambio es constante', no 'la serie es constante'.

    Cualquier rampa lineal o tramo interpolado la cumple, y en esos casos se
    devolvia el futuro exacto: sobre arange(0, 60, 0.5) el error maximo de la
    prevision era 0 con habilidad 0,99, con 0,50 y hasta con 0,01.
    """
    idx = pd.bdate_range("2024-01-01", periods=120)
    rampa = pd.Series(np.arange(0.0, 60.0, 0.5), index=idx, name="nivel")
    with pytest.raises(ErrorDatos, match="no se puede calibrar|No se puede calibrar"):
        prevision_exogena(rampa, habilidad=0.50, semilla=1)


def test_una_serie_de_dos_observaciones_no_se_preve_perfectamente():
    # Con ddof=1 y dos observaciones queda una sola diferencia, std devuelve
    # NaN y se tomaba tambien la rama de fuga: la prevision salia exacta.
    idx = pd.bdate_range("2024-01-01", periods=2)
    corta = pd.Series([10.0, 20.0], index=idx, name="nivel")
    with pytest.raises(ErrorDatos, match="No se puede calibrar"):
        prevision_exogena(corta, habilidad=0.90, semilla=1)


def test_la_habilidad_pedida_se_cumple_en_el_caso_normal():
    # La calibracion, que si estaba bien y conviene no romper.
    g = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=2000)
    serie = pd.Series(np.cumsum(g.standard_normal(2000)) * 0.5, index=idx, name="x")
    medidas = []
    for semilla in range(30):
        prevision = prevision_exogena(serie, habilidad=0.90, semilla=semilla)
        cambio_real = (serie.shift(-1) - serie).dropna()
        cambio_previsto = (prevision - serie).dropna()
        comun = cambio_real.index.intersection(cambio_previsto.index)
        medidas.append(np.corrcoef(cambio_real.loc[comun], cambio_previsto.loc[comun])[0, 1])
    assert np.mean(medidas) == pytest.approx(0.90, abs=0.02)
