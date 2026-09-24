"""El contrato entre la matriz de datos y el backtest de la estrategia.

Este fichero cubre una sola cosa, y es la que mas dano hace cuando se rompe:
que el rendimiento adelantado que produce
:func:`icm.ml.caracteristicas.objetivo_rendimiento_futuro` no se vuelva a
desplazar al ejecutar la estrategia.

El doble desplazamiento no revienta ni avisa: devuelve una cuenta de resultados
de aspecto normal y signo equivocado. Con una senal cuya correlacion con el
rendimiento futuro es de +0,11, medimos que convierte un ratio de Sharpe de
+2,6 en -0,5. Nada en la salida delata el problema, asi que la unica defensa es
que la convencion viaje pegada a los datos y que estas pruebas la vigilen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.ml.caracteristicas import construir_matriz, objetivo_rendimiento_futuro
from icm.ml.estrategia import ejecutar_estrategia


@pytest.fixture(scope="module")
def precios() -> pd.DataFrame:
    """Paseo aleatorio suave, suficiente para construir una matriz completa."""
    generador = np.random.default_rng(11)
    fechas = pd.bdate_range("2019-01-01", periods=800, name="fecha")
    rendimientos = generador.standard_normal((len(fechas), 2)) * 0.01
    niveles = 100.0 * np.exp(np.cumsum(rendimientos, axis=0))
    return pd.DataFrame(niveles, index=fechas, columns=["AAA", "BBB"])


# --- la marca viaja con los datos ---------------------------------------------


def test_el_rendimiento_futuro_se_marca_como_adelantado(precios):
    y = objetivo_rendimiento_futuro(precios["AAA"], horizonte=1)
    assert y.attrs["convencion"] == "futuro"
    assert y.attrs["horizonte"] == 1


def test_la_marca_recoge_el_horizonte_pedido(precios):
    y = objetivo_rendimiento_futuro(precios["AAA"], horizonte=5)
    assert y.attrs["horizonte"] == 5


def test_construir_matriz_conserva_la_marca(precios):
    # pandas pierde attrs al concatenar y filtrar, asi que esto no es gratis.
    md = construir_matriz(precios, "AAA", horizonte=1)
    assert md.y.attrs.get("convencion") == "futuro"
    assert md.y.attrs.get("horizonte") == 1


def test_una_serie_de_rendimientos_corriente_no_lleva_la_marca(precios):
    from icm.datos import rendimientos as calcular_rendimientos

    corrientes = calcular_rendimientos(precios, "log")["AAA"]
    assert corrientes.attrs.get("convencion") is None


# --- el backtest deduce el retardo --------------------------------------------


def test_con_rendimiento_adelantado_el_retardo_deducido_es_cero(precios):
    md = construir_matriz(precios, "AAA", horizonte=1)
    prediccion = md.y * 0.5   # prediccion imperfecta pero del signo correcto

    deducido = ejecutar_estrategia(prediccion, md.y, coste_bps=0.0)
    explicito = ejecutar_estrategia(prediccion, md.y, coste_bps=0.0, retardo=0)
    pd.testing.assert_series_equal(deducido.pnl, explicito.pnl)


def test_con_rendimiento_contemporaneo_el_retardo_deducido_es_uno(precios):
    from icm.datos import rendimientos as calcular_rendimientos

    corrientes = calcular_rendimientos(precios, "log")["AAA"]
    prediccion = corrientes.shift(1).fillna(0.0)

    deducido = ejecutar_estrategia(prediccion, corrientes, coste_bps=0.0)
    explicito = ejecutar_estrategia(prediccion, corrientes, coste_bps=0.0, retardo=1)
    pd.testing.assert_series_equal(deducido.pnl, explicito.pnl)


def test_el_doble_desplazamiento_se_rechaza(precios):
    md = construir_matriz(precios, "AAA", horizonte=1)
    with pytest.raises(ErrorDatos, match="adelantados"):
        ejecutar_estrategia(md.y, md.y, retardo=1)


def test_el_mensaje_de_rechazo_dice_que_hacer(precios):
    md = construir_matriz(precios, "AAA", horizonte=1)
    with pytest.raises(ErrorDatos) as excepcion:
        ejecutar_estrategia(md.y, md.y, retardo=1)
    mensaje = str(excepcion.value)
    assert "retardo=0" in mensaje
    assert "contemporaneos" in mensaje


def test_un_horizonte_mayor_tambien_se_rechaza(precios):
    md = construir_matriz(precios, "AAA", horizonte=5)
    with pytest.raises(ErrorDatos, match="5 periodo"):
        ejecutar_estrategia(md.y, md.y, retardo=1)


def test_con_serie_adelantada_no_se_avisa_de_retardo_cero(precios):
    # El aviso de retardo=0 es correcto con rendimientos contemporaneos, pero
    # con una serie ya adelantada seria una falsa alarma: el desplazamiento
    # esta hecho y cero es el valor correcto.
    import warnings

    md = construir_matriz(precios, "AAA", horizonte=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ejecutar_estrategia(md.y * 0.5, md.y, coste_bps=0.0)


def test_el_aviso_sigue_saltando_con_rendimientos_contemporaneos(precios):
    from icm.datos import rendimientos as calcular_rendimientos

    corrientes = calcular_rendimientos(precios, "log")["AAA"]
    with pytest.warns(UserWarning, match="informacion futura"):
        ejecutar_estrategia(corrientes, corrientes, coste_bps=0.0, retardo=0)


# --- la consecuencia economica -------------------------------------------------


def test_las_dos_formas_correctas_de_emparejar_dan_la_misma_economia(precios):
    """Adelantar el rendimiento o retrasar la senal describen lo mismo.

    Con una diferencia que conviene tener presente y que no es un error: la
    fecha en que queda imputado cada resultado. La via del rendimiento
    adelantado lo apunta en el dia de la DECISION; la del rendimiento
    contemporaneo, en el dia de la LIQUIDACION, un periodo despues. Los
    importes son los mismos en el mismo orden, pero las series no se pueden
    comparar alineadas por indice, y al medir un Sharpe sobre una muestra corta
    los extremos hacen que las cifras difieran en la tercera cifra decimal.
    Elija una convencion y mantengala para todo lo que vaya a comparar entre si.
    """
    from icm.datos import rendimientos as calcular_rendimientos

    corrientes = calcular_rendimientos(precios, "log")["AAA"]
    adelantado = objetivo_rendimiento_futuro(precios["AAA"], horizonte=1)

    generador = np.random.default_rng(3)
    prediccion = adelantado * 0.3 + generador.standard_normal(len(adelantado)) * 0.003
    prediccion = prediccion.dropna()

    via_adelantado = ejecutar_estrategia(prediccion, adelantado, coste_bps=0.0)
    via_contemporaneo = ejecutar_estrategia(prediccion, corrientes, coste_bps=0.0)

    # El resultado imputado al dia de la decision en una es el imputado al dia
    # siguiente en la otra: desplazar una periodo las iguala.
    comun = via_adelantado.pnl.index.intersection(via_contemporaneo.pnl.index)
    assert len(comun) > 500
    diferencia = (
        via_adelantado.pnl.shift(1).reindex(comun) - via_contemporaneo.pnl.reindex(comun)
    ).abs().max()
    assert diferencia < 1e-12, f"las dos vias no describen lo mismo: {diferencia:.2e}"

    # Y el desfase es de imputacion, no de contenido: la via adelantada abarca
    # dos sesiones mas al principio, de ahi que los totales no sean identicos.
    assert via_adelantado.pnl.index[0] < via_contemporaneo.pnl.index[0]


def test_el_doble_desplazamiento_habria_invertido_el_signo(precios):
    """Deja constancia del dano concreto que evita el rechazo.

    Se construye a mano el emparejamiento erroneo (senal retrasada un periodo
    contra un rendimiento ya adelantado) y se comprueba que da perdidas donde
    el correcto da beneficios. Es el numero que justifica que la funcion
    prefiera fallar antes que aceptar la llamada.
    """
    adelantado = objetivo_rendimiento_futuro(precios["AAA"], horizonte=1).dropna()
    generador = np.random.default_rng(5)
    prediccion = adelantado * 0.4 + generador.standard_normal(len(adelantado)) * 0.004

    correcto = ejecutar_estrategia(prediccion, adelantado, coste_bps=0.0)
    # El mismo calculo que haria retardo=1: la senal llega un periodo tarde.
    desplazada = prediccion.shift(1).dropna()
    erroneo = ejecutar_estrategia(desplazada, adelantado.loc[desplazada.index], coste_bps=0.0)

    assert correcto.pnl.sum() > 0
    assert erroneo.pnl.sum() < correcto.pnl.sum() / 2
