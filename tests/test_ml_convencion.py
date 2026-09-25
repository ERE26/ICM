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

    # El aviso de retardo=0 se emite por las dos vias: la marca dice de donde
    # salio la serie, no como esta alineada ahora.
    with pytest.warns(UserWarning, match="marcada como adelantada"):
        deducido = ejecutar_estrategia(prediccion, md.y, coste_bps=0.0)
    with pytest.warns(UserWarning, match="marcada como adelantada"):
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


def test_con_serie_adelantada_el_aviso_cambia_de_texto_pero_se_emite(precios):
    """Callar el aviso aqui era callarlo justo en el caso de fuga.

    Esta prueba consagraba lo contrario: que con una serie marcada como
    adelantada no se avisase de retardo=0, por ser el valor correcto. El
    razonamiento falla porque ``attrs['convencion']`` sobrevive a ``.shift()``,
    asi que la marca no prueba que la serie siga adelantada, solo de donde
    salio; y la unica senial del modulo quedaba suprimida precisamente cuando
    alguien la habia realineado. Ahora se avisa siempre que el retardo resuelto
    sea 0, con un texto que pide comprobar el alineamiento en lugar de acusar
    de leer el futuro.
    """
    md = construir_matriz(precios, "AAA", horizonte=1)

    with pytest.warns(UserWarning, match="marcada como") as avisos:
        ejecutar_estrategia(md.y * 0.5, md.y, coste_bps=0.0)

    assert len(avisos) == 1
    mensaje = str(avisos[0].message)
    assert "marcada como adelantada" in mensaje
    assert "attrs.pop('convencion', None)" in mensaje


def test_el_aviso_sigue_saltando_con_rendimientos_contemporaneos(precios):
    from icm.datos import rendimientos as calcular_rendimientos

    corrientes = calcular_rendimientos(precios, "log")["AAA"]
    with pytest.warns(UserWarning, match="informacion futura"):
        ejecutar_estrategia(corrientes, corrientes, coste_bps=0.0, retardo=0)


# --- la marca no prueba el alineamiento ---------------------------------------


def test_realinear_la_serie_conserva_la_marca_y_el_aviso_lo_delata(precios):
    """Una serie adelantada realineada con .shift(1) sigue diciendo 'futuro'.

    ``.shift(1)`` sobre el rendimiento adelantado deja el rendimiento
    contemporaneo de toda la vida, y es la operacion mas natural del mundo.
    Pero ``attrs`` viaja con el objeto: la serie llega marcada como adelantada,
    el retardo deducido es 0 y el pnl recoge la suma de los rendimientos en
    valor absoluto, que es la huella inconfundible de leer el futuro. Medido
    cuando el aviso estaba suprimido: pnl +5,5292, identico a la suma de |r|, y
    Sharpe publicado 66,3 sin un solo aviso.
    """
    adelantado = objetivo_rendimiento_futuro(precios["AAA"], horizonte=1)
    contemporanea = adelantado.shift(1)
    assert contemporanea.attrs["convencion"] == "futuro"

    # La prediccion es el rendimiento del propio periodo: informacion de t.
    prediccion = contemporanea.fillna(0.0)
    with pytest.warns(UserWarning, match="sigue estandolo"):
        resultado = ejecutar_estrategia(prediccion, contemporanea, coste_bps=0.0)

    maximo_extraible = contemporanea.reindex(resultado.pnl.index).abs().sum()
    assert resultado.pnl.sum() == pytest.approx(maximo_extraible)


def test_el_rechazo_de_retardo_uno_dice_como_borrar_una_marca_caducada(precios):
    """El mensaje empujaba a retardo=0, es decir, a la fuga.

    Quien realinea la serie y pide el retardo correcto se encontraba con un
    ``ErrorDatos`` que le mandaba usar retardo=0 sobre una serie
    contemporanea. Ahora el mensaje distingue los dos casos y dice como retirar
    una marca que ya no describe el alineamiento.
    """
    adelantado = objetivo_rendimiento_futuro(precios["AAA"], horizonte=1)
    contemporanea = adelantado.shift(1)
    prediccion = contemporanea.fillna(0.0)

    with pytest.raises(ErrorDatos) as excepcion:
        ejecutar_estrategia(prediccion, contemporanea, retardo=1)
    assert "attrs.pop('convencion', None)" in str(excepcion.value)

    # Y borrada la marca, la via honesta queda disponible y no hay fuga.
    limpia = contemporanea.copy()
    limpia.attrs.pop("convencion")
    honesta = ejecutar_estrategia(prediccion, limpia, coste_bps=0.0)
    assert abs(honesta.pnl.sum()) < 0.1 * contemporanea.abs().sum()


def test_el_horizonte_mayor_se_rechaza_tambien_por_la_via_por_omision(precios):
    """El camino por defecto era el peligroso, y era el que nadie probaba.

    Con horizonte h el valor en t y el de t+1 comparten h-1 periodos de
    movimiento, asi que ejecutar uno por fecha cobra cada movimiento hasta h
    veces, autocorrelaciona el pnl y deja sin sentido toda anualizacion. Solo
    se rechazaba ``retardo=1``; con el retardo deducido se aceptaba en
    silencio. Medido con comprar y mantener sobre un activo cuyo rendimiento
    log total es -0,1667: suma de pnl -0,8444 con h=5 (5,06 veces) y -3,5744
    con h=21 (21,44 veces), con la volatilidad anual pasando de 0,163 a 0,758.
    """
    md = construir_matriz(precios, "AAA", horizonte=5)
    comprar_y_mantener = pd.Series(1.0, index=md.y.index)

    with pytest.raises(ErrorDatos, match="se solapan") as excepcion:
        ejecutar_estrategia(comprar_y_mantener, md.y, coste_bps=0.0)

    mensaje = str(excepcion.value)
    assert "iloc[::5]" in mensaje
    assert "periodos_anio=50" in mensaje


def test_con_horizonte_uno_comprar_y_mantener_rinde_lo_que_el_activo(precios):
    """El contraste que da sentido al rechazo anterior: con h=1 cuadra exacto."""
    adelantado = objetivo_rendimiento_futuro(precios["AAA"], horizonte=1)
    comprar_y_mantener = pd.Series(1.0, index=adelantado.index)

    with pytest.warns(UserWarning, match="sigue estandolo"):
        resultado = ejecutar_estrategia(comprar_y_mantener, adelantado, coste_bps=0.0)

    total = float(np.log(precios["AAA"].iloc[-1] / precios["AAA"].iloc[0]))
    assert resultado.pnl.sum() == pytest.approx(total)


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

    with pytest.warns(UserWarning, match="marcada como adelantada"):
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


def test_adelantar_las_predicciones_de_la_validacion_es_exactamente_retardo_cero():
    """La receta que el docstring del modulo recomendaba era la trampa.

    ``ResultadoValidacion.predicciones`` se indexa con las filas de ``X``/``y``,
    y la ``y`` de :func:`construir_matriz` esta indexada en la fecha de
    DECISION, asi que esa serie ya viene en la convencion del backtest.
    Adelantarla con ``.shift(-1)`` y declarar ``retardo=1``, que es lo que
    ordenaba el docstring, produce fecha a fecha la MISMA cuenta de resultados
    que ``retardo=0``, la fuga que el modulo dedica cuarenta lineas a prohibir;
    y la produce sin aviso, porque el retardo declarado es 1.

    Se usa un proceso con momento real para que la via honesta tenga un
    resultado propio con el que contrastar: medido, Sharpe +4,22 con la receta
    frente a +1,57 emparejando bien.
    """
    from sklearn.linear_model import Ridge

    from icm.datos import rendimientos as calcular_rendimientos
    from icm.ml.validacion import ParticionWalkForward, validacion_walk_forward

    generador = np.random.default_rng(7)
    fechas = pd.bdate_range("2016-01-01", periods=600, name="fecha")
    ruido = generador.standard_normal(len(fechas)) * 0.01
    serie = np.zeros(len(fechas))
    for t in range(1, len(fechas)):
        serie[t] = 0.12 * serie[t - 1] + ruido[t]
    precios_momento = pd.DataFrame(
        {"ALFA": 100.0 * np.exp(np.cumsum(serie))}, index=fechas
    )

    md = construir_matriz(precios_momento, "ALFA", horizonte=1)
    validado = validacion_walk_forward(
        Ridge(alpha=1.0),
        md.X,
        md.y,
        ParticionWalkForward(n_particiones=3, tamano_prueba=80),
    )
    prediccion = validado.predicciones
    contemporaneos = calcular_rendimientos(precios_momento, "log")["ALFA"].reindex(
        prediccion.index
    )

    receta = ejecutar_estrategia(
        prediccion.shift(-1).dropna(), contemporaneos, coste_bps=0.0, retardo=1
    )
    with pytest.warns(UserWarning, match="informacion futura"):
        trampa = ejecutar_estrategia(prediccion, contemporaneos, coste_bps=0.0, retardo=0)
    honesta = ejecutar_estrategia(prediccion, contemporaneos, coste_bps=0.0, retardo=1)

    comun = receta.pnl.index.intersection(trampa.pnl.index)
    assert len(comun) > 200
    pd.testing.assert_series_equal(receta.pnl.reindex(comun), trampa.pnl.reindex(comun))

    # Y no es el mismo emparejamiento que el correcto: la receta cambiaba de
    # verdad la cuenta de resultados, solo que a la de la fuga.
    assert not np.allclose(
        receta.pnl.reindex(comun), honesta.pnl.reindex(comun), atol=1e-12
    )


def test_la_documentacion_no_manda_adelantar_una_serie_ya_indexada_en_la_decision():
    """El equivocado era el docstring, asi que el docstring es lo que se vigila.

    El modulo ordenaba literalmente ``prediccion =
    resultado.predicciones.shift(-1)`` y afirmaba que esa serie viene indexada
    en la fecha del rendimiento que predice. Las dos cosas son falsas para la
    tuberia de la libreria, y la prueba de arriba mide el dano: la receta da la
    cuenta de resultados de ``retardo=0``. Tambien sobraba la frase sobre la
    convencion de ``icm.ml.evaluacion``, que no tiene ninguna propia: compara
    termino a termino lo que le den.
    """
    from icm.ml import estrategia as modulo

    textos = {
        "modulo": modulo.__doc__,
        "ejecutar_estrategia": modulo.ejecutar_estrategia.__doc__,
    }
    for donde, texto in textos.items():
        assert "predicciones.shift(-1)" not in texto, donde
        assert "icm.ml.evaluacion" not in texto, donde

    # Y dice lo que si es cierto, para que nadie lo deduzca al reves.
    assert "ResultadoValidacion.predicciones" in textos["modulo"]
    assert "ya viene" in textos["modulo"]


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

    with pytest.warns(UserWarning, match="marcada como adelantada"):
        correcto = ejecutar_estrategia(prediccion, adelantado, coste_bps=0.0)
    # El mismo calculo que haria retardo=1: la senal llega un periodo tarde.
    desplazada = prediccion.shift(1).dropna()
    with pytest.warns(UserWarning, match="marcada como adelantada"):
        erroneo = ejecutar_estrategia(
            desplazada, adelantado.loc[desplazada.index], coste_bps=0.0
        )

    assert correcto.pnl.sum() > 0
    assert erroneo.pnl.sum() < correcto.pnl.sum() / 2
