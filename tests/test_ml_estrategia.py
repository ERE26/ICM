"""Pruebas del paso de la prediccion al dinero.

Lo que se comprueba aqui no son numeros sueltos sino las propiedades de las
que depende que un backtest signifique algo:

* con una prediccion perfecta y sin friccion se gana **todos** los periodos, y
  con friccion suficiente se pierde aunque la prediccion siga siendo perfecta;
* el retardo hace lo que dice hacer, y con retardo cero la estrategia se lleva
  exactamente la suma de los movimientos en valor absoluto, que es el maximo
  extraible de una serie y la huella inconfundible de que se esta leyendo el
  futuro;
* una prediccion sin informacion da un Sharpe indistinguible de cero sin
  costes, y negativo en cuanto se paga por operarla;
* quien no cambia de posicion no rota ni paga;
* el coste de equilibrio separa de verdad el Sharpe positivo del negativo,
  medido por el camino largo (ejecutando la estrategia con ese coste) y no por
  el atajo que usa la propia funcion.

Las pruebas estadisticas promedian varias semillas en lugar de fiarse de una:
un Sharpe muestral de una estrategia sin ventaja tiene una desviacion de
``sqrt(periodos_anio / n)``, que con 1500 observaciones diarias es 0,4, asi que
una sola semilla puede dar 0,8 sin que nada este roto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.cartera import volatilidad_anualizada
from icm.comun import ErrorDatos
from icm.ml.estrategia import (
    CLAVES_RESUMEN,
    ResultadoEstrategia,
    coste_de_equilibrio,
    comparar_estrategias,
    curva_beneficio,
    ejecutar_estrategia,
    objetivo_volatilidad,
    resumen_estrategia,
    senal_desde_prediccion,
)

N = 1500


@pytest.fixture(scope="module")
def rendimientos_ruido() -> pd.Series:
    """Rendimientos diarios sin ninguna estructura: nada que predecir."""
    generador = np.random.default_rng(11)
    indice = pd.bdate_range("2015-01-01", periods=N, name="fecha")
    return pd.Series(0.011 * generador.standard_normal(N), index=indice, name="rend")


@pytest.fixture(scope="module")
def caso_con_ventaja() -> tuple[pd.Series, pd.Series]:
    """Prediccion con ventaja real, en la convencion del modulo.

    ``prediccion[t]`` se construye con informacion de t y anticipa parte del
    rendimiento de t+1. La ventaja es deliberadamente grande (mucho mayor que
    cualquiera real) para que las propiedades se vean sin necesidad de series
    de decadas.
    """
    generador = np.random.default_rng(7)
    indice = pd.bdate_range("2018-01-01", periods=N, name="fecha")
    anticipado = 0.003 * generador.standard_normal(N)
    ruido = 0.010 * generador.standard_normal(N - 1)
    rendimientos = pd.Series(anticipado[:-1] + ruido, index=indice[1:], name="rend")
    prediccion = pd.Series(anticipado, index=indice, name="pred")
    return prediccion, rendimientos


# --- de la prediccion a la senal -----------------------------------------------


def test_la_senal_de_signo_solo_toma_tres_valores(caso_con_ventaja):
    prediccion, _ = caso_con_ventaja

    senal = senal_desde_prediccion(prediccion, modo="signo")

    assert set(np.unique(senal)) <= {-1.0, 0.0, 1.0}
    np.testing.assert_array_equal(np.sign(senal), np.sign(prediccion))


def test_la_senal_de_signo_respeta_el_limite_como_posicion_maxima(caso_con_ventaja):
    prediccion, _ = caso_con_ventaja

    senal = senal_desde_prediccion(prediccion, modo="signo", limite=3.0)

    assert set(np.unique(senal)) <= {-3.0, 0.0, 3.0}


def test_la_senal_proporcional_se_recorta_en_el_limite(caso_con_ventaja):
    prediccion, _ = caso_con_ventaja

    # Con una escala diminuta casi toda prediccion se sale del limite.
    senal = senal_desde_prediccion(prediccion, modo="proporcional", escala=1e-6, limite=1.5)

    assert (senal.abs() <= 1.5 + 1e-12).all()
    assert np.isclose(senal.abs(), 1.5).mean() > 0.99
    np.testing.assert_array_equal(np.sign(senal), np.sign(prediccion))

    # Y con la escala por defecto el limite se sigue respetando.
    por_defecto = senal_desde_prediccion(prediccion, modo="proporcional", limite=1.0)
    assert (por_defecto.abs() <= 1.0 + 1e-12).all()


def test_la_senal_proporcional_con_escala_uno_es_la_identidad(caso_con_ventaja):
    """Garantia de composicion: permite ejecutar posiciones ya calculadas.

    Es el mecanismo con el que se lleva la salida de objetivo_volatilidad a
    ejecutar_estrategia, que siempre construye la senal por su cuenta.
    """
    prediccion, _ = caso_con_ventaja

    senal = senal_desde_prediccion(prediccion, modo="proporcional", escala=1.0, limite=1e6)

    np.testing.assert_allclose(senal.to_numpy(), prediccion.to_numpy())


def test_el_modo_umbral_opera_estrictamente_menos_veces_que_el_modo_signo(caso_con_ventaja):
    """La banda muerta renuncia a las apuestas que no pagarian la horquilla.

    Con el umbral en la mediana de |prediccion| debe quedarse, por definicion
    de mediana, con la mitad de los periodos; y al operar menos, paga menos.
    """
    prediccion, rendimientos = caso_con_ventaja
    umbral = float(prediccion.abs().median())

    signo = senal_desde_prediccion(prediccion, modo="signo")
    umbralada = senal_desde_prediccion(prediccion, modo="umbral", umbral=umbral)

    assert (umbralada != 0).sum() < (signo != 0).sum()
    assert (umbralada != 0).sum() == pytest.approx(len(prediccion) / 2, rel=0.01)

    coste_signo = ejecutar_estrategia(prediccion, rendimientos, modo="signo").costes.sum()
    coste_umbral = ejecutar_estrategia(
        prediccion, rendimientos, modo="umbral", umbral=umbral
    ).costes.sum()
    assert coste_umbral < coste_signo


def test_la_prediccion_ausente_deja_la_posicion_plana():
    """Donde no hay opinion no se toma riesgo, y no se arrastra la de ayer."""
    prediccion = pd.Series([0.01, np.nan, -0.02, np.inf, 0.0])

    senal = senal_desde_prediccion(prediccion, modo="signo")

    np.testing.assert_array_equal(senal.to_numpy(), [1.0, 0.0, -1.0, 0.0, 0.0])


def test_las_opciones_incoherentes_de_la_senal_avisan(caso_con_ventaja):
    prediccion, _ = caso_con_ventaja

    with pytest.raises(ErrorDatos, match="desconocido"):
        senal_desde_prediccion(prediccion, modo="direccional")
    with pytest.raises(ErrorDatos, match="umbral"):
        senal_desde_prediccion(prediccion, umbral=-0.1)
    with pytest.raises(ErrorDatos, match="no filtra nada"):
        senal_desde_prediccion(prediccion, modo="umbral")
    with pytest.raises(ErrorDatos, match="no tiene banda muerta"):
        senal_desde_prediccion(prediccion, modo="proporcional", umbral=0.01)
    with pytest.raises(ErrorDatos, match="posicion maxima"):
        senal_desde_prediccion(prediccion, limite=0.0)
    with pytest.raises(ErrorDatos, match="escala"):
        senal_desde_prediccion(prediccion, modo="proporcional", escala=-1.0)
    with pytest.raises(ErrorDatos, match="ningun valor finito"):
        senal_desde_prediccion(pd.Series([np.nan, np.nan]))
    with pytest.raises(ErrorDatos, match="constante"):
        senal_desde_prediccion(pd.Series([0.5, 0.5, 0.5]), modo="proporcional")


# --- objetivo de volatilidad ---------------------------------------------------


@pytest.fixture(scope="module")
def caso_dos_regimenes() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rendimientos con volatilidad que cambia de regimen, y su valor exacto.

    Se le entrega al escalador la volatilidad verdadera del proceso, no una
    estimacion: asi la prueba mide lo que hace el escalador y no el error de la
    ventana movil, que es otro problema.
    """
    generador = np.random.default_rng(3)
    indice = pd.bdate_range("2012-01-01", periods=N, name="fecha")
    volatilidad_diaria = 0.004 + 0.012 * (0.5 + 0.5 * np.sin(np.arange(N) / 180.0))
    rendimientos = pd.Series(
        volatilidad_diaria * generador.standard_normal(N), index=indice, name="rend"
    )
    senal = pd.Series(np.where(generador.standard_normal(N) > 0, 1.0, -1.0), index=indice)
    volatilidad_anual = pd.Series(volatilidad_diaria * np.sqrt(252), index=indice)
    return senal, rendimientos, volatilidad_anual


def test_con_volatilidad_igual_al_objetivo_la_posicion_no_se_toca(caso_dos_regimenes):
    """Caso con solucion analitica: el multiplicador es exactamente 1."""
    senal, _, _ = caso_dos_regimenes

    posiciones = objetivo_volatilidad(
        senal, pd.Series(0.10, index=senal.index), objetivo_anual=0.10
    )

    np.testing.assert_allclose(posiciones.to_numpy(), senal.to_numpy())


def test_el_objetivo_de_volatilidad_escala_a_la_inversa_de_la_volatilidad(caso_dos_regimenes):
    senal, _, volatilidad = caso_dos_regimenes

    posiciones = objetivo_volatilidad(senal, volatilidad, objetivo_anual=0.10, limite=10.0)

    np.testing.assert_allclose(posiciones.abs().to_numpy(), 0.10 / volatilidad.to_numpy())
    # A mas volatilidad, menos posicion: la relacion es monotona decreciente.
    orden = np.argsort(volatilidad.to_numpy())
    assert (np.diff(posiciones.abs().to_numpy()[orden]) <= 1e-12).all()


def test_el_objetivo_de_volatilidad_acerca_la_volatilidad_realizada_al_objetivo(
    caso_dos_regimenes,
):
    """La prueba que justifica la funcion: el riesgo acaba donde se pidio."""
    senal, rendimientos, volatilidad = caso_dos_regimenes
    posiciones = objetivo_volatilidad(senal, volatilidad, objetivo_anual=0.10, limite=10.0)
    identidad = {"modo": "proporcional", "escala": 1.0, "limite": 10.0}

    escalada = ejecutar_estrategia(posiciones, rendimientos, coste_bps=0.0, **identidad)
    sin_escalar = ejecutar_estrategia(senal, rendimientos, coste_bps=0.0, **identidad)

    error_escalada = abs(volatilidad_anualizada(escalada.pnl) - 0.10)
    error_sin_escalar = abs(volatilidad_anualizada(sin_escalar.pnl) - 0.10)
    assert error_escalada < 0.01
    assert error_escalada < error_sin_escalar / 5.0


def test_el_tope_del_multiplicador_acota_la_posicion(caso_dos_regimenes):
    senal, _, volatilidad = caso_dos_regimenes

    posiciones = objetivo_volatilidad(senal, volatilidad, objetivo_anual=1.0, limite=2.0)

    assert posiciones.abs().max() == pytest.approx(2.0)


def test_el_objetivo_de_volatilidad_avisa_si_la_volatilidad_no_esta_anualizada(
    caso_dos_regimenes,
):
    """El error mas facil de cometer y el que menos se ve en el resultado."""
    senal, _, volatilidad_anual = caso_dos_regimenes
    por_periodo = volatilidad_anual / np.sqrt(252)

    with pytest.warns(UserWarning, match="anual"):
        posiciones = objetivo_volatilidad(senal, por_periodo, limite=2.0)

    # Sin el aviso, esto pasaria inadvertido: apalancada al maximo siempre.
    np.testing.assert_allclose(posiciones.abs().to_numpy(), 2.0)


def test_el_objetivo_de_volatilidad_rechaza_volatilidades_negativas(caso_dos_regimenes):
    senal, _, volatilidad = caso_dos_regimenes

    with pytest.raises(ErrorDatos, match="negativas"):
        objetivo_volatilidad(senal, -volatilidad)


def test_las_fechas_sin_volatilidad_utilizable_quedan_planas(caso_dos_regimenes):
    senal, _, volatilidad = caso_dos_regimenes
    con_huecos = volatilidad.copy()
    con_huecos.iloc[:5] = np.nan
    con_huecos.iloc[5:8] = 0.0

    posiciones = objetivo_volatilidad(senal, con_huecos, limite=10.0)

    assert (posiciones.iloc[:8] == 0.0).all()
    assert (posiciones.iloc[8:] != 0.0).all()


# --- ejecucion -----------------------------------------------------------------


def test_con_prediccion_perfecta_y_sin_costes_se_gana_todos_los_periodos(rendimientos_ruido):
    """Prediccion perfecta: la del rendimiento futuro, conocido de antemano.

    No es un caso realista sino la comprobacion de que las tuberias estan bien
    conectadas: si sabiendo el futuro la estrategia no ganase, el error estaria
    en el modulo y no en el modelo. La posicion es el signo del rendimiento
    siguiente, asi que el pnl de cada periodo es su valor absoluto.
    """
    perfecta = rendimientos_ruido.shift(-1)

    resultado = ejecutar_estrategia(perfecta, rendimientos_ruido, coste_bps=0.0)

    assert (resultado.pnl > 0).all()
    assert resumen_estrategia(resultado)["ratio_sharpe"] > 5.0
    np.testing.assert_allclose(
        resultado.pnl.to_numpy(), rendimientos_ruido.iloc[1:].abs().to_numpy()
    )


def test_con_prediccion_perfecta_y_costes_enormes_se_pierde_dinero(rendimientos_ruido):
    """Los costes muerden: saber el futuro no basta si operarlo cuesta mas.

    Con rendimientos diarios del orden del 1 % y una estrategia que se da la
    vuelta casi a diario (mueve dos veces el nominal), 500 puntos basicos por
    nominal movido se comen la ventaja entera varias veces.
    """
    perfecta = rendimientos_ruido.shift(-1)

    resultado = ejecutar_estrategia(perfecta, rendimientos_ruido, coste_bps=500.0)

    assert resultado.pnl_bruto.sum() > 0
    assert resultado.pnl.sum() < 0
    assert resumen_estrategia(resultado)["ratio_sharpe"] < 0


def test_el_retardo_cero_es_trampa_y_no_una_opcion(rendimientos_ruido):
    """retardo=0 no es una variante mas agresiva: es un backtest imposible.

    La prediccion de esta prueba es el rendimiento del propio periodo, es decir
    una serie perfectamente conocida al cierre de t. Bajo la convencion del
    modulo, ``prediccion[t]`` debe apostar por el rendimiento de t+1, asi que
    ejecutarla con retardo=0 la cobra contra el rendimiento de t, que ya habia
    ocurrido cuando se decidio: para haberlo cobrado habria que haber estado
    posicionado el periodo anterior, cuando ese rendimiento aun no se conocia.

    La prueba de que es imposible y no solo optimista esta en la cifra: con
    retardo=0 la estrategia se lleva **exactamente** la suma de los movimientos
    en valor absoluto, que es el maximo que cabe extraer de una serie de
    precios y que ninguna estrategia real roza. Con retardo=1, la misma senal
    no vale nada, que es lo que de verdad valia.

    Por eso retardo=0 no es una opcion legitima del backtest. Si lo que se
    tiene es una prediccion indexada en la fecha del rendimiento que predice
    (la convencion de icm.ml.evaluacion), hay que desplazarla con .shift(-1) y
    dejar retardo=1, no bajar el retardo.
    """
    contemporanea = rendimientos_ruido

    with pytest.warns(UserWarning, match="informacion futura"):
        trampa = ejecutar_estrategia(
            contemporanea, rendimientos_ruido, coste_bps=0.0, retardo=0
        )
    honesta = ejecutar_estrategia(contemporanea, rendimientos_ruido, coste_bps=0.0, retardo=1)

    assert trampa.pnl.sum() > honesta.pnl.sum()
    np.testing.assert_allclose(trampa.pnl.sum(), rendimientos_ruido.abs().sum())
    assert trampa.pnl.sum() > 5.0 * abs(honesta.pnl.sum())
    # El Sharpe de la version honesta esta dentro del ruido muestral: su
    # desviacion tipica es sqrt(252 / 1500) = 0,41, asi que la cota son mas de
    # siete desviaciones y no depende de la semilla. El de la trampa no cabe en
    # ninguna cota razonable, que es justo lo que delata el sesgo.
    assert resumen_estrategia(trampa)["ratio_sharpe"] > 10.0
    assert abs(resumen_estrategia(honesta)["ratio_sharpe"]) < 3.0


def test_la_posicion_es_la_senal_desplazada_y_las_cuentas_cuadran(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja
    coste_bps = 3.0

    resultado = ejecutar_estrategia(prediccion, rendimientos, coste_bps=coste_bps, retardo=1)

    esperada = senal_desde_prediccion(prediccion).shift(1).dropna()
    np.testing.assert_allclose(
        resultado.posiciones.to_numpy(),
        esperada.reindex(resultado.posiciones.index).to_numpy(),
    )
    np.testing.assert_allclose(
        resultado.pnl_bruto.to_numpy(),
        resultado.posiciones.to_numpy()
        * rendimientos.reindex(resultado.posiciones.index).to_numpy(),
    )
    np.testing.assert_allclose(
        resultado.costes.to_numpy(), resultado.rotacion.to_numpy() * coste_bps / 10_000.0
    )
    np.testing.assert_allclose(
        resultado.pnl.to_numpy(),
        (resultado.pnl_bruto - resultado.costes).to_numpy(),
    )


def test_un_retardo_mayor_retrasa_la_posicion_un_periodo_mas(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja

    uno = ejecutar_estrategia(prediccion, rendimientos, retardo=1)
    dos = ejecutar_estrategia(prediccion, rendimientos, retardo=2)

    comunes = uno.posiciones.index.intersection(dos.posiciones.index)
    np.testing.assert_allclose(
        dos.posiciones.reindex(comunes).to_numpy()[1:],
        uno.posiciones.reindex(comunes).to_numpy()[:-1],
    )


def test_una_prediccion_de_puro_ruido_da_un_sharpe_cercano_a_cero(rendimientos_ruido):
    """Sin informacion no hay dinero: la comprobacion de que no hay fugas.

    Se promedian doce semillas porque el Sharpe muestral de una estrategia sin
    ventaja tiene una desviacion tipica de sqrt(252 / 1500) = 0,41: con una
    sola semilla, un 0,8 no significaria nada.
    """
    sharpes = []
    for semilla in range(12):
        generador = np.random.default_rng(1000 + semilla)
        ruido = pd.Series(
            generador.standard_normal(len(rendimientos_ruido)), index=rendimientos_ruido.index
        )
        resultado = ejecutar_estrategia(ruido, rendimientos_ruido, coste_bps=0.0)
        sharpes.append(resumen_estrategia(resultado)["ratio_sharpe"])

    # Cotas a mas de cuatro desviaciones tipicas en los dos casos: 0,41 para un
    # Sharpe individual y 0,41 / sqrt(12) = 0,12 para la media de los doce.
    sharpes = np.asarray(sharpes)
    assert abs(sharpes.mean()) < 0.5
    assert np.abs(sharpes).max() < 2.0


def test_una_prediccion_de_puro_ruido_con_costes_pierde_dinero(rendimientos_ruido):
    """El corolario que importa: sin ventaja, operar es una perdida segura.

    El resultado bruto de una senal sin informacion es una moneda al aire con
    media cero, asi que sale positivo la mitad de las veces; los costes, en
    cambio, se pagan siempre. Con una rotacion de la posicion entera cada dia y
    10 puntos basicos, la friccion acumulada supera de largo la dispersion del
    resultado bruto y la perdida deja de depender de la suerte.

    De ahi que el coste de equilibrio del ruido sea pequenio pero no siempre
    exactamente cero: cuando el azar deja el bruto en positivo, la senal
    "aguanta" los dos o tres puntos basicos que ese azar paga. Es justamente el
    motivo de informar del coste de equilibrio y no del signo del Sharpe: dos o
    tres puntos basicos se leen como lo que son, ruido, mientras que un Sharpe
    positivo se lee como un hallazgo.
    """
    generador = np.random.default_rng(5)
    ruido = pd.Series(
        generador.standard_normal(len(rendimientos_ruido)), index=rendimientos_ruido.index
    )

    resultado = ejecutar_estrategia(ruido, rendimientos_ruido, coste_bps=10.0)

    assert resultado.pnl.sum() < 0
    assert resultado.costes.sum() > 0
    assert resultado.pnl.sum() < resultado.pnl_bruto.sum()
    assert coste_de_equilibrio(ruido, rendimientos_ruido) < 10.0


def test_una_posicion_que_no_cambia_no_rota_ni_paga_costes(rendimientos_ruido):
    """Comprar y mantener no paga friccion: la serie arranca ya invertida."""
    constante = pd.Series(1.0, index=rendimientos_ruido.index)

    resultado = ejecutar_estrategia(constante, rendimientos_ruido, coste_bps=25.0)

    assert (resultado.posiciones == 1.0).all()
    assert resultado.rotacion.abs().sum() == 0.0
    assert resultado.costes.abs().sum() == 0.0
    np.testing.assert_allclose(resultado.pnl.to_numpy(), resultado.pnl_bruto.to_numpy())
    resumen = resumen_estrategia(resultado)
    assert resumen["rotacion_media"] == 0.0
    assert resumen["coste_total_anualizado"] == 0.0
    assert resumen["merma_sharpe"] == 0.0


def test_los_huecos_del_rendimiento_no_se_operan_y_la_posicion_se_mantiene(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja
    con_hueco = rendimientos.copy()
    con_hueco.iloc[10] = np.nan

    resultado = ejecutar_estrategia(prediccion, con_hueco, coste_bps=0.0)

    assert rendimientos.index[10] not in resultado.pnl.index
    assert len(resultado.pnl) == len(ejecutar_estrategia(prediccion, rendimientos).pnl) - 1
    assert np.isfinite(resultado.pnl.to_numpy()).all()


def test_las_entradas_incoherentes_de_la_ejecucion_avisan(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja

    with pytest.raises(ErrorDatos, match="no negativo"):
        ejecutar_estrategia(prediccion, rendimientos, coste_bps=-1.0)
    with pytest.raises(ErrorDatos, match="entero"):
        ejecutar_estrategia(prediccion, rendimientos, retardo=1.5)
    with pytest.raises(ErrorDatos, match="leer el futuro"):
        ejecutar_estrategia(prediccion, rendimientos, retardo=-1)
    with pytest.raises(ErrorDatos, match="no comparten ninguna fecha"):
        ejecutar_estrategia(prediccion, rendimientos.reset_index(drop=True))
    with pytest.raises(ErrorDatos, match="repetidas"):
        ejecutar_estrategia(
            prediccion, pd.concat([rendimientos.iloc[:5], rendimientos.iloc[:5]])
        )
    with pytest.raises(ErrorDatos, match="hacen falta al menos"):
        ejecutar_estrategia(prediccion.iloc[:2], rendimientos.iloc[:2], retardo=2)


# --- medicion ------------------------------------------------------------------


def test_el_resumen_trae_todas_las_claves_esperadas_y_ningun_nan(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja

    resultado = ejecutar_estrategia(prediccion, rendimientos, coste_bps=2.0)
    resumen = resumen_estrategia(resultado)

    assert list(resumen.index) == list(CLAVES_RESUMEN)
    assert resumen.notna().all()
    assert resumen["observaciones"] == len(resultado)
    assert resumen["coste_bps"] == 2.0
    assert resumen["rotacion_media"] > 0
    assert resumen["maxima_caida"] <= 0


def test_el_resumen_acepta_una_frecuencia_por_nombre(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja
    resultado = ejecutar_estrategia(prediccion, rendimientos)

    por_nombre = resumen_estrategia(resultado, periodos_anio="diaria")
    por_numero = resumen_estrategia(resultado, periodos_anio=252)

    pd.testing.assert_series_equal(por_nombre, por_numero)


def test_el_sharpe_bruto_nunca_es_peor_que_el_neto_y_la_merma_lo_mide(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja

    barato = resumen_estrategia(ejecutar_estrategia(prediccion, rendimientos, coste_bps=1.0))
    caro = resumen_estrategia(ejecutar_estrategia(prediccion, rendimientos, coste_bps=20.0))

    assert barato["sharpe_bruto"] == pytest.approx(caro["sharpe_bruto"])
    assert barato["ratio_sharpe"] > caro["ratio_sharpe"]
    assert 0 < barato["merma_sharpe"] < caro["merma_sharpe"]
    assert caro["coste_total_anualizado"] == pytest.approx(
        20.0 * barato["coste_total_anualizado"]
    )


def test_comparar_estrategias_ordena_por_sharpe_neto(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja
    resultados = {
        "sin_costes": ejecutar_estrategia(prediccion, rendimientos, coste_bps=0.0),
        "cara": ejecutar_estrategia(prediccion, rendimientos, coste_bps=20.0),
        "barata": ejecutar_estrategia(prediccion, rendimientos, coste_bps=2.0),
    }

    tabla = comparar_estrategias(resultados)

    assert list(tabla.index) == ["sin_costes", "barata", "cara"]
    assert list(tabla.columns) == list(CLAVES_RESUMEN)
    assert tabla["ratio_sharpe"].is_monotonic_decreasing


def test_la_curva_de_beneficio_acumula_el_pnl_de_cada_estrategia(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja
    resultados = {
        "barata": ejecutar_estrategia(prediccion, rendimientos, coste_bps=2.0),
        "cara": ejecutar_estrategia(prediccion, rendimientos, coste_bps=50.0),
    }

    curvas = curva_beneficio(resultados)

    assert list(curvas.columns) == ["barata", "cara"]
    np.testing.assert_allclose(
        curvas["barata"].to_numpy(), resultados["barata"].pnl.cumsum().to_numpy()
    )
    # El coste solo resta, asi que la curva cara nunca va por encima.
    assert (curvas["cara"] <= curvas["barata"] + 1e-12).all()


def test_la_curva_de_beneficio_avisa_si_los_periodos_no_se_solapan(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja
    mitad = len(rendimientos) // 2
    resultados = {
        "primera": ejecutar_estrategia(prediccion.iloc[:mitad], rendimientos.iloc[:mitad]),
        "segunda": ejecutar_estrategia(prediccion.iloc[mitad:], rendimientos.iloc[mitad:]),
    }

    with pytest.warns(UserWarning, match="ninguna fecha"):
        curvas = curva_beneficio(resultados)

    assert curvas.notna().all().all()


def test_las_funciones_de_medicion_rechazan_lo_que_no_es_un_resultado(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja
    resultado = ejecutar_estrategia(prediccion, rendimientos)

    with pytest.raises(ErrorDatos, match="ResultadoEstrategia"):
        resumen_estrategia(resultado.pnl)
    with pytest.raises(ErrorDatos, match="diccionario"):
        comparar_estrategias([resultado])
    with pytest.raises(ErrorDatos, match="ninguna estrategia"):
        comparar_estrategias({})
    with pytest.raises(ErrorDatos, match="ResultadoEstrategia"):
        curva_beneficio({"mala": resultado.pnl})


# --- coste de equilibrio -------------------------------------------------------


def test_el_coste_de_equilibrio_separa_el_sharpe_positivo_del_negativo(caso_con_ventaja):
    """Por debajo del coste de equilibrio se gana; por encima, se pierde.

    Se comprueba por el camino largo, volviendo a ejecutar la estrategia con
    cada coste, para que la prueba valide tambien que el atajo analitico de
    coste_de_equilibrio coincide con el Sharpe que publica resumen_estrategia.
    """
    prediccion, rendimientos = caso_con_ventaja

    equilibrio = coste_de_equilibrio(prediccion, rendimientos)

    assert 0.0 < equilibrio < 100.0
    for factor, signo in ((0.9, 1.0), (1.1, -1.0)):
        resultado = ejecutar_estrategia(
            prediccion, rendimientos, coste_bps=factor * equilibrio
        )
        assert signo * resumen_estrategia(resultado)["ratio_sharpe"] > 0

    justo = ejecutar_estrategia(prediccion, rendimientos, coste_bps=equilibrio)
    assert resumen_estrategia(justo)["ratio_sharpe"] == pytest.approx(0.0, abs=1e-6)


def test_el_coste_de_equilibrio_es_cero_si_ni_sin_costes_se_gana(caso_con_ventaja):
    """Una estrategia que pierde sin friccion no tiene friccion que aguantar."""
    prediccion, rendimientos = caso_con_ventaja

    assert coste_de_equilibrio(-prediccion, rendimientos) == 0.0


def test_el_coste_de_equilibrio_devuelve_el_tope_y_avisa_si_no_lo_alcanza(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja

    with pytest.warns(UserWarning, match="al menos esto"):
        equilibrio = coste_de_equilibrio(prediccion, rendimientos, maximo_bps=5.0)

    assert equilibrio == 5.0


def test_la_banda_muerta_reduce_la_rotacion_y_por_tanto_los_costes(caso_con_ventaja):
    """El mecanismo por el que un umbral puede salvar una senal: rotar menos.

    Se comparan los dos modos con el **mismo** dimensionamiento, para que la
    unica diferencia sea la banda muerta y no el tamanio de las apuestas.

    Lo que no se afirma aqui, porque no es cierto en general, es que el umbral
    mejore el coste de equilibrio: renunciar a las apuestas pequenias ahorra
    rotacion pero tambien tira ventaja bruta, y quien gana depende de si esas
    predicciones pequenias llevaban informacion. En esta prueba la prediccion
    esta perfectamente calibrada (es la media condicionada exacta), asi que
    hasta la apuesta mas pequenia es rentable y el umbral no siempre compensa;
    con una prediccion real, que es casi toda ruido cerca de cero, compensa.
    Por eso el modulo ofrece coste_de_equilibrio para medirlo caso por caso en
    lugar de una regla general que seria falsa.
    """
    prediccion, rendimientos = caso_con_ventaja
    umbral = float(prediccion.abs().quantile(0.7))
    dimensionamiento = {"escala": float(prediccion.std()), "limite": 1.0}

    sin_banda = ejecutar_estrategia(
        prediccion, rendimientos, coste_bps=5.0, modo="proporcional", **dimensionamiento
    )
    con_banda = ejecutar_estrategia(
        prediccion,
        rendimientos,
        coste_bps=5.0,
        modo="umbral",
        umbral=umbral,
        **dimensionamiento,
    )

    assert con_banda.rotacion.mean() < sin_banda.rotacion.mean()
    assert con_banda.costes.sum() < sin_banda.costes.sum()
    assert (con_banda.posiciones == 0.0).mean() > (sin_banda.posiciones == 0.0).mean()


def test_el_coste_de_equilibrio_no_deja_que_se_le_fije_el_coste(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja

    with pytest.raises(ErrorDatos, match="es lo que esta funcion calcula"):
        coste_de_equilibrio(prediccion, rendimientos, coste_bps=3.0)
    with pytest.raises(ErrorDatos, match="maximo_bps"):
        coste_de_equilibrio(prediccion, rendimientos, maximo_bps=0.0)


def test_el_resultado_conoce_su_longitud(caso_con_ventaja):
    prediccion, rendimientos = caso_con_ventaja

    resultado = ejecutar_estrategia(prediccion, rendimientos)

    assert isinstance(resultado, ResultadoEstrategia)
    assert len(resultado) == len(resultado.pnl) == len(rendimientos) - 1
