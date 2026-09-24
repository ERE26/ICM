"""Pruebas del modulo de graficos de aprendizaje automatico.

No se comprueba el aspecto de las figuras, sino que lleven lo que deben llevar:
etiquetas en los dos ejes, leyenda solo cuando hay algo que distinguir, los
colores de la paleta en su orden y el numero exacto de marcas que corresponde a
los datos. Donde el grafico resume un calculo (acierto por grupo, pendiente de
la recta ajustada) se comprueba contra casos de solucion conocida a mano, que es
la unica forma de que la anotacion del grafico signifique lo que dice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from icm.comun import ErrorDatos  # noqa: E402
from icm.graficos import CHROME, ESTADO, PALETA  # noqa: E402
from icm.ml import graficos  # noqa: E402  (debe ir tras fijar el backend)


@pytest.fixture(autouse=True)
def _estilo_y_cierre():
    import matplotlib.pyplot as plt

    graficos.aplicar_estilo()
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def fechas() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=260)


@pytest.fixture(scope="module")
def observado(fechas: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(20240501)
    return pd.Series(rng.normal(0.0, 0.012, len(fechas)), index=fechas, name="real")


@pytest.fixture(scope="module")
def previsto(observado: pd.Series) -> pd.Series:
    """Prediccion con senial real pero encogida: correlacion positiva y pendiente < 1."""
    rng = np.random.default_rng(7)
    ruido = rng.normal(0.0, 0.004, len(observado))
    return pd.Series(0.45 * observado.to_numpy() + ruido, index=observado.index, name="previsto")


@pytest.fixture(scope="module")
def tabla_particiones() -> pd.DataFrame:
    """Cuatro particiones hacia delante, con embargo y sin solape."""
    filas = []
    for numero in range(4):
        inicio = pd.Timestamp("2018-01-01") + pd.DateOffset(months=6 * numero)
        fin_entrena = inicio + pd.DateOffset(months=12)
        filas.append(
            {
                "particion": numero,
                "inicio_entrena": inicio,
                "fin_entrena": fin_entrena,
                "inicio_prueba": fin_entrena + pd.Timedelta(days=5),
                "fin_prueba": fin_entrena + pd.DateOffset(months=6),
            }
        )
    return pd.DataFrame(filas).set_index("particion")


@pytest.fixture(scope="module")
def historial() -> pd.DataFrame:
    """Historial con sobreajuste evidente: la validacion toca fondo en la epoca 8."""
    epocas = np.arange(1, 21)
    entrenamiento = 1.0 / np.sqrt(epocas)
    validacion = 0.5 + (epocas - 8) ** 2 / 200.0
    return pd.DataFrame(
        {"perdida_entrenamiento": entrenamiento, "perdida_validacion": validacion},
        index=pd.Index(epocas, name="epoca"),
    )


#: Todas las funciones de dibujo del modulo, que las pruebas comunes recorren
#: una a una. Se deriva de ``__all__`` para que una funcion nueva sin prueba
#: rompa la bateria en lugar de pasar desapercibida.
NOMBRES = sorted(nombre for nombre in graficos.__all__ if nombre != "aplicar_estilo")


@pytest.fixture
def dibujos(observado, previsto, tabla_particiones, historial):
    """Una llamada valida por funcion publica, para las pruebas comunes."""
    curva = observado.cumsum()
    return {
        "particiones_walk_forward": lambda **kw: graficos.particiones_walk_forward(
            tabla_particiones, **kw
        ),
        "prediccion_vs_real": lambda **kw: graficos.prediccion_vs_real(
            observado, {"cresta": previsto}, **kw
        ),
        "dispersion_prediccion": lambda **kw: graficos.dispersion_prediccion(
            observado, previsto, **kw
        ),
        "importancia_variables": lambda **kw: graficos.importancia_variables(
            pd.Series({"a": 0.5, "b": 0.3, "c": 0.2}), **kw
        ),
        "curva_beneficio": lambda **kw: graficos.curva_beneficio({"modelo": curva}, **kw),
        "distribucion_errores": lambda **kw: graficos.distribucion_errores(
            {"modelo": observado - previsto}, **kw
        ),
        "acierto_por_magnitud": lambda **kw: graficos.acierto_por_magnitud(
            observado, previsto, **kw
        ),
        "historial_entrenamiento": lambda **kw: graficos.historial_entrenamiento(historial, **kw),
    }


# --- contrato comun ------------------------------------------------------------


@pytest.mark.parametrize("nombre", NOMBRES)
def test_cada_funcion_devuelve_figura_y_ejes_etiquetados(nombre, dibujos):
    figura, ax = dibujos[nombre]()
    assert figura is ax.figure
    assert ax.get_xlabel(), f"{nombre} no etiqueta el eje horizontal"
    assert ax.get_ylabel(), f"{nombre} no etiqueta el eje vertical"


@pytest.mark.parametrize("nombre", NOMBRES)
def test_se_puede_dibujar_sobre_unos_ejes_existentes(nombre, dibujos):
    import matplotlib.pyplot as plt

    figura_previa, ejes = plt.subplots(1, 2, figsize=(11, 4))
    figura, ax = dibujos[nombre](ax=ejes[1])
    assert ax is ejes[1]
    assert figura is figura_previa
    # El panel vecino se queda intacto: nada de dibujar en la figura entera.
    assert not ejes[0].lines and not ejes[0].patches


def test_las_pruebas_comunes_cubren_todas_las_funciones(dibujos):
    assert sorted(dibujos) == NOMBRES
    assert all(callable(getattr(graficos, nombre)) for nombre in NOMBRES)


def test_el_modulo_no_define_funciones_que_pytest_recogeria():
    assert not [nombre for nombre in dir(graficos) if nombre.startswith("test_")]


# --- colores y leyendas --------------------------------------------------------


def test_las_curvas_usan_los_colores_de_la_paleta_en_orden(observado):
    curvas = {
        f"estrategia {numero}": observado.cumsum() * (numero + 1) for numero in range(4)
    }
    _, ax = graficos.curva_beneficio(curvas)
    colores = [linea.get_color() for linea in ax.lines[: len(curvas)]]
    assert colores == PALETA[: len(curvas)]


def test_la_prediccion_usa_la_paleta_y_lo_real_va_en_gris(observado, previsto):
    predicciones = {f"modelo {numero}": previsto * (numero + 1) for numero in range(3)}
    _, ax = graficos.prediccion_vs_real(observado, predicciones)
    assert ax.lines[0].get_color() == CHROME["tinta_apagada"]
    assert [linea.get_color() for linea in ax.lines[1:]] == PALETA[:3]


def test_la_paleta_no_se_recorre_ciclicamente(observado):
    # Con mas series que colores, repetir uno haria indistinguibles dos curvas:
    # el modulo se niega en lugar de dibujar algo enganioso.
    curvas = {f"e{numero}": observado.cumsum() for numero in range(len(PALETA) + 1)}
    with pytest.raises(ErrorDatos, match="paleta"):
        graficos.curva_beneficio(curvas)


def test_una_sola_serie_no_lleva_leyenda(observado):
    _, ax = graficos.curva_beneficio({"unica": observado.cumsum()})
    assert ax.get_legend() is None
    _, ax = graficos.distribucion_errores({"unica": observado})
    assert ax.get_legend() is None


def test_varias_series_llevan_leyenda(observado):
    curvas = {"a": observado.cumsum(), "b": -observado.cumsum()}
    _, ax = graficos.curva_beneficio(curvas)
    assert ax.get_legend() is not None
    assert [texto.get_text() for texto in ax.get_legend().get_texts()] == ["a", "b"]

    _, ax = graficos.distribucion_errores({"a": observado, "b": observado * 2})
    assert ax.get_legend() is not None


def test_la_prediccion_frente_a_lo_real_siempre_lleva_leyenda(observado, previsto):
    # Aqui hay al menos dos series por construccion, y la gris no se distingue
    # de las demas por la etiqueta del eje.
    _, ax = graficos.prediccion_vs_real(observado, {"unico": previsto})
    etiquetas = [texto.get_text() for texto in ax.get_legend().get_texts()]
    assert etiquetas == ["Real", "unico"]


# --- particiones ---------------------------------------------------------------


def _recuadrada(barra) -> bool:
    """Si la barra lleva el recuadro del color de estado critico."""
    from matplotlib.colors import to_rgba

    return tuple(barra.get_edgecolor()) == to_rgba(ESTADO["critico"])


def test_las_particiones_dibujan_dos_barras_por_particion(tabla_particiones):
    _, ax = graficos.particiones_walk_forward(tabla_particiones)
    assert len(ax.patches) == 2 * len(tabla_particiones)
    etiquetas = [texto.get_text() for texto in ax.get_legend().get_texts()]
    assert etiquetas == ["Entrenamiento", "Prueba"]


def test_las_barras_de_particion_respetan_las_fechas(tabla_particiones):
    from matplotlib.dates import num2date

    _, ax = graficos.particiones_walk_forward(tabla_particiones)
    n = len(tabla_particiones)
    entrenamientos, pruebas = ax.patches[:n], ax.patches[n:]
    for barra, esperado in zip(entrenamientos, tabla_particiones["inicio_entrena"]):
        assert num2date(barra.get_x()).date() == esperado.date()
    for barra, esperado in zip(pruebas, tabla_particiones["inicio_prueba"]):
        assert num2date(barra.get_x()).date() == esperado.date()
    # El tramo de prueba es mas corto que el de entrenamiento en esta tabla.
    assert all(p.get_width() < e.get_width() for e, p in zip(entrenamientos, pruebas))


def test_una_particion_sin_solape_no_se_marca(tabla_particiones):
    _, ax = graficos.particiones_walk_forward(tabla_particiones)
    assert not [texto for texto in ax.texts if "Solape" in texto.get_text()]
    assert not [barra for barra in ax.patches if _recuadrada(barra)]


def test_el_solape_entre_entrenamiento_y_prueba_se_marca(tabla_particiones):
    # Fuga de informacion: la prueba empieza antes de acabar el entrenamiento.
    contaminada = tabla_particiones.copy()
    contaminada.loc[1, "inicio_prueba"] = contaminada.loc[1, "fin_entrena"] - pd.Timedelta(
        days=10
    )
    _, ax = graficos.particiones_walk_forward(contaminada)
    marcas = [texto for texto in ax.texts if "Solape" in texto.get_text()]
    assert len(marcas) == 1
    # Las dos barras de esa particion quedan recuadradas en el color critico.
    assert len([barra for barra in ax.patches if _recuadrada(barra)]) == 2


def test_las_particiones_aceptan_los_nombres_largos_de_columna(tabla_particiones):
    # Es la tabla que produce ParticionWalkForward.describir.
    larga = tabla_particiones.rename(
        columns={
            "inicio_entrena": "inicio_entrenamiento",
            "fin_entrena": "fin_entrenamiento",
        }
    )
    _, ax = graficos.particiones_walk_forward(larga)
    assert len(ax.patches) == 2 * len(larga)


def test_las_particiones_admiten_posiciones_en_lugar_de_fechas():
    tabla = pd.DataFrame(
        {
            "inicio_entrena": [0, 50],
            "fin_entrena": [100, 150],
            "inicio_prueba": [105, 155],
            "fin_prueba": [130, 180],
        }
    )
    _, ax = graficos.particiones_walk_forward(tabla)
    assert ax.patches[0].get_x() == 0.0
    assert ax.get_xlabel() == "Observacion"


def test_un_tramo_de_una_sola_observacion_sigue_siendo_visible():
    tabla = pd.DataFrame(
        {
            "inicio_entrena": [0],
            "fin_entrena": [100],
            "inicio_prueba": [110],
            "fin_prueba": [110],
        }
    )
    _, ax = graficos.particiones_walk_forward(tabla)
    assert ax.patches[1].get_width() > 0.0


def test_las_particiones_exigen_las_columnas_del_tramo(tabla_particiones):
    with pytest.raises(ErrorDatos, match="inicio_prueba"):
        graficos.particiones_walk_forward(tabla_particiones.drop(columns=["inicio_prueba"]))


def test_una_tabla_de_particiones_vacia_es_un_error():
    with pytest.raises(ErrorDatos, match="ninguna particion"):
        graficos.particiones_walk_forward(pd.DataFrame())


# --- dispersion ----------------------------------------------------------------


def test_la_dispersion_anota_la_correlacion(observado, previsto):
    _, ax = graficos.dispersion_prediccion(observado, previsto)
    textos = " ".join(texto.get_text() for texto in ax.texts)
    assert "correlacion" in textos
    assert "pendiente" in textos


def test_la_dispersion_mide_el_encogimiento_del_modelo():
    # Caso con solucion exacta: la prediccion es la mitad del valor real, asi
    # que la correlacion es 1 y la pendiente estimada es 0,5 exactamente.
    reales = pd.Series(np.linspace(-0.05, 0.05, 41))
    prediccion = reales * 0.5
    _, ax = graficos.dispersion_prediccion(reales, prediccion)
    anotacion = " ".join(texto.get_text() for texto in ax.texts)
    assert "correlacion 1.00" in anotacion
    assert "pendiente 0.50" in anotacion


def test_la_dispersion_dibuja_la_diagonal_y_la_recta_ajustada(observado, previsto):
    _, ax = graficos.dispersion_prediccion(observado, previsto)
    assert len(ax.lines) == 2
    diagonal, ajuste = ax.lines
    assert diagonal.get_linestyle() == "--"
    # La diagonal pasa por el origen con pendiente unitaria.
    x, y = diagonal.get_xdata(), diagonal.get_ydata()
    assert np.allclose(x, y)
    assert ajuste.get_color() == PALETA[1]


def test_la_dispersion_comparte_escala_en_los_dos_ejes(observado, previsto):
    # Sin ejes iguales la diagonal no se veria a 45 grados y la comparacion
    # visual enganiaria.
    _, ax = graficos.dispersion_prediccion(observado, previsto)
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_aspect() == 1.0


def test_la_dispersion_rechaza_una_prediccion_constante(observado):
    constante = pd.Series(0.01, index=observado.index)
    with pytest.raises(ErrorDatos, match="constante"):
        graficos.dispersion_prediccion(observado, constante)


def test_la_dispersion_cruza_por_fecha_y_no_por_posicion(observado, previsto):
    # Un hueco en la prediccion no debe desplazar el emparejamiento.
    con_hueco = previsto.drop(previsto.index[[5, 60, 200]])
    _, ax = graficos.dispersion_prediccion(observado, con_hueco)
    assert ax.collections[0].get_offsets().shape[0] == len(con_hueco)


def test_sin_fechas_comunes_la_dispersion_avisa(observado, previsto):
    desplazada = previsto.copy()
    desplazada.index = previsto.index + pd.Timedelta(hours=7)
    with pytest.raises(ErrorDatos, match="ninguna fecha"):
        graficos.dispersion_prediccion(observado, desplazada)


def test_longitudes_distintas_sin_indice_son_un_error():
    with pytest.raises(ErrorDatos, match="observaciones"):
        graficos.dispersion_prediccion(np.zeros(10), np.zeros(9))


# --- importancia ---------------------------------------------------------------


def test_la_importancia_dibuja_las_n_mas_importantes():
    importancia = pd.Series({f"v{numero}": float(numero) for numero in range(30)})
    _, ax = graficos.importancia_variables(importancia, n=7)
    assert len(ax.patches) == 7
    # De arriba abajo, de mayor a menor: la mas importante arriba.
    etiquetas = [texto.get_text() for texto in ax.get_yticklabels()]
    assert etiquetas[-1] == "v29"
    assert etiquetas[0] == "v23"


def test_la_importancia_ordena_por_valor_absoluto():
    # Un coeficiente muy negativo pesa tanto como uno muy positivo.
    importancia = pd.Series({"grande_negativa": -5.0, "pequenia": 0.1, "grande": 4.0})
    _, ax = graficos.importancia_variables(importancia, n=2)
    etiquetas = [texto.get_text() for texto in ax.get_yticklabels()]
    assert set(etiquetas) == {"grande_negativa", "grande"}


def test_la_importancia_con_signos_mezclados_distingue_los_colores():
    importancia = pd.Series({"a": 1.0, "b": -0.8, "c": 0.3})
    _, ax = graficos.importancia_variables(importancia)
    colores = {barra.get_facecolor() for barra in ax.patches}
    assert len(colores) == 2
    assert ax.get_legend() is not None


def test_la_importancia_toda_positiva_no_necesita_leyenda():
    _, ax = graficos.importancia_variables(pd.Series({"a": 1.0, "b": 0.5}))
    assert ax.get_legend() is None


def test_pedir_mas_variables_de_las_que_hay_dibuja_las_que_hay():
    _, ax = graficos.importancia_variables(pd.Series({"a": 1.0, "b": 0.5}), n=20)
    assert len(ax.patches) == 2


def test_la_importancia_exige_un_numero_de_barras_positivo():
    with pytest.raises(ErrorDatos, match="al menos 1"):
        graficos.importancia_variables(pd.Series({"a": 1.0}), n=0)


# --- curva de beneficio y errores ----------------------------------------------


def test_la_curva_de_beneficio_marca_el_cero(observado):
    _, ax = graficos.curva_beneficio({"a": observado.cumsum()})
    referencias = [
        linea
        for linea in ax.lines
        if len(linea.get_ydata()) == 2 and np.allclose(linea.get_ydata(), 0.0)
    ]
    assert len(referencias) == 1


def test_una_curva_de_beneficio_vacia_es_un_error():
    with pytest.raises(ErrorDatos, match="vacio"):
        graficos.curva_beneficio({})


def test_los_errores_comparten_intervalos_entre_modelos(observado):
    # Dos histogramas del mismo grafico con intervalos distintos no se pueden
    # comparar, por mucho que lo parezca.
    errores = {"estrecho": observado * 0.1, "ancho": observado * 3.0}
    _, ax = graficos.distribucion_errores(errores)
    assert len(ax.patches) == 2  # un contorno por modelo
    extremos = [barra.get_path().get_extents().intervalx for barra in ax.patches]
    assert np.allclose(extremos[0], extremos[1])
    assert len(ax.get_legend().get_texts()) == 2


def test_los_errores_marcan_el_cero(observado):
    _, ax = graficos.distribucion_errores({"a": observado + 0.05})
    verticales = [
        linea
        for linea in ax.lines
        if len(linea.get_xdata()) == 2 and np.allclose(linea.get_xdata(), 0.0)
    ]
    assert len(verticales) == 1


def test_una_serie_de_errores_sin_datos_validos_es_un_error(observado):
    vacia = pd.Series(np.nan, index=observado.index)
    with pytest.raises(ErrorDatos, match="valido"):
        graficos.distribucion_errores({"a": observado, "roto": vacia})


# --- acierto por magnitud ------------------------------------------------------


@pytest.mark.parametrize("n_grupos", [2, 3, 4, 5, 8])
def test_el_acierto_produce_exactamente_los_grupos_pedidos(observado, previsto, n_grupos):
    _, ax = graficos.acierto_por_magnitud(observado, previsto, n_grupos=n_grupos)
    assert len(ax.patches) == n_grupos
    assert len(ax.get_xticklabels()) == n_grupos


def test_los_grupos_no_colapsan_con_predicciones_repetidas():
    # Con cuantiles, las predicciones repetidas devolverian menos grupos de los
    # pedidos sin avisar; el reparto por orden no.
    reales = pd.Series(np.tile([1.0, -1.0], 10))
    prediccion = pd.Series(np.ones(20))
    _, ax = graficos.acierto_por_magnitud(reales, prediccion, n_grupos=5)
    assert len(ax.patches) == 5


def test_el_acierto_por_grupo_es_el_esperado():
    # Caso a mano: el modelo se equivoca en las diez predicciones mas pequenias
    # y acierta en las diez mas grandes.
    magnitudes = np.arange(1, 21, dtype=float)
    prediccion = pd.Series(magnitudes)
    reales = pd.Series(np.where(magnitudes <= 10, -1.0, 1.0))
    _, ax = graficos.acierto_por_magnitud(reales, prediccion, n_grupos=4)
    alturas = [barra.get_height() for barra in ax.patches]
    assert alturas == [0.0, 0.0, 1.0, 1.0]


def test_un_acierto_plano_se_dibuja_plano():
    # La confianza del modelo no informa: todos los grupos aciertan lo mismo.
    magnitudes = np.arange(1, 21, dtype=float)
    prediccion = pd.Series(magnitudes)
    reales = pd.Series(np.tile([1.0, -1.0], 10))
    _, ax = graficos.acierto_por_magnitud(reales, prediccion, n_grupos=5)
    alturas = [barra.get_height() for barra in ax.patches]
    assert alturas == [0.5] * 5


def test_una_prediccion_de_cero_cuenta_como_fallo():
    reales = pd.Series([1.0, 1.0, 1.0, 1.0])
    prediccion = pd.Series([0.0, 0.0, 2.0, 3.0])
    _, ax = graficos.acierto_por_magnitud(reales, prediccion, n_grupos=2)
    assert [barra.get_height() for barra in ax.patches] == [0.0, 1.0]


def test_el_acierto_se_dibuja_en_la_escala_completa(observado, previsto):
    # El eje va de 0 a 1: recortarlo exageraria diferencias de dos puntos.
    _, ax = graficos.acierto_por_magnitud(observado, previsto)
    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.yaxis.get_major_formatter()(0.55).endswith("%")


def test_el_acierto_dibuja_la_linea_del_azar(observado, previsto):
    _, ax = graficos.acierto_por_magnitud(observado, previsto)
    referencias = [linea for linea in ax.lines if np.allclose(linea.get_ydata(), 0.5)]
    assert len(referencias) == 1


def test_el_acierto_exige_al_menos_dos_grupos(observado, previsto):
    with pytest.raises(ErrorDatos, match="al menos 2"):
        graficos.acierto_por_magnitud(observado, previsto, n_grupos=1)


def test_no_se_pueden_formar_mas_grupos_que_observaciones():
    with pytest.raises(ErrorDatos, match="pares validos"):
        graficos.acierto_por_magnitud(pd.Series([1.0, -1.0]), pd.Series([1.0, 1.0]), n_grupos=5)


# --- historial de entrenamiento ------------------------------------------------


def test_el_historial_dibuja_las_dos_perdidas(historial):
    _, ax = graficos.historial_entrenamiento(historial)
    assert [linea.get_color() for linea in ax.lines[:2]] == PALETA[:2]
    etiquetas = [texto.get_text() for texto in ax.get_legend().get_texts()]
    assert etiquetas == ["Entrenamiento", "Validacion"]


def test_el_historial_marca_la_epoca_de_menor_perdida_de_validacion(historial):
    _, ax = graficos.historial_entrenamiento(historial)
    esperada = int(historial["perdida_validacion"].idxmin())
    textos = [texto.get_text() for texto in ax.texts]
    assert f"Mejor epoca: {esperada}" in textos


def test_el_historial_sin_validacion_no_lleva_leyenda(historial):
    solo_entrena = historial[["perdida_entrenamiento"]]
    _, ax = graficos.historial_entrenamiento(solo_entrena)
    assert ax.get_legend() is None
    assert len(ax.lines) == 1


def test_el_historial_acepta_los_nombres_breves(historial):
    breve = historial.rename(
        columns={
            "perdida_entrenamiento": "entrenamiento",
            "perdida_validacion": "validacion",
        }
    )
    _, ax = graficos.historial_entrenamiento(breve)
    assert len(ax.lines) == 3  # las dos curvas y la vertical de la mejor epoca


def test_el_historial_exige_la_perdida_de_entrenamiento():
    with pytest.raises(ErrorDatos, match="perdida_entrenamiento"):
        graficos.historial_entrenamiento(pd.DataFrame({"otra_cosa": [1.0, 0.5]}))


def test_un_historial_vacio_es_un_error():
    with pytest.raises(ErrorDatos, match="ninguna epoca"):
        graficos.historial_entrenamiento(pd.DataFrame({"perdida_entrenamiento": []}))
