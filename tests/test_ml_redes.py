"""Pruebas de las redes secuenciales de :mod:`icm.ml.redes`.

Se contrastan tres cosas distintas y de dificultad creciente:

1. Que la red *aprende*: sobre una relacion sintetica con solucion conocida
   (el objetivo es la suma exacta de la ventana) tiene que batir con holgura a
   predecir la media, que es el unico competidor honesto.
2. Que no mira al futuro: alterar una fecha posterior no puede cambiar la
   prediccion de una fecha anterior, y el tramo de validacion interna no puede
   influir ni en la normalizacion ni en los pesos.
3. Que es reproducible y clonable, que es lo que permite meterla en una
   validacion hacia delante sin adaptadores.

Todas las configuraciones son minusculas a proposito: lo que se comprueba son
propiedades del codigo, no el rendimiento de una red grande, y la bateria
completa del repositorio tiene que seguir tardando segundos.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

# Las redes de estas pruebas son minusculas (lotes de 32 y una decena de
# unidades), y con tensores tan pequenios repartir cada operacion entre los
# hilos de la maquina cuesta mas de lo que ahorra: medido, un solo hilo va unas
# tres veces mas rapido. Se fija aqui, en las pruebas, y no en el modulo, porque
# es una decision sobre este proceso y no sobre la biblioteca.
torch.set_num_threads(1)

from dataclasses import replace  # noqa: E402

from sklearn.base import clone  # noqa: E402

from icm.comun import ErrorDatos  # noqa: E402
from icm.ml import redes  # noqa: E402
from icm.ml.redes import (  # noqa: E402
    ARQUITECTURAS,
    ConfiguracionRed,
    RedSecuencial,
    torch_disponible,
)
from icm.ml.validacion import ParticionWalkForward, validacion_walk_forward  # noqa: E402

N_OBSERVACIONES = 500
VENTANA = 5


# --- datos ---------------------------------------------------------------------


def _datos_suma_ventana(
    n: int = N_OBSERVACIONES,
    ventana: int = VENTANA,
    semilla: int = 20240601,
) -> tuple[pd.DataFrame, pd.Series]:
    """Serie con un objetivo que es la suma exacta de la ventana pasada de ``x``.

    Se generan ``ventana - 1`` valores extra al principio y despues se recortan
    de ``X``: asi la suma de la primera fecha que la red puede usar como
    objetivo esta contenida por completo en las filas de ``X``, y la relacion
    que debe aprender es exacta, no aproximada. Se aniade una segunda columna de
    ruido puro para comprobar que la red maneja varias variables y no se
    desorienta con las que no informan.
    """
    generador = np.random.default_rng(semilla)
    crudos = generador.normal(size=n + ventana - 1)
    sumas = pd.Series(crudos).rolling(ventana).sum().to_numpy()[ventana - 1:]
    indice = pd.bdate_range("2018-01-01", periods=n, name="fecha")
    marco = pd.DataFrame(
        {"x": crudos[ventana - 1:], "ruido": generador.normal(size=n)},
        index=indice,
    )
    return marco, pd.Series(sumas, index=indice, name="objetivo")


def _datos_sin_senial(
    n: int = 300,
    semilla: int = 20240602,
) -> tuple[pd.DataFrame, pd.Series]:
    """Objetivo independiente de las variables: no hay nada que aprender.

    Es el caso en el que la parada temprana *tiene* que actuar: la perdida de
    validacion deja de mejorar en cuanto la red empieza a memorizar el ruido
    del tramo de entrenamiento.
    """
    generador = np.random.default_rng(semilla)
    indice = pd.bdate_range("2018-01-01", periods=n, name="fecha")
    marco = pd.DataFrame(
        {"a": generador.normal(size=n), "b": generador.normal(size=n)},
        index=indice,
    )
    return marco, pd.Series(generador.normal(size=n), index=indice, name="objetivo")


def _configuracion_aprendizaje() -> ConfiguracionRed:
    """Red minuscula pero suficiente para la suma de la ventana.

    ``abandono`` a cero porque la relacion es determinista: no hay ruido del que
    protegerse y el abandono solo aniadiria varianza. ``paciencia`` igual a
    ``epocas`` para que la parada temprana no intervenga y el historial tenga
    las 20 epocas completas, que es lo que otra prueba comprueba.
    """
    return ConfiguracionRed(
        arquitectura="lstm",
        ventana=VENTANA,
        unidades=16,
        capas=1,
        abandono=0.0,
        epocas=20,
        tamano_lote=32,
        tasa_aprendizaje=0.05,
        paciencia=20,
        proporcion_validacion=0.2,
        semilla=7,
    )


@pytest.fixture(scope="module")
def datos_suma() -> tuple[pd.DataFrame, pd.Series]:
    return _datos_suma_ventana()


@pytest.fixture(scope="module")
def red_ajustada(datos_suma: tuple[pd.DataFrame, pd.Series]) -> RedSecuencial:
    """Red entrenada una sola vez y compartida por varias pruebas."""
    X, y = datos_suma
    return RedSecuencial(_configuracion_aprendizaje()).ajustar(X, y)


# --- aprendizaje ---------------------------------------------------------------


def test_torch_disponible_es_cierto_cuando_se_puede_importar() -> None:
    assert torch_disponible() is True


def test_la_red_aprende_la_suma_de_la_ventana(
    red_ajustada: RedSecuencial,
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Debe batir con holgura a la prediccion ingenua de la media.

    La referencia es la desviacion tipica del objetivo, que es exactamente el
    error que comete quien predice siempre la media. El umbral (un 35 % de ese
    error) queda muy por encima de lo que la red consigue en realidad, de modo
    que la prueba no depende de una semilla afortunada.
    """
    X, y = datos_suma
    predicciones = red_ajustada.predecir(X)
    utiles = ~np.isnan(predicciones)
    reales = y.to_numpy(dtype=float)[utiles]

    error = float(np.sqrt(np.mean((predicciones[utiles] - reales) ** 2)))
    error_de_la_media = float(np.sqrt(np.mean((reales - reales.mean()) ** 2)))
    assert error < 0.35 * error_de_la_media


def test_el_historial_registra_las_epocas_y_la_perdida_de_entrenamiento_baja(
    red_ajustada: RedSecuencial,
) -> None:
    historial = red_ajustada.historial
    configuracion = red_ajustada.configuracion

    assert list(historial.index) == list(range(1, configuracion.epocas + 1))
    assert list(historial.columns) == ["perdida_entrenamiento", "perdida_validacion"]
    assert np.isfinite(historial.to_numpy()).all()

    perdida = historial["perdida_entrenamiento"]
    assert perdida.iloc[-1] < 0.5 * perdida.iloc[0]


@pytest.mark.parametrize("arquitectura", ARQUITECTURAS)
def test_las_tres_arquitecturas_ajustan_y_predicen(arquitectura: str) -> None:
    X, y = _datos_suma_ventana(n=200, ventana=4)
    configuracion = ConfiguracionRed(
        arquitectura=arquitectura,
        ventana=4,
        unidades=8,
        epocas=3,
        tamano_lote=32,
        paciencia=3,
        semilla=11,
    )
    red = RedSecuencial(configuracion).ajustar(X, y)
    predicciones = red.predecir(X)

    assert red.ajustada
    assert len(predicciones) == len(X)
    assert np.isfinite(predicciones[3:]).all()


# --- reproducibilidad y clonado -------------------------------------------------


def test_la_misma_semilla_da_exactamente_las_mismas_predicciones(
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Dos redes independientes con la misma semilla deben coincidir al bit.

    Si no coincidieran, cualquier comparacion entre dos configuraciones estaria
    midiendo en parte la suerte de la inicializacion.
    """
    X, y = datos_suma
    configuracion = replace(_configuracion_aprendizaje(), epocas=3, paciencia=3)
    primera = RedSecuencial(configuracion).ajustar(X, y).predecir(X)
    segunda = RedSecuencial(configuracion).ajustar(X, y).predecir(X)

    assert np.array_equal(primera[VENTANA - 1:], segunda[VENTANA - 1:])


def test_una_semilla_distinta_cambia_las_predicciones(
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Contrapartida de la prueba anterior: la semilla tiene que hacer algo.

    Sin esta comprobacion, un ``predecir`` que devolviese una constante pasaria
    la prueba de reproducibilidad sin merecerlo.
    """
    X, y = datos_suma
    configuracion = replace(_configuracion_aprendizaje(), epocas=3, paciencia=3)
    primera = RedSecuencial(configuracion).ajustar(X, y).predecir(X)
    otra = RedSecuencial(replace(configuracion, semilla=123)).ajustar(X, y).predecir(X)

    assert not np.array_equal(primera[VENTANA - 1:], otra[VENTANA - 1:])


def test_clone_de_sklearn_reproduce_la_configuracion_sin_el_ajuste(
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """``sklearn.base.clone`` es lo que usa la validacion hacia delante.

    Tiene que devolver un objeto con los mismos hiperparametros y **sin** los
    pesos: reutilizar una red ya entrenada particion tras particion la haria
    predecir fechas que ya habia visto.
    """
    X, y = datos_suma
    original = RedSecuencial(
        replace(_configuracion_aprendizaje(), arquitectura="gru", epocas=2, paciencia=2)
    ).ajustar(X, y)

    copia = clone(original)

    assert isinstance(copia, RedSecuencial)
    assert copia.get_params() == original.get_params()
    assert not copia.ajustada
    assert original.ajustada


def test_set_params_cambia_la_configuracion_y_descarta_el_ajuste(
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = datos_suma
    red = RedSecuencial(
        replace(_configuracion_aprendizaje(), epocas=2, paciencia=2)
    ).ajustar(X, y)

    red.set_params(arquitectura="tcn", unidades=4)

    assert red.configuracion.arquitectura == "tcn"
    assert red.configuracion.unidades == 4
    assert not red.ajustada


# --- ausencia de fugas del futuro ----------------------------------------------


def test_predecir_devuelve_nan_en_las_primeras_filas_y_una_prediccion_por_fila(
    red_ajustada: RedSecuencial,
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Las ``ventana - 1`` primeras fechas no tienen pasado suficiente."""
    X, _ = datos_suma
    predicciones = red_ajustada.predecir(X)

    assert predicciones.shape == (len(X),)
    assert np.isnan(predicciones[: VENTANA - 1]).all()
    assert np.isfinite(predicciones[VENTANA - 1:]).all()


def test_una_tabla_mas_corta_que_la_ventana_devuelve_solo_nan(
    red_ajustada: RedSecuencial,
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Sin historia no hay prediccion, y eso se dice con NaN, no con un error.

    Es lo que ocurre con un tramo de prueba muy corto en una validacion hacia
    delante: interrumpir el experimento seria peor que declarar el hueco.
    """
    X, _ = datos_suma
    predicciones = red_ajustada.predecir(X.iloc[: VENTANA - 1])

    assert len(predicciones) == VENTANA - 1
    assert np.isnan(predicciones).all()


def test_la_ventana_no_mira_al_futuro(
    red_ajustada: RedSecuencial,
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Alterar la ultima fecha solo puede cambiar la prediccion de esa fecha.

    Es la comprobacion directa de que la ventana se construye hacia atras. Con
    una ventana centrada, tocar la ultima fila cambiaria tambien las
    predicciones de las fechas anteriores, que son las que el modelo tendria que
    haber hecho sin conocerla.
    """
    X, _ = datos_suma
    alterada = X.copy()
    alterada.iloc[-1] = alterada.iloc[-1] + 10.0

    original = red_ajustada.predecir(X)
    perturbada = red_ajustada.predecir(alterada)

    assert np.array_equal(original[VENTANA - 1: -1], perturbada[VENTANA - 1: -1])
    assert original[-1] != perturbada[-1]


def test_la_normalizacion_y_el_entrenamiento_ignoran_el_tramo_de_validacion() -> None:
    """Cambiar el tramo final no puede alterar los pesos ni la normalizacion.

    Se ajusta con una sola epoca para que la perdida de validacion no pueda
    influir por la via legitima (la parada temprana) y quede aislado lo unico
    que se quiere medir: si la media y la desviacion se estimaron con toda la
    serie, multiplicar por 40 el ultimo tramo cambiaria la escala de *todas* las
    observaciones y, con ella, las predicciones de las fechas iniciales.

    La segunda mitad de la prueba es el control: tocar el principio de la serie
    si tiene que cambiar las predicciones. Sin ella, una red que devolviese
    siempre lo mismo pasaria la primera mitad sin merecerlo.
    """
    X, y = _datos_suma_ventana(n=200, ventana=4)
    configuracion = ConfiguracionRed(
        ventana=4, unidades=8, epocas=1, paciencia=1, tamano_lote=32, semilla=5
    )
    # Con el 20 % de validacion y el embargo, el entrenamiento acaba por la fila
    # 158 de 200; se altera desde la 180 para quedar holgadamente dentro del
    # tramo reservado y que la prueba no dependa del redondeo de la particion.
    corte = 180

    cola_alterada_X, cola_alterada_y = X.copy(), y.copy()
    cola_alterada_X.iloc[corte:] *= 40.0
    cola_alterada_y.iloc[corte:] += 500.0

    cabeza_alterada_X = X.copy()
    cabeza_alterada_X.iloc[:10] *= 40.0

    referencia = RedSecuencial(configuracion).ajustar(X, y).predecir(X)
    con_cola = RedSecuencial(configuracion).ajustar(cola_alterada_X, cola_alterada_y).predecir(X)
    con_cabeza = RedSecuencial(configuracion).ajustar(cabeza_alterada_X, y).predecir(X)

    assert np.array_equal(referencia[3:], con_cola[3:])
    assert not np.array_equal(referencia[3:], con_cabeza[3:])


# --- parada temprana -----------------------------------------------------------


def test_la_parada_temprana_corta_antes_del_maximo_y_restaura_los_mejores_pesos() -> None:
    """Sin senial que aprender, la validacion deja de mejorar y se corta.

    Se comprueban las tres piezas de la regla:

    * se para antes de agotar ``epocas``;
    * se para exactamente ``paciencia`` epocas despues de la mejor, que es la
      definicion de la paciencia;
    * los pesos que quedan son los de la mejor epoca y no los de la ultima.
      Esto se verifica volviendo a entrenar con ``epocas`` truncadas en la mejor:
      con la misma semilla y los mismos datos la trayectoria del optimizador es
      identica, asi que esa segunda red termina exactamente en los pesos que la
      primera deberia haber restaurado. Que la igualdad no sea trivial lo
      garantiza la ultima comprobacion: la perdida de validacion de la ultima
      epoca es estrictamente peor que la de la mejor, luego los pesos de una y
      otra son necesariamente distintos y la restauracion si ha hecho algo.
    """
    X, y = _datos_sin_senial()
    configuracion = ConfiguracionRed(
        arquitectura="lstm",
        ventana=VENTANA,
        unidades=16,
        epocas=20,
        tamano_lote=32,
        tasa_aprendizaje=0.02,
        paciencia=2,
        semilla=13,
    )

    red = RedSecuencial(configuracion).ajustar(X, y)
    historial = red.historial
    mejor_epoca = int(historial["perdida_validacion"].idxmin())

    assert len(historial) < configuracion.epocas
    assert historial.index[-1] - mejor_epoca == configuracion.paciencia

    predicciones = red.predecir(X)
    en_la_mejor = (
        RedSecuencial(replace(configuracion, epocas=mejor_epoca, paciencia=mejor_epoca + 1))
        .ajustar(X, y)
        .predecir(X)
    )

    assert np.array_equal(predicciones[VENTANA - 1:], en_la_mejor[VENTANA - 1:])
    perdidas = historial["perdida_validacion"]
    assert perdidas.iloc[-1] > perdidas.min()


def test_sin_validacion_interna_la_parada_temprana_vigila_el_entrenamiento() -> None:
    """Con ``proporcion_validacion`` a cero el ajuste sigue funcionando.

    Es el modo para series muy cortas: no hay tramo que reservar, asi que las
    dos columnas del historial coinciden.
    """
    X, y = _datos_suma_ventana(n=120, ventana=4)
    configuracion = ConfiguracionRed(
        ventana=4,
        unidades=8,
        epocas=4,
        paciencia=4,
        tamano_lote=32,
        proporcion_validacion=0.0,
        semilla=3,
    )
    red = RedSecuencial(configuracion).ajustar(X, y)
    historial = red.historial

    assert np.array_equal(
        historial["perdida_entrenamiento"].to_numpy(),
        historial["perdida_validacion"].to_numpy(),
    )
    assert np.isfinite(red.predecir(X)[3:]).all()


# --- integracion con la validacion hacia delante -------------------------------


def test_la_red_entra_en_la_validacion_walk_forward_sin_adaptadores() -> None:
    """La interfaz ``ajustar`` / ``predecir`` es la que espera el modulo vecino.

    Se comprueba tambien la consecuencia practica de los NaN del arranque: cada
    tramo de prueba empieza con ``ventana - 1`` fechas sin prediccion, asi que
    el numero de predicciones utiles es el esperado y no una sorpresa.
    """
    X, y = _datos_suma_ventana(n=300, ventana=4)
    red = RedSecuencial(
        ConfiguracionRed(ventana=4, unidades=8, epocas=2, paciencia=2, semilla=17)
    )
    particion = ParticionWalkForward(
        n_particiones=2, tamano_prueba=60, embargo=4, tamano_entrenamiento_minimo=100
    )

    resultado = validacion_walk_forward(red, X, y, particion)

    assert len(resultado.predicciones) == 120
    assert resultado.predicciones.isna().sum() == 2 * 3
    assert len(resultado.modelos) == 2
    assert not red.ajustada  # la validacion no toca el objeto que recibe


# --- errores -------------------------------------------------------------------


def test_importar_el_modulo_no_carga_pytorch() -> None:
    """La promesa central del modulo: ``import`` sin los 3 GB de PyTorch.

    Se comprueba en un interprete nuevo porque en este ya esta cargado: lo
    importa ``pytest.importorskip`` al principio del fichero. Es la unica
    prueba que abre un subproceso, y lo hace justo porque no puede importar
    PyTorch.
    """
    salida = subprocess.run(
        [sys.executable, "-c", "import icm.ml.redes, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert salida.stdout.strip() == "False"


def test_sin_pytorch_el_ajuste_explica_como_instalarlo(monkeypatch: pytest.MonkeyPatch) -> None:
    """El mensaje debe traer el comando, no un ImportError a secas."""
    monkeypatch.setattr(redes, "torch_disponible", lambda: False)
    X, y = _datos_suma_ventana(n=30, ventana=4)

    with pytest.raises(ErrorDatos, match="pip install torch"):
        RedSecuencial(ventana=4, epocas=1).ajustar(X, y)


def test_una_arquitectura_desconocida_se_rechaza() -> None:
    with pytest.raises(ErrorDatos, match="Arquitectura 'mlp' desconocida"):
        ConfiguracionRed(arquitectura="mlp")


@pytest.mark.parametrize(
    "parametros",
    [
        {"ventana": 0},
        {"unidades": 0},
        {"capas": 0},
        {"abandono": 1.0},
        {"epocas": 0},
        {"tamano_lote": 0},
        {"tasa_aprendizaje": 0.0},
        {"paciencia": 0},
        {"proporcion_validacion": 1.0},
    ],
)
def test_los_hiperparametros_fuera_de_rango_se_rechazan(parametros: dict) -> None:
    with pytest.raises(ErrorDatos):
        ConfiguracionRed(**parametros)


def test_un_hiperparametro_inexistente_se_rechaza() -> None:
    with pytest.raises(ErrorDatos, match="desconocidos: tasa_de_aprendizaje"):
        RedSecuencial(tasa_de_aprendizaje=0.1)


def test_una_ventana_mayor_que_la_serie_se_rechaza() -> None:
    X, y = _datos_suma_ventana(n=8, ventana=4)
    with pytest.raises(ErrorDatos, match="Hacen falta al menos 21 filas"):
        RedSecuencial(ventana=20, epocas=1).ajustar(X, y)


def test_los_ausentes_se_rechazan_en_lugar_de_rellenarse() -> None:
    """Rellenar por dentro inventaria historia; descartar desalinearia."""
    X, y = _datos_suma_ventana(n=60, ventana=4)
    X_con_hueco = X.copy()
    X_con_hueco.iloc[10, 0] = np.nan

    with pytest.raises(ErrorDatos, match="valores ausentes o infinitos"):
        RedSecuencial(ventana=4, epocas=1).ajustar(X_con_hueco, y)
    with pytest.raises(ErrorDatos, match="valores ausentes o infinitos"):
        RedSecuencial(ventana=4, epocas=1).ajustar(X, y.mask(y.index == y.index[5]))


def test_x_e_y_desalineados_se_rechazan() -> None:
    X, y = _datos_suma_ventana(n=60, ventana=4)

    with pytest.raises(ErrorDatos, match="filas e 'y' tiene"):
        RedSecuencial(ventana=4, epocas=1).ajustar(X, y.iloc[:-1])
    with pytest.raises(ErrorDatos, match="no comparten indice"):
        RedSecuencial(ventana=4, epocas=1).ajustar(
            X, pd.Series(y.to_numpy(), index=y.index + pd.Timedelta(days=1))
        )


def test_predecir_o_consultar_el_historial_sin_ajustar_falla() -> None:
    red = RedSecuencial(ventana=4)
    X, _ = _datos_suma_ventana(n=30, ventana=4)

    with pytest.raises(ErrorDatos, match="no esta ajustada"):
        red.predecir(X)
    with pytest.raises(ErrorDatos, match="no hay historial"):
        _ = red.historial


def test_predecir_sin_una_columna_del_ajuste_falla(
    red_ajustada: RedSecuencial,
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = datos_suma
    with pytest.raises(ErrorDatos, match="Faltan en 'X' columnas"):
        red_ajustada.predecir(X.drop(columns=["ruido"]))


def test_predecir_no_depende_del_orden_de_las_columnas(
    red_ajustada: RedSecuencial,
    datos_suma: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Las capas multiplican por posicion, asi que hay que reordenar por nombre."""
    X, _ = datos_suma
    original = red_ajustada.predecir(X)
    permutada = red_ajustada.predecir(X[["ruido", "x"]])

    assert np.array_equal(original[VENTANA - 1:], permutada[VENTANA - 1:])


def test_el_alcance_de_la_tcn_crece_de_forma_exponencial_con_las_capas() -> None:
    """Sirve para detectar una ventana mucho mayor que lo que la TCN alcanza a ver."""
    assert ConfiguracionRed(arquitectura="tcn", capas=1).alcance_tcn == 3
    assert ConfiguracionRed(arquitectura="tcn", capas=2).alcance_tcn == 7
    assert ConfiguracionRed(arquitectura="tcn", capas=5).alcance_tcn == 63
