"""Pruebas de los estimadores y de la comparacion contra la referencia.

La prueba que da valor al fichero entero es
:func:`test_con_ruido_puro_ningun_modelo_bate_a_la_referencia`. Todas las demas
comprueban que el codigo hace lo que dice; esa comprueba que **no** hace lo que
no deberia poder hacer. Si un modelo bate a "predecir cero" sobre ruido
independiente, no es que el modelo sea bueno: es que se ha colado informacion
del futuro por alguna parte, y da lo mismo lo bien que se lea el resto.

Por eso se contrasta con varias muestras de ruido y no con una: la conclusion
no puede depender de la semilla que haya tocado. Las combinaciones elegidas se
han comprobado sobre doce semillas independientes sin un solo falso positivo,
de modo que un fallo aqui es una senal, no mala suerte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from icm.comun import ErrorDatos
from icm.ml.modelos import (
    MODELOS,
    NOMBRE_REFERENCIA,
    PASO_ESCALADO,
    PASO_MODELO,
    ParticionSklearn,
    ajustar_y_predecir,
    buscar_hiperparametros,
    comparar_modelos,
    crear_modelo,
    importancia_variables,
)
from icm.ml.validacion import ParticionWalkForward

N_OBSERVACIONES = 400
CORTE = 300

#: Modelos con escalado obligatorio: los que dependen de la escala de cada
#: columna, es decir, todos los que penalizan coeficientes.
LINEALES = ("ridge", "lasso", "elastic_net")

#: Modelos que no lo necesitan: parten por umbrales sobre cada variable, asi
#: que cualquier transformacion monotona da los mismos cortes.
ARBOLES = ("arbol_potenciado", "bosque")


# --- datos ---------------------------------------------------------------------


def _indice(n: int = N_OBSERVACIONES) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-02", periods=n, name="fecha")


def datos_ruido(semilla: int, n: int = N_OBSERVACIONES, n_variables: int = 5):
    """Variables y objetivo independientes: no hay nada que aprender.

    El objetivo tiene la escala de un rendimiento logaritmico diario (2 %), para
    que las penalizaciones por defecto de los modelos trabajen en el orden de
    magnitud en el que se van a usar de verdad.
    """
    generador = np.random.default_rng(semilla)
    indice = _indice(n)
    X = pd.DataFrame(
        generador.standard_normal((n, n_variables)),
        index=indice,
        columns=[f"x{i}" for i in range(n_variables)],
    )
    y = pd.Series(generador.standard_normal(n) * 0.02, index=indice, name="rend_futuro_1")
    return X, y


@pytest.fixture(scope="module")
def particion() -> ParticionWalkForward:
    """Particion hacia delante, corta pero honesta, para que las pruebas vuelen."""
    return ParticionWalkForward(
        n_particiones=3,
        tamano_prueba=60,
        tamano_entrenamiento_minimo=150,
        embargo=1,
    )


@pytest.fixture(scope="module")
def datos_lineales() -> tuple[pd.DataFrame, pd.Series]:
    """Relacion lineal fuerte y conocida: ``y = 0.02*a - 0.01*b + ruido pequenio``.

    Dos variables relevantes y dos que no lo son, con el ruido diez veces menor
    que la senal. Es el caso con solucion analitica: se sabe que coeficientes
    tiene que recuperar el modelo y se sabe que la referencia debe perder.
    """
    generador = np.random.default_rng(1)
    indice = _indice()
    X = pd.DataFrame(
        generador.standard_normal((N_OBSERVACIONES, 4)),
        index=indice,
        columns=["a", "b", "c", "d"],
    )
    y = 0.02 * X["a"] - 0.01 * X["b"] + 0.002 * generador.standard_normal(N_OBSERVACIONES)
    return X, y.rename("rend_futuro_1")


@pytest.fixture(scope="module")
def datos_una_variable() -> tuple[pd.DataFrame, pd.Series]:
    """Cinco variables de las que solo ``v2`` explica el objetivo."""
    generador = np.random.default_rng(7)
    n = 300
    indice = _indice(n)
    X = pd.DataFrame(
        generador.standard_normal((n, 5)),
        index=indice,
        columns=[f"v{i}" for i in range(5)],
    )
    y = pd.Series(
        0.03 * X["v2"].to_numpy() + 0.003 * generador.standard_normal(n),
        index=indice,
        name="rend_futuro_1",
    )
    return X, y


# --- interfaz comun ------------------------------------------------------------


@pytest.mark.parametrize("nombre", sorted(MODELOS))
def test_todos_los_modelos_se_crean_ajustan_y_predicen(nombre, datos_lineales):
    X, y = datos_lineales
    modelo = crear_modelo(nombre)
    prediccion = ajustar_y_predecir(modelo, X.iloc[:CORTE], y.iloc[:CORTE], X.iloc[CORTE:])

    assert isinstance(prediccion, pd.Series)
    # La prediccion vuelve con las fechas de la prueba: emparejarla despues con
    # el rendimiento realizado por posicion es como se cruzan dias distintos.
    assert prediccion.index.equals(X.index[CORTE:])
    assert np.isfinite(prediccion.to_numpy()).all()


def test_media_cero_predice_exactamente_cero(datos_lineales):
    """Exactamente cero, no aproximadamente: es la definicion de la referencia."""
    X, y = datos_lineales
    prediccion = ajustar_y_predecir(
        crear_modelo(NOMBRE_REFERENCIA), X.iloc[:CORTE], y.iloc[:CORTE], X.iloc[CORTE:]
    )

    assert (prediccion.to_numpy() == 0.0).all()


def test_modelo_desconocido_lanza_error_datos():
    with pytest.raises(ErrorDatos, match="desconocido"):
        crear_modelo("red_neuronal_milagrosa")


def test_el_error_del_modelo_desconocido_enumera_los_disponibles():
    """Un error que no dice que se puede usar obliga a leer el codigo fuente."""
    with pytest.raises(ErrorDatos) as excepcion:
        crear_modelo("ridge_regression")
    mensaje = str(excepcion.value)
    assert all(nombre in mensaje for nombre in MODELOS)


def test_hiperparametro_inexistente_lanza_error_datos():
    with pytest.raises(ErrorDatos, match="no acepta"):
        crear_modelo("ridge", penalizacion=0.5)


# --- el escalado va dentro del pipeline ----------------------------------------


@pytest.mark.parametrize("nombre", LINEALES)
def test_los_lineales_llevan_el_escalado_dentro_del_pipeline(nombre):
    pasos = dict(crear_modelo(nombre).steps)

    assert list(pasos) == [PASO_ESCALADO, PASO_MODELO]
    assert isinstance(pasos[PASO_ESCALADO], StandardScaler)


@pytest.mark.parametrize("nombre", ARBOLES)
def test_los_arboles_no_llevan_escalado(nombre):
    """Escalar un arbol no mejora nada y aniade estado ajustado de mas."""
    assert list(dict(crear_modelo(nombre).steps)) == [PASO_MODELO]


def test_el_escalado_usa_solo_la_media_del_entrenamiento(datos_lineales):
    """La comprobacion que de verdad importa del escalado dentro del Pipeline.

    Que el paso exista no demuestra nada; lo que demuestra que no hay fuga es
    **con que cifras** se ajusta. Al ajustar con el tramo de entrenamiento, la
    media guardada tiene que ser la de ese tramo y no la de la muestra completa,
    que resume tambien las filas del futuro.
    """
    X, y = datos_lineales
    modelo = crear_modelo("ridge")
    modelo.fit(X.iloc[:CORTE], y.iloc[:CORTE])
    escalador = modelo[PASO_ESCALADO]

    esperada = X.iloc[:CORTE].to_numpy().mean(axis=0)
    assert np.allclose(escalador.mean_, esperada)
    # Y no la de toda la muestra: si coincidieran, la prueba no distinguiria nada.
    media_completa = X.to_numpy().mean(axis=0)
    assert not np.allclose(escalador.mean_, media_completa, atol=1e-4)


# --- senal conocida: el modelo la recupera y bate a la referencia ---------------


def test_ridge_recupera_los_coeficientes_de_una_relacion_lineal(datos_lineales):
    """Con senal fuerte y muchas observaciones, la penalizacion apenas encoge.

    Los coeficientes se leen destipificados (dividiendo por la escala que guarda
    el ``StandardScaler``), que es la unica forma de compararlos con los del
    proceso generador.
    """
    X, y = datos_lineales
    modelo = crear_modelo("ridge")
    modelo.fit(X, y)
    coeficientes = pd.Series(
        modelo[PASO_MODELO].coef_ / modelo[PASO_ESCALADO].scale_, index=X.columns
    )

    assert coeficientes["a"] == pytest.approx(0.02, rel=0.05)
    assert coeficientes["b"] == pytest.approx(-0.01, rel=0.05)
    assert abs(coeficientes["c"]) < 0.001
    assert abs(coeficientes["d"]) < 0.001


def test_ridge_bate_claramente_a_la_referencia_fuera_de_muestra(datos_lineales, particion):
    X, y = datos_lineales
    tabla, predicciones = comparar_modelos(X, y, ["ridge"], particion)

    assert set(predicciones) == {"ridge", NOMBRE_REFERENCIA}
    # MASE muy por debajo de 1 (menos de la mitad del error de no hacer nada) y
    # una diferencia que el contraste de Diebold-Mariano considera significativa.
    assert tabla.loc["ridge", "mase"] < 0.5
    assert tabla.loc["ridge", "p_valor_dm"] < 0.01
    assert tabla.index[0] == "ridge"
    assert tabla.loc[NOMBRE_REFERENCIA, "mase"] == pytest.approx(1.0)


# --- la prueba que protege del autoengano ---------------------------------------


@pytest.mark.parametrize("semilla", [1, 4, 9])
def test_con_ruido_puro_ningun_modelo_bate_a_la_referencia(semilla, particion):
    """Sin senal no puede haber prediccion, y menos significativa.

    Se considera que un modelo "bate" a la referencia si cumple las dos
    condiciones con las que se leeria la tabla en un informe: menos error
    absoluto (MASE por debajo de 1) y un p-valor de Diebold-Mariano por debajo
    de 0,05. Sobre ruido independiente ninguna de las semillas probadas produce
    un modelo asi, y no por casualidad: cualquier prediccion no nula anade error
    cuadratico esperado sobre predecir cero, asi que la diferencia esta sesgada
    en contra del modelo, no centrada en cero.

    Si esta prueba falla, el sitio donde buscar no es esta funcion: es la
    construccion de variables, la particion o el escalado.
    """
    X, y = datos_ruido(semilla)
    tabla, predicciones = comparar_modelos(X, y, particion=particion)

    # Sin 'nombres' se comparan todos los modelos del catalogo, la referencia
    # incluida: es la tabla que se leeria en un informe.
    assert set(tabla.index) == set(MODELOS) == set(predicciones)
    baten = tabla[(tabla["mase"] < 1.0) & (tabla["p_valor_dm"] < 0.05)]
    baten = [nombre for nombre in baten.index if nombre != NOMBRE_REFERENCIA]
    assert baten == [], (
        f"con ruido puro (semilla {semilla}) estos modelos baten a "
        f"'{NOMBRE_REFERENCIA}': {baten}. Sospeche de una fuga de informacion "
        "antes que de un hallazgo."
    )
    assert tabla.loc[NOMBRE_REFERENCIA, "mase"] == pytest.approx(1.0)


def test_con_ruido_puro_los_arboles_quedan_por_detras_de_la_referencia(particion):
    """No es un detalle: es el coste de la flexibilidad cuando no hay senal.

    Un modelo con capacidad para ajustar cualquier cosa ajusta el ruido del
    tramo de entrenamiento y lo proyecta sobre el de prueba, asi que sale
    **peor** que no predecir nada. Comprobarlo explicitamente evita la lectura
    perezosa de "el arbol empata con la referencia".
    """
    X, y = datos_ruido(1)
    tabla, _ = comparar_modelos(X, y, ["arbol_potenciado"], particion)

    assert tabla.loc["arbol_potenciado", "mase"] > 1.0
    assert tabla.index[0] == NOMBRE_REFERENCIA


# --- importancia de variables ---------------------------------------------------


@pytest.mark.parametrize("nombre", ["ridge", "arbol_potenciado"])
def test_la_importancia_senala_la_unica_variable_relevante(nombre, datos_una_variable):
    X, y = datos_una_variable
    importancia = importancia_variables(crear_modelo(nombre), X, y, n_repeticiones=5)

    assert list(importancia.index) == sorted(importancia.index, key=lambda c: -importancia[c])
    assert importancia.index[0] == "v2"
    # No basta con que gane: tiene que ganar por un orden de magnitud, porque lo
    # que se afirma es que las demas variables no aportan nada.
    resto = importancia.drop("v2").abs().max()
    assert importancia["v2"] > 10.0 * resto


def test_la_importancia_de_la_referencia_es_nula(datos_una_variable):
    """Quien predice siempre cero no usa ninguna variable, y se nota."""
    X, y = datos_una_variable
    importancia = importancia_variables(
        crear_modelo(NOMBRE_REFERENCIA), X, y, n_repeticiones=3
    )

    assert (importancia.abs() < 1e-12).all()


def test_la_importancia_admite_un_modelo_ya_ajustado(datos_una_variable):
    """Es el uso honesto: ajustar con el pasado y medir en el tramo no visto."""
    X, y = datos_una_variable
    modelo = crear_modelo("ridge")
    ajustar_y_predecir(modelo, X.iloc[:200], y.iloc[:200], X.iloc[200:])
    coeficientes_antes = modelo[PASO_MODELO].coef_.copy()

    importancia = importancia_variables(modelo, X.iloc[200:], y.iloc[200:], n_repeticiones=5)

    assert importancia.index[0] == "v2"
    # No se ha reajustado con el tramo de prueba: eso convertiria la medida
    # fuera de muestra en una medida dentro de muestra sin avisar.
    assert np.allclose(modelo[PASO_MODELO].coef_, coeficientes_antes)


def test_la_importancia_exige_nombres_de_variable(datos_una_variable):
    X, y = datos_una_variable
    with pytest.raises(ErrorDatos, match="DataFrame"):
        importancia_variables(crear_modelo("ridge"), X.to_numpy(), y)


def test_la_importancia_rechaza_cero_repeticiones(datos_una_variable):
    X, y = datos_una_variable
    with pytest.raises(ErrorDatos, match="al menos 1"):
        importancia_variables(crear_modelo("ridge"), X, y, n_repeticiones=0)


# --- ajuste y prediccion: las trampas que no producen excepcion -----------------


def test_predecir_fechas_ya_vistas_lanza_error_datos(datos_lineales):
    """Ajustar con todo y predecir sobre todo es el ajuste disfrazado de prediccion."""
    X, y = datos_lineales
    with pytest.raises(ErrorDatos, match="no es posterior"):
        ajustar_y_predecir(crear_modelo("ridge"), X, y, X)


def test_columnas_en_otro_orden_lanza_error_datos(datos_lineales):
    """Un lineal aplicaria cada coeficiente a la variable equivocada en silencio."""
    X, y = datos_lineales
    with pytest.raises(ErrorDatos, match="mismas variables"):
        ajustar_y_predecir(
            crear_modelo("ridge"),
            X.iloc[:CORTE],
            y.iloc[:CORTE],
            X.iloc[CORTE:, ::-1],
        )


def test_entrenamiento_desalineado_lanza_error_datos(datos_lineales):
    X, y = datos_lineales
    with pytest.raises(ErrorDatos, match="dropna"):
        ajustar_y_predecir(
            crear_modelo("ridge"), X.iloc[:CORTE], y.iloc[: CORTE - 1], X.iloc[CORTE:]
        )


def test_con_matrices_de_numpy_no_se_exige_el_orden_temporal(datos_lineales):
    """Sin indice no hay fechas, asi que la comprobacion no se puede hacer.

    Dos ``ndarray`` empiezan los dos en la posicion 0: exigirles orden seria
    rechazar todo uso legitimo de numpy. La garantia se paga trabajando con
    pandas, que es como salen las matrices de :mod:`icm.ml.caracteristicas`.
    """
    X, y = datos_lineales
    prediccion = ajustar_y_predecir(
        crear_modelo("ridge"),
        X.iloc[:CORTE].to_numpy(),
        y.iloc[:CORTE].to_numpy(),
        X.iloc[CORTE:].to_numpy(),
    )

    assert len(prediccion) == N_OBSERVACIONES - CORTE
    assert prediccion.index.equals(pd.RangeIndex(N_OBSERVACIONES - CORTE))


def test_ajustar_y_predecir_deja_el_modelo_ajustado(datos_lineales):
    """Convenio de scikit-learn, y lo que permite encadenar con la importancia."""
    X, y = datos_lineales
    modelo = crear_modelo("ridge")
    ajustar_y_predecir(modelo, X.iloc[:CORTE], y.iloc[:CORTE], X.iloc[CORTE:])

    assert hasattr(modelo[PASO_MODELO], "coef_")


# --- busqueda de hiperparametros -----------------------------------------------


def test_la_busqueda_elige_el_alpha_con_sentido(datos_lineales, particion):
    """Con senal lineal fuerte, la penalizacion enorme tiene que perder.

    La rejilla son dos extremos, 1e-6 y 1e6: con alpha 1e6 los coeficientes
    quedan aplastados y el modelo predice practicamente cero. No hay margen para
    la ambiguedad, asi que el resultado no depende del ruido de la muestra.
    """
    X, y = datos_lineales
    mejores = buscar_hiperparametros(
        "ridge", X, y, rejilla={"alpha": [1e-6, 1e6]}, particion=particion
    )

    assert mejores == {"alpha": 1e-6}
    # El diccionario esta en los nombres que acepta crear_modelo, sin el prefijo
    # del paso del Pipeline: encadenar las dos llamadas es el uso previsto.
    assert crear_modelo("ridge", **mejores)[PASO_MODELO].alpha == 1e-6


def test_la_busqueda_de_la_referencia_no_tiene_nada_que_elegir(datos_lineales, particion):
    X, y = datos_lineales
    assert buscar_hiperparametros(NOMBRE_REFERENCIA, X, y, particion=particion) == {}


def test_rejilla_con_hiperparametro_inexistente_lanza_error_datos(datos_lineales, particion):
    """Y el error enumera los que si existen, para no obligar a leer sklearn."""
    X, y = datos_lineales
    with pytest.raises(ErrorDatos, match="no tiene el hiperparametro") as excepcion:
        buscar_hiperparametros(
            "ridge", X, y, rejilla={"lambda": [0.1, 1.0]}, particion=particion
        )

    assert "alpha" in str(excepcion.value)


def test_rejilla_con_un_valor_sin_lista_lanza_error_datos(datos_lineales, particion):
    """El fallo silencioso seria peor: sklearn iteraria sobre un escalar."""
    X, y = datos_lineales
    with pytest.raises(ErrorDatos, match="lista"):
        buscar_hiperparametros("ridge", X, y, rejilla={"alpha": 1.0}, particion=particion)


def test_la_particion_adaptada_no_mira_al_futuro(datos_lineales, particion):
    """El adaptador traduce nombres de metodos; no reordena nada.

    Es la comprobacion que impide que alguien sustituya el ``cv`` por un entero
    (que en regresion significa ``KFold`` aleatorio) sin que nada proteste: aqui
    se exige que **todas** las posiciones de entrenamiento sean anteriores a
    todas las de prueba, con el embargo de por medio.
    """
    X, _ = datos_lineales
    adaptador = ParticionSklearn(particion)
    tramos = list(adaptador.split(X))

    assert len(tramos) == adaptador.get_n_splits(X) == 3
    for entrenamiento, prueba in tramos:
        assert entrenamiento.max() < prueba.min()
        assert prueba.min() - entrenamiento.max() > particion.embargo


def test_la_particion_adaptada_rechaza_un_kfold():
    with pytest.raises(ErrorDatos, match="KFold"):
        ParticionSklearn(KFold(n_splits=5, shuffle=True))


# --- comparacion ----------------------------------------------------------------


def test_comparar_modelos_anade_la_referencia_aunque_no_se_pida(datos_lineales, particion):
    X, y = datos_lineales
    tabla, predicciones = comparar_modelos(X, y, ["lasso", "bosque"], particion)

    assert NOMBRE_REFERENCIA in tabla.index
    assert NOMBRE_REFERENCIA in predicciones
    assert tabla.attrs["referencia"] == NOMBRE_REFERENCIA


def test_comparar_modelos_mide_a_todos_sobre_las_mismas_fechas(datos_lineales, particion):
    """Sin esto la tabla comparativa no compara: cada modelo tendria su periodo."""
    X, y = datos_lineales
    tabla, predicciones = comparar_modelos(X, y, ["ridge", "arbol_potenciado"], particion)

    indices = [serie.index for serie in predicciones.values()]
    assert all(indice.equals(indices[0]) for indice in indices)
    assert (tabla["n_observaciones"] == len(indices[0])).all()


def test_comparar_modelos_con_un_nombre_desconocido_lanza_error_datos(
    datos_lineales, particion
):
    X, y = datos_lineales
    with pytest.raises(ErrorDatos, match="desconocido"):
        comparar_modelos(X, y, ["ridge", "adivino"], particion)


def test_comparar_modelos_con_lista_vacia_lanza_error_datos(datos_lineales, particion):
    X, y = datos_lineales
    with pytest.raises(ErrorDatos, match="vacio"):
        comparar_modelos(X, y, [], particion)
