"""Pruebas de las particiones temporales y de la validacion hacia delante.

Lo que se contrasta aqui no es que el codigo no reviente, sino la propiedad que
justifica el modulo entero: que ninguna particion entrena con datos posteriores
a los que despues predice. Por eso el orden se comprueba con las **fechas** del
indice y no solo con las posiciones: una particion puede tener las posiciones
impecables y aun asi estar leyendo un indice desordenado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from icm.comun import ErrorDatos
from icm.ml.validacion import (
    ParticionWalkForward,
    validacion_walk_forward,
)

N_OBSERVACIONES = 600


# --- modelos de juguete --------------------------------------------------------


class ModeloPrimeraColumna:
    """Modelo estilo repositorio (``ajustar`` / ``predecir``) que copia una columna.

    Sirve para comprobar que la validacion acepta el convenio del repositorio
    sin adaptadores, y que las predicciones vuelven con la fecha correcta.
    """

    def __init__(self) -> None:
        self.columna: str | None = None

    def ajustar(self, X: pd.DataFrame, y: pd.Series) -> ModeloPrimeraColumna:
        self.columna = X.columns[0]
        return self

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.columna].to_numpy(dtype=float)


class ModeloContador:
    """Cuenta cuantas veces se le ha ajustado y lo devuelve como prediccion.

    Si la validacion reutilizase el mismo objeto en todas las particiones, la
    cuenta iria subiendo: es la forma mas directa de ver la contaminacion entre
    particiones, que de otro modo es invisible.
    """

    def __init__(self) -> None:
        self.ajustes = 0

    def ajustar(self, X, y) -> ModeloContador:
        self.ajustes += 1
        return self

    def predecir(self, X) -> np.ndarray:
        return np.full(len(X), float(self.ajustes))


class ModeloMudo:
    """Objeto sin ninguna de las dos interfaces reconocidas."""

    def entrenar(self, X, y) -> None:  # pragma: no cover - nunca debe llamarse
        raise AssertionError("la validacion no deberia haber llegado hasta aqui")


# --- datos ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def indice() -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=N_OBSERVACIONES, name="fecha")


@pytest.fixture(scope="module")
def X(indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Tres variables explicativas independientes y deterministas."""
    generador = np.random.default_rng(20240501)
    return pd.DataFrame(
        generador.normal(size=(N_OBSERVACIONES, 3)),
        index=indice,
        columns=["senal", "ruido", "tendencia"],
    )


@pytest.fixture(scope="module")
def y(X: pd.DataFrame) -> pd.Series:
    """Objetivo lineal en las variables, con un ruido pequenio y conocido."""
    generador = np.random.default_rng(99)
    valores = 0.5 * X["senal"] - 0.3 * X["ruido"] + 0.01 * generador.normal(size=len(X))
    return pd.Series(valores, index=X.index, name="objetivo")


# --- geometria de la particion -------------------------------------------------


def test_los_tramos_de_entrenamiento_y_prueba_nunca_se_solapan(X):
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50, embargo=7)

    for entrenamiento, prueba in particion.dividir(X):
        assert entrenamiento.size > 0 and prueba.size > 0
        assert not set(entrenamiento.tolist()) & set(prueba.tolist())


def test_el_entrenamiento_precede_temporalmente_a_la_prueba(X):
    """La comprobacion se hace con las fechas, no con las posiciones."""
    particion = ParticionWalkForward(n_particiones=4, tamano_prueba=60)

    for entrenamiento, prueba in particion.dividir(X):
        fechas_entrenamiento = X.index[entrenamiento]
        fechas_prueba = X.index[prueba]
        assert fechas_entrenamiento.max() < fechas_prueba.min()
        assert fechas_entrenamiento.is_monotonic_increasing
        assert fechas_prueba.is_monotonic_increasing


def test_los_tramos_de_prueba_son_contiguos_y_terminan_en_la_ultima_fecha(X):
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50)
    pruebas = [prueba for _, prueba in particion.dividir(X)]

    concatenadas = np.concatenate(pruebas)
    esperadas = np.arange(len(X) - 5 * 50, len(X))
    np.testing.assert_array_equal(concatenadas, esperadas)


def test_el_embargo_deja_exactamente_el_hueco_pedido(X):
    embargo = 10
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50, embargo=embargo)

    for entrenamiento, prueba in particion.dividir(X):
        descartadas = X.index[entrenamiento[-1] + 1 : prueba[0]]
        assert len(descartadas) == embargo
        assert prueba[0] - entrenamiento[-1] - 1 == embargo


def test_sin_embargo_el_entrenamiento_llega_hasta_la_vispera_de_la_prueba(X):
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50, embargo=0)

    for entrenamiento, prueba in particion.dividir(X):
        assert prueba[0] - entrenamiento[-1] == 1


def test_la_particion_expansiva_hace_crecer_el_entrenamiento(X):
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50, expansiva=True)
    tramos = list(particion.dividir(X))

    tamanos = [entrenamiento.size for entrenamiento, _ in tramos]
    assert tamanos == sorted(tamanos)
    assert len(set(tamanos)) == len(tamanos)
    assert all(entrenamiento[0] == 0 for entrenamiento, _ in tramos)


def test_la_particion_deslizante_mantiene_constante_el_entrenamiento(X):
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50, expansiva=False)
    tramos = list(particion.dividir(X))

    tamanos = {entrenamiento.size for entrenamiento, _ in tramos}
    assert len(tamanos) == 1
    inicios = [entrenamiento[0] for entrenamiento, _ in tramos]
    assert inicios[0] == 0
    assert inicios == sorted(inicios) and inicios[-1] > 0


def test_el_tamano_de_prueba_automatico_reparte_el_resto_a_partes_iguales(X):
    particion = ParticionWalkForward(n_particiones=5, tamano_entrenamiento_minimo=100)
    tramos = list(particion.dividir(X))

    assert [prueba.size for _, prueba in tramos] == [100] * 5
    assert tramos[0][0].size == 100


def test_n_divisiones_coincide_con_los_tramos_recorridos(X):
    particion = ParticionWalkForward(n_particiones=7, tamano_prueba=40)
    assert particion.n_divisiones(X) == len(list(particion.dividir(X))) == 7


# --- descripcion ---------------------------------------------------------------


def test_describir_devuelve_las_fechas_de_cada_tramo(X):
    particion = ParticionWalkForward(n_particiones=3, tamano_prueba=50, embargo=5)
    tabla = particion.describir(X)
    tramos = list(particion.dividir(X))

    assert len(tabla) == 3
    for (numero, fila), (entrenamiento, prueba) in zip(tabla.iterrows(), tramos):
        assert fila["inicio_entrenamiento"] == X.index[entrenamiento[0]]
        assert fila["fin_entrenamiento"] == X.index[entrenamiento[-1]]
        assert fila["inicio_prueba"] == X.index[prueba[0]]
        assert fila["fin_prueba"] == X.index[prueba[-1]]
        assert fila["n_entrenamiento"] == entrenamiento.size
        assert fila["n_prueba"] == prueba.size
        assert fila["fin_entrenamiento"] < fila["inicio_prueba"]
        assert numero == int(numero)


# --- validacion ----------------------------------------------------------------


def test_las_predicciones_fuera_de_muestra_cubren_el_indice_esperado(X, y):
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50)
    resultado = validacion_walk_forward(LinearRegression(), X, y, particion)

    esperado = X.index[len(X) - 250 :]
    assert resultado.predicciones.index.equals(esperado)
    assert resultado.reales.index.equals(esperado)
    assert resultado.predicciones.index.is_unique
    assert not resultado.predicciones.isna().any()
    # check_freq=False: concatenar los tramos pierde la marca de frecuencia del
    # indice, que es metadato y no dato.
    pd.testing.assert_series_equal(
        resultado.reales, y.loc[esperado], check_names=False, check_freq=False
    )


def test_la_validacion_recupera_un_modelo_que_memoriza(X):
    """Con ``y`` igual a la primera columna, el ajuste lineal es exacto."""
    objetivo = X["senal"].rename("objetivo")
    resultado = validacion_walk_forward(LinearRegression(), X, objetivo)

    assert resultado.por_particion["rmse"].max() < 1e-9
    assert resultado.por_particion["r2_fuera_muestra"].min() > 0.999
    np.testing.assert_allclose(
        resultado.predicciones.to_numpy(), resultado.reales.to_numpy(), atol=1e-9
    )


def test_la_validacion_recupera_la_relacion_lineal_conocida(X, y):
    resultado = validacion_walk_forward(LinearRegression(), X, y)

    for modelo in resultado.modelos:
        np.testing.assert_allclose(modelo.coef_, [0.5, -0.3, 0.0], atol=0.01)
    assert resultado.por_particion["r2_fuera_muestra"].min() > 0.99
    assert resultado.resumen["correlacion"] > 0.99


def test_el_modelo_estilo_repositorio_se_valida_sin_adaptador(X):
    objetivo = X["senal"].rename("objetivo")
    resultado = validacion_walk_forward(ModeloPrimeraColumna(), X, objetivo)

    np.testing.assert_allclose(
        resultado.predicciones.to_numpy(), resultado.reales.to_numpy()
    )
    assert all(isinstance(m, ModeloPrimeraColumna) for m in resultado.modelos)


def test_cada_particion_ajusta_una_copia_sin_contaminar_la_siguiente(X, y):
    contador = ModeloContador()
    resultado = validacion_walk_forward(contador, X, y)

    # Si se reutilizase el objeto, las predicciones irian 1, 2, 3, 4, 5.
    assert set(resultado.predicciones.unique()) == {1.0}
    assert contador.ajustes == 0
    assert len({id(m) for m in resultado.modelos}) == len(resultado.modelos)


def test_el_modelo_recibido_no_queda_ajustado(X, y):
    modelo = LinearRegression()
    validacion_walk_forward(modelo, X, y, ParticionWalkForward(n_particiones=3))

    assert not hasattr(modelo, "coef_")


def test_la_tabla_por_particion_recoge_tamanos_y_metricas(X, y):
    particion = ParticionWalkForward(n_particiones=4, tamano_prueba=50, embargo=3)
    resultado = validacion_walk_forward(LinearRegression(), X, y, particion)
    tabla = resultado.por_particion

    assert len(tabla) == len(resultado.modelos) == 4
    assert tabla["n_prueba"].tolist() == [50] * 4
    assert tabla["n_entrenamiento"].is_monotonic_increasing
    for columna in ("rmse", "mae", "sesgo", "correlacion", "acierto_direccional"):
        assert columna in tabla.columns
    assert (tabla["rmse"] >= 0).all()
    assert resultado.resumen["observaciones"] == 200


def test_la_validacion_admite_arrays_de_numpy(X, y):
    resultado = validacion_walk_forward(
        LinearRegression(), X.to_numpy(), y.to_numpy(), ParticionWalkForward(n_particiones=3)
    )

    assert len(resultado.predicciones) == len(resultado.reales)
    assert resultado.predicciones.index.is_unique


def test_una_prediccion_bidimensional_se_aplana_sin_desalinearse(X):
    class ModeloColumna(ModeloPrimeraColumna):
        def predecir(self, X: pd.DataFrame) -> np.ndarray:
            return super().predecir(X).reshape(-1, 1)

    objetivo = X["senal"].rename("objetivo")
    resultado = validacion_walk_forward(ModeloColumna(), X, objetivo)

    np.testing.assert_allclose(
        resultado.predicciones.to_numpy(), resultado.reales.to_numpy()
    )


# --- errores -------------------------------------------------------------------


def test_datos_insuficientes_dicen_cuantas_observaciones_harian_falta(X):
    corto = X.iloc[:50]
    particion = ParticionWalkForward(n_particiones=5, tamano_entrenamiento_minimo=100)

    with pytest.raises(ErrorDatos, match="105 observaciones"):
        list(particion.dividir(corto))


def test_un_tamano_de_prueba_demasiado_grande_dice_cuanto_falta(X):
    particion = ParticionWalkForward(
        n_particiones=5, tamano_prueba=50, tamano_entrenamiento_minimo=100
    )

    with pytest.raises(ErrorDatos, match="350 observaciones"):
        list(particion.dividir(X.iloc[:200]))


def test_la_falta_de_datos_se_detecta_antes_de_recorrer_nada(X):
    """El error salta en la llamada, no al consumir el iterador."""
    particion = ParticionWalkForward(n_particiones=5)

    with pytest.raises(ErrorDatos):
        particion.dividir(X.iloc[:50])
    with pytest.raises(ErrorDatos):
        particion.n_divisiones(X.iloc[:50])
    with pytest.raises(ErrorDatos):
        particion.describir(X.iloc[:50])


def test_el_embargo_puede_agotar_los_datos_disponibles(X):
    particion = ParticionWalkForward(
        n_particiones=5, tamano_prueba=100, embargo=50, tamano_entrenamiento_minimo=100
    )

    with pytest.raises(ErrorDatos, match="650 observaciones"):
        list(particion.dividir(X))


@pytest.mark.parametrize(
    "parametros",
    [
        {"n_particiones": 0},
        {"tamano_prueba": 0},
        {"embargo": -1},
        {"tamano_entrenamiento_minimo": 0},
    ],
)
def test_los_parametros_imposibles_se_rechazan_al_construir(parametros):
    with pytest.raises(ErrorDatos):
        ParticionWalkForward(**parametros)


def test_un_indice_desordenado_se_rechaza(X):
    desordenado = X.iloc[::-1]

    with pytest.raises(ErrorDatos, match="ordenado"):
        list(ParticionWalkForward().dividir(desordenado))


def test_un_indice_con_fechas_repetidas_se_rechaza(X):
    repetido = pd.concat([X.iloc[:300], X.iloc[:300]]).sort_index()

    with pytest.raises(ErrorDatos, match="repetidas"):
        list(ParticionWalkForward().dividir(repetido))


def test_un_modelo_sin_interfaz_conocida_se_rechaza(X, y):
    with pytest.raises(ErrorDatos, match="ajustar"):
        validacion_walk_forward(ModeloMudo(), X, y)


def test_x_e_y_de_distinta_longitud_se_rechazan(X, y):
    with pytest.raises(ErrorDatos, match="filas"):
        validacion_walk_forward(LinearRegression(), X, y.iloc[:-10])


def test_x_e_y_con_indices_distintos_se_rechazan(X, y):
    desplazado = y.copy()
    desplazado.index = desplazado.index + pd.Timedelta(days=1)

    with pytest.raises(ErrorDatos, match="indice"):
        validacion_walk_forward(LinearRegression(), X, desplazado)
