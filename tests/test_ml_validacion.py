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


class ModeloConHueco:
    """Imita a ``RedSecuencial``: NaN en las primeras filas de cada tramo.

    Es el modelo con el que se descubrio que ``acierto_direccional`` contaba los
    huecos como fallos, y por eso no es un caso inventado: la red del
    repositorio devuelve NaN en las ``ventana - 1`` primeras filas de la tabla
    que recibe, y cada tramo de prueba empieza con ese hueco.
    """

    def __init__(self, ventana: int = 10) -> None:
        self.ventana = ventana
        self.coeficientes: np.ndarray | None = None

    def ajustar(self, X: pd.DataFrame, y: pd.Series) -> ModeloConHueco:
        self.coeficientes = np.linalg.lstsq(
            X.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None
        )[0]
        return self

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        # El coseno degrada el ajuste a proposito: con una prediccion perfecta
        # el acierto verdadero seria 1.0 y no distinguiria un acierto bien
        # calculado de uno sesgado a la baja.
        salida = X.to_numpy(dtype=float) @ self.coeficientes + 0.12 * np.cos(
            np.arange(len(X))
        )
        salida[: self.ventana - 1] = np.nan
        return salida


class ModeloEtiquetado(ModeloPrimeraColumna):
    """Devuelve la prediccion como ``Series`` indexada, y opcionalmente reordenada.

    Reordenar la salida es lo que hace por dentro cualquier modelo que agrupe u
    ordene (``groupby``, ``sort_values``, ``sort_index``) antes de devolver:
    el tamanio cuadra y solo el indice delata el desorden.
    """

    def __init__(self, reordenar: bool = False) -> None:
        super().__init__()
        self.reordenar = reordenar

    def predecir(self, X: pd.DataFrame) -> pd.Series:
        salida = pd.Series(X[self.columna].to_numpy(dtype=float), index=X.index)
        return salida.sort_values() if self.reordenar else salida


class RedDeArranqueCaliente:
    """Modelo del convenio del repositorio, sin ``get_params`` y sin reiniciar.

    ``sklearn.base.clone`` no puede copiarlo (no expone ``get_params``), asi que
    cae en la via de ``copy.deepcopy``, y su ajuste sigue desde los pesos
    actuales en vez de reinicializarlos: si llega ya entrenado, cada particion
    arranca desde unos pesos que han visto las fechas de prueba.
    """

    def __init__(self, pasos: int = 25, tasa: float = 0.05) -> None:
        self.pasos = pasos
        self.tasa = tasa
        self.pesos: np.ndarray | None = None

    def ajustar(self, X, y) -> RedDeArranqueCaliente:
        matriz = np.asarray(X, dtype=float)
        objetivo = np.asarray(y, dtype=float)
        if self.pesos is None:
            self.pesos = np.zeros(matriz.shape[1])
        for _ in range(self.pasos):
            residuo = matriz @ self.pesos - objetivo
            self.pesos -= self.tasa * (matriz.T @ residuo) / len(objetivo)
        return self

    def predecir(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.pesos


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


def test_con_x_sin_indice_no_se_exige_que_el_modelo_reproduzca_las_posiciones(X):
    """Las fechas de un ``ndarray`` las inventa este modulo, no el modelo.

    El emparejamiento por etiqueta solo tiene sentido cuando las etiquetas
    vienen de los datos; con ``X`` sin indice se comprueba el numero de filas y
    se empareja por posicion, que es la unica nocion de tiempo que hay.
    """

    class ModeloSerieDesdeCero:
        def ajustar(self, X, y):
            self.media = float(np.asarray(y, dtype=float).mean())
            return self

        def predecir(self, X):
            return pd.Series(np.full(len(X), self.media))

    objetivo = X["senal"].rename("objetivo")
    resultado = validacion_walk_forward(
        ModeloSerieDesdeCero(),
        X.to_numpy(),
        objetivo.to_numpy(),
        ParticionWalkForward(n_particiones=3),
    )

    assert resultado.predicciones.nunique() == 3


def test_una_prediccion_bidimensional_se_aplana_sin_desalinearse(X):
    class ModeloColumna(ModeloPrimeraColumna):
        def predecir(self, X: pd.DataFrame) -> np.ndarray:
            return super().predecir(X).reshape(-1, 1)

    objetivo = X["senal"].rename("objetivo")
    resultado = validacion_walk_forward(ModeloColumna(), X, objetivo)

    np.testing.assert_allclose(
        resultado.predicciones.to_numpy(), resultado.reales.to_numpy()
    )


def test_las_fechas_sin_prediccion_no_cuentan_como_fallo_direccional(X, y):
    """El acierto debe medirse solo donde hay prediccion, no diluirse en huecos.

    ``np.sign(nan)`` es NaN y NaN nunca iguala al signo real, asi que cada hueco
    entraba en el denominador contado como fallo. Eso convertia el acierto en la
    unica metrica que no avisaba de un hueco: las demas salen NaN.
    """
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50)
    resultado = validacion_walk_forward(ModeloConHueco(ventana=10), X, y, particion)

    predicciones = resultado.predicciones.to_numpy(dtype=float)
    reales = resultado.reales.to_numpy(dtype=float)
    validas = np.isfinite(predicciones) & np.isfinite(reales) & (reales != 0.0)
    esperado = float(np.mean(np.sign(predicciones[validas]) == np.sign(reales[validas])))

    assert validas.sum() == 205  # 45 huecos de los 250; el 18 % de las filas
    assert resultado.resumen["acierto_direccional"] == pytest.approx(esperado)
    assert resultado.resumen["n_validas"] == 205
    assert resultado.resumen["observaciones"] == 250
    assert np.isnan(resultado.resumen["rmse"])
    assert (resultado.por_particion["n_validas"] == 41).all()


def test_una_prediccion_etiquetada_y_reordenada_se_realinea_por_fecha(X):
    """Si el modelo trae el indice, se usa para emparejar y no la posicion.

    Desetiquetar con ``np.asarray`` y volver a etiquetar por posicion pegaba
    cada prediccion a la fecha de otra en cuanto el modelo reordenaba su salida,
    con el tamanio cuadrando y sin ningun aviso.
    """
    objetivo = X["senal"].rename("objetivo")
    particion = ParticionWalkForward(n_particiones=5, tamano_prueba=50)

    ordenado = validacion_walk_forward(ModeloEtiquetado(), X, objetivo, particion)
    reordenado = validacion_walk_forward(
        ModeloEtiquetado(reordenar=True), X, objetivo, particion
    )

    pd.testing.assert_series_equal(reordenado.predicciones, ordenado.predicciones)
    np.testing.assert_allclose(
        reordenado.predicciones.to_numpy(), reordenado.reales.to_numpy()
    )
    assert reordenado.resumen["correlacion"] > 0.99


def test_una_prediccion_con_fechas_ajenas_al_tramo_se_rechaza(X):
    """Un indice que no es el del tramo no se puede emparejar por posicion."""

    class ModeloDesplazado(ModeloEtiquetado):
        def predecir(self, X: pd.DataFrame) -> pd.Series:
            salida = super().predecir(X)
            salida.index = salida.index + pd.Timedelta(days=400)
            return salida

    objetivo = X["senal"].rename("objetivo")

    with pytest.raises(ErrorDatos, match="no coincide"):
        validacion_walk_forward(ModeloDesplazado(), X, objetivo)


def test_una_prediccion_de_dos_columnas_no_se_aplana_a_la_callada(X):
    """Aplanar una tabla de dos columnas hacia cuadrar el tamanio es desalinear.

    ``reshape(-1)`` hacia que una salida de forma ``(n/2, 2)`` cuadrase con las
    ``n`` fechas del tramo, asi que el control de tamanio no la veia.
    """

    class ModeloDosColumnas(ModeloEtiquetado):
        def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
            valores = X[self.columna].to_numpy(dtype=float)
            mitad = len(valores) // 2
            return pd.DataFrame(
                {"a": valores[:mitad], "b": valores[mitad:]}, index=X.index[:mitad]
            )

    objetivo = X["senal"].rename("objetivo")

    with pytest.raises(ErrorDatos, match="filas de predicciones"):
        validacion_walk_forward(ModeloDosColumnas(), X, objetivo)


def test_un_modelo_ya_ajustado_sin_get_params_se_rechaza(X, y):
    """La copia por ``deepcopy`` arrastra el ajuste, asi que hay que rechazarlo.

    Con un modelo entrenado sobre toda la serie, cada particion arrancaria desde
    unos pesos que ya han visto las fechas de prueba; medido con ``y`` de ruido
    puro, la correlacion informada pasaba de -0,02 a +0,20.
    """
    preajustada = RedDeArranqueCaliente().ajustar(X, y)

    with pytest.raises(ErrorDatos, match="ya ajustado"):
        validacion_walk_forward(preajustada, X, y, ParticionWalkForward(n_particiones=3))

    sin_ajustar = RedDeArranqueCaliente()
    resultado = validacion_walk_forward(
        sin_ajustar, X, y, ParticionWalkForward(n_particiones=3)
    )
    assert len(resultado.modelos) == 3
    assert sin_ajustar.pesos is None


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


def test_un_panel_con_multiindex_se_rechaza(X):
    """El MultiIndex burla el control de duplicados sin dejar de romperlo.

    Los pares ``(fecha, activo)`` son unicos y crecientes, asi que la particion
    los aceptaba: cortaba una fecha por la mitad, dejando unos activos del dia
    ``t`` en entrenamiento y otros en prueba, y contaba el embargo en filas, de
    modo que compraba ``embargo / n_activos`` dias de hueco.
    """
    fechas = pd.bdate_range("2018-01-01", periods=140)
    activos = ["A", "B", "C", "D", "E"]
    panel_indice = pd.MultiIndex.from_product([fechas, activos], names=["fecha", "activo"])
    panel = pd.DataFrame(
        np.random.default_rng(7).normal(size=(len(panel_indice), 2)),
        index=panel_indice,
        columns=["u", "v"],
    )
    particion = ParticionWalkForward(n_particiones=4, tamano_prueba=48)

    with pytest.raises(ErrorDatos, match="panel"):
        list(particion.dividir(panel))
    with pytest.raises(ErrorDatos, match="panel"):
        particion.describir(panel)


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
