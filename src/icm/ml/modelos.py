"""Estimadores para prediccion de rendimientos, con su linea de flotacion.

Tres decisiones ordenan el modulo.

**La referencia es un modelo mas.** ``"media_cero"`` predice exactamente cero
siempre, que es lo que predice la hipotesis de paseo aleatorio sobre una serie
de rendimientos. Podria haber sido un numero calculado aparte, pero entonces
compararse con ella seria opcional, y lo opcional se omite justo cuando el
resultado no acompania. Expuesta como un estimador con la misma interfaz, entra
en la misma particion, se mide sobre las mismas fechas y aparece como una fila
mas de la tabla: batirla deja de ser una afirmacion y pasa a ser una columna.
Batirla es dificil; quien encuentra que la bate por mucho casi siempre ha
filtrado informacion del futuro en las variables.

**El escalado va dentro del Pipeline.** Nunca antes, sobre la matriz completa.
Vease :func:`crear_modelo`, donde esta el argumento entero.

**Los arboles potenciados son el modelo principal.** Con datos tabulares de
este tamanio (miles de filas, decenas de columnas, relaciones no lineales y
mucho ruido) el potenciado de gradiente sigue ganando en la practica a las
alternativas mas vistosas, y ademas cuesta minutos en lugar de horas. La
regresion regularizada no esta como adorno: es la referencia honesta que dice
cuanto de lo que aporta el arbol venia ya de una relacion lineal, y cuando la
senal es debil (que es el caso normal en rendimientos diarios) muchas veces
gana ella.

Convencion de nombres
---------------------
Las funciones de contraste de este repositorio se llaman ``contraste_...`` y no
``test_...``: pytest recoge como prueba cualquier funcion importable que empiece
por ``test_``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from icm.comun import ErrorDatos
from icm.ml.evaluacion import tabla_evaluacion
from icm.ml.validacion import ParticionWalkForward, validacion_walk_forward

#: Nombre del paso de escalado dentro del Pipeline de los modelos lineales.
PASO_ESCALADO = "escalado"

#: Nombre del paso del estimador dentro del Pipeline, en todos los modelos.
PASO_MODELO = "modelo"

#: Semilla por defecto de todo lo que tenga componente aleatoria, para que dos
#: ejecuciones del mismo experimento den la misma cifra.
SEMILLA = 42

#: Modelo trivial que sirve de linea de flotacion a todos los demas.
NOMBRE_REFERENCIA = "media_cero"


# --- el modelo trivial ---------------------------------------------------------


class MediaCero(RegressorMixin, BaseEstimator):
    """Predice exactamente 0.0 para cualquier entrada.

    Sobre rendimientos, "cero" no es una ocurrencia: es la prediccion del paseo
    aleatorio, es decir, la hipotesis de mercado eficiente en su version mas
    cruda. Vease :func:`icm.ml.evaluacion.prediccion_ingenua` para la distincion
    entre esa referencia y "el ultimo valor observado", que es la correcta sobre
    precios y un disparate sobre rendimientos.

    No predice la media del tramo de entrenamiento, que seria la otra referencia
    natural, porque esa media es ruido: con volatilidad diaria del 2 % y 500
    observaciones, su error tipico es del orden de la propia deriva que
    pretenderia estimar, asi que introduce un sesgo direccional arbitrario. Cero
    no tiene ese problema y es la cifra que se defiende sola.

    ``fit`` no aprende nada, pero valida ``X`` con ``validate_data`` de
    scikit-learn y con eso fija ``n_features_in_`` y ``feature_names_in_``: asi
    ``check_is_fitted`` distingue una instancia ajustada de una recien creada, y
    la referencia expone los mismos atributos que los cinco modelos con los que
    comparte tabla. Sin ellos era el unico Pipeline del catalogo cuyo
    ``n_features_in_`` lanzaba ``AttributeError``, de modo que cualquier codigo
    generico que recorriera :data:`MODELOS` se rompia solo en su fila.

    ``predict`` repite la validacion con ``reset=False``, que comprueba que
    ``X`` traiga las mismas columnas del ajuste. Que la prediccion sea siempre
    cero no hace inofensivo el desajuste: la referencia decide cuantas filas
    devuelve, y un vector de una dimension o una tabla de otras columnas daban
    en silencio una serie de ceros de la longitud equivocada, que luego se pega
    por posicion contra los valores realizados.

    La validacion eleva ``ValueError`` de scikit-learn y no ``ErrorDatos``
    porque aqui el contrato que se cumple es el de estimador: ``Pipeline``,
    ``GridSearchCV`` y ``check_estimator`` esperan esa excepcion, y quien llama
    lo hace a traves de ellos y no de la API de ``icm``.
    """

    def fit(self, X: Any, y: Any = None) -> MediaCero:
        # Se permiten NaN porque 'arbol_potenciado' los admite y
        # comparar_modelos anade siempre la referencia: si fuese mas estricta
        # que los modelos a los que sirve de linea de flotacion, la tabla no se
        # podria calcular con las entradas con las que si se calculan ellos.
        if y is None:
            X = validate_data(self, X, ensure_all_finite="allow-nan")
        else:
            X, _ = validate_data(
                self, X, y, ensure_all_finite="allow-nan", y_numeric=True
            )
        self.n_variables_ = self.n_features_in_
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return np.zeros(_n_filas(X), dtype=float)


# --- utilidades internas -------------------------------------------------------


def _n_filas(datos: Any, nombre: str = "X") -> int:
    """Numero de observaciones de la entrada, sea pandas o numpy."""
    if isinstance(datos, (pd.DataFrame, pd.Series)):
        return len(datos.index)
    valores = np.asarray(datos)
    if valores.ndim == 0:
        raise ErrorDatos(f"'{nombre}' debe ser una tabla o un vector, no un escalar.")
    return int(valores.shape[0])


def _indice(datos: Any, nombre: str) -> pd.Index:
    """Indice de la entrada; posiciones 0..n-1 si no tiene ninguno."""
    if isinstance(datos, (pd.DataFrame, pd.Series)):
        return datos.index
    return pd.RangeIndex(_n_filas(datos, nombre))


def _envolver(estimador: Any, escalar: bool) -> Pipeline:
    """Mete el estimador en un Pipeline, con o sin escalado previo."""
    pasos: list[tuple[str, Any]] = []
    if escalar:
        pasos.append((PASO_ESCALADO, StandardScaler()))
    pasos.append((PASO_MODELO, estimador))
    return Pipeline(pasos)


def _validar_nombre(nombre: Any) -> str:
    """Comprueba que el nombre sea uno de :data:`MODELOS`."""
    if not isinstance(nombre, str):
        raise ErrorDatos(
            f"El nombre del modelo debe ser una cadena, se ha recibido "
            f"{type(nombre).__name__}. Disponibles: {', '.join(MODELOS)}."
        )
    if nombre not in MODELOS:
        raise ErrorDatos(
            f"Modelo '{nombre}' desconocido. Disponibles: {', '.join(MODELOS)}."
        )
    return nombre


def _forma(datos: Any) -> tuple[int, ...]:
    """Forma de la entrada, sea pandas, numpy o una lista."""
    forma = getattr(datos, "shape", None)
    if forma is not None:
        return tuple(int(medida) for medida in forma)
    return tuple(int(medida) for medida in np.asarray(datos).shape)


def _exigir_objetivo_unidimensional(y: Any, sufijo: str = "") -> None:
    """Rechaza un objetivo que no sea una sola columna de numeros.

    ``df[["rend_futuro_1"]]`` en lugar de ``df["rend_futuro_1"]`` es el descuido
    mas frecuente de todos, y el mas silencioso: scikit-learn acepta la matriz de
    una columna aplanandola, asi que el modelo se ajusta y devuelve predicciones
    de aspecto impecable. Se exige aqui lo mismo que en
    :func:`icm.ml.validacion.validacion_walk_forward`, para que las dos puertas
    de entrada del repositorio pidan el objetivo en la misma forma.
    """
    forma = _forma(y)
    if len(forma) == 1:
        return
    raise ErrorDatos(
        f"'y{sufijo}' debe ser unidimensional; tiene forma {forma}. Seleccione una "
        f"columna antes de ajustar: y{sufijo}['rend_futuro_1'], no "
        f"y{sufijo}[['rend_futuro_1']]."
    )


def _alinear(X: Any, y: Any, sufijo: str = "") -> pd.Index:
    """Indice comun de ``X`` e ``y``, comprobando que de verdad lo sea.

    La particion y el emparejamiento se hacen por posicion, asi que dos entradas
    de la misma longitud pero de periodos distintos no fallarian: simplemente
    compararian la variable de una fecha con el objetivo de otra y darian un
    resultado sin sentido pero de aspecto impecable.

    La comparacion de indices cubre tambien un ``y`` que llegue como
    ``DataFrame``. Mirar solo las ``Series`` dejaba el guardian desactivado justo
    en el caso en que el usuario ya esta manejando columnas, que es cuando se
    equivoca: con un objetivo de otro periodo en un ``DataFrame`` de una columna
    el ajuste salia adelante sin un solo aviso y con la relacion real borrada.
    """
    indice_X = _indice(X, f"X{sufijo}")
    indice_y = _indice(y, f"y{sufijo}")
    if len(indice_X) == 0:
        raise ErrorDatos(f"'X{sufijo}' no contiene ninguna observacion.")
    if len(indice_X) != len(indice_y):
        raise ErrorDatos(
            f"'X{sufijo}' tiene {len(indice_X)} filas e 'y{sufijo}' tiene "
            f"{len(indice_y)}: alinee ambos con un unico 'dropna' conjunto antes "
            "de ajustar."
        )
    if isinstance(X, (pd.DataFrame, pd.Series)) and isinstance(y, (pd.Series, pd.DataFrame)):
        if not indice_X.equals(indice_y):
            raise ErrorDatos(
                f"'X{sufijo}' e 'y{sufijo}' tienen la misma longitud pero indices "
                "distintos: reindexelos al indice comun antes de ajustar, porque el "
                "emparejamiento es posicional y cruzaria fechas."
            )
    _exigir_objetivo_unidimensional(y, sufijo)
    return indice_X


def _validar_columnas(X_entrena: Any, X_prueba: Any) -> None:
    """Exige las mismas columnas y en el mismo orden en ambas matrices."""
    if not (isinstance(X_entrena, pd.DataFrame) and isinstance(X_prueba, pd.DataFrame)):
        return
    if list(X_entrena.columns) != list(X_prueba.columns):
        faltan = [c for c in X_entrena.columns if c not in X_prueba.columns]
        sobran = [c for c in X_prueba.columns if c not in X_entrena.columns]
        detalle = (
            f"faltan {faltan} y sobran {sobran}"
            if (faltan or sobran)
            else "estan en las dos pero en otro orden"
        )
        raise ErrorDatos(
            f"'X_prueba' no tiene las mismas variables que 'X_entrena': {detalle}. "
            "Reindexe con X_prueba.reindex(columns=X_entrena.columns): un modelo "
            "lineal aplicaria cada coeficiente a la variable equivocada sin avisar."
        )


def _contrastar_orden_temporal(X_entrena: Any, X_prueba: Any) -> None:
    """Rechaza predecir fechas que ya estaban en el tramo de entrenamiento.

    Es el error que produce los resultados espectaculares: se ajusta con toda la
    muestra, se predice sobre toda la muestra y se presenta el ajuste como
    capacidad predictiva. Aqui salta antes de gastar un segundo de calculo.

    Para medir el ajuste dentro de muestra a proposito (que es legitimo como
    diagnostico, no como resultado) se llama directamente a ``modelo.fit`` y
    ``modelo.predict``, que no llevan esta comprobacion.

    Con matrices de numpy no se comprueba nada, y no por dejadez: sin indice no
    hay fechas: las dos matrices empiezan en la posicion 0 y su orden temporal
    relativo es indeterminable. Quien quiera esta garantia (que es la que
    convierte un resultado en creible) trabaja con pandas, que es como salen las
    matrices de :mod:`icm.ml.caracteristicas`.
    """
    if not (
        isinstance(X_entrena, (pd.DataFrame, pd.Series))
        and isinstance(X_prueba, (pd.DataFrame, pd.Series))
    ):
        return
    indice_entrena, indice_prueba = X_entrena.index, X_prueba.index
    if len(indice_entrena) == 0 or len(indice_prueba) == 0:
        return
    try:
        se_solapa = bool(indice_prueba.min() <= indice_entrena.max())
    except TypeError:  # indices de tipos que no se pueden comparar entre si
        return
    if se_solapa:
        raise ErrorDatos(
            f"'X_prueba' empieza en {indice_prueba.min()}, que no es posterior al "
            f"final del entrenamiento ({indice_entrena.max()}): se estaria prediciendo "
            "sobre fechas ya vistas. Use icm.ml.validacion.ParticionWalkForward para "
            "partir la muestra hacia delante."
        )


# --- fabricas de modelos -------------------------------------------------------


def _crear_ridge(alpha: float = 1.0, **parametros: Any) -> Pipeline:
    """Regresion lineal con penalizacion cuadratica (cresta).

    Es la referencia lineal honesta del modulo. La penalizacion cuadratica no
    anula coeficientes, los encoge todos, que es lo adecuado cuando las
    variables estan muy correlacionadas entre si (retardos de la misma serie,
    medias moviles de ventanas contiguas): ahi los minimos cuadrados ordinarios
    reparten el efecto entre variables casi identicas con coeficientes enormes y
    de signo alterno, y el modelo se vuelve inestable de una particion a otra
    sin perder ajuste dentro de muestra.

    El ``alpha`` por defecto es deliberadamente suave: con las variables
    tipificadas y unos cientos de observaciones apenas encoge nada, de modo que
    el modelo recupera una relacion lineal real si existe. El valor que conviene
    de verdad sale de :func:`buscar_hiperparametros`.
    """
    return _envolver(Ridge(alpha=alpha, **parametros), escalar=True)


def _crear_lasso(alpha: float = 1e-4, max_iter: int = 5000, **parametros: Any) -> Pipeline:
    """Regresion lineal con penalizacion absoluta: selecciona variables.

    A diferencia de la cresta, el lasso lleva coeficientes exactamente a cero,
    asi que responde a una pregunta distinta y util: de las cincuenta variables
    construidas, cuales sobreviven. Con variables muy correlacionadas elige una
    del grupo casi al azar, asi que la lista de supervivientes no se debe leer
    como "estas son las importantes"; para eso esta
    :func:`importancia_variables`.

    El ``alpha`` por defecto no es el 1.0 de scikit-learn, y el motivo importa.
    La penalizacion compite con el error cuadratico medio, que en rendimientos
    logaritmicos diarios es del orden de 1e-4. Con alpha 1.0 el termino de
    penalizacion aplasta todos los coeficientes a cero, el modelo degenera en
    una constante, no salta ningun aviso y la conclusion que se saca es "el
    lasso no encuentra nada", cuando lo que ocurre es que no se le ha dejado
    buscar.
    """
    return _envolver(Lasso(alpha=alpha, max_iter=max_iter, **parametros), escalar=True)


def _crear_elastic_net(
    alpha: float = 1e-4,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    **parametros: Any,
) -> Pipeline:
    """Combinacion de las penalizaciones absoluta y cuadratica.

    Es el termino medio razonable cuando hay grupos de variables muy
    correlacionadas: la parte cuadratica reparte el peso dentro del grupo en vez
    de quedarse con un representante arbitrario, y la absoluta sigue
    descartando los grupos que no aportan. La misma advertencia sobre la escala
    de ``alpha`` que en :func:`_crear_lasso`.
    """
    return _envolver(
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, **parametros),
        escalar=True,
    )


def _crear_arbol_potenciado(
    learning_rate: float = 0.05,
    max_iter: int = 200,
    max_leaf_nodes: int = 15,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    **parametros: Any,
) -> Pipeline:
    """Potenciado de gradiente por histogramas: el modelo principal.

    Cada arbol corrige los errores del anterior, asi que el conjunto captura
    interacciones (el efecto del gas condicionado al dia de la semana) y umbrales
    (la volatilidad solo importa cuando pasa de cierto nivel) que una regresion
    lineal solo puede recoger si alguien las escribe a mano como variables.

    Los valores por defecto son conservadores a proposito, porque con
    rendimientos diarios la relacion senal-ruido es pesima y el sobreajuste es
    el estado natural: tasa de aprendizaje baja con muchas iteraciones (muchos
    pasos pequenios generalizan mejor que pocos grandes), hojas pocas y con un
    minimo de veinte observaciones (una hoja de tres dias memoriza tres dias) y
    penalizacion L2 activa.

    ``early_stopping`` se desactiva explicitamente, y es la decision menos obvia
    de la funcion. Activado, scikit-learn aparta un tramo de validacion con
    ``train_test_split`` **aleatorio**: dias de todo el periodo, entremezclados.
    En una serie temporal eso es exactamente la fuga que
    :mod:`icm.ml.validacion` existe para impedir, y ademas ocurriria dentro de
    cada particion, donde no se ve. Sin parada temprana el numero de
    iteraciones es un hiperparametro mas, que se elige hacia delante con
    :func:`buscar_hiperparametros`.
    """
    return _envolver(
        HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=SEMILLA,
            **parametros,
        ),
        escalar=False,
    )


def _crear_bosque(
    n_estimators: int = 200,
    min_samples_leaf: int = 20,
    max_features: float = 0.5,
    **parametros: Any,
) -> Pipeline:
    """Bosque aleatorio: muchos arboles independientes, promediados.

    Frente al potenciado tiene una virtud concreta en este dominio: al promediar
    arboles independientes reduce varianza sin perseguir el residuo, asi que con
    datos muy ruidosos se sobreajusta de forma menos agresiva y da una segunda
    opinion no lineal que no comparte el modo de fallar del potenciado. A cambio
    suele quedarse corto cuando hay estructura fina que aprovechar.

    ``max_features`` por debajo de 1 es lo que de verdad hace que el bosque sea
    un bosque: si cada arbol puede mirar todas las variables en cada corte,
    todos eligen los mismos cortes, quedan muy correlacionados entre si y el
    promedio deja de reducir varianza.
    """
    return _envolver(
        RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=SEMILLA,
            **parametros,
        ),
        escalar=False,
    )


def _crear_media_cero(**parametros: Any) -> Pipeline:
    """La referencia, envuelta en Pipeline como los demas.

    Va en un Pipeline aunque no tenga nada que transformar, para que no haya ni
    una diferencia de interfaz entre la referencia y los modelos que pretenden
    batirla: lo que se le pueda hacer a uno se le puede hacer a la otra.
    """
    if parametros:
        raise ErrorDatos(
            f"'{NOMBRE_REFERENCIA}' no tiene hiperparametros que ajustar (predice "
            f"siempre 0.0); se han recibido {sorted(parametros)}."
        )
    return _envolver(MediaCero(), escalar=False)


#: Fabricas de estimadores, por nombre. Cada valor es una funcion que acepta
#: hiperparametros por nombre y devuelve un ``Pipeline`` sin ajustar.
MODELOS: dict[str, Callable[..., Pipeline]] = {
    "ridge": _crear_ridge,
    "lasso": _crear_lasso,
    "elastic_net": _crear_elastic_net,
    "arbol_potenciado": _crear_arbol_potenciado,
    "bosque": _crear_bosque,
    NOMBRE_REFERENCIA: _crear_media_cero,
}

#: Rejillas por defecto de :func:`buscar_hiperparametros`. Deliberadamente
#: cortas: cada combinacion se ajusta una vez por particion, y una rejilla de
#: cien puntos explorada sobre cinco particiones no es una busqueda cuidadosa,
#: es una forma lenta de sobreajustar la propia validacion.
REJILLAS: dict[str, dict[str, Sequence[Any]]] = {
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"alpha": [1e-6, 1e-5, 1e-4, 1e-3]},
    "elastic_net": {"alpha": [1e-5, 1e-4, 1e-3], "l1_ratio": [0.2, 0.5, 0.8]},
    "arbol_potenciado": {
        "learning_rate": [0.02, 0.05, 0.1],
        "max_leaf_nodes": [7, 15, 31],
    },
    "bosque": {"min_samples_leaf": [10, 20, 50], "max_features": [0.3, 0.5, 1.0]},
    NOMBRE_REFERENCIA: {},
}


# --- construccion --------------------------------------------------------------


def crear_modelo(nombre: str, **parametros: Any) -> Pipeline:
    """Crea un modelo sin ajustar, listo para validar.

    Todos los modelos vuelven envueltos en un ``Pipeline``, y los lineales
    llevan dentro un ``StandardScaler``. Los arboles no lo necesitan: parten por
    umbrales sobre cada variable por separado, asi que cualquier
    transformacion monotona de una columna da exactamente los mismos cortes.
    Aniadirselo no mejoraria nada y crearia estado ajustado de mas.

    Por que el escalado va DENTRO del Pipeline
    ------------------------------------------
    Tipificar antes, sobre la matriz completa, es la fuga de informacion mas
    extendida del aprendizaje automatico aplicado a series temporales, y la mas
    dificil de ver, porque no toca la variable objetivo y no deja rastro en el
    codigo: una linea limpia, ``X = (X - X.mean()) / X.std()``, ejecutada una
    vez al principio.

    El problema es de quien es esa media. La media y la desviacion de toda la
    muestra resumen tambien las filas del futuro, las que cada particion de
    validacion va a usar como prueba. Al restarlas, el valor tipificado de la
    fecha ``t`` pasa a depender de lo que ocurrio despues de ``t``: si el ultimo
    trimestre fue muy volatil, la desviacion global es mayor y **todas** las
    observaciones anteriores quedan tipificadas hacia abajo. El modelo entrenado
    con ellas conoce, de forma difusa pero real, el regimen de volatilidad de su
    propio periodo de prueba. El resultado tipico es un error fuera de muestra
    entre un 5 % y un 20 % mejor que el verdadero, suficiente para dar por bueno
    un modelo que en produccion no funciona, e insuficiente para levantar
    sospechas.

    Dentro del ``Pipeline`` no puede pasar: ``fit`` calcula la media y la
    desviacion **solo** con el tramo de entrenamiento de esa particion, y
    ``predict`` aplica esas mismas cifras al tramo de prueba, que es exactamente
    lo que se podria hacer en produccion, donde el futuro todavia no existe.
    Esa es la razon de que :func:`icm.ml.caracteristicas.construir_matriz` no
    escale nada y lo deje escrito.

    Parametros
    ----------
    nombre:
        Una de las claves de :data:`MODELOS`.
    parametros:
        Hiperparametros del estimador, por nombre y sin el prefijo del paso del
        Pipeline: ``crear_modelo("ridge", alpha=10.0)``. Los que no se indiquen
        toman los valores por defecto de la fabrica correspondiente, que estan
        elegidos para rendimientos diarios y no son los de scikit-learn.

    Lanza
    -----
    ErrorDatos
        Si el nombre no esta en :data:`MODELOS`, o si el estimador no acepta
        alguno de los hiperparametros indicados.
    """
    nombre = _validar_nombre(nombre)
    try:
        return MODELOS[nombre](**parametros)
    except TypeError as exc:
        raise ErrorDatos(
            f"El modelo '{nombre}' no acepta alguno de estos hiperparametros: "
            f"{sorted(parametros)} ({exc}). Consulte la firma de la fabrica en "
            f"MODELOS['{nombre}'] o la documentacion del estimador de scikit-learn."
        ) from exc


# --- ajuste y prediccion -------------------------------------------------------


def ajustar_y_predecir(
    modelo: Any,
    X_entrena: pd.DataFrame,
    y_entrena: pd.Series,
    X_prueba: pd.DataFrame,
) -> pd.Series:
    """Ajusta con el pasado y predice el tramo siguiente, devolviendo una Series.

    El valor de la funcion no esta en las dos lineas de scikit-learn que
    envuelve, sino en las cuatro comprobaciones que hace antes: que ``X`` e
    ``y`` esten alineados de verdad y no solo tengan la misma longitud, que las
    columnas de prueba sean las mismas y en el mismo orden (un lineal aplicaria
    cada coeficiente a la variable equivocada sin protestar) y que el tramo de
    prueba sea **posterior** al de entrenamiento. Ninguno de esos tres fallos
    produce una excepcion por si mismo; los tres producen numeros mejores de lo
    que corresponde.

    Devolver una ``Series`` indexada por las fechas de ``X_prueba``, y no un
    ``ndarray``, tiene el mismo motivo: la prediccion viaja despues a
    :mod:`icm.ml.evaluacion` y a la cuenta de resultados de una estrategia, y
    ahi un vector sin fechas se acaba emparejando por posicion con el
    rendimiento realizado de otro dia.

    El modelo recibido **queda ajustado**, siguiendo el convenio de
    scikit-learn, de modo que se puede pasar despues a
    :func:`importancia_variables` sin volver a estimarlo.
    """
    _alinear(X_entrena, y_entrena, sufijo="_entrena")
    indice_prueba = _indice(X_prueba, "X_prueba")
    if len(indice_prueba) == 0:
        raise ErrorDatos("'X_prueba' no contiene ninguna fila que predecir.")
    _validar_columnas(X_entrena, X_prueba)
    _contrastar_orden_temporal(X_entrena, X_prueba)

    tiene_interfaz = callable(getattr(modelo, "fit", None)) and callable(
        getattr(modelo, "predict", None)
    )
    if not tiene_interfaz:
        raise ErrorDatos(
            f"El objeto {type(modelo).__name__} no tiene 'fit' y 'predict': cree el "
            "modelo con crear_modelo() o pase un estimador de scikit-learn."
        )

    modelo.fit(X_entrena, y_entrena)
    crudas = np.asarray(modelo.predict(X_prueba), dtype=float).reshape(-1)
    if crudas.size != len(indice_prueba):
        raise ErrorDatos(
            f"El modelo ha devuelto {crudas.size} predicciones para "
            f"{len(indice_prueba)} fechas: 'predict' debe devolver una por fila."
        )
    return pd.Series(crudas, index=indice_prueba, name="prediccion")


# --- importancia de variables --------------------------------------------------


def importancia_variables(
    modelo: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeticiones: int = 10,
    semilla: int = SEMILLA,
    dentro_de_muestra: bool = False,
) -> pd.Series:
    """Importancia por permutacion: cuanto empeora el modelo al romper cada variable.

    El procedimiento es el unico que mide lo que interesa: se baraja **una**
    columna, se vuelve a predecir y se mira cuanto se degrada el ajuste. Si la
    variable no aportaba nada, no pasa nada. Es comparable entre modelos
    (lineales y arboles se miden con la misma regla) y esta en unidades
    interpretables, la perdida de R cuadrado.

    Por que no se usa la importancia interna de los arboles
    -------------------------------------------------------
    ``feature_importances_`` mide la reduccion de impureza acumulada, y esa
    cifra esta sesgada hacia las variables con muchos valores distintos. El
    motivo es mecanico: una columna continua ofrece cientos de umbrales de corte
    posibles, asi que aunque sea ruido puro alguno de ellos reducira la impureza
    por azar, y ese premio se acumula arbol tras arbol. Un indicador binario
    ofrece un unico corte y no tiene esa oportunidad.

    En la matriz de :mod:`icm.ml.caracteristicas` esa mezcla es la norma: siete
    indicadores de dia de la semana y dos armonicos anuales conviven con
    volatilidades, momentos y distancias a medias moviles, todas continuas. El
    ranking por impureza coloca sistematicamente las continuas arriba y las de
    calendario abajo, y en energia el dia de la semana es una de las variables
    que de verdad explica el precio. No es un matiz academico: es el ranking
    dado la vuelta.

    Limitaciones que conviene tener presentes
    ----------------------------------------
    Con dos variables muy correlacionadas (dos retardos contiguos, dos medias
    moviles de ventanas parecidas) la permutacion reparte mal el credito: al
    romper una, la otra sostiene la prediccion y ambas parecen prescindibles.
    Lee, por tanto, "cuanto aporta esta variable **dado que estan las demas**",
    no "cuanto explica esta variable".

    Parametros
    ----------
    modelo:
        Tiene que llegar **ya ajustado**, y con ``(X, y)`` un tramo que el modelo
        no haya visto: esa es la importancia honesta, la que dice de que se fia
        el modelo cuando acierta de verdad. El encadenamiento previsto es
        :func:`ajustar_y_predecir` con el pasado y despues esta funcion con el
        tramo de prueba.
    n_repeticiones:
        Permutaciones por variable. La importancia es una media sobre
        barajados; con pocas repeticiones el ranking baila de una semilla a
        otra.
    dentro_de_muestra:
        Permite ajustar aqui mismo un modelo que llegue sin entrenar y permutar
        sobre ese mismo ``(X, y)``. Hay que pedirlo expresamente porque la cifra
        que sale entonces no es importancia predictiva, es cuanto memorizo el
        modelo: con ``X`` e ``y`` independientes, un arbol potenciado atribuye
        cerca de 0,3 de R cuadrado a **cada** variable cuando la respuesta
        correcta es ruido alrededor de cero. Es legitimo como diagnostico de
        sobreajuste, nunca como respuesta a "que mira el modelo". El resultado
        vuelve marcado y con un aviso.

    Devuelve
    --------
    Series indexada por nombre de variable y ordenada de mayor a menor. Los
    valores negativos no son un error: significan que el modelo predice mejor
    con esa columna barajada, es decir, que la estaba usando como ruido.

    ``attrs['dentro_de_muestra']`` dice si la cifra se midio sobre el mismo
    tramo con el que esta funcion ajusto el modelo. Vale ``False`` cuando el
    modelo llego ajustado, y entonces que la medida sea fuera de muestra depende
    de que quien llama haya pasado un tramo no visto: eso no se puede comprobar
    desde aqui, porque un estimador de scikit-learn no guarda las fechas con las
    que se ajusto.

    Lanza
    -----
    ErrorDatos
        Si el modelo llega sin ajustar y no se ha pedido ``dentro_de_muestra``.
    """
    if not isinstance(X, pd.DataFrame):
        raise ErrorDatos(
            "'X' debe ser un DataFrame para poder etiquetar la importancia con el "
            "nombre de cada variable; se ha recibido "
            f"{type(X).__name__}."
        )
    if not isinstance(n_repeticiones, (int, np.integer)) or isinstance(n_repeticiones, bool):
        raise ErrorDatos(
            f"'n_repeticiones' debe ser un entero, se ha recibido {n_repeticiones!r}."
        )
    if n_repeticiones < 1:
        raise ErrorDatos(
            f"'n_repeticiones' debe ser al menos 1, se ha recibido {n_repeticiones}."
        )
    _alinear(X, y)

    try:
        check_is_fitted(modelo)
        ajustado_aqui = False
    except NotFittedError:
        ajustado_aqui = True
    except TypeError as exc:
        raise ErrorDatos(
            f"El objeto {type(modelo).__name__} no es un estimador de scikit-learn: "
            f"{exc}. Creelo con crear_modelo()."
        ) from exc

    if ajustado_aqui and not dentro_de_muestra:
        raise ErrorDatos(
            f"El modelo {type(modelo).__name__} llega sin ajustar: ajustarlo con "
            "(X, y) y permutar sobre ese mismo tramo no mide de que se fia el modelo, "
            "mide cuanto ha memorizado, y sobre variables sin ninguna relacion con el "
            "objetivo devuelve importancias grandes. Ajustelo con el pasado "
            "(ajustar_y_predecir) y pase aqui un tramo posterior, o pase "
            "dentro_de_muestra=True si lo que quiere es el diagnostico dentro de "
            "muestra."
        )
    ajustado = modelo
    if ajustado_aqui:
        ajustado = modelo.fit(X, y)
        warnings.warn(
            "importancia_variables ha ajustado el modelo con (X, y) y permuta sobre "
            "ese mismo tramo: la cifra es importancia DENTRO de muestra y no dice "
            "que mira el modelo fuera de ella. Queda marcada en "
            "attrs['dentro_de_muestra'].",
            UserWarning,
            stacklevel=2,
        )

    resultado = permutation_importance(
        ajustado,
        X,
        y,
        scoring="r2",
        n_repeats=int(n_repeticiones),
        random_state=semilla,
    )
    importancias = pd.Series(
        resultado.importances_mean,
        index=X.columns,
        name="importancia",
    )
    importancias = importancias.sort_values(ascending=False)
    importancias.attrs["dentro_de_muestra"] = ajustado_aqui
    return importancias


# --- busqueda de hiperparametros -----------------------------------------------


class ParticionSklearn:
    """Adaptador que hace pasar a :class:`ParticionWalkForward` por un ``cv``.

    Existe por una razon defensiva. ``GridSearchCV`` acepta un entero en ``cv``,
    y ese atajo es la puerta de entrada al peor error posible en series
    temporales: ``cv=5`` significa ``KFold`` **aleatorio** en regresion, es
    decir, entrenar con 2023 para predecir 2019. El resultado es una busqueda
    que elige los hiperparametros que mejor memorizan el futuro, y como los
    numeros salen mejor, nadie lo revisa.

    Asi que aqui nunca se escribe un entero. Cuando hace falta un objeto con la
    interfaz de scikit-learn, se **adapta** la particion del repositorio: este
    envoltorio solo traduce ``dividir`` a ``split`` y aniade
    ``get_n_splits``. No cambia una sola posicion, y los tramos siguen avanzando
    hacia delante en el tiempo con su embargo.
    """

    def __init__(self, particion: ParticionWalkForward | None = None) -> None:
        if particion is not None and not isinstance(particion, ParticionWalkForward):
            raise ErrorDatos(
                "'particion' debe ser un ParticionWalkForward. No sustituya la "
                "particion por un KFold: barajar una serie temporal entrena con el "
                "futuro."
            )
        # Aqui no hay objetivo del que deducir el embargo: este envoltorio solo
        # traduce la particion que le den. Quien lo construye
        # (buscar_hiperparametros) ya ha resuelto el suyo con
        # particion_por_defecto antes de llegar aqui.
        self.particion = particion if particion is not None else ParticionWalkForward()

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[
        tuple[np.ndarray, np.ndarray]
    ]:
        return self.particion.dividir(X)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        if X is None:
            return int(self.particion.n_particiones)
        return int(self.particion.n_divisiones(X))


def _recortar(datos: Any, n: int) -> Any:
    """Primeras ``n`` filas de la entrada, sea pandas o numpy."""
    if isinstance(datos, (pd.DataFrame, pd.Series)):
        return datos.iloc[:n]
    return np.asarray(datos)[:n]


def _tramo_de_seleccion(
    particion: ParticionWalkForward, X: Any
) -> tuple[int, ParticionWalkForward]:
    """Filas utilizables para elegir hiperparametros, y con que particion.

    Devuelve cuantas filas iniciales puede ver la busqueda (las del
    entrenamiento de la **primera** particion de evaluacion, que ya excluyen el
    embargo) y una particion interna que quepa en ellas.

    La particion interna hereda ``embargo`` y ``expansiva`` de la de evaluacion,
    porque el criterio con el que se elige debe parecerse al criterio con el que
    luego se juzga, y reescala el entrenamiento minimo en la misma proporcion en
    que se ha recortado la muestra: el minimo de la particion de evaluacion esta
    pensado para la serie entera y dejaria sin sitio a los tramos internos.

    Del numero de particiones se toma el de la de evaluacion, pero bajando hasta
    el mayor que quepa en el tramo de seleccion. Reducir tramos no compromete
    nada (todos siguen midiendo hacia delante y ninguno toca las fechas del
    informe); quedarse sin ninguno si. Solo cuando no cabe ni una se lanza
    ``ErrorDatos``, porque entonces no hay forma honesta de elegir.
    """
    tramos = list(particion.dividir(X))
    fin_seleccion = int(tramos[0][0][-1]) + 1
    n = _n_filas(X)
    X_seleccion = _recortar(X, fin_seleccion)
    minimo = max(1, round(particion.tamano_entrenamiento_minimo * fin_seleccion / n))
    motivo: ErrorDatos | None = None
    for n_particiones in range(particion.n_particiones, 0, -1):
        interna = ParticionWalkForward(
            n_particiones=n_particiones,
            tamano_prueba=None,
            embargo=particion.embargo,
            expansiva=particion.expansiva,
            tamano_entrenamiento_minimo=minimo,
        )
        try:
            interna.n_divisiones(X_seleccion)
        except ErrorDatos as exc:
            motivo = exc
            continue
        return fin_seleccion, interna
    raise ErrorDatos(
        f"La busqueda solo puede mirar las {fin_seleccion} primeras observaciones de "
        f"{n}, las anteriores al primer tramo de prueba de 'particion', y ahi no cabe "
        f"ni una sola particion interna: {motivo} Alargue la serie, o reduzca "
        "'tamano_prueba' o 'embargo' en la particion de evaluacion para que quede mas "
        "historia antes de su primer tramo de prueba."
    ) from motivo


def buscar_hiperparametros(
    nombre: str,
    X: pd.DataFrame,
    y: pd.Series,
    rejilla: dict[str, Sequence[Any]] | None = None,
    particion: ParticionWalkForward | None = None,
) -> dict:
    """Elige hiperparametros por rejilla, validando siempre hacia delante.

    La busqueda se hace con :class:`ParticionSklearn`, que envuelve la particion
    *walk forward* del repositorio; vease ahi por que nunca se usa un ``KFold``
    aleatorio ni el atajo ``cv=5``.

    Que tramo ve la busqueda
    ------------------------
    Solo las fechas **anteriores** al primer tramo de prueba de ``particion``, y
    sobre ellas una particion interna hacia delante. La razon es que
    ``ParticionWalkForward`` pega sus tramos de prueba al final de la muestra, o
    sea exactamente las fechas que :func:`comparar_modelos` y
    :func:`icm.ml.validacion.validacion_walk_forward` presentan despues como
    resultado fuera de muestra. Buscar sobre la particion completa elegiria la
    combinacion que minimiza el error **en** el tramo que luego se reporta, que
    es la forma clasica de elegir mirando el conjunto de prueba: alterando
    ``y`` solo en el ultimo tramo de prueba, con el entrenamiento intacto, la
    eleccion de ``alpha`` saltaba de 1e-06 a 1e+06.

    Es un problema distinto del de ``refit``, y por eso no bastaba con avisar de
    aquel: ``refit`` afecta al ajuste final, mientras que aqui se contaminaba la
    **seleccion**. Con el recorte, el encadenamiento de mas abajo es legitimo:
    los hiperparametros se han elegido sin ver ninguna de las fechas sobre las
    que se va a informar.

    Se puntua con el error cuadratico medio negativo y no con el R cuadrado, que
    es lo que usaria scikit-learn por omision. El R cuadrado de cada tramo se
    calcula contra la media de **ese mismo tramo**, que en la fecha de la
    decision no se conoce; como referencia movil ademas hace que los tramos
    tranquilos y los agitados no pesen igual. El error cuadratico es
    comparable entre tramos y es lo que de verdad se quiere minimizar.

    No se reajusta el modelo al final (``refit=False``): lo que se pide aqui son
    los hiperparametros, y quien los recibe decide con que tramo ajustar. Un
    modelo devuelto ya ajustado, ademas, estaria ajustado solo con el tramo de
    seleccion, que es una parte de la muestra elegida por esta funcion y no por
    quien la llama.

    Parametros
    ----------
    rejilla:
        ``{nombre_de_hiperparametro: valores}`` **sin** el prefijo del paso del
        Pipeline: ``{"alpha": [0.1, 1.0]}``, no ``{"modelo__alpha": ...}``. Si se
        omite se usa la de :data:`REJILLAS`.
    particion:
        La particion con la que se va a **evaluar** despues. No se usa tal cual:
        de ella se toma el inicio de su primer tramo de prueba como frontera de
        lo que la busqueda puede mirar, y su forma para construir la particion
        interna. Pasar aqui la misma que se pasa a :func:`comparar_modelos` es
        lo correcto y lo previsto.
        Si se omite, se construye con :func:`particion_por_defecto`, que deduce
        el embargo del horizonte marcado en ``y``.

    Devuelve
    --------
    Diccionario de hiperparametros, con los nombres tal como los acepta
    :func:`crear_modelo`, de modo que se puede encadenar con la particion de
    evaluacion sin quemarla:

    >>> particion = ParticionWalkForward(n_particiones=5, embargo=1)  # doctest: +SKIP
    >>> mejores = buscar_hiperparametros("ridge", X, y, particion=particion)  # doctest: +SKIP
    >>> modelo = crear_modelo("ridge", **mejores)                     # doctest: +SKIP
    >>> resultado = validacion_walk_forward(modelo, X, y, particion)  # doctest: +SKIP

    Para ``"media_cero"`` devuelve un diccionario vacio: no hay nada que elegir,
    y eso es precisamente su virtud como referencia.

    Lanza
    -----
    ErrorDatos
        Si antes del primer tramo de prueba de ``particion`` no queda historia
        suficiente para una particion interna.
    """
    nombre = _validar_nombre(nombre)
    _alinear(X, y)
    rejilla = REJILLAS.get(nombre, {}) if rejilla is None else rejilla
    if not isinstance(rejilla, dict):
        raise ErrorDatos(
            "'rejilla' debe ser un diccionario {hiperparametro: valores}, se ha "
            f"recibido {type(rejilla).__name__}."
        )

    modelo = crear_modelo(nombre)
    prefijo = f"{PASO_MODELO}__"
    admitidos = {
        clave.removeprefix(prefijo)
        for clave in modelo.get_params(deep=True)
        if clave.startswith(prefijo)
    }

    rejilla_pipeline: dict[str, list[Any]] = {}
    for clave, valores in rejilla.items():
        # Se comprueba el nombre antes de ajustar nada: scikit-learn tambien lo
        # detecta, pero a mitad de la busqueda y sin decir que si existe.
        if clave not in admitidos:
            raise ErrorDatos(
                f"El modelo '{nombre}' no tiene el hiperparametro '{clave}'. "
                f"Disponibles: {', '.join(sorted(admitidos))}."
            )
        if isinstance(valores, (str, bytes)) or not isinstance(valores, Sequence):
            raise ErrorDatos(
                f"Los valores de '{clave}' en la rejilla deben ser una lista, se ha "
                f"recibido {valores!r}. Use {{'{clave}': [{valores!r}]}} para un unico valor."
            )
        if len(valores) == 0:
            raise ErrorDatos(f"La rejilla de '{clave}' esta vacia: indique al menos un valor.")
        rejilla_pipeline[f"{prefijo}{clave}"] = list(valores)

    particion = particion if particion is not None else particion_por_defecto(y)
    fin_seleccion, interna = _tramo_de_seleccion(particion, X)
    busqueda = GridSearchCV(
        modelo,
        rejilla_pipeline,
        scoring="neg_mean_squared_error",
        cv=ParticionSklearn(interna),
        refit=False,
    )
    busqueda.fit(_recortar(X, fin_seleccion), _recortar(y, fin_seleccion))
    return {
        clave.removeprefix(prefijo): valor for clave, valor in busqueda.best_params_.items()
    }


# --- comparacion ---------------------------------------------------------------


def particion_por_defecto(y: Any) -> ParticionWalkForward:
    """Particion walk-forward con el embargo deducido del propio objetivo.

    Una particion construida sin argumentos deja ``embargo=0``, y eso
    contradice la regla que el propio :class:`ParticionWalkForward` documenta:
    con un objetivo a ``h`` periodos, las ultimas ``h`` observaciones de
    entrenamiento tienen su resultado medido dentro del tramo de prueba. Quien
    no pasa particion se llevaba el defecto neutro sin enterarse.

    Aqui el horizonte no hay que pedirlo: viene marcado en la serie.
    :func:`icm.ml.caracteristicas.objetivo_rendimiento_futuro` graba
    ``attrs["horizonte"]``, asi que el camino canonico del paquete
    (``construir_matriz`` y despues ``comparar_modelos``) queda protegido sin
    que el usuario tenga que acordarse de nada.

    Sin esa marca no se puede adivinar el horizonte, asi que se mantiene el
    embargo en cero: suponer uno seria descartar observaciones por un
    solapamiento que quiza no existe.

    Cuanto importa, medido sobre un objetivo solapado y variables de puro
    ruido, comparando el error en las filas del borde con el mismo tramo de
    prueba en ambos casos: a horizonte 1 no hay solape y no cambia nada; a 21
    la diferencia no se distingue del azar (p = 0,48 con 60 semillas); a 63 el
    entrenamiento sin embargo aparenta un 33 % menos de error, con
    p = 6,2e-06 sobre 250 semillas. Es decir, el sesgo es real y crece con el
    horizonte, y se vuelve medible mucho antes de volverse visible.
    """
    horizonte = 0
    marca = getattr(y, "attrs", {}).get("horizonte")
    if isinstance(marca, (int, np.integer)) and not isinstance(marca, bool):
        horizonte = max(int(marca), 0)
    return ParticionWalkForward(embargo=horizonte)


def comparar_modelos(
    X: pd.DataFrame,
    y: pd.Series,
    nombres: Sequence[str] | str | None = None,
    particion: ParticionWalkForward | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Valida varios modelos hacia delante y los pone en la misma tabla.

    Todos los modelos recorren **la misma** particion, asi que sus predicciones
    cubren exactamente las mismas fechas y la comparacion es una comparacion y
    no un artefacto del calendario de cada uno.

    Si no se pasa ``particion`` se construye con :func:`particion_por_defecto`,
    que lee el horizonte marcado en ``y`` y pone ese embargo. Antes se
    construia una particion neutra, con embargo cero, que es justamente lo que
    la regla documentada desaconseja para un objetivo solapado.

    ``"media_cero"`` se aniade siempre, aunque ``nombres`` no la incluya, y se
    usa como referencia de la tabla. Una tabla de modelos sin linea de flotacion
    no se puede leer: invita a comparar el mejor con el segundo, que es una
    pregunta interesante solo si alguno de los dos bate a no hacer nada.

    Devuelve
    --------
    ``(tabla, predicciones)``, donde ``tabla`` es la de
    :func:`icm.ml.evaluacion.tabla_evaluacion` (una fila por modelo, ordenada
    por MASE ascendente, con el p-valor de Diebold-Mariano frente a la
    referencia) y ``predicciones`` es ``{nombre: Series}`` con las predicciones
    fuera de muestra concatenadas, listas para pasar a una evaluacion de
    estrategia.

    Como se lee la tabla: un modelo merece atencion si su MASE esta por debajo
    de 1 **y** su p-valor por debajo de 0,05. Con lo primero solo, la ventaja
    puede ser ruido muestral. Con datos reales de rendimientos diarios, lo
    normal es que ninguna fila cumpla las dos cosas; eso es un resultado
    legitimo, no un fallo del experimento.
    """
    _alinear(X, y)
    if nombres is None:
        elegidos = list(MODELOS)
    else:
        candidatos = [nombres] if isinstance(nombres, str) else list(nombres)
        if not candidatos:
            raise ErrorDatos(
                "'nombres' esta vacio: indique al menos un modelo, o deje None para "
                f"comparar todos ({', '.join(MODELOS)})."
            )
        elegidos = [_validar_nombre(n) for n in candidatos]
        if NOMBRE_REFERENCIA not in elegidos:
            elegidos.append(NOMBRE_REFERENCIA)
    elegidos = list(dict.fromkeys(elegidos))

    particion = particion if particion is not None else particion_por_defecto(y)
    predicciones: dict[str, pd.Series] = {}
    # Los valores realizados los fija el primer modelo validado; como todos
    # recorren la misma particion, los demas devuelven exactamente los mismos.
    reales = pd.Series(dtype=float)
    for nombre in elegidos:
        resultado = validacion_walk_forward(crear_modelo(nombre), X, y, particion)
        predicciones[nombre] = resultado.predicciones.rename(nombre)
        if predicciones[nombre].size and reales.empty:
            reales = resultado.reales

    tabla = tabla_evaluacion(reales, predicciones, referencia=NOMBRE_REFERENCIA)
    return tabla, predicciones
