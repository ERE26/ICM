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
from sklearn.utils.validation import check_is_fitted

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

    ``fit`` no aprende nada, pero si guarda el numero de variables: asi
    ``check_is_fitted`` distingue una instancia ajustada de una recien creada, y
    el objeto se comporta como cualquier otro estimador de scikit-learn en la
    validacion, en la busqueda de hiperparametros y en la tabla comparativa.
    """

    def fit(self, X: Any, y: Any = None) -> MediaCero:
        forma = getattr(X, "shape", None)
        self.n_variables_ = int(forma[1]) if forma is not None and len(forma) > 1 else 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
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


def _alinear(X: Any, y: Any, sufijo: str = "") -> pd.Index:
    """Indice comun de ``X`` e ``y``, comprobando que de verdad lo sea.

    La particion y el emparejamiento se hacen por posicion, asi que dos entradas
    de la misma longitud pero de periodos distintos no fallarian: simplemente
    compararian la variable de una fecha con el objetivo de otra y darian un
    resultado sin sentido pero de aspecto impecable.
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
    if isinstance(X, (pd.DataFrame, pd.Series)) and isinstance(y, pd.Series):
        if not indice_X.equals(indice_y):
            raise ErrorDatos(
                f"'X{sufijo}' e 'y{sufijo}' tienen la misma longitud pero indices "
                "distintos: reindexelos al indice comun antes de ajustar, porque el "
                "emparejamiento es posicional y cruzaria fechas."
            )
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
        Si ya esta ajustado, se usa tal cual, y entonces ``X`` e ``y`` deberian
        ser un tramo que el modelo **no** haya visto: esa es la importancia
        honesta, la que dice de que se fia el modelo cuando acierta de verdad.
        Si no lo esta, se ajusta aqui mismo sobre ``(X, y)`` (y queda ajustado,
        por el convenio de scikit-learn) y la cifra pasa a medir en que se apoya
        el modelo dentro de muestra, que no es lo mismo: un arbol sobreajustado
        puede apoyarse mucho en una variable que fuera de muestra no vale nada.
    n_repeticiones:
        Permutaciones por variable. La importancia es una media sobre
        barajados; con pocas repeticiones el ranking baila de una semilla a
        otra.

    Devuelve
    --------
    Series indexada por nombre de variable y ordenada de mayor a menor. Los
    valores negativos no son un error: significan que el modelo predice mejor
    con esa columna barajada, es decir, que la estaba usando como ruido.
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
        ajustado = modelo
    except NotFittedError:
        ajustado = modelo.fit(X, y)
    except TypeError as exc:
        raise ErrorDatos(
            f"El objeto {type(modelo).__name__} no es un estimador de scikit-learn: "
            f"{exc}. Creelo con crear_modelo()."
        ) from exc

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
    return importancias.sort_values(ascending=False)


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
        self.particion = particion if particion is not None else ParticionWalkForward()

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[
        tuple[np.ndarray, np.ndarray]
    ]:
        return self.particion.dividir(X)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        if X is None:
            return int(self.particion.n_particiones)
        return int(self.particion.n_divisiones(X))


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

    Se puntua con el error cuadratico medio negativo y no con el R cuadrado, que
    es lo que usaria scikit-learn por omision. El R cuadrado de cada tramo se
    calcula contra la media de **ese mismo tramo**, que en la fecha de la
    decision no se conoce; como referencia movil ademas hace que los tramos
    tranquilos y los agitados no pesen igual. El error cuadratico es
    comparable entre tramos y es lo que de verdad se quiere minimizar.

    No se reajusta el modelo al final (``refit=False``): lo que se pide aqui son
    los hiperparametros, y quien los recibe decide con que tramo ajustar. Un
    reajuste automatico sobre la muestra completa incluiria el ultimo tramo de
    prueba y convertiria en silencio un resultado fuera de muestra en uno dentro.

    Parametros
    ----------
    rejilla:
        ``{nombre_de_hiperparametro: valores}`` **sin** el prefijo del paso del
        Pipeline: ``{"alpha": [0.1, 1.0]}``, no ``{"modelo__alpha": ...}``. Si se
        omite se usa la de :data:`REJILLAS`.

    Devuelve
    --------
    Diccionario de hiperparametros, con los nombres tal como los acepta
    :func:`crear_modelo`, de modo que se puede encadenar directamente:

    >>> mejores = buscar_hiperparametros("ridge", X, y)   # doctest: +SKIP
    >>> modelo = crear_modelo("ridge", **mejores)         # doctest: +SKIP

    Para ``"media_cero"`` devuelve un diccionario vacio: no hay nada que elegir,
    y eso es precisamente su virtud como referencia.
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

    busqueda = GridSearchCV(
        modelo,
        rejilla_pipeline,
        scoring="neg_mean_squared_error",
        cv=ParticionSklearn(particion),
        refit=False,
    )
    busqueda.fit(X, y)
    return {
        clave.removeprefix(prefijo): valor for clave, valor in busqueda.best_params_.items()
    }


# --- comparacion ---------------------------------------------------------------


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

    particion = particion if particion is not None else ParticionWalkForward()
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
