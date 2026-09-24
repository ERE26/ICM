"""Particiones temporales: validar sin entrenar con el futuro.

La validacion cruzada al azar es el error mas comun y mas destructivo del
aprendizaje automatico aplicado a los mercados. Barajar las filas de una serie
temporal deja dias de 2023 en el tramo de entrenamiento y dias de 2019 en el de
prueba: el modelo ve el futuro y despues lo "predice". El resultado es un R
cuadrado espectacular que no se puede cobrar, porque en produccion las fechas
llegan en un solo orden y nunca al reves.

El fallo es tan barato de cometer (``cross_val_score`` baraja por defecto en
problemas de regresion) y tan caro de detectar (nada falla, nada avisa, los
numeros solo salen *mejor*) que este modulo existe para que no se pueda cometer
por descuido: la unica particion que ofrece avanza hacia delante en el tiempo, y
:meth:`ParticionWalkForward.describir` deja las fechas de cada tramo a la vista
para poder auditarlas a ojo.

La segunda fuente de optimismo, mas sutil, es el solapamiento entre tramos
contiguos; se corrige con el embargo, que se explica en
:class:`ParticionWalkForward`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos

#: Nombres de los metodos de ajuste y prediccion que se reconocen, en el orden
#: en que se buscan: primero el convenio de scikit-learn, despues el del repositorio.
INTERFACES = (("fit", "predict"), ("ajustar", "predecir"))


# --- utilidades internas -------------------------------------------------------


def _indice_temporal(datos: Any, nombre: str) -> pd.Index:
    """Indice de la entrada, exigiendo orden creciente y fechas sin repetir.

    Las dos exigencias no son celo burocratico, son la particion misma. Con las
    filas desordenadas, "las primeras 400 posiciones" deja de significar "los
    primeros 400 dias" y el tramo de entrenamiento se llena de fechas
    posteriores a la prueba sin que nada lo delate. Con fechas repetidas, la
    misma observacion puede caer a los dos lados del corte.

    Una entrada sin indice (un ``ndarray``) recibe posiciones 0..n-1: ahi el
    orden de las filas es la unica nocion de tiempo disponible y se toma como
    valida.
    """
    if isinstance(datos, (pd.DataFrame, pd.Series)):
        indice = datos.index
    else:
        valores = np.asarray(datos)
        if valores.ndim == 0:
            raise ErrorDatos(f"'{nombre}' debe ser una tabla o un vector, no un escalar.")
        indice = pd.RangeIndex(len(valores))

    if len(indice) == 0:
        raise ErrorDatos(f"'{nombre}' no contiene ninguna observacion.")
    if indice.has_duplicates:
        repetidas = [str(f) for f in indice[indice.duplicated()].unique()[:3]]
        raise ErrorDatos(
            f"El indice de '{nombre}' tiene fechas repetidas ({', '.join(repetidas)}...): "
            "agregue o elimine los duplicados antes de partir la serie, porque una misma "
            "fecha puede acabar en entrenamiento y en prueba a la vez."
        )
    if not indice.is_monotonic_increasing:
        raise ErrorDatos(
            f"El indice de '{nombre}' no esta ordenado de menor a mayor. Ordenelo con "
            ".sort_index(): una particion hacia delante sobre filas desordenadas entrena "
            "con fechas posteriores a las de prueba."
        )
    return indice


def _subconjunto(datos: Any, posiciones: np.ndarray) -> Any:
    """Selecciona filas por posicion conservando el tipo de la entrada.

    Se conserva el ``DataFrame`` cuando lo es para que el modelo reciba los
    nombres de las columnas: scikit-learn los valida, y perderlos por el camino
    convierte un error claro en un aviso confuso.
    """
    if isinstance(datos, (pd.DataFrame, pd.Series)):
        return datos.iloc[posiciones]
    return np.asarray(datos)[posiciones]


def _metodos_del_modelo(modelo: Any) -> tuple[str, str]:
    """Nombres de los metodos de ajuste y prediccion que expone el modelo.

    Se acepta tanto el convenio de scikit-learn (``fit`` / ``predict``) como el
    de este repositorio (``ajustar`` / ``predecir``), de modo que un
    ``Pipeline`` y una red de :mod:`icm.ml.redes` entren sin adaptadores. Si un
    objeto expone los dos, se usa el de scikit-learn.
    """
    for ajuste, prediccion in INTERFACES:
        if callable(getattr(modelo, ajuste, None)) and callable(getattr(modelo, prediccion, None)):
            return ajuste, prediccion
    raise ErrorDatos(
        f"El modelo {type(modelo).__name__} no se puede validar: necesita los metodos "
        "'fit' y 'predict' (estilo scikit-learn) o 'ajustar' y 'predecir' (estilo de "
        "este repositorio)."
    )


def _clonar(modelo: Any) -> Any:
    """Copia sin ajustar del modelo, para empezar cada particion desde cero.

    Reajustar el mismo objeto particion tras particion arruina el experimento
    de la forma mas silenciosa posible: al llegar a la tercera prueba, el modelo
    ya ha visto los datos de la segunda, es decir, datos posteriores a los de su
    propio entrenamiento. Las estimaciones fuera de muestra dejan de serlo.

    ``sklearn.base.clone`` es la via preferida porque copia los hiperparametros
    y **no** el estado ajustado; para lo que no es un estimador de scikit-learn
    (una red de PyTorch, una clase propia) se recurre a ``copy.deepcopy``.
    """
    from sklearn.base import clone

    try:
        return clone(modelo)
    except TypeError:
        return copy.deepcopy(modelo)


def _correlacion(reales: np.ndarray, predicciones: np.ndarray) -> float:
    """Correlacion de Pearson, devolviendo NaN en vez de dividir por cero.

    Un modelo que predice siempre lo mismo (constante) tiene desviacion nula y
    su correlacion no esta definida. Calcularla sin mas emitiria un
    ``RuntimeWarning`` y un NaN por la puerta de atras; es mas honesto decir que
    no hay correlacion que medir.
    """
    if reales.size < 2:
        return float("nan")
    desviacion_real = float(reales.std())
    desviacion_prediccion = float(predicciones.std())
    if desviacion_real <= 0.0 or desviacion_prediccion <= 0.0:
        return float("nan")
    centrado = (reales - reales.mean()) * (predicciones - predicciones.mean())
    return float(centrado.mean() / (desviacion_real * desviacion_prediccion))


def _acierto_direccional(reales: np.ndarray, predicciones: np.ndarray) -> float:
    """Proporcion de fechas en las que se acierta el signo del rendimiento.

    En trading el signo decide la posicion, asi que esta cifra se acerca mas al
    dinero que el error cuadratico. Las fechas de rendimiento real nulo se
    excluyen: no hay direccion que acertar.
    """
    utiles = reales != 0.0
    if not utiles.any():
        return float("nan")
    return float(np.mean(np.sign(predicciones[utiles]) == np.sign(reales[utiles])))


def _metricas(
    reales: np.ndarray,
    predicciones: np.ndarray,
    referencia: float,
) -> dict[str, float]:
    """Metricas de una particion frente a una prediccion de referencia.

    ``referencia`` es la media del objetivo **en el tramo de entrenamiento**, que
    es la unica prediccion ingenua que se podia haber hecho en esa fecha. El R
    cuadrado calculado contra ella (Campbell y Thompson, 2008) responde a la
    pregunta que importa: si el modelo aporta algo sobre no haber modelado nada.
    Usar la media del tramo de prueba, como hace el R cuadrado habitual, seria
    otra fuga de informacion del futuro, pequenia pero de la misma familia.
    """
    error = predicciones - reales
    suma_cuadrados_error = float(np.sum(error**2))
    suma_cuadrados_total = float(np.sum((reales - referencia) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "sesgo": float(np.mean(error)),
        "correlacion": _correlacion(reales, predicciones),
        "acierto_direccional": _acierto_direccional(reales, predicciones),
        "r2_fuera_muestra": (
            float("nan")
            if suma_cuadrados_total <= 0.0
            else 1.0 - suma_cuadrados_error / suma_cuadrados_total
        ),
    }


# --- particion -----------------------------------------------------------------


@dataclass
class ParticionWalkForward:
    """Particiones sucesivas hacia delante en el tiempo (*walk forward*).

    Cada particion entrena con el pasado y prueba con el tramo inmediatamente
    posterior. Los tramos de prueba son contiguos, no se solapan y terminan en
    la ultima observacion disponible, de modo que la concatenacion de todas las
    predicciones es una serie fuera de muestra sin huecos ni fechas repetidas.

    Parametros
    ----------
    n_particiones:
        Numero de tramos de prueba. Mas particiones dan una estimacion menos
        ruidosa del error, pero acortan cada tramo y dejan menos historia al
        primer entrenamiento.
    tamano_prueba:
        Observaciones de cada tramo de prueba. Si se omite, se reparte a partes
        iguales lo que sobra tras reservar ``tamano_entrenamiento_minimo`` y el
        embargo.
    embargo:
        Observaciones que se **descartan** entre el final del entrenamiento y el
        principio de la prueba. Con un objetivo a ``h`` dias, ``y[t]`` es lo que
        ocurre entre ``t`` y ``t+h``, asi que las ultimas ``h`` observaciones del
        tramo de entrenamiento tienen su objetivo medido ya dentro del tramo de
        prueba: el modelo se entrena con el resultado de los dias que luego debe
        predecir. El solapamiento aparece tambien con cualquier variable
        construida con ventanas moviles, que mezclan fechas de los dos lados del
        corte. La regla practica es ``embargo = h`` (o la ventana mas larga, si es
        mayor); cuesta ``h`` observaciones por particion, una cifra ridicula
        frente al sesgo que elimina.
    expansiva:
        Si es cierto, el entrenamiento arranca siempre en la primera observacion
        y crece en cada particion: usa toda la historia, que es lo razonable
        cuando el proceso es estable. Si es falso, la ventana de entrenamiento
        conserva un tamanio constante y se desplaza, lo que se adapta mejor a un
        mercado que cambia de regimen y ademas mantiene comparables los errores
        de las distintas particiones, porque todas se estiman con la misma
        cantidad de datos.
    tamano_entrenamiento_minimo:
        Observaciones que debe tener el entrenamiento de la primera particion.
        Evita la tentacion de exprimir una serie corta con un modelo ajustado
        sobre treinta dias.
    """

    n_particiones: int = 5
    tamano_prueba: int | None = None
    embargo: int = 0
    expansiva: bool = True
    tamano_entrenamiento_minimo: int = 100

    def __post_init__(self) -> None:
        if self.n_particiones < 1:
            raise ErrorDatos(
                f"'n_particiones' debe ser al menos 1; se ha recibido {self.n_particiones}."
            )
        if self.tamano_prueba is not None and self.tamano_prueba < 1:
            raise ErrorDatos(
                "'tamano_prueba' debe ser al menos 1, o None para repartir el resto "
                f"a partes iguales; se ha recibido {self.tamano_prueba}."
            )
        if self.embargo < 0:
            raise ErrorDatos(
                f"'embargo' no puede ser negativo; se ha recibido {self.embargo}. "
                "Use 0 para no descartar nada entre entrenamiento y prueba."
            )
        if self.tamano_entrenamiento_minimo < 1:
            raise ErrorDatos(
                "'tamano_entrenamiento_minimo' debe ser al menos 1; se ha recibido "
                f"{self.tamano_entrenamiento_minimo}."
            )

    def _disposicion(self, n: int) -> list[tuple[int, int, int, int]]:
        """Limites ``(inicio_ent, fin_ent, inicio_prueba, fin_prueba)`` por particion.

        Todos los limites son posiciones, con el final excluido al estilo de
        ``range``. El calculo se hace de atras hacia delante: los tramos de
        prueba se pegan al final de la muestra, porque los datos recientes son
        los que de verdad interesa saber si el modelo predice.
        """
        minimo = self.tamano_entrenamiento_minimo
        if self.tamano_prueba is None:
            tamano = (n - minimo - self.embargo) // self.n_particiones
            if tamano < 1:
                necesarias = minimo + self.embargo + self.n_particiones
                raise ErrorDatos(
                    f"No hay datos suficientes: con {n} observaciones, {self.n_particiones} "
                    f"particiones, embargo de {self.embargo} y un entrenamiento minimo de "
                    f"{minimo}, a cada tramo de prueba le corresponderian "
                    f"{max(tamano, 0)} observaciones. Hacen falta al menos {necesarias} "
                    "observaciones, o reduzca 'n_particiones' o "
                    "'tamano_entrenamiento_minimo'."
                )
        else:
            tamano = int(self.tamano_prueba)

        inicio_pruebas = n - self.n_particiones * tamano
        fin_primer_entrenamiento = inicio_pruebas - self.embargo
        if fin_primer_entrenamiento < minimo:
            necesarias = minimo + self.embargo + self.n_particiones * tamano
            raise ErrorDatos(
                f"No hay datos suficientes: {self.n_particiones} particiones de {tamano} "
                f"observaciones de prueba, un embargo de {self.embargo} y un entrenamiento "
                f"minimo de {minimo} exigen al menos {necesarias} observaciones; hay {n}. "
                "Alargue la serie, o reduzca 'n_particiones', 'tamano_prueba' o "
                "'tamano_entrenamiento_minimo'."
            )

        tramos: list[tuple[int, int, int, int]] = []
        for k in range(self.n_particiones):
            inicio_prueba = inicio_pruebas + k * tamano
            fin_entrenamiento = inicio_prueba - self.embargo
            inicio_entrenamiento = (
                0 if self.expansiva else fin_entrenamiento - fin_primer_entrenamiento
            )
            tramos.append(
                (inicio_entrenamiento, fin_entrenamiento, inicio_prueba, inicio_prueba + tamano)
            )
        return tramos

    def dividir(self, X: pd.DataFrame | pd.Series | np.ndarray) -> Iterator[
        tuple[np.ndarray, np.ndarray]
    ]:
        """Posiciones de entrenamiento y de prueba de cada particion.

        Devuelve posiciones enteras, no etiquetas: se usan con ``.iloc`` en
        pandas y con indexacion directa en ``numpy``, igual que en scikit-learn.

        La disposicion se calcula antes de devolver el iterador, asi que una
        serie demasiado corta falla en la llamada y no a mitad del recorrido,
        cuando ya se ha ajustado medio experimento.
        """
        indice = _indice_temporal(X, "X")
        tramos = self._disposicion(len(indice))
        return iter(
            [
                (np.arange(inicio_ent, fin_ent), np.arange(inicio_pru, fin_pru))
                for inicio_ent, fin_ent, inicio_pru, fin_pru in tramos
            ]
        )

    def n_divisiones(self, X: pd.DataFrame | pd.Series | np.ndarray) -> int:
        """Numero de particiones que se obtendran de ``X``.

        Comprueba tambien que haya datos suficientes, de modo que preguntar
        cuantas particiones hay nunca devuelve una cifra que luego resulte
        imposible de cumplir.
        """
        indice = _indice_temporal(X, "X")
        return len(self._disposicion(len(indice)))

    def describir(self, X: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame:
        """Fechas y tamanios de cada tramo, para auditar la particion a ojo.

        Las fechas de inicio y fin son **inclusivas**, que es como se leen en un
        calendario. Mirar esta tabla es la forma mas rapida de descubrir que la
        particion no es la que uno creia: un fin de entrenamiento posterior al
        inicio de la prueba, un embargo que no esta, un primer tramo ridiculo.
        """
        indice = _indice_temporal(X, "X")
        tramos = self._disposicion(len(indice))
        filas = []
        for numero, (inicio_ent, fin_ent, inicio_pru, fin_pru) in enumerate(tramos):
            filas.append(
                {
                    "particion": numero,
                    "inicio_entrenamiento": indice[inicio_ent],
                    "fin_entrenamiento": indice[fin_ent - 1],
                    "n_entrenamiento": fin_ent - inicio_ent,
                    "embargo": self.embargo,
                    "inicio_prueba": indice[inicio_pru],
                    "fin_prueba": indice[fin_pru - 1],
                    "n_prueba": fin_pru - inicio_pru,
                }
            )
        return pd.DataFrame(filas).set_index("particion")


# --- validacion ----------------------------------------------------------------


@dataclass
class ResultadoValidacion:
    """Todo lo que arroja una validacion hacia delante.

    ``predicciones`` y ``reales`` son series fuera de muestra concatenadas e
    indexadas por fecha: cada valor lo produjo un modelo que no habia visto esa
    fecha ni ninguna posterior. Es la serie sobre la que se puede evaluar una
    estrategia sin enganiarse.
    """

    predicciones: pd.Series
    reales: pd.Series
    por_particion: pd.DataFrame
    modelos: list

    def __len__(self) -> int:
        return len(self.predicciones)

    @property
    def resumen(self) -> pd.Series:
        """Metricas sobre el conjunto de las predicciones fuera de muestra.

        No incluye el R cuadrado: cada particion lo mide contra la media de su
        propio entrenamiento, y esas referencias no son la misma cifra, asi que
        agregarlas en un solo numero seria inventarselo. Esta por particion en
        ``por_particion``.
        """
        reales = self.reales.to_numpy(dtype=float)
        predicciones = self.predicciones.to_numpy(dtype=float)
        error = predicciones - reales
        return pd.Series(
            {
                "n_particiones": float(len(self.modelos)),
                "observaciones": float(len(reales)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "sesgo": float(np.mean(error)),
                "correlacion": _correlacion(reales, predicciones),
                "acierto_direccional": _acierto_direccional(reales, predicciones),
            },
            name="validacion_walk_forward",
        )


def _alinear(X: Any, y: Any) -> pd.Index:
    """Indice comun de ``X`` e ``y``, comprobando que de verdad lo sea."""
    indice = _indice_temporal(X, "X")
    indice_y = _indice_temporal(y, "y")
    if len(indice) != len(indice_y):
        raise ErrorDatos(
            f"'X' tiene {len(indice)} filas e 'y' tiene {len(indice_y)}: alinee ambos "
            "con un unico 'dropna' conjunto antes de validar."
        )
    if isinstance(X, (pd.DataFrame, pd.Series)) and isinstance(y, pd.Series):
        if not indice.equals(indice_y):
            raise ErrorDatos(
                "'X' e 'y' no comparten indice pese a tener la misma longitud: "
                "reindexelos antes de validar, porque la particion se hace por posicion "
                "y emparejaria fechas distintas."
            )
    return indice


def validacion_walk_forward(
    modelo: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    particion: ParticionWalkForward | None = None,
) -> ResultadoValidacion:
    """Ajusta y evalua un modelo particion a particion, siempre hacia delante.

    Parametros
    ----------
    modelo:
        Cualquier objeto con ``fit``/``predict`` (estilo scikit-learn) o con
        ``ajustar``/``predecir`` (estilo de este repositorio). No se toca: en
        cada particion se valida una copia sin ajustar, de modo que al terminar
        el objeto recibido sigue tal como llego.
    particion:
        Esquema de division. Por omision, :class:`ParticionWalkForward` con sus
        valores por defecto.

    Notas
    -----
    Las predicciones de todas las particiones se concatenan en una sola serie.
    Como los tramos de prueba son contiguos y disjuntos, esa serie cubre sin
    huecos la cola de la muestra y ninguna fecha aparece dos veces, que es
    justo lo que hace falta para pasarsela despues a una evaluacion de
    estrategia sin volver a pensar en el alineamiento.
    """
    particion = particion if particion is not None else ParticionWalkForward()
    indice = _alinear(X, y)
    nombre_ajuste, nombre_prediccion = _metodos_del_modelo(modelo)

    valores_y = (
        y.to_numpy(dtype=float) if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
    )
    if valores_y.ndim != 1:
        raise ErrorDatos(
            f"'y' debe ser unidimensional; tiene forma {valores_y.shape}. Seleccione una "
            "columna antes de validar."
        )

    predicciones: list[pd.Series] = []
    reales: list[pd.Series] = []
    modelos: list = []
    filas: list[dict] = []

    for numero, (pos_ent, pos_pru) in enumerate(particion.dividir(X)):
        copia = _clonar(modelo)
        devuelto = getattr(copia, nombre_ajuste)(
            _subconjunto(X, pos_ent), _subconjunto(y, pos_ent)
        )
        # Se admiten las dos convenciones de ajuste: mutar el objeto y devolver
        # self (scikit-learn) o devolver un objeto nuevo ya ajustado.
        ajustado = (
            devuelto
            if devuelto is not None and callable(getattr(devuelto, nombre_prediccion, None))
            else copia
        )

        crudas = getattr(ajustado, nombre_prediccion)(_subconjunto(X, pos_pru))
        crudas = np.asarray(crudas, dtype=float).reshape(-1)
        if crudas.size != pos_pru.size:
            raise ErrorDatos(
                f"En la particion {numero} el modelo ha devuelto {crudas.size} predicciones "
                f"para {pos_pru.size} fechas. '{nombre_prediccion}' debe devolver una "
                "prediccion por fila."
            )

        fechas = indice[pos_pru]
        observados = valores_y[pos_pru]
        predicciones.append(pd.Series(crudas, index=fechas))
        reales.append(pd.Series(observados, index=fechas))
        modelos.append(ajustado)
        filas.append(
            {
                "particion": numero,
                "n_entrenamiento": int(pos_ent.size),
                "n_prueba": int(pos_pru.size),
                "inicio_prueba": fechas[0],
                "fin_prueba": fechas[-1],
                **_metricas(observados, crudas, float(valores_y[pos_ent].mean())),
            }
        )

    return ResultadoValidacion(
        predicciones=pd.concat(predicciones).rename("prediccion"),
        reales=pd.concat(reales).rename("real"),
        por_particion=pd.DataFrame(filas).set_index("particion"),
        modelos=modelos,
    )
