"""Redes secuenciales (LSTM, GRU, TCN) para predecir series de precios.

Por que el import de PyTorch es diferido
----------------------------------------
Este modulo es el unico del repositorio que necesita PyTorch, y el paquete de
PyPI arrastra las bibliotecas de CUDA: unos 3 GB para un entorno que aqui
siempre corre en CPU. Importar ``torch`` en la cabecera obligaria a bajarse esos
3 GB a quien solo quiera calcular un VaR, asi que la importacion vive **dentro**
de las funciones. ``import icm.ml.redes`` funciona sin PyTorch instalado; lo que
falla, con un mensaje que incluye el comando de instalacion, es intentar ajustar
una red.

Que hace este modulo distinto de un tutorial
--------------------------------------------
Casi todos los ejemplos de redes recurrentes aplicados a mercados contienen al
menos una de estas cuatro fugas de informacion del futuro, y las cuatro estan
cerradas aqui a proposito:

1. **Ventanas centradas.** La muestra de la fecha ``t`` usa exclusivamente
   ``t-ventana+1 .. t``. Una ventana centrada en ``t`` incluiria fechas
   posteriores, que en produccion no existen todavia.
2. **Normalizacion con toda la serie.** La media y la desviacion tipica se
   estiman **solo** con el tramo de entrenamiento. Estandarizar antes de partir
   filtra el nivel y la volatilidad del futuro dentro de cada observacion
   pasada; es una fuga pequenia, constante y practicamente invisible en las
   metricas.
3. **Validacion interna al azar.** El conjunto de validacion que gobierna la
   parada temprana es el **ultimo tramo temporal**, nunca una muestra
   aleatoria. Barajar una serie temporal para elegir la validacion es la misma
   trampa que la validacion cruzada al azar que :mod:`icm.ml.validacion` existe
   para impedir, solo que cometida un nivel mas abajo.
4. **Tabla en orden descendente.** Las ventanas se construyen por *posicion*,
   asi que con la tabla de mas reciente a mas antigua -el orden en que la sirven
   casi todos los proveedores de datos- el "hacia atras" en posicion es hacia
   delante en el calendario. Se exige indice creciente y sin fechas repetidas,
   igual que :mod:`icm.ml.validacion` y :mod:`icm.ml.caracteristicas`.

La interfaz es ``ajustar`` / ``predecir``, la del repositorio, de modo que
:func:`icm.ml.validacion.validacion_walk_forward` acepta una red sin
adaptadores; y ``get_params`` / ``set_params`` estan implementados para que
``sklearn.base.clone`` pueda producir la copia sin ajustar que esa validacion
necesita en cada particion.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import warnings
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos

#: Arquitecturas admitidas en :attr:`ConfiguracionRed.arquitectura`.
ARQUITECTURAS = ("lstm", "gru", "tcn")

#: Anchura del nucleo de convolucion de la TCN. No se expone como parametro
#: porque el alcance de la red se regula mejor con el numero de capas, que lo
#: hace crecer de forma exponencial en lugar de lineal.
NUCLEO_TCN = 3

#: Como instalar PyTorch sin las bibliotecas de CUDA, que aqui no se usan.
COMANDO_INSTALACION = "pip install torch --index-url https://download.pytorch.org/whl/cpu"

#: Muestras de entrenamiento por debajo de las cuales el ajuste se rechaza. Con
#: una sola muestra la red memoriza su unico ejemplo, la desviacion del objetivo
#: es nula y :func:`_escala` la sustituye por 1, de modo que la escala de ``y``
#: se queda sin estimar; el resultado es un ajuste degenerado que no se
#: distingue por fuera de uno bueno.
MINIMO_ENTRENAMIENTO = 2

#: Clases de ``torch.nn`` definidas en la primera llamada que las necesita.
#: Viven en un diccionario y no en el espacio del modulo porque heredan de
#: ``torch.nn.Module``, que no puede existir mientras PyTorch no se importe.
_MODULOS: dict[str, Any] = {}


def torch_disponible() -> bool:
    """Indica si PyTorch se puede importar en este entorno.

    Comprueba la especificacion del modulo en lugar de importarlo: cargar
    PyTorch tarda varios segundos y reserva memoria, un precio absurdo para
    responder a una pregunta de si o no. Permite que un notebook o una bateria
    de pruebas se salte la parte de redes sin provocar un error.
    """
    return importlib.util.find_spec("torch") is not None


def _importar_torch():
    """Importa PyTorch o explica como instalarlo.

    El error es :class:`~icm.comun.ErrorDatos` y no ``ImportError`` para que
    quien use el repositorio solo tenga que capturar un tipo de excepcion, y
    lleva el comando literal porque la version por defecto de PyPI descarga
    unos 3 GB de CUDA que en este entorno no sirven para nada.
    """
    if not torch_disponible():
        raise ErrorDatos(
            "icm.ml.redes necesita PyTorch y no esta instalado. La version de CPU "
            f"ocupa unos 200 MB en lugar de 3 GB: {COMANDO_INSTALACION}"
        )
    import torch

    return torch


# --- configuracion -------------------------------------------------------------


@dataclass
class ConfiguracionRed:
    """Hiperparametros de :class:`RedSecuencial`.

    Parametros
    ----------
    arquitectura:
        ``"lstm"`` y ``"gru"`` son redes recurrentes; la GRU tiene una puerta
        menos, asi que ajusta menos parametros y suele ir mejor con series
        cortas. ``"tcn"`` es una red convolucional causal: entrena mas rapido
        porque no recorre la secuencia paso a paso, pero su memoria esta
        limitada por el alcance del nucleo, ``(NUCLEO_TCN - 1) * (2**capas - 1)
        + 1`` observaciones. Con los valores por defecto (nucleo 3, una capa)
        ese alcance es de 3 fechas: si la senial que se busca vive mas atras en
        la ventana, hacen falta mas capas o una arquitectura recurrente.
    ventana:
        Numero de fechas que ve cada muestra, contadas **hacia atras** desde la
        fecha que se predice. Fija tambien el arranque: las primeras
        ``ventana - 1`` filas de cualquier tabla no tienen historia suficiente.
    unidades, capas, abandono:
        Tamanio del estado oculto, numero de capas apiladas y proporcion de
        abandono (*dropout*). Con series financieras, donde la relacion
        senial-ruido es minuscula, las redes pequenias casi siempre ganan: mas
        capacidad se emplea en memorizar el ruido del tramo de entrenamiento.
    epocas, tamano_lote, tasa_aprendizaje:
        Pasadas maximas sobre los datos, tamanio del minilote y paso de Adam.
        ``epocas`` es un maximo, no un objetivo: lo normal es que la parada
        temprana corte antes.
    paciencia:
        Epocas consecutivas sin mejorar la perdida de validacion que se toleran
        antes de parar. Al terminar se restauran los pesos de la **mejor**
        epoca, no los de la ultima, que es el motivo de que exista la parada
        temprana: los ultimos pesos son precisamente los que ya han empezado a
        sobreajustar. Se ignora si ``proporcion_validacion`` es 0, porque sin
        tramo reservado no hay ninguna perdida que vigilar.
    proporcion_validacion:
        Fraccion final de la muestra que se reserva para vigilar el
        sobreajuste. Es el ultimo tramo **temporal**. Con 0 no hay validacion
        interna, y entonces **la parada temprana queda desactivada**: se
        ejecutan las ``epocas`` completas y se conservan los pesos de la
        ultima. Es el modo para series muy cortas, donde reservar un tramo
        costaria demasiadas muestras, y vale a cambio de renunciar a la
        vigilancia del sobreajuste. Vigilar en su lugar la perdida de
        entrenamiento, que es lo que se hacia antes, no es una version debil de
        la parada temprana sino una averia: esa perdida baja casi siempre, asi
        que no avisa de nada, y su ruido de minilote y de abandono basta para
        agotar la paciencia y cortar el entrenamiento por sorpresa (medido:
        con abandono 0.2 y paciencia 2 paraba en la epoca 14 de 30 y se quedaba
        con unos pesos de perdida real 0.0139 cuando la epoca 26 llegaba a
        0.0063).
    semilla:
        Se aplica a PyTorch y al generador de numpy que ordena los minilotes,
        de modo que dos ajustes con la misma configuracion y los mismos datos
        dan exactamente las mismas predicciones.
    """

    arquitectura: str = "lstm"
    ventana: int = 30
    unidades: int = 32
    capas: int = 1
    abandono: float = 0.1
    epocas: int = 100
    tamano_lote: int = 64
    tasa_aprendizaje: float = 1e-3
    paciencia: int = 10
    proporcion_validacion: float = 0.2
    semilla: int = 42

    def __post_init__(self) -> None:
        # Se comprueba sin convertir los valores: ``sklearn.base.clone`` exige
        # que los hiperparametros que devuelve ``get_params`` sean los mismos
        # objetos que se le pasaron, y una conversion los sustituiria.
        if self.arquitectura not in ARQUITECTURAS:
            raise ErrorDatos(
                f"Arquitectura '{self.arquitectura}' desconocida. "
                f"Disponibles: {', '.join(ARQUITECTURAS)}."
            )
        if self.ventana < 1:
            raise ErrorDatos(
                f"'ventana' debe ser al menos 1; se ha recibido {self.ventana}. "
                "Es el numero de fechas pasadas que ve cada muestra."
            )
        if self.unidades < 1:
            raise ErrorDatos(f"'unidades' debe ser al menos 1; se ha recibido {self.unidades}.")
        if self.capas < 1:
            raise ErrorDatos(f"'capas' debe ser al menos 1; se ha recibido {self.capas}.")
        if not 0.0 <= self.abandono < 1.0:
            raise ErrorDatos(
                f"'abandono' debe estar en [0, 1); se ha recibido {self.abandono}. "
                "Con 1 se apagarian todas las unidades y la red no podria aprender nada."
            )
        if self.epocas < 1:
            raise ErrorDatos(f"'epocas' debe ser al menos 1; se ha recibido {self.epocas}.")
        if self.tamano_lote < 1:
            raise ErrorDatos(
                f"'tamano_lote' debe ser al menos 1; se ha recibido {self.tamano_lote}."
            )
        if self.tasa_aprendizaje <= 0.0:
            raise ErrorDatos(
                f"'tasa_aprendizaje' debe ser positiva; se ha recibido {self.tasa_aprendizaje}."
            )
        if self.paciencia < 1:
            raise ErrorDatos(
                f"'paciencia' debe ser al menos 1; se ha recibido {self.paciencia}. "
                "Use 'epocas' para entrenar sin parada temprana."
            )
        if not 0.0 <= self.proporcion_validacion < 1.0:
            raise ErrorDatos(
                "'proporcion_validacion' debe estar en [0, 1); se ha recibido "
                f"{self.proporcion_validacion}. Con 1 no quedarian datos de entrenamiento."
            )
        # Es un aviso y no un ErrorDatos porque la configuracion no es invalida:
        # la red se ajusta y predice, solo que con menos memoria de la que el
        # usuario cree haber pedido. Rechazarla dejaria sin construir
        # ConfiguracionRed(arquitectura='tcn') con los valores por defecto, que
        # es la forma en que el repositorio y sus pruebas la piden hoy.
        if self.arquitectura == "tcn" and self.alcance_tcn < self.ventana:
            capas_necesarias = math.ceil(math.log2(1 + (self.ventana - 1) / (NUCLEO_TCN - 1)))
            descartadas = self.ventana - self.alcance_tcn
            warnings.warn(
                f"La TCN configurada solo alcanza a ver {self.alcance_tcn} de las "
                f"{self.ventana} fechas de la ventana: {descartadas} de cada "
                f"{self.ventana}, las mas antiguas de cada muestra, no pueden influir en "
                f"la prediccion, y sin embargo cuestan las {self.ventana - 1} filas de "
                "arranque que 'predecir' devuelve como NaN. Suba 'capas' a "
                f"{capas_necesarias} para cubrir la ventana, o baje 'ventana' a "
                f"{self.alcance_tcn} para pagar solo por lo que la red mira.",
                UserWarning,
                stacklevel=3,
            )

    @property
    def alcance_tcn(self) -> int:
        """Fechas hacia atras que alcanza a ver la TCN configurada.

        Sirve para detectar la incoherencia mas comun de esta arquitectura: una
        ventana de 30 fechas con un alcance de 3 significa que 27 columnas de
        cada muestra no influyen en la prediccion.
        """
        return (NUCLEO_TCN - 1) * (2**self.capas - 1) + 1


#: Nombres de los hiperparametros, que son tambien los de ``get_params``.
NOMBRES_HIPERPARAMETROS = tuple(campo.name for campo in fields(ConfiguracionRed))


def _validar_nombres(parametros: dict[str, Any]) -> None:
    """Rechaza hiperparametros inexistentes antes de que se pierdan en silencio.

    Sin esta comprobacion, un ``RedSecuencial(tasa_de_aprendizaje=0.1)`` (con la
    preposicion de mas) se aceptaria y entrenaria con la tasa por defecto, que
    es el peor desenlace posible: ni error, ni el comportamiento pedido.
    """
    desconocidos = sorted(set(parametros) - set(NOMBRES_HIPERPARAMETROS))
    if desconocidos:
        raise ErrorDatos(
            f"Hiperparametros desconocidos: {', '.join(desconocidos)}. "
            f"Disponibles: {', '.join(sorted(NOMBRES_HIPERPARAMETROS))}."
        )


# --- construccion de las redes -------------------------------------------------


def _clases_torch() -> dict[str, Any]:
    """Define (una sola vez) las clases de ``torch.nn`` que usa el modulo.

    Las definiciones estan dentro de una funcion por la misma razon que el
    import: heredar de ``torch.nn.Module`` en el cuerpo del modulo obligaria a
    tener PyTorch instalado para poder importar ``icm.ml.redes``. El resultado
    se memoriza porque redefinir las clases en cada ajuste crearia un tipo
    nuevo cada vez.
    """
    if _MODULOS:
        return _MODULOS

    torch = _importar_torch()
    from torch import nn
    from torch.nn import functional as F

    class RedRecurrente(nn.Module):
        """LSTM o GRU que predice a partir del ultimo estado oculto.

        Se lee solo el estado del ultimo paso porque es el unico que ha visto
        la ventana completa; los intermedios corresponden a fechas anteriores y
        usarlos equivaldria a predecir ``t`` con informacion de ``t-3``.
        """

        def __init__(
            self,
            tipo: str,
            n_variables: int,
            unidades: int,
            capas: int,
            abandono: float,
        ) -> None:
            super().__init__()
            constructor = nn.LSTM if tipo == "lstm" else nn.GRU
            self.recurrente = constructor(
                input_size=n_variables,
                hidden_size=unidades,
                num_layers=capas,
                batch_first=True,
                # El abandono interno de PyTorch solo actua *entre* capas, asi
                # que con una sola capa debe ir a cero (si no, PyTorch avisa de
                # que el parametro no tiene efecto).
                dropout=float(abandono) if capas > 1 else 0.0,
            )
            self.abandono = nn.Dropout(float(abandono))
            self.salida = nn.Linear(unidades, 1)

        def forward(self, x):
            secuencia, _ = self.recurrente(x)
            return self.salida(self.abandono(secuencia[:, -1, :])).squeeze(-1)

    class BloqueCausal(nn.Module):
        """Convolucion dilatada con relleno solo por la izquierda.

        Aqui es donde la TCN deja de mirar al futuro: una ``Conv1d`` con
        ``padding`` simetrico rellena tambien por la derecha, y entonces la
        salida de la fecha ``t`` depende de fechas posteriores. Rellenando solo
        por delante, la convolucion es causal.
        """

        def __init__(
            self,
            entradas: int,
            salidas: int,
            dilatacion: int,
            abandono: float,
        ) -> None:
            super().__init__()
            self.relleno = (NUCLEO_TCN - 1) * dilatacion
            self.conv = nn.Conv1d(entradas, salidas, NUCLEO_TCN, dilation=dilatacion)
            self.abandono = nn.Dropout(float(abandono))
            # Atajo residual: permite apilar capas sin que el gradiente se
            # diluya, y adapta el numero de canales cuando cambia.
            self.atajo = nn.Conv1d(entradas, salidas, 1) if entradas != salidas else nn.Identity()

        def forward(self, x):
            y = F.pad(x, (self.relleno, 0))
            y = self.abandono(torch.relu(self.conv(y)))
            return y + self.atajo(x)

    class RedTCN(nn.Module):
        """Pila de bloques causales con dilatacion creciente.

        La dilatacion se duplica en cada capa para que el alcance temporal
        crezca de forma exponencial: ver 30 fechas atras cuesta cinco capas en
        lugar de quince.
        """

        def __init__(
            self,
            n_variables: int,
            unidades: int,
            capas: int,
            abandono: float,
        ) -> None:
            super().__init__()
            bloques = [
                BloqueCausal(
                    n_variables if indice == 0 else unidades,
                    unidades,
                    2**indice,
                    abandono,
                )
                for indice in range(capas)
            ]
            self.bloques = nn.Sequential(*bloques)
            self.salida = nn.Linear(unidades, 1)

        def forward(self, x):
            # Las convoluciones esperan (lote, canales, tiempo) y las ventanas
            # llegan como (lote, tiempo, variables).
            activaciones = self.bloques(x.transpose(1, 2))
            return self.salida(activaciones[:, :, -1]).squeeze(-1)

    _MODULOS.update({"RedRecurrente": RedRecurrente, "RedTCN": RedTCN})
    return _MODULOS


def _crear_red(configuracion: ConfiguracionRed, n_variables: int):
    """Instancia la arquitectura indicada, siempre en CPU.

    No se busca GPU a proposito: este entorno no tiene, y con las redes
    minusculas que exigen las series financieras el trasiego de memoria a la
    tarjeta cuesta mas de lo que ahorra el calculo.
    """
    clases = _clases_torch()
    if configuracion.arquitectura == "tcn":
        red = clases["RedTCN"](
            n_variables,
            configuracion.unidades,
            configuracion.capas,
            configuracion.abandono,
        )
    else:
        red = clases["RedRecurrente"](
            configuracion.arquitectura,
            n_variables,
            configuracion.unidades,
            configuracion.capas,
            configuracion.abandono,
        )
    return red.to("cpu")


# --- utilidades de datos -------------------------------------------------------


def _validar_indice_temporal(indice: pd.Index, nombre: str) -> None:
    """Exige un indice creciente y sin fechas repetidas.

    Es la cuarta fuga de la cabecera, y la unica que ninguna de las otras tres
    defensas puede cerrar: :func:`_ventanas` recorta por **posicion**, asi que
    con la tabla ordenada de mas reciente a mas antigua la ventana "hacia atras"
    es hacia delante en el calendario y la fecha ``t`` se predice con ``t+1 ..
    t+ventana-1``. Con fechas repetidas no hay forma de saber cual de las filas
    iguales es el pasado de cual.

    Ordenar por dentro y en silencio seria peor: ``predecir`` devuelve un vector
    alineado por posicion con la tabla que recibe, asi que el llamante volveria
    a pegar esas predicciones sobre su propio orden, que ya no seria el mismo.

    Una entrada sin indice (un ``ndarray``) recibe el ``RangeIndex`` que le pone
    :func:`_como_marco`, que siempre pasa las dos comprobaciones: ahi el orden
    de las filas es la unica nocion de tiempo disponible y se toma como valida,
    igual que en :func:`icm.ml.validacion._indice_temporal`.

    Un ``MultiIndex`` de panel se rechaza aparte, porque burla las dos
    comprobaciones sin dejar de romper lo que protegen: los pares
    ``(fecha, activo)`` son unicos y crecientes aunque cada fecha se repita una
    vez por activo. Una ventana por posicion sobre esa tabla no recorre la
    historia de un activo, sino filas alternas de todos: con tres activos, una
    ventana de 5 mezcla dos fechas de cada uno. :mod:`icm.ml.validacion` ya
    rechazaba este caso, y que un modulo lo aceptara y el otro no era la
    incoherencia mas facil de aprovechar por descuido.
    """
    if isinstance(indice, pd.MultiIndex):
        raise ErrorDatos(
            f"El indice de '{nombre}' es un MultiIndex y esta red no sabe de paneles: "
            "construye la ventana por posicion, asi que cada muestra mezclaria filas de "
            "activos distintos como si fuesen la historia de uno solo. Ajuste una red por "
            "activo, o aplane el panel a una sola serie por fecha."
        )
    if indice.has_duplicates:
        repetidas = [str(f) for f in indice[indice.duplicated()].unique()[:3]]
        raise ErrorDatos(
            f"El indice de '{nombre}' tiene fechas repetidas ({', '.join(repetidas)}...): "
            "agregue o elimine los duplicados antes de ajustar, porque la ventana se "
            "construye por posicion y no hay forma de saber cual de las filas repetidas "
            "es el pasado de cual."
        )
    if not indice.is_monotonic_increasing:
        raise ErrorDatos(
            f"El indice de '{nombre}' no esta ordenado de menor a mayor. Ordenelo con "
            ".sort_index(): la ventana se construye por posicion, asi que sobre filas "
            "desordenadas la muestra de una fecha trae valores posteriores en el tiempo."
        )


def _como_marco(X: Any, nombre: str = "X") -> pd.DataFrame:
    """Convierte la entrada en un ``DataFrame`` de float sin tocar las filas.

    A diferencia de :func:`icm.comun.como_matriz`, aqui **no** se eliminan los
    ausentes: borrar filas desalinearia las ventanas y, peor, desalinearia las
    predicciones respecto al indice que el llamante espera recibir. Los
    ausentes se denuncian para que se decida fuera que hacer con ellos.

    El indice se valida aqui y no en cada llamante porque ``ajustar`` y
    ``predecir`` sufren la fuga del orden de las filas por separado: una red
    entrenada con la tabla bien ordenada sigue prediciendo el futuro si despues
    le dan la tabla al reves.
    """
    if isinstance(X, pd.DataFrame):
        marco = X.astype(float)
    elif isinstance(X, pd.Series):
        marco = X.to_frame().astype(float)
    else:
        valores = np.asarray(X, dtype=float)
        if valores.ndim == 1:
            valores = valores.reshape(-1, 1)
        if valores.ndim != 2:
            raise ErrorDatos(
                f"'{nombre}' debe ser una tabla de dos dimensiones; tiene forma "
                f"{valores.shape}."
            )
        marco = pd.DataFrame(valores)
    if marco.shape[1] == 0:
        raise ErrorDatos(f"'{nombre}' no tiene ninguna columna: no hay nada con lo que predecir.")
    _validar_indice_temporal(marco.index, nombre)
    if not np.isfinite(marco.to_numpy(dtype=float)).all():
        raise ErrorDatos(
            f"'{nombre}' contiene valores ausentes o infinitos. Elimine esas fechas con un "
            "'dropna' conjunto de X e y antes de ajustar: rellenarlas aqui inventaria "
            "historia, y descartarlas por dentro desalinearia las predicciones."
        )
    return marco


def _ventanas(valores: np.ndarray, ventana: int) -> np.ndarray:
    """Tensor ``(muestras, ventana, variables)`` de ventanas que solo miran atras.

    La ventana numero ``i`` son las filas ``i .. i+ventana-1``, es decir, la
    muestra cuyo "presente" es la fila ``i+ventana-1``. De ahi que las primeras
    ``ventana - 1`` filas no generen ninguna muestra: no tienen pasado
    suficiente, y rellenarlo con ceros o con la media seria inventar historia
    que el modelo tomaria por real.
    """
    deslizantes = np.lib.stride_tricks.sliding_window_view(valores, ventana, axis=0)
    return np.ascontiguousarray(deslizantes.transpose(0, 2, 1), dtype=np.float32)


def _escala(valores: np.ndarray) -> np.ndarray:
    """Desviaciones tipicas utilizables como divisor.

    Una columna constante tiene desviacion nula; dividir por ella daria
    infinitos que envenenan el gradiente entero. Se sustituye por 1, con lo que
    la columna queda centrada en cero y simplemente no aporta nada, que es
    exactamente lo que una constante puede aportar.
    """
    escalas = valores.std(axis=0, ddof=0)
    return np.where(escalas > 0.0, escalas, 1.0)


# --- modelo --------------------------------------------------------------------


class RedSecuencial:
    """Red neuronal secuencial con la interfaz ``ajustar`` / ``predecir``.

    Parametros
    ----------
    configuracion:
        Hiperparametros. Si se omite, :class:`ConfiguracionRed` por defecto.
    **kwargs:
        Atajo para sobrescribir campos concretos de la configuracion, tanto
        para escribir ``RedSecuencial(arquitectura="gru", ventana=10)`` como
        para que ``sklearn.base.clone`` pueda reconstruir el objeto a partir de
        ``get_params``.

    Notas
    -----
    ``predecir`` devuelve ``NaN`` en las ``ventana - 1`` primeras filas de la
    tabla que recibe, porque ahi no hay historia suficiente. Esto importa al
    combinarla con :func:`icm.ml.validacion.validacion_walk_forward`: cada
    tramo de prueba empieza con ese hueco, asi que conviene usar tramos de
    prueba bastante mas largos que la ventana y descartar los ausentes antes de
    calcular metricas.

    Ejemplo
    -------
    >>> red = RedSecuencial(arquitectura="gru", ventana=5, epocas=3)
    >>> red.get_params()["arquitectura"]
    'gru'
    """

    def __init__(self, configuracion: ConfiguracionRed | None = None, **kwargs: Any) -> None:
        base = configuracion if configuracion is not None else ConfiguracionRed()
        if not isinstance(base, ConfiguracionRed):
            raise ErrorDatos(
                "'configuracion' debe ser una ConfiguracionRed; se ha recibido "
                f"{type(base).__name__}. Pase los hiperparametros como argumentos con "
                "nombre si no quiere construirla."
            )
        _validar_nombres(kwargs)
        self.configuracion = replace(base, **kwargs) if kwargs else base

        self._red: Any = None
        self._columnas: list[Any] = []
        self._media_X: np.ndarray | None = None
        self._escala_X: np.ndarray | None = None
        self._media_y: float = 0.0
        self._escala_y: float = 1.0
        self._historial: list[dict[str, float]] = []

    # --- compatibilidad con scikit-learn ---------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Hiperparametros planos, al estilo de scikit-learn.

        Se devuelven los campos sueltos y no el objeto ``ConfiguracionRed``
        porque ``clone`` reconstruye el estimador con ``Clase(**parametros)`` y
        despues comprueba que cada parametro recuperado sea *el mismo objeto*
        que le paso. Con escalares eso se cumple; con un dataclass, que ``clone``
        copiaria en profundidad, no tendria por que.
        """
        return {nombre: getattr(self.configuracion, nombre) for nombre in NOMBRES_HIPERPARAMETROS}

    def set_params(self, **parametros: Any) -> RedSecuencial:
        """Cambia hiperparametros y descarta el ajuste anterior.

        El ajuste se descarta porque unos pesos entrenados con otra ventana u
        otra arquitectura no significan nada: conservarlos permitiria llamar a
        ``predecir`` y obtener numeros que no corresponden a la configuracion
        que el objeto declara tener.
        """
        if not parametros:
            return self
        _validar_nombres(parametros)
        self.configuracion = replace(self.configuracion, **parametros)
        self._red = None
        self._historial = []
        return self

    # --- ajuste ----------------------------------------------------------------

    def _preparar(self, X: Any, y: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Valida y alinea las entradas del ajuste."""
        marco = _como_marco(X, "X")
        if isinstance(y, pd.Series):
            # Con X sin indice (un ndarray) esta es la unica fecha disponible, y
            # un 'y' descendente invierte igualmente el emparejamiento.
            _validar_indice_temporal(y.index, "y")
            objetivo = y.to_numpy(dtype=float)
        else:
            objetivo = np.asarray(y, dtype=float).reshape(-1)
        if objetivo.ndim != 1:
            raise ErrorDatos(
                f"'y' debe ser unidimensional; tiene forma {objetivo.shape}. Seleccione una "
                "columna antes de ajustar."
            )
        if len(objetivo) != len(marco):
            raise ErrorDatos(
                f"'X' tiene {len(marco)} filas e 'y' tiene {len(objetivo)}: alinee ambos con "
                "un unico 'dropna' conjunto antes de ajustar."
            )
        if isinstance(X, (pd.DataFrame, pd.Series)) and isinstance(y, pd.Series):
            if not marco.index.equals(y.index):
                raise ErrorDatos(
                    "'X' e 'y' no comparten indice pese a tener la misma longitud: la ventana "
                    "se construye por posicion y emparejaria fechas distintas."
                )
        if not np.isfinite(objetivo).all():
            raise ErrorDatos(
                "'y' contiene valores ausentes o infinitos. Descarte esas fechas de X e y a la "
                "vez antes de ajustar."
            )
        return marco, objetivo

    def _tramos(self, n_muestras: int) -> tuple[int, int]:
        """Muestras de entrenamiento y de validacion, en ese orden temporal.

        Entre los dos tramos se descartan ``ventana - 1`` muestras a modo de
        embargo, por la misma razon que en
        :class:`icm.ml.validacion.ParticionWalkForward`: la primera ventana de
        validacion comparte filas con las ultimas de entrenamiento, asi que sin
        ese margen la perdida de validacion se mide en parte sobre datos ya
        vistos y la parada temprana se vuelve demasiado indulgente. Si la
        muestra es tan corta que el embargo dejaria el entrenamiento vacio, se
        recorta: es preferible un embargo incompleto a no poder ajustar.
        """
        configuracion = self.configuracion
        n_validacion = 0
        if configuracion.proporcion_validacion > 0.0:
            n_validacion = max(
                1, int(round(n_muestras * configuracion.proporcion_validacion))
            )
        if n_validacion >= n_muestras:
            n_validacion = n_muestras - 1
        # Sin tramo de validacion no hay nada de lo que separarse, y aplicar el
        # embargo solo tiraria muestras de entrenamiento a la basura.
        embargo = 0
        if n_validacion:
            embargo = min(configuracion.ventana - 1, max(0, n_muestras - n_validacion - 1))
        return n_muestras - n_validacion - embargo, n_validacion

    def ajustar(self, X: pd.DataFrame, y: pd.Series) -> RedSecuencial:
        """Entrena la red sobre ventanas pasadas de ``X`` para predecir ``y``.

        Se estandarizan variables y objetivo con la media y la desviacion del
        **tramo de entrenamiento** unicamente. La estandarizacion no es cosmetica
        en una red: con variables de escalas dispares, el gradiente de las
        grandes domina el de las pequenias y la optimizacion se atasca. Y hacerla
        con toda la serie seria una fuga de informacion del futuro, porque la
        media y la volatilidad de los anios siguientes acabarian metidas en cada
        observacion pasada.

        La perdida se minimiza sobre el objetivo estandarizado, asi que las
        cifras de :attr:`historial` estan en esa escala; para leerlas en las
        unidades de ``y`` hay que multiplicarlas por su varianza.
        """
        torch = _importar_torch()
        from torch import nn

        configuracion = self.configuracion
        marco, objetivo = self._preparar(X, y)
        ventana = configuracion.ventana
        n_muestras = len(marco) - ventana + 1
        if n_muestras < 2:
            raise ErrorDatos(
                f"Con una ventana de {ventana} y {len(marco)} filas solo se pueden formar "
                f"{max(n_muestras, 0)} muestras. Hacen falta al menos {ventana + 1} filas "
                "para formar dos muestras, y unas cuantas mas para que quede tramo de "
                "entrenamiento despues de reservar la validacion y el embargo; o reduzca "
                "'ventana'."
            )

        n_entrenamiento, n_validacion = self._tramos(n_muestras)
        # El control anterior mira el total; despues de restar la validacion y el
        # embargo puede quedar una sola muestra, y ese ajuste es degenerado (vease
        # MINIMO_ENTRENAMIENTO) sin que nada por fuera lo delate.
        if n_entrenamiento < MINIMO_ENTRENAMIENTO:
            embargadas = n_muestras - n_validacion - n_entrenamiento
            raise ErrorDatos(
                f"Con una ventana de {ventana} y {len(marco)} filas quedan {n_muestras} "
                f"muestras, y de ellas solo {n_entrenamiento} para entrenar: "
                f"{n_validacion} van al tramo de validacion y {embargadas} las descarta el "
                f"embargo. Hacen falta al menos {MINIMO_ENTRENAMIENTO}: alargue la serie, "
                "reduzca 'ventana' o baje 'proporcion_validacion'."
            )
        # Las ventanas de entrenamiento abarcan las filas [0, filas_entrenamiento)
        # de la tabla, y sus objetivos son los de [ventana - 1, filas_entrenamiento):
        # las primeras ventana - 1 fechas aportan historia pero nunca son
        # objetivo. Solo esos dos tramos pueden intervenir en la estandarizacion.
        filas_entrenamiento = n_entrenamiento + ventana - 1
        valores = marco.to_numpy(dtype=float)
        objetivos_entrenamiento = objetivo[ventana - 1: filas_entrenamiento]
        self._columnas = list(marco.columns)
        self._media_X = valores[:filas_entrenamiento].mean(axis=0)
        self._escala_X = _escala(valores[:filas_entrenamiento])
        self._media_y = float(objetivos_entrenamiento.mean())
        self._escala_y = float(_escala(objetivos_entrenamiento.reshape(-1, 1))[0])

        ventanas = _ventanas((valores - self._media_X) / self._escala_X, ventana)
        etiquetas = ((objetivo[ventana - 1:] - self._media_y) / self._escala_y).astype(np.float32)

        entradas = torch.from_numpy(ventanas)
        salidas = torch.from_numpy(etiquetas)
        X_entrenamiento = entradas[:n_entrenamiento]
        y_entrenamiento = salidas[:n_entrenamiento]
        X_validacion = entradas[n_muestras - n_validacion:]
        y_validacion = salidas[n_muestras - n_validacion:]

        torch.manual_seed(configuracion.semilla)
        generador = np.random.default_rng(configuracion.semilla)
        red = _crear_red(configuracion, len(self._columnas))
        optimizador = torch.optim.Adam(red.parameters(), lr=configuracion.tasa_aprendizaje)
        criterio = nn.MSELoss()

        historial: list[dict[str, float]] = []
        mejor_perdida = float("inf")
        # Arranca en None y no con los pesos de la red recien creada: si ninguna
        # epoca llega a mejorar -lo que ocurre cuando la perdida sale inf o NaN
        # en todas, porque ni 'inf < inf' ni 'nan < inf' son ciertos-, restaurar
        # esa copia devolveria la inicializacion ALEATORIA como si fuera el
        # resultado del entrenamiento, con 'ajustada' a True y predicciones de
        # aspecto normal.
        mejores_pesos: dict[str, Any] | None = None
        epocas_sin_mejora = 0
        # Sin tramo de validacion no hay parada temprana en absoluto: se corren
        # las epocas pedidas y se conservan los pesos de la ultima. Sustituir la
        # perdida de validacion por la de entrenamiento, como se hacia, no es
        # una parada temprana peor sino una averia con el nombre de la buena. Por
        # un lado no protege: la perdida de entrenamiento baja casi siempre, que
        # es justo lo que hace un modelo al sobreajustar. Por otro hace dano: es
        # la media de los minilotes medida en modo train, o sea con el abandono
        # activo y con los pesos de ANTES de cada paso del optimizador, asi que
        # su ruido agota la paciencia y corta el entrenamiento a media
        # convergencia acreditando a los pesos guardados una cifra que
        # corresponde a otro estado de la red.
        parada_temprana = n_validacion > 0

        for epoca in range(1, configuracion.epocas + 1):
            red.train()
            # Barajar los minilotes no es hacer trampas: el orden en que se
            # presentan las muestras no cambia que informacion contiene cada
            # una. Lo que jamas se baraja es la eleccion del tramo de
            # validacion, que sigue siendo el ultimo en el tiempo.
            orden = generador.permutation(n_entrenamiento)
            acumulada = 0.0
            for inicio in range(0, n_entrenamiento, configuracion.tamano_lote):
                posiciones = orden[inicio: inicio + configuracion.tamano_lote]
                indices = torch.from_numpy(posiciones)
                optimizador.zero_grad()
                perdida = criterio(red(X_entrenamiento[indices]), y_entrenamiento[indices])
                perdida.backward()
                optimizador.step()
                acumulada += float(perdida.detach()) * len(posiciones)
            perdida_entrenamiento = acumulada / n_entrenamiento

            perdida_validacion = float("nan")
            if parada_temprana:
                red.eval()
                with torch.no_grad():
                    perdida_validacion = float(criterio(red(X_validacion), y_validacion))

            historial.append(
                {
                    "epoca": epoca,
                    "perdida_entrenamiento": perdida_entrenamiento,
                    "perdida_validacion": perdida_validacion,
                }
            )

            if not parada_temprana:
                continue
            if perdida_validacion < mejor_perdida:
                mejor_perdida = perdida_validacion
                mejores_pesos = copy.deepcopy(red.state_dict())
                epocas_sin_mejora = 0
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= configuracion.paciencia:
                    break

        if parada_temprana:
            if mejores_pesos is None:
                raise ErrorDatos(
                    f"La perdida de validacion no fue finita en ninguna de las "
                    f"{len(historial)} epocas, asi que no hay ninguna 'mejor epoca' que "
                    "restaurar y devolver la red con los pesos de su inicializacion "
                    "aleatoria seria peor que fallar. La causa habitual es un valor extremo "
                    "en 'y' que desborda al elevarse al cuadrado sobre la escala del "
                    "entrenamiento (con validacion interna, en su ultimo tramo): recortelo, "
                    "pase a logaritmos o baje 'tasa_aprendizaje'."
                )
            # Los pesos que se conservan son los de la mejor epoca. Quedarse con
            # los ultimos haria inutil la parada temprana: son justamente los que
            # ya habian empezado a sobreajustar durante las epocas de paciencia.
            red.load_state_dict(mejores_pesos)
        red.eval()
        self._red = red
        self._historial = historial
        return self

    # --- prediccion ------------------------------------------------------------

    def _alinear_columnas(self, X: Any) -> np.ndarray:
        """Valores de ``X`` en el orden de columnas con el que se ajusto la red.

        Las capas de una red multiplican por posicion, no por nombre: un
        ``DataFrame`` con las mismas columnas en otro orden daria predicciones
        sin sentido y sin ningun aviso.
        """
        marco = _como_marco(X, "X")
        if isinstance(X, (pd.DataFrame, pd.Series)):
            faltan = [c for c in self._columnas if c not in marco.columns]
            if faltan:
                raise ErrorDatos(
                    f"Faltan en 'X' columnas con las que se ajusto la red: {faltan}."
                )
            marco = marco.reindex(columns=self._columnas)
        elif marco.shape[1] != len(self._columnas):
            raise ErrorDatos(
                f"La red se ajusto con {len(self._columnas)} variables y 'X' trae "
                f"{marco.shape[1]}."
            )
        return marco.to_numpy(dtype=float)

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones, una por fila de ``X``, con ``NaN`` al principio.

        Las ``ventana - 1`` primeras filas reciben ``NaN`` en lugar de un valor
        inventado. Es informacion, no un fallo: dice que en esas fechas el
        modelo no puede pronunciarse porque le falta pasado. Rellenarlas
        completando la ventana con ceros o con la media daria un numero de
        apariencia normal, imposible de distinguir despues de una prediccion
        legitima, y contaminaria cualquier metrica o simulacion de estrategia
        que se calcule con el.
        """
        torch = _importar_torch()
        if self._red is None:
            raise ErrorDatos(
                "La red no esta ajustada: llame a 'ajustar(X, y)' antes de 'predecir(X)'."
            )
        valores = self._alinear_columnas(X)
        ventana = self.configuracion.ventana
        salida = np.full(len(valores), np.nan)
        if len(valores) < ventana:
            return salida

        ventanas = _ventanas((valores - self._media_X) / self._escala_X, ventana)
        with torch.no_grad():
            crudas = self._red(torch.from_numpy(ventanas)).numpy().reshape(-1)
        salida[ventana - 1:] = crudas.astype(float) * self._escala_y + self._media_y
        return salida

    # --- diagnostico -----------------------------------------------------------

    @property
    def historial(self) -> pd.DataFrame:
        """Perdida de entrenamiento y de validacion por epoca.

        El indice es la epoca, empezando en 1, y su longitud dice si la parada
        temprana actuo (menos filas que ``epocas``) o si el maximo se agoto, que
        suele significar que conviene subir ``epocas``. Las dos columnas juntas
        son el diagnostico habitual de sobreajuste: la de entrenamiento baja y
        la de validacion empieza a subir.

        Con ``proporcion_validacion`` a 0, ``perdida_validacion`` es ``NaN``
        entera: no hay tramo reservado sobre el que medirla. Antes se rellenaba
        con una copia de la de entrenamiento, que daba dos columnas identicas y
        aparentaba un diagnostico de sobreajuste donde no habia ninguno.

        La de entrenamiento es la media ponderada de los minilotes de la epoca,
        cada uno medido antes de su paso del optimizador y con el abandono
        activo, asi que describe el recorrido de la epoca y no el estado en que
        la red la termina. Sirve para ver si converge; no para comparar epocas
        entre si.
        """
        if not self._historial:
            raise ErrorDatos(
                "Todavia no hay historial: llame a 'ajustar(X, y)' antes de consultarlo."
            )
        return pd.DataFrame(self._historial).set_index("epoca")

    @property
    def ajustada(self) -> bool:
        """Indica si la red ya tiene pesos entrenados."""
        return self._red is not None
