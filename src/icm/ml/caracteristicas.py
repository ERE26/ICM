"""Ingenieria de variables para series temporales financieras.

La regla que gobierna el modulo entero
--------------------------------------
El valor de una variable en la fecha ``t`` solo puede depender de informacion
disponible en ``t``. Esa regla se incumple con una facilidad asombrosa: un
``shift`` con el signo cambiado, una media movil centrada, una tipificacion
hecha con la media de toda la muestra. Ninguno de esos tres fallos rompe el
codigo ni deja NaN sospechosos; simplemente convierten el modelo en un oraculo
que en produccion no funciona.

Por eso la garantia no se deja escrita en un comentario, sino que se somete a
contraste: :func:`comprobar_ausencia_de_fuga` reconstruye la matriz con la
serie recortada y con el futuro alterado, y compara. Si una variable de la
fecha ``t`` reacciona a lo que pasa despues de ``t``, salta un
:class:`~icm.comun.ErrorDatos` que dice que columna y que fecha.

La variable objetivo es la unica excepcion, y por definicion: predecir es
justamente poner en ``t`` una cifra que no se conocera hasta ``t+h``. Por eso
la comprobacion audita ``X`` y nunca ``y``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos

#: Nombres de los dias de la semana, en el orden de ``DatetimeIndex.dayofweek``.
DIAS_SEMANA = (
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
)

#: Familias de variables que sabe construir :func:`construir_matriz`.
FAMILIAS = ("retardos", "medias", "volatilidad", "momento", "calendario", "cruzadas")

RETARDOS_PROPIOS = (1, 2, 3, 5, 10)
VENTANAS_MEDIAS = (5, 10, 21)
VENTANAS_VOLATILIDAD = (5, 21, 63)
VENTANAS_MOMENTO = (5, 21, 63)

#: Retardos con los que entran los demas activos del marco. Empiezan en 1: los
#: mercados no cierran a la vez, asi que dar por conocido el cierre de hoy de
#: otro mercado es una suposicion que conviene no hacer gratis.
RETARDOS_CRUZADOS = (1, 2, 5)

# Nombre interno del objetivo mientras viaja pegado a X; se renombra al final.
# Evita que choque con una columna que se llame igual.
_CLAVE_OBJETIVO = "__objetivo__"

# Las tolerancias de la comprobacion de fuga. Las ventanas moviles de pandas se
# calculan de forma incremental, asi que dos recorridos sobre los mismos datos
# pueden diferir en el ultimo bit; una fuga real cambia cifras significativas.
_TOLERANCIA_RELATIVA = 1e-9
_TOLERANCIA_ABSOLUTA = 1e-12

# Cuantos cortes usa la auditoria cuando no se le indican fechas. Son muchos
# mas que los tres de antes porque una fuga condicionada a los datos (un
# relleno hacia atras, una interpolacion) solo se delata si algun corte cae
# junto al hueco que la dispara. Y estan acotados porque cada corte cuesta dos
# reconstrucciones completas de la matriz: sin tope, auditar una serie larga
# dejaria de caber en una bateria de pruebas y se dejaria de ejecutar.
_MAX_CORTES_POR_OMISION = 48

# Dias que se desplazan las fechas posteriores al corte en el tercer
# experimento. Cualquier valor positivo sirve; siete deja el hueco bien visible
# sin romper el orden del indice.
_DIAS_CALENDARIO_ALTERADO = 7


# --- utilidades internas -------------------------------------------------------


def _como_marco(datos, nombre: str, base: str = "serie") -> pd.DataFrame:
    """Convierte la entrada en DataFrame de float **conservando los ausentes**.

    A diferencia de :func:`icm.comun.como_matriz`, aqui no se descartan filas:
    los NaN del periodo de calentamiento de cada ventana son informacion, no
    suciedad, y quien decide que fila sobrevive es :func:`construir_matriz`
    cuando ya estan todas las variables alineadas.
    """
    if isinstance(datos, pd.Series):
        serie = datos if datos.name is not None else datos.rename(base)
        marco = serie.to_frame()
    elif isinstance(datos, pd.DataFrame):
        marco = datos.copy()
    else:
        raise ErrorDatos(
            f"'{nombre}' debe ser una Series o un DataFrame de pandas con indice de "
            f"fechas; se ha recibido {type(datos).__name__}."
        )
    if marco.empty:
        raise ErrorDatos(f"'{nombre}' no contiene ninguna observacion.")
    try:
        return marco.astype(float)
    except (TypeError, ValueError) as exc:
        raise ErrorDatos(f"'{nombre}' contiene valores no numericos: {exc}.") from exc


def _como_serie(datos, nombre: str, base: str = "serie") -> pd.Series:
    """Igual que :func:`_como_marco` pero exigiendo una unica columna."""
    marco = _como_marco(datos, nombre, base)
    if marco.shape[1] != 1:
        raise ErrorDatos(
            f"Se esperaba una unica serie en '{nombre}', se han recibido "
            f"{marco.shape[1]} columnas. Seleccione una columna antes de llamar."
        )
    return marco.iloc[:, 0]


def _validar_enteros(valores, nombre: str, minimo: int, motivo: str) -> tuple[int, ...]:
    """Valida una secuencia de enteros, sin duplicados y en el orden recibido."""
    if isinstance(valores, (str, bytes)) or not isinstance(valores, Sequence):
        try:
            valores = list(valores)
        except TypeError as exc:
            raise ErrorDatos(
                f"'{nombre}' debe ser una secuencia de enteros, por ejemplo (1, 2, 5)."
            ) from exc
    lista = list(valores)
    if not lista:
        raise ErrorDatos(f"'{nombre}' esta vacio: indique al menos un valor.")
    salida: list[int] = []
    for valor in lista:
        try:
            numero = float(valor)
        except (TypeError, ValueError) as exc:
            raise ErrorDatos(f"'{nombre}' solo admite enteros; se ha recibido {valor!r}.") from exc
        if not numero.is_integer():
            raise ErrorDatos(f"'{nombre}' solo admite enteros; se ha recibido {valor!r}.")
        entero = int(numero)
        if entero < minimo:
            raise ErrorDatos(f"'{nombre}' no admite el valor {entero}: {motivo}")
        salida.append(entero)
    return tuple(dict.fromkeys(salida))


def _validar_columnas_unicas(marco: pd.DataFrame, nombre: str) -> None:
    """Exige nombres de columna distintos.

    Con dos columnas homonimas, ``marco[objetivo]`` devuelve un DataFrame en vez
    de una Series y la primera comprobacion que se haga sobre el revienta con un
    mensaje de pandas que no dice ni cual es la columna culpable ni que hacer.
    """
    repetidas = sorted({str(c) for c in marco.columns[marco.columns.duplicated()]})
    if repetidas:
        raise ErrorDatos(
            f"'{nombre}' tiene columnas con el mismo nombre ({', '.join(repetidas)}): no "
            "se sabe cual de ellas es el activo. Renombrelas o quedese con una, por "
            f"ejemplo con {nombre}.loc[:, ~{nombre}.columns.duplicated()]."
        )


def _validar_indice_sin_duplicados(indice: pd.Index, nombre: str) -> None:
    """Exige un indice sin etiquetas repetidas.

    Va aparte del orden porque hay entradas que se alinean por etiqueta y a las
    que el desorden no afecta (una exogena se incorpora con ``reindex``, que
    trabaja por etiqueta), mientras que las etiquetas repetidas rompen ese
    mismo ``reindex`` y ademas dejan ambiguo que valor corresponde a la fecha.
    """
    if indice.has_duplicates:
        repetidas = [_texto(f) for f in indice[indice.duplicated()].unique()[:3]]
        raise ErrorDatos(
            f"El indice de '{nombre}' tiene fechas repetidas ({', '.join(repetidas)}...): "
            "agregue o elimine los duplicados antes de construir variables."
        )


def _validar_indice_temporal(indice: pd.Index, nombre: str) -> None:
    """Exige un indice ordenado y sin fechas repetidas.

    No es celo burocratico. Con las filas desordenadas, ``shift(1)`` no trae el
    dato anterior sino el de la fila de arriba, que puede ser posterior en el
    tiempo: la fuga se cuela sin que nadie haya escrito un retardo negativo.
    Ordenar por dentro y en silencio seria peor, porque el resto del analisis
    seguiria trabajando con el orden original.
    """
    _validar_indice_sin_duplicados(indice, nombre)
    if not indice.is_monotonic_increasing:
        raise ErrorDatos(
            f"El indice de '{nombre}' no esta ordenado de menor a mayor. Ordenelo con "
            ".sort_index(): con las filas desordenadas un retardo puede traer un valor "
            "posterior en el tiempo."
        )


def huella_alineamiento(serie: pd.Series) -> tuple:
    """Fechas del primer y del ultimo valor de ``serie``: su huella de alineamiento.

    Sirve para desconfiar de una marca caducada. ``attrs`` viaja pegado al
    objeto y sobrevive intacto a ``.shift()``, ``.dropna()``, ``.reindex()`` y a
    la aritmetica, asi que ``attrs["convencion"] = "futuro"`` dice de donde
    salio la serie y no como esta alineada **ahora**: quien la realinea con
    ``.shift(1)`` se queda con una serie contemporanea que sigue diciendo
    "futuro", y el consumidor
    (:func:`icm.ml.estrategia.ejecutar_estrategia`) la ejecuta con retardo 0,
    que es leer el futuro.

    Estas dos fechas si se mueven con cualquier desplazamiento (``.shift()``
    cambia al menos una de las dos, tambien con ``fill_value``) y en cambio no
    cambian al limpiar los ausentes ni al reescalar, que son operaciones que no
    tocan el alineamiento. Por eso son la huella: comparar
    ``serie.attrs.get("huella")`` con ``huella_alineamiento(serie)`` distingue
    una serie tal como la produjo este modulo de una que ya se ha realineado,
    sin dar la alarma por operaciones inocentes.
    """
    return (serie.first_valid_index(), serie.last_valid_index())


def _marcar_alineamiento(serie: pd.Series) -> pd.Series:
    """Graba la huella del alineamiento actual de la serie en ``attrs``."""
    serie.attrs["huella"] = huella_alineamiento(serie)
    return serie


def _nombre_base(columna, prefijo: str | None, era_serie: bool) -> str:
    """Nombre con el que se bautizan las variables derivadas de una columna.

    Con una Series el prefijo sustituye al nombre (solo hay uno que sustituir);
    con un DataFrame lo antepone, para no perder de que activo viene cada
    columna.
    """
    if prefijo is None:
        return str(columna)
    return prefijo if era_serie else f"{prefijo}_{columna}"


# --- familias de variables -----------------------------------------------------


def retardos(
    datos: pd.Series | pd.DataFrame,
    retardos: Sequence[int] = (1, 2, 3, 5, 10),
    prefijo: str | None = None,
) -> pd.DataFrame:
    """Valores pasados de una o varias series.

    Es la familia mas simple y la mas dificil de equivocar, con una excepcion:
    el signo. Un retardo negativo traeria el valor de una fecha posterior, que
    es exactamente la fuga que este modulo persigue, asi que se rechaza. El
    retardo 0 (el valor del propio dia) si se admite, porque hay variables que
    de verdad se conocen a tiempo; vease la discusion de ``retardos_exogenas``
    en :func:`construir_matriz`.

    Parametros
    ----------
    prefijo:
        Con una Series sustituye a su nombre; con un DataFrame se antepone al
        de cada columna. Las columnas resultantes se llaman ``base_ret{k}``.
    """
    era_serie = isinstance(datos, pd.Series)
    marco = _como_marco(datos, "datos", base=prefijo or "serie")
    desplazamientos = _validar_enteros(
        retardos,
        "retardos",
        minimo=0,
        motivo=(
            "un retardo negativo tomaria el valor de una fecha posterior, que es "
            "informacion del futuro. Use retardos mayores o iguales que cero."
        ),
    )

    salida: dict[str, pd.Series] = {}
    for columna in marco.columns:
        base = _nombre_base(columna, prefijo, era_serie)
        for k in desplazamientos:
            salida[f"{base}_ret{k}"] = marco[columna].shift(k)
    return pd.DataFrame(salida, index=marco.index)


def medias_moviles(
    serie: pd.Series,
    ventanas: Sequence[int] = (5, 10, 21),
    relativa: bool = True,
) -> pd.DataFrame:
    """Medias moviles de una serie de nivel (tipicamente, precios).

    Parametros
    ----------
    relativa:
        Si es cierto se devuelve ``precio / media - 1`` en lugar de la media.
        Importa mucho mas de lo que parece. La media movil en niveles hereda
        la escala y la tendencia del precio: vale 23 en el gas y 140 en la
        electricidad, y en la misma serie vale el doble tres anios despues. Un
        modelo entrenado con eso aprende el nivel de la muestra de
        entrenamiento, no el comportamiento, y se rompe en cuanto el precio
        sale de ese rango. La distancia relativa es adimensional, estacionaria
        y directamente comparable entre activos y entre epocas: "el precio esta
        un 3 % por encima de su media de 21 sesiones" significa lo mismo en
        2015 y en 2022, en el gas y en la electricidad.

    Notas
    -----
    La ventana es retrospectiva y **cerrada en ``t``**: incluye el dato del
    propio dia, que a cierre ya se conoce, y ninguno posterior. Las primeras
    ``ventana - 1`` fechas quedan a NaN a proposito; rellenarlas hacia atras
    seria colar informacion del futuro por la puerta de servicio.
    """
    serie = _como_serie(serie, "serie", base="serie")
    anchuras = _validar_enteros(
        ventanas, "ventanas", minimo=1, motivo="una ventana debe abarcar al menos un periodo."
    )
    if relativa and bool((serie <= 0).any()):
        raise ErrorDatos(
            "La media movil relativa divide por la media, asi que exige una serie "
            "estrictamente positiva (precios, no rendimientos). Use relativa=False "
            "o pase la serie de precios."
        )

    base = str(serie.name)
    salida: dict[str, pd.Series] = {}
    for ventana in anchuras:
        media = serie.rolling(ventana).mean()
        if relativa:
            salida[f"{base}_mm{ventana}_rel"] = serie / media - 1.0
        else:
            salida[f"{base}_mm{ventana}"] = media
    return pd.DataFrame(salida, index=serie.index)


def volatilidad_realizada(
    rendimientos: pd.Series | pd.DataFrame,
    ventanas: Sequence[int] = (5, 21, 63),
) -> pd.DataFrame:
    """Desviacion tipica movil de los rendimientos, anualizada.

    Las tres ventanas por defecto son aproximadamente una semana, un mes y un
    trimestre de sesiones. Tenerlas a la vez no es redundante: su cociente es
    lo que informa de si la volatilidad esta subiendo o bajando de regimen, que
    en energia es la variable que manda.

    La anualizacion (multiplicar por la raiz de 252) no aniade ni quita
    informacion, solo pone la cifra en las unidades en las que se cotiza la
    volatilidad, de modo que un 0.45 se lee directamente como "45 %".
    """
    marco = _como_marco(rendimientos, "rendimientos", base="rend")
    anchuras = _validar_enteros(
        ventanas, "ventanas", minimo=2, motivo="la desviacion tipica necesita al menos dos datos."
    )

    factor = float(np.sqrt(DIAS_HABILES_ANIO))
    salida: dict[str, pd.Series] = {}
    for columna in marco.columns:
        for ventana in anchuras:
            salida[f"{columna}_vol{ventana}"] = marco[columna].rolling(ventana).std() * factor
    return pd.DataFrame(salida, index=marco.index)


def momento(
    rendimientos: pd.Series | pd.DataFrame,
    ventanas: Sequence[int] = (5, 21, 63),
) -> pd.DataFrame:
    """Rendimiento acumulado en las ultimas ``ventana`` sesiones.

    Con rendimientos logaritmicos la suma movil **es** exactamente el
    rendimiento acumulado del periodo, sin aproximacion ninguna: esa aditividad
    temporal es la razon de usar logaritmos. Con rendimientos simples la suma
    es solo una aproximacion de primer orden, aceptable en horizontes cortos.
    """
    marco = _como_marco(rendimientos, "rendimientos", base="rend")
    anchuras = _validar_enteros(
        ventanas, "ventanas", minimo=1, motivo="una ventana debe abarcar al menos un periodo."
    )

    salida: dict[str, pd.Series] = {}
    for columna in marco.columns:
        for ventana in anchuras:
            salida[f"{columna}_mom{ventana}"] = marco[columna].rolling(ventana).sum()
    return pd.DataFrame(salida, index=marco.index)


def calendario(indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables de calendario: dia de la semana y posicion en el anio.

    Dia de la semana
    ----------------
    Se codifica en indicadores ``0/1``, uno por dia, **quitando el lunes**. Con
    los siete indicadores la suma seria siempre 1, es decir, identica a la
    constante del modelo: la matriz queda singular y los coeficientes dejan de
    estar identificados. Quitando una categoria, cada coeficiente se lee como
    "cuanto se aparta este dia del lunes".

    Las siete categorias se emiten siempre, aunque el indice sea de dias
    habiles y las columnas de sabado y domingo queden a cero. Es deliberado: el
    conjunto de variables no debe depender de los datos que le hayan tocado a
    la muestra, que es la misma disciplina que no mirar al futuro. Una columna
    constante recibe coeficiente nulo y no hace dano.

    Posicion en el anio
    -------------------
    Se codifica con el seno y el coseno del angulo ``2*pi*dia/dias_del_anio``,
    no con el numero de mes. La razon es que el calendario es circular: el 31
    de diciembre y el 1 de enero son dias consecutivos. Con un numero de mes
    quedan a 11 unidades de distancia, la maxima posible, y el modelo aprende
    un salto que en la realidad no existe. Con seno y coseno son dos puntos
    contiguos de la circunferencia, a una distancia de unos 0,017, que es la
    verdad del calendario. La estacionalidad anual (invierno frio, verano de
    aire acondicionado) se recupera como combinacion lineal de ambos, con
    amplitud y fase libres.
    """
    if not isinstance(indice, pd.DatetimeIndex):
        raise ErrorDatos(
            "El calendario necesita un DatetimeIndex. Convierta el indice con "
            "pd.to_datetime(...) antes de construir las variables de calendario."
        )
    if indice.empty:
        raise ErrorDatos("El indice no contiene ninguna fecha.")

    salida: dict[str, np.ndarray] = {}
    dia_semana = indice.dayofweek.to_numpy()
    for numero, nombre in enumerate(DIAS_SEMANA):
        if numero == 0:  # el lunes es la categoria de referencia
            continue
        salida[f"dia_{nombre}"] = (dia_semana == numero).astype(float)

    dias_del_anio = np.where(indice.is_leap_year, 366.0, 365.0)
    angulo = 2.0 * np.pi * (indice.dayofyear.to_numpy() - 1) / dias_del_anio
    salida["sin_anio"] = np.sin(angulo)
    salida["cos_anio"] = np.cos(angulo)
    return pd.DataFrame(salida, index=indice)


# --- variable objetivo ---------------------------------------------------------


def objetivo_rendimiento_futuro(
    precios: pd.Series,
    horizonte: int = 1,
    tipo: str = "log",
) -> pd.Series:
    """Rendimiento de ``t`` a ``t+horizonte``, **indexado por ``t``**.

    El alineamiento es la mitad del trabajo de un modelo de prediccion y la
    fuente de errores mas cara que existe. Aqui la convencion es explicita:
    ``y[t]`` es lo que ocurrira entre ``t`` y ``t+horizonte``, guardado en la
    fila ``t``, porque ``t`` es la fecha en la que se toma la decision y en la
    que estan medidas todas las variables explicativas. Indexarlo por
    ``t+horizonte`` (el error habitual) desplaza el modelo entero un periodo y
    produce un R cuadrado estupendo que no se puede cobrar.

    Las ultimas ``horizonte`` fechas quedan a NaN: su futuro todavia no ha
    ocurrido. Son justamente las filas sobre las que se predice en produccion.

    La serie sale marcada en ``attrs``: ``convencion="futuro"``, ``horizonte``,
    ``tipo_rendimiento`` y ``huella``, la huella del alineamiento que describe
    la marca (vease :func:`huella_alineamiento`). El consumidor de la marca,
    :func:`icm.ml.estrategia.ejecutar_estrategia`, deduce de ella el retardo de
    ejecucion; con la huella puede comprobar ademas que la serie no se ha
    realineado por el camino, porque ``attrs`` sobrevive a ``.shift()`` y la
    marca sola no distingue una serie adelantada de una que ya no lo esta.
    """
    serie = _como_serie(precios, "precios", base="precio")
    if tipo not in {"log", "simple"}:
        raise ErrorDatos(f"Tipo de rendimiento '{tipo}' desconocido: use 'log' o 'simple'.")
    (paso,) = _validar_enteros(
        [horizonte],
        "horizonte",
        minimo=1,
        motivo="el horizonte de prediccion debe ser de al menos un periodo hacia delante.",
    )
    if bool((serie <= 0).any()):
        raise ErrorDatos(
            "Hay precios nulos o negativos: no se puede calcular el rendimiento futuro."
        )

    futuro = serie.shift(-paso)
    valores = np.log(futuro / serie) if tipo == "log" else futuro / serie - 1.0
    salida = valores.rename(f"rend_futuro_{paso}")
    # Se deja constancia de que la serie esta ADELANTADA: su valor en t es el
    # rendimiento de t a t+paso, no el de t-1 a t. Sin esta marca,
    # icm.ml.estrategia.ejecutar_estrategia no puede distinguirla de una serie
    # de rendimientos corriente y aplicaria su retardo por defecto, desplazando
    # dos veces y destruyendo el resultado sin dar ninguna senial de alarma.
    salida.attrs["convencion"] = "futuro"
    salida.attrs["horizonte"] = paso
    salida.attrs["tipo_rendimiento"] = tipo
    # Y junto a la marca, la huella del alineamiento que describe: la marca sola
    # sobrevive a .shift() y entonces afirma un adelanto que ya no existe.
    # Vease :func:`huella_alineamiento`.
    return _marcar_alineamiento(salida)


# --- matriz de datos -----------------------------------------------------------


@dataclass
class MatrizDatos:
    """Matriz de variables, objetivo y mapa de familias, ya alineados.

    ``nombres_grupo`` permite tratar las variables por bloques sin volver a
    adivinar nombres: quitar el calendario, penalizar solo las cruzadas o medir
    cuanto aporta cada familia.
    """

    X: pd.DataFrame
    y: pd.Series
    nombres_grupo: dict[str, list[str]]

    def __post_init__(self) -> None:
        if not self.X.index.equals(self.y.index):
            raise ErrorDatos(
                "X e y no comparten indice: alinee ambos con un unico 'dropna' "
                "conjunto antes de construir MatrizDatos."
            )

    def __len__(self) -> int:
        return len(self.X)


def construir_matriz(
    precios: pd.DataFrame,
    objetivo: str,
    horizonte: int = 1,
    exogenas: pd.DataFrame | pd.Series | None = None,
    retardos_exogenas: Sequence[int] = (0, 1),
    incluir: Sequence[str] = (
        "retardos",
        "medias",
        "volatilidad",
        "momento",
        "calendario",
        "cruzadas",
    ),
) -> MatrizDatos:
    """Matriz completa de variables explicativas y objetivo para un activo.

    Parametros
    ----------
    precios:
        Un activo por columna. Del activo ``objetivo`` salen las familias
        propias; del resto, las cruzadas.
    incluir:
        Familias a construir, de :data:`FAMILIAS`. ``"cruzadas"`` son los
        retardos de los **demas** activos del marco, y en energia no son un
        adorno: buena parte de la senal del precio electrico viene de fuera del
        propio precio electrico, porque el gas fija el coste de la tecnologia
        marginal y se traslada con retardo de horas o dias. Un modelo
        univariante del precio electrico esta tirando esa informacion.
    retardos_exogenas:
        Retardos con los que entran las variables de ``exogenas``. Incluir el
        **retardo 0** significa usar el valor del mismo dia, y eso solo es
        legitimo si en la practica se conoce por adelantado. El caso tipico, y
        la razon de que sea el valor por defecto, es una **prevision**
        meteorologica: la prevision de temperatura o de viento para maniana
        esta publicada hoy, asi que al decidir en ``t`` se conoce el valor
        previsto para ``t``. Con una exogena **observada** (la temperatura que
        de verdad hizo, la demanda liquidada, un dato macro que se publica con
        retraso) el retardo 0 es fuga pura y hay que empezar en 1. Ante la
        duda, empiece en 1: perder un dia de informacion cuesta mucho menos que
        un modelo que no se puede ejecutar.

    Notas
    -----
    Aqui no se tipifica ni se escala nada. Cualquier estadistico calculado
    sobre la muestra entera (media, desviacion, minimo, maximo) mete
    informacion del conjunto de prueba en el de entrenamiento. El escalado es
    trabajo del ``Pipeline``, que lo ajusta solo con el tramo de entrenamiento
    de cada particion.
    """
    marco = _como_marco(precios, "precios", base="precio")
    _validar_indice_temporal(marco.index, "precios")
    _validar_columnas_unicas(marco, "precios")
    if objetivo not in marco.columns:
        disponibles = ", ".join(str(c) for c in marco.columns)
        raise ErrorDatos(
            f"La columna objetivo '{objetivo}' no esta en 'precios'. "
            f"Columnas disponibles: {disponibles}."
        )
    familias = list(incluir)
    desconocidas = [f for f in familias if f not in FAMILIAS]
    if desconocidas:
        raise ErrorDatos(
            f"Familias de variables desconocidas: {desconocidas}. "
            f"Disponibles: {', '.join(FAMILIAS)}."
        )
    if not familias and exogenas is None:
        raise ErrorDatos(
            "No se ha pedido ninguna variable: 'incluir' esta vacio y no hay exogenas."
        )

    serie = marco[objetivo]
    if bool((serie <= 0).any()):
        raise ErrorDatos(
            f"La serie '{objetivo}' tiene precios nulos o negativos: no se pueden "
            "calcular rendimientos logaritmicos. Revise los datos o trabaje en niveles."
        )
    rendimientos = np.log(serie).diff().rename("rend")

    grupos: dict[str, list[str]] = {}
    bloques: list[pd.DataFrame] = []

    def agregar(familia: str, variables: pd.DataFrame) -> None:
        if variables.shape[1] == 0:
            return
        bloques.append(variables)
        grupos[familia] = list(variables.columns)

    if "retardos" in familias:
        agregar("retardos", retardos(rendimientos, RETARDOS_PROPIOS))
    if "medias" in familias:
        agregar("medias", medias_moviles(serie.rename("precio"), VENTANAS_MEDIAS, relativa=True))
    if "volatilidad" in familias:
        agregar("volatilidad", volatilidad_realizada(rendimientos, VENTANAS_VOLATILIDAD))
    if "momento" in familias:
        agregar("momento", momento(rendimientos, VENTANAS_MOMENTO))
    if "calendario" in familias:
        agregar("calendario", calendario(marco.index))
    if "cruzadas" in familias:
        otros = [c for c in marco.columns if c != objetivo]
        if otros:
            if bool((marco[otros] <= 0).any().any()):
                raise ErrorDatos(
                    "Hay precios nulos o negativos en los activos cruzados. Quite "
                    "'cruzadas' de 'incluir' o depure esas columnas."
                )
            cruzados = np.log(marco[otros]).diff()
            agregar("cruzadas", retardos(cruzados, RETARDOS_CRUZADOS, prefijo="cruz"))

    if exogenas is not None:
        exo = _como_marco(exogenas, "exogenas", base="exogena")
        # El desorden no importa (el reindex posterior alinea por etiqueta, no
        # por posicion), pero las fechas repetidas si: rompen ese reindex.
        _validar_indice_sin_duplicados(exo.index, "exogenas")
        comunes = exo.index.intersection(marco.index)
        if len(comunes) == 0:
            raise ErrorDatos(
                "El indice de 'exogenas' no se solapa con el de 'precios': no hay "
                "ninguna fecha en comun. Reindexe las exogenas al calendario de precios."
            )
        agregar("exogenas", retardos(exo.reindex(marco.index), retardos_exogenas, prefijo="exo"))

    if not bloques:
        # Pasa con incluir=('cruzadas',) y un solo activo: la familia existe, se
        # pide, y no aporta ninguna columna porque no hay ningun OTRO activo.
        # El control de arriba no lo ve, porque 'incluir' no esta vacio.
        raise ErrorDatos(
            "Las familias pedidas no han producido ninguna variable con "
            f"{marco.shape[1]} activo(s) en 'precios': 'cruzadas' son los retardos de "
            f"los DEMAS activos y '{objetivo}' es el unico que hay. Aniada al marco los "
            "activos cruzados que quiera usar, o pida alguna familia mas de "
            f"{{{', '.join(f for f in FAMILIAS if f != 'cruzadas')}}}."
        )

    X = pd.concat(bloques, axis=1)
    repetidas = sorted(set(X.columns[X.columns.duplicated()]))
    if repetidas:
        raise ErrorDatos(
            f"Hay nombres de variable repetidos: {repetidas}. Renombre las columnas "
            "de 'precios' o de 'exogenas' para que no colisionen."
        )

    y = objetivo_rendimiento_futuro(serie, horizonte, "log")
    completo = pd.concat([X, y.rename(_CLAVE_OBJETIVO)], axis=1).dropna(how="any")
    if completo.empty:
        raise ErrorDatos(
            f"No queda ninguna fila completa: con {len(marco)} fechas no hay margen "
            "para el calentamiento de las ventanas mas largas mas el horizonte de "
            f"{horizonte}. Use una serie mas larga o reduzca 'incluir'."
        )

    objetivo_alineado = completo[_CLAVE_OBJETIVO].rename(y.name)
    # pandas no arrastra attrs a traves de la concatenacion y el filtrado, asi
    # que la marca de convencion se vuelve a poner a mano: es la que evita el
    # doble desplazamiento en el backtest. La huella, en cambio, se recalcula:
    # la de 'y' describe la serie sin alinear y aqui sobran las filas del
    # calentamiento, asi que copiarla tal cual acusaria de realineada a la
    # unica serie que con seguridad no lo esta.
    objetivo_alineado.attrs.update(y.attrs)
    _marcar_alineamiento(objetivo_alineado)

    return MatrizDatos(
        X=completo.loc[:, X.columns],
        y=objetivo_alineado,
        nombres_grupo=grupos,
    )


# --- auditoria de fuga de informacion ------------------------------------------


def _reparto(posiciones: Sequence[int], maximo: int) -> list[int]:
    """Adelgaza una lista de posiciones, dejandolas repartidas por todo el tramo."""
    lista = list(posiciones)
    if len(lista) <= maximo:
        return lista
    elegidas = np.linspace(0, len(lista) - 1, maximo)
    return sorted({lista[int(round(float(i)))] for i in elegidas})


def _posiciones_de_hueco(marco: pd.DataFrame | None) -> list[int]:
    """Posiciones con algun ausente, y la inmediatamente anterior a cada una.

    Un relleno hacia atras o una interpolacion solo mienten en las fechas con
    hueco: para cualquier otra fila los tres calculos (serie completa, truncada
    y perturbada) dan el mismo numero, porque el valor que se arrastra es
    anterior al corte. Cortar justo en el hueco y el dia antes pone el dato que
    se arrastra al otro lado de la frontera, que es lo unico que lo delata.
    """
    if marco is None:
        return []
    ausentes = np.flatnonzero(np.asarray(marco.isna().to_numpy()).any(axis=1))
    return sorted({int(p) for p in ausentes} | {int(p) - 1 for p in ausentes})


def _fechas_de_corte(indice: pd.Index, fechas, datos: pd.DataFrame | None = None) -> list:
    """Fechas en las que se parte la serie para auditar el constructor.

    Por omision se barre la segunda mitad de la muestra (que es donde hay
    calentamiento suficiente para que la matriz truncada tenga filas) y se
    aniaden los cortes pegados a los huecos de ``datos``, esten donde esten:
    esos son los unicos que ven una fuga condicionada a los ausentes.
    """
    if fechas is None:
        n = len(indice)
        if n < 4:
            raise ErrorDatos("La serie es demasiado corta para auditar la fuga de informacion.")
        candidatas = set(_reparto(range(n // 2, n - 1), _MAX_CORTES_POR_OMISION))
        candidatas.update(_reparto(_posiciones_de_hueco(datos), _MAX_CORTES_POR_OMISION))
        posiciones = sorted(p for p in candidatas if 0 < p < n - 1)
        if not posiciones:
            raise ErrorDatos("La serie es demasiado corta para auditar la fuga de informacion.")
        return [indice[p] for p in posiciones]

    convertidas = []
    for fecha in fechas:
        valor = pd.Timestamp(fecha) if isinstance(indice, pd.DatetimeIndex) else fecha
        if valor < indice[0] or valor >= indice[-1]:
            raise ErrorDatos(
                f"La fecha de corte {_texto(valor)} esta fuera del tramo auditable "
                f"[{_texto(indice[0])}, {_texto(indice[-1])}): hace falta que queden "
                "datos posteriores al corte para poder anadirlos."
            )
        convertidas.append(valor)
    if not convertidas:
        raise ErrorDatos("No se ha indicado ninguna fecha de corte.")
    return convertidas


def _texto(fecha) -> str:
    """Formatea una fecha para el mensaje de error, sin la hora si no la tiene."""
    if isinstance(fecha, pd.Timestamp) and fecha == fecha.normalize():
        return fecha.date().isoformat()
    return str(fecha)


def _perturbar_futuro(marco: pd.DataFrame, corte, semilla: int = 0) -> pd.DataFrame:
    """Reescribe los valores posteriores al corte dejando intacto lo anterior.

    La transformacion es ``valor * factor + desplazamiento`` con factor mayor
    que uno y desplazamiento positivo: cambia con seguridad cualquier valor,
    incluido el cero, y conserva el signo de una serie de precios.
    """
    posterior = np.asarray(marco.index > corte)
    generador = np.random.default_rng(semilla)
    forma = (int(posterior.sum()), marco.shape[1])
    factores = generador.uniform(1.2, 1.6, size=forma)
    desplazamientos = generador.uniform(1.0, 2.0, size=forma)
    copia = marco.copy()
    copia.iloc[posterior] = marco.to_numpy()[posterior] * factores + desplazamientos
    return copia


def _alterar_calendario_futuro(datos, corte):
    """Cambia **cuando** ocurre el futuro, no solo cuanto vale.

    Reescribir los valores deja el calendario intacto, asi que una variable que
    use la POSICION de las observaciones posteriores en lugar de sus valores
    (cuantos dias faltan para el siguiente dato, si maniana hay cotizacion,
    cuantas fechas quedan en la muestra) sale igual en las tres llamadas y
    atraviesa la auditoria sin dejar rastro. El experimento 1 tampoco la ve,
    porque la unica fila que la delataria, la del borde, desaparece al eliminar
    los NaN.

    Se hacen las dos cosas que un calendario puede hacer distinto: se elimina
    una de cada dos fechas posteriores al corte, empezando por la primera (que
    es la que fija "cuando vuelve a haber dato", y ademas cambia cuantas
    quedan), y se desplazan unos dias las que sobreviven. Lo anterior al corte
    no se toca: sus fechas siguen siendo las mismas, asi que una variable
    causal da exactamente el mismo numero y solo cambian las que miran el
    calendario de despues. El desplazamiento es hacia delante y uniforme, de
    modo que el indice sigue creciendo y sin repetidos, que es lo que exige
    cualquier constructor razonable.
    """
    indice = datos.index
    posteriores = np.flatnonzero(np.asarray(indice > corte))
    if posteriores.size == 0:
        return datos
    sobran = {int(p) for p in posteriores[::2]} if posteriores.size > 1 else set()
    salida = datos.iloc[[p for p in range(len(indice)) if p not in sobran]]
    conservado = salida.index
    if isinstance(conservado, pd.DatetimeIndex):
        # .where sobre el propio indice para no perder la zona horaria por el
        # camino: reconstruirlo desde numpy la dejaria en UTC y entonces ni una
        # sola fecha anterior al corte casaria con la matriz de referencia.
        desplazado = conservado.where(
            np.asarray(conservado <= corte),
            conservado + pd.Timedelta(days=_DIAS_CALENDARIO_ALTERADO),
        )
        salida = salida.set_axis(pd.DatetimeIndex(desplazado, name=conservado.name), axis=0)
    return salida


def _datos_de_kwargs(kwargs: dict) -> dict[str, pd.Series | pd.DataFrame]:
    """Separa los argumentos del constructor que son datos de los que son opciones.

    Solo los datos hay que truncar y perturbar. Un entero, una cadena o una
    secuencia de retardos no puede esconder informacion del futuro.
    """
    return {n: v for n, v in kwargs.items() if isinstance(v, (pd.Series, pd.DataFrame))}


def _exigir_auditable(datos, nombre: str, indice: pd.Index) -> None:
    """Rechaza un argumento de datos que la auditoria no sabe alterar.

    Un visto bueno que no significa nada es peor que un error: si un argumento
    llega identico a las tres llamadas, las columnas que salgan de el coinciden
    por construccion en las tres matrices y la comparacion no puede ver nada,
    ni siquiera un ``shift(-1)`` descarado.
    """
    try:
        bool(np.asarray(datos.index <= indice[-1]).any())
    except TypeError as exc:
        raise ErrorDatos(
            f"La auditoria no puede garantizar nada sobre el argumento '{nombre}': su "
            f"indice ({datos.index.dtype}) no se puede comparar con el de 'precios' "
            f"({indice.dtype}), asi que no hay forma de truncarlo en cada corte. "
            "Reindexelo al calendario de 'precios' antes de auditar."
        ) from exc
    try:
        _como_marco(datos, nombre, base=nombre)
    except ErrorDatos as exc:
        raise ErrorDatos(
            f"La auditoria no puede garantizar nada sobre el argumento '{nombre}' "
            f"porque no sabe reescribir sus valores: {exc} Pase solo datos numericos "
            "o incorporelos ya calculados a 'precios'."
        ) from exc


def _perturbar_datos(datos, nombre: str, corte, semilla: int):
    """Perturba un argumento de datos **conservando el tipo que se recibio**.

    Si la llamada de referencia recibe una Series y la perturbada un DataFrame,
    un constructor que bautice las columnas segun el tipo (vease
    :func:`_nombre_base`) devuelve nombres distintos en las dos matrices, y la
    comparacion se queda sin nada que mirar precisamente en las variables que
    salen del argumento sospechoso.
    """
    perturbado = _perturbar_futuro(
        _como_marco(datos, nombre, base=nombre), corte, semilla=semilla
    )
    if isinstance(datos, pd.Series):
        return perturbado.iloc[:, 0].rename(datos.name)
    return perturbado


def _comparar(
    referencia: MatrizDatos,
    alterada: MatrizDatos,
    corte,
    experimento: str,
) -> None:
    """Compara dos matrices hasta el corte y denuncia la primera discrepancia.

    Si los dos conjuntos de columnas no coinciden se aborta en vez de intersecar:
    una columna que se queda fuera de la comparacion no da ningun aviso, asi que
    el desajuste seria indistinguible de la ausencia de fuga, que es justo lo que
    la auditoria tiene que poder afirmar.
    """
    columnas = list(referencia.X.columns)
    faltan = [c for c in columnas if c not in set(alterada.X.columns)]
    sobran = [c for c in alterada.X.columns if c not in set(columnas)]
    if faltan or sobran:
        raise ErrorDatos(
            f"La auditoria no puede comparar la matriz del corte {_texto(corte)}: el "
            f"constructor devuelve columnas distintas segun los datos que reciba "
            f"(faltan {faltan}, aparecen {sobran}). Esas columnas se quedarian sin "
            "auditar y eso no se distingue de que no haya fuga: haga que el "
            "constructor emita siempre el mismo conjunto de nombres, sin que dependa "
            "de los datos que le lleguen."
        )
    fechas = referencia.X.index.intersection(alterada.X.index)
    fechas = fechas[fechas <= corte]
    if len(fechas) == 0:
        raise ErrorDatos(
            f"En el corte {_texto(corte)} no queda ninguna fila comparable. Use fechas "
            "de corte mas tardias, para que la matriz truncada tenga filas."
        )

    esperado = referencia.X.loc[fechas, columnas].to_numpy(dtype=float)
    obtenido = alterada.X.loc[fechas, columnas].to_numpy(dtype=float)
    distintos = ~np.isclose(
        esperado,
        obtenido,
        rtol=_TOLERANCIA_RELATIVA,
        atol=_TOLERANCIA_ABSOLUTA,
        equal_nan=True,
    )
    if not distintos.any():
        return

    fila = int(np.argmax(distintos.any(axis=1)))
    columna = int(np.argmax(distintos[fila]))
    raise ErrorDatos(
        f"Fuga de informacion del futuro en la variable '{columnas[columna]}': "
        f"en la fecha {_texto(fechas[fila])} vale {esperado[fila, columna]:.6g} con la "
        f"serie completa y {obtenido[fila, columna]:.6g} {experimento} "
        f"{_texto(corte)}. Una variable de esa fecha no puede cambiar porque cambien "
        "datos posteriores: revise si usa un shift negativo, una ventana centrada, un "
        "relleno hacia atras o una tipificacion con toda la muestra."
    )


def comprobar_ausencia_de_fuga(
    constructor: Callable[..., MatrizDatos],
    precios: pd.DataFrame,
    objetivo: str,
    fechas: Sequence | None = None,
    **kwargs,
) -> None:
    """Audita que ninguna variable de ``X`` use informacion posterior a su fecha.

    Es la funcion importante del modulo: convierte la promesa de causalidad en
    algo que se puede ejecutar en la bateria de pruebas, tanto sobre
    :func:`construir_matriz` como sobre cualquier constructor que escriba el
    usuario, con la firma ``(precios, objetivo, **kwargs) -> MatrizDatos``.

    Para cada fecha de corte se hacen tres experimentos:

    1. **Truncar.** Se reconstruye la matriz con la serie cortada en ``t`` y se
       comparan los valores. Detecta todo lo que dependa de la muestra
       completa: tipificaciones globales, minimos y maximos, ajustes hechos de
       una vez sobre todo el historico.
    2. **Reescribir el futuro.** Se deja la serie con su longitud pero se
       alteran los valores posteriores al corte. Detecta lo que la truncacion
       no puede ver, que es justo el borde: una variable que use ``shift(-1)``
       da los mismos valores en la serie truncada (porque su ultima fila, la
       unica que la delataria, desaparece al eliminar los NaN) y en cambio
       reacciona de inmediato a que el futuro cambie.
    3. **Reescribir el futuro y su calendario.** Lo mismo, pero moviendo
       ademas las fechas posteriores al corte (se elimina una de cada dos y se
       desplazan las demas). Detecta las variables que leen el calendario de
       despues en lugar de los valores: cuantos dias faltan para el siguiente
       dato, si maniana hay cotizacion, cuantas fechas quedan en la muestra.
       En una serie de energia con calendario irregular, "cuando vuelve a
       haber dato" no es informacion neutra.

    Los tres experimentos se aplican a ``precios`` y a **todos** los argumentos
    de ``kwargs`` que sean Series o DataFrame, no solo al que se llame
    ``exogenas``: un argumento que llegue igual a las tres llamadas produce por
    construccion las mismas columnas en las tres matrices, y entonces la
    comparacion no puede ver nada. Si alguno de esos argumentos no se puede
    truncar ni reescribir (indice incomparable con el de ``precios``, valores no
    numericos) se lanza :class:`~icm.comun.ErrorDatos` en lugar de dar un visto
    bueno que no significaria nada.

    Solo se audita ``X``. La variable objetivo mira al futuro por definicion:
    si no lo hiciera, no habria nada que predecir.

    Alcance
    -------
    La auditoria cubre lo que el constructor hace con los datos que recibe, y
    solo eso. Cualquier relleno, interpolacion o reindexacion aplicada **antes**
    de la llamada queda fuera: si se audita
    ``comprobar_ausencia_de_fuga(constructor, precios.interpolate(), ...)``, la
    fuga viaja ya cocinada dentro de ``precios`` y no hay eleccion de cortes que
    la encuentre, porque lo que se contrasta es el constructor y no su entrada.
    Interpole despues de auditar, o mejor, dentro del constructor, donde si se
    ve. Un "pasa" dice que el constructor es causal con los datos que recibio,
    no que los datos que recibio sean limpios.

    Parametros
    ----------
    fechas:
        Cortes a probar. Por omision se barre la segunda mitad de la muestra
        (hasta :data:`_MAX_CORTES_POR_OMISION` cortes repartidos, que es donde
        hay calentamiento suficiente para que la matriz truncada tenga filas) y
        se aniade un corte en cada fecha ausente de ``precios`` y en la
        anterior, porque una fuga por relleno hacia atras solo se delata ahi.

    Lanza
    -----
    ErrorDatos
        Si alguna variable cambia de valor en una fecha al cambiar lo que pasa
        despues, indicando la columna y la fecha concretas.
    """
    marco = _como_marco(precios, "precios", base="precio")
    _validar_indice_temporal(marco.index, "precios")
    datos_extra = _datos_de_kwargs(kwargs)
    for nombre, valor in datos_extra.items():
        _exigir_auditable(valor, nombre, marco.index)

    referencia = constructor(marco, objetivo, **kwargs)
    if not isinstance(referencia, MatrizDatos):
        raise ErrorDatos(
            "El constructor debe devolver un MatrizDatos con los atributos X, y y "
            f"nombres_grupo; ha devuelto {type(referencia).__name__}."
        )

    for corte in _fechas_de_corte(marco.index, fechas, marco):
        mascara = np.asarray(marco.index <= corte)

        truncados = {**kwargs}
        for nombre, valor in datos_extra.items():
            truncados[nombre] = valor.loc[np.asarray(valor.index <= corte)]
        try:
            alterada = constructor(marco.loc[mascara], objetivo, **truncados)
        except ErrorDatos as exc:
            raise ErrorDatos(
                f"Con la serie truncada en {_texto(corte)} el constructor ha fallado: "
                f"{exc} Elija fechas de corte mas tardias o una serie mas larga."
            ) from exc
        _comparar(referencia, alterada, corte, experimento="al truncar la serie en")

        perturbados = {**kwargs}
        for nombre, valor in datos_extra.items():
            perturbados[nombre] = _perturbar_datos(valor, nombre, corte, semilla=1)
        alterada = constructor(_perturbar_futuro(marco, corte), objetivo, **perturbados)
        _comparar(referencia, alterada, corte, experimento="al reescribir lo posterior a")

        # Misma perturbacion, pero moviendo tambien el calendario del futuro:
        # sin esto, una variable que lea cuando llega el siguiente dato en vez
        # de cuanto vale pasa las dos pruebas anteriores.
        recalendarizados = {**kwargs}
        for nombre, valor in datos_extra.items():
            recalendarizados[nombre] = _alterar_calendario_futuro(
                _perturbar_datos(valor, nombre, corte, semilla=2), corte
            )
        alterada = constructor(
            _alterar_calendario_futuro(_perturbar_futuro(marco, corte, semilla=2), corte),
            objetivo,
            **recalendarizados,
        )
        _comparar(
            referencia,
            alterada,
            corte,
            experimento="al cambiar los valores y las fechas posteriores a",
        )
