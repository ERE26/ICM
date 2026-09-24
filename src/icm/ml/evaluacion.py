"""Evaluacion de predicciones: si aciertan y si el acierto es significativo.

Tres ideas sostienen este modulo.

**Una prediccion sola no se puede juzgar.** Un error cuadratico medio de 0,012
no es bueno ni malo mientras no se sepa cuanto habria errado no hacer nada. La
referencia natural en series financieras es el paseo aleatorio, que sobre
rendimientos se traduce en "el rendimiento esperado es cero"; batirla de forma
sostenida es mucho mas dificil de lo que sugiere la literatura, y buena parte
de los trabajos que presumen de R cuadrado alto simplemente no la han
calculado.

**La metrica debe ser adimensional y estar anclada.** De ahi el MASE
(:func:`error_escalado_medio`): dice "comete un 20 % menos error que la
referencia", que se entiende sin saber en que unidades esta la serie ni cuanta
volatilidad tuvo el periodo. Un MAE de 3 EUR/MWh no se puede comparar con el de
otro mercado, ni siquiera con el del mismo mercado otro anio.

**La diferencia entre dos modelos hay que contrastarla.** "Mi modelo tiene 0,02
menos de RMSE" no significa nada por si solo: con 250 observaciones ruidosas,
diferencias de ese tamanio aparecen y desaparecen al cambiar el tramo de
prueba. Para eso esta :func:`contraste_diebold_mariano`.

Convencion de nombres
---------------------
El contraste se llama ``contraste_diebold_mariano`` y no ``test_...`` a
proposito: pytest recoge como prueba cualquier funcion importable que empiece
por ``test_``, y una funcion de libreria colada en la bateria de pruebas la
rompe con un error de fixtures desconcertante.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from icm.comun import ErrorDatos

#: Metodos admitidos por :func:`correlacion_informacion`.
METODOS_CORRELACION = ("spearman", "pearson", "kendall")

#: Nombre con el que se anade la prediccion ingenua a una tabla que no la trae.
NOMBRE_INGENUA = "ingenua"

#: Valores reconocidos en ``attrs["tipo_serie"]`` para desactivar la deteccion
#: automatica de :func:`prediccion_ingenua`.
TIPOS_DE_SERIE = ("rendimiento", "precio")


# --- utilidades internas -------------------------------------------------------


def _entero_positivo(valor: Any, nombre: str) -> int:
    """Valida un parametro entero de al menos 1."""
    if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)):
        raise ErrorDatos(f"'{nombre}' debe ser un numero entero, se ha recibido {valor!r}.")
    if valor < 1:
        raise ErrorDatos(f"'{nombre}' debe ser como minimo 1, se ha recibido {valor}.")
    return int(valor)


def _como_serie(datos: Any, nombre: str) -> pd.Series:
    """Convierte la entrada en una Series de float **sin** descartar ausentes.

    A diferencia de :func:`icm.comun.como_serie`, aqui los NaN se conservan:
    quitarlos de cada serie por separado desplazaria una respecto de la otra y
    acabaria comparando la prediccion de un dia con el valor real de otro. El
    descarte se hace mas tarde, y siempre por parejas.
    """
    if isinstance(datos, pd.Series):
        serie = datos.astype(float)
    elif isinstance(datos, pd.DataFrame):
        if datos.shape[1] != 1:
            raise ErrorDatos(
                f"'{nombre}' tiene {datos.shape[1]} columnas: pase una sola serie, "
                "por ejemplo con .iloc[:, 0] o seleccionando la columna por su nombre."
            )
        serie = datos.iloc[:, 0].astype(float)
    else:
        try:
            valores = np.asarray(datos, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ErrorDatos(
                f"'{nombre}' no se puede interpretar como una serie numerica: {exc}"
            ) from exc
        if valores.ndim != 1:
            raise ErrorDatos(
                f"'{nombre}' tiene {valores.ndim} dimensiones: se esperaba una serie de una sola."
            )
        serie = pd.Series(valores, name=nombre)
    if serie.empty:
        raise ErrorDatos(f"'{nombre}' no contiene ninguna observacion.")
    serie.attrs.update(getattr(datos, "attrs", {}))
    return serie


def _emparejar_varias(
    y: Any,
    predicciones: dict[str, Any],
    nombre_y: str = "y",
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.Index]:
    """Alinea el objetivo con varias predicciones sobre las MISMAS fechas.

    Dos exigencias y una limpieza:

    * misma longitud, porque el emparejamiento es posicional;
    * mismo indice cuando ambas entradas lo tienen, porque dos series de igual
      longitud pueden cubrir periodos distintos y nada lo delataria;
    * se descartan las fechas en las que falta (o es infinito) cualquiera de
      los valores implicados, de golpe para todas las series.

    Ese descarte conjunto es deliberado: si cada modelo se midiese sobre las
    fechas que a el le convienen, el que necesita mas historia para arrancar
    competiria en un periodo distinto del de los demas y la comparacion dejaria
    de ser una comparacion.
    """
    serie_y = _como_serie(y, nombre_y)
    if not predicciones:
        raise ErrorDatos("No se ha pasado ninguna prediccion que evaluar.")

    columnas: dict[str, np.ndarray] = {}
    for nombre, pred in predicciones.items():
        serie = _como_serie(pred, str(nombre))
        if len(serie) != len(serie_y):
            raise ErrorDatos(
                f"'{nombre_y}' tiene {len(serie_y)} observaciones y '{nombre}' tiene "
                f"{len(serie)}: recorte ambas al mismo tramo antes de evaluar, por ejemplo "
                "con .reindex(indice_comun), porque el emparejamiento es posicional y "
                "compararia fechas distintas."
            )
        if (
            isinstance(y, pd.Series)
            and isinstance(pred, pd.Series)
            and not serie.index.equals(serie_y.index)
        ):
            raise ErrorDatos(
                f"'{nombre_y}' y '{nombre}' tienen la misma longitud pero indices distintos: "
                "reindexelos al indice comun (y.index.intersection(prediccion.index)) antes "
                "de evaluar."
            )
        columnas[str(nombre)] = serie.to_numpy()

    valores_y = serie_y.to_numpy()
    validos = np.isfinite(valores_y)
    for valores in columnas.values():
        validos &= np.isfinite(valores)
    if not validos.any():
        raise ErrorDatos(
            "No queda ninguna fecha con dato real y prediccion a la vez: compruebe el "
            "periodo de calentamiento de las predicciones (una media movil de 21 sesiones "
            "no predice las 21 primeras)."
        )
    return valores_y[validos], {n: v[validos] for n, v in columnas.items()}, serie_y.index[validos]


def _emparejar(
    y: Any,
    pred: Any,
    nombre_y: str = "y",
    nombre_pred: str = "prediccion",
) -> tuple[np.ndarray, np.ndarray]:
    """Caso de dos series de :func:`_emparejar_varias`."""
    reales, columnas, _ = _emparejar_varias(y, {nombre_pred: pred}, nombre_y)
    return reales, columnas[nombre_pred]


def _es_serie_de_rendimientos(serie: pd.Series) -> bool:
    """Decide si la serie contiene rendimientos (True) o niveles (False).

    Por orden de preferencia:

    1. ``attrs["tipo_serie"]``, que el usuario pone a mano. Es la via explicita
       y la unica que no puede equivocarse.
    2. ``attrs["tipo_rendimiento"]``, la marca que deja
       :func:`icm.datos.precios.rendimientos`.
    3. Una regla de escala: son rendimientos las series centradas en cero
       (media menor que su desviacion tipica) cuyas magnitudes se mueven por
       debajo de la unidad. Un precio, aunque sea negativo como llega a serlo
       la electricidad, ni esta centrado en cero ni vive en el intervalo
       (-1, 1).
    """
    marca = serie.attrs.get("tipo_serie")
    if marca is not None:
        if marca not in TIPOS_DE_SERIE:
            raise ErrorDatos(
                f"attrs['tipo_serie'] vale '{marca}' y no se reconoce: "
                f"use uno de {', '.join(TIPOS_DE_SERIE)}."
            )
        return marca == "rendimiento"
    if serie.attrs.get("tipo_rendimiento") in {"log", "simple"}:
        return True

    valores = serie.to_numpy()
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        raise ErrorDatos("La serie no contiene ningun valor finito del que deducir su naturaleza.")
    centrada = abs(float(valores.mean())) < float(valores.std())
    escala_pequenia = float(np.quantile(np.abs(valores), 0.99)) < 1.0
    return centrada and escala_pequenia


# --- predicciones de referencia ------------------------------------------------


def prediccion_ingenua(y: pd.Series, horizonte: int = 1) -> pd.Series:
    """Prediccion del paseo aleatorio, la referencia que hay que batir.

    La sutileza que mas errores causa en la practica: el paseo aleatorio se
    formula sobre el **precio** (``p[t+h] = p[t] + ruido``). Sobre precios, por
    tanto, predice el ultimo precio observado; sobre **rendimientos** la misma
    hipotesis predice **cero**, porque el rendimiento esperado de un paseo
    aleatorio es nulo.

    Confundirlas cambia por completo la dificultad del ejercicio. Aplicar "el
    ultimo valor" a una serie de rendimientos construye una referencia
    artificialmente pesima, que arrastra el ruido de ayer y correlaciona
    negativamente con lo que va a pasar: cualquier modelo la bate y el MASE
    sale muy por debajo de 1 sin que haya capacidad predictiva ninguna. Al
    reves, comparar un modelo de precios contra "cero" es batir a un disparate
    y el MASE sale absurdamente bajo tambien.

    La naturaleza de la serie se deduce de ``y.attrs`` y, en su defecto, de la
    escala de los valores (vease :func:`_es_serie_de_rendimientos`). Para no
    depender de esa deduccion, marque la serie:

    >>> y.attrs["tipo_serie"] = "precio"   # o "rendimiento"  # doctest: +SKIP

    Las ``horizonte`` primeras fechas quedan a NaN tambien en el caso de los
    rendimientos, donde el cero no necesita historia: asi las dos ramas cubren
    exactamente las mismas fechas que cualquier otra prediccion con arranque, y
    ninguna referencia compite con observaciones de mas.
    """
    serie = _como_serie(y, "y")
    horizonte = _entero_positivo(horizonte, "horizonte")
    if horizonte >= len(serie):
        raise ErrorDatos(
            f"El horizonte ({horizonte}) alcanza o supera la longitud de la serie "
            f"({len(serie)}): no quedaria ninguna prediccion que evaluar."
        )

    if _es_serie_de_rendimientos(serie):
        prediccion = pd.Series(0.0, index=serie.index, name=NOMBRE_INGENUA)
        prediccion.iloc[:horizonte] = np.nan
    else:
        prediccion = serie.shift(horizonte).rename(NOMBRE_INGENUA)
    prediccion.attrs["horizonte"] = horizonte
    return prediccion


def prediccion_media_movil(y: pd.Series, ventana: int = 21) -> pd.Series:
    """Media de las ``ventana`` observaciones **anteriores** a cada fecha.

    El desplazamiento de un periodo es lo unico que separa una referencia
    honesta de una fuga de informacion: sin el, la media que "predice" el dia t
    incluye el propio dia t, y la referencia pasa a conocer lo que aun no ha
    ocurrido. Con series muy ruidosas el efecto es enorme, porque la media
    contaminada se pega al dato del dia.

    Como referencia tiene caracter distinto segun la serie: sobre rendimientos
    es una estimacion lenta de la deriva, dificil de batir cuando hay momento;
    sobre precios con tendencia va siempre por detras del nivel y suele ser
    **peor** que el paseo aleatorio, asi que no conviene usarla como unica
    referencia para declarar bueno un modelo.
    """
    serie = _como_serie(y, "y")
    ventana = _entero_positivo(ventana, "ventana")
    if ventana >= len(serie):
        raise ErrorDatos(
            f"La ventana ({ventana}) alcanza o supera la longitud de la serie "
            f"({len(serie)}): no quedaria ninguna prediccion que evaluar."
        )
    prediccion = serie.rolling(ventana).mean().shift(1)
    prediccion.name = f"media_movil_{ventana}"
    prediccion.attrs["ventana"] = ventana
    return prediccion


# --- metricas de error ---------------------------------------------------------


def _mae(reales: np.ndarray, prediccion: np.ndarray) -> float:
    return float(np.mean(np.abs(prediccion - reales)))


def _rmse(reales: np.ndarray, prediccion: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediccion - reales) ** 2)))


def error_absoluto_medio(y: pd.Series, pred: pd.Series) -> float:
    """Error absoluto medio (MAE), en las unidades de la serie.

    Es la metrica que menos castiga los fallos grandes, asi que describe mejor
    el error "de un dia cualquiera". Por eso mismo no sirve para decidir si un
    modelo es bueno: depende por completo de la escala y de la volatilidad del
    periodo. Para eso esta :func:`error_escalado_medio`.
    """
    reales, prediccion = _emparejar(y, pred)
    return _mae(reales, prediccion)


def raiz_error_cuadratico_medio(y: pd.Series, pred: pd.Series) -> float:
    """Raiz del error cuadratico medio (RMSE), en las unidades de la serie.

    Al elevar al cuadrado, unos pocos dias muy fallados pesan mas que muchos
    dias casi acertados; en gestion de riesgos eso suele ser lo que se quiere,
    porque los dias malos son los que rompen la cuenta. Siempre es mayor o
    igual que el MAE, y la distancia entre ambos mide lo desigual que es el
    reparto de los errores.
    """
    reales, prediccion = _emparejar(y, pred)
    return _rmse(reales, prediccion)


def error_escalado_medio(
    y: pd.Series,
    pred: pd.Series,
    referencia: pd.Series | None = None,
) -> float:
    """MASE: error absoluto del modelo dividido por el de la referencia.

    Es la metrica que de verdad informa, por dos motivos. Es adimensional, asi
    que 0,80 significa "comete un 20 % menos error absoluto que la referencia"
    tanto en EUR/MWh como en rendimientos logaritmicos. Y esta anclada a algo:
    por debajo de 1 el modelo bate a la referencia, por encima de 1 no, y el
    numero que separa ambos mundos no depende del mercado ni del anio.

    Si no se indica ``referencia`` se usa :func:`prediccion_ingenua` con
    horizonte 1. El numerador y el denominador se calculan sobre las **mismas**
    fechas; medir el error del modelo en un tramo y el de la referencia en otro
    es la forma mas comoda de fabricar un MASE favorable sin darse cuenta.

    Ojo: un MASE por debajo de 1 dice que el modelo gana en la muestra, no que
    la ventaja sea real. Eso lo responde :func:`contraste_diebold_mariano`.
    """
    if referencia is None:
        referencia = prediccion_ingenua(_como_serie(y, "y"))
    reales, columnas, _ = _emparejar_varias(y, {"prediccion": pred, "referencia": referencia})
    error_referencia = _mae(reales, columnas["referencia"])
    if error_referencia <= 0.0:
        raise ErrorDatos(
            "La referencia acierta sin ningun error y el MASE no esta definido "
            "(division por cero). Con datos reales esto solo pasa con series constantes: "
            "use otra referencia o compruebe que la serie no es degenerada."
        )
    return _mae(reales, columnas["prediccion"]) / error_referencia


def _acierto_direccional(reales: np.ndarray, prediccion: np.ndarray) -> float:
    utiles = (reales != 0.0) & (prediccion != 0.0)
    if not utiles.any():
        return float("nan")
    return float(np.mean(np.sign(prediccion[utiles]) == np.sign(reales[utiles])))


def acierto_direccional(y: pd.Series, pred: pd.Series) -> float:
    """Proporcion de fechas en las que se acierta el signo del movimiento.

    En trading el signo decide la posicion, asi que esta cifra se acerca mucho
    mas al dinero que el error cuadratico: un modelo con RMSE mediocre pero 55 %
    de acierto direccional gana dinero, y uno con RMSE excelente que se
    equivoca de lado en los dias grandes lo pierde.

    Se excluyen del computo las fechas sin direccion que acertar (el valor real
    es exactamente cero) y aquellas en las que el modelo no apuesta (predice
    exactamente cero). Excluir estas ultimas no es cosmetico: si contasen como
    fallo, una prediccion y su contraria fallarian las dos en la misma fecha y
    la medida dejaria de cumplir que invertir el signo da el complementario.
    Como efecto secundario, la prediccion ingenua sobre rendimientos, que es
    identicamente cero, no tiene acierto direccional definido: es exactamente
    lo que se quiere decir de quien no apuesta nunca.
    """
    reales, prediccion = _emparejar(y, pred)
    return _acierto_direccional(reales, prediccion)


def _correlacion_informacion(reales: np.ndarray, prediccion: np.ndarray, metodo: str) -> float:
    if metodo not in METODOS_CORRELACION:
        raise ErrorDatos(
            f"Metodo de correlacion '{metodo}' desconocido. "
            f"Disponibles: {', '.join(METODOS_CORRELACION)}."
        )
    # Con una serie constante la correlacion no esta definida: scipy avisa con
    # un RuntimeWarning y devuelve NaN por la puerta de atras. Es mas limpio
    # detectarlo antes y decir que no hay nada que correlacionar.
    if reales.size < 2 or np.unique(reales).size < 2 or np.unique(prediccion).size < 2:
        return float("nan")
    if metodo == "spearman":
        return float(stats.spearmanr(reales, prediccion)[0])
    if metodo == "pearson":
        return float(stats.pearsonr(reales, prediccion)[0])
    return float(stats.kendalltau(reales, prediccion)[0])


def correlacion_informacion(y: pd.Series, pred: pd.Series, metodo: str = "spearman") -> float:
    """Coeficiente de informacion (IC): correlacion entre previsto y ocurrido.

    Se usa Spearman por defecto porque en gestion lo que se explota es el
    **orden**: si el modelo ordena bien los dias (o los activos) de peor a
    mejor, la cartera se construye y gana, aunque la magnitud prevista este mal
    calibrada. Pearson, en cambio, se lo juega todo a la magnitud y un solo dia
    extremo puede inventarse o destruir la correlacion entera.

    Las cifras habituales sorprenden a quien viene de otros dominios: un IC
    entre 0,03 y 0,05 sostenido en el tiempo ya es una senal explotable en
    renta variable diaria. Un IC de 0,4 en datos financieros casi siempre
    significa que se ha colado informacion del futuro.

    Devuelve NaN si alguna de las dos series es constante: ahi no hay orden que
    comparar.
    """
    reales, prediccion = _emparejar(y, pred)
    return _correlacion_informacion(reales, prediccion, metodo)


# --- contraste de precision predictiva -----------------------------------------


def contraste_diebold_mariano(
    errores_a: pd.Series,
    errores_b: pd.Series,
    horizonte: int = 1,
    potencia: float = 2.0,
) -> dict:
    """Contrasta si dos predicciones tienen precision distinta (Diebold-Mariano).

    Por que hace falta: "mi modelo tiene 0,02 menos de RMSE" no significa
    absolutamente nada por si solo. Con unos cientos de observaciones ruidosas,
    diferencias de ese tamanio surgen del puro azar muestral, y la prueba esta
    en que cambian de signo al mover el tramo de prueba unos meses. El
    contraste pone un numero a esa duda: la hipotesis nula es que las dos
    predicciones tienen la misma precision esperada, y el p-valor es la
    probabilidad de observar una ventaja como la observada si en realidad no
    hubiera ninguna.

    La mecanica es sencilla: se construye la diferencia de perdidas
    ``d[t] = |e_a[t]|^potencia - |e_b[t]|^potencia`` y se contrasta que su
    media sea cero. Los dos refinamientos que trae la funcion no son adornos:

    * **Varianza de largo plazo.** Las predicciones a ``horizonte`` periodos se
      solapan, asi que sus errores estan autocorrelacionados hasta el retardo
      ``horizonte - 1``. Ignorarlo subestima la varianza y multiplica los
      falsos hallazgos. Se suman las autocovarianzas hasta ese retardo.
    * **Correccion de Harvey-Leybourne-Newbold (1997).** El estadistico es
      asintotico y con muestras pequenas rechaza demasiado; la correccion lo
      encoge y se usa la t de Student con ``n - 1`` grados de libertad en lugar
      de la normal.

    Parametros
    ----------
    errores_a, errores_b:
        Errores (real menos previsto, o al reves: solo se usa su valor
        absoluto) de las dos predicciones, sobre las mismas fechas.
    potencia:
        2 compara errores cuadraticos (el caso habitual, equivalente a comparar
        RMSE); 1 compara errores absolutos, mucho mas robusto si hay dias
        atipicos que dominan la suma de cuadrados.

    Devuelve
    --------
    Un diccionario con ``estadistico``, ``p_valor``, ``mejor`` ("a", "b" o
    "empate", segun quien tenga menor perdida media, con independencia de la
    significacion), ``diferencia_media`` (la media de ``d``, negativa si gana
    "a") y ``rechaza_al_5pct``.

    Notas
    -----
    Si la diferencia de perdidas no tiene ninguna variabilidad muestral (dos
    predicciones identicas, o una varianza de largo plazo que sale negativa por
    la suma de autocovarianzas sin ponderar), no hay nada que contrastar: la
    funcion devuelve estadistico 0 y p-valor 1 en lugar de dividir por cero.
    Es la respuesta prudente, la que nunca afirma una significacion que la
    muestra no sostiene.
    """
    horizonte = _entero_positivo(horizonte, "horizonte")
    potencia = float(potencia)
    if potencia <= 0.0:
        raise ErrorDatos(f"'potencia' debe ser positiva, se ha recibido {potencia}.")

    valores_a, valores_b = _emparejar(errores_a, errores_b, "errores_a", "errores_b")
    diferencia = np.abs(valores_a) ** potencia - np.abs(valores_b) ** potencia
    n = diferencia.size
    if n < 2:
        raise ErrorDatos(
            "Hacen falta al menos 2 fechas comparables para contrastar dos predicciones; "
            f"solo hay {n}."
        )

    media = float(diferencia.mean())
    centrado = diferencia - media
    varianza = float(np.mean(centrado**2))
    for retardo in range(1, min(horizonte, n)):
        varianza += 2.0 * float(np.mean(centrado[retardo:] * centrado[:-retardo]))

    factor = (n + 1.0 - 2.0 * horizonte + horizonte * (horizonte - 1.0) / n) / n
    if factor <= 0.0:
        raise ErrorDatos(
            f"La muestra ({n} fechas) es demasiado corta para un horizonte de {horizonte}: "
            "la correccion de Harvey-Leybourne-Newbold no esta definida. Alargue el tramo "
            "de prueba o contraste a un horizonte menor."
        )

    if varianza <= 0.0:
        estadistico, p_valor = 0.0, 1.0
    else:
        estadistico = media / float(np.sqrt(varianza / n)) * float(np.sqrt(factor))
        p_valor = float(2.0 * stats.t.sf(abs(estadistico), df=n - 1))

    if media < 0.0:
        mejor = "a"
    elif media > 0.0:
        mejor = "b"
    else:
        mejor = "empate"

    return {
        "estadistico": float(estadistico),
        "p_valor": p_valor,
        "mejor": mejor,
        "diferencia_media": media,
        "rechaza_al_5pct": bool(p_valor < 0.05),
    }


# --- tabla comparativa ---------------------------------------------------------


def tabla_evaluacion(
    y: pd.Series,
    predicciones: dict[str, pd.Series],
    referencia: str = NOMBRE_INGENUA,
) -> pd.DataFrame:
    """Compara varios modelos entre si y contra una referencia, en una tabla.

    Una fila por modelo con el MAE, el RMSE, el MASE, el acierto direccional,
    el coeficiente de informacion y el p-valor de Diebold-Mariano frente a la
    referencia, ordenada por MASE ascendente: el mejor arriba.

    Si ``referencia`` es ``"ingenua"`` y no esta entre las predicciones, se
    calcula con :func:`prediccion_ingenua` y se anade como una fila mas. Su
    fila sale siempre con MASE exactamente 1 y p-valor 1 (se contrasta contra
    si misma), lo que la convierte en la linea de flotacion visible de la
    tabla.

    Todos los modelos se miden sobre las mismas fechas: la interseccion de
    aquellas en las que hay dato real y todas las predicciones. Conviene
    saberlo, porque anadir a la comparacion un modelo con un calentamiento
    largo acorta el periodo de evaluacion de todos los demas.

    Como se lee: un modelo merece atencion si su MASE esta por debajo de 1
    **y** su p-valor por debajo de 0,05. Con lo primero solo, la ventaja puede
    ser ruido; con lo segundo solo, la diferencia es fiable pero puede ser
    fiablemente irrelevante.
    """
    if not isinstance(predicciones, dict):
        raise ErrorDatos(
            "'predicciones' debe ser un diccionario {nombre: serie} para poder etiquetar "
            "cada fila de la tabla."
        )
    if not predicciones:
        raise ErrorDatos("No se ha pasado ninguna prediccion que evaluar.")

    todas = dict(predicciones)
    if referencia not in todas:
        if referencia != NOMBRE_INGENUA:
            raise ErrorDatos(
                f"La referencia '{referencia}' no esta entre las predicciones "
                f"({', '.join(map(str, todas))}). Anadala al diccionario o deje la "
                f"referencia por defecto '{NOMBRE_INGENUA}', que se calcula sola."
            )
        todas[NOMBRE_INGENUA] = prediccion_ingenua(_como_serie(y, "y"))

    reales, columnas, indice = _emparejar_varias(y, todas)
    errores = {nombre: valores - reales for nombre, valores in columnas.items()}
    error_referencia = _mae(reales, columnas[referencia])
    if error_referencia <= 0.0:
        raise ErrorDatos(
            f"La referencia '{referencia}' no comete ningun error y el MASE no estaria "
            "definido: elija otra referencia."
        )

    filas = []
    for nombre, prediccion in columnas.items():
        contraste = contraste_diebold_mariano(errores[nombre], errores[referencia])
        filas.append(
            {
                "modelo": nombre,
                "mae": _mae(reales, prediccion),
                "rmse": _rmse(reales, prediccion),
                "mase": _mae(reales, prediccion) / error_referencia,
                "acierto_direccional": _acierto_direccional(reales, prediccion),
                "ic": _correlacion_informacion(reales, prediccion, "spearman"),
                "p_valor_dm": contraste["p_valor"],
                "n_observaciones": len(indice),
            }
        )
    tabla = pd.DataFrame(filas).set_index("modelo").sort_values("mase")
    tabla.attrs["referencia"] = referencia
    return tabla
