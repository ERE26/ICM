"""Graficos del flujo de aprendizaje automatico y de trading.

Modulo de presentacion, pero no decorativo: varios de estos graficos son la
unica forma de ver errores que los numeros esconden. Un R cuadrado alto no
distingue un modelo que anticipa de uno que copia el valor anterior; una tabla
de particiones no delata un solape de dos dias; un error medio pequenio no dice
si el modelo encoge sus predicciones hacia la media. Las figuras si.

Se reutilizan la paleta, el estilo y los ayudantes de :mod:`icm.graficos`: la
paleta esta validada para daltonismo y los colores se asignan **en orden fijo,
nunca ciclicamente**, de modo que la novena serie no reaparece con el color de
la primera. Si hacen falta mas series que colores, el modulo se niega y pide
repartirlas en varios paneles, que es lo que habria que hacer de todas formas.

Como en :mod:`icm.graficos`, matplotlib se importa dentro de cada funcion: el
resto de la libreria sigue siendo utilizable sin dependencias de dibujo.

Todas las funciones aceptan pandas y numpy corrientes (no objetos de
:mod:`icm.ml`) y devuelven la pareja ``(figura, ax)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos
from icm.graficos import (
    CHROME,
    ESTADO,
    PALETA,
    _crear_ejes,
    _formato_porcentaje,
    aplicar_estilo,
)

__all__ = [
    "acierto_por_magnitud",
    "aplicar_estilo",
    "curva_beneficio",
    "dispersion_prediccion",
    "distribucion_errores",
    "historial_entrenamiento",
    "importancia_variables",
    "particiones_walk_forward",
    "prediccion_vs_real",
]

#: Nombres admitidos para cada columna de la tabla de particiones. El primero
#: es el canonico; el segundo es el que produce
#: ``ParticionWalkForward.describir``, que es de donde suele salir la tabla.
_COLUMNAS_PARTICION = {
    "inicio_entrena": ("inicio_entrena", "inicio_entrenamiento"),
    "fin_entrena": ("fin_entrena", "fin_entrenamiento"),
    "inicio_prueba": ("inicio_prueba",),
    "fin_prueba": ("fin_prueba",),
}

#: Nombres admitidos para las dos curvas del historial de entrenamiento.
_COLUMNAS_HISTORIAL = {
    "entrenamiento": ("perdida_entrenamiento", "entrenamiento", "perdida_entrena", "entrena"),
    "validacion": ("perdida_validacion", "validacion", "valida"),
}


# --- utilidades internas -------------------------------------------------------


def _colores(n_series: int) -> list[str]:
    """Los ``n_series`` primeros colores de la paleta, sin repetir ninguno.

    La paleta no se recorre ciclicamente a proposito. Repetir color es peor que
    no dibujar: el lector cree estar comparando dos series distinguibles y no lo
    son. Con mas series que colores, la respuesta correcta no es otro color, son
    varios paneles.
    """
    if n_series < 1:
        raise ErrorDatos("No hay ninguna serie que dibujar: pase al menos una.")
    if n_series > len(PALETA):
        raise ErrorDatos(
            f"Se han pedido {n_series} series y la paleta tiene {len(PALETA)} colores. "
            "Reparta las series en varios paneles (o quedese con las principales): "
            "repetir color impide distinguirlas."
        )
    return list(PALETA[:n_series])


def _como_serie(datos, nombre: str, indice=None) -> pd.Series:
    """Convierte la entrada en ``Series`` de float conservando su indice.

    A diferencia de :func:`icm.comun.como_serie`, aqui **no** se descartan los
    ausentes por su cuenta: en un grafico temporal un hueco debe verse como un
    hueco, no cerrarse en falso uniendo los dos extremos.
    """
    if isinstance(datos, pd.Series):
        return datos.astype(float)
    if isinstance(datos, pd.DataFrame):
        if datos.shape[1] != 1:
            raise ErrorDatos(
                f"Se esperaba una unica columna en '{nombre}', "
                f"se han recibido {datos.shape[1]}: seleccione la columna a dibujar."
            )
        return datos.iloc[:, 0].astype(float)
    valores = np.asarray(datos, dtype=float)
    if valores.ndim != 1:
        raise ErrorDatos(f"'{nombre}' debe ser unidimensional, tiene {valores.ndim} dimensiones.")
    if indice is not None and len(indice) == len(valores):
        return pd.Series(valores, index=indice, name=nombre)
    return pd.Series(valores, name=nombre)


def _series_validas(series, nombre: str) -> dict[str, pd.Series]:
    """Normaliza un diccionario ``nombre -> serie`` y comprueba que no este vacio."""
    if not isinstance(series, dict):
        raise ErrorDatos(
            f"'{nombre}' debe ser un diccionario 'nombre del modelo' -> serie, "
            f"se ha recibido {type(series).__name__}."
        )
    if not series:
        raise ErrorDatos(f"'{nombre}' esta vacio: no hay nada que dibujar.")
    return {str(clave): _como_serie(valor, str(clave)) for clave, valor in series.items()}


def _par_alineado(reales, prediccion, minimo: int = 3) -> tuple[pd.Series, pd.Series]:
    """Empareja valor real y prediccion descartando los pares incompletos.

    Si ambos llegan con indice se cruzan por fecha, que es la unica forma de no
    comparar el martes con el miercoles cuando una de las dos series tiene un
    hueco. Si no, se exige la misma longitud y se emparejan por posicion.
    """
    serie_real = _como_serie(reales, "reales")
    serie_pred = _como_serie(prediccion, "prediccion")

    if isinstance(reales, (pd.Series, pd.DataFrame)) and isinstance(
        prediccion, (pd.Series, pd.DataFrame)
    ):
        comun = serie_real.index.intersection(serie_pred.index)
        if len(comun) == 0:
            raise ErrorDatos(
                "'reales' y 'prediccion' no comparten ninguna fecha: compruebe que las dos "
                "series vienen del mismo tramo de prueba antes de compararlas."
            )
        serie_real, serie_pred = serie_real.loc[comun], serie_pred.loc[comun]
    else:
        if len(serie_real) != len(serie_pred):
            raise ErrorDatos(
                f"'reales' tiene {len(serie_real)} observaciones y 'prediccion' "
                f"{len(serie_pred)}: recorte la mas larga o pase ambas como Series con "
                "indice para cruzarlas por fecha."
            )
        serie_pred = pd.Series(serie_pred.to_numpy(), index=serie_real.index, name="prediccion")

    validos = serie_real.notna() & serie_pred.notna()
    serie_real, serie_pred = serie_real[validos], serie_pred[validos]
    if len(serie_real) < minimo:
        raise ErrorDatos(
            f"Solo quedan {len(serie_real)} pares validos y hacen falta al menos {minimo}: "
            "amplie el tramo de prueba o revise los valores ausentes."
        )
    return serie_real, serie_pred


def _etiqueta_eje_x(indice: pd.Index) -> str:
    """'Fecha' cuando el indice son fechas; si no, 'Observacion'."""
    return "Fecha" if isinstance(indice, pd.DatetimeIndex) else "Observacion"


def _a_numeros(columna: pd.Series, nombre: str) -> tuple[np.ndarray, bool]:
    """Pasa una columna de fechas o de numeros a coordenadas del eje.

    Devuelve tambien si eran fechas, porque en ese caso hay que decirselo al eje
    para que ponga anios y meses en lugar de numeros de serie.
    """
    if pd.api.types.is_numeric_dtype(columna):
        return columna.to_numpy(dtype=float), False

    from matplotlib.dates import date2num

    fechas = columna if pd.api.types.is_datetime64_any_dtype(columna) else None
    if fechas is None:
        try:
            fechas = pd.to_datetime(columna)
        except (ValueError, TypeError) as exc:
            raise ErrorDatos(
                f"La columna '{nombre}' no contiene ni fechas ni numeros ({exc}): "
                "conviertala con pd.to_datetime antes de dibujar."
            ) from exc
    return np.asarray(date2num(fechas.to_numpy())), True


def _leyenda(ax, n_series: int, **opciones) -> None:
    """Pone leyenda solo si hay mas de una serie.

    Con una sola serie la leyenda repite lo que ya dice el titulo o la etiqueta
    del eje, y gasta espacio y atencion en no decir nada.
    """
    if n_series > 1:
        ax.legend(**opciones)


# --- particiones y prediccion --------------------------------------------------


def particiones_walk_forward(descripcion: pd.DataFrame, ax=None, figsize=(10, 4)):
    """Dibuja cada particion como dos barras: entrenamiento y prueba.

    Es el grafico que permite comprobar de un vistazo que la validacion avanza
    hacia delante y que no hay viaje al pasado. Vale mas que cualquier
    afirmacion en un informe: el solape entre el fin del entrenamiento y el
    inicio de la prueba es de las pocas cosas que la vista detecta al instante y
    que ninguna metrica delata, porque un modelo que ha visto el futuro no falla
    ninguna prueba, solo saca mejor nota.

    Parametros
    ----------
    descripcion:
        Una fila por particion, con las columnas ``inicio_entrena``,
        ``fin_entrena``, ``inicio_prueba`` y ``fin_prueba``. Se aceptan tambien
        los nombres largos ``inicio_entrenamiento`` y ``fin_entrenamiento`` que
        produce ``ParticionWalkForward.describir``. Los extremos se entienden
        inclusivos y pueden ser fechas o numeros de observacion.

    Las particiones cuya prueba empieza antes de acabar el entrenamiento se
    marcan en el color de estado critico: ahi hay fuga de informacion.
    """
    if not isinstance(descripcion, pd.DataFrame):
        raise ErrorDatos(
            "'descripcion' debe ser un DataFrame con una fila por particion, "
            f"se ha recibido {type(descripcion).__name__}."
        )
    if descripcion.empty:
        raise ErrorDatos("'descripcion' no tiene ninguna particion que dibujar.")

    tramos: dict[str, np.ndarray] = {}
    es_fecha = False
    for canonico, alternativos in _COLUMNAS_PARTICION.items():
        presente = next((nombre for nombre in alternativos if nombre in descripcion.columns), None)
        if presente is None:
            raise ErrorDatos(
                f"Falta la columna '{canonico}' en 'descripcion' "
                f"(nombres admitidos: {', '.join(alternativos)}). Use el resultado de "
                "ParticionWalkForward.describir(X) o construya la tabla con esas columnas."
            )
        tramos[canonico], columna_es_fecha = _a_numeros(descripcion[presente], presente)
        es_fecha = es_fecha or columna_es_fecha

    figura, ax = _crear_ejes(ax, figsize)

    extension = float(
        np.nanmax(tramos["fin_prueba"]) - np.nanmin(tramos["inicio_entrena"])
    )
    # Un tramo de una sola observacion tiene anchura cero y se volveria
    # invisible: se le da un grosor minimo para que siga contandose a ojo.
    minimo = max(extension * 0.004, 1e-9)
    color_entrena, color_prueba = _colores(2)
    altura = 0.34

    filas = np.arange(len(descripcion))
    bandas = [
        ("inicio_entrena", "fin_entrena", color_entrena, "Entrenamiento", +0.5),
        ("inicio_prueba", "fin_prueba", color_prueba, "Prueba", -0.5),
    ]
    barras = {}
    for clave_inicio, clave_fin, color, etiqueta, desplazamiento in bandas:
        anchuras = np.maximum(tramos[clave_fin] - tramos[clave_inicio], minimo)
        barras[etiqueta] = ax.barh(
            filas + desplazamiento * (altura + 0.04),
            anchuras,
            height=altura,
            left=tramos[clave_inicio],
            color=color,
            label=etiqueta,
            zorder=3,
        )

    solapes = tramos["inicio_prueba"] <= tramos["fin_entrena"]
    for fila in filas[solapes]:
        for contenedor in barras.values():
            contenedor[fila].set_edgecolor(ESTADO["critico"])
            contenedor[fila].set_linewidth(2.0)
        ax.annotate(
            "Solape",
            (tramos["inicio_prueba"][fila], fila - 0.44),
            textcoords="offset points",
            xytext=(3, 0),
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color=ESTADO["critico"],
            zorder=5,
        )

    ax.set_yticks(filas)
    ax.set_yticklabels([str(etiqueta) for etiqueta in descripcion.index])
    ax.invert_yaxis()
    ax.set_xlabel("Fecha" if es_fecha else "Observacion")
    ax.set_ylabel("Particion")
    if es_fecha:
        # Se rotan solo las marcas de estos ejes, no las de toda la figura:
        # el grafico puede ser un panel de una rejilla ajena.
        ax.xaxis_date()
        ax.tick_params(axis="x", rotation=25)
    # La leyenda va fuera del area de dibujo: dentro taparia siempre la ultima
    # particion, que es justo la que se mira primero.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    ax.grid(axis="y", visible=False)
    return figura, ax


def prediccion_vs_real(
    reales,
    predicciones: dict[str, pd.Series],
    ax=None,
    figsize=(11, 4.5),
):
    """Superpone el valor observado y lo que predijo cada modelo.

    El valor real va en gris y por detras: es el fondo contra el que se juzgan
    las predicciones, no una serie mas a comparar. Dibujado asi se ve enseguida
    el fallo tipico de las series financieras, una prediccion que reproduce la
    serie real con un periodo de retraso, es decir, un modelo que no anticipa
    nada y solo repite el ultimo valor.
    """
    series = _series_validas(predicciones, "predicciones")
    serie_real = _como_serie(reales, "reales")
    colores = _colores(len(series))

    figura, ax = _crear_ejes(ax, figsize)
    ax.plot(
        serie_real.index,
        serie_real.to_numpy(),
        color=CHROME["tinta_apagada"],
        linewidth=1.2,
        label="Real",
        zorder=2,
    )
    for color, (nombre, serie) in zip(colores, series.items()):
        ax.plot(serie.index, serie.to_numpy(), color=color, linewidth=1.6, label=nombre, zorder=3)

    ax.set_xlabel(_etiqueta_eje_x(serie_real.index))
    ax.set_ylabel("Valor")
    ax.legend(loc="upper left", ncols=min(len(series) + 1, 4))
    ax.grid(axis="x", visible=False)
    return figura, ax


def dispersion_prediccion(reales, prediccion, ax=None, figsize=(6, 6)):
    """Nube de puntos con la diagonal y la recta ajustada, y sus dos numeros.

    La recta se ajusta por minimos cuadrados de la **prediccion sobre el valor
    real**, de modo que su pendiente es directamente el factor de encogimiento
    del modelo. Una pendiente muy por debajo de 1 describe un modelo que empuja
    sus predicciones hacia la media: acierta la direccion pero infravalora la
    magnitud. No es un defecto menor ni cosmetico, porque cambia como hay que
    dimensionar la posicion: con predicciones encogidas, una regla que invierta
    proporcionalmente a la prediccion se queda sistematicamente corta.

    Los dos ejes comparten escala y la figura es cuadrada, sin lo cual la
    diagonal no se veria a 45 grados y la comparacion visual enganiaria.
    """
    serie_real, serie_pred = _par_alineado(reales, prediccion)
    x = serie_real.to_numpy()
    y = serie_pred.to_numpy()

    if np.isclose(x.std(), 0.0):
        raise ErrorDatos(
            "El valor real es constante en el tramo: no hay dispersion que explicar "
            "ni pendiente que estimar. Amplie el tramo de prueba."
        )
    if np.isclose(y.std(), 0.0):
        raise ErrorDatos(
            "La prediccion es constante: el modelo no discrimina y la correlacion no esta "
            "definida. Revise el ajuste antes de dibujar la dispersion."
        )

    correlacion = float(np.corrcoef(x, y)[0, 1])
    pendiente = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    ordenada = float(y.mean() - pendiente * x.mean())

    figura, ax = _crear_ejes(ax, figsize)
    ax.scatter(
        x,
        y,
        s=22,
        color=PALETA[0],
        alpha=0.55,
        edgecolors="none",
        zorder=3,
        label="Observaciones",
    )

    minimo = float(min(x.min(), y.min()))
    maximo = float(max(x.max(), y.max()))
    margen = (maximo - minimo) * 0.06 or 1e-6
    limites = (minimo - margen, maximo + margen)
    recta = np.array(limites)

    ax.plot(
        recta,
        recta,
        color=CHROME["tinta_apagada"],
        linewidth=1.4,
        linestyle="--",
        zorder=2,
        label="Prediccion perfecta",
    )
    ax.plot(
        recta,
        ordenada + pendiente * recta,
        color=PALETA[1],
        linewidth=2.0,
        zorder=4,
        label="Ajuste por minimos cuadrados",
    )

    ax.annotate(
        f"correlacion {correlacion:.2f}\npendiente {pendiente:.2f}",
        (0.04, 0.96),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        fontweight="bold",
        color=CHROME["tinta"],
        zorder=6,
    )

    ax.set_xlim(limites)
    ax.set_ylim(limites)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Prediccion")
    ax.legend(loc="lower right")
    return figura, ax


def importancia_variables(importancia: pd.Series, n: int = 20, ax=None, figsize=(8, 6)):
    """Las ``n`` variables mas importantes, en barras horizontales ordenadas.

    Se ordena por valor absoluto y se dibuja el valor con su signo: en un modelo
    lineal, un coeficiente muy negativo es tan informativo como uno muy
    positivo, y ordenar por el valor con signo esconderia justo las variables
    que mas pesan hacia abajo.
    """
    serie = _como_serie(importancia, "importancia").dropna()
    if serie.empty:
        raise ErrorDatos("'importancia' no contiene ningun valor valido que dibujar.")
    if int(n) < 1:
        raise ErrorDatos(f"'n' debe ser al menos 1, se ha recibido {n}.")

    seleccion = serie.reindex(serie.abs().sort_values(ascending=False).index).head(int(n))
    # Se invierte para que la mas importante quede arriba al dibujar hacia
    # arriba desde el origen del eje vertical.
    seleccion = seleccion.iloc[::-1]

    hay_negativos = bool((seleccion < 0).any())
    hay_positivos = bool((seleccion >= 0).any())
    color_positivo, color_negativo = _colores(2)
    colores = [color_positivo if valor >= 0 else color_negativo for valor in seleccion]

    figura, ax = _crear_ejes(ax, figsize)
    posiciones = np.arange(len(seleccion))
    ax.barh(posiciones, seleccion.to_numpy(), height=0.7, color=colores, zorder=3)

    if hay_negativos:
        ax.axvline(0.0, color=CHROME["eje"], linewidth=1.0, zorder=2)
        if hay_positivos:
            from matplotlib.patches import Patch

            ax.legend(
                handles=[
                    Patch(color=color_positivo, label="Aporta en positivo"),
                    Patch(color=color_negativo, label="Aporta en negativo"),
                ],
                loc="lower right",
            )

    ax.set_yticks(posiciones)
    ax.set_yticklabels([str(etiqueta) for etiqueta in seleccion.index])
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    ax.grid(axis="y", visible=False)
    return figura, ax


# --- resultados de la estrategia -----------------------------------------------


def curva_beneficio(curvas: dict[str, pd.Series], ax=None, figsize=(11, 4.5)):
    """Beneficio acumulado de cada estrategia, con el cero marcado.

    La linea del cero no es decoracion: sin ella, un eje que empieza en 0,02
    hace pasar por estrategia rentable lo que apenas se separa de no hacer nada.
    """
    series = _series_validas(curvas, "curvas")
    colores = _colores(len(series))

    figura, ax = _crear_ejes(ax, figsize)
    for color, (nombre, serie) in zip(colores, series.items()):
        ax.plot(serie.index, serie.to_numpy(), color=color, label=nombre, zorder=3)
    ax.axhline(0.0, color=CHROME["tinta_apagada"], linewidth=1.2, linestyle="--", zorder=1)

    primera = next(iter(series.values()))
    ax.set_xlabel(_etiqueta_eje_x(primera.index))
    ax.set_ylabel("Beneficio acumulado")
    _leyenda(ax, len(series), loc="upper left", ncols=min(len(series), 4))
    ax.grid(axis="x", visible=False)
    return figura, ax


def distribucion_errores(errores: dict[str, pd.Series], ax=None, figsize=(9, 4.5)):
    """Histograma de los errores de prediccion de cada modelo.

    Todos los modelos comparten los mismos intervalos, calculados sobre el
    conjunto de los errores: con intervalos distintos, dos histogramas del mismo
    grafico no se pueden comparar por mucho que lo parezca. Con varios modelos
    se dibuja el contorno en lugar del relleno, porque tres rellenos
    translucidos superpuestos no dejan leer ninguno.

    El cero marcado es lo que revela el sesgo: una distribucion estrecha pero
    desplazada es un modelo que se equivoca poco y siempre en la misma
    direccion, y eso se corrige restando una constante, no cambiando de modelo.
    """
    series = _series_validas(errores, "errores")
    limpias = {nombre: serie.dropna() for nombre, serie in series.items()}
    vacias = [nombre for nombre, serie in limpias.items() if serie.empty]
    if vacias:
        raise ErrorDatos(
            f"Las series de error {vacias} no tienen ningun valor valido: "
            "elimine esas entradas o revise el calculo del error."
        )
    colores = _colores(len(limpias))

    todos = np.concatenate([serie.to_numpy() for serie in limpias.values()])
    intervalos = np.histogram_bin_edges(todos, bins=min(40, max(10, int(np.sqrt(todos.size)))))

    figura, ax = _crear_ejes(ax, figsize)
    varias = len(limpias) > 1
    for color, (nombre, serie) in zip(colores, limpias.items()):
        ax.hist(
            serie.to_numpy(),
            bins=intervalos,
            color=color,
            label=nombre,
            histtype="step" if varias else "bar",
            linewidth=1.8 if varias else 0.0,
            alpha=1.0 if varias else 0.75,
            zorder=3,
        )
    ax.axvline(0.0, color=CHROME["tinta"], linewidth=1.2, linestyle="--", zorder=4)

    ax.set_xlabel("Error de prediccion (real - previsto)")
    ax.set_ylabel("Frecuencia")
    _leyenda(ax, len(limpias), loc="upper right")
    ax.grid(axis="x", visible=False)
    return figura, ax


def acierto_por_magnitud(
    reales,
    prediccion,
    n_grupos: int = 5,
    ax=None,
    figsize=(8, 4.5),
):
    """Acierto direccional por grupo de magnitud de la prediccion.

    Es el grafico que decide si una estrategia tiene sentido. Si el acierto
    crece con la magnitud, el modelo sabe cuando sabe y conviene operar solo por
    encima de un umbral. Si es plano, la confianza del modelo no informa de
    nada: filtrar por umbral no mejora el acierto y solo reduce el numero de
    operaciones sobre las que repartir el coste fijo. Un acierto global del 55 %
    puede ser cualquiera de los dos casos, y son estrategias opuestas.

    Los grupos se forman partiendo las observaciones ordenadas por
    ``|prediccion|`` en ``n_grupos`` tramos de tamanio casi igual. Se hace asi, y
    no con cuantiles, porque con predicciones repetidas los cuantiles colapsan y
    devuelven menos grupos de los pedidos sin avisar.

    Un cero en la prediccion o en el valor real cuenta como fallo: no hay
    direccion que acertar.
    """
    n_grupos = int(n_grupos)
    if n_grupos < 2:
        raise ErrorDatos(f"'n_grupos' debe ser al menos 2, se ha recibido {n_grupos}.")
    serie_real, serie_pred = _par_alineado(reales, prediccion, minimo=n_grupos)

    orden = np.argsort(np.abs(serie_pred.to_numpy()), kind="stable")
    aciertos = (serie_real.to_numpy() * serie_pred.to_numpy() > 0.0).astype(float)
    magnitudes = np.abs(serie_pred.to_numpy())

    grupos = np.array_split(orden, n_grupos)
    tasas = np.array([aciertos[posiciones].mean() for posiciones in grupos])
    cortes = [magnitudes[posiciones] for posiciones in grupos]

    figura, ax = _crear_ejes(ax, figsize)
    posiciones = np.arange(n_grupos)
    ax.bar(posiciones, tasas, width=0.68, color=PALETA[0], zorder=3)
    ax.axhline(0.5, color=CHROME["tinta_apagada"], linewidth=1.3, linestyle="--", zorder=4)
    ax.annotate(
        "Azar",
        (-0.58, 0.5),
        textcoords="offset points",
        xytext=(2, 4),
        ha="left",
        fontsize=8,
        color=CHROME["tinta_apagada"],
    )

    for posicion, tasa in zip(posiciones, tasas):
        ax.text(
            posicion,
            tasa,
            f"{tasa:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=CHROME["tinta_secundaria"],
        )

    ax.set_xticks(posiciones)
    ax.set_xticklabels(
        [f"G{numero + 1}\n|p|>={tramo.min():.3g}" for numero, tramo in enumerate(cortes)]
    )
    ax.set_xlim(-0.6, n_grupos - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Grupo por magnitud de la prediccion (G1 = las mas pequenias)")
    ax.set_ylabel("Acierto direccional")
    _formato_porcentaje(ax)
    ax.grid(axis="x", visible=False)
    return figura, ax


def historial_entrenamiento(historial: pd.DataFrame, ax=None, figsize=(8, 4.5)):
    """Perdida de entrenamiento y de validacion por epoca.

    Mientras las dos curvas bajan juntas, el modelo aprende. Cuando la de
    entrenamiento sigue bajando y la de validacion se da la vuelta, ha dejado de
    aprender y ha empezado a memorizar: a partir de ese punto cada epoca empeora
    el modelo en produccion aunque la perdida que se imprime por pantalla siga
    mejorando. El minimo de la curva de validacion se marca porque es la epoca
    que habria que conservar.

    Parametros
    ----------
    historial:
        Una fila por epoca. Se admiten las columnas ``perdida_entrenamiento`` y
        ``perdida_validacion`` o, mas breves, ``entrenamiento`` y
        ``validacion``. La de validacion es opcional.
    """
    if not isinstance(historial, pd.DataFrame):
        raise ErrorDatos(
            "'historial' debe ser un DataFrame con una fila por epoca, "
            f"se ha recibido {type(historial).__name__}."
        )
    if historial.empty:
        raise ErrorDatos("'historial' no tiene ninguna epoca que dibujar.")

    columnas: dict[str, str | None] = {}
    for canonico, alternativos in _COLUMNAS_HISTORIAL.items():
        columnas[canonico] = next(
            (nombre for nombre in alternativos if nombre in historial.columns), None
        )
    if columnas["entrenamiento"] is None:
        admitidos = ", ".join(_COLUMNAS_HISTORIAL["entrenamiento"])
        raise ErrorDatos(
            f"Falta la perdida de entrenamiento en 'historial' (nombres admitidos: "
            f"{admitidos}). Registre la perdida de cada epoca en una de esas columnas."
        )

    epocas = historial.index.to_numpy()
    entrenamiento = historial[columnas["entrenamiento"]].astype(float)
    color_entrena, color_valida = _colores(2)

    figura, ax = _crear_ejes(ax, figsize)
    ax.plot(
        epocas, entrenamiento.to_numpy(), color=color_entrena, label="Entrenamiento", zorder=3
    )

    n_series = 1
    if columnas["validacion"] is not None:
        validacion = historial[columnas["validacion"]].astype(float)
        ax.plot(epocas, validacion.to_numpy(), color=color_valida, label="Validacion", zorder=3)
        n_series = 2
        if validacion.notna().any():
            posicion = int(np.nanargmin(validacion.to_numpy()))
            ax.axvline(
                epocas[posicion],
                color=CHROME["tinta_apagada"],
                linewidth=1.2,
                linestyle=":",
                zorder=2,
            )
            ax.annotate(
                f"Mejor epoca: {epocas[posicion]}",
                (epocas[posicion], float(validacion.iloc[posicion])),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold",
                color=CHROME["tinta"],
                zorder=5,
            )

    ax.set_xlabel("Epoca")
    ax.set_ylabel("Perdida")
    _leyenda(ax, n_series, loc="upper right")
    ax.grid(axis="x", visible=False)
    return figura, ax
