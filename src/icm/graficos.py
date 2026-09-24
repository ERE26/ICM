"""Graficos para los notebooks de analisis.

Modulo opcional: requiere matplotlib, que se instala con el extra ``graficos``.
El resto de la libreria no lo importa, de modo que ``icm`` sigue siendo
utilizable sin dependencias de dibujo.

La paleta esta fijada y validada para daltonismo: los colores se asignan
siempre en el mismo orden, nunca ciclicamente, para que un activo conserve su
color aunque cambie el numero de series del grafico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Paleta categorica, en orden fijo. Las tres primeras posiciones son las
#: unicas seguras cuando todas las series pueden compararse entre si a la vez
#: (nubes de puntos); a partir de la cuarta conviene separar en varios paneles.
PALETA = [
    "#2a78d6",  # azul
    "#eb6834",  # naranja
    "#1baf7a",  # aguamarina
    "#eda100",  # amarillo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#4a3aa7",  # violeta
    "#e34948",  # rojo
]

#: Colores de estado, reservados: nunca se usan como una serie mas.
ESTADO = {
    "bien": "#0ca30c",
    "aviso": "#fab219",
    "grave": "#ec835a",
    "critico": "#d03b3b",
}

#: Tinta y superficies del grafico.
CHROME = {
    "superficie": "#fcfcfb",
    "tinta": "#0b0b0b",
    "tinta_secundaria": "#52514e",
    "tinta_apagada": "#898781",
    "rejilla": "#e1e0d9",
    "eje": "#c3c2b7",
}


def aplicar_estilo() -> None:
    """Fija los parametros de matplotlib: rejilla discreta, ejes limpios."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": CHROME["superficie"],
            "axes.facecolor": CHROME["superficie"],
            "axes.edgecolor": CHROME["eje"],
            "axes.labelcolor": CHROME["tinta_secundaria"],
            "axes.titlecolor": CHROME["tinta"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": CHROME["rejilla"],
            "grid.linewidth": 0.8,
            "xtick.color": CHROME["tinta_apagada"],
            "ytick.color": CHROME["tinta_apagada"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "figure.dpi": 110,
            "font.size": 10,
        }
    )


def _crear_ejes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _formato_porcentaje(ax, eje: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter

    formateador = FuncFormatter(lambda valor, _: f"{valor:.0%}")
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(formateador)


def frontera_eficiente(
    frontera: pd.DataFrame,
    carteras: dict[str, tuple[float, float]] | None = None,
    activos: dict[str, tuple[float, float]] | None = None,
    recta=None,
    ax=None,
    figsize=(8, 5.5),
):
    """Dibuja la frontera eficiente y, opcionalmente, carteras y activos sueltos.

    ``carteras`` y ``activos`` son diccionarios ``nombre -> (volatilidad,
    rentabilidad)``. Las carteras destacadas se etiquetan directamente sobre el
    grafico, de modo que la identidad no depende solo del color.
    """
    figura, ax = _crear_ejes(ax, figsize)

    ax.plot(
        frontera["volatilidad"],
        frontera["rentabilidad"],
        color=PALETA[0],
        linewidth=2.0,
        label="Frontera eficiente",
        zorder=3,
    )

    if recta is not None:
        ax.plot(
            recta["volatilidad"],
            recta["rentabilidad"],
            color=CHROME["tinta_apagada"],
            linewidth=1.5,
            linestyle="--",
            label="Linea del mercado de capitales",
            zorder=2,
        )

    if activos:
        ax.scatter(
            [v for v, _ in activos.values()],
            [r for _, r in activos.values()],
            s=45,
            color=CHROME["tinta_apagada"],
            edgecolors=CHROME["superficie"],
            linewidths=2,
            label="Activos individuales",
            zorder=4,
        )
        for nombre, (volatilidad, rentabilidad) in activos.items():
            ax.annotate(
                nombre,
                (volatilidad, rentabilidad),
                textcoords="offset points",
                xytext=(8, -3),
                fontsize=8,
                color=CHROME["tinta_apagada"],
            )

    if carteras:
        for indice, (nombre, (volatilidad, rentabilidad)) in enumerate(carteras.items()):
            color = PALETA[(indice + 1) % 3]
            ax.scatter(
                [volatilidad],
                [rentabilidad],
                s=130,
                color=color,
                edgecolors=CHROME["superficie"],
                linewidths=2,
                zorder=5,
            )
            ax.annotate(
                nombre,
                (volatilidad, rentabilidad),
                textcoords="offset points",
                xytext=(10, 6),
                fontsize=9,
                fontweight="bold",
                color=CHROME["tinta"],
            )

    ax.set_xlabel("Volatilidad anualizada")
    ax.set_ylabel("Rentabilidad esperada anual")
    _formato_porcentaje(ax, "x")
    _formato_porcentaje(ax, "y")
    ax.legend(loc="lower right")
    return figura, ax


def contribuciones_riesgo(tabla: pd.DataFrame, ax=None, figsize=(8, 4.5)):
    """Compara el peso en capital con la contribucion al riesgo de cada activo.

    Es el grafico que desmonta la intuicion de que repartir el capital a partes
    iguales reparte el riesgo a partes iguales.
    """
    figura, ax = _crear_ejes(ax, figsize)

    posiciones = np.arange(len(tabla))
    ancho = 0.38
    # Separacion de 2 px entre barras adyacentes mediante un hueco explicito.
    ax.bar(
        posiciones - ancho / 2 - 0.01,
        tabla["peso"],
        ancho,
        color=PALETA[0],
        label="Peso en capital",
    )
    ax.bar(
        posiciones + ancho / 2 + 0.01,
        tabla["contribucion_relativa"],
        ancho,
        color=PALETA[1],
        label="Contribucion al riesgo",
    )

    for posicion, (peso, contribucion) in enumerate(
        zip(tabla["peso"], tabla["contribucion_relativa"])
    ):
        ax.text(
            posicion - ancho / 2 - 0.01, peso, f"{peso:.0%}",
            ha="center", va="bottom", fontsize=8, color=CHROME["tinta_secundaria"],
        )
        ax.text(
            posicion + ancho / 2 + 0.01, contribucion, f"{contribucion:.0%}",
            ha="center", va="bottom", fontsize=8, color=CHROME["tinta_secundaria"],
        )

    ax.set_xticks(posiciones)
    ax.set_xticklabels(tabla.index)
    ax.set_ylabel("Proporcion del total")
    _formato_porcentaje(ax)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return figura, ax


def series_temporales(
    series: dict[str, pd.Series],
    etiqueta_y: str = "",
    porcentaje: bool = False,
    ax=None,
    figsize=(10, 4.5),
):
    """Superpone varias series en un unico eje.

    Un solo eje siempre: comparar dos magnitudes de escalas distintas en el
    mismo grafico con dos ejes verticales es la forma mas rapida de enganiar al
    lector, incluido uno mismo.
    """
    figura, ax = _crear_ejes(ax, figsize)
    for indice, (nombre, serie) in enumerate(series.items()):
        ax.plot(serie.index, serie.to_numpy(), color=PALETA[indice % len(PALETA)], label=nombre)
    ax.set_ylabel(etiqueta_y)
    if porcentaje:
        _formato_porcentaje(ax)
    if len(series) > 1:
        ax.legend(loc="upper left", ncols=min(len(series), 4))
    ax.grid(axis="x", visible=False)
    return figura, ax


def caidas(serie_caidas: pd.Series, ax=None, figsize=(10, 3.8)):
    """Grafico 'bajo el agua': distancia al maximo historico en cada momento."""
    figura, ax = _crear_ejes(ax, figsize)
    ax.fill_between(
        serie_caidas.index,
        serie_caidas.to_numpy(),
        0.0,
        color=ESTADO["critico"],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(serie_caidas.index, serie_caidas.to_numpy(), color=ESTADO["critico"], linewidth=1.5)

    peor = serie_caidas.idxmin()
    ax.annotate(
        f"Maxima caida {serie_caidas.min():.1%}",
        (peor, serie_caidas.min()),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        color=CHROME["tinta"],
        fontweight="bold",
    )
    ax.set_ylabel("Caida desde maximos")
    _formato_porcentaje(ax)
    ax.grid(axis="x", visible=False)
    return figura, ax


def distribucion_simulada(
    simulaciones,
    var: float | None = None,
    es: float | None = None,
    ax=None,
    figsize=(9, 4.5),
):
    """Histograma de una distribucion simulada, con el VaR y el ES marcados."""
    figura, ax = _crear_ejes(ax, figsize)
    valores = np.asarray(simulaciones, dtype=float)

    ax.hist(valores, bins=120, color=PALETA[0], alpha=0.75, linewidth=0)
    if var is not None:
        ax.axvline(-var, color=ESTADO["critico"], linewidth=2, linestyle="--")
        ax.annotate(
            f"VaR {var:.1%}",
            (-var, ax.get_ylim()[1] * 0.92),
            textcoords="offset points",
            xytext=(-6, 0),
            ha="right",
            fontsize=9,
            color=ESTADO["critico"],
            fontweight="bold",
        )
    if es is not None:
        ax.axvline(-es, color=CHROME["tinta"], linewidth=2, linestyle=":")
        ax.annotate(
            f"ES {es:.1%}",
            (-es, ax.get_ylim()[1] * 0.78),
            textcoords="offset points",
            xytext=(-6, 0),
            ha="right",
            fontsize=9,
            color=CHROME["tinta"],
            fontweight="bold",
        )
    ax.set_xlabel("Rendimiento simulado")
    ax.set_ylabel("Frecuencia")
    _formato_porcentaje(ax, "x")
    ax.grid(axis="x", visible=False)
    return figura, ax


def backtest(
    rendimientos: pd.Series,
    var_estimado: pd.Series,
    excepciones: pd.Series | None = None,
    ax=None,
    figsize=(11, 4.5),
):
    """Rendimientos frente al VaR previsto, destacando las excepciones.

    Es la imagen que resume un backtest: si los puntos rojos aparecen en
    racimos, el modelo no reacciona a los cambios de regimen, por mucho que el
    numero total de excepciones cuadre.
    """
    figura, ax = _crear_ejes(ax, figsize)
    comun = rendimientos.index.intersection(var_estimado.index)
    reales = rendimientos.loc[comun]
    umbral = -var_estimado.loc[comun]

    ax.plot(
        reales.index,
        reales.to_numpy(),
        color=CHROME["tinta_apagada"],
        linewidth=0.7,
        label="Rendimiento diario",
    )
    ax.plot(umbral.index, umbral.to_numpy(), color=PALETA[0], linewidth=2, label="VaR previsto")

    if excepciones is None:
        excepciones = (reales < umbral).astype(int)
    fallos = excepciones[excepciones == 1].index.intersection(comun)
    ax.scatter(
        fallos,
        reales.loc[fallos].to_numpy(),
        s=28,
        color=ESTADO["critico"],
        edgecolors=CHROME["superficie"],
        linewidths=1.2,
        zorder=5,
        label=f"Excepciones ({len(fallos)})",
    )

    ax.set_ylabel("Rendimiento")
    _formato_porcentaje(ax)
    ax.legend(loc="lower left", ncols=3)
    ax.grid(axis="x", visible=False)
    return figura, ax
