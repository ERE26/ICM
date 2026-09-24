"""Caidas desde maximos (drawdowns).

El drawdown es la medida de riesgo que el inversor percibe de verdad: no la
volatilidad, sino cuanto ha llegado a perder desde el ultimo maximo y cuanto
ha tardado en recuperarlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos, como_serie


def curva_valor(rendimientos, tipo: str | None = None, valor_inicial: float = 1.0) -> pd.Series:
    """Evolucion acumulada de una unidad invertida."""
    serie = como_serie(rendimientos, "rendimientos")
    if tipo is None:
        tipo = serie.attrs.get("tipo_rendimiento", "simple")
    if tipo not in {"log", "simple"}:
        raise ErrorDatos(f"Tipo de rendimiento '{tipo}' desconocido: use 'log' o 'simple'.")
    simples = np.expm1(serie) if tipo == "log" else serie
    curva = valor_inicial * (1.0 + simples).cumprod()
    curva.name = "valor"
    return curva


def serie_caidas(rendimientos, tipo: str | None = None) -> pd.Series:
    """Caida relativa respecto al maximo alcanzado hasta cada momento.

    Los valores son negativos o cero: -0.18 significa estar un 18 % por debajo
    del maximo historico.
    """
    curva = curva_valor(rendimientos, tipo)
    maximo = curva.cummax()
    caidas = curva / maximo - 1.0
    caidas.name = "caida"
    return caidas


def maxima_caida(rendimientos, tipo: str | None = None) -> float:
    """Peor caida desde maximos del periodo, como numero negativo."""
    return float(serie_caidas(rendimientos, tipo).min())


def tabla_caidas(rendimientos, tipo: str | None = None, n: int = 5) -> pd.DataFrame:
    """Las ``n`` peores caidas, con sus fechas y su duracion.

    Para cada episodio se identifica el maximo previo, el suelo y, si llego, la
    fecha de recuperacion. Un episodio sin recuperar aparece con
    ``recuperacion`` a ``NaT``.
    """
    curva = curva_valor(rendimientos, tipo)
    maximo = curva.cummax()
    caidas = curva / maximo - 1.0

    # Cada episodio arranca cuando la curva marca un maximo nuevo.
    en_caida = caidas < 0
    inicio_episodio = en_caida & ~en_caida.shift(1, fill_value=False)
    episodios = inicio_episodio.cumsum().where(en_caida)

    filas = []
    for identificador, tramo in caidas.groupby(episodios):
        if pd.isna(identificador):
            continue
        posicion_suelo = tramo.idxmin()
        anteriores = curva.loc[:posicion_suelo]
        fecha_pico = anteriores.idxmax()
        posteriores = curva.loc[posicion_suelo:]
        recuperados = posteriores[posteriores >= curva.loc[fecha_pico]]
        fecha_recuperacion = recuperados.index[0] if len(recuperados) else pd.NaT

        filas.append(
            {
                "pico": fecha_pico,
                "suelo": posicion_suelo,
                "recuperacion": fecha_recuperacion,
                "caida": float(tramo.min()),
                "periodos_hasta_suelo": int(curva.index.get_loc(posicion_suelo) - curva.index.get_loc(fecha_pico)),
                "periodos_recuperacion": (
                    int(curva.index.get_loc(fecha_recuperacion) - curva.index.get_loc(posicion_suelo))
                    if not pd.isna(fecha_recuperacion)
                    else np.nan
                ),
            }
        )

    if not filas:
        return pd.DataFrame(
            columns=[
                "pico",
                "suelo",
                "recuperacion",
                "caida",
                "periodos_hasta_suelo",
                "periodos_recuperacion",
            ]
        )
    tabla = pd.DataFrame(filas).sort_values("caida").head(n).reset_index(drop=True)
    return tabla


def duracion_maxima_caida(rendimientos, tipo: str | None = None) -> int:
    """Mayor numero de periodos consecutivos por debajo de un maximo previo.

    Mide el tiempo que el inversor pasa "bajo el agua", que suele pesar tanto
    como la profundidad de la caida.
    """
    caidas = serie_caidas(rendimientos, tipo)
    bajo_agua = (caidas < 0).to_numpy()
    maxima_racha = racha = 0
    for valor in bajo_agua:
        racha = racha + 1 if valor else 0
        maxima_racha = max(maxima_racha, racha)
    return int(maxima_racha)


def ulcer_index(rendimientos, tipo: str | None = None) -> float:
    """Raiz de la media de las caidas al cuadrado.

    Penaliza mas las caidas profundas y prolongadas que la maxima caida, que
    solo mira un instante concreto.
    """
    caidas = serie_caidas(rendimientos, tipo)
    return float(np.sqrt(np.mean(caidas.to_numpy() ** 2)))
