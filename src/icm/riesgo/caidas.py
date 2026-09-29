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


def _curva_con_capital_inicial(
    rendimientos, tipo: str | None, valor_inicial: float = 1.0
) -> pd.Series:
    """La curva de valor con el capital invertido como primer punto.

    ``curva_valor`` devuelve el valor DESPUES de cada rendimiento, asi que su
    primer elemento ya incorpora el primer periodo y el capital de partida no
    aparece en ninguna fila. Tomar el maximo movil sobre esa curva hace que el
    maximo de partida sea el valor ya movido, y la perdida del primer periodo
    deja de existir: con ``[-0,50, +0,30, +0,30, +0,30]``, ``maxima_caida``
    devolvia **0,0** para una cartera que perdio la mitad el primer dia y
    nunca la recupero.

    El capital invertido es, obviamente, un valor alcanzado. Se antepone
    fechado un periodo antes del primer rendimiento.
    """
    curva = curva_valor(rendimientos, tipo, valor_inicial)
    if len(curva.index) >= 2:
        paso = curva.index[1] - curva.index[0]
        inicio = curva.index[0] - paso
    else:  # pragma: no cover - una sola observacion
        inicio = curva.index[0]
    con_inicio = pd.concat([pd.Series([valor_inicial], index=[inicio]), curva])
    con_inicio.index.name = curva.index.name
    con_inicio.name = curva.name
    return con_inicio


def serie_caidas(rendimientos, tipo: str | None = None) -> pd.Series:
    """Caida relativa respecto al maximo alcanzado hasta cada momento.

    Los valores son negativos o cero: -0.18 significa estar un 18 % por debajo
    del maximo historico. El capital invertido cuenta como maximo de partida,
    de modo que una perdida en el primer periodo si es una caida.
    """
    con_inicio = _curva_con_capital_inicial(rendimientos, tipo)
    caidas = con_inicio / con_inicio.cummax() - 1.0
    # La fila sintetica del capital inicial no se publica: su caida es cero
    # por construccion y su fecha no existe en los datos.
    caidas = caidas.iloc[1:]
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

    Si la serie empieza cayendo, el maximo previo es el capital invertido, que
    por definicion es anterior a la primera observacion. En ese caso ``pico``
    lleva una fecha **un periodo antes del inicio de los datos**: no es una
    sesion del mercado, es el momento en que se puso el dinero. Sin ella ese
    episodio no existia, y era justo el peor.
    """
    curva = _curva_con_capital_inicial(rendimientos, tipo)
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
        # La ULTIMA fecha en la que se alcanzo el maximo, no la primera: con
        # una curva plana en el techo antes de caer, 'idxmax' databa la caida
        # desde el primer dia de la meseta e inflaba su duracion.
        cima = anteriores[anteriores == anteriores.max()]
        fecha_pico = cima.index[-1]
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
