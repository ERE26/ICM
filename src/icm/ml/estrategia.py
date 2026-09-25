"""De la prediccion al dinero: senal, posicion, costes y cuenta de resultados.

Un modelo con mejor RMSE puede perder dinero y uno con peor RMSE ganarlo, y no
es ninguna paradoja. El error cuadratico reparte el peso entre todos los dias
por igual; la cuenta de resultados lo concentra en los dias de movimiento
grande y descuenta friccion cada vez que se cruza la horquilla. Acertar mucho
en los dias planos no paga las comisiones de haberlo intentado.

Este modulo es el paso que casi nadie da y el unico que contesta a la pregunta
que decide: cuanto dinero deja la prediccion, y cuanta friccion aguanta antes
de dejar de dejarlo (:func:`coste_de_equilibrio`).

Convencion temporal
-------------------
``prediccion[t]`` se construye con informacion **hasta t incluido** y apuesta
por el rendimiento de ``t+1``. La posicion se toma al cierre de t y cobra el
rendimiento de t+1. De ahi ``retardo=1`` con rendimientos contemporaneos, que
es el unico valor correcto bajo esta convencion.

``ResultadoValidacion.predicciones`` **ya viene** en esa convencion y por eso
se pasa tal cual, sin desplazarla. Conviene saber por que: esa serie se indexa
con las filas de ``X``/``y``, y la ``y`` de
:func:`icm.ml.caracteristicas.construir_matriz` esta indexada en la fecha de
DECISION (``y[t]`` es el rendimiento de t a t+h). Hay dos formas correctas de
traerla aqui:

>>> r = ejecutar_estrategia(p.predicciones, md.y)                  # doctest: +SKIP
>>> r = ejecutar_estrategia(p.predicciones, contemporaneos, retardo=1)  # doctest: +SKIP

En la primera el retardo deducido es 0 porque el desplazamiento lo lleva ya la
serie de rendimientos; en la segunda lo pone ``retardo``. Las dos imputan el
mismo importe, una en la fecha de decision y otra en la de liquidacion.

Lo que **nunca** hay que hacer es adelantar esa serie con ``.shift(-1)``:
produce exactamente la cuenta de resultados de ``retardo=0``, la trampa que
este modulo dedica cuarenta lineas a prohibir, y encima sin el aviso, porque
el retardo declarado sigue siendo 1. Medido sobre un proceso con momento y un
Ridge validado hacia delante: Sharpe +4,22 con ``.shift(-1)`` frente a +1,57
emparejando bien, y el pnl coincide fecha a fecha con el de ``retardo=0``.

``.shift(-1)`` solo corrige una prediccion indexada en la fecha del
rendimiento que predice, que es lo que sale de construirla sobre una ``y``
contemporanea. Ahi desplazarla y quedarse con ``retardo=1`` es preferible a
compensar con ``retardo=0``: lo primero deja una sola convencion en todo el
montaje, lo segundo deja dos y la proxima serie que entre lo hara por la
equivocada.

Convencion de signos y unidades
-------------------------------
La posicion es una exposicion por unidad de capital: 1 es estar invertido, -1
estar vendido, 2 estar apalancado el doble. El ``pnl`` es el rendimiento
simple del capital en cada periodo, ya neto de costes, y el coste de un
periodo es ``|cambio de posicion| * coste_bps / 10000``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from icm.cartera.metricas import resumen_metricas
from icm.comun import DIAS_HABILES_ANIO, ErrorDatos, periodos_por_anio

#: Modos de construccion de la senal que admite :func:`senal_desde_prediccion`.
MODOS_SENAL = ("signo", "proporcional", "umbral")

#: Puntos basicos que hay en la unidad. El coste se expresa en bps porque es
#: como lo cotizan las mesas: 2 bps es 0,02 % del nominal movido.
BPS = 10_000.0

#: Volatilidad anual por debajo de la cual se sospecha que la serie que se ha
#: pasado a :func:`objetivo_volatilidad` no esta anualizada. Nada negociable
#: tiene un 2 % de volatilidad anual; una serie de volatilidades diarias, si.
VOLATILIDAD_ANUAL_SOSPECHOSA = 0.02

#: Precision de la biseccion de :func:`coste_de_equilibrio`, en bps. Afinar
#: mas no tiene sentido: nadie cotiza una horquilla con seis decimales.
TOLERANCIA_BPS = 1e-6

#: Claves que devuelve :func:`resumen_estrategia`, en orden.
CLAVES_RESUMEN = (
    "observaciones",
    "rentabilidad_anualizada",
    "volatilidad_anualizada",
    "ratio_sharpe",
    "sharpe_bruto",
    "merma_sharpe",
    "maxima_caida",
    "var_historico_95",
    "rotacion_media",
    "coste_total_anualizado",
    "coste_bps",
)


# --- utilidades internas -------------------------------------------------------


def _como_serie_con_huecos(datos: Any, nombre: str) -> pd.Series:
    """Convierte la entrada en una Series de float **conservando** los huecos.

    A diferencia de :func:`icm.comun.como_serie`, aqui los NaN no se descartan.
    Descartarlos adelantaria las fechas siguientes y el desplazamiento de
    ``retardo`` periodos dejaria de significar lo que dice: una prediccion sin
    valor no es una fecha que no exista, es una fecha en la que no hay opinion.
    """
    if isinstance(datos, pd.Series):
        serie = datos.astype(float)
    elif isinstance(datos, pd.DataFrame):
        if datos.shape[1] != 1:
            raise ErrorDatos(
                f"'{nombre}' tiene {datos.shape[1]} columnas: pase una sola serie, "
                "por ejemplo seleccionando la columna por su nombre o con .iloc[:, 0]."
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
                f"'{nombre}' tiene {valores.ndim} dimensiones: se esperaba una serie de una."
            )
        serie = pd.Series(valores, name=nombre)
    if serie.empty:
        raise ErrorDatos(f"'{nombre}' no contiene ninguna observacion.")
    return serie


def _indice_comun(a: pd.Series, b: pd.Series, nombre_a: str, nombre_b: str) -> pd.Index:
    """Fechas presentes en las dos series, ordenadas y sin repeticiones.

    Se exige un indice sin duplicados porque todo este modulo se sostiene sobre
    un desplazamiento temporal, y con dos filas para la misma fecha "el periodo
    anterior" deja de estar definido.
    """
    for serie, nombre in ((a, nombre_a), (b, nombre_b)):
        if serie.index.has_duplicates:
            repetidas = serie.index[serie.index.duplicated()].unique()[:3].tolist()
            raise ErrorDatos(
                f"'{nombre}' tiene fechas repetidas en el indice (por ejemplo {repetidas}): "
                "agregue o elimine los duplicados antes de operar, porque con dos filas por "
                "fecha el desplazamiento de un periodo no esta definido."
            )
    indice = a.index.intersection(b.index)
    if len(indice) == 0:
        raise ErrorDatos(
            f"'{nombre_a}' y '{nombre_b}' no comparten ninguna fecha. Suele ser un problema de "
            "indice, no de datos: frecuencias distintas, marcas de hora con y sin zona, o una "
            "serie indexada por posicion y la otra por fecha. Alinee ambas al mismo indice."
        )
    return indice.sort_values()


def _como_entero(valor: Any, nombre: str) -> int:
    """Comprueba que un parametro sea entero de verdad, no un 1.5 redondeado.

    Se rechaza ``bool`` explicitamente porque en Python es un entero y
    ``retardo=True`` colaria como ``retardo=1``.
    """
    if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)):
        raise ErrorDatos(
            f"'{nombre}' debe ser un numero entero de periodos, se ha recibido {valor!r}."
        )
    return int(valor)


def _sharpe_aritmetico(pnl: pd.Series, periodos_anio: int) -> float:
    """Sharpe del ``pnl``, con numerador aritmetico e invariante a la escala.

    Aqui no se reutiliza :func:`icm.cartera.metricas.ratio_sharpe` a proposito.
    Su numerador es la CAGR geometrica, que es la cifra correcta para una
    cartera que compone, pero el ``pnl`` de este modulo es aditivo por unidad de
    capital y :func:`curva_beneficio` lo dice expresamente. Mezclar las dos
    convenciones rompe justo la propiedad por la que un Sharpe sirve para
    comparar: al multiplicar la posicion por k el denominador se multiplica por
    k exactamente, pero el numerador geometrico no, porque el arrastre de
    volatilidad crece con el cuadrado.

    Medido con la misma senal y el mismo coste a x1, x2 y x3 de apalancamiento:
    la via geometrica publicaba 7,39 / 12,02 / 20,08 y ``comparar_estrategias``
    ordenaba ``['x3', 'x2', 'x1']``, es decir, repartia capital por
    apalancamiento; con este estimador las tres dan 4,71. Y en el extremo el
    arrastre volvia negativo el numerador, con lo que
    :func:`coste_de_equilibrio` caia de 32,8 a 19,1 bps sobre la misma senal
    (equilibrio aritmetico 33,5 bps en los cinco casos) y devolvia exactamente
    0 bps en cuanto un unico periodo tenia ``pnl <= -1``, porque ahi la CAGR se
    trunca a -1.

    La CAGR sigue publicandose, pero solo donde significa lo que dice: en la
    clave ``rentabilidad_anualizada``.
    """
    valores = pnl.to_numpy(dtype=float)
    if valores.size < 2:
        return float("nan")
    desviacion = float(np.std(valores, ddof=1))
    if not np.isfinite(desviacion) or desviacion <= 0.0:
        # Volatilidad nula: no hay Sharpe que comparar.
        return float("nan")
    return float(np.mean(valores) / desviacion * np.sqrt(periodos_anio))


def _escala_de_la_senal(serie: pd.Series, escala: float | None) -> float:
    """Divisor de la senal proporcional; por defecto, su desviacion tipica.

    Hace falta un divisor porque una prediccion viene en las unidades de la
    serie (EUR/MWh, rendimientos logaritmicos) y una posicion no: sin
    normalizar, el mismo codigo da posiciones de 0,0002 en una serie y de 40 en
    otra. Dividir por la desviacion tipica de la prediccion convierte la
    magnitud prevista en "cuantas desviaciones de conviccion", que ya es
    comparable entre mercados.

    La desviacion se calcula con toda la muestra, asi que conviene ser honesto
    sobre lo que eso implica: es un unico numero positivo que divide la serie
    entera, de modo que no puede cambiar ningun signo ni reordenar nada en el
    tiempo (no fabrica capacidad de anticipacion), pero si significa que el
    nivel de apalancamiento se ha elegido sabiendo como fue todo el periodo.
    Para un backtest sin esa concesion, pase ``escala`` a mano con un valor
    fijado de antemano.
    """
    if escala is not None:
        valor = float(escala)
        if not np.isfinite(valor) or valor <= 0.0:
            raise ErrorDatos(
                f"'escala' debe ser un numero positivo y finito, se ha recibido {escala!r}. "
                "Es el divisor que convierte la prediccion en posicion."
            )
        return valor

    valores = serie.to_numpy(dtype=float)
    finitos = valores[np.isfinite(valores)]
    if finitos.size < 2:
        raise ErrorDatos(
            "Hacen falta al menos 2 predicciones validas para deducir la escala de la senal; "
            f"solo hay {finitos.size}. Pase 'escala' explicitamente."
        )
    desviacion = float(np.std(finitos, ddof=1))
    if desviacion <= 0.0:
        raise ErrorDatos(
            "La prediccion es constante y no tiene escala de la que deducir el tamanio de la "
            "posicion. Use modo='signo' o pase 'escala' explicitamente."
        )
    return desviacion


# --- de la prediccion a la senal -----------------------------------------------


def senal_desde_prediccion(
    prediccion,
    modo: str = "signo",
    umbral: float = 0.0,
    escala: float | None = None,
    limite: float = 1.0,
) -> pd.Series:
    """Traduce una prediccion en una posicion deseada, periodo a periodo.

    Parametros
    ----------
    modo:
        ``"signo"``
            Posicion ``+limite``, ``-limite`` o 0 segun el signo de la
            prediccion, y 0 tambien cuando ``|prediccion| <= umbral``. Apuesta
            siempre lo mismo: solo explota la direccion, no la magnitud, lo que
            la hace muy robusta a una calibracion mala.
        ``"proporcional"``
            Posicion proporcional a la prediccion (dividida por ``escala``) y
            recortada a ``+-limite``. Apuesta mas cuando preve mas, que es lo
            optimo si la magnitud prevista esta bien calibrada; si no lo esta,
            concentra el riesgo justo en las previsiones mas exageradas, que
            son las menos fiables.
        ``"umbral"``
            Como ``"proporcional"``, pero con una banda muerta: exactamente 0
            mientras ``|prediccion| <= umbral``. Exige un ``umbral`` positivo.
    umbral:
        Conviccion minima, en las unidades de la prediccion, para molestarse en
        operar.
    escala:
        Divisor de la prediccion en los modos proporcionales. Por defecto, su
        desviacion tipica (vease :func:`_escala_de_la_senal`).
    limite:
        Posicion maxima en valor absoluto, en cualquiera de los tres modos.

    Por que el umbral importa en cuanto hay costes
    ----------------------------------------------
    Sin costes, operar por una prediccion diminuta es gratis y en promedio
    ligeramente favorable, asi que lo optimo es apostar siempre. Con costes
    deja de serlo: cruzar la horquilla cuesta lo mismo si se preve un
    movimiento de 5 puntos basicos que si se preve uno de 200, de modo que toda
    prediccion cuyo valor esperado no pague la friccion resta dinero **aunque
    el signo sea correcto**. Una senal con un 55 % de acierto direccional puede
    perder dinero sin mas que operarla demasiado a menudo.

    El umbral es la respuesta directa a eso: renuncia a las apuestas que no se
    pagan y, de paso, evita la rotacion que genera una prediccion que oscila
    alrededor de cero, donde el modelo no esta diciendo nada y el backtest
    acaba comprando y vendiendo lo mismo a diario. Para elegirlo,
    :func:`coste_de_equilibrio` dice cuanta friccion aguanta cada variante.

    Notas
    -----
    Las predicciones ausentes o infinitas dan posicion 0. Es la lectura
    conservadora: donde no hay opinion no se toma riesgo, y nunca se arrastra
    la opinion de ayer, que es lo que haria un ``ffill`` silencioso.
    """
    serie = _como_serie_con_huecos(prediccion, "prediccion")

    if modo not in MODOS_SENAL:
        raise ErrorDatos(
            f"Modo de senal '{modo}' desconocido. Disponibles: {', '.join(MODOS_SENAL)}."
        )
    umbral = float(umbral)
    if not np.isfinite(umbral) or umbral < 0.0:
        raise ErrorDatos(
            f"'umbral' es una distancia a cero y debe ser finito y no negativo, se ha "
            f"recibido {umbral}. Para no filtrar nada, dejelo en 0."
        )
    limite = float(limite)
    if not np.isfinite(limite) or limite <= 0.0:
        raise ErrorDatos(
            f"'limite' es la posicion maxima y debe ser positivo, se ha recibido {limite}."
        )
    if modo == "umbral" and umbral == 0.0:
        raise ErrorDatos(
            "El modo 'umbral' con umbral 0 no filtra nada y es exactamente el modo "
            "'proporcional': fije un umbral positivo (la mediana de |prediccion| es un "
            "punto de partida razonable) o cambie de modo."
        )
    if modo == "proporcional" and umbral > 0.0:
        raise ErrorDatos(
            "El modo 'proporcional' no tiene banda muerta y el umbral se quedaria sin "
            "efecto: use modo='umbral' si quiere dejar de operar las predicciones "
            "pequenias, o deje 'umbral' en 0."
        )

    valores = serie.to_numpy(dtype=float)
    finitos = np.isfinite(valores)
    if not finitos.any():
        raise ErrorDatos(
            "La prediccion no tiene ningun valor finito: no hay nada que operar. Compruebe "
            "el periodo de calentamiento del modelo o el alineamiento de las fechas."
        )

    posiciones = np.zeros_like(valores)
    opera = finitos & (np.abs(valores) > umbral)
    if modo == "signo":
        posiciones[opera] = np.sign(valores[opera]) * limite
        escala_usada = float("nan")
    else:
        escala_usada = _escala_de_la_senal(serie, escala)
        posiciones[opera] = np.clip(valores[opera] / escala_usada, -limite, limite)

    senal = pd.Series(posiciones, index=serie.index, name="senal")
    senal.attrs.update(
        {"modo": modo, "umbral": umbral, "limite": limite, "escala": escala_usada}
    )
    return senal


def objetivo_volatilidad(
    senal,
    volatilidad_realizada,
    objetivo_anual: float = 0.10,
    limite: float = 2.0,
) -> pd.Series:
    """Escala la posicion para apuntar a una volatilidad constante.

    La posicion se multiplica por ``objetivo_anual / volatilidad_realizada``,
    recortado a ``limite``. La idea es vieja y sencilla: si la senal dice "largo"
    con la misma fuerza en un mercado al 12 % de volatilidad y en uno al 60 %, la
    segunda posicion arriesga cinco veces mas sin que nadie lo haya decidido. Al
    normalizar, el riesgo deja de depender del regimen y pasa a ser una eleccion
    explicita.

    Es tambien lo que hace comparables dos estrategias distintas. Sin normalizar,
    la que gane mas puede estar ganando solo por ir mas apalancada, y comparar
    sus Sharpe no arregla nada porque la rotacion, los costes y la maxima caida
    no escalan igual que el Sharpe. Con las dos apuntando al mismo 10 % anual, lo
    que queda de diferencia es senal.

    Parametros
    ----------
    senal:
        Posicion deseada antes de normalizar, tipicamente de
        :func:`senal_desde_prediccion`.
    volatilidad_realizada:
        Volatilidad **anualizada** estimada con informacion disponible en cada
        fecha, indexada como la senal. Por ejemplo
        ``icm.riesgo.volatilidad.volatilidad_movil(rendimientos, 21,
        anualizar=True)``. Si los numeros parecen volatilidades por periodo se
        avisa con un ``UserWarning``, porque el error es facil de cometer y su
        efecto (multiplicador enorme, recorte permanente en ``limite``) no salta
        a la vista en el resultado.
    objetivo_anual:
        Volatilidad anual a la que se apunta. 0.10 es un objetivo tipico de un
        fondo sistematico diversificado.
    limite:
        Tope del **multiplicador**, no de la posicion final: con una senal
        acotada en 1 y ``limite=2`` la posicion no pasa de 2. El tope no es
        cosmetico: cuando la volatilidad estimada se acerca a cero, el
        multiplicador se dispara y una estrategia de libro se convierte en una
        apuesta apalancada veinte veces sobre un tramo de precios planos.

    Notas
    -----
    La volatilidad debe estar calculada con informacion de hasta t, igual que la
    prediccion; el desplazamiento que separa la decision del rendimiento lo pone
    despues :func:`ejecutar_estrategia` con ``retardo``. Las fechas sin
    volatilidad utilizable (ausente o cero) quedan planas: una volatilidad
    realizada de cero casi siempre es un precio estancado en los datos, y
    apalancarse al maximo sobre un defecto de los datos es justo como se
    fabrica un backtest espectacular.

    El resultado ya son posiciones, no una prediccion. Para ejecutarlo hay que
    pedirle a :func:`ejecutar_estrategia` que no vuelva a transformarlo:

    >>> posiciones = objetivo_volatilidad(senal, volatilidad)        # doctest: +SKIP
    >>> resultado = ejecutar_estrategia(                             # doctest: +SKIP
    ...     posiciones, rendimientos, modo="proporcional", escala=1.0, limite=2.0
    ... )

    con ``escala=1.0`` y un ``limite`` no menor que el de aqui, combinacion en la
    que :func:`senal_desde_prediccion` es la identidad.
    """
    serie_senal = _como_serie_con_huecos(senal, "senal")
    serie_vol = _como_serie_con_huecos(volatilidad_realizada, "volatilidad_realizada")

    objetivo_anual = float(objetivo_anual)
    if not np.isfinite(objetivo_anual) or objetivo_anual <= 0.0:
        raise ErrorDatos(
            f"'objetivo_anual' es una volatilidad y debe ser positiva, se ha recibido "
            f"{objetivo_anual}. Un 0.10 es un 10 % anual."
        )
    limite = float(limite)
    if not np.isfinite(limite) or limite <= 0.0:
        raise ErrorDatos(
            f"'limite' acota el multiplicador y debe ser positivo, se ha recibido {limite}."
        )

    indice = _indice_comun(serie_senal, serie_vol, "senal", "volatilidad_realizada")
    valores_senal = serie_senal.reindex(indice).to_numpy(dtype=float)
    valores_vol = serie_vol.reindex(indice).to_numpy(dtype=float)

    negativas = np.isfinite(valores_vol) & (valores_vol < 0.0)
    if negativas.any():
        raise ErrorDatos(
            f"Hay {int(negativas.sum())} volatilidades negativas en 'volatilidad_realizada': "
            "una volatilidad es una desviacion tipica. Puede que haya pasado varianzas con "
            "signo o rendimientos en lugar de volatilidades."
        )

    utiles = np.isfinite(valores_vol) & (valores_vol > 0.0)
    if not utiles.any():
        raise ErrorDatos(
            "No hay ninguna volatilidad utilizable en las fechas comunes: no se puede "
            "dimensionar la posicion. Compruebe el calentamiento de la ventana movil."
        )
    mediana = float(np.median(valores_vol[utiles]))
    if mediana < VOLATILIDAD_ANUAL_SOSPECHOSA:
        warnings.warn(
            f"La volatilidad mediana es {mediana:.5f}, demasiado baja para ser anual: "
            "parece volatilidad por periodo. Anualicela (volatilidad_movil(..., "
            "anualizar=True) o multiplicando por la raiz de los periodos por anio), porque "
            "si no el multiplicador se va al tope y la estrategia va apalancada al maximo "
            "de principio a fin.",
            UserWarning,
            stacklevel=2,
        )

    multiplicador = np.zeros_like(valores_vol)
    multiplicador[utiles] = np.minimum(objetivo_anual / valores_vol[utiles], limite)
    posiciones = pd.Series(
        np.nan_to_num(valores_senal, nan=0.0, posinf=0.0, neginf=0.0) * multiplicador,
        index=indice,
        name="posicion_objetivo_volatilidad",
    )
    posiciones.attrs.update({"objetivo_anual": objetivo_anual, "limite": limite})
    return posiciones


# --- ejecucion -----------------------------------------------------------------


@dataclass
class ResultadoEstrategia:
    """Cuenta de resultados completa de una estrategia, periodo a periodo.

    Todas las series comparten indice. ``pnl`` es lo unico que se cobra:
    ``pnl_bruto``, ``costes`` y ``rotacion`` estan para poder responder a la
    pregunta siguiente, que es cuanta de la ventaja bruta se queda la friccion.
    """

    posiciones: pd.Series
    pnl: pd.Series
    pnl_bruto: pd.Series
    costes: pd.Series
    rotacion: pd.Series
    coste_bps: float

    def __len__(self) -> int:
        return len(self.pnl)


def ejecutar_estrategia(
    prediccion,
    rendimientos,
    coste_bps: float = 2.0,
    retardo: int | None = None,
    **opciones_senal: Any,
) -> ResultadoEstrategia:
    """Convierte una prediccion en una cuenta de resultados neta de costes.

    Parametros
    ----------
    prediccion:
        Prevision indexada en la **fecha de decision**: ``prediccion[t]`` se ha
        construido con informacion hasta t y apuesta por el rendimiento de t+1.
    rendimientos:
        Rendimientos realizados del activo, en la convencion que sea (simple o
        logaritmica); lo que se multiplica por la posicion es el numero tal cual.
    coste_bps:
        Coste de transaccion en puntos basicos del nominal movido, ida o vuelta.
        El coste de un periodo es ``|cambio de posicion| * coste_bps / 10000``:
        pasar de 0 a 1 mueve un nominal y paga una vez; darse la vuelta de +1 a
        -1 mueve dos y paga el doble, que es lo que de verdad ocurre. Como
        referencia, 2 bps es un futuro liquido y 20-50 bps un mercado de energia
        estrecho.
    retardo:
        Periodos entre la fecha en que esta indexada la prediccion y el
        rendimiento contra el que se ejecuta. Vease mas abajo.
        Si se deja en ``None`` se deduce de la propia serie de rendimientos:
        ``1`` con rendimientos contemporaneos y ``0`` si vienen ya adelantados
        (los que produce
        :func:`icm.ml.caracteristicas.objetivo_rendimiento_futuro`, marcados
        con ``attrs["convencion"] = "futuro"``). Pasar un retardo positivo
        junto con una serie adelantada es un error y se rechaza, porque
        desplazaria dos veces sin dar ninguna senial. Una serie adelantada con
        ``attrs["horizonte"] > 1`` se rechaza siempre, con retardo o sin el:
        sus tramos se solapan y no hay retardo que lo arregle.
    **opciones_senal:
        Se pasan tal cual a :func:`senal_desde_prediccion` (``modo``, ``umbral``,
        ``escala``, ``limite``).

    Por que ``retardo=1`` y por que ``retardo=0`` es el error clasico
    ----------------------------------------------------------------
    Con ``retardo=1``, la posicion del periodo t es la senal de t-1: se decide
    con lo que se sabia al cierre de t-1 y se cobra el rendimiento de t, que
    entonces todavia no se conocia. Es la unica secuencia que se puede
    reproducir con dinero de verdad.

    Con ``retardo=0``, la posicion del periodo t es la senal de t, y la senal de
    t se construye con informacion de t: se esta cobrando el rendimiento del
    mismo periodo cuya informacion se ha usado para decidir. La estrategia deja
    de tener que anticipar nada y se limita a cobrar el movimiento que ya ha
    ocurrido. El resultado no es una estrategia optimista, es una imposibilidad:
    en el limite recoge la suma de los movimientos en valor absoluto, que es el
    maximo que puede extraerse de una serie y que ninguna estrategia real
    alcanza ni de lejos. Es el fallo mas frecuente de los backtests de
    estudiante, y es silencioso: no da error, da un Sharpe de 8.

    Por eso ``retardo=0`` avisa con un ``UserWarning`` en lugar de estar
    prohibido: sirve para medir el tamanio del sesgo (comparelo con
    ``retardo=1`` y vea de donde salia el beneficio), nunca para presentar un
    resultado. El aviso salta **tambien** cuando el 0 lo ha deducido la marca
    de la serie de rendimientos, con otro texto; vease mas abajo por que.

    Lo que **no** arregla nada es adelantar la prediccion con ``.shift(-1)``
    cuando ya esta indexada en la fecha de decision, como
    ``ResultadoValidacion.predicciones``: eso da la misma cuenta de resultados
    que ``retardo=0``, y sin aviso, porque el retardo declarado es 1.
    ``.shift(-1)`` corrige unicamente una prediccion indexada en la fecha del
    rendimiento que predice, es decir construida sobre una ``y``
    contemporanea.

    Un retardo mayor que 1 es legitimo y a veces obligatorio: modela que la
    orden no se ejecuta hasta la apertura siguiente, o que el dato del que vive
    el modelo se publica con un dia de demora.

    Por que la marca de la serie no basta para callar el aviso
    ---------------------------------------------------------
    ``attrs["convencion"]`` dice de donde salio la serie, no como esta
    alineada ahora: sobrevive intacta a ``.shift()``, ``.dropna()``,
    ``.reindex()``, ``.rename()``, ``.astype()``, ``.copy()`` y a la
    aritmetica. Quien realinea la serie con ``.shift(1)`` -- la operacion mas
    natural del mundo, y la que deja el rendimiento contemporaneo de toda la
    vida -- se queda con una serie que sigue diciendo "futuro". Medido: sobre
    esa serie el retardo deducido es 0, el pnl sumo 5,5292, exactamente la
    suma de los rendimientos en valor absoluto (la huella inconfundible de
    leer el futuro), el Sharpe publicado fue 66,3 y no se emitio ni un aviso.
    De ahi que el aviso se emita siempre que el retardo resuelto sea 0, y que
    el rechazo de ``retardo=1`` explique como borrar la marca en lugar de
    mandar al usuario a ``retardo=0``.

    Notas
    -----
    Solo se opera en las fechas comunes a las dos series con rendimiento
    conocido; los huecos del rendimiento no se pueden operar, y el
    desplazamiento se aplica sobre lo que queda, de modo que la posicion se
    mantiene a traves del hueco, que es lo que le pasa a una cartera cuando el
    mercado cierra.

    La rotacion del primer periodo se cuenta como cero: la serie empieza con la
    posicion ya puesta. Cargar ahi la entrada haria que una cartera de comprar y
    mantener pagase un coste unico que, anualizado sobre una muestra corta,
    ensuciaria la comparacion; y sobre todo haria falso lo que si debe ser
    cierto, que quien no cambia de posicion no paga nada.
    """
    coste_bps = float(coste_bps)
    if not np.isfinite(coste_bps) or coste_bps < 0.0:
        raise ErrorDatos(
            f"'coste_bps' es un coste y debe ser finito y no negativo, se ha recibido "
            f"{coste_bps}. Para un backtest sin friccion, pase 0."
        )
    # Una serie de rendimientos ADELANTADA (la que produce
    # icm.ml.caracteristicas.objetivo_rendimiento_futuro: su valor en t es el
    # rendimiento de t a t+h) ya lleva incorporado el desplazamiento. Aplicarle
    # ademas el retardo por defecto desplaza dos veces, y eso no da un error ni
    # un aviso: da un resultado plausible y equivocado. Con una senal real
    # medida en +0,11 de correlacion, el doble desplazamiento convierte un
    # Sharpe de +2,6 en -0,5. De ahi que la convencion se lea de la propia
    # serie en lugar de confiarla al usuario.
    adelantada = rendimientos.attrs.get("convencion") == "futuro" if hasattr(rendimientos, "attrs") else False
    horizonte_serie = int(rendimientos.attrs.get("horizonte", 1)) if adelantada else 1

    # Con horizonte h > 1 los tramos de rendimiento se solapan h-1 periodos, asi
    # que ejecutar uno por fecha cobra cada movimiento del precio hasta h veces y
    # autocorrelaciona el pnl, con lo que ninguna de las cifras anualizadas del
    # resumen significa nada. Medido con comprar y mantener sobre un activo cuyo
    # rendimiento log total es -0,1667: suma de pnl -0,1667 con h=1, -0,8444 con
    # h=5 (5,06 veces) y -3,5744 con h=21 (21,44 veces), con la autocorrelacion
    # del pnl subiendo a +0,955 y la volatilidad anual de 0,163 a 0,758. Una
    # posicion fija de 1 no puede rendir 21 veces lo que rinde el activo.
    if adelantada and horizonte_serie != 1:
        raise ErrorDatos(
            f"Los rendimientos vienen adelantados {horizonte_serie} periodos y por tanto se "
            f"solapan: el valor en t y el de t+1 comparten {horizonte_serie - 1} periodos de "
            f"movimiento. Ejecutar uno por fecha cobra cada movimiento hasta {horizonte_serie} "
            "veces y el pnl deja de ser reproducible con una unidad de capital. Quedese con "
            f"una decision cada {horizonte_serie} periodos (prediccion.iloc[::"
            f"{horizonte_serie}] y rendimientos.iloc[::{horizonte_serie}], reponiendo attrs "
            "con .attrs.update(...) si se pierden) y anualice el resumen con "
            f"periodos_anio={DIAS_HABILES_ANIO // horizonte_serie} si los datos son diarios; "
            "o construya el objetivo con horizonte=1 si lo que quiere es operar todas las "
            "fechas."
        )

    if retardo is None:
        retardo = 0 if adelantada else 1
    else:
        retardo = _como_entero(retardo, "retardo")
        if adelantada and retardo > 0:
            raise ErrorDatos(
                f"Los rendimientos vienen marcados como adelantados {horizonte_serie} "
                f"periodo(s) (su valor en t seria el rendimiento de t a t+{horizonte_serie}), "
                f"asi que el desplazamiento ya estaria hecho y con retardo={retardo} se "
                "desplazaria dos veces: el resultado seria plausible y falso. Si la serie "
                "sigue adelantada, pase retardo=0, o use los rendimientos contemporaneos con "
                "retardo=1. Si la ha realineado usted (por ejemplo con .shift(1)), lo que "
                "pasa es que la marca ha sobrevivido al desplazamiento y ya no describe el "
                "alineamiento: borrela con rendimientos.attrs.pop('convencion', None) y "
                "vuelva a llamar, porque si no se ejecutaria con retardo=0 sobre una serie "
                "contemporanea, que es exactamente leer el futuro."
            )

    if retardo < 0:
        raise ErrorDatos(
            f"'retardo' no puede ser negativo (se ha recibido {retardo}): seria ejecutar la "
            "posicion contra un rendimiento anterior a la decision, es decir, leer el futuro."
        )
    # El aviso se emite siempre que el retardo resuelto sea 0, tambien por la via
    # de la marca: attrs sobrevive a .shift(), asi que una serie ya realineada
    # llega aqui diciendo "futuro" y callar el aviso por eso era suprimirlo justo
    # en el caso de fuga (vease el docstring).
    if retardo == 0:
        if adelantada:
            warnings.warn(
                "Se ejecuta con retardo=0 porque la serie de rendimientos viene marcada como "
                "adelantada (attrs['convencion'] = 'futuro'). Compruebe que sigue estandolo: "
                "esa marca dice de donde salio la serie, no como esta alineada ahora, y "
                "sobrevive intacta a .shift(), .dropna(), .reindex() y a la aritmetica. Si la "
                "ha realineado a rendimientos contemporaneos, borre la marca con "
                "rendimientos.attrs.pop('convencion', None) y deje el retardo por defecto; si "
                "no, el resultado incluye informacion futura y no es replicable.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "retardo=0 ejecuta la posicion contra el rendimiento del mismo periodo cuya "
                "informacion ha usado la prediccion: el resultado incluye informacion futura "
                "y no es replicable. Si la prediccion viene indexada en la fecha del "
                "rendimiento que predice, desplacela con .shift(-1) y deje retardo=1; si ya "
                "viene indexada en la fecha de decision, como ResultadoValidacion."
                "predicciones, no la desplace y deje el retardo por defecto.",
                UserWarning,
                stacklevel=2,
            )

    senal = senal_desde_prediccion(prediccion, **opciones_senal)
    serie_rendimientos = _como_serie_con_huecos(rendimientos, "rendimientos")
    indice = _indice_comun(senal, serie_rendimientos, "prediccion", "rendimientos")

    senal = senal.reindex(indice)
    rendimientos_comunes = serie_rendimientos.reindex(indice)
    conocidos = np.isfinite(rendimientos_comunes.to_numpy(dtype=float))
    senal, rendimientos_comunes = senal[conocidos], rendimientos_comunes[conocidos]

    minimo = retardo + 2
    if len(rendimientos_comunes) < minimo:
        raise ErrorDatos(
            f"Quedan {len(rendimientos_comunes)} fechas con prediccion y rendimiento conocido, "
            f"y con retardo={retardo} hacen falta al menos {minimo} para que haya dos periodos "
            "operados. Alargue el periodo o reduzca el calentamiento del modelo."
        )

    posiciones = senal.shift(retardo).iloc[retardo:].rename("posicion")
    rendimientos_operados = rendimientos_comunes.iloc[retardo:]

    pnl_bruto = (posiciones * rendimientos_operados).rename("pnl_bruto")
    # La primera rotacion es cero: la serie arranca con la posicion ya puesta.
    rotacion = posiciones.diff().abs().fillna(0.0).rename("rotacion")
    costes = (rotacion * (coste_bps / BPS)).rename("costes")
    pnl = (pnl_bruto - costes).rename("pnl")

    # El pnl es un rendimiento simple del capital aunque los rendimientos de
    # entrada fuesen logaritmicos: el doble de un rendimiento logaritmico no es
    # el rendimiento logaritmico de una posicion doble. Se marca para que
    # resumen_metricas no lo trate como logaritmico por herencia de attrs.
    for serie in (pnl, pnl_bruto):
        serie.attrs["tipo_rendimiento"] = "simple"

    return ResultadoEstrategia(
        posiciones=posiciones,
        pnl=pnl,
        pnl_bruto=pnl_bruto,
        costes=costes,
        rotacion=rotacion,
        coste_bps=coste_bps,
    )


# --- medicion ------------------------------------------------------------------


def _validar_resultado(resultado: Any, nombre: str = "resultado") -> ResultadoEstrategia:
    if not isinstance(resultado, ResultadoEstrategia):
        raise ErrorDatos(
            f"'{nombre}' debe ser un ResultadoEstrategia devuelto por ejecutar_estrategia, "
            f"se ha recibido {type(resultado).__name__}."
        )
    return resultado


def _validar_resultados(resultados: Any) -> dict[str, ResultadoEstrategia]:
    if not isinstance(resultados, dict):
        raise ErrorDatos(
            "'resultados' debe ser un diccionario {nombre: ResultadoEstrategia} para poder "
            "etiquetar cada estrategia en la comparacion."
        )
    if not resultados:
        raise ErrorDatos("No se ha pasado ninguna estrategia que comparar.")
    return {str(nombre): _validar_resultado(r, str(nombre)) for nombre, r in resultados.items()}


def resumen_estrategia(
    resultado: ResultadoEstrategia,
    periodos_anio: int | str = DIAS_HABILES_ANIO,
) -> pd.Series:
    """Las cifras que deciden si una estrategia vale, en una sola columna.

    Se apoya en :func:`icm.cartera.resumen_metricas` para la rentabilidad, la
    volatilidad, la maxima caida y el VaR historico, y se le anaden las tres
    cifras propias de una estrategia: la rotacion, el coste anualizado y el
    Sharpe que se habria presentado sin costes. Los dos Sharpe se calculan aqui
    (:func:`_sharpe_aritmetico`) y no se toman de ``resumen_metricas``, por el
    motivo que explica esa funcion.

    Ese ultimo par es el que hay que leer junto. ``sharpe_bruto`` es lo que mide
    la capacidad predictiva; ``ratio_sharpe`` es lo que se cobra; y
    ``merma_sharpe``, la diferencia, es lo que se queda la friccion. Una
    estrategia con un Sharpe bruto de 1,4 y uno neto de 0,1 no es una buena
    estrategia con mala ejecucion: es una senal que no sobrevive a operarse, y
    el remedio no es negociar comisiones sino operar menos (un ``umbral``, vease
    :func:`senal_desde_prediccion`).

    ``coste_total_anualizado`` es el coste medio por periodo llevado a un anio,
    es decir, cuantos puntos de rentabilidad anual se pagan en friccion.
    Cuidado al restarlo de ``rentabilidad_anualizada``: aquel es aritmetico y
    esta es una CAGR, asi que la caida real de la CAGR al pagar ese coste no es
    la misma cifra (medido: 0,0493 declarado frente a 0,0780 de caida efectiva,
    un 37 % menos). Para leerlos juntos sin sorpresas, compare
    ``coste_total_anualizado`` con ``rotacion_media * coste_bps / 10000 * n``,
    que es la misma cuenta, o mire directamente ``merma_sharpe``.
    ``rotacion_media`` es el nominal medio movido por periodo: 1,0 significa
    renovar la posicion entera cada periodo, y con datos diarios eso son unas
    252 veces al anio.

    El ``pnl`` se trata como rendimiento simple del capital (vease la nota de
    :func:`ejecutar_estrategia`). Los dos Sharpe salen de
    :func:`_sharpe_aritmetico`, con numerador aritmetico y no CAGR, para que
    sean invariantes a la escala de la posicion y para que su diferencia
    signifique solo lo que dice significar.
    """
    resultado = _validar_resultado(resultado)
    n = periodos_por_anio(periodos_anio)

    neto = resumen_metricas(resultado.pnl, periodos_anio=n, tipo="simple")
    sharpe_neto = _sharpe_aritmetico(resultado.pnl, n)
    sharpe_bruto = _sharpe_aritmetico(resultado.pnl_bruto, n)

    resumen = pd.Series(
        {
            "observaciones": float(len(resultado.pnl)),
            "rentabilidad_anualizada": float(neto["rentabilidad_anualizada"]),
            "volatilidad_anualizada": float(neto["volatilidad_anualizada"]),
            "ratio_sharpe": sharpe_neto,
            "sharpe_bruto": sharpe_bruto,
            "merma_sharpe": float(sharpe_bruto - sharpe_neto),
            "maxima_caida": float(neto["maxima_caida"]),
            "var_historico_95": float(neto["var_historico_95"]),
            "rotacion_media": float(resultado.rotacion.mean()),
            "coste_total_anualizado": float(resultado.costes.mean() * n),
            "coste_bps": float(resultado.coste_bps),
        },
        name=resultado.pnl.name or "estrategia",
    )
    return resumen.reindex(list(CLAVES_RESUMEN))


def comparar_estrategias(
    resultados: dict[str, ResultadoEstrategia],
    periodos_anio: int | str = DIAS_HABILES_ANIO,
) -> pd.DataFrame:
    """Una fila por estrategia con el resumen de cada una, la mejor arriba.

    Ordena por Sharpe neto descendente, que es el criterio con el que se
    reparte capital. Conviene mirar la tabla en dos pasadas: primero la columna
    ``ratio_sharpe`` para ordenar, y despues ``merma_sharpe`` y
    ``rotacion_media`` para entender por que la que gana gana. No es raro que la
    mejor bruta sea la peor neta.
    """
    resultados = _validar_resultados(resultados)
    tabla = pd.DataFrame(
        {nombre: resumen_estrategia(r, periodos_anio) for nombre, r in resultados.items()}
    ).T
    tabla.index.name = "estrategia"
    return tabla.sort_values("ratio_sharpe", ascending=False)


def curva_beneficio(resultados: dict[str, ResultadoEstrategia]) -> pd.DataFrame:
    """Beneficio acumulado de cada estrategia, una columna por estrategia.

    Se acumula por suma y no por composicion, a proposito: la posicion esta
    definida como exposicion por unidad de capital, asi que la serie dice
    "beneficio acumulado por unidad de capital" sin suponer ninguna politica de
    reinversion, que es una decision aparte y que con posiciones cortas o
    apalancadas cambia el resultado bastante. Para la curva de valor compuesta
    esta :func:`icm.riesgo.caidas.curva_valor` sobre ``resultado.pnl``, que es
    la que usa por dentro la maxima caida del resumen.

    Las estrategias pueden cubrir periodos distintos; se toma la union de las
    fechas y se prolonga la ultima cifra conocida de cada una, con cero antes de
    que empiece. Si ni una fecha tienen en comun se avisa, porque entonces la
    grafica pone una al lado de otra dos cosas que no son comparables.
    """
    resultados = _validar_resultados(resultados)

    comun: pd.Index | None = None
    for r in resultados.values():
        comun = r.pnl.index if comun is None else comun.intersection(r.pnl.index)
    if comun is not None and len(comun) == 0:
        warnings.warn(
            "Las estrategias comparadas no comparten ninguna fecha: la grafica pondra juntas "
            "series de periodos distintos, que no se pueden comparar. Recorte todas al mismo "
            "tramo antes de dibujarlas.",
            UserWarning,
            stacklevel=2,
        )

    curvas = pd.DataFrame({nombre: r.pnl.cumsum() for nombre, r in resultados.items()})
    return curvas.sort_index().ffill().fillna(0.0)


def _sharpe_con_coste(
    resultado: ResultadoEstrategia,
    coste_bps: float,
    periodos_anio: int = DIAS_HABILES_ANIO,
) -> float:
    """Sharpe neto que tendria el resultado con otro coste de transaccion.

    No hace falta volver a ejecutar la estrategia: ni la senal ni las posiciones
    ni la rotacion dependen del coste, asi que el pnl con cualquier coste es el
    bruto menos la rotacion escalada. Se usa el mismo :func:`_sharpe_aritmetico`
    que :func:`resumen_estrategia` para que las dos cifras sean la misma cifra.
    """
    pnl = resultado.pnl_bruto - resultado.rotacion * (coste_bps / BPS)
    return _sharpe_aritmetico(pnl, periodos_anio)


def coste_de_equilibrio(
    prediccion,
    rendimientos,
    maximo_bps: float = 100.0,
    **opciones: Any,
) -> float:
    """Coste en puntos basicos que deja el Sharpe de la estrategia en cero.

    Es la forma mas honesta de presentar una estrategia. Un Sharpe depende del
    coste que se le haya supuesto, y el coste supuesto es el parametro mas
    facil de elegir a conveniencia de todo el ejercicio; el coste de equilibrio
    le da la vuelta a la pregunta y no deja margen: "esta senal aguanta 18 puntos
    basicos de friccion". Quien la lea puede comparar esa cifra con lo que sabe
    que cuesta operar en su mercado y decidir el solo, sin discutir supuestos.

    Ademas separa dos cosas que suelen confundirse. Una senal con poca ventaja
    pero muy poca rotacion puede aguantar mucho coste, y una senal con mucha
    ventaja que se renueva a diario puede no aguantar ninguno. El Sharpe solo no
    distingue esos dos casos; esta cifra si.

    Parametros
    ----------
    maximo_bps:
        Extremo superior de la busqueda. Si la estrategia sigue teniendo Sharpe
        positivo con ese coste se devuelve ``maximo_bps`` y se avisa: la
        respuesta es "al menos esto", y para el valor exacto hay que subir el
        tope.
    **opciones:
        Se pasan a :func:`ejecutar_estrategia` (``retardo``, ``modo``,
        ``umbral``, ``escala``, ``limite``). ``coste_bps`` no, porque es
        justamente la incognita.

    Devuelve
    --------
    El coste en bps que anula el Sharpe, o ``0.0`` si ni con coste cero el
    Sharpe es positivo: una estrategia que pierde dinero sin friccion no tiene
    ninguna friccion que aguantar, y devolver un numero positivo ahi seria
    enganioso.

    Notas
    -----
    La busqueda es una biseccion, que supone que el Sharpe decrece con el coste.
    Lo hace: la rotacion no depende del coste, asi que subirlo solo resta a cada
    periodo una cantidad no negativa fija. La volatilidad del pnl puede moverse
    un poco al hacerlo, asi que la monotonia es practica y no un teorema; con
    una rotacion normal el Sharpe es decreciente y suave.

    No hay parametro ``periodos_anio`` porque no cambiaria el resultado: el
    signo del Sharpe no depende de la anualizacion, que solo multiplica
    numerador y denominador por constantes positivas. Por el mismo motivo la
    cifra es invariante a la escala de la posicion, que es lo que se espera de
    ella: el Sharpe que se usa aqui tiene numerador aritmetico (vease
    :func:`_sharpe_aritmetico`), asi que apalancar la misma senal no mueve el
    coste de equilibrio.
    """
    maximo_bps = float(maximo_bps)
    if not np.isfinite(maximo_bps) or maximo_bps <= 0.0:
        raise ErrorDatos(
            f"'maximo_bps' acota la busqueda y debe ser positivo, se ha recibido {maximo_bps}."
        )
    if "coste_bps" in opciones:
        raise ErrorDatos(
            "'coste_bps' es lo que esta funcion calcula: no se puede fijar. Use 'maximo_bps' "
            "para acotar la busqueda, o ejecutar_estrategia si quiere un coste concreto."
        )

    base = ejecutar_estrategia(prediccion, rendimientos, coste_bps=0.0, **opciones)

    sin_costes = _sharpe_con_coste(base, 0.0)
    if not sin_costes > 0.0:  # cubre tambien el NaN de una volatilidad nula
        return 0.0
    if _sharpe_con_coste(base, maximo_bps) > 0.0:
        warnings.warn(
            f"La estrategia sigue teniendo Sharpe positivo con {maximo_bps:g} bps de coste: se "
            "devuelve el tope de la busqueda, que hay que leer como 'al menos esto'. Suba "
            "'maximo_bps' si necesita el valor exacto.",
            UserWarning,
            stacklevel=2,
        )
        return maximo_bps

    bajo, alto = 0.0, maximo_bps
    while alto - bajo > TOLERANCIA_BPS:
        medio = 0.5 * (bajo + alto)
        if _sharpe_con_coste(base, medio) > 0.0:
            bajo = medio
        else:
            alto = medio
    return 0.5 * (bajo + alto)
