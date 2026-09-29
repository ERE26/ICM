"""Valor en Riesgo (VaR) y Perdida Esperada (Expected Shortfall).

Convencion de signos
--------------------
Todas las funciones devuelven la perdida como **numero positivo**. Un VaR al
99 % de 0.031 significa "se espera perder mas del 3,1 % solo el 1 % de los
dias". Es la convencion de los comites de riesgos y evita el baile de signos
que tanto confunde al comparar modelos.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos, como_serie, validar_nivel

METODOS = ("historico", "parametrico_normal", "parametrico_t", "cornish_fisher")


#: Definicion de cuantil que se usa en todo el modulo.
#:
#: ``inverted_cdf`` es la funcion de cuantiles empirica tal y como se define:
#: ``Q(p) = inf{x : F(x) >= p}``, es decir el orden estadistico ``ceil(n*p)``.
#: El valor por omision de ``numpy`` es ``linear``, que interpola entre dos
#: ordenes estadisticos y **siempre cae del lado menos extremo**: con 250
#: observaciones al 99 %, el indice es ``0,01 * 249 = 2,49``, o sea un punto
#: entre el tercer y el cuarto peor dia en lugar del tercero. Eso no es un
#: detalle de redondeo. Medido sobre 30 000 normales iid con ventana movil de
#: 250 sesiones: 401 excepciones frente a 297,5 esperadas y Kupiec
#: ``p = 1,0e-08`` con ``linear``, contra 348 y ``p = 0,0042`` con
#: ``inverted_cdf``; el VaR medio sale un 2,9 % por debajo.
METODO_CUANTIL = "inverted_cdf"


def var_historico(rendimientos, nivel: float = 0.99) -> float:
    """VaR por simulacion historica: el cuantil empirico de las perdidas.

    No supone ninguna distribucion, asi que recoge de forma natural la
    asimetria y las colas gruesas. A cambio, no puede decir nada de perdidas
    mayores que la peor observada en la muestra.

    Sobre el cuantil, y lo que no arregla
    -------------------------------------
    Se usa la funcion de cuantiles empirica (``METODO_CUANTIL``), no la
    interpolacion por omision de ``numpy``. Corregirlo quita un sesgo del
    estimador, pero **no** hace que un VaR historico a 250 sesiones pase el
    contraste de Kupiec al 99 %, y conviene no confundir las dos cosas: con
    250 observaciones el cuantil al 99 % es el tercer peor dia, cuya
    probabilidad de ser superado es ``3/251 = 1,20 %``, no el 1,00 %. Medido,
    la tasa de excepciones sale 1,170 %. Ese exceso es una propiedad de la
    ventana, no un defecto del codigo: se quita alargando la ventana o bajando
    el nivel, no cambiando el estimador.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    return float(-np.quantile(serie, 1.0 - nivel, method=METODO_CUANTIL))


def es_historico(rendimientos, nivel: float = 0.99) -> float:
    """Perdida media condicionada a superar el VaR historico.

    A diferencia del VaR, el ES si mira dentro de la cola, y es una medida
    coherente (subaditiva): diversificar nunca puede empeorarlo. Es la medida
    que Basilea III adopto en sustitucion del VaR.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    umbral = np.quantile(serie, 1.0 - nivel, method=METODO_CUANTIL)
    cola = serie[serie <= umbral]
    if cola.empty:  # pragma: no cover - solo con muestras minusculas
        return float(-umbral)
    return float(-cola.mean())


def var_parametrico(
    rendimientos,
    nivel: float = 0.99,
    distribucion: str = "normal",
) -> float:
    """VaR suponiendo una distribucion parametrica de los rendimientos.

    Parametros
    ----------
    distribucion:
        ``"normal"`` es el modelo clasico de varianza-covarianza; ``"t"`` ajusta
        una t de Student por maxima verosimilitud, lo que da colas mas gruesas
        y un VaR mas prudente cuando los datos no son normales.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    alfa = 1.0 - nivel

    if distribucion == "normal":
        media, desviacion = float(serie.mean()), float(serie.std(ddof=1))
        return float(-(media + desviacion * stats.norm.ppf(alfa)))
    if distribucion == "t":
        gl, loc, escala = _ajustar_t(serie)
        return float(-(loc + escala * stats.t.ppf(alfa, gl)))
    raise ErrorDatos(f"Distribucion '{distribucion}' desconocida: use 'normal' o 't'.")


#: Por debajo de estos grados de libertad la t no tiene media, asi que el ES
#: -- que es una media condicionada -- no existe. La formula cerrada divide
#: entre ``gl - 1`` y no lo comprobaba: con una muestra de una t de 0,8 grados
#: el ajuste daba 0,832 y el ES salia **-3,0654** con un VaR de +0,6194, es
#: decir un ES menor que el VaR, que es imposible por definicion.
GL_MINIMOS_ES = 1.0

#: Por debajo de dos grados la t no tiene varianza. El VaR sigue estando bien
#: definido (es un cuantil), pero conviene decirlo.
GL_MINIMOS_VARIANZA = 2.0


def _ajustar_t(serie: pd.Series) -> tuple[float, float, float]:
    """Ajusta una t por maxima verosimilitud avisando si sale degenerada."""
    gl, loc, escala = stats.t.fit(serie)
    if not np.isfinite([gl, loc, escala]).all():
        raise ErrorDatos(
            "El ajuste de la t de Student no ha convergido a parametros finitos. "
            "Use el metodo historico."
        )
    if gl < GL_MINIMOS_VARIANZA:
        warnings.warn(
            f"La t ajustada tiene {gl:.3f} grados de libertad, por debajo de "
            f"{GL_MINIMOS_VARIANZA:.0f}: no tiene varianza finita. El cuantil sigue "
            "estando definido, pero cualquier cifra que dependa de la desviacion "
            "tipica -- escalar a otro horizonte, entre otras -- deja de tener "
            "sentido. Compare con el metodo historico.",
            UserWarning,
            stacklevel=3,
        )
    return float(gl), float(loc), float(escala)


def es_parametrico(
    rendimientos,
    nivel: float = 0.99,
    distribucion: str = "normal",
) -> float:
    """Perdida esperada bajo la distribucion parametrica ajustada."""
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    alfa = 1.0 - nivel

    if distribucion == "normal":
        media, desviacion = float(serie.mean()), float(serie.std(ddof=1))
        densidad = stats.norm.pdf(stats.norm.ppf(alfa))
        return float(-(media - desviacion * densidad / alfa))
    if distribucion == "t":
        gl, loc, escala = _ajustar_t(serie)
        if gl <= GL_MINIMOS_ES:
            raise ErrorDatos(
                f"La t ajustada tiene {gl:.3f} grados de libertad: por debajo de "
                f"{GL_MINIMOS_ES:.0f} no tiene media, asi que la perdida esperada no "
                f"existe (la formula cerrada divide entre gl - 1 y devolvia un numero "
                f"con el signo cambiado). Use 'es_historico', que no supone nada."
            )
        cuantil = stats.t.ppf(alfa, gl)
        # Formula cerrada del ES de la t de Student estandarizada.
        factor = (gl + cuantil**2) / (gl - 1.0) * stats.t.pdf(cuantil, gl) / alfa
        return float(-(loc - escala * factor))
    raise ErrorDatos(f"Distribucion '{distribucion}' desconocida: use 'normal' o 't'.")


#: Minimo de observaciones para que los estimadores insesgados de asimetria y
#: curtosis esten definidos. Por debajo devolvian constantes que no son
#: estimaciones de nada -- con dos observaciones, asimetria 0,0 y curtosis
#: -2,0 siempre -- y Cornish-Fisher construia un "VaR" encima sin avisar.
MINIMO_MOMENTOS = 4


def _momentos(serie: pd.Series) -> tuple[float, float, float, float]:
    """Media, desviacion tipica, asimetria y exceso de curtosis de la serie."""
    if len(serie) < MINIMO_MOMENTOS:
        raise ErrorDatos(
            f"Hacen falta al menos {MINIMO_MOMENTOS} observaciones para estimar "
            f"asimetria y curtosis; hay {len(serie)}. Con menos, los estimadores "
            f"devuelven constantes que no dependen de los datos."
        )
    return (
        float(serie.mean()),
        float(serie.std(ddof=1)),
        float(stats.skew(serie, bias=False)),
        float(stats.kurtosis(serie, fisher=True, bias=False)),
    )


def _cuantil_cornish_fisher(probabilidad, asimetria: float, curtosis: float):
    """Cuantil normal corregido por los momentos de orden tres y cuatro."""
    z = stats.norm.ppf(probabilidad)
    return (
        z
        + (z**2 - 1.0) * asimetria / 6.0
        + (z**3 - 3.0 * z) * curtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * asimetria**2 / 36.0
    )


def dominio_valido_cornish_fisher(
    asimetria: float,
    curtosis: float,
    z_desde: float | None = None,
    z_hasta: float | None = None,
) -> bool:
    """Indica si la expansion de Cornish-Fisher sigue siendo una funcion de cuantiles.

    La expansion es asintotica: solo vale para asimetria y curtosis moderadas.
    Pasado cierto punto el polinomio deja de ser creciente, y una funcion de
    cuantiles que no crece no es una funcion de cuantiles: el "VaR" resultante
    puede ser menor al 99 % que al 95 %, o dispararse a decenas de sigmas.

    La derivada del polinomio respecto a ``z`` es la cuadratica
    ``a z^2 + b z + c`` con ``a = k/8 - s^2/6``, ``b = s/3`` y
    ``c = 1 - k/8 + 5 s^2/36``. El criterio es que sea positiva donde se
    evalua.

    Sobre el rango
    --------------
    Sin ``z_desde`` ni ``z_hasta`` se exige monotonia en **toda** la recta,
    que es el criterio de Maillard (2012): la cuadratica sin raices reales y
    abierta hacia arriba. Es la pregunta matematica limpia, pero como criterio
    practico es inservible, porque ``a < 0`` en cuanto la curtosis de exceso
    es negativa **por poco que sea**, y la mitad de las muestras normales lo
    son. Medido: el criterio global rechazaba el **57 % de 2 000 muestras
    normales de 250 observaciones** (53 % con 1 000). Y lo rechazaba por nada:
    con ``s = 0`` y ``k = -0,02``, la monotonia se rompe en ``|z| = 20``,
    cuando el VaR al 99,99 % vive en ``z = -3,72``.

    Por eso quien llama pasa el rango que de verdad va a evaluar --
    :func:`var_cornish_fisher` el de su nivel, :func:`es_cornish_fisher` el de
    toda la cola que integra -- y la comprobacion se hace ahi.

    >>> dominio_valido_cornish_fisher(0.0, 0.0)              # la normal
    True
    >>> dominio_valido_cornish_fisher(-1.2, 29.0)            # colas muy gruesas
    False
    >>> dominio_valido_cornish_fisher(0.0, -0.02)            # criterio global
    False
    >>> dominio_valido_cornish_fisher(0.0, -0.02, -2.33, 0)  # donde se usa
    True
    """
    s, k = float(asimetria), float(curtosis)
    if not (np.isfinite(s) and np.isfinite(k)):
        return False
    a = k / 8.0 - s**2 / 6.0
    b = s / 3.0
    c = 1.0 - k / 8.0 + 5.0 * s**2 / 36.0

    if z_desde is None or z_hasta is None:
        if abs(a) < 1e-12:
            return abs(b) < 1e-12 and c >= 0.0
        if a < 0.0:
            return False
        return (b**2 - 4.0 * a * c) <= 0.0

    inferior, superior = sorted((float(z_desde), float(z_hasta)))
    # El minimo de una cuadratica en un intervalo esta en un extremo o, si se
    # abre hacia arriba y el vertice cae dentro, en el vertice.
    candidatos = [inferior, superior]
    if a > 0.0:
        vertice = -b / (2.0 * a)
        if inferior <= vertice <= superior:
            candidatos.append(vertice)
    return min(a * z**2 + b * z + c for z in candidatos) > 0.0


def _avisar_si_fuera_de_dominio(
    asimetria: float,
    curtosis: float,
    z_desde: float,
    z_hasta: float = 0.0,
) -> None:
    if dominio_valido_cornish_fisher(asimetria, curtosis, z_desde, z_hasta):
        return
    warnings.warn(
        f"Cornish-Fisher fuera de su dominio de validez (asimetria {asimetria:.2f}, "
        f"exceso de curtosis {curtosis:.2f}): la expansion deja de ser monotona entre "
        f"z = {min(z_desde, z_hasta):.2f} y z = {max(z_desde, z_hasta):.2f}, que es el "
        "tramo que se esta evaluando, y el valor devuelto no es un cuantil fiable. Use "
        "el metodo historico o la t de Student con colas tan gruesas.",
        UserWarning,
        stacklevel=3,
    )


def var_cornish_fisher(rendimientos, nivel: float = 0.99) -> float:
    """VaR normal corregido por asimetria y curtosis.

    La expansion de Cornish-Fisher desplaza el cuantil normal en funcion de los
    momentos de tercer y cuarto orden. Es un termino medio barato entre el
    parametrico normal y el historico: mantiene la formula cerrada pero deja de
    suponer simetria y colas normales.

    Ojo: es una expansion asintotica. Con colas muy gruesas se sale de su
    dominio de validez y el resultado deja de ser un cuantil; en ese caso se
    emite un ``UserWarning``. Vease :func:`dominio_valido_cornish_fisher`.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    media, desviacion, s, k = _momentos(serie)
    # Se comprueba la monotonia desde la mediana hasta el cuantil pedido, que
    # es el tramo del que depende este numero: si el polinomio crece ahi, el
    # VaR crece con el nivel de confianza, que es lo que tiene que pasar.
    _avisar_si_fuera_de_dominio(s, k, float(stats.norm.ppf(1.0 - nivel)))
    return float(-(media + desviacion * _cuantil_cornish_fisher(1.0 - nivel, s, k)))


def es_cornish_fisher(rendimientos, nivel: float = 0.99, n_tramos: int = 1000) -> float:
    """ES obtenido integrando numericamente el VaR de Cornish-Fisher en la cola.

    No existe una formula cerrada, asi que se promedia el VaR de Cornish-Fisher
    sobre los niveles de la cola, que es la definicion misma del ES. Los
    momentos se calculan una sola vez: solo cambia el cuantil.
    """
    serie = como_serie(rendimientos, "rendimientos")
    nivel = validar_nivel(nivel)
    if n_tramos < 2:
        raise ErrorDatos("Hacen falta al menos dos tramos para integrar la cola.")
    alfa = 1.0 - nivel
    media, desviacion, s, k = _momentos(serie)
    # Regla del punto medio, que ademas evita el extremo alfa=0 donde el
    # cuantil diverge. Con los extremos derechos de cada tramo se tomaba
    # siempre el cuantil menos extremo y la integral salia corta de forma
    # sistematica, igual que pasaba en 'es_garch': medido con 1 000 tramos,
    # -0,043 % frente a -0,003 % por el punto medio, y siempre hacia el lado
    # inseguro.
    probabilidades = (np.arange(n_tramos) + 0.5) / n_tramos * alfa
    # Aqui el tramo evaluado llega mucho mas lejos que en el VaR: el primer
    # punto medio de la cola esta en z = -4,42 con alfa = 1 % y 1 000 tramos.
    _avisar_si_fuera_de_dominio(s, k, float(stats.norm.ppf(probabilidades[0])))
    cuantiles = _cuantil_cornish_fisher(probabilidades, s, k)
    return float(-(media + desviacion * cuantiles.mean()))


def var(
    rendimientos,
    nivel: float = 0.99,
    metodo: str = "historico",
) -> float:
    """Punto de entrada unico a los distintos metodos de VaR."""
    if metodo == "historico":
        return var_historico(rendimientos, nivel)
    if metodo == "parametrico_normal":
        return var_parametrico(rendimientos, nivel, "normal")
    if metodo == "parametrico_t":
        return var_parametrico(rendimientos, nivel, "t")
    if metodo == "cornish_fisher":
        return var_cornish_fisher(rendimientos, nivel)
    raise ErrorDatos(f"Metodo '{metodo}' desconocido. Disponibles: {', '.join(METODOS)}.")


def expected_shortfall(
    rendimientos,
    nivel: float = 0.99,
    metodo: str = "historico",
) -> float:
    """Punto de entrada unico a los distintos metodos de ES."""
    if metodo == "historico":
        return es_historico(rendimientos, nivel)
    if metodo == "parametrico_normal":
        return es_parametrico(rendimientos, nivel, "normal")
    if metodo == "parametrico_t":
        return es_parametrico(rendimientos, nivel, "t")
    if metodo == "cornish_fisher":
        return es_cornish_fisher(rendimientos, nivel)
    raise ErrorDatos(f"Metodo '{metodo}' desconocido. Disponibles: {', '.join(METODOS)}.")


def escalar_horizonte(
    medida: float,
    dias: int,
    metodo: str = "raiz_tiempo",
    media_diaria: float | None = None,
) -> float:
    """Lleva una medida de riesgo diaria a un horizonte de varios dias.

    La regla de la raiz del tiempo supone rendimientos independientes y sin
    autocorrelacion. Con volatilidad agrupada (que es lo normal) infravalora el
    riesgo a horizontes largos, asi que conviene usarla solo para 1-10 dias,
    que es justo el rango que exige la normativa.

    Parametros
    ----------
    media_diaria:
        La media de los rendimientos diarios, si la medida la lleva dentro.

        Un VaR es ``-(mu + sigma z)``. A T dias la media se acumula
        **linealmente** y la desviacion tipica con la raiz, asi que el VaR
        correcto es ``-(T mu + sqrt(T) sigma z)``, no ``sqrt(T)`` veces el
        diario: eso escala tambien la media, y con mu positiva (lo habitual en
        una serie con deriva) el resultado se va hacia arriba. Medido con
        ``mu = 0,0005`` y ``sigma = 0,01``: a 10 dias el error es del +4,99 %,
        y a un anio del **+48,53 %** -- 0,3614 frente a 0,2433.

        Pasando la media se aplica la formula correcta,
        ``sqrt(T) (VaR_1 + mu) - T mu``, que con ``mu = 0`` se reduce a la
        raiz del tiempo de siempre. Dejandola en ``None`` se escala tal cual,
        que es lo correcto **solo** si la medida se calculo con media nula.
    """
    if dias < 1:
        raise ErrorDatos("El horizonte debe ser de al menos un dia.")
    if metodo != "raiz_tiempo":
        raise ErrorDatos(f"Metodo de escalado '{metodo}' desconocido.")
    if media_diaria is None:
        return float(medida * np.sqrt(dias))
    mu = float(media_diaria)
    if not np.isfinite(mu):
        raise ErrorDatos("La media diaria tiene que ser un numero finito.")
    return float(np.sqrt(dias) * (medida + mu) - dias * mu)


def tabla_medidas(
    rendimientos,
    niveles=(0.95, 0.99),
    metodos=METODOS,
    valor_cartera: float | None = None,
) -> pd.DataFrame:
    """Compara VaR y ES de todos los metodos y niveles en una sola tabla.

    Si se indica ``valor_cartera``, se aniaden las cifras en unidades
    monetarias, que es como se presentan en un comite.
    """
    serie = como_serie(rendimientos, "rendimientos")
    filas = []
    for metodo in metodos:
        for nivel in niveles:
            fila = {
                "metodo": metodo,
                "nivel": nivel,
                "var": var(serie, nivel, metodo),
                "es": expected_shortfall(serie, nivel, metodo),
            }
            if valor_cartera is not None:
                fila["var_monetario"] = fila["var"] * valor_cartera
                fila["es_monetario"] = fila["es"] * valor_cartera
            filas.append(fila)
    return pd.DataFrame(filas).set_index(["metodo", "nivel"])


def var_incremental(
    rendimientos_activos: pd.DataFrame,
    pesos,
    nivel: float = 0.99,
    metodo: str = "historico",
) -> pd.Series:
    """Cuanto cambia el VaR de la cartera al eliminar cada activo.

    Un valor positivo significa que el activo aumenta el VaR total; uno
    negativo, que lo reduce (diversifica).

    Los pesos de los que quedan se **renormalizan** para volver a sumar 1, asi
    que la cifra mezcla dos cosas: quitar el activo y repartir su peso entre
    los demas. Es la convencion habitual -- comparar dos carteras invertidas
    al 100 % -- pero conviene saberlo: si el activo que se quita pesa poco y
    los que quedan son volatiles, el reparto puede subir el VaR y la
    contribucion sale positiva sin que el activo aportase riesgo por si
    mismo. Para descomponer el riesgo sin ese efecto, use
    :func:`icm.cartera.contribuciones_riesgo`, que reparte la volatilidad por
    la identidad de Euler y suma exactamente el total.
    """
    from icm.comun import normalizar_pesos
    from icm.datos.precios import rendimientos_cartera

    marco = rendimientos_activos
    serie_pesos = normalizar_pesos(pesos, marco.columns)
    var_total = var(rendimientos_cartera(marco, serie_pesos), nivel, metodo)

    incrementos = {}
    for activo in marco.columns:
        restantes = [c for c in marco.columns if c != activo]
        if not restantes:
            incrementos[activo] = var_total
            continue
        pesos_sin = normalizar_pesos(serie_pesos.loc[restantes], restantes)
        var_sin = var(rendimientos_cartera(marco[restantes], pesos_sin), nivel, metodo)
        incrementos[activo] = var_total - var_sin
    return pd.Series(incrementos, name=f"var_incremental_{nivel:.0%}")


def var_anualizado(rendimientos, nivel: float = 0.99, metodo: str = "historico") -> float:
    """VaR diario escalado a un anio, con la media escalada linealmente.

    Se ofrece por comodidad, pero con la advertencia de :func:`escalar_horizonte`:
    a un anio vista la regla de la raiz del tiempo es poco fiable.

    La media de la muestra se pasa a ``escalar_horizonte`` porque el VaR la
    lleva dentro y no escala con la raiz. Antes no se pasaba, y sobre una
    serie con deriva diaria del 0,05 % el VaR anual salia un 48,5 % por
    encima del correcto.
    """
    serie = como_serie(rendimientos, "rendimientos")
    return escalar_horizonte(
        var(serie, nivel, metodo),
        DIAS_HABILES_ANIO,
        media_diaria=float(serie.mean()),
    )
