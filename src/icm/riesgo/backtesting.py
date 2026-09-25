"""Contraste de los modelos de VaR contra lo que de verdad ocurrio.

Un modelo de riesgo no vale por lo elegante que sea, sino por si acierta. Un
VaR al 99 % debe ser superado aproximadamente un 1 % de los dias: si se supera
mucho mas, infravalora el riesgo; si mucho menos, inmoviliza capital sin
motivo. Y las excepciones deben aparecer dispersas: si se concentran en una
semana, el modelo no reacciona a los cambios de regimen.

Los dos contrastes clasicos cubren esas dos preguntas: Kupiec mira cuantas
excepciones hay, Christoffersen mira si estan agrupadas.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from icm.comun import ErrorDatos, como_serie, validar_nivel

METODOS_BACKTEST = (
    "historico",
    "parametrico_normal",
    "parametrico_t",
    "cornish_fisher",
    "ewma",
    "garch",
)


def serie_var_movil(
    rendimientos,
    nivel: float = 0.99,
    ventana: int = 250,
    metodo: str = "historico",
    frecuencia_reajuste: int = 20,
    factor_decaimiento: float = 0.94,
) -> pd.Series:
    """Estima el VaR de cada dia usando solo la informacion anterior.

    Es la parte delicada de cualquier backtest: si el VaR del dia ``t`` se
    calcula con datos que incluyen el propio dia ``t``, el contraste da
    resultados excelentes y completamente falsos.

    Parametros
    ----------
    ventana:
        Numero de observaciones pasadas con las que se estima. 250 sesiones
        (un anio) es el estandar supervisor.
    frecuencia_reajuste:
        Solo para ``metodo="garch"``: cada cuantos dias se vuelve a estimar el
        modelo. Reestimarlo a diario es correcto pero muy costoso, y en la
        practica los parametros apenas se mueven de un dia para otro.
    """
    serie = como_serie(rendimientos, "rendimientos")
    if not serie.index.is_monotonic_increasing:
        # Las ventanas se construyen por POSICION (valores[t-ventana:t]), asi
        # que un indice desordenado convierte cada estimacion en una mezcla de
        # pasado y futuro. Medido sobre 1.200 rendimientos permutados: 947 de
        # las 950 ventanas contenian observaciones posteriores a la fecha que
        # estimaban, con una media de 126,7 futuras de 250, y el semaforo de
        # Basilea cambiaba de verde a amarilla segun la permutacion. No se
        # ordena en silencio porque un indice desordenado suele ser sintoma de
        # algo peor aguas arriba.
        raise ErrorDatos(
            "El indice de los rendimientos no esta ordenado por fecha. Las ventanas "
            "moviles se toman por posicion, asi que el VaR de cada fecha se estimaria "
            "con observaciones posteriores a ella. Ordene con sort_index() antes de "
            "llamar, y revise de donde sale la serie."
        )
    if serie.index.has_duplicates:
        raise ErrorDatos(
            "El indice de los rendimientos tiene fechas repetidas: no hay una ventana "
            "de 'las N observaciones anteriores' bien definida."
        )
    nivel = validar_nivel(nivel)
    if metodo not in METODOS_BACKTEST:
        raise ErrorDatos(
            f"Metodo '{metodo}' desconocido. Disponibles: {', '.join(METODOS_BACKTEST)}."
        )
    if ventana < 30:
        raise ErrorDatos("La ventana debe tener al menos 30 observaciones.")
    if len(serie) <= ventana:
        raise ErrorDatos(
            f"Hacen falta mas de {ventana} observaciones para el backtest; hay {len(serie)}."
        )

    if metodo == "ewma":
        return _var_movil_ewma(serie, nivel, ventana, factor_decaimiento)
    if metodo == "garch":
        return _var_movil_garch(serie, nivel, ventana, frecuencia_reajuste)
    if metodo == "parametrico_t":
        return _var_movil_t(serie, nivel, ventana)
    if metodo == "cornish_fisher":
        return _var_movil_cornish_fisher(serie, nivel, ventana)

    from icm.riesgo.var import var

    estimaciones = {}
    valores = serie.to_numpy(float)
    for posicion in range(ventana, len(serie)):
        historia = pd.Series(valores[posicion - ventana : posicion])
        estimaciones[serie.index[posicion]] = var(historia, nivel, metodo)
    return pd.Series(estimaciones, name=f"var_{metodo}_{nivel:.0%}")


def _var_movil_t(serie: pd.Series, nivel: float, ventana: int) -> pd.Series:
    """VaR parametrico t, reaprovechando la estimacion de la ventana anterior.

    Ajustar una t por maxima verosimilitud en cada una de miles de ventanas es
    lo mas caro de todo el backtest. Como dos ventanas consecutivas se
    diferencian en una sola observacion, arrancar el optimizador en la solucion
    anterior lo acelera unas tres veces sin cambiar el resultado.
    """
    valores = serie.to_numpy(float)
    alfa = 1.0 - nivel
    estimacion: tuple[float, float, float] | None = None
    estimaciones: dict = {}

    for posicion in range(ventana, len(serie)):
        tramo = valores[posicion - ventana : posicion]
        if estimacion is None:
            estimacion = stats.t.fit(tramo)
        else:
            gl, loc, escala = estimacion
            estimacion = stats.t.fit(tramo, gl, loc=loc, scale=escala)
        gl, loc, escala = estimacion
        estimaciones[serie.index[posicion]] = -(loc + escala * stats.t.ppf(alfa, gl))
    return pd.Series(estimaciones, name=f"var_parametrico_t_{nivel:.0%}")


def _var_movil_cornish_fisher(serie: pd.Series, nivel: float, ventana: int) -> pd.Series:
    """VaR de Cornish-Fisher ventana a ventana, avisando una sola vez.

    Llamar a :func:`~icm.riesgo.var.var_cornish_fisher` en cada ventana emitiria
    un aviso por ventana cuando los momentos se salen de su dominio, lo que
    llena la pantalla sin aportar nada. Aqui se cuentan y se resume en un unico
    mensaje al final, que ademas dice en que proporcion de la muestra ocurre.
    """
    from icm.riesgo.var import (
        _cuantil_cornish_fisher,
        _momentos,
        dominio_valido_cornish_fisher,
    )

    valores = serie.to_numpy(float)
    alfa = 1.0 - nivel
    estimaciones: dict = {}
    fuera_de_dominio = 0

    for posicion in range(ventana, len(serie)):
        tramo = pd.Series(valores[posicion - ventana : posicion])
        media, desviacion, s, k = _momentos(tramo)
        if not dominio_valido_cornish_fisher(s, k):
            fuera_de_dominio += 1
        estimaciones[serie.index[posicion]] = -(
            media + desviacion * _cuantil_cornish_fisher(alfa, s, k)
        )

    if fuera_de_dominio:
        proporcion = fuera_de_dominio / len(estimaciones)
        warnings.warn(
            f"Cornish-Fisher se sale de su dominio de validez en {fuera_de_dominio} de "
            f"{len(estimaciones)} ventanas ({proporcion:.0%}): en ellas el valor no es "
            "un cuantil fiable y el backtest resultante no dice gran cosa. Con colas "
            "tan gruesas, compare con el metodo historico o con la t de Student.",
            UserWarning,
            stacklevel=3,
        )
    return pd.Series(estimaciones, name=f"var_cornish_fisher_{nivel:.0%}")


def _var_movil_ewma(
    serie: pd.Series, nivel: float, ventana: int, factor_decaimiento: float
) -> pd.Series:
    """VaR normal con volatilidad EWMA: el modelo RiskMetrics original."""
    from icm.riesgo.volatilidad import volatilidad_ewma

    volatilidad = volatilidad_ewma(serie, factor_decaimiento)
    cuantil = stats.norm.ppf(1.0 - nivel)
    estimaciones = -(volatilidad * cuantil)
    return estimaciones.iloc[ventana:].rename(f"var_ewma_{nivel:.0%}")


def _var_movil_garch(
    serie: pd.Series, nivel: float, ventana: int, frecuencia_reajuste: int
) -> pd.Series:
    """VaR condicional GARCH, reestimando cada ``frecuencia_reajuste`` dias."""
    from icm.riesgo.volatilidad import ajustar_garch

    if frecuencia_reajuste < 1:
        raise ErrorDatos("La frecuencia de reajuste debe ser de al menos un dia.")

    estimaciones: dict = {}
    ajuste = None
    cuantil = media = np.nan
    for posicion in range(ventana, len(serie)):
        ventana_actual = serie.iloc[posicion - ventana : posicion]
        if ajuste is None or (posicion - ventana) % frecuencia_reajuste == 0:
            ajuste = ajustar_garch(ventana_actual)
            cuantil = ajuste.cuantil_estandarizado(1.0 - nivel)
            media = float(ajuste.resultado.params.get("mu", 0.0)) / ajuste.escala
        # Los parametros se conservan entre reajustes, pero la varianza
        # condicional se actualiza cada dia con la recursion del modelo: de
        # otro modo el VaR no reaccionaria a las ultimas sesiones, que es
        # justamente lo que aporta un GARCH.
        volatilidad = _volatilidad_recursiva(ajuste, ventana_actual)
        estimaciones[serie.index[posicion]] = -(media + volatilidad * cuantil)
    return pd.Series(estimaciones, name=f"var_garch_{nivel:.0%}")


def _volatilidad_recursiva(ajuste, historia: pd.Series) -> float:
    """Aplica la recursion GARCH(1,1) a la historia sin reestimar parametros.

    Solo se usa con la especificacion simetrica; para las demas se recurre a la
    prevision del propio modelo, que queda algo desfasada entre reajustes.
    """
    params = ajuste.resultado.params
    if ajuste.especificacion != "garch" or not {"omega", "alpha[1]", "beta[1]"} <= set(params.index):
        prevision = ajuste.resultado.forecast(horizon=1, reindex=False)
        return float(np.sqrt(prevision.variance.iloc[-1, 0])) / ajuste.escala

    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    mu = float(params.get("mu", 0.0))

    valores = historia.to_numpy(float) * ajuste.escala
    varianza = omega / max(1.0 - alpha - beta, 1e-6)
    for valor in valores:
        varianza = omega + alpha * (valor - mu) ** 2 + beta * varianza
    return float(np.sqrt(varianza)) / ajuste.escala


def excepciones(rendimientos, var_estimado) -> pd.Series:
    """Marca los dias en que la perdida supero al VaR previsto.

    Ambas series se alinean por fecha, de modo que da igual el orden en que se
    hayan calculado, y el resultado sale **en orden cronologico**.

    Ese ultimo detalle no es cosmetico. Antes se devolvia en el orden del
    primer argumento, y ``contraste_christoffersen_independencia`` cuenta
    transiciones de un dia al siguiente sobre la secuencia que recibe: con las
    mismas 12 excepciones agrupadas en una racha, el estadistico pasaba de
    86,93 (p = 0,000, rechaza) a 0,74 (p = 0,388, no rechaza) solo porque el
    marco venia sin ordenar. Un modelo con las excepciones perfectamente
    agrupadas aprobaba el contraste de independencia.
    """
    serie = como_serie(rendimientos, "rendimientos")
    estimado = como_serie(var_estimado, "var")
    for nombre, indice in (("rendimientos", serie.index), ("var", estimado.index)):
        if indice.has_duplicates:
            raise ErrorDatos(
                f"El indice de '{nombre}' tiene fechas repetidas: al alinear por fecha "
                "no se sabria con cual emparejar."
            )
    comun = serie.index.intersection(estimado.index).sort_values()
    if len(comun) == 0:
        raise ErrorDatos("Los rendimientos y las estimaciones de VaR no comparten fechas.")
    fallos = (serie.loc[comun] < -estimado.loc[comun]).astype(int)
    fallos.name = "excepcion"
    return fallos


def contraste_kupiec(excepciones_serie, nivel: float = 0.99) -> dict[str, float]:
    """Contraste de cobertura incondicional (Kupiec, 1995).

    Hipotesis nula: la proporcion de excepciones es exactamente ``1 - nivel``.
    El estadistico es una razon de verosimilitudes que sigue una chi-cuadrado
    con un grado de libertad.
    """
    fallos = np.asarray(excepciones_serie, dtype=float)
    nivel = validar_nivel(nivel)
    n = len(fallos)
    if n == 0:
        raise ErrorDatos("No hay observaciones sobre las que contrastar.")
    x = float(fallos.sum())
    p = 1.0 - nivel
    proporcion = x / n

    if x == 0:
        estadistico = -2.0 * n * np.log(1.0 - p)
    elif np.isclose(proporcion, 1.0):
        estadistico = -2.0 * x * np.log(p)
    else:
        log_nula = x * np.log(p) + (n - x) * np.log(1.0 - p)
        log_alternativa = x * np.log(proporcion) + (n - x) * np.log(1.0 - proporcion)
        estadistico = -2.0 * (log_nula - log_alternativa)

    p_valor = float(1.0 - stats.chi2.cdf(estadistico, df=1))
    return {
        "observaciones": float(n),
        "excepciones": x,
        "excepciones_esperadas": n * p,
        "proporcion_observada": proporcion,
        "proporcion_esperada": p,
        "estadistico_lr": float(estadistico),
        "p_valor": p_valor,
        "rechaza_al_5pct": bool(p_valor < 0.05),
    }


def contraste_christoffersen_independencia(excepciones_serie) -> dict[str, float]:
    """Contraste de independencia de las excepciones (Christoffersen, 1998).

    Hipotesis nula: que hoy haya excepcion no depende de si la hubo ayer. Se
    rechaza cuando las excepciones se agrupan, senial tipica de un modelo que
    no actualiza la volatilidad.
    """
    fallos = np.asarray(excepciones_serie, dtype=int)
    if len(fallos) < 2:
        raise ErrorDatos("Hacen falta al menos 2 observaciones para contrastar la independencia.")

    # n_ij = transiciones del estado i al estado j.
    anterior, actual = fallos[:-1], fallos[1:]
    n00 = int(np.sum((anterior == 0) & (actual == 0)))
    n01 = int(np.sum((anterior == 0) & (actual == 1)))
    n10 = int(np.sum((anterior == 1) & (actual == 0)))
    n11 = int(np.sum((anterior == 1) & (actual == 1)))

    pi_01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _log(valor: float) -> float:
        return np.log(valor) if valor > 0 else 0.0

    log_nula = (n00 + n10) * _log(1.0 - pi) + (n01 + n11) * _log(pi)
    log_alternativa = (
        n00 * _log(1.0 - pi_01) + n01 * _log(pi_01) + n10 * _log(1.0 - pi_11) + n11 * _log(pi_11)
    )
    estadistico = -2.0 * (log_nula - log_alternativa)
    p_valor = float(1.0 - stats.chi2.cdf(estadistico, df=1))

    return {
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
        "prob_excepcion_tras_normal": pi_01,
        "prob_excepcion_tras_excepcion": pi_11,
        "estadistico_lr": float(estadistico),
        "p_valor": p_valor,
        "rechaza_al_5pct": bool(p_valor < 0.05),
    }


def contraste_christoffersen_condicional(excepciones_serie, nivel: float = 0.99) -> dict[str, float]:
    """Contraste conjunto de cobertura correcta e independencia.

    El estadistico es la suma de los de Kupiec y de independencia, y se
    distribuye como una chi-cuadrado con dos grados de libertad. Es el que se
    reporta cuando hay que dar una sola cifra.
    """
    kupiec = contraste_kupiec(excepciones_serie, nivel)
    independencia = contraste_christoffersen_independencia(excepciones_serie)
    estadistico = kupiec["estadistico_lr"] + independencia["estadistico_lr"]
    p_valor = float(1.0 - stats.chi2.cdf(estadistico, df=2))
    return {
        "estadistico_lr": float(estadistico),
        "p_valor": p_valor,
        "rechaza_al_5pct": bool(p_valor < 0.05),
        "lr_cobertura": kupiec["estadistico_lr"],
        "lr_independencia": independencia["estadistico_lr"],
    }


def zona_basilea(n_excepciones: int, n_observaciones: int = 250, nivel: float = 0.99) -> dict:
    """Semaforo supervisor de Basilea sobre el numero de excepciones.

    Las zonas se definen por la probabilidad acumulada binomial de observar
    tantas excepciones o menos si el modelo fuese correcto: verde por debajo
    del 95 %, amarilla hasta el 99,99 % y roja por encima. Para la ventana
    estandar de 250 dias al 99 % esto reproduce los umbrales de 0-4, 5-9 y 10
    o mas excepciones, y el recargo sobre el capital regulatorio.
    """
    p = 1.0 - validar_nivel(nivel)
    acumulada = float(stats.binom.cdf(n_excepciones, n_observaciones, p))

    if acumulada < 0.95:
        zona, recargo = "verde", 0.00
    elif acumulada < 0.9999:
        # El recargo crece con el numero de excepciones dentro de la zona.
        escala = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
        zona, recargo = "amarilla", escala.get(int(n_excepciones), 0.85)
    else:
        zona, recargo = "roja", 1.00

    return {
        "excepciones": int(n_excepciones),
        "observaciones": int(n_observaciones),
        "probabilidad_acumulada": acumulada,
        "zona": zona,
        "recargo_multiplicador": recargo,
    }


def diagnostico_es(rendimientos, var_estimado, es_estimado) -> dict[str, float]:
    """Compara la perdida media en los dias de excepcion con el ES previsto.

    No es un contraste formal con distribucion conocida, sino el diagnostico
    que se usa en la practica: si la perdida media observada en la cola supera
    de forma sistematica al ES previsto, el modelo se queda corto justo donde
    mas importa. El intervalo se obtiene por bootstrap sobre las excepciones,
    que suelen ser pocas.
    """
    serie = como_serie(rendimientos, "rendimientos")
    var_serie = como_serie(var_estimado, "var")
    es_serie = como_serie(es_estimado, "es") if not np.isscalar(es_estimado) else None

    fallos = excepciones(serie, var_serie)
    fechas_fallo = fallos[fallos == 1].index
    if len(fechas_fallo) == 0:
        raise ErrorDatos("No ha habido ninguna excepcion: no se puede diagnosticar el ES.")

    perdidas = -serie.loc[fechas_fallo]
    es_previsto = (
        float(es_serie.loc[fechas_fallo].mean()) if es_serie is not None else float(es_estimado)
    )

    generador = np.random.default_rng(42)
    muestras = generador.choice(perdidas.to_numpy(float), size=(2000, len(perdidas)), replace=True)
    medias = muestras.mean(axis=1)

    return {
        "excepciones": float(len(fechas_fallo)),
        "perdida_media_observada": float(perdidas.mean()),
        "es_previsto_medio": es_previsto,
        "ratio_observado_previsto": float(perdidas.mean() / es_previsto) if es_previsto else np.nan,
        "ic_inferior_95": float(np.quantile(medias, 0.025)),
        "ic_superior_95": float(np.quantile(medias, 0.975)),
        "es_dentro_del_ic": bool(
            np.quantile(medias, 0.025) <= es_previsto <= np.quantile(medias, 0.975)
        ),
    }


@dataclass
class ResultadoBacktest:
    """Todo lo que arroja el contraste de un modelo de VaR."""

    metodo: str
    nivel: float
    var: pd.Series
    excepciones: pd.Series
    kupiec: dict
    independencia: dict
    condicional: dict
    basilea: dict

    @property
    def resumen(self) -> pd.Series:
        """Las cifras clave en una sola fila, comoda para comparar modelos."""
        return pd.Series(
            {
                "metodo": self.metodo,
                "nivel": self.nivel,
                "observaciones": self.kupiec["observaciones"],
                "excepciones": self.kupiec["excepciones"],
                "excepciones_esperadas": self.kupiec["excepciones_esperadas"],
                "p_kupiec": self.kupiec["p_valor"],
                "p_independencia": self.independencia["p_valor"],
                "p_condicional": self.condicional["p_valor"],
                "zona_basilea": self.basilea["zona"],
                "var_medio": float(self.var.mean()),
                "modelo_valido": not self.condicional["rechaza_al_5pct"],
            },
            name=self.metodo,
        )


def backtest_var(
    rendimientos,
    nivel: float = 0.99,
    ventana: int = 250,
    metodo: str = "historico",
    **parametros,
) -> ResultadoBacktest:
    """Ejecuta el contraste completo de un metodo de VaR sobre una serie."""
    serie = como_serie(rendimientos, "rendimientos")
    var_serie = serie_var_movil(serie, nivel, ventana, metodo, **parametros)
    fallos = excepciones(serie, var_serie)

    return ResultadoBacktest(
        metodo=metodo,
        nivel=nivel,
        var=var_serie,
        excepciones=fallos,
        kupiec=contraste_kupiec(fallos, nivel),
        independencia=contraste_christoffersen_independencia(fallos),
        condicional=contraste_christoffersen_condicional(fallos, nivel),
        basilea=zona_basilea(int(fallos.sum()), len(fallos), nivel),
    )


def comparar_metodos_var(
    rendimientos,
    nivel: float = 0.99,
    ventana: int = 250,
    metodos=("historico", "parametrico_normal", "parametrico_t", "cornish_fisher", "ewma"),
    **parametros,
) -> pd.DataFrame:
    """Contrasta varios metodos sobre los mismos datos y los pone en una tabla.

    Es la forma honesta de elegir modelo: no por su sofisticacion, sino por el
    numero de excepciones que produce y por como se reparten en el tiempo.
    """
    filas = []
    for metodo in metodos:
        resultado = backtest_var(rendimientos, nivel, ventana, metodo, **parametros)
        filas.append(resultado.resumen)
    return pd.DataFrame(filas)
