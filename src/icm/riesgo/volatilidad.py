"""Modelos de volatilidad condicional: media movil, EWMA y GARCH.

La volatilidad de los mercados no es constante: se agrupa en rachas. Suponerla
fija, como hace el VaR parametrico clasico, infravalora el riesgo en periodos
turbulentos y lo exagera en los tranquilos. Los modelos de este modulo estiman
la volatilidad **del dia siguiente** condicionada a lo ocurrido hasta hoy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos, como_serie, validar_nivel

#: Factor de decaimiento de RiskMetrics para datos diarios.
FACTOR_RISKMETRICS = 0.94

#: ``arch`` trabaja mucho mejor con rendimientos en puntos porcentuales: con
#: valores del orden de 0.01 el optimizador tiene problemas de escala.
ESCALA_ARCH = 100.0


def volatilidad_movil(rendimientos, ventana: int = 21, anualizar: bool = False) -> pd.Series:
    """Desviacion tipica en una ventana movil.

    Es el estimador mas simple y el mas facil de explicar, pero tiene un
    defecto conocido: cuando un dia extremo sale de la ventana, la volatilidad
    cae de golpe sin que haya pasado nada en el mercado.
    """
    serie = como_serie(rendimientos, "rendimientos")
    if ventana < 2:
        raise ErrorDatos("La ventana debe ser de al menos 2 periodos.")
    volatilidad = serie.rolling(ventana).std(ddof=1)
    if anualizar:
        volatilidad = volatilidad * np.sqrt(DIAS_HABILES_ANIO)
    volatilidad.name = f"volatilidad_movil_{ventana}"
    return volatilidad.dropna()


def volatilidad_ewma(
    rendimientos,
    factor_decaimiento: float = FACTOR_RISKMETRICS,
    anualizar: bool = False,
) -> pd.Series:
    """Volatilidad con ponderacion exponencial, al estilo RiskMetrics.

    Cada observacion pesa ``(1 - lambda) * lambda**k``, de modo que la
    influencia de un dia se desvanece de forma suave en lugar de desaparecer
    de golpe. Con ``lambda = 0.94`` la vida media es de unos 11 dias.

    La serie devuelta esta desplazada un periodo: el valor de la fecha ``t`` es
    la prediccion para ``t`` hecha con informacion hasta ``t-1``, que es lo que
    hace falta para calcular un VaR utilizable y para poder contrastarlo.
    """
    serie = como_serie(rendimientos, "rendimientos")
    if not 0.0 < factor_decaimiento < 1.0:
        raise ErrorDatos("El factor de decaimiento debe estar entre 0 y 1.")

    valores = serie.to_numpy(dtype=float)
    varianzas = np.empty_like(valores)
    # Se arranca con la varianza muestral de las primeras observaciones.
    arranque = min(len(valores), 30)
    varianzas[0] = float(np.var(valores[:arranque], ddof=1)) if arranque > 1 else valores[0] ** 2
    for t in range(1, len(valores)):
        varianzas[t] = (
            factor_decaimiento * varianzas[t - 1]
            + (1.0 - factor_decaimiento) * valores[t - 1] ** 2
        )

    volatilidad = pd.Series(np.sqrt(varianzas), index=serie.index, name="volatilidad_ewma")
    if anualizar:
        volatilidad = volatilidad * np.sqrt(DIAS_HABILES_ANIO)
    return volatilidad


@dataclass
class AjusteGarch:
    """Resultado de estimar un modelo GARCH sobre una serie de rendimientos."""

    resultado: object  # arch.univariate.base.ARCHModelResult
    escala: float
    especificacion: str
    distribucion: str
    indice: pd.Index = field(repr=False)

    @property
    def parametros(self) -> pd.Series:
        """Parametros estimados, en la escala original de los rendimientos."""
        return self.resultado.params

    @property
    def resumen(self) -> str:
        """Salida textual completa de la estimacion."""
        return str(self.resultado.summary())

    @property
    def persistencia(self) -> float:
        """Suma de los coeficientes alfa y beta.

        Cuanto mas cerca de 1, mas lentamente se disipan los choques de
        volatilidad. Un valor por encima de 1 indica un modelo no estacionario
        y, casi siempre, que algo va mal en la estimacion.

        En el EGARCH la persistencia la determina unicamente el coeficiente
        beta, porque la ecuacion esta escrita sobre el logaritmo de la varianza
        y los terminos de choque no se acumulan igual.
        """
        params = self.resultado.params
        if self.especificacion == "egarch":
            return float(sum(params[k] for k in params.index if k.startswith("beta")))
        total = 0.0
        for clave in params.index:
            if clave.startswith(("alpha", "beta")):
                total += float(params[clave])
            elif clave.startswith("gamma"):
                # El termino asimetrico de GJR solo actua en los choques
                # negativos, que bajo simetria son la mitad de los casos.
                total += 0.5 * float(params[clave])
        return float(total)

    def volatilidad_condicional(self, anualizar: bool = False) -> pd.Series:
        """Volatilidad estimada dentro de la muestra, periodo a periodo."""
        serie = pd.Series(
            np.asarray(self.resultado.conditional_volatility, dtype=float) / self.escala,
            index=self.indice,
            name="volatilidad_garch",
        ).dropna()
        return serie * np.sqrt(DIAS_HABILES_ANIO) if anualizar else serie

    def cuantil_estandarizado(self, probabilidad: float) -> float:
        """Cuantil de la distribucion de las innovaciones, ya estandarizada."""
        distribucion = self.resultado.model.distribution
        n_parametros = int(distribucion.num_params)
        parametros = (
            self.resultado.params.to_numpy()[-n_parametros:] if n_parametros else np.array([])
        )
        return float(np.asarray(distribucion.ppf(probabilidad, parametros)).ravel()[0])


def ajustar_garch(
    rendimientos,
    especificacion: str = "garch",
    p: int = 1,
    q: int = 1,
    distribucion: str = "t",
    media: str = "constante",
) -> AjusteGarch:
    """Estima un modelo de volatilidad condicional de la familia GARCH.

    Parametros
    ----------
    especificacion:
        ``"garch"`` es el modelo simetrico habitual; ``"gjr"`` aniade un termino
        que deja que las caidas eleven mas la volatilidad que las subidas (el
        efecto apalancamiento, muy marcado en renta variable); ``"egarch"``
        modeliza el logaritmo de la varianza, lo que garantiza positividad sin
        imponer restricciones a los parametros.
    distribucion:
        ``"t"`` (por defecto) o ``"normal"``, ``"skewt"``, ``"ged"``. Con datos
        financieros la normal casi siempre se queda corta en las colas.
    media:
        ``"constante"``, ``"cero"`` o ``"ar"`` (autorregresiva de orden 1).
    """
    serie = como_serie(rendimientos, "rendimientos")
    if len(serie) < 100:
        raise ErrorDatos(
            f"Estimar un GARCH con {len(serie)} observaciones no es fiable; "
            "use al menos 100, y preferiblemente varios cientos."
        )

    from arch import arch_model

    modelos = {"garch": "GARCH", "gjr": "GARCH", "egarch": "EGARCH"}
    if especificacion not in modelos:
        raise ErrorDatos(
            f"Especificacion '{especificacion}' desconocida: use 'garch', 'gjr' o 'egarch'."
        )
    medias = {"constante": "Constant", "cero": "Zero", "ar": "AR"}
    if media not in medias:
        raise ErrorDatos(f"Modelo de media '{media}' desconocido: use 'constante', 'cero' o 'ar'.")
    distribuciones = {"t": "t", "normal": "normal", "skewt": "skewt", "ged": "ged"}
    if distribucion not in distribuciones:
        raise ErrorDatos(f"Distribucion '{distribucion}' desconocida.")

    modelo = arch_model(
        serie * ESCALA_ARCH,
        mean=medias[media],
        lags=1 if media == "ar" else 0,
        vol=modelos[especificacion],
        p=p,
        o=1 if especificacion == "gjr" else 0,
        q=q,
        dist=distribuciones[distribucion],
        rescale=False,
    )
    resultado = modelo.fit(disp="off", show_warning=False)
    return AjusteGarch(
        resultado=resultado,
        escala=ESCALA_ARCH,
        especificacion=especificacion,
        distribucion=distribucion,
        indice=serie.index,
    )


def prevision_volatilidad(ajuste: AjusteGarch, horizonte: int = 1) -> pd.Series:
    """Volatilidad prevista para los proximos ``horizonte`` periodos."""
    if horizonte < 1:
        raise ErrorDatos("El horizonte debe ser de al menos un periodo.")
    prevision = ajuste.resultado.forecast(horizon=horizonte, reindex=False)
    varianzas = np.asarray(prevision.variance.iloc[-1], dtype=float)
    return pd.Series(
        np.sqrt(varianzas) / ajuste.escala,
        index=pd.RangeIndex(1, horizonte + 1, name="periodos_adelante"),
        name="volatilidad_prevista",
    )


def var_garch(
    rendimientos,
    nivel: float = 0.99,
    horizonte: int = 1,
    ajuste: AjusteGarch | None = None,
    **parametros_ajuste,
) -> float:
    """VaR condicional: usa la volatilidad prevista por el GARCH, no la media historica.

    Es la diferencia practica mas importante frente al VaR historico: tras una
    semana turbulenta este VaR sube de inmediato, mientras que el historico
    tarda tantos dias como tenga la ventana en reaccionar.

    Para horizontes de varios dias se acumulan la media y la varianza previstas,
    lo que supone innovaciones independientes: el GARCH modeliza la varianza
    cambiante, pero no una autocorrelacion de los rendimientos.
    """
    nivel = validar_nivel(nivel)
    ajuste = ajuste if ajuste is not None else ajustar_garch(rendimientos, **parametros_ajuste)

    prevision = ajuste.resultado.forecast(horizon=horizonte, reindex=False)
    media_acumulada = float(np.sum(np.asarray(prevision.mean.iloc[-1], dtype=float))) / ajuste.escala
    varianza_acumulada = float(
        np.sum(np.asarray(prevision.variance.iloc[-1], dtype=float))
    ) / ajuste.escala**2
    desviacion = float(np.sqrt(max(varianza_acumulada, 0.0)))

    cuantil = ajuste.cuantil_estandarizado(1.0 - nivel)
    return float(-(media_acumulada + desviacion * cuantil))


def es_garch(
    rendimientos,
    nivel: float = 0.99,
    horizonte: int = 1,
    ajuste: AjusteGarch | None = None,
    n_tramos: int = 1000,
    **parametros_ajuste,
) -> float:
    """ES condicional, integrando el VaR del GARCH sobre la cola."""
    nivel = validar_nivel(nivel)
    ajuste = ajuste if ajuste is not None else ajustar_garch(rendimientos, **parametros_ajuste)
    alfa = 1.0 - nivel
    probabilidades = np.linspace(alfa / n_tramos, alfa, n_tramos)

    prevision = ajuste.resultado.forecast(horizon=horizonte, reindex=False)
    media = float(np.sum(np.asarray(prevision.mean.iloc[-1], dtype=float))) / ajuste.escala
    desviacion = float(
        np.sqrt(max(np.sum(np.asarray(prevision.variance.iloc[-1], dtype=float)), 0.0))
    ) / ajuste.escala

    cuantiles = np.array([ajuste.cuantil_estandarizado(p) for p in probabilidades])
    return float(-(media + desviacion * cuantiles.mean()))


def comparar_modelos_volatilidad(
    rendimientos,
    especificaciones=("garch", "gjr", "egarch"),
    distribucion: str = "t",
) -> pd.DataFrame:
    """Ajusta varias especificaciones y las compara por criterios de informacion.

    AIC y BIC penalizan el numero de parametros: entre dos modelos que explican
    los datos igual de bien, gana el mas sencillo. BIC penaliza mas.
    """
    filas = {}
    for especificacion in especificaciones:
        ajuste = ajustar_garch(rendimientos, especificacion, distribucion=distribucion)
        filas[especificacion] = {
            "log_verosimilitud": float(ajuste.resultado.loglikelihood),
            "aic": float(ajuste.resultado.aic),
            "bic": float(ajuste.resultado.bic),
            "n_parametros": int(len(ajuste.resultado.params)),
            "persistencia": ajuste.persistencia,
        }
    return pd.DataFrame(filas).T.sort_values("bic")
