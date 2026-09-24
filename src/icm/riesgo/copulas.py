"""Copulas gaussiana y t de Student para modelizar la dependencia.

Una matriz de correlacion resume la dependencia media entre activos, pero no
dice nada de lo que ocurre en los extremos. Y precisamente ahi esta el
problema: en las crisis las correlaciones se disparan y todo cae a la vez.

Las copulas separan el problema en dos: por un lado el comportamiento
individual de cada activo (las marginales, que aqui se toman directamente de
los datos), por otro la estructura de dependencia. La copula gaussiana no
tiene dependencia de cola: por mucha correlacion que se le ponga, la
probabilidad de que dos activos sufran su peor dia a la vez tiende a cero. La
copula t si la tiene, y esa diferencia explica buena parte de lo que fallo en
los modelos de riesgo de 2008.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from icm.comun import ErrorDatos, como_matriz, normalizar_pesos, validar_nivel


def pseudo_observaciones(rendimientos) -> pd.DataFrame:
    """Transforma cada columna a rangos en (0, 1) mediante su funcion empirica.

    Se divide por ``n + 1`` en lugar de por ``n`` para que ningun valor caiga
    exactamente en 0 o en 1, donde los cuantiles de las distribuciones que se
    usan despues serian infinitos.
    """
    marco = como_matriz(rendimientos, "rendimientos")
    return marco.rank(method="average") / (len(marco) + 1.0)


def correlacion_por_kendall(rendimientos) -> pd.DataFrame:
    """Matriz de correlacion a partir de la tau de Kendall.

    Para las copulas elipticas se cumple ``rho = sin(pi * tau / 2)``. Esta ruta
    es preferible a la correlacion de Pearson porque solo depende del orden de
    las observaciones: no se deja arrastrar por un puniado de dias extremos ni
    presupone marginales normales.
    """
    marco = como_matriz(rendimientos, "rendimientos")
    tau = marco.corr(method="kendall").to_numpy(float)
    rho = np.sin(np.pi * tau / 2.0)
    np.fill_diagonal(rho, 1.0)
    rho = _proyectar_definida_positiva(rho)
    return pd.DataFrame(rho, index=marco.columns, columns=marco.columns)


def _proyectar_definida_positiva(matriz: np.ndarray, minimo: float = 1e-8) -> np.ndarray:
    """Corrige autovalores negativos y vuelve a normalizar la diagonal a 1.

    La transformacion de Kendall se aplica entrada a entrada, asi que el
    resultado no tiene por que ser una matriz de correlacion valida.
    """
    autovalores, autovectores = np.linalg.eigh(matriz)
    if autovalores.min() >= minimo:
        return matriz
    autovalores = np.clip(autovalores, minimo, None)
    reparada = autovectores @ np.diag(autovalores) @ autovectores.T
    escala = np.sqrt(np.diag(reparada))
    return reparada / np.outer(escala, escala)


@dataclass
class CopulaGaussiana:
    """Dependencia gaussiana: correlacion lineal y **sin** dependencia de cola."""

    correlacion: pd.DataFrame

    @property
    def activos(self) -> list[str]:
        return [str(c) for c in self.correlacion.columns]

    def simular_uniformes(self, n: int, semilla: int | None = 42) -> pd.DataFrame:
        generador = np.random.default_rng(semilla)
        factor = np.linalg.cholesky(_proyectar_definida_positiva(self.correlacion.to_numpy(float)))
        normales = generador.standard_normal((n, len(self.activos))) @ factor.T
        return pd.DataFrame(stats.norm.cdf(normales), columns=self.activos)

    def dependencia_cola(self) -> float:
        """Siempre 0: es la limitacion esencial de esta copula."""
        return 0.0

    def log_verosimilitud(self, uniformes: pd.DataFrame) -> float:
        z = stats.norm.ppf(uniformes.to_numpy(float))
        matriz = _proyectar_definida_positiva(self.correlacion.to_numpy(float))
        inversa = np.linalg.inv(matriz)
        signo, log_det = np.linalg.slogdet(matriz)
        if signo <= 0:  # pragma: no cover - imposible tras la proyeccion
            raise ErrorDatos("La matriz de correlacion no es definida positiva.")
        forma = np.einsum("ij,jk,ik->i", z, inversa - np.eye(matriz.shape[0]), z)
        return float(np.sum(-0.5 * log_det - 0.5 * forma))


@dataclass
class CopulaT:
    """Dependencia t de Student: misma correlacion, pero con colas conjuntas.

    Los grados de libertad gobiernan la dependencia de cola: cuanto menores,
    mayor es la probabilidad de que los activos se hundan a la vez. Por encima
    de 30 el comportamiento es practicamente el de la gaussiana.
    """

    correlacion: pd.DataFrame
    grados_libertad: float

    @property
    def activos(self) -> list[str]:
        return [str(c) for c in self.correlacion.columns]

    def simular_uniformes(self, n: int, semilla: int | None = 42) -> pd.DataFrame:
        generador = np.random.default_rng(semilla)
        gl = self.grados_libertad
        factor = np.linalg.cholesky(_proyectar_definida_positiva(self.correlacion.to_numpy(float)))
        normales = generador.standard_normal((n, len(self.activos))) @ factor.T
        mezcla = generador.chisquare(gl, size=(n, 1)) / gl
        t_multivariante = normales / np.sqrt(mezcla)
        return pd.DataFrame(stats.t.cdf(t_multivariante, gl), columns=self.activos)

    def dependencia_cola(self, par: tuple[str, str] | None = None) -> float:
        """Probabilidad limite de que un activo hunda dado que el otro se hunde.

        Formula cerrada de la copula t bivariante. Si no se indica el par se
        usa la correlacion media fuera de la diagonal.
        """
        if par is None:
            valores = self.correlacion.to_numpy(float)
            fuera = valores[~np.eye(valores.shape[0], dtype=bool)]
            rho = float(np.mean(fuera))
        else:
            rho = float(self.correlacion.loc[par[0], par[1]])
        gl = self.grados_libertad
        argumento = -np.sqrt((gl + 1.0) * (1.0 - rho) / (1.0 + rho))
        return float(2.0 * stats.t.cdf(argumento, gl + 1.0))

    def log_verosimilitud(self, uniformes: pd.DataFrame) -> float:
        return _log_verosimilitud_t(
            uniformes.to_numpy(float),
            _proyectar_definida_positiva(self.correlacion.to_numpy(float)),
            self.grados_libertad,
        )


def _log_verosimilitud_t(uniformes: np.ndarray, correlacion: np.ndarray, gl: float) -> float:
    """Log-verosimilitud de la copula t: densidad conjunta menos las marginales."""
    d = correlacion.shape[0]
    x = stats.t.ppf(uniformes, gl)
    if not np.all(np.isfinite(x)):
        return -np.inf
    inversa = np.linalg.inv(correlacion)
    signo, log_det = np.linalg.slogdet(correlacion)
    if signo <= 0:
        return -np.inf

    forma = np.einsum("ij,jk,ik->i", x, inversa, x)
    constante = (
        special.gammaln((gl + d) / 2.0)
        + (d - 1.0) * special.gammaln(gl / 2.0)
        - d * special.gammaln((gl + 1.0) / 2.0)
        - 0.5 * log_det
    )
    conjunta = -(gl + d) / 2.0 * np.log1p(forma / gl)
    marginales = np.sum((gl + 1.0) / 2.0 * np.log1p(x**2 / gl), axis=1)
    return float(np.sum(constante + conjunta + marginales))


def ajustar_copula_gaussiana(rendimientos) -> CopulaGaussiana:
    """Estima una copula gaussiana por el metodo de los rangos."""
    return CopulaGaussiana(correlacion=correlacion_por_kendall(rendimientos))


def ajustar_copula_t(
    rendimientos,
    gl_minimos: float = 2.1,
    gl_maximos: float = 60.0,
) -> CopulaT:
    """Estima una copula t: correlacion por rangos y grados de libertad por maxima verosimilitud.

    Es el procedimiento en dos etapas habitual: fijada la matriz de
    correlacion, se busca el unico parametro restante maximizando la
    verosimilitud, lo que reduce un problema multidimensional a uno escalar y
    bien condicionado.
    """
    correlacion = correlacion_por_kendall(rendimientos)
    uniformes = pseudo_observaciones(rendimientos).to_numpy(float)
    matriz = _proyectar_definida_positiva(correlacion.to_numpy(float))

    def objetivo(gl: float) -> float:
        return -_log_verosimilitud_t(uniformes, matriz, float(gl))

    resultado = optimize.minimize_scalar(
        objetivo, bounds=(gl_minimos, gl_maximos), method="bounded"
    )
    gl = float(resultado.x) if resultado.success else float(gl_maximos)
    return CopulaT(correlacion=correlacion, grados_libertad=gl)


def dependencia_cola_empirica(
    rendimientos,
    cuantil: float = 0.05,
    cola: str = "inferior",
) -> pd.DataFrame:
    """Frecuencia observada con que dos activos entran juntos en su cola.

    Es la contrapartida muestral de :meth:`CopulaT.dependencia_cola`. Comparar
    ambas cifras es la forma mas directa de ver si la copula elegida recoge
    bien el comportamiento conjunto en los extremos.
    """
    if cola not in {"inferior", "superior"}:
        raise ErrorDatos("La cola debe ser 'inferior' o 'superior'.")
    cuantil = validar_nivel(cuantil)
    uniformes = pseudo_observaciones(rendimientos)
    if cola == "superior":
        uniformes = 1.0 - uniformes

    columnas = list(uniformes.columns)
    salida = pd.DataFrame(np.eye(len(columnas)), index=columnas, columns=columnas)
    for i, primera in enumerate(columnas):
        for segunda in columnas[i + 1 :]:
            extremo_segunda = uniformes[segunda] <= cuantil
            n_condicionante = int(extremo_segunda.sum())
            if n_condicionante == 0:
                valor = np.nan
            else:
                conjunto = int(((uniformes[primera] <= cuantil) & extremo_segunda).sum())
                valor = conjunto / n_condicionante
            salida.loc[primera, segunda] = valor
            salida.loc[segunda, primera] = valor
    return salida


def simular_rendimientos_copula(
    rendimientos,
    copula: CopulaGaussiana | CopulaT | None = None,
    n_simulaciones: int = 10_000,
    tipo: str = "t",
    semilla: int | None = 42,
) -> pd.DataFrame:
    """Simula rendimientos con la dependencia de la copula y las marginales reales.

    Enfoque semiparametrico: la copula aporta la estructura de dependencia y
    los cuantiles empiricos de cada activo aportan las marginales, sin imponer
    ninguna forma funcional a estas ultimas.
    """
    marco = como_matriz(rendimientos, "rendimientos")
    if copula is None:
        if tipo == "t":
            copula = ajustar_copula_t(marco)
        elif tipo == "gaussiana":
            copula = ajustar_copula_gaussiana(marco)
        else:
            raise ErrorDatos(f"Tipo de copula '{tipo}' desconocido: use 't' o 'gaussiana'.")

    uniformes = copula.simular_uniformes(n_simulaciones, semilla)
    simulados = {
        activo: np.quantile(marco[activo].to_numpy(float), uniformes[activo].to_numpy(float))
        for activo in marco.columns
    }
    return pd.DataFrame(simulados, columns=marco.columns)


def var_copula(
    rendimientos,
    pesos,
    nivel: float = 0.99,
    tipo: str = "t",
    n_simulaciones: int = 50_000,
    semilla: int | None = 42,
) -> dict[str, float]:
    """VaR y ES de la cartera bajo el modelo de copula.

    Comparar el resultado con ``tipo="gaussiana"`` y con ``tipo="t"`` cuantifica
    cuanto riesgo se pasa por alto al ignorar la dependencia de cola.
    """
    marco = como_matriz(rendimientos, "rendimientos")
    vector_pesos = normalizar_pesos(pesos, marco.columns)
    simulados = simular_rendimientos_copula(marco, None, n_simulaciones, tipo, semilla)
    cartera = simulados.mul(vector_pesos, axis=1).sum(axis=1)

    from icm.riesgo.montecarlo import es_montecarlo, var_montecarlo

    return {
        "var": var_montecarlo(cartera, nivel),
        "es": es_montecarlo(cartera, nivel),
        "nivel": float(nivel),
        "tipo": tipo,
    }


def comparar_copulas(
    rendimientos,
    pesos,
    nivel: float = 0.99,
    n_simulaciones: int = 50_000,
    semilla: int | None = 42,
) -> pd.DataFrame:
    """Enfrenta la copula gaussiana y la t sobre los mismos datos.

    Devuelve verosimilitud, criterios de informacion y las cifras de riesgo de
    cada una, que es la comparacion que justifica elegir una u otra.
    """
    marco = como_matriz(rendimientos, "rendimientos")
    uniformes = pseudo_observaciones(marco)
    n_parametros_corr = marco.shape[1] * (marco.shape[1] - 1) / 2

    gaussiana = ajustar_copula_gaussiana(marco)
    t_student = ajustar_copula_t(marco)

    filas = {}
    for nombre, copula, n_extra in (
        ("gaussiana", gaussiana, 0),
        ("t", t_student, 1),
    ):
        log_ver = copula.log_verosimilitud(uniformes)
        k = n_parametros_corr + n_extra
        riesgo = var_copula(marco, pesos, nivel, nombre, n_simulaciones, semilla)
        filas[nombre] = {
            "log_verosimilitud": log_ver,
            "n_parametros": k,
            "aic": 2 * k - 2 * log_ver,
            "bic": k * np.log(len(marco)) - 2 * log_ver,
            "dependencia_cola": (
                t_student.dependencia_cola() if nombre == "t" else gaussiana.dependencia_cola()
            ),
            "grados_libertad": t_student.grados_libertad if nombre == "t" else np.nan,
            f"var_{nivel:.0%}": riesgo["var"],
            f"es_{nivel:.0%}": riesgo["es"],
        }
    return pd.DataFrame(filas).T
