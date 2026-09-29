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

import warnings
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


def _alinear_uniformes(uniformes, correlacion: pd.DataFrame) -> np.ndarray:
    """Ordena las columnas por NOMBRE antes de cruzarlas con la correlacion.

    La verosimilitud tomaba ``uniformes.to_numpy()`` y lo cruzaba con la
    matriz por POSICION, aunque los dos objetos llevan etiquetas. Bastaba que
    el marco llegase con las columnas en otro orden -- un ``reindex``, un
    ``groupby``, un ``merge`` -- para evaluar la verosimilitud de otro modelo:
    medido sobre tres activos, la gaussiana daba +788,19 con el orden natural
    y -3654,30 con el inverso, sobre exactamente los mismos datos. Eso cambia
    el signo del AIC en ``comparar_copulas``.
    """
    if not isinstance(uniformes, pd.DataFrame):
        return np.asarray(uniformes, dtype=float)
    esperadas = [str(c) for c in correlacion.columns]
    recibidas = {str(c): c for c in uniformes.columns}
    faltan = [c for c in esperadas if c not in recibidas]
    if faltan:
        raise ErrorDatos(
            f"Faltan columnas en las uniformes: {faltan}. La copula esta definida "
            f"sobre {esperadas}."
        )
    return uniformes[[recibidas[c] for c in esperadas]].to_numpy(float)


#: Por debajo de este autovalor la matriz ya no es una matriz de correlacion
#: con ruido numerico encima, sino otra cosa. La traza de una correlacion vale
#: d, asi que el autovalor medio es 1: -0,1 es una direccion del espacio con
#: varianza francamente negativa, no un error de redondeo.
AUTOVALOR_INADMISIBLE = -0.1


def _proyectar_definida_positiva(
    matriz: np.ndarray,
    minimo: float = 1e-8,
    nombre: str = "La matriz de correlacion",
) -> np.ndarray:
    """Corrige autovalores negativos y vuelve a normalizar la diagonal a 1.

    La transformacion de Kendall se aplica entrada a entrada, asi que el
    resultado no tiene por que ser una matriz de correlacion valida.

    La reparacion **no es gratis**: mueve las correlaciones que se le han dado
    y todo lo que salga despues -- simulaciones, VaR, dependencia de cola --
    describe la matriz reparada, no la que pidio quien llamo. Antes esto
    ocurria en silencio, asi que una matriz inventada a mano y claramente
    imposible producia numeros de riesgo con toda la apariencia de ser
    correctos. Ahora una desviacion pequenia avisa y una grande se rechaza.
    """
    autovalores, autovectores = np.linalg.eigh(matriz)
    menor = float(autovalores.min())
    if menor >= minimo:
        return matriz
    if menor < AUTOVALOR_INADMISIBLE:
        raise ErrorDatos(
            f"{nombre} no es semidefinida positiva ni por aproximacion: su menor "
            f"autovalor es {menor:.4f}. Esa combinacion de correlaciones no la "
            f"puede tener ningun conjunto de activos; repararla daria cifras de "
            f"riesgo de una matriz distinta de la pedida."
        )
    reparada = matriz
    for _ in range(10):
        autovalores, autovectores = np.linalg.eigh(reparada)
        if autovalores.min() >= minimo:
            break
        autovalores = np.clip(autovalores, 2.0 * minimo, None)
        reparada = autovectores @ np.diag(autovalores) @ autovectores.T
        escala = np.sqrt(np.diag(reparada))
        reparada = reparada / np.outer(escala, escala)
        np.fill_diagonal(reparada, 1.0)
    desvio = float(np.abs(reparada - matriz).max())
    warnings.warn(
        f"{nombre} tenia un autovalor {menor:.2e} y se ha proyectado sobre las "
        f"definidas positivas; la correlacion que mas se mueve lo hace en "
        f"{desvio:.2e}. Los resultados describen la matriz proyectada.",
        RuntimeWarning,
        stacklevel=3,
    )
    return reparada


def _correlacion_valida(correlacion, nombre: str = "La matriz de correlacion") -> pd.DataFrame:
    """Comprueba forma, simetria y diagonal, y devuelve ya la version proyectada.

    Se llama una sola vez, al construir la copula, para que la matriz que
    guarda el objeto sea exactamente la que usan la simulacion, la
    verosimilitud y la dependencia de cola. Cuando cada metodo proyectaba por
    su cuenta y ``dependencia_cola`` no lo hacia, el objeto declaraba una
    dependencia de cola de 0,5945 y simulaba otra de 0,164.
    """
    marco = correlacion if isinstance(correlacion, pd.DataFrame) else pd.DataFrame(correlacion)
    if marco.shape[0] != marco.shape[1]:
        raise ErrorDatos(f"{nombre} debe ser cuadrada, y es {marco.shape[0]}x{marco.shape[1]}.")
    if marco.shape[0] < 2:
        raise ErrorDatos(f"{nombre} necesita al menos dos activos.")
    if list(marco.index) != list(marco.columns):
        raise ErrorDatos(f"{nombre} debe llevar las mismas etiquetas en filas y columnas.")
    valores = marco.to_numpy(float)
    if not np.all(np.isfinite(valores)):
        raise ErrorDatos(f"{nombre} contiene valores no finitos.")
    if not np.allclose(valores, valores.T, atol=1e-8):
        raise ErrorDatos(f"{nombre} no es simetrica.")
    if not np.allclose(np.diag(valores), 1.0, atol=1e-8):
        raise ErrorDatos(f"{nombre} debe tener unos en la diagonal.")
    proyectada = _proyectar_definida_positiva(valores, nombre=nombre)
    return pd.DataFrame(proyectada, index=marco.index, columns=marco.columns)


@dataclass
class CopulaGaussiana:
    """Dependencia gaussiana: correlacion lineal y **sin** dependencia de cola."""

    correlacion: pd.DataFrame

    def __post_init__(self) -> None:
        self.correlacion = _correlacion_valida(self.correlacion)

    @property
    def activos(self) -> list[str]:
        return [str(c) for c in self.correlacion.columns]

    def simular_uniformes(self, n: int, semilla: int | None = 42) -> pd.DataFrame:
        generador = np.random.default_rng(semilla)
        factor = np.linalg.cholesky(self.correlacion.to_numpy(float))
        normales = generador.standard_normal((n, len(self.activos))) @ factor.T
        return pd.DataFrame(stats.norm.cdf(normales), columns=self.activos)

    def dependencia_cola(self) -> float:
        """Siempre 0: es la limitacion esencial de esta copula."""
        return 0.0

    def log_verosimilitud(self, uniformes) -> float:
        z = stats.norm.ppf(_alinear_uniformes(uniformes, self.correlacion))
        matriz = self.correlacion.to_numpy(float)
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

    def __post_init__(self) -> None:
        self.correlacion = _correlacion_valida(self.correlacion)
        self.grados_libertad = float(self.grados_libertad)
        if not np.isfinite(self.grados_libertad) or self.grados_libertad <= 2.0:
            raise ErrorDatos(
                f"Los grados de libertad de la copula t deben ser mayores que 2 "
                f"(recibido {self.grados_libertad}); por debajo la t no tiene "
                f"varianza y la dependencia de cola deja de estar definida."
            )

    @property
    def activos(self) -> list[str]:
        return [str(c) for c in self.correlacion.columns]

    def simular_uniformes(self, n: int, semilla: int | None = 42) -> pd.DataFrame:
        generador = np.random.default_rng(semilla)
        gl = self.grados_libertad
        factor = np.linalg.cholesky(self.correlacion.to_numpy(float))
        normales = generador.standard_normal((n, len(self.activos))) @ factor.T
        mezcla = generador.chisquare(gl, size=(n, 1)) / gl
        t_multivariante = normales / np.sqrt(mezcla)
        return pd.DataFrame(stats.t.cdf(t_multivariante, gl), columns=self.activos)

    def dependencia_cola(self, par: tuple[str, str] | None = None) -> float:
        """Probabilidad limite de que un activo hunda dado que el otro se hunde.

        Formula cerrada de la copula t bivariante. Si no se indica el par se
        usa la correlacion media fuera de la diagonal.
        """
        valores = self.correlacion.to_numpy(float)
        if par is None:
            fuera = valores[~np.eye(valores.shape[0], dtype=bool)]
            rho = float(np.mean(fuera))
        else:
            etiquetas = {str(c): i for i, c in enumerate(self.correlacion.columns)}
            faltan = [str(c) for c in par if str(c) not in etiquetas]
            if faltan:
                raise ErrorDatos(
                    f"Activos desconocidos para esta copula: {faltan}. Los que hay "
                    f"son {self.activos}."
                )
            rho = float(valores[etiquetas[str(par[0])], etiquetas[str(par[1])]])
        gl = self.grados_libertad
        # rho = -1 haria infinito el argumento; el limite es dependencia nula y
        # es mejor devolverlo que dividir por cero.
        rho = float(np.clip(rho, -1.0 + 1e-12, 1.0))
        argumento = -np.sqrt((gl + 1.0) * (1.0 - rho) / (1.0 + rho))
        return float(2.0 * stats.t.cdf(argumento, gl + 1.0))

    def log_verosimilitud(self, uniformes) -> float:
        return _log_verosimilitud_t(
            _alinear_uniformes(uniformes, self.correlacion),
            self.correlacion.to_numpy(float),
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
    if gl_minimos <= 2.0:
        raise ErrorDatos("El limite inferior de grados de libertad debe superar 2.")
    correlacion = correlacion_por_kendall(rendimientos)
    uniformes = _alinear_uniformes(pseudo_observaciones(rendimientos), correlacion)
    matriz = correlacion.to_numpy(float)

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

    esperados = [str(c) for c in marco.columns]
    if copula.activos != esperados:
        raise ErrorDatos(
            f"La copula esta definida sobre {copula.activos} y los rendimientos "
            f"traen {esperados}; alinee las columnas antes de simular."
        )

    # La copula devuelve las columnas con el nombre pasado por ``str``, asi que
    # se cruzan por POSICION y se reetiquetan con las columnas originales. Con
    # ``uniformes[activo]`` y un marco de columnas numericas -- una matriz de
    # rendimientos sin nombrar, por ejemplo -- esto era un KeyError: 0.
    uniformes = copula.simular_uniformes(n_simulaciones, semilla).to_numpy(float)
    simulados = pd.DataFrame(
        {
            activo: np.quantile(marco[activo].to_numpy(float), uniformes[:, i])
            for i, activo in enumerate(marco.columns)
        },
        columns=marco.columns,
    )
    simulados.attrs.update(marco.attrs)
    return simulados


def var_copula(
    rendimientos,
    pesos,
    nivel: float = 0.99,
    tipo: str = "t",
    n_simulaciones: int = 50_000,
    semilla: int | None = 42,
    copula: CopulaGaussiana | CopulaT | None = None,
) -> dict[str, float]:
    """VaR y ES de la cartera bajo el modelo de copula.

    Comparar el resultado con ``tipo="gaussiana"`` y con ``tipo="t"`` cuantifica
    cuanto riesgo se pasa por alto al ignorar la dependencia de cola.
    """
    marco = como_matriz(rendimientos, "rendimientos")
    if marco.attrs.get("tipo_rendimiento") is None:
        marco.attrs.update(getattr(rendimientos, "attrs", {}))
    vector_pesos = normalizar_pesos(pesos, marco.columns)
    simulados = simular_rendimientos_copula(marco, copula, n_simulaciones, tipo, semilla)

    # Un rendimiento logaritmico no se puede promediar entre activos: el de la
    # cartera no es la media ponderada de los logaritmos sino el logaritmo de la
    # media ponderada de los factores. Se pasa a simple, se pondera y se vuelve.
    if str(marco.attrs.get("tipo_rendimiento", "")) == "log":
        simulados = np.expm1(simulados)
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
        riesgo = var_copula(marco, pesos, nivel, nombre, n_simulaciones, semilla, copula)
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
