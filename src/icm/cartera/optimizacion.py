"""Construccion de carteras optimas.

Convencion de unidades: ``mu`` (rentabilidades esperadas) y ``cov`` (matriz de
covarianzas) deben estar expresadas en la **misma** frecuencia. Lo habitual es
anualizar ambas antes de optimizar; los pesos resultantes son los mismos, pero
las cifras de rentabilidad y volatilidad que devuelven estas funciones heredan
la frecuencia de la entrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from icm.comun import ErrorDatos, ErrorOptimizacion, normalizar_pesos


@dataclass(frozen=True)
class Restricciones:
    """Limites que debe respetar la cartera.

    Atributos
    ---------
    permitir_cortos:
        Si es ``False`` (lo habitual en un fondo long-only) los pesos no pueden
        ser negativos.
    peso_maximo:
        Tope por activo, para evitar carteras concentradas en un solo nombre.
    peso_minimo:
        Suelo por activo. Solo tiene sentido con ``permitir_cortos=True`` o
        para forzar una presencia minima.
    """

    permitir_cortos: bool = False
    peso_maximo: float | None = None
    peso_minimo: float | None = None

    def limites(self, n_activos: int) -> list[tuple[float, float]]:
        inferior = -1.0 if self.permitir_cortos else 0.0
        superior = 1.0
        if self.peso_minimo is not None:
            inferior = float(self.peso_minimo)
        if self.peso_maximo is not None:
            superior = float(self.peso_maximo)
        if superior * n_activos < 1.0 - 1e-9:
            raise ErrorDatos(
                f"Con un peso maximo de {superior} y {n_activos} activos los pesos "
                "no pueden sumar 1."
            )
        if inferior > superior:
            raise ErrorDatos("El peso minimo no puede superar al maximo.")
        return [(inferior, superior)] * n_activos


def _preparar(mu, cov) -> tuple[np.ndarray | None, np.ndarray, list[str]]:
    """Alinea rentabilidades esperadas y covarianzas y valida su forma."""
    marco_cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(np.asarray(cov, float))
    activos = [str(c) for c in marco_cov.columns]
    matriz = marco_cov.to_numpy(dtype=float)
    if matriz.shape[0] != matriz.shape[1]:
        raise ErrorDatos("La matriz de covarianzas debe ser cuadrada.")
    if not np.allclose(matriz, matriz.T, atol=1e-10):
        raise ErrorDatos("La matriz de covarianzas debe ser simetrica.")
    if np.min(np.linalg.eigvalsh(matriz)) < -1e-10:
        raise ErrorDatos(
            "La matriz de covarianzas no es semidefinida positiva. "
            "Pruebe con el estimador 'ledoit_wolf'."
        )

    if mu is None:
        return None, matriz, activos
    vector = mu.reindex(activos).to_numpy(float) if isinstance(mu, pd.Series) else np.asarray(mu, float)
    if vector.shape != (len(activos),):
        raise ErrorDatos("Las rentabilidades esperadas no encajan con la matriz de covarianzas.")
    if np.isnan(vector).any():
        raise ErrorDatos("Hay rentabilidades esperadas ausentes para algun activo.")
    return vector, matriz, activos


def rentabilidad_cartera(pesos, mu) -> float:
    """Rentabilidad esperada de la cartera."""
    vector = normalizar_pesos(pesos).to_numpy(float) if not isinstance(pesos, np.ndarray) else pesos
    return float(vector @ np.asarray(mu, float))


def volatilidad_cartera(pesos, cov) -> float:
    """Desviacion tipica de la cartera."""
    vector = normalizar_pesos(pesos).to_numpy(float) if not isinstance(pesos, np.ndarray) else pesos
    matriz = cov.to_numpy(float) if isinstance(cov, pd.DataFrame) else np.asarray(cov, float)
    varianza = float(vector @ matriz @ vector)
    return float(np.sqrt(max(varianza, 0.0)))


def contribuciones_riesgo(pesos, cov) -> pd.DataFrame:
    """Descompone la volatilidad de la cartera activo a activo.

    La contribucion marginal es la derivada de la volatilidad respecto al peso;
    multiplicada por el propio peso da la contribucion absoluta, y estas suman
    exactamente la volatilidad total (teorema de Euler, porque la volatilidad
    es homogenea de grado uno en los pesos).
    """
    marco_cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(np.asarray(cov, float))
    serie_pesos = normalizar_pesos(pesos, marco_cov.columns)
    vector = serie_pesos.to_numpy(float)
    matriz = marco_cov.to_numpy(float)

    volatilidad = float(np.sqrt(max(vector @ matriz @ vector, 0.0)))
    if volatilidad == 0:
        raise ErrorDatos("La cartera tiene volatilidad nula: no hay riesgo que repartir.")
    marginal = (matriz @ vector) / volatilidad
    absoluta = vector * marginal
    return pd.DataFrame(
        {
            "peso": serie_pesos,
            "contribucion_marginal": marginal,
            "contribucion_absoluta": absoluta,
            "contribucion_relativa": absoluta / volatilidad,
        },
        index=marco_cov.columns,
    )


def _optimizar(
    funcion_objetivo,
    n_activos: int,
    restricciones: Restricciones,
    restricciones_extra: list[dict] | None = None,
    inicio: np.ndarray | None = None,
) -> np.ndarray:
    """Envoltorio de SLSQP con la restriccion de presupuesto siempre presente."""
    limites = restricciones.limites(n_activos)
    condiciones = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if restricciones_extra:
        condiciones.extend(restricciones_extra)
    punto_inicial = inicio if inicio is not None else np.full(n_activos, 1.0 / n_activos)

    resultado = optimize.minimize(
        funcion_objetivo,
        punto_inicial,
        method="SLSQP",
        bounds=limites,
        constraints=condiciones,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not resultado.success:
        raise ErrorOptimizacion(f"El optimizador no ha convergido: {resultado.message}")
    pesos = np.asarray(resultado.x, dtype=float)
    pesos[np.abs(pesos) < 1e-10] = 0.0
    return pesos / pesos.sum()


def cartera_pesos_iguales(activos) -> pd.Series:
    """Cartera 1/N: sencilla, sin estimaciones y sorprendentemente dificil de batir."""
    lista = list(activos.columns) if isinstance(activos, pd.DataFrame) else list(activos)
    if not lista:
        raise ErrorDatos("No se ha indicado ningun activo.")
    return pd.Series(1.0 / len(lista), index=lista, name="pesos_iguales")


def minima_varianza(cov, restricciones: Restricciones | None = None) -> pd.Series:
    """Cartera de minima volatilidad, sin usar rentabilidades esperadas.

    Es la mas robusta de las carteras optimas precisamente por eso: estimar
    rentabilidades esperadas es mucho mas ruidoso que estimar covarianzas.
    """
    restricciones = restricciones or Restricciones()
    _, matriz, activos = _preparar(None, cov)
    pesos = _optimizar(
        lambda w: float(w @ matriz @ w),
        len(activos),
        restricciones,
    )
    return pd.Series(pesos, index=activos, name="minima_varianza")


def maximo_sharpe(
    mu,
    cov,
    tasa_libre_riesgo: float = 0.0,
    restricciones: Restricciones | None = None,
) -> pd.Series:
    """Cartera tangente: la que maximiza el exceso por unidad de volatilidad."""
    restricciones = restricciones or Restricciones()
    vector, matriz, activos = _preparar(mu, cov)

    def objetivo(w: np.ndarray) -> float:
        volatilidad = np.sqrt(max(w @ matriz @ w, 1e-16))
        return -float((w @ vector - tasa_libre_riesgo) / volatilidad)

    pesos = _optimizar(objetivo, len(activos), restricciones)
    return pd.Series(pesos, index=activos, name="maximo_sharpe")


def minima_varianza_con_objetivo(
    mu,
    cov,
    rentabilidad_objetivo: float,
    restricciones: Restricciones | None = None,
) -> pd.Series:
    """Cartera de minima varianza sujeta a alcanzar una rentabilidad dada."""
    restricciones = restricciones or Restricciones()
    vector, matriz, activos = _preparar(mu, cov)
    condicion = {
        "type": "eq",
        "fun": lambda w: float(w @ vector - rentabilidad_objetivo),
    }
    pesos = _optimizar(
        lambda w: float(w @ matriz @ w),
        len(activos),
        restricciones,
        [condicion],
    )
    return pd.Series(pesos, index=activos, name=f"objetivo_{rentabilidad_objetivo:.4f}")


def paridad_riesgo(cov, presupuesto=None, restricciones: Restricciones | None = None) -> pd.Series:
    """Cartera en la que cada activo aporta la misma parte del riesgo total.

    A diferencia de 1/N, que iguala el capital, aqui se iguala la contribucion
    al riesgo: los activos volatiles pesan menos. Con ``presupuesto`` se puede
    pedir un reparto distinto del riesgo (por ejemplo 60/40 en vez de 50/50).
    """
    restricciones = restricciones or Restricciones(permitir_cortos=False)
    if restricciones.permitir_cortos:
        raise ErrorDatos("La paridad de riesgo requiere pesos no negativos.")
    _, matriz, activos = _preparar(None, cov)
    n = len(activos)

    if presupuesto is None:
        objetivo_riesgo = np.full(n, 1.0 / n)
    else:
        objetivo_riesgo = normalizar_pesos(presupuesto, activos).to_numpy(float)
        if np.any(objetivo_riesgo <= 0):
            raise ErrorDatos("El presupuesto de riesgo debe ser estrictamente positivo.")

    def objetivo(w: np.ndarray) -> float:
        volatilidad = np.sqrt(max(w @ matriz @ w, 1e-16))
        contribucion = w * (matriz @ w) / volatilidad
        # Se minimiza la distancia entre el reparto observado y el deseado.
        return float(np.sum((contribucion / volatilidad - objetivo_riesgo) ** 2))

    pesos = _optimizar(objetivo, n, restricciones)
    return pd.Series(pesos, index=activos, name="paridad_riesgo")


def comparar_carteras(carteras: dict[str, pd.Series], mu, cov, tasa_libre_riesgo: float = 0.0) -> pd.DataFrame:
    """Tabla con rentabilidad, volatilidad y Sharpe de varias carteras."""
    vector, matriz, activos = _preparar(mu, cov)
    filas = {}
    for nombre, pesos in carteras.items():
        w = normalizar_pesos(pesos, activos).to_numpy(float)
        rentabilidad = float(w @ vector)
        volatilidad = float(np.sqrt(max(w @ matriz @ w, 0.0)))
        filas[nombre] = {
            "rentabilidad": rentabilidad,
            "volatilidad": volatilidad,
            "sharpe": (rentabilidad - tasa_libre_riesgo) / volatilidad if volatilidad else np.nan,
            "activos_con_peso": int(np.sum(np.abs(w) > 1e-6)),
            "peso_maximo": float(np.max(w)),
        }
    return pd.DataFrame(filas).T
