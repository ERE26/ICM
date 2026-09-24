"""Frontera eficiente de Markowitz y linea del mercado de capitales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from icm.comun import ErrorOptimizacion
from icm.cartera.optimizacion import (
    Restricciones,
    _preparar,
    maximo_sharpe,
    minima_varianza,
    minima_varianza_con_objetivo,
)


def frontera_eficiente(
    mu,
    cov,
    n_puntos: int = 40,
    restricciones: Restricciones | None = None,
    incluir_pesos: bool = True,
) -> pd.DataFrame:
    """Recorre la frontera entre la cartera de minima varianza y la de mayor rentabilidad.

    Devuelve un marco con la rentabilidad, la volatilidad y, si se pide, los
    pesos de cada cartera eficiente. Los puntos que el optimizador no logra
    resolver se descartan en silencio: con restricciones exigentes es normal
    que algunos objetivos de rentabilidad sean inalcanzables.
    """
    restricciones = restricciones or Restricciones()
    vector, _, activos = _preparar(mu, cov)

    pesos_min_var = minima_varianza(cov, restricciones)
    rentabilidad_minima = float(pesos_min_var.to_numpy() @ vector)
    rentabilidad_maxima = float(np.max(vector))
    if restricciones.peso_maximo is not None:
        # Con tope por activo no se puede concentrar todo en el mejor activo:
        # el extremo alcanzable es la mejor combinacion que respeta el tope.
        orden = np.sort(vector)[::-1]
        tope = restricciones.peso_maximo
        capacidad = np.minimum(tope, np.maximum(0.0, 1.0 - tope * np.arange(len(orden))))
        rentabilidad_maxima = float(orden @ capacidad)

    if rentabilidad_maxima <= rentabilidad_minima:
        rentabilidad_maxima = rentabilidad_minima * 1.0001 + 1e-6

    objetivos = np.linspace(rentabilidad_minima, rentabilidad_maxima, int(n_puntos))
    filas = []
    for objetivo in objetivos:
        try:
            pesos = minima_varianza_con_objetivo(mu, cov, float(objetivo), restricciones)
        except ErrorOptimizacion:
            continue
        w = pesos.to_numpy(float)
        matriz = cov.to_numpy(float) if isinstance(cov, pd.DataFrame) else np.asarray(cov, float)
        fila = {
            "rentabilidad": float(w @ vector),
            "volatilidad": float(np.sqrt(max(w @ matriz @ w, 0.0))),
        }
        if incluir_pesos:
            fila.update({f"peso_{activo}": float(peso) for activo, peso in zip(activos, w)})
        filas.append(fila)

    if not filas:
        raise ErrorOptimizacion(
            "No se ha podido calcular ningun punto de la frontera con esas restricciones."
        )
    frontera = pd.DataFrame(filas).sort_values("volatilidad").reset_index(drop=True)
    frontera.attrs["activos"] = activos
    return frontera


def linea_mercado_capitales(
    mu,
    cov,
    tasa_libre_riesgo: float = 0.0,
    volatilidad_maxima: float | None = None,
    n_puntos: int = 25,
    restricciones: Restricciones | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Recta que une el activo sin riesgo con la cartera tangente.

    Devuelve la recta muestreada y los pesos de la cartera tangente. Cada punto
    representa una combinacion entre depositar al tipo sin riesgo y invertir en
    la cartera tangente; mas alla de esta ultima se esta apalancando.
    """
    restricciones = restricciones or Restricciones()
    vector, matriz, activos = _preparar(mu, cov)
    tangente = maximo_sharpe(mu, cov, tasa_libre_riesgo, restricciones)

    w = tangente.to_numpy(float)
    rentabilidad_tangente = float(w @ vector)
    volatilidad_tangente = float(np.sqrt(max(w @ matriz @ w, 0.0)))
    if volatilidad_tangente == 0:
        raise ErrorOptimizacion("La cartera tangente no tiene riesgo: la recta no esta definida.")

    pendiente = (rentabilidad_tangente - tasa_libre_riesgo) / volatilidad_tangente
    tope = volatilidad_maxima if volatilidad_maxima is not None else volatilidad_tangente * 1.5
    volatilidades = np.linspace(0.0, tope, int(n_puntos))
    recta = pd.DataFrame(
        {
            "volatilidad": volatilidades,
            "rentabilidad": tasa_libre_riesgo + pendiente * volatilidades,
            "peso_en_tangente": volatilidades / volatilidad_tangente,
        }
    )
    recta.attrs["sharpe_tangente"] = pendiente
    return recta, tangente
