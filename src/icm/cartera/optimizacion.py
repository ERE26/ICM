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
        """Traduce las restricciones a los limites por activo que espera scipy.

        Dos cosas que antes se colaban en silencio:

        - ``peso_minimo`` **sobrescribia** el suelo sin mirar
          ``permitir_cortos``, asi que
          ``Restricciones(permitir_cortos=False, peso_minimo=-0.5)`` devolvia
          ``[(-0.5, 1.0), ...]``: una cartera con cortos pedida explicitamente
          sin cortos. Ahora es un error.
        - Con ``permitir_cortos=True`` el tope quedaba en 1,0, de modo que un
          130/30 -- la razon de ser de permitir cortos -- era infactible sin
          decirlo. El tope pasa a ser el maximo que de verdad puede alcanzar un
          peso dado el suelo de los demas, ``1 - (n-1) * inferior``.
        """
        if n_activos < 1:
            raise ErrorDatos("Hace falta al menos un activo.")
        inferior = -1.0 if self.permitir_cortos else 0.0
        if self.peso_minimo is not None:
            inferior = float(self.peso_minimo)
            if inferior < 0.0 and not self.permitir_cortos:
                raise ErrorDatos(
                    f"Se ha pedido un peso minimo de {inferior} con "
                    f"permitir_cortos=False: son incompatibles. Ponga "
                    f"permitir_cortos=True si de verdad quiere posiciones cortas."
                )
        # Un peso no puede pasar de 1 menos el suelo que ocupan los demas.
        superior = 1.0 - (n_activos - 1) * inferior
        if self.peso_maximo is not None:
            superior = float(self.peso_maximo)
        if inferior * n_activos > 1.0 + 1e-9:
            raise ErrorDatos(
                f"Con un peso minimo de {inferior} y {n_activos} activos los pesos "
                "no pueden sumar 1."
            )
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
    if isinstance(mu, pd.Series):
        # Se reindexa por las etiquetas ORIGINALES de la matriz, no por su
        # 'str()'. Con columnas e indice numericos, 'reindex(["0", "1"])'
        # devolvia todo NaN y el error que salia -- "hay rentabilidades
        # esperadas ausentes" -- mandaba a buscar un hueco en los datos que no
        # existia: lo que no encajaba era el tipo de la etiqueta.
        alineado = mu.reindex(marco_cov.columns)
        if alineado.isna().all() and not mu.isna().all():
            raise ErrorDatos(
                f"Las etiquetas de las rentabilidades esperadas no coinciden con las "
                f"de la matriz de covarianzas: {list(mu.index)[:5]} frente a "
                f"{list(marco_cov.columns)[:5]}. Compruebe tambien el tipo (texto "
                f"frente a numero)."
            )
        vector = alineado.to_numpy(float)
    else:
        vector = np.asarray(mu, float)
    if vector.shape != (len(activos),):
        raise ErrorDatos("Las rentabilidades esperadas no encajan con la matriz de covarianzas.")
    if np.isnan(vector).any():
        ausentes = [a for a, v in zip(activos, vector) if np.isnan(v)]
        raise ErrorDatos(f"Hay rentabilidades esperadas ausentes para {ausentes}.")
    return vector, matriz, activos


def _pesos_alineados(pesos, etiquetas) -> np.ndarray:
    """Ordena los pesos segun las etiquetas del otro operando.

    ``normalizar_pesos(pesos)`` sin etiquetas devuelve el vector en el orden en
    que venga el diccionario o la Series, y ese vector se multiplicaba despues
    por una matriz que esta en el orden de SUS columnas. Con
    ``{A: 0,9, B: 0,1}`` sobre dos activos de volatilidad 20 % y 60 %, la
    cartera daba 0,1897 -- correcto -- y escribiendo los mismos pesos como
    ``{B: 0,1, A: 0,9}`` daba **0,5404**: un factor 2,85 decidido por el orden
    de las claves de un diccionario. La rentabilidad pasaba de 0,0600 a
    0,1400. ``contribuciones_riesgo``, en el mismo modulo, si alineaba.

    Un ``ndarray`` se acepta tal cual, por posicion: es como lo pasan las
    rutinas internas, que ya trabajan en el orden de ``activos``.
    """
    if isinstance(pesos, np.ndarray):
        vector = np.asarray(pesos, dtype=float)
        if etiquetas is not None and vector.shape != (len(etiquetas),):
            raise ErrorDatos(
                f"Se han dado {vector.size} pesos para {len(etiquetas)} activos."
            )
        return vector
    return normalizar_pesos(pesos, etiquetas).to_numpy(float)


def rentabilidad_cartera(pesos, mu) -> float:
    """Rentabilidad esperada de la cartera.

    Los pesos se alinean por nombre con el indice de ``mu`` cuando este lo
    tiene; vease :func:`_pesos_alineados`.
    """
    etiquetas = mu.index if isinstance(mu, pd.Series) else None
    vector = _pesos_alineados(pesos, etiquetas)
    return float(vector @ np.asarray(mu, float))


def volatilidad_cartera(pesos, cov) -> float:
    """Desviacion tipica de la cartera.

    Los pesos se alinean por nombre con las columnas de ``cov`` cuando esta es
    un ``DataFrame``; vease :func:`_pesos_alineados`.
    """
    es_marco = isinstance(cov, pd.DataFrame)
    matriz = cov.to_numpy(float) if es_marco else np.asarray(cov, float)
    vector = _pesos_alineados(pesos, cov.columns if es_marco else None)
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
    vector = _pesos_alineados(pesos, marco_cov.columns)
    serie_pesos = pd.Series(vector, index=marco_cov.columns)
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

    acotada = restricciones.peso_maximo is not None or restricciones.peso_minimo is not None
    pesos = _paridad_convexa(matriz, objetivo_riesgo)

    if acotada:
        # Con tope o suelo por activo la formulacion logaritmica ya no sirve,
        # asi que se vuelve a SLSQP -- pero arrancando de la solucion convexa,
        # que suele estar cerca, y comprobando el resultado despues.
        def objetivo(w: np.ndarray) -> float:
            varianza = max(w @ matriz @ w, 1e-16)
            contribucion = w * (matriz @ w) / varianza
            return float(np.sum((contribucion - objetivo_riesgo) ** 2))

        pesos = _optimizar(objetivo, n, restricciones, inicio=pesos)

    _verificar_paridad(pesos, matriz, objetivo_riesgo, activos, acotada)
    return pd.Series(pesos, index=activos, name="paridad_riesgo")


#: Desviacion maxima admisible entre la contribucion relativa al riesgo de cada
#: activo y la que se le pidio. Un punto porcentual sobre el total.
TOLERANCIA_PARIDAD = 0.01


def _paridad_convexa(matriz: np.ndarray, presupuesto: np.ndarray) -> np.ndarray:
    """Paridad de riesgo por la formulacion logaritmica, que si es convexa.

    El objetivo que habia -- la suma de cuadrados de las diferencias entre las
    contribuciones relativas y las deseadas -- **no es convexo**. SLSQP se
    quedaba en caras del simplex con activos expulsados a peso cero y devolvia
    ``success=True``, y lo que uno miraria para validarlo (que las
    contribuciones de los supervivientes salgan iguales) tambien cuadraba.
    Medido sobre 20 matrices de 10 activos sin correlacion y con
    volatilidades entre el 0,3 % y el 50 %: **6 de 20 con activos a peso cero**
    y hasta 0,25 de dispersion entre contribuciones que deberian ser
    identicas. Con 3 activos, 1 de 20.

    La reformulacion estandar (Maillard et al. 2010; Spinu 2013) minimiza

        f(y) = 1/2 y' S y - sum_i b_i log(y_i)     con y > 0

    que es **estrictamente convexa** en el ortante positivo, tiene minimo
    unico y cuyo punto critico cumple ``y_i (S y)_i = b_i`` exactamente, que es
    la condicion de paridad. Los pesos salen de reescalar ``y`` para que sumen
    1, porque la condicion es invariante a la escala. Se resuelve con el
    gradiente analitico ``S y - b / y``, que es barato y exacto.
    """
    n = matriz.shape[0]
    # La condicion de paridad es invariante a la escala de la matriz, asi que
    # se normaliza a diagonal media 1 antes de resolver. Sin esto, con
    # volatilidades entre el 0,3 % y el 50 % los dos terminos del objetivo
    # -- la forma cuadratica y el logaritmo -- viven a cuatro ordenes de
    # magnitud de distancia y la busqueda lineal aborta.
    factor = float(np.mean(np.diag(matriz)))
    if factor <= 0:
        raise ErrorOptimizacion("La matriz de covarianzas tiene diagonal nula o negativa.")
    normalizada = matriz / factor

    def objetivo_y(y: np.ndarray) -> float:
        return float(0.5 * y @ normalizada @ y - presupuesto @ np.log(y))

    def gradiente(y: np.ndarray) -> np.ndarray:
        return normalizada @ y - presupuesto / y

    resultado = optimize.minimize(
        objetivo_y,
        np.full(n, 1.0 / np.sqrt(n)),
        jac=gradiente,
        method="L-BFGS-B",
        bounds=[(1e-10, None)] * n,
        options={"maxiter": 10_000, "maxfun": 20_000},
    )
    y = np.asarray(resultado.x, dtype=float)
    if not np.isfinite(y).all() or y.sum() <= 0:
        raise ErrorOptimizacion(
            f"La paridad de riesgo no ha convergido: {resultado.message}"
        )
    # El indicador de scipy NO es el criterio: con este objetivo puede decir
    # 'ABNORMAL' porque la busqueda lineal no logra mejorar cuando ya esta en
    # el optimo. Quien juzga es '_verificar_paridad', que comprueba la
    # condicion de primer orden sobre el resultado.
    return y / y.sum()


def _verificar_paridad(
    pesos: np.ndarray,
    matriz: np.ndarray,
    presupuesto: np.ndarray,
    activos: list[str],
    acotada: bool,
) -> None:
    """Comprueba que el reparto del riesgo sea el pedido, y no solo que scipy lo diga.

    Es la comprobacion que faltaba: ``success=True`` no dice nada sobre si el
    optimo encontrado reparte el riesgo, y con un objetivo no convexo eran dos
    cosas distintas.
    """
    varianza = float(pesos @ matriz @ pesos)
    if varianza <= 0:
        raise ErrorOptimizacion("La cartera de paridad sale sin riesgo: revise la matriz.")
    relativa = pesos * (matriz @ pesos) / varianza
    desviacion = float(np.max(np.abs(relativa - presupuesto)))
    if desviacion <= TOLERANCIA_PARIDAD:
        return
    peor = int(np.argmax(np.abs(relativa - presupuesto)))
    detalle = (
        "Con tope o suelo por activo la paridad exacta puede ser inalcanzable: "
        "relaje las restricciones o acepte el reparto aproximado."
        if acotada
        else "Revise el condicionamiento de la matriz de covarianzas."
    )
    raise ErrorOptimizacion(
        f"La paridad de riesgo no se ha alcanzado: '{activos[peor]}' aporta el "
        f"{relativa[peor]:.4f} del riesgo cuando se le pidio {presupuesto[peor]:.4f} "
        f"(desviacion maxima {desviacion:.4f}, tolerancia {TOLERANCIA_PARIDAD}). "
        f"{detalle}"
    )


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
