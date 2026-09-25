"""Constantes y utilidades compartidas por todos los modulos."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Numero de sesiones bursatiles en un anio natural.
DIAS_HABILES_ANIO = 252

#: Periodos por anio segun la frecuencia de la serie.
PERIODOS_POR_ANIO = {
    "diaria": DIAS_HABILES_ANIO,
    "semanal": 52,
    "mensual": 12,
    "trimestral": 4,
    "anual": 1,
}


class ErrorDatos(Exception):
    """Los datos de entrada no tienen la forma o el contenido esperados."""


class ErrorDescarga(Exception):
    """No se han podido obtener los precios del proveedor externo."""


class ErrorOptimizacion(Exception):
    """El optimizador no ha encontrado una solucion valida."""


def periodos_por_anio(frecuencia: str | int) -> int:
    """Traduce una frecuencia a numero de periodos por anio.

    Acepta tanto un nombre ("diaria", "mensual", ...) como un entero, que se
    devuelve tal cual para permitir frecuencias no contempladas.
    """
    if isinstance(frecuencia, bool):
        # En Python bool es subclase de int, asi que True colaba como "un
        # periodo al anio" en lugar de como el error que es.
        raise ErrorDatos("Se ha recibido un booleano como frecuencia.")
    if isinstance(frecuencia, (int, np.integer)):
        if frecuencia <= 0:
            raise ErrorDatos("El numero de periodos por anio debe ser positivo.")
        return int(frecuencia)
    clave = frecuencia.lower()
    if clave not in PERIODOS_POR_ANIO:
        opciones = ", ".join(sorted(PERIODOS_POR_ANIO))
        raise ErrorDatos(f"Frecuencia '{frecuencia}' desconocida. Opciones: {opciones}.")
    return PERIODOS_POR_ANIO[clave]


def _a_float(datos, nombre: str):
    """Convierte a array de float traduciendo los fallos a ``ErrorDatos``.

    La libreria promete ``ErrorDatos`` ante una entrada mal formada, pero
    ``np.asarray(..., dtype=float)`` dejaba escapar el ``ValueError`` crudo de
    numpy ("could not convert string to float"), que ningun `except` de la
    libreria recoge.
    """
    try:
        return np.asarray(datos, dtype=float)
    except (TypeError, ValueError) as error:
        raise ErrorDatos(f"'{nombre}' contiene valores que no son numeros: {error}") from error


def _denunciar_infinitos(valores: np.ndarray, nombre: str) -> None:
    """Rechaza los infinitos, que ``dropna`` no ve.

    Un infinito no es un hueco: es el resultado de una division por cero o de
    un ``log(0)``, y en esta libreria eso significa que en algun sitio hubo un
    precio nulo. Descartarlo en silencio seria peor que un NaN, porque el NaN
    al menos se propaga y hace ruido: medido, una serie de 500 rendimientos
    con **un** ``+inf`` inyectado devolvia exactamente el mismo VaR historico
    que la serie limpia, con la cifra intacta y la serie corrompida.
    """
    ausentes = np.isnan(valores)
    infinitos = ~np.isfinite(valores) & ~ausentes
    if infinitos.any():
        raise ErrorDatos(
            f"'{nombre}' contiene {int(infinitos.sum())} valores infinitos. Suelen venir "
            "de dividir por un precio cero o de un logaritmo de cero; en una serie "
            "electrica eso pasa de verdad. Revise los precios de origen, o use la "
            "diferencia en vez del rendimiento."
        )


def como_serie(datos, nombre: str = "serie") -> pd.Series:
    """Convierte la entrada en una Series de float, descartando los ausentes.

    Los infinitos **no** se descartan: se denuncian. Ver
    :func:`_denunciar_infinitos`.
    """
    if isinstance(datos, pd.Series):
        serie = datos.astype(float)
    elif isinstance(datos, pd.DataFrame):
        if datos.shape[1] != 1:
            raise ErrorDatos(
                f"Se esperaba una unica columna en '{nombre}', se han recibido {datos.shape[1]}."
            )
        serie = datos.iloc[:, 0].astype(float)
    else:
        valores = _a_float(datos, nombre)
        if valores.ndim != 1:
            raise ErrorDatos(
                f"Se esperaba una serie de una dimension en '{nombre}', se han recibido "
                f"{valores.ndim}. Seleccione una columna."
            )
        serie = pd.Series(valores, name=nombre)
    _denunciar_infinitos(serie.to_numpy(), nombre)
    serie = serie.dropna()
    if serie.empty:
        raise ErrorDatos(f"La serie '{nombre}' no contiene observaciones validas.")
    return serie


def como_matriz(datos, nombre: str = "matriz") -> pd.DataFrame:
    """Convierte la entrada en un DataFrame de float sin filas incompletas.

    Igual que :func:`como_serie`, los infinitos se denuncian en lugar de
    colarse: ``dropna`` no los ve.
    """
    if isinstance(datos, pd.DataFrame):
        marco = datos.astype(float)
    elif isinstance(datos, pd.Series):
        marco = datos.to_frame().astype(float)
    else:
        marco = pd.DataFrame(_a_float(datos, nombre))
    _denunciar_infinitos(marco.to_numpy(), nombre)
    marco = marco.dropna(how="any")
    if marco.empty:
        raise ErrorDatos(f"'{nombre}' no contiene filas completas.")
    return marco


def validar_nivel(nivel: float) -> float:
    """Comprueba que un nivel de confianza este en el intervalo (0, 1)."""
    nivel = float(nivel)
    if not 0.0 < nivel < 1.0:
        raise ErrorDatos(f"El nivel de confianza debe estar entre 0 y 1, se ha recibido {nivel}.")
    return nivel


def normalizar_pesos(pesos, activos=None) -> pd.Series:
    """Devuelve los pesos como Series indexada por activo y con suma unitaria.

    Se exige que la suma sea **positiva**. Dividir por una suma negativa
    invierte el signo de todos los pesos, y como el resultado sigue sumando
    1,0 no hay nada en el que delate lo ocurrido: la posicion larga sale corta
    y la corta, larga.

    No es hipotetico. ``Restricciones(permitir_cortos=True)`` es una opcion
    documentada del optimizador, y esta funcion es el portero de
    ``rendimientos_cartera``, ``volatilidad_cartera``, ``contribuciones_riesgo``,
    ``var_incremental`` y los tres puntos de entrada de ``riesgo.montecarlo``.
    Medido de punta a punta con pesos ``{A: +0,3, B: -1,3}``, la serie de
    resultados salia con el signo cambiado en todas las fechas y un acumulado
    real del -1,17 % se publicaba como **+1,03 %**. La volatilidad si cuadraba,
    porque es cuadratica en los pesos, de modo que la comprobacion habitual
    ("la vol encaja, luego los pesos estan bien") no detectaba nada.

    Una cartera cuya exposicion neta es negativa no admite "peso relativo":
    no hay capital propio sobre el que repartir. Si lo que se quiere es
    expresar posiciones que ya suman uno, paselas asi y no se tocaran.
    """
    if isinstance(pesos, pd.Series):
        serie = pesos.astype(float)
    elif isinstance(pesos, dict):
        serie = pd.Series(pesos, dtype=float)
    else:
        valores = _a_float(pesos, "pesos")
        indice = list(activos) if activos is not None else range(len(valores))
        serie = pd.Series(valores, index=indice)
    if activos is not None:
        faltan = [a for a in activos if a not in serie.index]
        if faltan:
            raise ErrorDatos(f"Faltan pesos para los activos: {faltan}.")
        serie = serie.reindex(list(activos))

    # Antes que nada los ausentes: 'sum' los ignora, asi que un peso a NaN
    # producia una Series con un NaN dentro cuya suma valia 1,0 y parecia
    # perfectamente normalizada.
    valores = serie.to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        malos = [str(a) for a, v in serie.items() if not np.isfinite(v)]
        raise ErrorDatos(
            f"Hay pesos ausentes o infinitos en: {malos}. La suma los ignoraria y el "
            "resultado pareceria normalizado."
        )

    suma = float(serie.sum())
    # La comprobacion no puede ser "la suma es cero" con una tolerancia
    # absoluta: lo que rompe la normalizacion no es el cero exacto sino una
    # exposicion NETA despreciable frente a la BRUTA. Una cartera neutral
    # descuadrada por 1e-7 tiene suma 1e-7 y bruta 2, asi que pasaba el filtro
    # de np.isclose (atol=1e-8) y salia con pesos del orden de 1e7 sin que
    # nada avisase. El criterio es por tanto el cociente entre las dos.
    bruta = float(np.abs(valores).sum())
    if bruta == 0.0 or abs(suma) < 1e-6 * bruta:
        raise ErrorDatos(
            f"La exposicion neta ({suma:.3g}) es despreciable frente a la bruta "
            f"({bruta:.3g}): normalizar multiplicaria los pesos por {bruta / max(abs(suma), 1e-300):.3g} "
            "y el resultado no significaria nada. Una cartera neutral no tiene capital "
            "neto sobre el que repartir pesos relativos: trabaje con las posiciones sin "
            "normalizar."
        )
    if suma < 0.0:
        raise ErrorDatos(
            f"Los pesos suman {suma:.6g}, que es negativo, y dividir por esa suma "
            "invertiria el signo de todos: la posicion larga saldria corta y la corta, "
            "larga, con el resultado sumando 1,0 y sin ninguna senial de lo ocurrido. "
            "Una cartera con exposicion neta negativa no admite pesos relativos; si las "
            "posiciones ya son las que quiere, paselas sumando uno."
        )
    return serie / suma
