"""Interfaz comun a todas las fuentes de precios.

El resto de la libreria nunca habla con Yahoo Finance ni con ninguna otra
fuente concreta: habla con :class:`ProveedorDatos`. Cambiar de proveedor es
por tanto escribir una clase nueva, sin tocar el analisis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import pandas as pd

from icm.comun import ErrorDatos


def normalizar_tickers(tickers: str | Iterable[str]) -> list[str]:
    """Acepta un ticker suelto o una coleccion y devuelve una lista limpia."""
    if isinstance(tickers, str):
        lista = [tickers]
    else:
        lista = list(tickers)
    limpios: list[str] = []
    for ticker in lista:
        texto = str(ticker).strip().upper()
        if not texto:
            raise ErrorDatos("Se ha recibido un ticker vacio.")
        if texto not in limpios:
            limpios.append(texto)
    if not limpios:
        raise ErrorDatos("No se ha indicado ningun ticker.")
    return limpios


class ProveedorDatos(ABC):
    """Fuente de precios historicos.

    Una implementacion solo tiene que resolver :meth:`descargar` y devolver un
    DataFrame de precios de cierre ajustados, con el indice de fechas ordenado
    y una columna por ticker.
    """

    #: Nombre legible que aparece en los mensajes y en los metadatos del cache.
    nombre: str = "proveedor"

    @abstractmethod
    def descargar(
        self,
        tickers: Sequence[str],
        inicio: str | pd.Timestamp | None = None,
        fin: str | pd.Timestamp | None = None,
        intervalo: str = "1d",
    ) -> pd.DataFrame:
        """Devuelve los precios de cierre ajustados de los tickers indicados."""

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return f"{type(self).__name__}(nombre={self.nombre!r})"


def validar_marco_precios(precios: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Comprueba la forma del marco devuelto por un proveedor y lo ordena.

    Selecciona las columnas pedidas (descartando las demas), ordena el indice
    por fecha y rechaza lo que no se puede analizar: una columna entera sin un
    solo dato, o dos filas con la misma fecha.

    **Los precios cero y negativos se aceptan**, y esto es deliberado. Antes se
    rechazaban para proteger a ``np.log`` aguas abajo, pero eso hacia la
    libreria inservible justo para el mercado que le da nombre: de las 102.167
    horas del pool espaniol entre 2018 y 2026, 2.315 casaron a cero exacto y
    3.553 en negativo, el 5,74 % repartido por 390 dias distintos. El WTI
    tambien cerro a -36,98 el 20 de abril de 2020. Un precio negativo no es un
    error del fichero: es lo que hace un mercado electrico cuando sobra
    renovable, y era precisamente el dia que no se podia estudiar.

    La proteccion vive ahora donde importa, en
    :func:`icm.datos.precios.rendimientos` con ``tipo="log"``, que es la unica
    operacion que de verdad no admite un cero.
    """
    if not isinstance(precios, pd.DataFrame):
        raise ErrorDatos("El proveedor debe devolver un DataFrame de precios.")
    faltan = [t for t in tickers if t not in precios.columns]
    if faltan:
        raise ErrorDatos(f"El proveedor no ha devuelto datos para: {faltan}.")
    precios = precios.loc[:, list(tickers)].sort_index()
    # Se descarta cualquier frecuencia asociada al indice: no sobrevive a un
    # viaje de ida y vuelta por Parquet, y dejarla haria que un marco recien
    # descargado y el mismo marco leido del cache no se considerasen iguales.
    precios.index = pd.DatetimeIndex(_indice_temporal(precios.index), freq=None, name="fecha")

    vacias = [t for t in tickers if precios[t].isna().all()]
    if vacias:
        raise ErrorDatos(
            f"El proveedor ha devuelto columnas sin un solo dato: {vacias}. Suele "
            "significar que el ticker no existe en la fuente o que no cotiza en la "
            "ventana pedida."
        )
    if precios.index.has_duplicates:
        repetidas = precios.index[precios.index.duplicated()].unique()[:3]
        raise ErrorDatos(
            f"El indice tiene fechas repetidas (por ejemplo {[str(f) for f in repetidas]}). "
            "Con dos precios para la misma fecha no hay serie que analizar."
        )
    return precios.astype(float)


def _indice_temporal(indice) -> pd.DatetimeIndex:
    """Convierte el indice a fechas sin caer en la trampa de los enteros.

    ``pd.to_datetime`` lee un entero como nanosegundos desde 1970, asi que un
    indice ``AAAAMMDD`` -- lo que trae un Parquet de cache o un proveedor
    propio -- se convertia en 1970 sin protestar. Es el mismo fallo que ya se
    habia corregido en ``convertir_fechas`` y que aqui seguia vivo.
    """
    if isinstance(indice, pd.DatetimeIndex):
        return indice
    valores = pd.Index(indice)
    if pd.api.types.is_integer_dtype(valores):
        enteros = valores.dropna()
        if len(enteros) and enteros.min() >= 19000101 and enteros.max() <= 21001231:
            return pd.to_datetime(valores.astype(str), format="%Y%m%d")
        raise ErrorDatos(
            f"El indice es entero y no tiene forma de fecha AAAAMMDD (rango "
            f"{valores.min()}..{valores.max()}). Conviertalo antes a fechas."
        )
    return pd.DatetimeIndex(pd.to_datetime(valores))
