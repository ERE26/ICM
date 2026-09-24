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

    Se exige indice temporal creciente, columnas exactamente iguales a los
    tickers pedidos y ausencia de precios no positivos, que romperian los
    rendimientos logaritmicos aguas abajo.
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
    precios.index = pd.DatetimeIndex(pd.to_datetime(precios.index), freq=None, name="fecha")
    if (precios.dropna(how="all") <= 0).any().any():
        raise ErrorDatos("Se han encontrado precios nulos o negativos en los datos descargados.")
    return precios.astype(float)
