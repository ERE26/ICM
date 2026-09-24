"""Cache local en Parquet para no repetir descargas.

Yahoo limita las peticiones y acaba bloqueando temporalmente al que abusa, asi
que conviene bajar cada serie una sola vez. El cache es un envoltorio sobre
cualquier :class:`~icm.datos.proveedor.ProveedorDatos`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from icm.datos.proveedor import ProveedorDatos, normalizar_tickers, validar_marco_precios

#: Carpeta por defecto, relativa al directorio de trabajo.
DIRECTORIO_CACHE = Path("datos_cache")


class CacheParquet:
    """Almacen de marcos de precios en disco, indexado por clave de consulta."""

    def __init__(self, directorio: str | Path = DIRECTORIO_CACHE) -> None:
        self.directorio = Path(directorio)

    def clave(
        self,
        proveedor: str,
        tickers: Sequence[str],
        inicio,
        fin,
        intervalo: str,
    ) -> str:
        """Huella estable de una consulta, usada como nombre de fichero."""
        descriptor = json.dumps(
            {
                "proveedor": proveedor,
                "tickers": sorted(tickers),
                "inicio": str(inicio),
                "fin": str(fin),
                "intervalo": intervalo,
            },
            sort_keys=True,
        )
        resumen = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:16]
        etiqueta = "-".join(sorted(tickers))[:40].replace("/", "_").replace(".", "_")
        return f"{proveedor}_{etiqueta}_{resumen}"

    def ruta(self, clave: str) -> Path:
        return self.directorio / f"{clave}.parquet"

    def leer(self, clave: str, caducidad_horas: float | None = None) -> pd.DataFrame | None:
        """Devuelve el marco guardado, o ``None`` si no existe o ha caducado."""
        ruta = self.ruta(clave)
        if not ruta.exists():
            return None
        if caducidad_horas is not None:
            antiguedad = (time.time() - ruta.stat().st_mtime) / 3600.0
            if antiguedad > caducidad_horas:
                return None
        return pd.read_parquet(ruta)

    def escribir(self, clave: str, precios: pd.DataFrame) -> Path:
        self.directorio.mkdir(parents=True, exist_ok=True)
        ruta = self.ruta(clave)
        precios.to_parquet(ruta)
        return ruta

    def vaciar(self) -> int:
        """Borra todo el contenido del cache y devuelve cuantos ficheros elimino."""
        if not self.directorio.exists():
            return 0
        ficheros = list(self.directorio.glob("*.parquet"))
        for fichero in ficheros:
            fichero.unlink()
        return len(ficheros)


class ProveedorConCache(ProveedorDatos):
    """Envuelve un proveedor y guarda en disco lo que descarga.

    Parametros
    ----------
    proveedor:
        Fuente real de los datos.
    cache:
        Almacen a utilizar. Por defecto, ``datos_cache/`` en el directorio
        de trabajo.
    caducidad_horas:
        Pasadas estas horas se vuelve a descargar. ``None`` significa que lo
        guardado no caduca nunca, que es lo razonable para series historicas
        ya cerradas.
    """

    def __init__(
        self,
        proveedor: ProveedorDatos,
        cache: CacheParquet | None = None,
        caducidad_horas: float | None = 24.0,
    ) -> None:
        self.proveedor = proveedor
        self.cache = cache if cache is not None else CacheParquet()
        self.caducidad_horas = caducidad_horas

    @property
    def nombre(self) -> str:  # type: ignore[override]
        return f"cache:{self.proveedor.nombre}"

    def descargar(
        self,
        tickers: Sequence[str],
        inicio: str | pd.Timestamp | None = None,
        fin: str | pd.Timestamp | None = None,
        intervalo: str = "1d",
    ) -> pd.DataFrame:
        tickers = normalizar_tickers(tickers)
        clave = self.cache.clave(self.proveedor.nombre, tickers, inicio, fin, intervalo)

        guardado = self.cache.leer(clave, self.caducidad_horas)
        if guardado is not None:
            return validar_marco_precios(guardado, tickers)

        precios = self.proveedor.descargar(tickers, inicio, fin, intervalo)
        self.cache.escribir(clave, precios)
        return precios
