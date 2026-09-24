"""Proveedor de precios basado en Yahoo Finance a traves de ``yfinance``.

Yahoo no publica un contrato de API estable, asi que la descarga se aisla aqui
y se traduce cualquier fallo a :class:`~icm.comun.ErrorDescarga`. Si el dia de
maniana Yahoo deja de servir datos, basta con escribir otro proveedor.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Sequence

import pandas as pd

from icm.comun import ErrorDescarga
from icm.datos.proveedor import ProveedorDatos, normalizar_tickers, validar_marco_precios

#: Registros por los que ``yfinance`` escribe sus propios avisos de error.
_REGISTROS_YFINANCE = ("yfinance", "peewee", "urllib3")


@contextlib.contextmanager
def _yfinance_callado(activo: bool):
    """Sube el umbral de los registros de yfinance mientras dura la descarga.

    yfinance informa de cada ticker fallido por su cuenta. Como aqui esos
    fallos se recogen y se traducen a :class:`ErrorDescarga`, dejarlos pasar
    solo duplica el mensaje y, cuando no hay red, llena la pantalla.
    """
    if not activo:
        yield
        return
    niveles = {}
    for nombre in _REGISTROS_YFINANCE:
        registro = logging.getLogger(nombre)
        niveles[nombre] = registro.level
        registro.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for nombre, nivel in niveles.items():
            logging.getLogger(nombre).setLevel(nivel)


class ProveedorYahoo(ProveedorDatos):
    """Descarga precios de cierre ajustados de Yahoo Finance.

    Parametros
    ----------
    reintentos:
        Numero de intentos antes de rendirse. Yahoo limita las peticiones y
        devuelve errores esporadicos, asi que reintentar compensa.
    espera_inicial:
        Segundos de espera tras el primer fallo. Se duplica en cada reintento.
    silencioso:
        Silencia los mensajes que ``yfinance`` escribe por su cuenta. Como esta
        clase ya traduce cualquier fallo a :class:`ErrorDescarga`, esa salida
        solo aniade ruido; ponerlo a ``False`` la deja pasar para depurar.
    """

    nombre = "yahoo"

    def __init__(
        self,
        reintentos: int = 3,
        espera_inicial: float = 1.0,
        silencioso: bool = True,
    ) -> None:
        self.reintentos = max(1, int(reintentos))
        self.espera_inicial = float(espera_inicial)
        self.silencioso = bool(silencioso)

    def descargar(
        self,
        tickers: Sequence[str],
        inicio: str | pd.Timestamp | None = None,
        fin: str | pd.Timestamp | None = None,
        intervalo: str = "1d",
    ) -> pd.DataFrame:
        tickers = normalizar_tickers(tickers)
        espera = self.espera_inicial
        ultimo_error: Exception | None = None

        for intento in range(1, self.reintentos + 1):
            try:
                with _yfinance_callado(self.silencioso):
                    crudo = self._descargar_una_vez(tickers, inicio, fin, intervalo)
            except Exception as exc:  # yfinance lanza excepciones muy variadas
                ultimo_error = exc
            else:
                precios = self._extraer_cierres(crudo, tickers)
                if not precios.dropna(how="all").empty:
                    return validar_marco_precios(precios, tickers)
                ultimo_error = ErrorDescarga("Yahoo ha devuelto una respuesta vacia.")

            if intento < self.reintentos:
                time.sleep(espera)
                espera *= 2

        raise ErrorDescarga(
            f"No se han podido descargar {tickers} de Yahoo Finance tras "
            f"{self.reintentos} intentos: {ultimo_error}"
        ) from ultimo_error

    @staticmethod
    def _descargar_una_vez(
        tickers: list[str],
        inicio: str | pd.Timestamp | None,
        fin: str | pd.Timestamp | None,
        intervalo: str,
    ) -> pd.DataFrame:
        import yfinance  # import diferido: la libreria tarda en cargar

        return yfinance.download(
            tickers=tickers,
            start=inicio,
            end=fin,
            interval=intervalo,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

    @staticmethod
    def _extraer_cierres(crudo: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Aplana la respuesta de yfinance y se queda con el cierre ajustado.

        Con ``auto_adjust=True`` la columna 'Close' ya viene ajustada por
        dividendos y splits. yfinance devuelve columnas simples para un unico
        ticker y un MultiIndex para varios, de ahi los dos caminos.
        """
        if crudo is None or crudo.empty:
            return pd.DataFrame(columns=tickers)

        if isinstance(crudo.columns, pd.MultiIndex):
            nivel_campos = 0 if "Close" in crudo.columns.get_level_values(0) else 1
            cierres = crudo.xs("Close", axis=1, level=nivel_campos)
        elif "Close" in crudo.columns:
            cierres = crudo[["Close"]]
            cierres.columns = tickers[:1]
        else:  # pragma: no cover - forma inesperada de la respuesta
            return pd.DataFrame(columns=tickers)

        cierres = cierres.copy()
        cierres.columns = [str(c).upper() for c in cierres.columns]
        for ticker in tickers:
            if ticker not in cierres.columns:
                cierres[ticker] = pd.NA
        return cierres.loc[:, tickers]
