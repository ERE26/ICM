"""Carga de datos propios: ficheros en disco o marcos ya en memoria.

Yahoo Finance no es la unica fuente posible. Lo habitual en un trabajo es tener
un CSV o un Excel con las series ya descargadas, o un marco construido en el
propio cuaderno. Estos proveedores los envuelven con la misma interfaz que el
resto, de modo que todo el analisis funciona igual sin cambiar una linea.

Se admiten dos formas de organizar la tabla:

``ancho``
    una fila por fecha y una columna por activo. Es el formato natural para el
    analisis y el que devuelven casi todas las descargas.
``largo``
    tres columnas: fecha, identificador del activo y valor. Es el formato de
    las bases de datos y el que sale de muchas consultas SQL.

Y dos tipos de contenido: precios o rendimientos. Si se entregan rendimientos
se reconstruye un indice de precios, porque el resto de la libreria trabaja
sobre precios y asi la conversion queda hecha en un unico sitio.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos
from icm.datos.proveedor import ProveedorDatos, normalizar_tickers, validar_marco_precios

#: Nombres que se prueban para localizar la columna de fechas.
NOMBRES_FECHA = ("fecha", "date", "fechas", "dates", "time", "datetime", "periodo", "dia")

#: Extensiones reconocidas y como leerlas.
LECTORES = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
}


def convertir_fechas(serie: pd.Series, dia_primero: bool = False) -> pd.Series:
    """Interpreta una columna como fechas, cubriendo el caso ``AAAAMMDD``.

    Los enteros merecen un trato aparte: ``pd.to_datetime`` los lee como
    nanosegundos desde 1970, asi que ``20240102`` se convertiria en una fecha
    de 1970 sin protestar. Cuando todos los valores caen en el rango de fechas
    plausibles escritas asi, se interpretan con ese formato.
    """
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie

    if pd.api.types.is_integer_dtype(serie):
        valores = serie.dropna()
        if not valores.empty and valores.between(19000101, 21001231).all():
            return pd.to_datetime(serie.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        # Un entero que no tiene forma de AAAAMMDD no es una fecha fiable.
        return pd.Series(pd.NaT, index=serie.index)

    if pd.api.types.is_numeric_dtype(serie):
        return pd.Series(pd.NaT, index=serie.index)

    return pd.to_datetime(serie, errors="coerce", dayfirst=dia_primero)


def _localizar_columna_fecha(
    marco: pd.DataFrame, indicada: str | None, dia_primero: bool = False
) -> str:
    """Devuelve el nombre de la columna de fechas."""
    if indicada is not None:
        if indicada not in marco.columns:
            raise ErrorDatos(
                f"La columna de fechas '{indicada}' no esta en el fichero. "
                f"Columnas disponibles: {list(marco.columns)}."
            )
        return indicada

    for columna in marco.columns:
        if str(columna).strip().lower() in NOMBRES_FECHA:
            return str(columna)

    # Ultimo recurso: la primera columna, que es donde suele ir la fecha.
    primera = marco.columns[0]
    if convertir_fechas(marco[primera], dia_primero).notna().mean() > 0.9:
        return str(primera)

    raise ErrorDatos(
        "No se ha encontrado la columna de fechas. Indiquela con 'columna_fecha'. "
        f"Columnas disponibles: {list(marco.columns)}."
    )


def _a_formato_ancho(
    marco: pd.DataFrame,
    formato: str,
    columna_fecha: str | None,
    columna_ticker: str | None,
    columna_valor: str | None,
    dia_primero: bool,
) -> pd.DataFrame:
    """Normaliza la tabla a una fila por fecha y una columna por activo."""
    if formato not in {"ancho", "largo"}:
        raise ErrorDatos(f"Formato '{formato}' desconocido: use 'ancho' o 'largo'.")

    marco = marco.copy()
    marco.columns = [str(c).strip() for c in marco.columns]

    # Si el indice ya es temporal y no hay columna de fechas, se respeta.
    if isinstance(marco.index, pd.DatetimeIndex) and formato == "ancho":
        ancho = marco
    else:
        fecha = _localizar_columna_fecha(marco, columna_fecha, dia_primero)
        marco[fecha] = convertir_fechas(marco[fecha], dia_primero)
        if marco[fecha].isna().all():
            raise ErrorDatos(f"No se ha podido interpretar ninguna fecha en la columna '{fecha}'.")
        marco = marco.dropna(subset=[fecha])

        if formato == "ancho":
            ancho = marco.set_index(fecha)
        else:
            if columna_ticker is None or columna_valor is None:
                restantes = [c for c in marco.columns if c != fecha]
                if len(restantes) != 2:
                    raise ErrorDatos(
                        "En formato largo hay que indicar 'columna_ticker' y 'columna_valor' "
                        f"salvo que solo queden dos columnas; quedan {restantes}."
                    )
                columna_ticker, columna_valor = restantes
            ancho = marco.pivot_table(
                index=fecha, columns=columna_ticker, values=columna_valor, aggfunc="last"
            )

    ancho.index = pd.DatetimeIndex(ancho.index, freq=None, name="fecha")
    ancho.columns = [str(c).strip().upper() for c in ancho.columns]
    return ancho.sort_index()


def _a_numerico(marco: pd.DataFrame) -> pd.DataFrame:
    """Convierte a numero las columnas que hayan llegado como texto.

    Es lo habitual con los CSV exportados desde una hoja de calculo espaniola:
    el separador de miles y el decimal viajan dentro de la cadena.
    """
    convertido = {}
    for columna in marco.columns:
        serie = marco[columna]
        if pd.api.types.is_numeric_dtype(serie):
            convertido[columna] = serie.astype(float)
            continue
        texto = (
            serie.astype(str)
            .str.strip()
            .str.replace(" ", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        # Si aparecen las dos, el ultimo en aparecer es el decimal.
        con_ambos = texto.str.contains(r"\.", na=False) & texto.str.contains(",", na=False)
        texto = texto.mask(
            con_ambos & (texto.str.rfind(",") > texto.str.rfind(".")),
            texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        )
        texto = texto.mask(
            con_ambos & (texto.str.rfind(".") > texto.str.rfind(",")),
            texto.str.replace(",", "", regex=False),
        )
        texto = texto.mask(~con_ambos, texto.str.replace(",", ".", regex=False))
        convertido[columna] = pd.to_numeric(texto, errors="coerce")

    salida = pd.DataFrame(convertido, index=marco.index)
    vacias = [c for c in salida.columns if salida[c].isna().all()]
    if vacias:
        raise ErrorDatos(f"Estas columnas no contienen numeros interpretables: {vacias}.")
    return salida


def _rendimientos_a_precios(
    rendimientos: pd.DataFrame,
    tipo: str,
    precio_inicial: float,
) -> pd.DataFrame:
    """Reconstruye un indice de precios a partir de una tabla de rendimientos."""
    if tipo not in {"simple", "log"}:
        raise ErrorDatos(f"Tipo de rendimiento '{tipo}' desconocido: use 'simple' o 'log'.")
    simples = np.expm1(rendimientos) if tipo == "log" else rendimientos
    if (simples <= -1).any().any():
        raise ErrorDatos(
            "Hay rendimientos simples menores o iguales a -100 %. Si en realidad son "
            "porcentajes (por ejemplo 2.5 para un 2,5 %), dividalos antes por 100."
        )
    return precio_inicial * (1.0 + simples).cumprod()


def preparar_precios(
    datos: pd.DataFrame,
    formato: str = "ancho",
    contenido: str = "precios",
    tipo_rendimiento: str = "simple",
    columna_fecha: str | None = None,
    columna_ticker: str | None = None,
    columna_valor: str | None = None,
    precio_inicial: float = 100.0,
    dia_primero: bool = False,
) -> pd.DataFrame:
    """Convierte una tabla cualquiera en un marco de precios listo para usar.

    Es la funcion que hace el trabajo sucio: localiza las fechas, pivota si el
    formato es largo, arregla los numeros escritos como texto y, si lo que hay
    son rendimientos, reconstruye el indice de precios.
    """
    if contenido not in {"precios", "rendimientos"}:
        raise ErrorDatos(f"Contenido '{contenido}' desconocido: use 'precios' o 'rendimientos'.")

    ancho = _a_formato_ancho(
        datos, formato, columna_fecha, columna_ticker, columna_valor, dia_primero
    )
    numerico = _a_numerico(ancho)
    numerico = numerico[~numerico.index.duplicated(keep="last")]

    if contenido == "rendimientos":
        # Aqui NO se arrastra el valor anterior: repetir un rendimiento inventa
        # un movimiento que no ocurrio y descuadra la serie acumulada. Las
        # fechas incompletas simplemente se descartan.
        numerico = numerico.dropna(how="any")
        if numerico.empty:
            raise ErrorDatos("No quedan filas completas tras limpiar los rendimientos.")
        return _rendimientos_a_precios(numerico, tipo_rendimiento, precio_inicial)

    # Con precios si tiene sentido arrastrar: un festivo de un mercado que no lo
    # es de otro deja el precio donde estaba, no lo borra. Los huecos del
    # arranque no se pueden rellenar y esas filas caen.
    numerico = numerico.ffill().dropna(how="any")
    if numerico.empty:
        raise ErrorDatos(
            "No hay ninguna fecha con precio para todos los activos. Revise si las "
            "series cubren periodos distintos."
        )
    return numerico


class ProveedorMarco(ProveedorDatos):
    """Sirve precios de un ``DataFrame`` que ya esta en memoria.

    Util cuando los datos vienen de una base de datos, de otra libreria o de
    un fichero que se ha leido a mano, y se quiere pasar por todo el analisis
    sin escribirlo antes a disco.

    >>> import pandas as pd
    >>> tabla = pd.DataFrame(
    ...     {"AAA": [100.0, 101.0], "BBB": [50.0, 49.5]},
    ...     index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ... )
    >>> ProveedorMarco(tabla).descargar(["AAA"]).shape
    (2, 1)
    """

    nombre = "marco"

    def __init__(self, datos: pd.DataFrame, **opciones) -> None:
        self.precios = preparar_precios(datos, **opciones)

    @property
    def activos(self) -> list[str]:
        return list(self.precios.columns)

    def descargar(
        self,
        tickers: Sequence[str],
        inicio: str | pd.Timestamp | None = None,
        fin: str | pd.Timestamp | None = None,
        intervalo: str = "1d",
    ) -> pd.DataFrame:
        tickers = normalizar_tickers(tickers)
        faltan = [t for t in tickers if t not in self.precios.columns]
        if faltan:
            raise ErrorDatos(
                f"Los datos no contienen {faltan}. Disponibles: {self.activos}."
            )

        recorte = self.precios.loc[:, tickers]
        if inicio is not None:
            recorte = recorte.loc[recorte.index >= pd.Timestamp(inicio)]
        if fin is not None:
            recorte = recorte.loc[recorte.index <= pd.Timestamp(fin)]
        if recorte.empty:
            raise ErrorDatos("La ventana solicitada no contiene ninguna fecha de los datos.")
        return validar_marco_precios(recorte, tickers)


class ProveedorFichero(ProveedorMarco):
    """Lee los precios de un CSV, un Excel o un Parquet en disco.

    Parametros
    ----------
    ruta:
        Fichero a leer. La extension decide el lector.
    separador:
        Solo para CSV. Por defecto se deduce, lo que cubre tanto la coma como
        el punto y coma de las exportaciones espaniolas.
    decimal:
        Separador decimal del fichero. Aunque se deje en ``"."`` los numeros
        escritos con coma se reconocen igualmente al convertir.
    hoja:
        Nombre o indice de la hoja, solo para Excel.
    dia_primero:
        ``True`` si las fechas vienen en formato dia/mes/anio.
    **opciones:
        El resto se pasa a :func:`preparar_precios`: ``formato``,
        ``contenido``, ``tipo_rendimiento``, ``columna_fecha``, etc.
    """

    nombre = "fichero"

    def __init__(
        self,
        ruta: str | Path,
        separador: str | None = None,
        decimal: str = ".",
        hoja: str | int = 0,
        **opciones,
    ) -> None:
        self.ruta = Path(ruta)
        datos = leer_tabla(self.ruta, separador, decimal, hoja)
        super().__init__(datos, **opciones)

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return f"ProveedorFichero({str(self.ruta)!r}, activos={self.activos})"


def leer_tabla(
    ruta: str | Path,
    separador: str | None = None,
    decimal: str = ".",
    hoja: str | int = 0,
) -> pd.DataFrame:
    """Lee un fichero tabular sin interpretar su contenido."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise ErrorDatos(f"No existe el fichero '{ruta}'.")

    lector = LECTORES.get(ruta.suffix.lower())
    if lector is None:
        raise ErrorDatos(
            f"Extension '{ruta.suffix}' no reconocida. "
            f"Admitidas: {', '.join(sorted(LECTORES))}."
        )

    if lector == "csv":
        return pd.read_csv(
            ruta,
            sep=separador,
            decimal=decimal,
            engine="python" if separador is None else "c",
        )
    if lector == "excel":
        return pd.read_excel(ruta, sheet_name=hoja)
    return pd.read_parquet(ruta)


def cargar_precios_fichero(
    ruta: str | Path,
    tickers: Sequence[str] | None = None,
    inicio: str | pd.Timestamp | None = None,
    fin: str | pd.Timestamp | None = None,
    **opciones,
) -> pd.DataFrame:
    """Atajo para leer un fichero y quedarse con los precios directamente.

    >>> # cargar_precios_fichero("datos/precios.csv", ["AAPL", "MSFT"])
    """
    proveedor = ProveedorFichero(ruta, **opciones)
    return proveedor.descargar(tickers if tickers is not None else proveedor.activos, inicio, fin)
