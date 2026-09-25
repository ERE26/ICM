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

Nada se descarta en silencio
----------------------------
Una revision adversarial de septiembre de 2026 encontro que los cuatro
formatos reales del mercado espaniol (OMIE diario, OMIE horario, MIBGAS y
ESIOS) fallaban, y tres de ellos **sin decir nada**: un anio de OMIE entraba
con 366 dias y salia con 144. El diagnostico era comun a casi todos los
defectos: esta capa resolvia sola las ambiguedades y no contaba lo que perdia
por el camino.

De ahi la regla que ahora gobierna el modulo: **toda fila descartada, toda
celda ilegible, todo hueco rellenado y toda fecha duplicada se cuenta**, queda
en el :class:`InformeCarga` que viaja en ``attrs["informe_carga"]`` del marco
devuelto, y si pasa de un umbral pequenio detiene la carga con
:class:`~icm.comun.ErrorDatos` en lugar de devolver un marco de aspecto
impecable. Lo que no se puede decidir sin adivinar -- si ``01/02/2024`` es
enero o febrero, si las dos columnas de un formato largo son (activo, valor) o
(valor, activo) -- ya no se adivina: se pregunta.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
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

#: Codificaciones que se prueban por orden cuando no se indica ninguna. La
#: primera come el BOM que Excel escribe en su "CSV UTF-8"; la segunda es lo
#: que Excel en Espanol produce por defecto.
CODIFICACIONES = ("utf-8-sig", "cp1252", "latin-1")

#: Un numero escrito con el punto como separador de miles: grupos de tres
#: digitos exactos. "11.250" encaja (11250); "45.67" no, porque .67 son dos.
PATRON_MILES = re.compile(r"^-?\d{1,3}(\.\d{3})+$")

#: Proporcion de filas que se puede perder sin detener la carga. Un fichero
#: bien leido no pierde ninguna; perder el 1 % ya merece una explicacion.
TOLERANCIA_DESCARTES = 0.01


@dataclass
class InformeCarga:
    """Lo que ha pasado al leer un fichero, contado.

    Existe porque el fallo mas peligroso de esta capa no era equivocarse: era
    acertar a medias y callarse. Un marco con 144 de las 366 filas del fichero
    tiene exactamente el mismo aspecto que uno completo, y la unica forma de
    notarlo era contar las filas a mano.

    Viaja en ``attrs["informe_carga"]`` del marco devuelto, asi que siempre se
    puede preguntar despues con ``precios.attrs["informe_carga"].resumen()``.
    """

    filas_leidas: int = 0
    filas_sin_fecha: int = 0
    filas_incompletas: int = 0
    fechas_duplicadas: int = 0
    interpretacion_fechas: str = ""
    codificacion: str = ""
    filas_saltadas: int = 0
    celdas_ilegibles: dict[str, int] = field(default_factory=dict)
    ejemplos_ilegibles: dict[str, list[str]] = field(default_factory=dict)
    celdas_con_miles: dict[str, int] = field(default_factory=dict)
    celdas_con_porcentaje: dict[str, int] = field(default_factory=dict)
    huecos_rellenados: dict[str, int] = field(default_factory=dict)
    columnas_no_pedidas: list[str] = field(default_factory=list)

    def como_dict(self) -> dict:
        """Version serializable, que es como viaja en ``attrs``.

        Un dataclass no sobrevive a un viaje por Parquet: pandas serializa
        ``attrs`` a JSON y, al no poder, **descarta el diccionario entero** con
        un simple aviso. Guardando un dict se conserva, y el cache sigue
        funcionando.
        """
        from dataclasses import asdict

        return asdict(self)

    @property
    def filas_devueltas(self) -> int:
        return self.filas_leidas - self.filas_sin_fecha - self.filas_incompletas - self.fechas_duplicadas

    def resumen(self) -> str:
        """Una linea por cosa que merece saberse, y nada si todo fue limpio."""
        lineas = [f"Filas leidas del fichero: {self.filas_leidas}"]
        if self.filas_saltadas:
            lineas.append(f"Filas de preambulo saltadas: {self.filas_saltadas}")
        if self.codificacion:
            lineas.append(f"Codificacion usada: {self.codificacion}")
        if self.interpretacion_fechas:
            lineas.append(f"Fechas interpretadas como: {self.interpretacion_fechas}")
        for etiqueta, valor in (
            ("sin fecha interpretable", self.filas_sin_fecha),
            ("incompletas tras limpiar", self.filas_incompletas),
            ("con fecha duplicada", self.fechas_duplicadas),
        ):
            if valor:
                lineas.append(f"Filas descartadas por estar {etiqueta}: {valor}")
        for nombre, mapa in (
            ("Celdas ilegibles", self.celdas_ilegibles),
            ("Celdas con separador de miles", self.celdas_con_miles),
            ("Celdas en porcentaje", self.celdas_con_porcentaje),
            ("Huecos rellenados con el valor anterior", self.huecos_rellenados),
        ):
            if mapa:
                lineas.append(f"{nombre}: {mapa}")
        if self.ejemplos_ilegibles:
            lineas.append(f"Ejemplos de celda ilegible: {self.ejemplos_ilegibles}")
        if self.columnas_no_pedidas:
            lineas.append(f"Columnas del fichero no solicitadas: {self.columnas_no_pedidas}")
        return "\n".join(lineas)

    def comprobar(self, tolerancia: float = TOLERANCIA_DESCARTES) -> None:
        """Detiene la carga si se ha perdido mas de lo tolerable.

        El umbral se compara contra el total de filas del fichero, no contra
        las que sobreviven: si de 366 dias salen 144, lo que interesa es que
        faltan 222, no que hay 144.
        """
        if self.filas_leidas == 0:
            return
        umbral = max(1, int(tolerancia * self.filas_leidas))

        # Las celdas ilegibles van primero porque suelen ser la CAUSA de que
        # luego caigan filas, y avisar del sintoma manda a buscar en el sitio
        # equivocado.
        ilegibles = sum(self.celdas_ilegibles.values())
        if ilegibles > umbral:
            raise ErrorDatos(
                f"Hay {ilegibles} celdas que no se han podido leer como numero y se "
                "habrian tapado con el valor del dia anterior, que es indistinguible de "
                "un dato real.\n\n" + self.resumen() + "\n\n"
                "Revise el fichero, o indique 'decimal' y 'miles' si el problema es el "
                "formato de los numeros."
            )

        perdidas = self.filas_sin_fecha + self.filas_incompletas
        if perdidas > umbral:
            raise ErrorDatos(
                f"Se han descartado {perdidas} de las {self.filas_leidas} filas del fichero "
                f"({100.0 * perdidas / self.filas_leidas:.1f} %), que es demasiado para seguir "
                "sin revisarlo.\n\n" + self.resumen() + "\n\n"
                "Si las fechas van en formato dia/mes/anio indiquelo con dia_primero=True. "
                "Si el descarte es intencionado, suba 'tolerancia_descartes'."
            )


def detectar_preambulo(ruta: Path, separador: str | None = None, maximo: int = 6) -> int:
    """Cuantas lineas hay antes de la cabecera de la tabla.

    Los informes de OMIE y MIBGAS abren con una linea suelta de fecha de
    emision, y sin saltarla la cabecera real se lee como datos. Se detecta
    contando separadores: una linea de preambulo tiene muchos menos que las de
    la tabla, asi que la primera que alcanza la cuenta habitual es la
    cabecera.

    Se mira el fichero en crudo en lugar de reintentar la lectura con saltos
    crecientes, que es tentador y peligroso: un fichero con dos columnas
    llamadas igual "funciona" si se le salta la cabecera, y el usuario recibe
    un marco cuyas columnas son la primera fila de datos.
    """
    try:
        lineas = []
        with ruta.open("r", encoding="utf-8", errors="replace") as fichero:
            for _ in range(maximo + 4):
                linea = fichero.readline()
                if not linea:
                    break
                lineas.append(linea.rstrip("\n"))
    except OSError:  # pragma: no cover - el fichero existe al llegar aqui
        return 0
    if len(lineas) < 2:
        return 0

    candidatos = [separador] if separador else [";", ",", "\t", "|"]
    mejor = max(candidatos, key=lambda s: sum(linea.count(s) for linea in lineas))
    cuentas = [linea.count(mejor) for linea in lineas]
    if max(cuentas) == 0:
        return 0
    # La cabecera es la primera linea que ya trae todas las columnas.
    habitual = max(cuentas)
    for posicion, cuenta in enumerate(cuentas[:maximo]):
        if cuenta == habitual:
            return posicion
    return 0


def resumen_carga(precios: pd.DataFrame) -> str:
    """Lee el informe que viaja en ``attrs`` de un marco y lo formatea.

    >>> import pandas as pd
    >>> from icm.datos.ficheros import preparar_precios, resumen_carga
    >>> tabla = pd.DataFrame({"fecha": ["2024-01-02", "2024-01-03"], "AAA": [10.0, 11.0]})
    >>> print(resumen_carga(preparar_precios(tabla)))
    Filas leidas del fichero: 2
    Fechas interpretadas como: ISO anio-mes-dia
    """
    guardado = precios.attrs.get("informe_carga")
    if guardado is None:
        return "Este marco no lleva informe de carga."
    informe = InformeCarga(**guardado) if isinstance(guardado, dict) else guardado
    return informe.resumen()


def _clasificar_fechas(serie: pd.Series, dia_primero: bool) -> pd.Series:
    """Convierte con una interpretacion concreta, sin decidir nada.

    Deliberadamente **sin** ``format="mixed"``: ese modo infiere el formato
    elemento a elemento, de modo que lee ``31/01/2024`` como dia primero
    aunque se le pida lo contrario. Con el, las dos lecturas que compara
    :func:`convertir_fechas` interpretaban siempre el mismo numero de filas y
    la deteccion nunca podia decidir. Un formato unico por columna es ademas
    lo correcto: un fichero no cambia de convencion a mitad.
    """
    return pd.to_datetime(serie, errors="coerce", dayfirst=dia_primero)


def convertir_fechas(
    serie: pd.Series,
    dia_primero: bool | None = None,
    informe: InformeCarga | None = None,
) -> pd.Series:
    """Interpreta una columna como fechas, sin adivinar el orden dia/mes.

    Los enteros merecen un trato aparte: ``pd.to_datetime`` los lee como
    nanosegundos desde 1970, asi que ``20240102`` se convertiria en una fecha
    de 1970 sin protestar. Cuando todos los valores caen en el rango de fechas
    plausibles escritas asi, se interpretan con ese formato.

    Con ``dia_primero=None`` (lo que hay por omision) no se elige a ciegas. Se
    prueban las dos lecturas y se toma la que interpreta mas filas; si ambas
    interpretan lo mismo pero dan fechas distintas -- el caso de un fichero
    donde todos los dias son 12 o menos, como ``01/02/2024`` -- el orden es
    genuinamente ambiguo y se exige decidirlo, porque cualquier eleccion
    silenciosa intercambia dia y mes en todo el calendario sin dejar rastro.
    """
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie

    if pd.api.types.is_integer_dtype(serie):
        valores = serie.dropna()
        if not valores.empty and valores.between(19000101, 21001231).all():
            if informe is not None:
                informe.interpretacion_fechas = "enteros AAAAMMDD"
            return pd.to_datetime(
                serie.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
            )
        # Un entero que no tiene forma de AAAAMMDD no es una fecha fiable.
        return pd.Series(pd.NaT, index=serie.index)

    if pd.api.types.is_numeric_dtype(serie):
        return pd.Series(pd.NaT, index=serie.index)

    if dia_primero is not None:
        if informe is not None:
            informe.interpretacion_fechas = "dia/mes/anio" if dia_primero else "mes/dia/anio"
        return _clasificar_fechas(serie, dia_primero)

    # Una fecha ISO (AAAA-MM-DD) no admite mas que una lectura, pero pandas
    # aplica 'dayfirst' igualmente y la reinterpreta, asi que las dos pruebas
    # de abajo la declararian ambigua sin serlo. Se resuelve antes.
    texto = serie.dropna().astype(str).str.strip()
    if len(texto) and texto.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").all():
        if informe is not None:
            informe.interpretacion_fechas = "ISO anio-mes-dia"
        return pd.to_datetime(serie, errors="coerce", format="mixed", yearfirst=True)

    mes_primero = _clasificar_fechas(serie, False)
    dia_antes = _clasificar_fechas(serie, True)
    validas_mes, validas_dia = int(mes_primero.notna().sum()), int(dia_antes.notna().sum())

    if validas_dia > validas_mes:
        if informe is not None:
            informe.interpretacion_fechas = (
                f"dia/mes/anio (interpreta {validas_dia} filas frente a {validas_mes})"
            )
        return dia_antes
    if validas_mes > validas_dia:
        if informe is not None:
            informe.interpretacion_fechas = (
                f"mes/dia/anio (interpreta {validas_mes} filas frente a {validas_dia})"
            )
        return mes_primero

    # Empatan en cuantas leen. Si ademas coinciden, el orden da igual.
    if mes_primero.equals(dia_antes):
        if informe is not None:
            informe.interpretacion_fechas = "sin ambiguedad (ISO o mes no numerico)"
        return mes_primero

    distintas = int((mes_primero != dia_antes).sum())
    ejemplos = [str(v) for v in serie.dropna().astype(str).head(3)]
    raise ErrorDatos(
        f"El orden de dia y mes es ambiguo en {distintas} de las {len(serie)} fechas: "
        f"las dos lecturas son validas y dan resultados distintos (ejemplos: {ejemplos}). "
        "Indique dia_primero=True para dia/mes/anio (lo habitual en Espania) o "
        "dia_primero=False para mes/dia/anio. Adivinarlo intercambiaria dia y mes en "
        "todo el calendario sin dejar ningun rastro."
    )


def _localizar_columna_fecha(
    marco: pd.DataFrame, indicada: str | None, dia_primero: bool | None = None
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
    try:
        if convertir_fechas(marco[primera], dia_primero).notna().mean() > 0.9:
            return str(primera)
    except ErrorDatos:
        # Ambigua, pero es una columna de fechas: que lo diga quien la convierta.
        return str(primera)

    raise ErrorDatos(
        "No se ha encontrado la columna de fechas. Indiquela con 'columna_fecha'. "
        f"Columnas disponibles: {list(marco.columns)}. Si el fichero empieza con "
        "lineas de cabecera antes de la tabla, saltelas con 'filas_a_saltar'."
    )


#: Cuantos minutos dura un periodo segun cuantos quepan en el dia. Un dia
#: normal trae 24 periodos horarios o 96 cuarto-horarios; los del cambio de
#: hora traen 23 o 25 (y 92 o 100 en cuarto-horario), y siguen durando lo
#: mismo cada uno: lo que cambia es el largo del dia, no el del periodo.
DURACION_PERIODO = {23: 60, 24: 60, 25: 60, 92: 15, 96: 15, 100: 15}


def _marca_intradiaria(
    fechas: pd.Series, periodos: pd.Series, huso: str | None, informe: InformeCarga
) -> pd.Series:
    """Combina fecha y numero de periodo en una marca de tiempo.

    Con ``huso`` se localiza la medianoche de cada dia en esa zona (la
    medianoche nunca es ambigua: el cambio de hora ocurre a las 02:00) y el
    desplazamiento se suma en tiempo absoluto. Asi el dia de 25 periodos del
    ultimo domingo de octubre ocupa 25 horas reales y su ultimo periodo cae
    donde debe, en vez de pisar la medianoche del dia siguiente.

    Sin ``huso`` la marca es local ingenua, y esos dias producen dos filas con
    la misma etiqueta. No se resuelve solo: es el unico caso en que el propio
    calendario obliga a elegir, y el mensaje de fechas repetidas lo explica.
    """
    desplazamiento = _desplazamiento_intradiario(fechas, periodos, informe)
    base = fechas
    if huso is not None:
        base = base.dt.tz_localize(huso)
        informe.interpretacion_fechas += f" | marcas en huso {huso}"
    return base + desplazamiento


def _desplazamiento_intradiario(
    fechas: pd.Series, periodos: pd.Series, informe: InformeCarga
) -> pd.Series:
    """Convierte un numero de periodo (1..24, 1..96) en un desplazamiento.

    No se puede suponer que un periodo dure una hora. El 1 de octubre de 2025
    OMIE paso de casar 24 periodos horarios a casar 96 cuarto-horarios, asi que
    una misma serie descargada hoy trae los dos regimenes: tomar el numero de
    periodo como horas convertiria el periodo 96 en cuatro dias despues.

    La duracion se deduce dia a dia del numero de periodos que tiene ese dia,
    que es lo unico que la distingue y viene en los propios datos.
    """
    por_dia = periodos.groupby(fechas).transform("max")
    minutos = por_dia.map(DURACION_PERIODO)
    desconocidos = sorted(set(por_dia[minutos.isna()].unique()))
    if desconocidos:
        raise ErrorDatos(
            f"Hay dias con un numero de periodos que no se reconoce: {desconocidos}. "
            f"Se esperan {sorted(DURACION_PERIODO)} (24 u 96 en un dia normal, 23/25 o "
            "92/100 en los del cambio de hora). Combine la fecha y la hora usted mismo "
            "y pase una columna de marcas de tiempo."
        )
    regimenes = sorted(set(minutos.unique()))
    if len(regimenes) > 1:
        informe.interpretacion_fechas += (
            f" | la serie mezcla periodos de {regimenes} minutos"
        )
    return pd.to_timedelta((periodos - 1) * minutos, unit="m")


def _elegir_ticker_y_valor(marco: pd.DataFrame, restantes: list[str]) -> tuple[str, str]:
    """Decide cual de las dos columnas es el activo y cual el valor.

    Asignarlas por posicion era un defecto real: ESIOS publica
    ``(datetime, value, indicator)``, con el identificador **detras** y
    numerico, asi que la lectura posicional pivotaba al reves y devolvia un
    marco cuyas columnas se llamaban por los precios. Como todo era numerico,
    nada protestaba.

    El criterio es la cardinalidad: un identificador de activo se repite una
    vez por fecha, mientras que un precio es casi siempre distinto. Cuando la
    diferencia no es clara, se exige indicarlo a mano.
    """
    cardinalidades = {c: int(marco[c].nunique(dropna=True)) for c in restantes}
    bajo, alto = sorted(restantes, key=lambda c: cardinalidades[c])
    if cardinalidades[bajo] == 0 or cardinalidades[alto] < 2 * max(cardinalidades[bajo], 1):
        raise ErrorDatos(
            "No se puede deducir cual de las dos columnas es el activo y cual el valor: "
            f"{cardinalidades}. Indiquelas con 'columna_ticker' y 'columna_valor'. "
            "Asignarlas por su posicion pivotaria la tabla al reves si el identificador "
            "va detras del valor, como publica ESIOS."
        )
    return bajo, alto


def _a_formato_ancho(
    marco: pd.DataFrame,
    formato: str,
    columna_fecha: str | None,
    columna_ticker: str | None,
    columna_valor: str | None,
    dia_primero: bool | None,
    columna_hora: str | None,
    huso: str | None,
    informe: InformeCarga,
) -> pd.DataFrame:
    """Normaliza la tabla a una fila por fecha y una columna por activo."""
    if formato not in {"ancho", "largo"}:
        raise ErrorDatos(f"Formato '{formato}' desconocido: use 'ancho' o 'largo'.")

    marco = marco.copy()
    marco.columns = [str(c).strip() for c in marco.columns]
    informe.filas_leidas = len(marco)

    # Si el indice ya es temporal y no hay columna de fechas, se respeta.
    if isinstance(marco.index, pd.DatetimeIndex) and formato == "ancho":
        ancho = marco
    else:
        fecha = _localizar_columna_fecha(marco, columna_fecha, dia_primero)
        marco[fecha] = convertir_fechas(marco[fecha], dia_primero, informe)
        if marco[fecha].isna().all():
            raise ErrorDatos(f"No se ha podido interpretar ninguna fecha en la columna '{fecha}'.")

        if columna_hora is not None:
            if columna_hora not in marco.columns:
                raise ErrorDatos(
                    f"La columna de horas '{columna_hora}' no esta en el fichero. "
                    f"Columnas disponibles: {list(marco.columns)}."
                )
            periodos = pd.to_numeric(marco[columna_hora], errors="coerce")
            if periodos.isna().any():
                raise ErrorDatos(
                    f"La columna de horas '{columna_hora}' tiene valores no numericos."
                )
            marco[fecha] = _marca_intradiaria(marco[fecha], periodos, huso, informe)
            marco = marco.drop(columns=[columna_hora])

        informe.filas_sin_fecha = int(marco[fecha].isna().sum())
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
                columna_ticker, columna_valor = _elegir_ticker_y_valor(marco, restantes)
            duplicadas = int(marco.duplicated(subset=[fecha, columna_ticker]).sum())
            if duplicadas:
                raise ErrorDatos(
                    f"Hay {duplicadas} pares (fecha, activo) repetidos en el fichero. "
                    "Agregarlos en silencio elegiria una de las dos publicaciones por su "
                    "posicion en el fichero, que no es la mas reciente. Decida antes cual "
                    "se queda, o pase 'resolver_duplicados'."
                )
            ancho = marco.pivot(index=fecha, columns=columna_ticker, values=columna_valor)

    ancho.index = pd.DatetimeIndex(ancho.index, freq=None, name="fecha")
    nombres = [str(c).strip().upper() for c in ancho.columns]
    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
    if repetidos:
        raise ErrorDatos(
            f"Estas columnas quedan con el mismo nombre al normalizar: {repetidos}. "
            "Renombrelas en el fichero: con nombres repetidos no hay forma de saber "
            "cual es cual."
        )
    ancho.columns = nombres
    return ancho.sort_index()


def _limpiar_columna_texto(
    serie: pd.Series, columna: str, informe: InformeCarga
) -> pd.Series:
    """Convierte una columna de texto a numero contando lo que no entiende."""
    texto = (
        serie.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
    )

    # El porcentaje no se borra, se divide. Quitar el simbolo y dejar el
    # numero convertia un 1,5 % en un 1,5 y, con contenido="rendimientos",
    # en un +150 % diario.
    es_porcentaje = texto.str.endswith("%")
    if es_porcentaje.any():
        informe.celdas_con_porcentaje[columna] = int(es_porcentaje.sum())
        texto = texto.str.rstrip("%")

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

    # Sin coma decimal, un punto entre grupos de tres digitos exactos es
    # separador de miles: "11.250" son once mil doscientos cincuenta, no
    # 11,25. Leerlo como decimal dividia el volumen por mil.
    solo_punto = ~con_ambos
    es_miles = solo_punto & texto.str.fullmatch(PATRON_MILES.pattern).fillna(False)
    if es_miles.any():
        informe.celdas_con_miles[columna] = int(es_miles.sum())
        texto = texto.mask(es_miles, texto.str.replace(".", "", regex=False))

    texto = texto.mask(solo_punto & ~es_miles, texto.str.replace(",", ".", regex=False))
    numero = pd.to_numeric(texto, errors="coerce")
    if es_porcentaje.any():
        numero = numero.mask(es_porcentaje, numero / 100.0)

    # Una celda que tenia algo escrito y no se ha podido leer es un dato
    # perdido, no un hueco: hay que contarla antes de que el ffill la tape
    # con el valor de ayer.
    habia_algo = serie.notna() & (serie.astype(str).str.strip() != "")
    perdidas = habia_algo & numero.isna()
    if perdidas.any():
        informe.celdas_ilegibles[columna] = int(perdidas.sum())
        informe.ejemplos_ilegibles[columna] = [
            str(v) for v in serie[perdidas].astype(str).head(3)
        ]
    return numero


def _a_numerico(marco: pd.DataFrame, informe: InformeCarga) -> pd.DataFrame:
    """Convierte a numero las columnas que hayan llegado como texto.

    Es lo habitual con los CSV exportados desde una hoja de calculo espaniola:
    el separador de miles y el decimal viajan dentro de la cadena.
    """
    convertido = {}
    for columna in marco.columns:
        serie = marco[columna]
        if pd.api.types.is_numeric_dtype(serie):
            convertido[columna] = serie.astype(float)
        else:
            convertido[columna] = _limpiar_columna_texto(serie, columna, informe)

    salida = pd.DataFrame(convertido, index=marco.index)
    vacias = [c for c in salida.columns if salida[c].isna().all()]
    if vacias:
        raise ErrorDatos(
            f"Estas columnas no contienen numeros interpretables: {vacias}. "
            "Si son columnas de texto (un identificador, una etiqueta), el fichero "
            "esta en formato largo: pase formato='largo'."
        )
    return salida


def _rendimientos_a_precios(
    rendimientos: pd.DataFrame,
    tipo: str,
    precio_inicial: float,
) -> pd.DataFrame:
    """Reconstruye un indice de precios a partir de una tabla de rendimientos.

    Se antepone una fila con el ``precio_inicial``, fechada un periodo antes
    del primer rendimiento, para que el ciclo sea reversible: sin ella el
    primer rendimiento se consumia al fijar el nivel y no habia forma de
    recuperarlo.
    """
    if tipo not in {"simple", "log"}:
        raise ErrorDatos(f"Tipo de rendimiento '{tipo}' desconocido: use 'simple' o 'log'.")
    simples = np.expm1(rendimientos) if tipo == "log" else rendimientos
    if (simples <= -1).any().any():
        raise ErrorDatos(
            "Hay rendimientos simples menores o iguales a -100 %. Si en realidad son "
            "porcentajes (por ejemplo 2.5 para un 2,5 %), dividalos antes por 100."
        )
    precios = precio_inicial * (1.0 + simples).cumprod()
    if len(precios.index) >= 2:
        paso = precios.index[1] - precios.index[0]
        inicial = pd.DataFrame(
            [[precio_inicial] * precios.shape[1]],
            index=[precios.index[0] - paso],
            columns=precios.columns,
        )
        precios = pd.concat([inicial, precios])
        precios.index.name = "fecha"
    return precios


def preparar_precios(
    datos: pd.DataFrame,
    formato: str = "ancho",
    contenido: str = "precios",
    tipo_rendimiento: str = "simple",
    columna_fecha: str | None = None,
    columna_ticker: str | None = None,
    columna_valor: str | None = None,
    columna_hora: str | None = None,
    huso: str | None = None,
    precio_inicial: float = 100.0,
    dia_primero: bool | None = None,
    resolver_duplicados: str = "error",
    limite_relleno: int | None = 5,
    tolerancia_descartes: float = TOLERANCIA_DESCARTES,
    limpiar: bool = True,
) -> pd.DataFrame:
    """Convierte una tabla cualquiera en un marco de precios listo para usar.

    Es la funcion que hace el trabajo sucio: localiza las fechas, pivota si el
    formato es largo, arregla los numeros escritos como texto y, si lo que hay
    son rendimientos, reconstruye el indice de precios.

    Parametros
    ----------
    columna_hora:
        Nombre de una columna de horas que se combina con la fecha para formar
        una marca de tiempo. Es el formato nativo de OMIE (``Fecha;Hora;Precio``),
        que sin esto colapsaba las 24 filas de cada dia en una sola.
    huso:
        Zona horaria con la que fechar los periodos intradiarios, por ejemplo
        ``"Europe/Madrid"``. Hace falta el dia del cambio de hora de octubre,
        que tiene 25 periodos y sin huso deja dos con la misma etiqueta.
    dia_primero:
        ``True`` para dia/mes/anio, ``False`` para mes/dia/anio, ``None`` (lo
        que hay por omision) para deducirlo y detenerse si es ambiguo.
    resolver_duplicados:
        Que hacer con dos filas de la misma fecha: ``"error"`` (por omision),
        ``"ultimo"``, ``"primero"`` o ``"media"``. Antes se quedaba con la
        ultima del fichero en silencio, lo que en un fichero ordenado de mas
        reciente a mas antiguo se queda con la version **vieja**, y en un
        fichero horario del cambio de hora de octubre borra una hora real.
    limite_relleno:
        Maximo de periodos consecutivos que se arrastra un precio hacia
        delante. ``None`` para no limitarlo. Sin limite, un hueco largo se
        convertia en una meseta perfecta indistinguible de un mercado parado.
    tolerancia_descartes:
        Proporcion de filas que se puede perder antes de detener la carga.
    limpiar:
        ``False`` devuelve el marco sin arrastrar precios ni tirar fechas
        incompletas, para que quien vaya a quedarse con un subconjunto de
        activos pueda limpiarlo despues sin que las demas series le recorten
        la suya. Es lo que hace :class:`ProveedorMarco`.

    El marco devuelto lleva el recuento de todo lo descartado en
    ``attrs["informe_carga"]``.
    """
    if contenido not in {"precios", "rendimientos"}:
        raise ErrorDatos(f"Contenido '{contenido}' desconocido: use 'precios' o 'rendimientos'.")
    if resolver_duplicados not in {"error", "ultimo", "primero", "media"}:
        raise ErrorDatos(
            f"resolver_duplicados='{resolver_duplicados}' desconocido: use 'error', "
            "'ultimo', 'primero' o 'media'."
        )

    informe = InformeCarga()
    ancho = _a_formato_ancho(
        datos, formato, columna_fecha, columna_ticker, columna_valor,
        dia_primero, columna_hora, huso, informe,
    )
    numerico = _a_numerico(ancho, informe)

    repetidas = int(numerico.index.duplicated().sum())
    if repetidas:
        informe.fechas_duplicadas = repetidas
        if resolver_duplicados == "error":
            muestra = numerico.index[numerico.index.duplicated()][:3]
            raise ErrorDatos(
                f"Hay {repetidas} fechas repetidas en el fichero (por ejemplo "
                f"{[str(f) for f in muestra]}). Quedarse con una en silencio elige por "
                "posicion en el fichero, no por fecha de publicacion, y en una serie "
                "horaria borra la hora repetida del cambio de hora de octubre. Decida con "
                "resolver_duplicados='ultimo', 'primero' o 'media', o use 'columna_hora' "
                "si lo que hay son varias horas por dia."
            )
        if resolver_duplicados == "media":
            numerico = numerico.groupby(level=0).mean()
        else:
            mantener = "last" if resolver_duplicados == "ultimo" else "first"
            numerico = numerico[~numerico.index.duplicated(keep=mantener)]

    if contenido == "rendimientos":
        # Aqui NO se arrastra el valor anterior: repetir un rendimiento inventa
        # un movimiento que no ocurrio y descuadra la serie acumulada. Las
        # fechas incompletas simplemente se descartan.
        antes = len(numerico)
        numerico = numerico.dropna(how="any")
        informe.filas_incompletas = antes - len(numerico)
        informe.comprobar(tolerancia_descartes)
        if numerico.empty:
            raise ErrorDatos("No quedan filas completas tras limpiar los rendimientos.")
        salida = _rendimientos_a_precios(numerico, tipo_rendimiento, precio_inicial)
        salida.attrs["informe_carga"] = informe.como_dict()
        return salida

    if limpiar:
        numerico = limpiar_precios(numerico, limite_relleno, informe, tolerancia_descartes)
    numerico.attrs["informe_carga"] = informe.como_dict()
    return numerico


def limpiar_precios(
    numerico: pd.DataFrame,
    limite_relleno: int | None,
    informe: InformeCarga,
    tolerancia: float = TOLERANCIA_DESCARTES,
) -> pd.DataFrame:
    """Arrastra los precios hacia delante y tira las fechas incompletas.

    Con precios si tiene sentido arrastrar: un festivo de un mercado que no lo
    es de otro deja el precio donde estaba, no lo borra. Los huecos del
    arranque no se pueden rellenar y esas filas caen.

    Va aparte de :func:`preparar_precios` porque hay que poder aplicarla
    **despues** de elegir los activos. Hacerla antes significaba que una serie
    que empieza mas tarde recortaba a las demas aunque nadie la hubiese
    pedido: un fichero con ES y FR devolvia 2 de los 4 dias de ES por culpa de
    FR.
    """
    for columna in numerico.columns:
        huecos = int(numerico[columna].isna().sum())
        if huecos:
            informe.huecos_rellenados[columna] = huecos
    numerico = numerico.ffill(limit=limite_relleno)
    antes = len(numerico)
    numerico = numerico.dropna(how="any")
    informe.filas_incompletas = antes - len(numerico)
    informe.comprobar(tolerancia)
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

    La limpieza final (arrastrar precios y tirar filas incompletas) se aplica
    **sobre los activos que se piden**, no sobre todo el fichero. Hacerlo antes
    significaba que una serie que empieza mas tarde recortaba las demas aunque
    nadie la hubiese pedido: un fichero con ES y FR devolvia 2 de los 4 dias de
    ES por culpa de FR.

    >>> import pandas as pd
    >>> tabla = pd.DataFrame(
    ...     {"AAA": [100.0, 101.0], "BBB": [50.0, 49.5]},
    ...     index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ... )
    >>> ProveedorMarco(tabla).descargar(["AAA"]).shape
    (2, 1)
    """

    _nombre_base = "marco"

    def __init__(self, datos: pd.DataFrame, **opciones) -> None:
        self._limite_relleno = opciones.get("limite_relleno", 5)
        self._tolerancia = opciones.get("tolerancia_descartes", TOLERANCIA_DESCARTES)
        # Sin limpiar: el relleno y el descarte de fechas incompletas se
        # aplican en 'descargar', ya sobre los activos que se piden.
        self.precios = preparar_precios(datos, limpiar=False, **opciones)

    @property
    def nombre(self) -> str:  # type: ignore[override]
        """Nombre con huella del contenido, para que el cache distinga marcos.

        ``CacheParquet.clave`` se construye con el nombre del proveedor, los
        tickers y las fechas. Con un nombre fijo, dos marcos distintos con los
        mismos activos compartian entrada y el segundo recibia los datos del
        primero. La huella (forma, columnas y extremos del indice) es barata y
        basta para separarlos.
        """
        return f"{self._nombre_base}_{self._huella()}"

    def _huella(self) -> str:
        import hashlib

        indice = self.precios.index
        descriptor = "|".join(
            [
                str(self.precios.shape),
                ",".join(map(str, self.precios.columns)),
                str(indice.min()) if len(indice) else "",
                str(indice.max()) if len(indice) else "",
            ]
        )
        return hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:12]

    @property
    def informe_carga(self) -> InformeCarga | None:
        """El recuento de lo que paso al leer, o ``None`` si no hay."""
        guardado = self.precios.attrs.get("informe_carga")
        return InformeCarga(**guardado) if isinstance(guardado, dict) else guardado

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

        informe = self.informe_carga or InformeCarga()
        recorte = limpiar_precios(
            self.precios.loc[:, tickers], self._limite_relleno, informe, self._tolerancia
        )
        if inicio is not None:
            recorte = recorte.loc[recorte.index >= _al_huso(inicio, recorte.index)]
        if fin is not None:
            recorte = recorte.loc[recorte.index <= _al_huso(fin, recorte.index)]
        if recorte.empty:
            raise ErrorDatos("La ventana solicitada no contiene ninguna fecha de los datos.")
        salida = validar_marco_precios(recorte, tickers)
        informe.columnas_no_pedidas = [c for c in self.precios.columns if c not in tickers]
        salida.attrs["informe_carga"] = informe.como_dict()
        return salida


def _al_huso(momento, indice: pd.DatetimeIndex) -> pd.Timestamp:
    """Pone un limite de fecha en el mismo huso que el indice.

    Comparar un ``Timestamp`` sin huso con un indice que si lo tiene lanzaba un
    ``TypeError`` de pandas, que no es un ``ErrorDatos`` y por tanto se escapaba
    del respaldo de ``cargar_precios``.
    """
    marca = pd.Timestamp(momento)
    if indice.tz is not None and marca.tz is None:
        return marca.tz_localize(indice.tz)
    if indice.tz is None and marca.tz is not None:
        return marca.tz_convert(None)
    return marca


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
    codificacion:
        ``None`` para probar utf-8 y luego cp1252, que es lo que exporta Excel
        en Espania. Sin esto, un CSV con una tilde en la cabecera no habia
        forma de leerlo.
    miles:
        Separador de miles que pasar al lector. Independiente de la deteccion
        por celda, que funciona igual sin indicarlo.
    filas_a_saltar:
        Lineas de preambulo antes de la tabla. ``None`` las busca: los informes
        de OMIE y MIBGAS empiezan con una linea de fecha de emision.
    dia_primero:
        ``True`` si las fechas vienen en formato dia/mes/anio.
    **opciones:
        El resto se pasa a :func:`preparar_precios`: ``formato``,
        ``contenido``, ``columna_hora``, ``resolver_duplicados``, etc.
    """

    _nombre_base = "fichero"

    @property
    def nombre(self) -> str:  # type: ignore[override]
        """Nombre con la ruta y el estado del fichero dentro.

        Antes valia ``"fichero"`` para todos, asi que dos CSV distintos con los
        mismos activos compartian entrada de cache y el segundo recibia los
        datos del primero. Y como el cache solo miraba su propia antiguedad,
        modificar el fichero de origen no invalidaba nada: con
        ``caducidad_horas=None`` se servian los datos viejos para siempre.
        Metiendo la fecha de modificacion y el tamanio en la huella, cualquier
        cambio del origen cambia la clave y el problema se resuelve solo.
        """
        import hashlib

        try:
            estado = self.ruta.stat()
            marca = f"{estado.st_mtime_ns}:{estado.st_size}"
        except OSError:  # pragma: no cover - el fichero existia al construir
            marca = "sin_estado"
        descriptor = f"{self.ruta.resolve()}|{marca}"
        return f"fichero_{hashlib.sha256(descriptor.encode('utf-8')).hexdigest()[:12]}"

    def __init__(
        self,
        ruta: str | Path,
        separador: str | None = None,
        decimal: str = ".",
        hoja: str | int = 0,
        codificacion: str | None = None,
        miles: str | None = None,
        filas_a_saltar: int | None = None,
        **opciones,
    ) -> None:
        self.ruta = Path(ruta)
        # Solo en ficheros de texto: abrir un .xlsx como texto y contar
        # separadores dentro del binario del zip da un numero arbitrario, y
        # como ese binario cambia entre escrituras el fallo resultante es
        # intermitente.
        es_texto = LECTORES.get(self.ruta.suffix.lower()) == "csv"
        salto = (
            filas_a_saltar
            if filas_a_saltar is not None
            else (detectar_preambulo(self.ruta, separador) if es_texto else 0)
        )
        datos = leer_tabla(self.ruta, separador, decimal, hoja, codificacion, miles, salto)
        super().__init__(datos, **opciones)
        self._anotar_salto(salto)

    def _anotar_salto(self, salto: int) -> None:
        """Deja constancia de cuantas lineas de preambulo se saltaron.

        Se escribe en el diccionario de ``attrs``, no sobre la propiedad
        ``informe_carga``, que devuelve un objeto reconstruido: asignarle un
        atributo no llegaria al marco.
        """
        guardado = self.precios.attrs.get("informe_carga")
        if isinstance(guardado, dict):
            guardado["filas_saltadas"] = salto

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return f"ProveedorFichero({str(self.ruta)!r}, activos={self.activos})"


def leer_tabla(
    ruta: str | Path,
    separador: str | None = None,
    decimal: str = ".",
    hoja: str | int = 0,
    codificacion: str | None = None,
    miles: str | None = None,
    filas_a_saltar: int = 0,
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
        candidatas = (codificacion,) if codificacion is not None else CODIFICACIONES
        ultimo: Exception | None = None
        for cod in candidatas:
            try:
                return pd.read_csv(
                    ruta,
                    sep=separador,
                    decimal=decimal,
                    thousands=miles,
                    encoding=cod,
                    skiprows=filas_a_saltar,
                    engine="python" if separador is None else "c",
                )
            except UnicodeDecodeError as error:
                ultimo = error
            except (ValueError, csv.Error, pd.errors.ParserError) as error:
                raise ErrorDatos(f"No se ha podido interpretar '{ruta}': {error}") from error
        raise ErrorDatos(
            f"No se ha podido decodificar '{ruta}' con {list(candidatas)}. "
            "Indique la codificacion con 'codificacion'."
        ) from ultimo

    if lector == "excel":
        if hoja is None:
            raise ErrorDatos(
                "hoja=None leeria todas las hojas a la vez. Indique el nombre o el "
                "indice de una sola."
            )
        try:
            return pd.read_excel(ruta, sheet_name=hoja, skiprows=filas_a_saltar)
        except ValueError as error:
            raise ErrorDatos(f"No se ha podido leer la hoja '{hoja}' de '{ruta}': {error}") from error
    return pd.read_parquet(ruta)


def cargar_precios_fichero(
    ruta: str | Path,
    tickers: Sequence[str] | None = None,
    inicio: str | pd.Timestamp | None = None,
    fin: str | pd.Timestamp | None = None,
    **opciones,
) -> pd.DataFrame:
    """Atajo para leer un fichero y quedarse con los precios directamente.

    Las opciones de :class:`ProveedorFichero` y de :func:`preparar_precios`
    valen aqui: ``dia_primero``, ``columna_hora``, ``resolver_duplicados``,
    ``codificacion``, ``filas_a_saltar``...

    >>> # cargar_precios_fichero("datos/omie.csv", dia_primero=True)
    """
    proveedor = ProveedorFichero(ruta, **opciones)
    return proveedor.descargar(tickers if tickers is not None else proveedor.activos, inicio, fin)
