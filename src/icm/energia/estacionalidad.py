"""Estacionalidad de las series de energia: perfiles, descomposicion y contraste.

Por que un modulo propio
------------------------
Una accion no sabe en que mes vive. La energia si: hace frio todos los
inviernos y calor todos los veranos, la demanda sigue al termometro y el
calendario explica una parte enorme de la varianza del precio. Esa parte es
**predecible sin informacion de mercado**, y ahi esta el problema: si no se
separa, un modelo de aprendizaje automatico se gasta toda su capacidad en
reaprender el calendario (que ya se sabe) en lugar de la senal que queda
cuando el calendario se descuenta (que es la que da dinero). Un modelo que
"acierta" porque sabe que en enero los precios son altos no esta prediciendo
nada; esta leyendo el almanaque.

De ahi las tres cosas que hace este modulo, en este orden:

1. **Describir** el patron: :func:`perfil_estacional` y :func:`tabla_perfiles`
   dicen cuanto vale la serie en cada mes, dia de la semana o trimestre.
2. **Separarlo**: :func:`descomponer` parte la serie en tendencia, estacional
   y residuo, y :func:`desestacionalizar` devuelve la serie sin el calendario,
   que es la que hay que modelar.
3. **Contrastarlo**: :func:`contraste_estacionalidad` responde a si el patron
   que se ve en el grafico es real o es ruido, con un contraste que no exige
   normalidad porque los precios de energia no son normales ni de lejos.

Convenciones
------------
* Las funciones que agrupan por calendario (``perfil_estacional``,
  ``tabla_perfiles``, ``indice_estacional``, ``contraste_estacionalidad``)
  exigen un ``DatetimeIndex``: sin fechas no hay calendario.
* Las funciones que descomponen (``descomponer``, ``desestacionalizar``)
  trabajan con un ``periodo`` en **numero de observaciones**, no en dias
  naturales: 252 para un ciclo anual de datos diarios habiles, 12 para datos
  mensuales, 7 para un ciclo semanal de datos diarios.
* El modelo ``"multiplicativo"`` exige serie estrictamente positiva. No es un
  capricho de implementacion: dividir por una tendencia que pasa por cero
  produce factores estacionales infinitos, y con precios negativos (que en el
  pool electrico existen) el cociente cambia de signo y deja de significar
  nada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from icm.comun import ErrorDatos, como_serie

#: Agrupaciones de calendario admitidas.
AGRUPACIONES = ("mes", "dia_semana", "dia_anio", "trimestre")

#: Estadisticos de resumen admitidos en los perfiles.
ESTADISTICOS = ("media", "mediana")

#: Formas de combinar tendencia y estacionalidad.
MODELOS = ("aditivo", "multiplicativo")

#: Nombres de los meses, en el orden de ``DatetimeIndex.month`` (1 = enero).
MESES = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)

#: Nombres de los dias de la semana, en el orden de ``DatetimeIndex.dayofweek``.
#: Se repiten aqui en vez de importarlos de ``icm.ml`` para que el analisis
#: descriptivo de un mercado no dependa del modulo de modelos.
DIAS_SEMANA = (
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
)

#: Base de calefaccion convencional del sector, en grados Celsius (los 65 F
#: estadounidenses). Por debajo de ella se enciende la caldera.
BASE_CALEFACCION = 15.5


# --- validaciones ---------------------------------------------------------------


def _serie_fechada(datos, nombre: str = "serie") -> pd.Series:
    """Serie de float con indice de fechas, que es lo que exige el calendario."""
    serie = como_serie(datos, nombre)
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ErrorDatos(
            f"'{nombre}' necesita un DatetimeIndex para agrupar por calendario y "
            f"tiene un indice de tipo {type(serie.index).__name__}. Convierta el "
            "indice con pd.to_datetime(...) o pase la serie con sus fechas."
        )
    return serie


def _validar_opcion(valor: str, opciones: tuple[str, ...], nombre: str) -> str:
    if valor not in opciones:
        raise ErrorDatos(
            f"'{nombre}' no admite el valor '{valor}'. Use uno de: {', '.join(opciones)}."
        )
    return valor


def _validar_entero(valor, nombre: str, minimo: int = 1) -> int:
    try:
        entero = int(valor)
    except (TypeError, ValueError) as exc:
        raise ErrorDatos(f"'{nombre}' debe ser un entero, se ha recibido {valor!r}.") from exc
    if entero != valor or entero < minimo:
        raise ErrorDatos(
            f"'{nombre}' debe ser un entero mayor o igual que {minimo}, "
            f"se ha recibido {valor!r}."
        )
    return entero


def _exigir_positiva(serie: pd.Series, contexto: str) -> None:
    """El modelo multiplicativo solo tiene sentido con valores positivos."""
    if (serie <= 0.0).any():
        minimo = float(serie.min())
        raise ErrorDatos(
            f"El modelo multiplicativo de {contexto} exige una serie estrictamente "
            f"positiva y el minimo es {minimo:g}. Con ceros o negativos el cociente "
            "serie/tendencia se dispara o cambia de signo y el factor estacional deja "
            "de significar nada. Use modelo='aditivo', que admite cualquier signo, o "
            "desplace la serie a positivos antes de descomponerla."
        )


# --- perfiles de calendario -----------------------------------------------------


def _claves(indice: pd.DatetimeIndex, por: str) -> pd.Index:
    """Clave de agrupacion de cada observacion segun la particion pedida."""
    _validar_opcion(por, AGRUPACIONES, "por")
    if por == "mes":
        valores = indice.month
    elif por == "dia_semana":
        valores = indice.dayofweek
    elif por == "dia_anio":
        valores = indice.dayofyear
    else:
        valores = indice.quarter
    return pd.Index(np.asarray(valores), name=por)


def etiqueta_grupo(por: str, clave: int) -> str:
    """Nombre legible de un grupo, para la tabla que acaba en el informe."""
    _validar_opcion(por, AGRUPACIONES, "por")
    if por == "mes":
        return MESES[int(clave) - 1]
    if por == "dia_semana":
        return DIAS_SEMANA[int(clave)]
    if por == "trimestre":
        return f"T{int(clave)}"
    return f"dia {int(clave)}"


def perfil_estacional(
    serie,
    por: str = "mes",
    estadistico: str = "media",
) -> pd.Series:
    """Valor tipico de la serie en cada posicion del calendario.

    Parametros
    ----------
    por:
        ``"mes"`` para el ciclo anual de la demanda termica, ``"dia_semana"``
        para el ciclo de actividad economica (el pool cae los domingos porque
        la industria para), ``"dia_anio"`` para el detalle fino de la curva
        anual y ``"trimestre"`` para la agregacion con la que cotizan los
        productos a plazo.
    estadistico:
        ``"media"`` o ``"mediana"``.

    Por que la mediana suele ser mejor
    ----------------------------------
    Los precios de energia tienen picos: un dia de ola de frio con la
    interconexion saturada puede multiplicar por diez el precio horario. Ese
    dia entra en la media de su mes con todo su peso y desplaza el perfil
    entero, de modo que "enero" acaba describiendo un episodio y no un mes.
    La mediana tiene un punto de ruptura del 50 %: haria falta que la mitad de
    los eneros de la muestra fuesen extremos para moverla. Para describir lo
    que pasa **normalmente** en enero, que es para lo que sirve un perfil, la
    mediana es la eleccion prudente; la media se reserva para cuando hay que
    conservar la suma (facturacion, volumenes, balances de energia), porque la
    mediana no es aditiva.
    """
    valores = _serie_fechada(serie, "serie")
    _validar_opcion(estadistico, ESTADISTICOS, "estadistico")
    claves = _claves(valores.index, por)

    agrupado = valores.groupby(claves)
    perfil = agrupado.median() if estadistico == "mediana" else agrupado.mean()
    perfil = perfil.sort_index()
    perfil.index.name = por
    perfil.name = f"{estadistico}_por_{por}"
    return perfil


def tabla_perfiles(serie) -> pd.DataFrame:
    """Perfiles por mes y por dia de la semana en una sola tabla.

    Son las dos agrupaciones que se presentan juntas en un informe porque
    responden a preguntas distintas: el mes recoge el ciclo **termico** (frio,
    calor, mantenimiento de centrales en primavera y otono) y el dia de la
    semana el ciclo **economico** (la industria no trabaja el domingo). Verlas
    a la vez evita la confusion clasica de atribuir al mes lo que en realidad
    es un efecto de composicion de dias habiles.

    Se devuelven media y mediana una al lado de la otra a proposito: cuando se
    separan mucho en un grupo, ese grupo tiene picos y hay que mirarlo antes de
    contar nada. La columna ``indice_aditivo`` es la desviacion del grupo
    respecto a la media global, que es como se lee un perfil de un vistazo
    (``+8,4 EUR/MWh en enero``), y funciona con cualquier signo, tambien con
    los precios negativos del pool.
    """
    valores = _serie_fechada(serie, "serie")
    media_global = float(valores.mean())

    filas = []
    for por in ("mes", "dia_semana"):
        claves = _claves(valores.index, por)
        agrupado = valores.groupby(claves)
        medias = agrupado.mean().sort_index()
        medianas = agrupado.median().sort_index()
        desviaciones = agrupado.std(ddof=1).sort_index()
        conteos = agrupado.count().sort_index()
        for clave in medias.index:
            filas.append(
                {
                    "agrupacion": por,
                    "grupo": int(clave),
                    "etiqueta": etiqueta_grupo(por, clave),
                    "observaciones": int(conteos.loc[clave]),
                    "media": float(medias.loc[clave]),
                    "mediana": float(medianas.loc[clave]),
                    "desviacion": float(desviaciones.loc[clave]),
                    "indice_aditivo": float(medias.loc[clave]) - media_global,
                }
            )

    tabla = pd.DataFrame(filas).set_index(["agrupacion", "grupo"])
    tabla.attrs["media_global"] = media_global
    return tabla


def indice_estacional(
    serie,
    por: str = "mes",
    modelo: str = "multiplicativo",
) -> pd.Series:
    """Perfil normalizado: media 1 si es multiplicativo, media 0 si es aditivo.

    Es la forma en que se publica un patron estacional, y la razon es que asi
    se puede leer sin conocer el nivel: "enero es un 18 % mas caro que un mes
    medio" se entiende y se compara entre mercados y entre anios, mientras que
    "enero vale 84 EUR/MWh" solo describe la muestra.

    La normalizacion se hace con la media de las medias de grupo, **no** con la
    media de la serie. Los grupos no son igual de grandes (los meses tienen
    entre 28 y 31 dias, el calendario habil reparte fiestas de forma desigual,
    los anios bisiestos aniaden un dia), asi que dividir por la media global
    daria un indice que no promedia exactamente 1 y una suma de desviaciones
    que no es cero: el patron parece tener un sesgo que solo viene de la
    longitud de los meses.

    El argumento ``modelo`` decide la escala: ``"multiplicativo"`` para series
    cuya amplitud estacional crece con el nivel (lo normal en precios) y
    ``"aditivo"`` para series en unidades fisicas o con signo variable.
    """
    _validar_opcion(modelo, MODELOS, "modelo")
    valores = _serie_fechada(serie, "serie")
    if modelo == "multiplicativo":
        _exigir_positiva(valores, "el indice estacional")
    perfil = perfil_estacional(valores, por=por, estadistico="media")

    if modelo == "multiplicativo":
        indice = perfil / float(perfil.mean())
    else:
        indice = perfil - float(perfil.mean())

    indice.name = f"indice_{modelo}_por_{por}"
    return indice


# --- descomposicion clasica -----------------------------------------------------


def _media_movil_centrada(serie: pd.Series, ventana: int) -> pd.Series:
    """Media movil centrada, con la correccion 2xM cuando la ventana es par.

    Una media movil centrada de ventana par no existe tal cual: con 12 meses el
    centro cae entre dos observaciones. La solucion clasica es promediar dos
    ventanas consecutivas, lo que equivale a pesos ``1/2, 1, ..., 1, 1/2``
    divididos por la ventana. Sin esa correccion la "tendencia" queda
    desplazada medio periodo y se cuela una parte de la estacionalidad dentro
    de ella, que es justo lo que la descomposicion pretende evitar.
    """
    if ventana % 2 == 1:
        pesos = np.full(ventana, 1.0 / ventana)
    else:
        pesos = np.full(ventana + 1, 1.0 / ventana)
        pesos[0] = pesos[-1] = 0.5 / ventana

    n = len(serie)
    ancho = len(pesos)
    if ancho > n:
        raise ErrorDatos(
            f"La ventana de tendencia efectiva ({ancho} observaciones) no cabe en una "
            f"serie de {n}. Reduzca 'ventana_tendencia' o alargue la muestra."
        )

    centro = (ancho - 1) // 2
    suavizada = np.full(n, np.nan)
    suavizada[centro : n - centro] = np.convolve(serie.to_numpy(float), pesos, mode="valid")
    return pd.Series(suavizada, index=serie.index, name="tendencia")


@dataclass
class Descomposicion:
    """Las tres piezas en que se parte una serie, mas como recomponerlas.

    Atributos
    ---------
    tendencia:
        Media movil centrada. Vale ``NaN`` en los extremos que la ventana no
        cubre: media ventana por cada lado. No se rellena ni se extrapola, y es
        deliberado: un valor inventado en el extremo es exactamente donde mas
        dano hace, porque el extremo derecho es el presente.
    estacional:
        Patron que se repite cada ``periodo`` observaciones, definido en toda
        la muestra. Suma cero en el modelo aditivo y promedia uno en el
        multiplicativo.
    residuo:
        Lo que queda. Es la serie que de verdad hay que modelar: ni la
        tendencia ni el calendario son noticia.
    periodo:
        Longitud del ciclo en numero de observaciones.
    modelo:
        ``"aditivo"`` o ``"multiplicativo"``. Se guarda porque sin el la
        reconstruccion tendria que adivinar si las piezas se suman o se
        multiplican, y las dos opciones dan resultados plausibles.
    """

    tendencia: pd.Series
    estacional: pd.Series
    residuo: pd.Series
    periodo: int
    modelo: str = "aditivo"

    def __post_init__(self) -> None:
        _validar_opcion(self.modelo, MODELOS, "modelo")

    def reconstruir(self) -> pd.Series:
        """Recompone la serie original a partir de las tres piezas.

        Devuelve la serie de partida salvo en los extremos, donde la media
        movil no llega y la tendencia es ``NaN``. Que la reconstruccion cuadre
        no es un detalle estetico: es la comprobacion de que la descomposicion
        no ha perdido ni inventado nada por el camino, y es lo primero que hay
        que mirar antes de fiarse de un residuo.
        """
        if self.modelo == "multiplicativo":
            serie = self.tendencia * self.estacional * self.residuo
        else:
            serie = self.tendencia + self.estacional + self.residuo
        return serie.rename("reconstruida")

    def a_marco(self) -> pd.DataFrame:
        """Las tres componentes en un DataFrame, que es como se grafican."""
        return pd.DataFrame(
            {
                "tendencia": self.tendencia,
                "estacional": self.estacional,
                "residuo": self.residuo,
            }
        )

    def fuerza_estacional(self) -> float:
        """Cuanto pesa la estacionalidad frente al residuo, entre 0 y 1.

        ``1 - Var(residuo) / Var(estacional + residuo)``, la medida de Wang,
        Smith y Hyndman. Se compara con el residuo y no con la serie entera
        porque una tendencia fuerte inflaria cualquier cociente y haria parecer
        estacional a una serie que solo sube.
        """
        if self.modelo == "multiplicativo":
            combinado = self.estacional * self.residuo
        else:
            combinado = self.estacional + self.residuo
        valida = combinado.dropna()
        varianza = float(valida.var(ddof=1))
        if varianza <= 0.0:
            return 0.0
        residual = float(self.residuo.dropna().var(ddof=1))
        return float(np.clip(1.0 - residual / varianza, 0.0, 1.0))


def descomponer(
    serie,
    periodo: int = 252,
    modelo: str = "aditivo",
    ventana_tendencia: int | None = None,
) -> Descomposicion:
    """Descomposicion clasica por medias moviles en tendencia, estacional y residuo.

    El procedimiento es el de siempre, y sigue siendo el que mejor se explica en
    un comite: (1) se estima la tendencia con una media movil centrada de una
    ventana igual al periodo, que promedia un ciclo completo y por tanto lo
    cancela; (2) se quita la tendencia de la serie; (3) se promedia lo que queda
    en cada posicion del ciclo, lo que cancela el ruido y deja el patron; (4) se
    centra ese patron y se llama residuo a lo que sobra.

    Parametros
    ----------
    periodo:
        Longitud del ciclo en **observaciones**: 252 para el ciclo anual de una
        serie diaria de dias habiles, 365 si la serie es de dias naturales, 12
        si es mensual, 7 para el ciclo semanal de una serie diaria.
    modelo:
        ``"multiplicativo"`` si la amplitud estacional **crece con el nivel**
        (el invierno aniade un 20 % al precio, valga este 30 o 300 EUR/MWh: es
        lo tipico de los precios, y se reconoce porque en el grafico los dientes
        de sierra se ensanchan cuando la serie sube). ``"aditivo"`` si la
        amplitud es constante en unidades de la serie (el invierno aniade 8
        EUR/MWh haga el precio lo que haga: es lo tipico de los volumenes, los
        grados dia y los spreads). Elegir mal no da error, da un residuo con
        estacionalidad residual visible: ese es el sintoma.

        El multiplicativo **exige serie estrictamente positiva**. En energia
        conviene recordarlo: los precios horarios del pool pueden ser negativos
        cuando sobra renovable, y ahi el multiplicativo no es una opcion.
    ventana_tendencia:
        Ventana de la media movil. Por defecto el propio ``periodo``, que es lo
        correcto para que la media anule el ciclo. Se deja abrir para suavizar
        mas (una ventana mayor da una tendencia mas lisa y carga mas variacion
        en el residuo) en series muy ruidosas.

    Errores
    -------
    Se exigen dos ciclos completos. Con menos, hay posiciones del ciclo para las
    que la media movil no deja ningun dato con el que estimar el factor
    estacional, y lo que sale no es una estacionalidad mal estimada sino un
    hueco.
    """
    valores = como_serie(serie, "serie")
    periodo = _validar_entero(periodo, "periodo", minimo=2)
    _validar_opcion(modelo, MODELOS, "modelo")

    n = len(valores)
    if n < 2 * periodo:
        raise ErrorDatos(
            f"Hacen falta al menos dos ciclos completos para descomponer "
            f"(2 x periodo = {2 * periodo} observaciones) y la serie tiene {n}. "
            "Reduzca 'periodo' (252 para diario habil, 12 para mensual, 7 para el "
            "ciclo semanal) o alargue la muestra."
        )

    if ventana_tendencia is None:
        ventana = periodo
    else:
        ventana = _validar_entero(ventana_tendencia, "ventana_tendencia", minimo=2)
    if modelo == "multiplicativo":
        _exigir_positiva(valores, "la descomposicion")

    tendencia = _media_movil_centrada(valores, ventana)
    sin_tendencia = valores / tendencia if modelo == "multiplicativo" else valores - tendencia

    posicion = np.arange(n) % periodo
    factores = sin_tendencia.groupby(posicion).mean()
    if factores.isna().any():
        huecos = int(factores.isna().sum())
        raise ErrorDatos(
            f"{huecos} posiciones del ciclo se quedan sin ninguna observacion con "
            "tendencia estimada. Reduzca 'ventana_tendencia' o alargue la muestra."
        )

    # Centrar el patron es lo que separa nivel de estacionalidad: sin esto, una
    # parte constante del ciclo se colaria en el factor estacional y la
    # tendencia dejaria de ser el nivel de la serie.
    if modelo == "multiplicativo":
        factores = factores / float(factores.mean())
    else:
        factores = factores - float(factores.mean())

    estacional = pd.Series(
        factores.to_numpy()[posicion], index=valores.index, name="estacional"
    )
    if modelo == "multiplicativo":
        residuo = valores / (tendencia * estacional)
    else:
        residuo = valores - tendencia - estacional

    return Descomposicion(
        tendencia=tendencia.rename("tendencia"),
        estacional=estacional,
        residuo=residuo.rename("residuo"),
        periodo=periodo,
        modelo=modelo,
    )


def desestacionalizar(
    serie,
    periodo: int = 252,
    modelo: str = "aditivo",
) -> pd.Series:
    """Serie sin el componente estacional: tendencia mas residuo.

    Es la serie que hay que llevar a un modelo. Dos razones. La primera es que
    lo que queda es lo unico que no se sabia de antemano: descontar el
    calendario impide que el modelo se puntue a si mismo por acertar algo que
    un almanaque tambien acierta. La segunda es que la comparacion entre
    periodos deja de mentir: un diciembre frente a un noviembre no se pueden
    comparar en bruto, y sobre la serie desestacionalizada si.

    A diferencia de la tendencia, la serie desestacionalizada esta definida en
    **toda** la muestra, tambien en los extremos: el factor estacional se
    conoce en cada fecha aunque la media movil no llegue.
    """
    valores = como_serie(serie, "serie")
    piezas = descomponer(valores, periodo=periodo, modelo=modelo)
    if piezas.modelo == "multiplicativo":
        ajustada = valores / piezas.estacional
    else:
        ajustada = valores - piezas.estacional
    return ajustada.rename("desestacionalizada")


# --- contraste ------------------------------------------------------------------


def contraste_estacionalidad(serie, por: str = "mes") -> dict:
    """Contrasta si las diferencias entre grupos del calendario son significativas.

    Contrasta la hipotesis nula de que todos los grupos proceden de la misma
    distribucion, con el contraste de **Kruskal-Wallis**: el ANOVA de un factor
    aplicado a los rangos en lugar de a los valores. La eleccion no es
    cosmetica. El ANOVA clasico supone normalidad y varianzas iguales en todos
    los grupos, y una serie de precios de energia incumple las dos cosas de
    manera escandalosa: colas derechas larguisimas por los picos de escasez y
    una varianza de invierno que multiplica varias veces a la de verano.
    Trabajar con rangos hace que un pico de 700 EUR/MWh cuente como "el valor
    mas alto" y no como "cuarenta desviaciones tipicas", que es lo que hace
    falta para que un solo dia no decida el contraste.

    Devuelve
    --------
    dict con ``estadistico`` (H), ``p_valor``, ``rechaza_al_5pct`` y
    ``tamano_efecto``, ademas de los grados de libertad, el numero de grupos y
    las observaciones.

    El tamanio del efecto es ``epsilon`` cuadrado, ``H / (n - 1)``: la fraccion
    de la variabilidad de los rangos que explica la agrupacion, entre 0 y 1. Va
    aparte del p-valor porque responden a preguntas distintas y confundirlas es
    el error habitual: con 4.000 observaciones diarias, una estacionalidad
    ridicula sale significativa. El p-valor dice si el patron existe; el
    tamanio del efecto, si importa. Como referencia orientativa, 0,01 es
    pequenio, 0,06 medio y 0,14 grande.
    """
    valores = _serie_fechada(serie, "serie")
    claves = _claves(valores.index, por)
    grupos = [np.asarray(trozo, dtype=float) for _, trozo in valores.groupby(claves)]

    if len(grupos) < 2:
        raise ErrorDatos(
            f"Solo hay un grupo con por='{por}': no hay nada que comparar. "
            "Alargue la muestra o use una agrupacion mas fina."
        )
    if max(len(g) for g in grupos) < 2:
        # Con un unico dato por grupo los rangos son una permutacion de 1..k y
        # el estadistico vale siempre k-1, gane quien gane: el contraste no
        # mira los datos.
        raise ErrorDatos(
            f"Con por='{por}' cada grupo tiene una sola observacion y el contraste "
            "es vacio (H valdria k-1 con cualquier dato). Use una agrupacion mas "
            "gruesa (por='trimestre' o por='mes') o alargue la muestra."
        )
    if valores.nunique() == 1:
        raise ErrorDatos(
            "Todos los valores de la serie son identicos: no hay variabilidad que "
            "repartir entre grupos y el contraste no esta definido."
        )

    estadistico, p_valor = stats.kruskal(*grupos)
    n = len(valores)
    k = len(grupos)
    epsilon_cuadrado = float(np.clip(float(estadistico) / (n - 1), 0.0, 1.0))

    return {
        "estadistico": float(estadistico),
        "p_valor": float(p_valor),
        "rechaza_al_5pct": bool(p_valor < 0.05),
        "tamano_efecto": epsilon_cuadrado,
        "grados_libertad": k - 1,
        "n_grupos": k,
        "observaciones": n,
        "por": por,
    }


# --- meteorologia ---------------------------------------------------------------


def grados_dia_acumulados(
    temperatura,
    base: float = BASE_CALEFACCION,
    ventana: int = 30,
) -> pd.Series:
    """Suma movil de los grados dia de calefaccion de los ultimos ``ventana`` dias.

    Los grados dia de calefaccion de un dia son ``max(base - temperatura, 0)``:
    cuanto frio ha hecho por debajo del umbral en que se enciende la caldera.
    Lo que anticipa la demanda, sin embargo, no es el frio de hoy sino el
    **acumulado**, y por razones fisicas y contractuales bien concretas:

    * Los edificios tienen inercia termica. Un dia frio aislado apenas mueve el
      consumo porque la estructura todavia guarda calor; una semana seguida de
      frio vacia esa reserva y la calefaccion pasa a funcionar sin descanso.
    * El almacenamiento de gas se vacia por integracion, no por derivada. Lo
      que decide si hay que comprar en el mercado spot es cuanto se lleva
      retirado del almacen en las ultimas semanas.
    * Los contratos de suministro y las coberturas meteorologicas se liquidan
      contra grados dia acumulados de un periodo, no contra la temperatura de
      un dia.

    Por eso un modelo alimentado solo con la temperatura del dia infravalora
    sistematicamente las olas de frio largas, que son justo los episodios en
    los que se gana o se pierde el anio.

    Los primeros ``ventana - 1`` valores son ``NaN``. No se rellenan con sumas
    parciales a proposito: un acumulado de siete dias no es comparable con uno
    de treinta, y mezclarlos meteria un salto artificial al principio de la
    muestra que el modelo aprenderia como si fuese informacion.

    >>> import pandas as pd
    >>> frio = pd.Series([10.5] * 5, index=pd.date_range("2024-01-01", periods=5))
    >>> float(grados_dia_acumulados(frio, ventana=3).iloc[-1])
    15.0
    """
    serie = como_serie(temperatura, "temperatura")
    ventana = _validar_entero(ventana, "ventana", minimo=1)
    try:
        umbral = float(base)
    except (TypeError, ValueError) as exc:
        raise ErrorDatos(f"'base' debe ser un numero, se ha recibido {base!r}.") from exc
    if not np.isfinite(umbral):
        raise ErrorDatos(f"'base' debe ser un numero finito, se ha recibido {base!r}.")

    if ventana > len(serie):
        raise ErrorDatos(
            f"La ventana de acumulacion ({ventana}) supera las {len(serie)} "
            "observaciones de la serie de temperatura: no saldria ningun valor. "
            "Reduzca 'ventana' o alargue la muestra."
        )

    grados_dia = (umbral - serie).clip(lower=0.0)
    acumulado = grados_dia.rolling(ventana).sum()
    acumulado.attrs["base"] = umbral
    acumulado.attrs["ventana"] = ventana
    return acumulado.rename(f"hdd_acumulado_{ventana}")
