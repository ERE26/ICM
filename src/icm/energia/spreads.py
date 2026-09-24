"""Spreads de generacion y de refino: el margen es el activo.

Una central de ciclo combinado no gana dinero con el precio de la
electricidad ni con el del gas, sino con la **diferencia** entre lo que cobra
por el megavatio que vende y lo que paga por el gas que quema para
producirlo. Ese margen es el *spark spread*, y es lo que de verdad se negocia:
el productor vende electricidad a plazo y compra gas a plazo a la vez, de modo
que su posicion economica es el spread, no ninguna de las dos patas.

El mismo razonamiento con carbon da el *dark spread*, y restando el coste de
los derechos de emision se obtienen sus versiones "limpias", que son las que
deciden de verdad si una planta arranca en un sistema con precio del CO2.

Convenciones
------------
* Todos los spreads se devuelven en **euros por MWh electrico** (el MWh que la
  planta vende), no por MWh termico. Por eso el precio del combustible se
  divide por el rendimiento: vease :func:`spark_spread`.
* El signo es el del margen: positivo significa que producir cubre el
  combustible, las emisiones y los costes variables.
* Las entradas pueden ser ``pd.Series`` con indice de fechas, secuencias o
  escalares. Las series se alinean por fecha (interseccion), que es lo unico
  correcto cuando la electricidad cotiza los dias habiles del pool y el gas
  sigue otro calendario.

Lo que este modulo no hace
--------------------------
No modela el despacho intertemporal (minimos tecnicos, rampas, tiempos
minimos de arranque y parada). :func:`valor_opcionalidad` recoge solo la
opcionalidad periodo a periodo, que es la aproximacion habitual de mercado
("strip of spark spread options") y una cota superior del valor real de una
planta con restricciones fisicas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from icm.comun import ErrorDatos, como_serie

#: Galones estadounidenses por barril de crudo. La constante que hace falta
#: para comparar un crudo cotizado en USD/barril con unos productos refinados
#: cotizados en USD/galon; olvidarla da un crack spread 42 veces menor.
GALONES_POR_BARRIL = 42.0


def _como_float(valor, nombre: str) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError) as exc:
        raise ErrorDatos(f"'{nombre}' debe ser un numero, se ha recibido {valor!r}.") from exc


def _validar_eficiencia(eficiencia: float, nombre: str = "eficiencia") -> float:
    """Comprueba que el rendimiento termico este en (0, 1]."""
    valor = _como_float(eficiencia, nombre)
    if not 0.0 < valor <= 1.0:
        raise ErrorDatos(
            f"'{nombre}' debe estar en (0, 1] y se ha recibido {valor}. "
            "Exprese el rendimiento termico en tanto por uno (0.50 para un ciclo "
            "combinado, 0.38 para carbon), no en porcentaje: una eficiencia mayor "
            "que 1 crearia energia de la nada y una nula daria un coste infinito."
        )
    return valor


def _validar_no_negativo(valor: float, nombre: str) -> float:
    """Comprueba que un coste o un factor fisico no sea negativo."""
    numero = _como_float(valor, nombre)
    if not numero >= 0.0:
        raise ErrorDatos(
            f"'{nombre}' debe ser mayor o igual que cero y se ha recibido {numero}. "
            "Revise las unidades y el signo: los costes se introducen en positivo "
            "y ya se restan del margen."
        )
    return numero


@dataclass(frozen=True)
class ParametrosPlanta:
    """Ficha tecnica minima de una unidad de generacion termica.

    Atributos
    ---------
    nombre:
        Descripcion legible de la tecnologia.
    eficiencia:
        Rendimiento termico en tanto por uno (energia electrica producida por
        unidad de energia del combustible). Un ciclo combinado moderno ronda
        0.55-0.60 sobre poder calorifico inferior, pero el mercado cotiza el
        spark spread de referencia al 50 %, y esa es la convencion que se toma
        por defecto aqui.
    factor_emision:
        Toneladas de CO2 por MWh **termico** de combustible quemado. No por
        MWh electrico: la conversion la hace la eficiencia.
    costes_variables:
        Operacion y mantenimiento variables en EUR por MWh electrico. Incluye
        consumibles, agua y peajes; no incluye los costes fijos, que no entran
        en la decision de arrancar.
    """

    nombre: str
    eficiencia: float
    factor_emision: float
    costes_variables: float = 0.0

    def __post_init__(self) -> None:
        _validar_eficiencia(self.eficiencia)
        _validar_no_negativo(self.factor_emision, "factor_emision")
        _validar_no_negativo(self.costes_variables, "costes_variables")


#: Tecnologias de referencia.
#:
#: Los factores de emision salen de los valores por defecto del IPCC para
#: combustion estacionaria, pasados de kg de CO2 por gigajulio a toneladas por
#: MWh termico (1 MWh = 3,6 GJ): gas natural ~56 kg/GJ -> 0,20 tCO2/MWh;
#: hulla ~94 kg/GJ -> 0,34 tCO2/MWh. Las eficiencias son las convenciones con
#: las que cotizan los spreads en ICE y Argus (50 % para el gas, 38 % para el
#: carbon), no el rendimiento de una unidad concreta; para valorar un activo
#: real hay que sustituirlas por su curva de consumo especifico.
PLANTAS: dict[str, ParametrosPlanta] = {
    "ciclo_combinado": ParametrosPlanta(
        nombre="Ciclo combinado de gas (referencia de mercado, 50 %)",
        eficiencia=0.50,
        factor_emision=0.20,
        costes_variables=2.0,
    ),
    "ciclo_combinado_antiguo": ParametrosPlanta(
        nombre="Ciclo combinado de gas de primera generacion",
        eficiencia=0.42,
        factor_emision=0.20,
        costes_variables=2.5,
    ),
    "turbina_gas": ParametrosPlanta(
        nombre="Turbina de gas en ciclo abierto (punta)",
        eficiencia=0.35,
        factor_emision=0.20,
        costes_variables=3.0,
    ),
    "carbon": ParametrosPlanta(
        nombre="Central de carbon pulverizado (referencia de mercado, 38 %)",
        eficiencia=0.38,
        factor_emision=0.34,
        costes_variables=3.5,
    ),
}


# --- alineacion de las patas del spread ----------------------------------------


def _alinear(valores: dict[str, object]) -> pd.DataFrame:
    """Alinea por fecha las patas de un spread y difunde los escalares.

    Las series se cruzan por interseccion de indices: un spread solo existe en
    las fechas en que cotizan **todas** sus patas, y rellenar los huecos hacia
    delante inventaria margenes con precios rancios. Las secuencias sin indice
    se interpretan por posicion sobre el indice resultante, y los escalares se
    difunden (util para un precio de CO2 fijo en un escenario).
    """
    series: dict[str, pd.Series] = {}
    brutos: dict[str, np.ndarray] = {}
    escalares: dict[str, float] = {}

    for nombre, valor in valores.items():
        if isinstance(valor, (pd.Series, pd.DataFrame)):
            series[nombre] = como_serie(valor, nombre).rename(nombre)
        elif np.ndim(valor) == 0:
            escalares[nombre] = _como_float(valor, nombre)
        else:
            bruto = np.asarray(valor, dtype=float)
            if bruto.ndim != 1:
                raise ErrorDatos(
                    f"'{nombre}' debe ser una serie de una sola dimension, "
                    f"se han recibido {bruto.ndim} dimensiones."
                )
            brutos[nombre] = bruto

    if series:
        marco = pd.concat(series.values(), axis=1, join="inner")
        if marco.empty:
            raise ErrorDatos(
                f"Las series de {', '.join(series)} no comparten ninguna fecha. "
                "Reindexelas al mismo calendario (misma zona horaria y misma "
                "frecuencia) antes de calcular el spread."
            )
    else:
        longitudes = {len(v) for v in brutos.values()}
        if len(longitudes) > 1:
            raise ErrorDatos(
                f"Las series de {', '.join(brutos)} tienen longitudes distintas "
                f"({sorted(longitudes)}): recortelas a la misma muestra."
            )
        marco = pd.DataFrame(index=pd.RangeIndex(longitudes.pop() if longitudes else 1))

    for nombre, bruto in brutos.items():
        if len(bruto) != len(marco.index):
            raise ErrorDatos(
                f"'{nombre}' tiene {len(bruto)} valores y las fechas comunes son "
                f"{len(marco.index)}. Pase '{nombre}' como pd.Series con su indice "
                "de fechas para que se alinee sola."
            )
        marco[nombre] = bruto
    for nombre, escalar in escalares.items():
        marco[nombre] = escalar

    marco = marco.dropna(how="any")
    if marco.empty:
        raise ErrorDatos(
            f"No queda ninguna observacion completa tras alinear {', '.join(valores)}: "
            "compruebe los valores ausentes de las series de precios."
        )
    return marco.loc[:, list(valores)]


# --- spreads de generacion -----------------------------------------------------


def _spread_termico(
    precio_electricidad,
    precio_combustible,
    precio_co2,
    eficiencia: float,
    factor_emision: float,
    costes_variables: float,
    combustible: str,
    nombre: str,
) -> pd.Series:
    """Nucleo comun del spark y del dark spread, con o sin coste de emisiones."""
    eficiencia = _validar_eficiencia(eficiencia)
    costes_variables = _validar_no_negativo(costes_variables, "costes_variables")

    patas: dict[str, object] = {
        "electricidad": precio_electricidad,
        combustible: precio_combustible,
    }
    if precio_co2 is not None:
        factor_emision = _validar_no_negativo(factor_emision, "factor_emision")
        patas["co2"] = precio_co2

    datos = _alinear(patas)
    spread = datos["electricidad"] - datos[combustible] / eficiencia - costes_variables
    if precio_co2 is not None:
        spread = spread - datos["co2"] * factor_emision / eficiencia
    return spread.rename(nombre)


def spark_spread(
    precio_electricidad,
    precio_gas,
    eficiencia: float = 0.50,
    costes_variables: float = 0.0,
) -> pd.Series:
    """Margen bruto de un ciclo combinado, en EUR/MWh electrico.

    .. math:: S = P_{electricidad} - P_{gas} / \\eta - C_{variables}

    **Por que se divide por la eficiencia y no se multiplica.** El precio del
    gas cotiza por MWh *termico* (la energia contenida en el combustible) y el
    de la electricidad por MWh *electrico* (la energia que sale por la linea).
    Con un rendimiento del 50 %, producir 1 MWh electrico exige quemar
    1 / 0.50 = 2 MWh termicos de gas, asi que el coste de combustible por MWh
    vendido es el doble del precio del gas. Multiplicar por la eficiencia en
    lugar de dividir es el error clasico: convierte un coste de 2 EUR en uno
    de 0,5 EUR y produce margenes absurdamente altos que sobreviven a todas
    las revisiones porque el numero "parece" razonable.

    >>> float(spark_spread(100.0, 30.0, eficiencia=0.50).iloc[0])
    40.0
    """
    return _spread_termico(
        precio_electricidad,
        precio_gas,
        None,
        eficiencia,
        0.0,
        costes_variables,
        "gas",
        "spark_spread",
    )


def dark_spread(
    precio_electricidad,
    precio_carbon,
    eficiencia: float = 0.38,
    costes_variables: float = 0.0,
) -> pd.Series:
    """Margen bruto de una central de carbon, en EUR/MWh electrico.

    Identico al spark spread cambiando el combustible. La eficiencia de
    referencia es menor (38 % frente al 50 % del gas) porque un ciclo de vapor
    aprovecha peor el calor, y esa diferencia es la razon de fondo por la que
    el carbon aguanta peor un precio alto del CO2: quema mas combustible y
    ademas mas sucio por unidad quemada.
    """
    return _spread_termico(
        precio_electricidad,
        precio_carbon,
        None,
        eficiencia,
        0.0,
        costes_variables,
        "carbon",
        "dark_spread",
    )


def spark_spread_limpio(
    precio_electricidad,
    precio_gas,
    precio_co2,
    eficiencia: float = 0.50,
    factor_emision: float = 0.20,
    costes_variables: float = 0.0,
) -> pd.Series:
    """Spark spread despues del coste de los derechos de emision.

    Las emisiones por MWh **electrico** son ``factor_emision / eficiencia``:
    el factor esta expresado por MWh termico quemado y hacen falta
    ``1 / eficiencia`` MWh termicos por cada MWh vendido. Con los valores por
    defecto, 0.20 / 0.50 = 0,40 tCO2 por MWh electrico, de modo que cada 10
    EUR/t de CO2 se comen 4 EUR/MWh de margen.

    Con precios del CO2 altos el spread limpio puede ser negativo mientras el
    sucio sigue siendo positivo, y esa diferencia es exactamente lo que decide
    si la planta arranca: el derecho de emision es un coste variable mas, y
    quien lo ignore concluira que una central gana dinero justo el dia en que
    pierde.
    """
    return _spread_termico(
        precio_electricidad,
        precio_gas,
        precio_co2,
        eficiencia,
        factor_emision,
        costes_variables,
        "gas",
        "spark_spread_limpio",
    )


def dark_spread_limpio(
    precio_electricidad,
    precio_carbon,
    precio_co2,
    eficiencia: float = 0.38,
    factor_emision: float = 0.34,
    costes_variables: float = 0.0,
) -> pd.Series:
    """Dark spread despues del coste de los derechos de emision.

    El castigo es mayor que en el gas por partida doble: mas toneladas por MWh
    termico (0.34 frente a 0.20) y mas MWh termicos por MWh electrico (1/0.38
    frente a 1/0.50). Con los valores por defecto son 0,89 tCO2 por MWh
    electrico, mas del doble que un ciclo combinado.
    """
    return _spread_termico(
        precio_electricidad,
        precio_carbon,
        precio_co2,
        eficiencia,
        factor_emision,
        costes_variables,
        "carbon",
        "dark_spread_limpio",
    )


def spread_cambio_combustible(spark_limpio, dark_limpio) -> pd.Series:
    """Diferencia entre el spark y el dark limpios: el "climate spread".

    Dice que tecnologia resulta marginal. Si es **positivo**, el gas gana mas
    que el carbon con los mismos precios, entra antes en el despacho y desplaza
    al carbon al margen del sistema; si es **negativo**, manda el carbon.

    Es la variable que mas mueve las emisiones de un sistema electrico a corto
    plazo: el parque esta construido y no cambia, pero el orden en que arranca
    si, y sustituir un MWh de carbon por uno de gas ahorra del orden de medio
    tonelada de CO2 sin invertir un euro. Por eso el nivel del precio del CO2
    al que este spread cambia de signo (el *fuel switching price*) es el numero
    que miran los reguladores del mercado de derechos.
    """
    datos = _alinear({"spark_limpio": spark_limpio, "dark_limpio": dark_limpio})
    spread = datos["spark_limpio"] - datos["dark_limpio"]
    return spread.rename("spread_cambio_combustible")


# --- refino y estructura de mercado --------------------------------------------


def crack_spread(crudo, gasolina, destilado, ratio: Sequence[float] = (3, 2, 1)) -> pd.Series:
    """Margen de refino 3-2-1, en dolares por barril de crudo procesado.

    El 3-2-1 es la aproximacion estandar al rendimiento de una refineria
    sencilla: de tres barriles de crudo salen dos de gasolina y uno de
    destilado medio (gasoleo o queroseno).

    **Unidades.** El crudo cotiza en USD por barril y los productos refinados
    en USD por **galon**, asi que hay que multiplicar los productos por los
    42 galones que tiene un barril antes de restar. El resultado se divide por
    los barriles de crudo del ratio para expresarlo por barril procesado, que
    es como se compara entre refinerias y con el coste de operacion.

    .. math::
        C = \\frac{2 \\cdot 42 \\cdot P_{gasolina}
                  + 1 \\cdot 42 \\cdot P_{destilado}
                  - 3 \\cdot P_{crudo}}{3}

    Parametros
    ----------
    ratio:
        Tripleta ``(barriles de crudo, de gasolina, de destilado)``. El 5-3-2
        se usa para refinerias con mas capacidad de conversion.
    """
    valores = [_como_float(v, "ratio") for v in ratio]
    if len(valores) != 3:
        raise ErrorDatos(
            f"'ratio' debe tener tres elementos (crudo, gasolina, destilado) y "
            f"se han recibido {len(valores)}. El estandar es (3, 2, 1)."
        )
    if any(v <= 0.0 for v in valores):
        raise ErrorDatos(
            f"Todos los elementos de 'ratio' deben ser positivos, se ha recibido {tuple(valores)}."
        )
    partes_crudo, partes_gasolina, partes_destilado = valores

    datos = _alinear({"crudo": crudo, "gasolina": gasolina, "destilado": destilado})
    ingresos = GALONES_POR_BARRIL * (
        partes_gasolina * datos["gasolina"] + partes_destilado * datos["destilado"]
    )
    spread = (ingresos - partes_crudo * datos["crudo"]) / partes_crudo
    return spread.rename("crack_spread")


def spread_calendario(cercano, lejano) -> pd.Series:
    """Diferencia entre el vencimiento cercano y el lejano de la misma curva.

    Positivo es *backwardation* (el mercado paga mas por tener la mercancia
    ahora, sintoma de escasez fisica o de inventarios bajos) y negativo es
    *contango*, que remunera almacenar: si el descuento supera el coste de
    almacenamiento y financiacion, comprar fisico y vender el futuro es un
    arbitraje. En energia el signo cambia con las estaciones, porque el gas de
    verano se inyecta para venderlo en invierno.
    """
    datos = _alinear({"cercano": cercano, "lejano": lejano})
    spread = datos["cercano"] - datos["lejano"]
    return spread.rename("spread_calendario")


def spread_locacional(precio_a, precio_b, coste_transporte: float = 0.0) -> pd.Series:
    """Diferencia de precio entre dos zonas, neta del coste de transporte.

    Es el margen de mover energia de B a A. Si es positivo de forma
    persistente, la interconexion esta saturada: el mercado querria mandar mas
    y no puede, y ese diferencial es justo lo que remuneran los derechos de
    capacidad de transporte.
    """
    coste_transporte = _validar_no_negativo(coste_transporte, "coste_transporte")
    datos = _alinear({"a": precio_a, "b": precio_b})
    spread = datos["a"] - datos["b"] - coste_transporte
    return spread.rename("spread_locacional")


# --- lectura economica de un spread --------------------------------------------


def horas_en_dinero(spread) -> float:
    """Proporcion de periodos con spread estrictamente positivo.

    Es como el sector resume de un vistazo si una planta tiene sentido
    economico: "el ciclo combinado estuvo en dinero el 38 % de las horas del
    anio". Los periodos con spread exactamente nulo no cuentan: nadie arranca
    una turbina para cubrir justo el combustible.
    """
    serie = como_serie(spread, "spread")
    return float((serie > 0.0).mean())


def valor_opcionalidad(spread, coste_arranque: float = 0.0) -> pd.Series:
    """Pago periodo a periodo de la planta: ``max(spread - coste_arranque, 0)``.

    Una central no esta obligada a producir. Si el spread es negativo, para; si
    es positivo, arranca. Ese derecho a no producir es una **opcion de compra
    sobre el spread con precio de ejercicio cero**, y tiene valor precisamente
    porque el spread es volatil.

    De ahi el error sistematico que conviene evitar: valorar la planta como si
    produjese siempre, multiplicando el spread medio por las horas del anio,
    la **infravalora**. Por la desigualdad de Jensen, la media de
    ``max(spread, 0)`` es siempre mayor o igual que ``max(media, 0)``, y la
    diferencia crece con la volatilidad. Una planta con spread medio negativo
    puede valer mucho dinero si el spread se dispara algunas horas: es
    exactamente el caso de las puntas del ciclo abierto.

    ``coste_arranque`` descuenta el coste de encender la unidad, repartido
    entre los MWh del arranque. Con el, el pago deja de dominar al spread: la
    planta renuncia a margenes positivos pero pequenios.
    """
    coste_arranque = _validar_no_negativo(coste_arranque, "coste_arranque")
    serie = como_serie(spread, "spread")
    pago = (serie - coste_arranque).clip(lower=0.0)
    return pago.rename("valor_opcionalidad")


def resumen_spread(spread, nombre: str | None = None) -> pd.Series:
    """Ficha resumen de un spread: nivel, dispersion, colas y opcionalidad.

    Los percentiles 5 y 95 importan mas que la desviacion tipica porque la
    distribucion de un spread es muy asimetrica: acotada por abajo en la
    practica y con una cola derecha larguisima que concentra casi todo el
    beneficio de la planta.
    """
    serie = como_serie(spread, "spread")
    etiqueta = nombre or getattr(spread, "name", None) or "spread"
    return pd.Series(
        {
            "observaciones": float(len(serie)),
            "media": float(serie.mean()),
            "mediana": float(serie.median()),
            "desviacion": float(serie.std(ddof=1)) if len(serie) > 1 else float("nan"),
            "percentil_5": float(np.quantile(serie, 0.05)),
            "percentil_95": float(np.quantile(serie, 0.95)),
            "horas_en_dinero": horas_en_dinero(serie),
            "opcionalidad_acumulada": float(valor_opcionalidad(serie).sum()),
        },
        name=str(etiqueta),
    )


def margen_planta(
    precios: pd.DataFrame,
    planta: str | ParametrosPlanta,
    columna_electricidad: str,
    columna_combustible: str,
    columna_co2: str | None = None,
) -> pd.DataFrame:
    """Margen completo de una unidad a partir de un marco de precios.

    Devuelve el spread sucio, el coste de emisiones y el spread limpio (si se
    indica la columna de CO2), el pago de la opcionalidad y el margen
    acumulado. El acumulado suma la **opcionalidad**, no el spread: una planta
    que para cuando pierde dinero no acumula perdidas, y sumar spreads
    negativos describiria una central obligada a producir que no existe.

    Parametros
    ----------
    planta:
        Clave de :data:`PLANTAS` o un :class:`ParametrosPlanta` propio.
    columna_co2:
        Si se omite, todo se calcula sobre el spread sucio. En un mercado con
        derechos de emision eso sobrestima el margen, asi que conviene
        indicarla siempre que haya curva de CO2.
    """
    ficha = _resolver_planta(planta)
    marco = precios if isinstance(precios, pd.DataFrame) else pd.DataFrame(precios)

    columnas = [columna_electricidad, columna_combustible]
    if columna_co2 is not None:
        columnas.append(columna_co2)
    faltan = [c for c in columnas if c not in marco.columns]
    if faltan:
        raise ErrorDatos(
            f"Faltan las columnas {faltan} en el marco de precios. "
            f"Disponibles: {list(marco.columns)}."
        )

    sucio = _spread_termico(
        marco[columna_electricidad],
        marco[columna_combustible],
        None,
        ficha.eficiencia,
        0.0,
        ficha.costes_variables,
        "combustible",
        "spread_sucio",
    )
    salida = pd.DataFrame({"spread_sucio": sucio})

    if columna_co2 is not None:
        limpio = _spread_termico(
            marco[columna_electricidad],
            marco[columna_combustible],
            marco[columna_co2],
            ficha.eficiencia,
            ficha.factor_emision,
            ficha.costes_variables,
            "combustible",
            "spread_limpio",
        )
        salida = salida.reindex(limpio.index)
        salida["coste_co2"] = salida["spread_sucio"] - limpio
        salida["spread_limpio"] = limpio
        referencia = limpio
    else:
        referencia = sucio

    salida["valor_opcionalidad"] = valor_opcionalidad(referencia)
    salida["margen_acumulado"] = salida["valor_opcionalidad"].cumsum()
    salida.attrs["planta"] = ficha.nombre
    salida.attrs["eficiencia"] = ficha.eficiencia
    return salida


def _resolver_planta(planta: str | ParametrosPlanta) -> ParametrosPlanta:
    """Acepta indistintamente una clave del catalogo o una ficha propia."""
    if isinstance(planta, ParametrosPlanta):
        return planta
    if planta in PLANTAS:
        return PLANTAS[planta]
    raise ErrorDatos(
        f"Planta '{planta}' desconocida. Use una de {sorted(PLANTAS)} o construya "
        "un ParametrosPlanta con la eficiencia y el factor de emision de su unidad."
    )
