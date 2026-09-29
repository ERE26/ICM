"""Simulador de precios de energia con estructura predecible.

Este entorno no tiene salida a internet, asi que los modelos de
:mod:`icm.ml` necesitan un banco de pruebas propio. Un paseo aleatorio no
sirve: si el proceso generador no tiene senal, ningun modelo puede batir a la
referencia ingenua y lo unico que se demuestra es que nada funciona. Aqui el
precio depende de variables fisicas observables (el frio que hace, el viento
que sopla, el gas que queda almacenado) con sensibilidades conocidas, de modo
que se puede contrastar lo que el modelo aprende contra la verdad con la que
se genero la serie.

Calibrado contra el mercado espanol
-----------------------------------
Una revision adversarial encontro que este simulador inflaba los resultados
del cuaderno 03 por dos vias, y que llevaba dentro una hipotesis de mercado
falsa que nadie habia declarado. Se rehizo contra datos reales: 3.185 dias del
pool iberico, 3.877 del gas espaniol y 2.100 de demanda del sistema. Lo que
cambio:

* **El precio responde a la PREVISION, no a lo realizado.** Antes se sorteaba
  la exogena realizada, el precio se fijaba sobre ella y la prevision se
  fabricaba despues a partir de ese mismo valor futuro. De ahi salia, por pura
  algebra, que la rentabilidad de operar con previsiones era el parametro
  ``habilidad`` multiplicado por una constante. Medido sobre OMIE y la
  prevision de demanda que publica el operador, el R cuadrado del precio
  contra la prevision es 0,2608 y contra la demanda realizada 0,2628: la
  subasta se casa **con** la prevision sobre la mesa.
* **Calendario natural con ciclo semanal.** Antes solo habia dias habiles, y
  eso dejaba fuera el efecto mas grande del pool: 19 EUR/MWh de recorrido
  entre el domingo y el martes. El crudo y las acciones siguen sin cotizar en
  fin de semana y salen como ``NaN``, igual que en una descarga real.
* **Estacionalidad estocastica.** El ciclo anual era un coseno identico cada
  anio, de modo que ocho columnas de calendario reproducian el 80 % del
  resultado del cuaderno. En los datos, la correlacion entre el perfil mensual
  de un anio y el de otro es de +0,08: hay estacionalidad, pero su forma no se
  conoce de antemano.
* **Volatilidades y colas de los datos.** El catalogo se quedaba entre 1,5 y
  10 veces corto (OMIE declaraba 0,75 y la realizada es 7,69), y por eso los
  precios nunca llegaban a la cola inferior: el minimo real de OMIE es el
  0,2 % de su media y el simulado no bajaba del 54 %.
* **Factores por bloque de mercado.** Un unico factor global hacia que
  mercados sin relacion correlacionasen: +0,31 entre Henry Hub y Brent frente
  a +0,04 real.
* **Reversion a la media** tipo Ornstein-Uhlenbeck, arrancando ya en la
  distribucion estacionaria, y **grados dia** (HDD/CDD) como senal termica,
  que es la variable con la que el sector contrata.
* **Saltos asimetricos**: al alza en materias primas, a la baja en acciones.
  Se parametrizan por la cuota de varianza que se llevan y por su frecuencia.

Lo que **no** reproduce, y conviene saber: el agrupamiento de volatilidad se
queda corto. La autocorrelacion de los rendimientos en valor absoluto sale
entre 0,07 y 0,13, cuando en los datos reales va de 0,25 a 0,44.

Cobertura de datos reales
-------------------------
Los activos con clave ``yahoo`` se pueden descargar cuando hay red. **TTF
(Paises Bajos), MIBGAS (Espana) y OMIE (pool electrico espanol) no tienen
feed gratuito en Yahoo Finance**: sus curvas se publican en las webs de ICE
Endex, MIBGAS y OMIE, y hay que descargarlas a mano y cargarlas con
:func:`icm.datos.cargar_precios_fichero`. :func:`activos_sin_yahoo` devuelve
justo esa lista.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from icm.comun import DIAS_HABILES_ANIO, ErrorDatos, como_serie
from icm.datos.proveedor import ProveedorDatos, normalizar_tickers, validar_marco_precios

# La dinamica de la volatilidad es la misma que la del simulador general: se
# reutiliza en lugar de duplicarla para que ambos mundos compartan un unico
# GARCH(1,1) de varianza incondicional unitaria.
from icm.datos.simulacion import _factor_volatilidad_garch as factor_volatilidad_garch

#: Duracion del ciclo estacional en dias naturales (no habiles: el frio no
#: sabe de fines de semana).
DIAS_ANIO = 365.25

#: Dia del anio en que el invierno pesa mas: maximo de HDD y de precio.
_PICO_INVIERNO = 15.0

#: Dia del anio en que el almacenamiento de gas esta mas lleno, al terminar la
#: campania de inyeccion y antes de empezar el invierno.
_PICO_ALMACENAMIENTO = 300.0

# Persistencia casi unitaria. Ajustando un GARCH(1,1) a los datos reales de
# OMIE y MIBGAS, alfa+beta sale 1,0000 exacto en ambos: el limite IGARCH. Con
# 0,97 el agrupamiento de volatilidad simulado se quedaba en una
# autocorrelacion de |r| de 0,03, frente al 0,25-0,44 que se mide.
_GARCH_ALFA = 0.10
_GARCH_BETA = 0.895

#: Habilidad por defecto de las previsiones a 24 horas: correlacion entre lo
#: previsto y lo que acaba ocurriendo. Vease :func:`prevision_exogena`.
HABILIDAD_PREVISION = 0.90

#: Habilidad por variable. El mensaje de error de :func:`prevision_exogena`
#: prescribia 0,95 para temperatura y 0,85 para viento, y el codigo aplicaba
#: 0,90 a las tres: ahora se respeta lo que el propio modulo recomienda.
HABILIDAD_POR_VARIABLE = {"temperatura": 0.95, "viento": 0.85}

#: Exogenas de las que se publica prevision. El almacenamiento no esta: se
#: conoce con rezago semanal, no por adelantado. Los grados-dia no aparecen
#: porque ya no se preven por separado: se derivan de la temperatura prevista,
#: que es como se hace en la realidad y evita previsiones de grados-dia
#: negativas o que anuncien calefaccion y refrigeracion el mismo dia.
EXOGENAS_PREVISTAS = ("temperatura", "viento")

#: Los saltos se parametrizan por la CUOTA de varianza que se llevan y por su
#: frecuencia, y de ahi se deduce su tamanio. Antes su tamanio era un
#: logaritmo absoluto, de modo que recalibrar la volatilidad de OMIE de 0,75 a
#: 7,70 dejaba su salto de 0,11 en un cuarto de sigma y la curtosis se hundia
#: de 14,9 -- la real -- a 0,5; y al expresarlo en sigmas, un salto grande se
#: comia el presupuesto entero y la volatilidad caia a la mitad. Con la cuota
#: fijada, la frecuencia decide la forma de la cola: pocos saltos grandes dan
#: mas curtosis que muchos pequenios, a igualdad de varianza.

#: Cuanto del precio responde a la SORPRESA (lo realizado menos lo previsto),
#: relativo a como responde a la prevision. Cero significa un mercado diario
#: perfectamente eficiente: la subasta se casa con la prevision y lo que luego
#: ocurra no puede cambiar un precio ya fijado.
#:
#: Medido sobre OMIE y la prevision de demanda publicada por el operador (1.004
#: dias de 2018 a 2020): el R cuadrado del precio contra la prevision es 0,2608
#: y contra la demanda realizada 0,2628, asi que conocer la realidad aniade dos
#: decimas de punto. El coeficiente de la sorpresa sale en +2,13 frente a +2,41
#: el de la prevision, pero con un intervalo de confianza de [-0,39, +4,65] que
#: contiene tanto el cero como el coeficiente de la prevision: los datos no los
#: distinguen porque la sorpresa tiene una decima parte de la variacion. El 0,3
#: reproduce esa brecha de R cuadrado sin pretender mas precision de la que hay.
PESO_SORPRESA = 0.3

#: Desviacion en logaritmos del precio medio segun el dia de la semana, de
#: lunes a domingo. Medida sobre los datos reales del periodo 2018-2026: 3.185
#: dias de OMIE y 3.877 de MIBGAS.
#:
#: Es el efecto mas grande del mercado electrico espaniol (19 EUR/MWh de
#: recorrido sobre un nivel medio de 79) y el simulador no podia contenerlo,
#: porque generaba el calendario con ``pd.bdate_range`` y no tenia fines de
#: semana. En gas el ciclo existe pero es un orden de magnitud menor.
CICLO_SEMANAL: dict[str, tuple[float, ...]] = {
    "electricidad": (0.0638, 0.1101, 0.0971, 0.0802, 0.0384, -0.1360, -0.2557),
    "gas": (0.0245, 0.0020, 0.0039, 0.0055, -0.0004, -0.0229, -0.0125),
}

#: Tipos de activo que no cotizan en fin de semana. Sus precios salen como
#: ``NaN`` el sabado y el domingo, que es lo que devuelve cualquier descarga
#: real y obliga a que el analisis case calendarios distintos, como en el
#: mundo. La electricidad y el gas si casan los 365 dias del anio.
SIN_FIN_DE_SEMANA = ("crudo", "accion")

#: Bloques de mercados que comparten factor. Un unico factor global hacia que
#: mercados sin relacion correlacionasen: el simulador daba +0,31 entre Henry
#: Hub y Brent cuando lo real es +0,04, y a la vez se quedaba corto en el par
#: que si se mueve junto, +0,52 entre WTI y Brent frente a +0,69 real.
#: Medido sobre 1.652 dias comunes de datos reales.
BLOQUES_FACTOR: dict[str, tuple[str, ...]] = {
    "crudo": ("WTI", "BRENT", "XLE"),
    "gas_europeo": ("TTF", "MIBGAS", "OMIE"),
    "gas_americano": ("HENRY_HUB",),
    "servicios": ("XLU",),
}

CATALOGO: dict[str, dict] = {
    "WTI": {
        "nombre": "Crudo West Texas Intermediate",
        "tipo": "crudo",
        "mercado": "NYMEX",
        "yahoo": "CL=F",
        "unidad": "USD/barril",
        "estacionalidad": 0.03,
    },
    "BRENT": {
        "nombre": "Crudo Brent del Mar del Norte",
        "tipo": "crudo",
        "mercado": "ICE Futures Europe",
        "yahoo": "BZ=F",
        "unidad": "USD/barril",
        "estacionalidad": 0.03,
    },
    "HENRY_HUB": {
        "nombre": "Gas natural Henry Hub",
        "tipo": "gas",
        "mercado": "NYMEX",
        "yahoo": "NG=F",
        "unidad": "USD/MMBtu",
        "estacionalidad": 0.22,
    },
    "TTF": {
        "nombre": "Gas natural Title Transfer Facility",
        "tipo": "gas",
        "mercado": "ICE Endex (Paises Bajos)",
        "yahoo": None,
        "unidad": "EUR/MWh",
        "estacionalidad": 0.26,
    },
    "MIBGAS": {
        "nombre": "Gas natural MIBGAS diario",
        "tipo": "gas",
        "mercado": "MIBGAS (Espana)",
        "yahoo": None,
        "unidad": "EUR/MWh",
        "estacionalidad": 0.22,
    },
    "OMIE": {
        "nombre": "Electricidad mercado diario peninsular",
        "tipo": "electricidad",
        "mercado": "OMIE (Espana)",
        "yahoo": None,
        "unidad": "EUR/MWh",
        "estacionalidad": 0.15,
    },
    "XLE": {
        "nombre": "Energy Select Sector SPDR",
        "tipo": "accion",
        "mercado": "NYSE Arca",
        "yahoo": "XLE",
        "unidad": "USD/participacion",
        "estacionalidad": 0.0,
    },
    "XLU": {
        "nombre": "Utilities Select Sector SPDR",
        "tipo": "accion",
        "mercado": "NYSE Arca",
        "yahoo": "XLU",
        "unidad": "USD/participacion",
        "estacionalidad": 0.0,
    },
}

# Perfil dinamico por tipo de activo. Las sensibilidades son elasticidades del
# log-precio al nivel de la exogena, asi que coinciden con el efecto de una
# variacion unitaria de esa exogena sobre el rendimiento del dia: es lo que
# una regresion de rendimientos sobre diferencias deberia recuperar.
_PERFIL_TIPO: dict[str, dict] = {
    "crudo": {
        "reversion": 0.010,
        "volatilidad": 0.52,
        "sensibilidad_hdd": 0.0005,
        "sensibilidad_cdd": 0.0006,
        "sensibilidad_viento": 0.0,
        "sensibilidad_almacenamiento": 0.0,
        "carga_factor": 0.78,
        "propension_saltos": 0.80,
        "cuota_saltos": 0.25,
        "signo_salto": 1.0,
        "deriva_anual": 0.0,
    },
    "gas": {
        "reversion": 0.030,
        "volatilidad": 1.20,
        "sensibilidad_hdd": 0.0100,
        "sensibilidad_cdd": 0.0025,
        "sensibilidad_viento": -0.06,
        "sensibilidad_almacenamiento": -0.0018,
        "carga_factor": 0.45,
        "propension_saltos": 0.50,
        "cuota_saltos": 0.40,
        "signo_salto": 1.0,
        "deriva_anual": 0.0,
    },
    "electricidad": {
        "reversion": 0.100,
        "volatilidad": 7.70,
        "sensibilidad_hdd": 0.0090,
        "sensibilidad_cdd": 0.0070,
        "sensibilidad_viento": -0.25,
        "sensibilidad_almacenamiento": -0.0004,
        "carga_factor": 0.30,
        "propension_saltos": 0.80,
        "cuota_saltos": 0.30,
        "signo_salto": 1.0,
        "deriva_anual": 0.0,
    },
    "accion": {
        "reversion": 0.0,
        "volatilidad": 0.24,
        "sensibilidad_hdd": 0.0002,
        "sensibilidad_cdd": 0.0001,
        "sensibilidad_viento": 0.0,
        "sensibilidad_almacenamiento": 0.0,
        "carga_factor": 0.70,
        "propension_saltos": 0.20,
        # Las acciones saltan a la BAJA. Con saltos Gamma, que son siempre
        # positivos, XLU acababa siendo el activo con mas asimetria positiva
        # del catalogo (+1,81), justo al reves que una utility real.
        "cuota_saltos": 0.30,
        "signo_salto": -1.0,
        "deriva_anual": 0.07,
    },
}

# Lo que distingue a cada activo dentro de su tipo. "nivel" es el precio medio
# en la unidad de cotizacion del catalogo.
# Las volatilidades son las realizadas sobre datos reales 2018-2026,
# anualizadas con 365 dias: WTI 0,538, Brent 0,515, Henry Hub 1,396, MIBGAS
# 1,163 y OMIE 7,685. Las anteriores (0,34 / 0,55 / 0,75) se quedaban entre
# 1,5 y 10 veces cortas, y por eso el simulador nunca llegaba a la cola
# inferior: el minimo real de OMIE es el 0,2 % de su media y el simulado no
# bajaba del 54 %. Las de TTF, XLE y XLU se mantienen proporcionadas a su
# comparable porque no hay serie real para calibrarlas.
_AJUSTE_ACTIVO: dict[str, dict] = {
    "WTI": {"nivel": 72.0, "volatilidad": 0.54},
    "BRENT": {"nivel": 77.0, "volatilidad": 0.52, "carga_factor": 0.88},
    "HENRY_HUB": {"nivel": 3.2, "volatilidad": 1.40},
    "TTF": {
        "nivel": 32.0,
        "volatilidad": 1.30,
        "sensibilidad_hdd": 0.0125,
        "sensibilidad_almacenamiento": -0.0022,
    },
    "MIBGAS": {
        "nivel": 30.0,
        "volatilidad": 1.16,
        "sensibilidad_hdd": 0.0110,
        # Su curtosis real es 9,1, la mas baja del grupo: saltos mas
        # frecuentes y mas pequenios a igualdad de varianza.
        "propension_saltos": 1.6,
        "cuota_saltos": 0.30,
    },
    # La carga sube de 0,30 a 0,55 para reproducir el vinculo iberico entre el
    # gas y la electricidad, que en los datos reales es de +0,169 y con 0,30
    # salia +0,075: el gas fija la tecnologia marginal buena parte del tiempo.
    "OMIE": {"nivel": 85.0, "volatilidad": 7.70, "carga_factor": 0.48},
    "XLE": {"nivel": 85.0, "volatilidad": 0.28},
    "XLU": {"nivel": 68.0, "volatilidad": 0.17, "deriva_anual": 0.05},
}


def activos_con_yahoo() -> list[str]:
    """Activos del catalogo descargables de Yahoo Finance."""
    return [clave for clave, ficha in CATALOGO.items() if ficha["yahoo"] is not None]


def activos_sin_yahoo() -> list[str]:
    """Activos que hay que cargar de fichero: no hay feed gratuito de ellos.

    Son las referencias europeas (TTF, MIBGAS, OMIE). Se descargan de la web
    del mercado y se leen con :func:`icm.datos.cargar_precios_fichero`.
    """
    return [clave for clave, ficha in CATALOGO.items() if ficha["yahoo"] is None]


def _ciclo_anual(dia_del_anio: np.ndarray, pico: float) -> np.ndarray:
    """Coseno anual que vale 1 en el dia ``pico`` y -1 medio anio despues."""
    return np.cos(2.0 * np.pi * (dia_del_anio - pico) / DIAS_ANIO)


def _ar1(
    n_periodos: int,
    persistencia: float,
    desviacion: float,
    generador: np.random.Generator,
) -> np.ndarray:
    """Proceso AR(1) de media cero cuya desviacion tipica *incondicional* es ``desviacion``.

    La innovacion se escala por ``sqrt(1 - persistencia**2)`` y el primer valor
    se extrae ya de la distribucion estacionaria. Asi el parametro significa lo
    que su nombre dice, la serie no arrastra un transitorio inicial y cambiar
    la persistencia no cambia de paso la escala del proceso.
    """
    valores = np.empty(n_periodos)
    choques = generador.standard_normal(n_periodos) * desviacion
    valores[0] = choques[0]
    escala = np.sqrt(1.0 - persistencia**2)
    for t in range(1, n_periodos):
        valores[t] = persistencia * valores[t - 1] + escala * choques[t]
    return valores


def _temperatura(
    dia_del_anio: np.ndarray,
    generador: np.random.Generator,
    media_anual: float,
    amplitud: float,
    persistencia: float,
    ruido: float,
) -> np.ndarray:
    """Nucleo de :func:`simular_temperatura`, con el generador ya construido.

    Se separa de la funcion publica para que :func:`simular_energia` pueda
    extraer las exogenas y los choques de precio del *mismo* generador. Con dos
    generadores distintos sembrados con la misma semilla, ambos consumirian la
    misma secuencia de normales y el tiempo quedaria correlacionado con los
    choques de mercado, que es justo la trampa que arruinaria el ejercicio.
    """
    estacional = media_anual - amplitud * _ciclo_anual(dia_del_anio, _PICO_INVIERNO)
    return estacional + _ar1(len(dia_del_anio), persistencia, ruido, generador)


def simular_temperatura(
    n_periodos: int,
    inicio: str | pd.Timestamp = "2018-01-01",
    semilla: int | None = 42,
    media_anual: float = 14.0,
    amplitud: float = 9.0,
    persistencia: float = 0.85,
    ruido: float = 2.5,
) -> pd.Series:
    """Temperatura diaria en grados Celsius: ciclo anual mas un AR(1) alrededor.

    El ciclo recoge las estaciones (minimo a mediados de enero, maximo a
    mediados de julio, hemisferio norte) y el AR(1) recoge que el tiempo es
    persistente: a un dia frio le sigue otro dia frio. Esa persistencia es la
    que hace que la temperatura sea *predecible a corto plazo* y, por tanto,
    utilizable como variable explicativa.

    Parametros
    ----------
    amplitud:
        Semiamplitud del ciclo anual: la diferencia entre el verano y el
        invierno medios es el doble.
    persistencia:
        Coeficiente autorregresivo de la desviacion respecto al ciclo.
    ruido:
        Desviacion tipica *incondicional* de esa desviacion, en grados.
    """
    if n_periodos < 2:
        raise ErrorDatos("Hacen falta al menos 2 periodos para simular una temperatura.")
    if not 0.0 <= persistencia < 1.0:
        raise ErrorDatos(
            f"La persistencia debe estar en [0, 1) para que la temperatura sea "
            f"estacionaria; se ha recibido {persistencia}."
        )
    if amplitud < 0.0 or ruido < 0.0:
        raise ErrorDatos("La amplitud estacional y el ruido no pueden ser negativos.")

    fechas = pd.bdate_range(start=inicio, periods=n_periodos, name="fecha")
    valores = _temperatura(
        fechas.dayofyear.to_numpy(dtype=float),
        np.random.default_rng(semilla),
        media_anual,
        amplitud,
        persistencia,
        ruido,
    )
    return pd.Series(valores, index=fechas, name="temperatura")


def grados_dia(
    temperatura,
    base_calefaccion: float = 15.5,
    base_refrigeracion: float = 22.0,
) -> pd.DataFrame:
    """Convierte temperatura en grados dia de calefaccion (HDD) y refrigeracion (CDD).

    Es la transformacion con la que el sector traduce meteorologia en demanda:
    por debajo de la base de calefaccion se enciende la caldera y por encima de
    la de refrigeracion el aire acondicionado, mientras que en la banda
    intermedia la temperatura no mueve el consumo. La no linealidad importa:
    pasar de 20 a 18 grados no cambia nada y pasar de 2 a 0 dispara la demanda,
    algo que una regresion sobre la temperatura en bruto no puede captar.

    Las bases son las convencionales del sector (15,5 C y 22 C), heredadas de
    los 65 F y 72 F estadounidenses.
    """
    serie = como_serie(temperatura, "temperatura")
    if base_calefaccion > base_refrigeracion:
        raise ErrorDatos(
            f"La base de calefaccion ({base_calefaccion}) no puede superar a la de "
            f"refrigeracion ({base_refrigeracion}): dejaria dias contando a la vez "
            "como frios y como calurosos."
        )
    return pd.DataFrame(
        {
            "hdd": np.maximum(base_calefaccion - serie, 0.0),
            "cdd": np.maximum(serie - base_refrigeracion, 0.0),
        },
        index=serie.index,
    )


@dataclass
class EscenarioEnergia:
    """Una realizacion completa del mercado simulado.

    Atributos
    ---------
    precios:
        Precios diarios, indice de fechas habiles y una columna por activo.
    exogenas:
        Variables observables que mueven el precio: ``temperatura``, ``hdd``,
        ``cdd``, ``viento`` y ``almacenamiento``, mas las columnas
        ``prevision_*`` descritas en :func:`prevision_exogena`. Comparten
        indice con los precios, de modo que se pueden concatenar sin
        reindexar.
    parametros:
        La verdad con la que se genero el escenario. Sirve para lo unico que
        no permite un mercado real: comparar lo que el modelo estima con lo
        que de verdad hay detras.
    """

    precios: pd.DataFrame
    exogenas: pd.DataFrame
    parametros: dict

    def __post_init__(self) -> None:
        if not self.precios.index.equals(self.exogenas.index):
            raise ErrorDatos(
                "Los precios y las exogenas de un escenario deben compartir indice; "
                "reindexe antes de construir el EscenarioEnergia."
            )


def prevision_exogena(
    serie: pd.Series,
    habilidad: float = HABILIDAD_PREVISION,
    horizonte: int = 1,
    semilla: int | None = None,
) -> pd.Series:
    """Prevision publicada en ``t`` para el valor de ``t + horizonte``.

    Esta es la pieza que distingue al trading de energia del de renta variable,
    y conviene entender bien por que no es mirar al futuro.

    Nadie puede predecir el precio del gas de manana. Pero el frio que hara
    manana **si** se sabe hoy: el operador del sistema y los servicios
    meteorologicos publican previsiones de temperatura, demanda y produccion
    renovable antes de que cierre el mercado diario. Esa prevision es un dato
    disponible en ``t``, y una mesa real la usa. Excluirla no haria el analisis
    mas honesto, lo haria irreal.

    La frontera esta en la habilidad. Usar el valor futuro exacto si seria
    trampa, porque ninguna prevision acierta del todo. La prevision se
    construye como el valor futuro mas un error:

        prevision(t) = x(t + horizonte) + error

    y la habilidad se mide **sobre el cambio**, no sobre el nivel:

        corr(prevision(t) - x(t),  x(t + horizonte) - x(t)) = habilidad

    Esa eleccion no es un detalle tecnico. Medir la habilidad sobre el nivel
    daria una cifra enganiosamente alta: con una variable persistente como la
    temperatura, decir "manana hara como hoy" ya correlaciona 0,85 con lo que
    ocurre, sin aportar nada. Lo unico que aniade valor es acertar el CAMBIO, y
    es ahi donde hay que exigir la habilidad. Es el mismo principio que el
    error escalado de :mod:`icm.ml.evaluacion`: una medida solo informa si esta
    anclada a la referencia ingenua que hay que batir.

    Parametros
    ----------
    habilidad:
        Correlacion entre el cambio previsto y el cambio realizado, en (0, 1].
        ``0`` seria no saber nada mas que la persistencia y ``1``, acertar el
        cambio exactamente. Las previsiones meteorologicas a 24 horas se mueven
        en 0,85-0,95.
        Con ``habilidad=1`` no hay error y el resultado **si** constituye fuga
        de informacion: se permite solo para poder medir cuanto valdria una
        prevision perfecta, nunca para presentar resultados.
    semilla:
        Fija el error de prevision, para que el escenario sea reproducible.

    Devuelve
    --------
    Serie indexada por ``t`` cuyo valor se refiere a ``t + horizonte``. Las
    ultimas ``horizonte`` posiciones quedan a ``NaN``: la prevision de un dia
    posterior al final de la muestra no existe.

    Notas
    -----
    El nombre lleva el prefijo ``prevision_``, pero conviene saber que **ese
    prefijo no lo mira nadie**: ``icm.ml.caracteristicas.construir_matriz``
    aplica ``retardos_exogenas`` por igual a todas las columnas que recibe, sin
    inspeccionar nombres. Lo que hace que la columna sirva con retardo cero no
    es como se llama, sino como esta fechada: la fila de ``t`` trae la
    prevision para ``t+1``, publicada en ``t``. Si se construye la prevision a
    mano con esta funcion, hay que desplazarla igual antes de usarla como
    variable, o pasarla en una llamada aparte con retardo cero.
    """
    if not isinstance(serie, pd.Series):
        raise ErrorDatos("prevision_exogena espera una Series indexada por fecha.")
    habilidad = float(habilidad)
    if not 0.0 < habilidad <= 1.0:
        raise ErrorDatos(
            f"La habilidad de la prevision debe estar en (0, 1], se ha recibido {habilidad}. "
            "Use 0,95 para temperatura a 24 horas y 0,85 para viento."
        )
    if horizonte < 1:
        raise ErrorDatos("El horizonte de la prevision debe ser de al menos un periodo.")

    futuro = serie.shift(-horizonte)
    if habilidad == 1.0:
        salida = futuro.copy()
    else:
        cambio = futuro - serie
        desviacion = float(cambio.std(ddof=1))
        constante = serie.nunique(dropna=True) <= 1
        if constante:
            # Una serie que no cambia es perfectamente previsible, y eso no es
            # un caso patologico: los grados-dia de refrigeracion valen cero
            # todo un invierno. Prever exactamente un cero no aporta ninguna
            # informacion, asi que aqui la prevision puede ser el valor futuro.
            salida = futuro.copy()
        elif not np.isfinite(desviacion) or desviacion == 0.0:
            # Con NIVEL variable, en cambio, devolver el futuro es una fuga
            # perfecta. La condicion anterior era 'desviacion == 0', que se
            # cumple siempre que el CAMBIO es constante: cualquier rampa
            # lineal, cualquier tramo interpolado. Medido: sobre la rampa
            # arange(0, 60, 0.5), con habilidad 0,99, 0,50 y hasta 0,01, el
            # error maximo de la prevision era 0 en los tres casos. Y con dos
            # observaciones, ddof=1 deja una sola diferencia valida, std
            # devuelve NaN y se tomaba tambien esta rama.
            raise ErrorDatos(
                "No se puede calibrar una habilidad de prevision sobre esta serie: su "
                f"nivel varia ({serie.nunique(dropna=True)} valores distintos) pero el "
                "cambio a un horizonte de "
                f"{horizonte} es constante o no calculable (desviacion={desviacion}). "
                "Devolver el valor futuro seria una prevision perfecta, no una con la "
                "habilidad pedida. Si la serie es una rampa o un tramo interpolado, "
                "previea la serie de la que salio; si tiene menos de tres "
                "observaciones, alarguela."
            )
        else:
            # corr(d + e, d) = sd(d) / sqrt(sd(d)^2 + sd(e)^2) = habilidad
            escala_error = desviacion * np.sqrt(1.0 / habilidad**2 - 1.0)
            generador = np.random.default_rng(semilla)
            error = generador.standard_normal(len(serie)) * escala_error
            salida = futuro + error
            salida[futuro.isna()] = np.nan

    nombre = serie.name if serie.name else "exogena"
    salida.name = f"prevision_{nombre}"
    return salida


def _validar_activos(activos: Sequence[str] | None) -> list[str]:
    if activos is None:
        return list(CATALOGO)
    claves = normalizar_tickers(activos)
    desconocidos = [c for c in claves if c not in CATALOGO]
    if desconocidos:
        raise ErrorDatos(
            f"Activos fuera del catalogo de energia: {desconocidos}. "
            f"Use alguno de: {', '.join(CATALOGO)}."
        )
    return claves


def _perfil_activo(clave: str) -> dict:
    """Parametros dinamicos de un activo: perfil de su tipo mas sus ajustes."""
    ficha = CATALOGO[clave]
    perfil = dict(_PERFIL_TIPO[ficha["tipo"]])
    perfil.update(_AJUSTE_ACTIVO.get(clave, {}))
    perfil["amplitud_estacional"] = float(ficha["estacionalidad"])
    # El tipo decide el ciclo semanal y si el mercado casa en fin de semana.
    perfil["tipo"] = ficha["tipo"]
    return perfil


def _por_activo(claves: Sequence[str], valores: np.ndarray) -> dict[str, float]:
    """Empareja un vector de parametros con sus activos, en tipos de Python."""
    return {clave: float(valor) for clave, valor in zip(claves, valores)}


def _viento(dia_del_anio: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Factor de carga eolica en (0, 1), mas alto en invierno y persistente.

    Se modela como una logistica de un AR(1) para que quede acotado entre 0 y 1
    sin recortes bruscos: un factor de carga no puede ser negativo ni superar
    la potencia instalada.
    """
    latente = _ar1(len(dia_del_anio), 0.60, 1.0, generador)
    senal = -0.9 + 0.45 * _ciclo_anual(dia_del_anio, _PICO_INVIERNO) + 0.8 * latente
    return 1.0 / (1.0 + np.exp(-senal))


def _almacenamiento(dia_del_anio: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Nivel de llenado de los almacenamientos de gas, en porcentaje.

    Se inyecta en verano y se extrae en invierno, asi que el ciclo va medio
    anio desfasado respecto al de la demanda. Es la variable que explica por
    que un invierno frio con los depositos llenos no mueve el precio y con los
    depositos vacios lo dispara.
    """
    nivel = 55.0 + 30.0 * _ciclo_anual(dia_del_anio, _PICO_ALMACENAMIENTO)
    nivel = nivel + _ar1(len(dia_del_anio), 0.95, 6.0, generador)
    return np.clip(nivel, 5.0, 100.0)


def _ciclo_semanal(fechas: pd.DatetimeIndex, tipo: str) -> np.ndarray:
    """Desviacion logaritmica del precio segun el dia de la semana."""
    perfil = CICLO_SEMANAL.get(tipo)
    if perfil is None:
        return np.zeros(len(fechas))
    return np.asarray(perfil, dtype=float)[fechas.dayofweek.to_numpy()]


def _estacional_estocastica(
    fechas: pd.DatetimeIndex, amplitud: float, generador: np.random.Generator
) -> np.ndarray:
    """Ciclo anual con amplitud y fase que cambian de un anio a otro.

    El ciclo era un coseno determinista identico cada anio, de modo que la
    base ``sin_anio``/``cos_anio`` que emite ``icm.ml.caracteristicas.calendario``
    lo recuperaba exactamente: con datos simulados, ocho columnas de fecha
    reproducian el 80 % del Sharpe del cuaderno sin mirar ni un precio.

    En el mercado real el perfil mensual **no se repite**: medido sobre los
    precios diarios de OMIE y MIBGAS de 2018 a 2026, la correlacion media
    entre el perfil de un anio y el de otro es de +0,08, con casos de -0,85 y
    de +0,85. Hay estacionalidad, pero su forma concreta no se conoce de
    antemano, y eso es lo que aqui se reproduce sorteando cada anio su propia
    amplitud y su propio desfase.
    """
    if amplitud == 0.0:
        return np.zeros(len(fechas))
    anios = fechas.year.to_numpy()
    salida = np.zeros(len(fechas))
    for anio in np.unique(anios):
        mascara = anios == anio
        # Amplitud lognormal alrededor de la nominal y pico desplazado hasta
        # mes y medio: dos inviernos no se parecen.
        escala = float(np.exp(generador.normal(0.0, 0.45)))
        desfase = float(generador.uniform(-90.0, 90.0))
        dia = fechas[mascara].dayofyear.to_numpy(dtype=float)
        salida[mascara] = amplitud * escala * _ciclo_anual(dia, _PICO_INVIERNO + desfase)
    # Centrar quita la deriva que induce empezar la muestra en el pico del
    # ciclo: con inicio el 1 de enero, el gas arrastraba una tendencia bajista
    # de casi el 7 % anual que ningun parametro declaraba.
    return salida - salida.mean()


def _generador_con_nombre(semilla: int, nombre: str) -> np.random.Generator:
    """Un generador reproducible derivado de una semilla y una etiqueta.

    La etiqueta entra en la secuencia de semillas, de modo que el flujo de
    'OMIE' es el mismo se pidan uno o los ocho activos, y el de un bloque no
    depende de cuantos de sus miembros se hayan pedido.
    """
    huella = int.from_bytes(nombre.encode("utf-8"), "little") % (2**63)
    return np.random.default_rng(np.random.SeedSequence([int(semilla), huella]))


def _generadores_por_activo(
    semilla: int, claves: Sequence[str]
) -> dict[str, np.random.Generator]:
    """Un generador propio por activo, derivado de su nombre y no de su posicion.

    Antes el sorteo idiosincratico salia de una matriz ``(n_periodos, n_activos)``
    extraida de un generador compartido, asi que pedir ``["OMIE"]`` y pedir el
    catalogo entero con la misma semilla daban mercados distintos para OMIE, y
    ``["OMIE", "WTI"]`` y ``["WTI", "OMIE"]`` tambien. Derivando la semilla del
    nombre, cada activo es el mismo se pida como se pida.
    """
    return {clave: _generador_con_nombre(semilla, f"activo:{clave}") for clave in claves}


def _factores_comunes(
    n_periodos: int, claves: Sequence[str], semilla: int
) -> np.ndarray:
    """Innovacion comun de cada activo segun el bloque de mercado al que pertenece.

    Con un unico factor global, todo correlacionaba con todo: el simulador daba
    +0,31 entre Henry Hub y Brent cuando lo real es +0,04, y a la vez se
    quedaba en +0,52 entre WTI y Brent, que en los datos van a +0,69. Un factor
    por bloque separa los mercados que de verdad estan separados -- el gas
    americano y el europeo lo estan -- y deja que los que comparten fisica se
    muevan juntos.
    """
    bloques = sorted(BLOQUES_FACTOR)
    sorteos = {
        b: _generador_con_nombre(semilla, f"bloque:{b}").standard_normal(n_periodos)
        for b in bloques
    }
    de_activo = {a: b for b, activos in BLOQUES_FACTOR.items() for a in activos}
    return np.column_stack(
        [sorteos[de_activo.get(clave, bloques[0])] for clave in claves]
    )


def simular_energia(
    activos: Sequence[str] | None = None,
    n_periodos: int = 1500,
    inicio: str | pd.Timestamp = "2018-01-01",
    semilla: int | None = 42,
    intensidad_saltos: float = 0.02,
    habilidad_prevision: float | None = None,
) -> EscenarioEnergia:
    """Genera un mercado de energia completo: precios, exogenas y la verdad que los liga.

    El log-precio de cada activo es

    ``log P = log(nivel) + estacional(t) + sensibilidades @ exogenas_centradas(t) + x(t)``

    donde ``x`` es un Ornstein-Uhlenbeck discreto con volatilidad GARCH comun,
    un factor comun entre activos y saltos de Poisson al alza. Como las
    exogenas entran en el nivel del log-precio, la sensibilidad de cada activo
    es exactamente el coeficiente que una regresion del rendimiento diario
    sobre la *variacion* de la exogena deberia recuperar: esa es la senal que
    el ML tiene que encontrar, y ``parametros`` guarda la respuesta.

    Parametros
    ----------
    activos:
        Claves del :data:`CATALOGO`. Por omision, todas.
    intensidad_saltos:
        Numero medio de saltos por dia, antes de escalar por la propension de
        cada tipo de activo (la electricidad salta mucho mas que una accion).
        Con 0 no hay saltos y las colas quedan solo con las del GARCH.

    Notas
    -----
    Los saltos se compensan restando su media: asi la asimetria al alza se
    traduce en colas derechas y no en una deriva espuria que un modelo podria
    confundir con tendencia.
    """
    claves = _validar_activos(activos)
    if n_periodos < 2:
        raise ErrorDatos("Hacen falta al menos 2 periodos para simular un escenario.")
    if intensidad_saltos < 0.0:
        raise ErrorDatos(
            f"La intensidad de saltos es un numero medio de saltos por dia y no puede "
            f"ser negativa; se ha recibido {intensidad_saltos}."
        )

    # Calendario NATURAL. Antes era 'pd.bdate_range', y eso dejaba fuera el
    # efecto mas grande del mercado electrico espaniol: el fin de semana. De
    # paso, el 20 % de las "previsiones a 24 horas" eran en realidad a 72,
    # porque el salto del viernes al lunes contaba como un periodo.
    fechas = pd.date_range(start=inicio, periods=n_periodos, freq="D", name="fecha")
    dia_del_anio = fechas.dayofyear.to_numpy(dtype=float)
    if semilla is None:
        # Materializar la semilla: guardarla como None hacia que 'parametros'
        # no describiese el escenario que acompania, y el docstring de
        # EscenarioEnergia promete que si.
        semilla = int(np.random.SeedSequence().entropy % (2**32))
    generador = np.random.default_rng(semilla)

    # LA PREVISION SE GENERA PRIMERO Y LA REALIDAD SE DERIVA DE ELLA.
    #
    # Antes era al reves: se sorteaba la exogena realizada, el precio se fijaba
    # sobre ella, y la prevision se fabricaba a partir de ese mismo valor
    # futuro con ruido. De ahi salia, algebraicamente,
    #     r(prevision, rendimiento) = habilidad x r(contemporanea, rendimiento)
    # es decir, que la rentabilidad de operar con previsiones era el parametro
    # 'habilidad' multiplicado por una constante. No era un hallazgo: era una
    # constante del fichero.
    #
    # Detras habia una hipotesis de mercado no declarada y falsa: que el precio
    # ignora la informacion publica hasta que se materializa. Medido sobre
    # OMIE y la prevision de demanda que publica el operador (1.004 dias), el
    # R cuadrado del precio contra la prevision es 0,2608 y contra la demanda
    # realizada 0,2628. La subasta se casa CON la prevision sobre la mesa.
    #
    # Invirtiendo la causalidad, 'habilidad' pasa a ser lo que dice ser -- una
    # propiedad fisica de la meteorologia -- y deja de gobernar el resultado.
    def habilidad_de(nombre: str) -> float:
        # None deja que cada variable use su habilidad propia; un valor
        # explicito la impone a todas, que es como se barre el parametro para
        # comprobar que ya no gobierna el resultado.
        valor = (
            habilidad_prevision
            if habilidad_prevision is not None
            else HABILIDAD_POR_VARIABLE.get(nombre, HABILIDAD_PREVISION)
        )
        if not 0.0 < valor <= 1.0:
            raise ErrorDatos(
                f"La habilidad de prevision de '{nombre}' debe estar en (0, 1], "
                f"se ha recibido {valor}."
            )
        return float(valor)

    # El reparto conserva la varianza del clima. La anomalia se parte en una
    # parte PREVISIBLE, con desviacion h por unidad, y una SORPRESA con
    # desviacion sqrt(1 - h^2), independientes entre si: la realizada vuelve a
    # tener desviacion uno sea cual sea h, y corr(prevista, realizada) = h.
    #
    # Escalar la sorpresa hacia arriba, como se hacia al principio de este
    # arreglo, tenia un efecto perverso: una prevision peor hacia el clima mas
    # variable, y con la sorpresa dominando tanto la realidad como el precio
    # la correlacion contemporanea SUBIA al empeorar la prevision. Una
    # prevision mala no hace que llueva mas.
    def repartir(estacional, anomalia_previsible, anomalia_sorpresa, habilidad):
        predecible = habilidad * anomalia_previsible
        return (
            estacional + predecible,
            estacional + predecible + np.sqrt(1.0 - habilidad**2) * anomalia_sorpresa,
        )

    h_temp = habilidad_de("temperatura")
    estacional_temp = 14.0 - 9.0 * _ciclo_anual(dia_del_anio, _PICO_INVIERNO)
    temp_prev, temp_real = repartir(
        estacional_temp,
        _ar1(n_periodos, 0.85, 2.5, generador),
        _ar1(n_periodos, 0.85, 2.5, generador),
        h_temp,
    )

    # El viento se reparte en el espacio LATENTE y despues pasa por la
    # logistica, para que las dos series sigan acotadas en (0, 1).
    h_viento = habilidad_de("viento")
    senal_estacional = -0.9 + 0.45 * _ciclo_anual(dia_del_anio, _PICO_INVIERNO)
    lat_prev, lat_real = repartir(
        senal_estacional,
        0.8 * _ar1(n_periodos, 0.60, 1.0, generador),
        0.8 * _ar1(n_periodos, 0.60, 1.0, generador),
        h_viento,
    )
    logistica = lambda x: 1.0 / (1.0 + np.exp(-x))

    previstas = {"temperatura": temp_prev, "viento": logistica(lat_prev)}
    realizadas = {"temperatura": temp_real, "viento": logistica(lat_real)}

    # Los grados-dia previstos se DERIVAN de la temperatura prevista, no se
    # preven por separado. Antes eran tres previsiones con tres ruidos
    # independientes sobre una sola temperatura, y salian grados-dia de
    # calefaccion negativos el 21 % de los dias, de refrigeracion el 45 %, y un
    # 2,4 % de dias que anunciaban calefaccion y refrigeracion a la vez.
    gd_previstos = grados_dia(pd.Series(previstas["temperatura"], index=fechas))
    gd_realizados = grados_dia(pd.Series(realizadas["temperatura"], index=fechas))

    almacenamiento = _almacenamiento(dia_del_anio, generador)

    # CONVENCION DE FECHA DE LAS PREVISIONES. Internamente, 'previstas[t]' es
    # la prevision PARA el dia t, que es la que entra en la subasta casada el
    # dia t-1 y por tanto la que fija el precio de t.
    #
    # Pero una fila de variables fechada en t tiene que contener lo que se
    # sabe EN t, no lo que se supo ayer. Por eso la columna publicada se
    # adelanta un periodo: 'prevision_hdd[t]' es la prevision para t+1,
    # publicada hoy. Asi la columna sirve con retardo cero y predice el
    # rendimiento de manana sin mirar al futuro. La ultima fila sale NaN
    # porque la prevision del dia siguiente al final de la muestra no existe.
    def publicada(valores: np.ndarray) -> np.ndarray:
        desplazada = np.full_like(np.asarray(valores, dtype=float), np.nan)
        desplazada[:-1] = np.asarray(valores, dtype=float)[1:]
        return desplazada

    exogenas = pd.DataFrame(
        {
            "temperatura": realizadas["temperatura"],
            "hdd": gd_realizados["hdd"].to_numpy(),
            "cdd": gd_realizados["cdd"].to_numpy(),
            "viento": realizadas["viento"],
            "almacenamiento": almacenamiento,
            "prevision_temperatura": publicada(previstas["temperatura"]),
            "prevision_hdd": publicada(gd_previstos["hdd"].to_numpy()),
            "prevision_cdd": publicada(gd_previstos["cdd"].to_numpy()),
            "prevision_viento": publicada(previstas["viento"]),
        },
        index=fechas,
    )

    # El precio usa la prevision PARA cada dia, no la publicada ese dia.
    conductoras_previstas = np.column_stack(
        [
            gd_previstos["hdd"].to_numpy(),
            gd_previstos["cdd"].to_numpy(),
            previstas["viento"],
            almacenamiento,
        ]
    )

    perfiles = [_perfil_activo(clave) for clave in claves]

    def campo(nombre: str) -> np.ndarray:
        """Un parametro de todos los activos como vector alineado con ``claves``."""
        return np.array([perfil[nombre] for perfil in perfiles], dtype=float)

    reversion = campo("reversion")
    carga = campo("carga_factor")
    cuota_saltos = campo("cuota_saltos")
    signo_salto = campo("signo_salto")
    lambda_salto = intensidad_saltos * campo("propension_saltos")
    # El calendario es natural, asi que la desanualizacion va con 365 dias.
    deriva = campo("deriva_anual") / DIAS_ANIO
    sensibilidades = np.vstack(
        [
            campo("sensibilidad_hdd"),
            campo("sensibilidad_cdd"),
            campo("sensibilidad_viento"),
            campo("sensibilidad_almacenamiento"),
        ]
    )

    # EL PRECIO RESPONDE A LA PREVISION, no a lo realizado. La sorpresa entra
    # con un peso pequenio: la subasta del dia D se casa en D-1 y lo que luego
    # ocurra no puede mover un precio ya fijado, aunque el mercado de ajustes
    # devuelva algo de esa desviacion al promedio diario.
    previstas_marco = conductoras_previstas
    realizadas_marco = exogenas[["hdd", "cdd", "viento", "almacenamiento"]].to_numpy()
    sorpresa_marco = realizadas_marco - previstas_marco
    efecto_exogeno = (
        (previstas_marco - previstas_marco.mean(axis=0)) @ sensibilidades
        + PESO_SORPRESA * (sorpresa_marco - sorpresa_marco.mean(axis=0)) @ sensibilidades
    )

    # PRESUPUESTO DE VOLATILIDAD. 'volatilidad' es la del activo entero, no la
    # del choque: si se usa tal cual para el choque, las exogenas y los saltos
    # aniaden varianza por encima y la realizada supera a la declarada. Medido
    # antes de este arreglo: +52 % en OMIE, +24 % en Henry Hub, +3 % en XLE, un
    # sesgo desigual que distorsionaba cualquier comparacion entre activos.
    var_objetivo = (campo("volatilidad") ** 2) / DIAS_ANIO
    var_saltos = cuota_saltos * var_objetivo * (lambda_salto > 0.0)
    # Lo que aportan al RENDIMIENTO diario es la varianza de su PRIMERA
    # DIFERENCIA, no la del nivel: un ciclo semanal grande pero suave mueve
    # poco de un dia al siguiente, y uno con un escalon de fin de semana mueve
    # mucho.
    var_exogena = np.diff(efecto_exogeno, axis=0).var(axis=0, ddof=1)
    var_semanal = np.array(
        [np.diff(_ciclo_semanal(fechas, perfil["tipo"])).var(ddof=1) for perfil in perfiles]
    )
    var_choque = np.maximum(
        var_objetivo - var_saltos - var_exogena - var_semanal, var_objetivo * 0.05
    )
    volatilidad = np.sqrt(var_choque)

    # Un factor por BLOQUE de mercado mas una parte idiosincratica. Con un solo
    # factor global, mercados sin relacion correlacionaban.
    factores = _factores_comunes(n_periodos, claves, semilla)
    rng_activo = _generadores_por_activo(semilla, claves)
    propio = np.column_stack(
        [rng_activo[clave].standard_normal(n_periodos) for clave in claves]
    )
    innovaciones = carga * factores + np.sqrt(1.0 - carga**2) * propio
    garch = factor_volatilidad_garch(
        n_periodos, _GARCH_ALFA, _GARCH_BETA, _generador_con_nombre(semilla, "garch")
    )
    choques = volatilidad * garch[:, None] * innovaciones

    # Suma de k exponenciales = Gamma(k, escala). El signo del tamanio decide
    # la direccion: las materias primas saltan al alza y las acciones a la baja.
    # De la cuota de varianza y la frecuencia sale el tamanio: para un Poisson
    # compuesto con magnitudes exponenciales, var = 2 * lambda * escala^2.
    escala_salto = np.where(
        lambda_salto > 0.0,
        np.sqrt(var_saltos / np.maximum(2.0 * lambda_salto, 1e-12)),
        0.0,
    )
    # Y la intensidad sigue al factor GARCH: una crisis trae volatilidad alta
    # y saltos a la vez, que es de donde sale el agrupamiento que se observa.
    intensidad = np.clip(lambda_salto[None, :] * garch[:, None], 0.0, None)
    cuenta = np.column_stack([
        rng_activo[clave].poisson(intensidad[:, i]) for i, clave in enumerate(claves)
    ])
    magnitudes = np.column_stack([
        rng_activo[clave].gamma(np.maximum(cuenta[:, i], 1), escala_salto[i])
        for i, clave in enumerate(claves)
    ])
    saltos = signo_salto * (
        np.where(cuenta > 0, magnitudes, 0.0) - lambda_salto * escala_salto
    )

    # Ornstein-Uhlenbeck discreto. El primer valor sale ya de la distribucion
    # ESTACIONARIA para los activos con reversion: arrancar en cero dejaba un
    # transitorio de mas de 400 sesiones en el crudo, de modo que la primera
    # particion de un walk-forward entrenaba en un regimen de menos varianza
    # que aquel en el que se la evaluaba.
    desviacion = np.empty((n_periodos, len(claves)))
    persistencia = 1.0 - reversion
    sd_estacionaria = np.where(
        reversion > 0.0,
        volatilidad / np.sqrt(np.maximum(1.0 - persistencia**2, 1e-12)),
        0.0,
    )
    desviacion[0] = sd_estacionaria * np.array(
        [rng_activo[clave].standard_normal() for clave in claves]
    )
    for t in range(1, n_periodos):
        desviacion[t] = persistencia * desviacion[t - 1] + deriva + choques[t] + saltos[t]

    estacional = np.column_stack(
        [
            _estacional_estocastica(
                fechas,
                perfil["amplitud_estacional"],
                _generador_con_nombre(semilla, f"estacional:{clave}"),
            )
            for clave, perfil in zip(claves, perfiles)
        ]
    )
    semanal = np.column_stack(
        [_ciclo_semanal(fechas, perfil["tipo"]) for perfil in perfiles]
    )
    log_precio = (
        np.log(campo("nivel")) + estacional + semanal + efecto_exogeno + desviacion
    )
    precios = pd.DataFrame(np.exp(log_precio), index=fechas, columns=claves)

    # Los mercados que no casan en fin de semana no publican precio: sale NaN,
    # que es lo que devuelve cualquier descarga real. La electricidad y el gas
    # si casan los 365 dias.
    fin_de_semana = fechas.dayofweek >= 5
    for posicion, perfil in enumerate(perfiles):
        if perfil["tipo"] in SIN_FIN_DE_SEMANA:
            precios.iloc[fin_de_semana, posicion] = np.nan

    # Las sensibilidades van por unidad de la exogena (grado dia, punto de
    # factor de carga, punto porcentual de llenado); la volatilidad objetivo es
    # anualizada y solo cubre la parte estocastica, asi que la realizada sale
    # algo mayor: las exogenas y los saltos aniaden varianza por encima.
    parametros = {
        "activos": list(claves),
        "inicio": str(fechas[0].date()),
        "n_periodos": int(n_periodos),
        "semilla": semilla,
        "garch": {"alfa": _GARCH_ALFA, "beta": _GARCH_BETA},
        "habilidad_prevision": (
            float(habilidad_prevision) if habilidad_prevision is not None
            else dict(HABILIDAD_POR_VARIABLE)
        ),
        "exogenas_previstas": list(EXOGENAS_PREVISTAS),
        "sensibilidad_hdd": _por_activo(claves, sensibilidades[0]),
        "sensibilidad_cdd": _por_activo(claves, sensibilidades[1]),
        "sensibilidad_viento": _por_activo(claves, sensibilidades[2]),
        "sensibilidad_almacenamiento": _por_activo(claves, sensibilidades[3]),
        "velocidad_reversion": _por_activo(claves, reversion),
        "volatilidad_objetivo": _por_activo(claves, campo("volatilidad")),
        "intensidad_saltos": _por_activo(claves, lambda_salto),
        "cuota_varianza_saltos": _por_activo(claves, cuota_saltos),
        "tamanio_medio_salto": _por_activo(claves, signo_salto * escala_salto),
        "carga_factor_comun": _por_activo(claves, carga),
        "amplitud_estacional": _por_activo(claves, campo("amplitud_estacional")),
        "deriva_anual": _por_activo(claves, campo("deriva_anual")),
        "nivel_medio": _por_activo(claves, campo("nivel")),
    }
    return EscenarioEnergia(
        precios=validar_marco_precios(precios, claves),
        exogenas=exogenas,
        parametros=parametros,
    )


class ProveedorEnergiaSimulado(ProveedorDatos):
    """Proveedor que genera precios de energia en lugar de descargarlos.

    Encaja con :func:`icm.datos.cargar_precios` como cualquier otra fuente, de
    modo que un cuaderno se puede escribir contra la interfaz real y ejecutarse
    sin red. El escenario completo de la ultima llamada queda en
    :attr:`ultimo_escenario`, que es como se recuperan las exogenas y los
    parametros verdaderos despues de haber pedido solo los precios.
    """

    nombre = "energia_simulada"

    def __init__(self, semilla: int | None = 42, **parametros) -> None:
        self.semilla = semilla
        self.parametros = parametros
        self.ultimo_escenario: EscenarioEnergia | None = None

    def descargar(
        self,
        tickers: Sequence[str],
        inicio: str | pd.Timestamp | None = None,
        fin: str | pd.Timestamp | None = None,
        intervalo: str = "1d",
    ) -> pd.DataFrame:
        if intervalo != "1d":
            raise ErrorDatos("El proveedor de energia simulada solo genera series diarias.")
        parametros = dict(self.parametros)
        if inicio is not None:
            parametros.setdefault("inicio", inicio)
        if inicio is not None and fin is not None:
            habiles = len(pd.bdate_range(start=inicio, end=fin))
            parametros.setdefault("n_periodos", max(habiles, 2))

        escenario = simular_energia(activos=tickers, semilla=self.semilla, **parametros)
        self.ultimo_escenario = escenario
        precios = escenario.precios
        if inicio is not None:
            precios = precios.loc[precios.index >= pd.Timestamp(inicio)]
        if fin is not None:
            precios = precios.loc[precios.index <= pd.Timestamp(fin)]
        if precios.empty:
            raise ErrorDatos("La ventana solicitada no contiene sesiones simuladas.")
        return precios
