"""Mercados de energia: catalogo de activos, simulador, spreads y curvas.

La energia no se comporta como una accion. Sus precios revierten a una media
que se mueve con las estaciones, dan saltos bruscos cuando el sistema se
tensiona y dependen de variables fisicas observables (el frio que hace, cuanto
viento sopla, cuanto gas queda almacenado). Ese ultimo punto es el que hace
interesante aplicar aprendizaje automatico aqui: a diferencia de una accion,
hay variables exogenas con poder predictivo real.

``simulacion``
    Escenarios sinteticos con senal predecible de verdad, para trabajar sin
    acceso a datos de mercado y para contrastar lo que un modelo aprende
    contra la verdad con la que se genero la serie.
``spreads``
    Lo que de verdad se negocia: el margen entre la electricidad que se vende
    y el combustible que se quema, limpio de derechos de emision.
``curvas``
    Estructura temporal de futuros y el rolado de contratos, sin el cual un
    backtest de futuros se inventa el beneficio.
``curvas_oferta``
    La pila de ofertas del diario de OMIE y cuanto mueve el precio operar en
    ella. Es lo que permite dejar de suponer que el activo es precio
    aceptante, hipotesis que a 500 MW se come el 35 % del margen.
``estacionalidad``
    Perfiles, descomposicion y contraste de la estacionalidad anual y semanal.
"""

from icm.energia.almacenamiento import (
    ALMACENAMIENTOS,
    Despacho,
    ParametrosAlmacenamiento,
    ParametrosInversion,
    curva_margen_por_capacidad,
    despacho_optimo,
    liquidar_plan,
    valor_de_la_prevision,
    valor_van_de_la_prevision,
    van_almacenamiento,
)
from icm.energia.curvas_oferta import (
    ELASTICIDAD_OMIE,
    casacion,
    contrastar_casacion,
    impacto_estimado,
    impacto_precio,
    leer_curva_agregada,
    margen_con_impacto,
    tramo_de_precio,
)
from icm.energia.curvas import (
    descomponer_rendimiento,
    dias_a_vencimiento,
    estado_curva,
    interpolar_curva,
    pendiente_curva,
    rendimiento_roll,
    serie_continua,
)
from icm.energia.estacionalidad import (
    Descomposicion,
    contraste_estacionalidad,
    descomponer,
    desestacionalizar,
    desestacionalizar_recursiva,
    periodo_del_indice,
    grados_dia_acumulados,
    indice_estacional,
    perfil_estacional,
    tabla_perfiles,
)
from icm.energia.simulacion import (
    CATALOGO,
    HABILIDAD_PREVISION,
    EscenarioEnergia,
    ProveedorEnergiaSimulado,
    activos_con_yahoo,
    activos_sin_yahoo,
    grados_dia,
    prevision_exogena,
    simular_energia,
    simular_temperatura,
)
from icm.energia.spreads import (
    PLANTAS,
    ParametrosPlanta,
    crack_spread,
    dark_spread,
    dark_spread_limpio,
    horas_en_dinero,
    margen_planta,
    resumen_spread,
    spark_spread,
    spark_spread_limpio,
    spread_calendario,
    spread_cambio_combustible,
    spread_locacional,
    valor_opcionalidad,
)

__all__ = [
    "ELASTICIDAD_OMIE",
    "casacion",
    "contrastar_casacion",
    "impacto_estimado",
    "impacto_precio",
    "leer_curva_agregada",
    "margen_con_impacto",
    "tramo_de_precio",
    "van_almacenamiento",
    "valor_van_de_la_prevision",
    "curva_margen_por_capacidad",
    "ParametrosInversion",
    "valor_de_la_prevision",
    "liquidar_plan",
    "despacho_optimo",
    "ParametrosAlmacenamiento",
    "Despacho",
    "ALMACENAMIENTOS",
    "CATALOGO",
    "Descomposicion",
    "HABILIDAD_PREVISION",
    "EscenarioEnergia",
    "PLANTAS",
    "ParametrosPlanta",
    "ProveedorEnergiaSimulado",
    "activos_con_yahoo",
    "activos_sin_yahoo",
    "contraste_estacionalidad",
    "crack_spread",
    "dark_spread",
    "dark_spread_limpio",
    "descomponer",
    "descomponer_rendimiento",
    "desestacionalizar",
    "desestacionalizar_recursiva",
    "periodo_del_indice",
    "dias_a_vencimiento",
    "estado_curva",
    "grados_dia",
    "grados_dia_acumulados",
    "horas_en_dinero",
    "indice_estacional",
    "interpolar_curva",
    "margen_planta",
    "pendiente_curva",
    "perfil_estacional",
    "prevision_exogena",
    "rendimiento_roll",
    "resumen_spread",
    "serie_continua",
    "simular_energia",
    "simular_temperatura",
    "spark_spread",
    "spark_spread_limpio",
    "spread_calendario",
    "spread_cambio_combustible",
    "spread_locacional",
    "tabla_perfiles",
    "valor_opcionalidad",
]
