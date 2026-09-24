"""Aprendizaje automatico para prediccion de precios y estrategias de trading.

El orden de los modulos es el orden en que se usan:

``caracteristicas``
    De una serie de precios a una matriz de variables explicativas, con la
    garantia comprobable de que ninguna usa informacion del futuro.
``validacion``
    Como partir una serie temporal sin hacer trampas: hacia delante, nunca al
    azar, y con un margen entre entrenamiento y prueba.
``modelos``
    Los estimadores: regresion regularizada como referencia honesta y arboles
    potenciados como modelo principal.
``redes``
    Redes recurrentes y convolucionales temporales. Requiere PyTorch, que se
    importa solo si se usa, asi que este paquete funciona sin el.
``evaluacion``
    Si la prediccion es buena, y si lo es de forma estadisticamente
    significativa frente a una referencia ingenua.
``estrategia``
    De la prediccion al dinero: senal, posicion, costes y cuenta de
    resultados, que es la unica metrica que decide de verdad.
``graficos``
    Los graficos del flujo, con la paleta ya validada del paquete.

El modulo ``redes`` NO se importa aqui: cargar PyTorch cuesta segundos y
varios cientos de megas de memoria, y la mayoria de los analisis no lo
necesitan. Importalo explicitamente cuando lo vayas a usar::

    from icm.ml.redes import RedSecuencial, torch_disponible
"""

from icm.ml.caracteristicas import (
    MatrizDatos,
    calendario,
    comprobar_ausencia_de_fuga,
    construir_matriz,
    medias_moviles,
    momento,
    objetivo_rendimiento_futuro,
    retardos,
    volatilidad_realizada,
)
from icm.ml.estrategia import (
    ResultadoEstrategia,
    comparar_estrategias,
    coste_de_equilibrio,
    curva_beneficio,
    ejecutar_estrategia,
    objetivo_volatilidad,
    resumen_estrategia,
    senal_desde_prediccion,
)
from icm.ml.evaluacion import (
    acierto_direccional,
    contraste_diebold_mariano,
    correlacion_informacion,
    error_absoluto_medio,
    error_escalado_medio,
    prediccion_ingenua,
    prediccion_media_movil,
    raiz_error_cuadratico_medio,
    tabla_evaluacion,
)
from icm.ml.modelos import (
    MODELOS,
    REJILLAS,
    MediaCero,
    ajustar_y_predecir,
    buscar_hiperparametros,
    comparar_modelos,
    crear_modelo,
    importancia_variables,
)
from icm.ml.validacion import (
    ParticionWalkForward,
    ResultadoValidacion,
    validacion_walk_forward,
)

__all__ = [
    "MODELOS",
    "MatrizDatos",
    "MediaCero",
    "ParticionWalkForward",
    "REJILLAS",
    "ResultadoEstrategia",
    "ResultadoValidacion",
    "acierto_direccional",
    "ajustar_y_predecir",
    "buscar_hiperparametros",
    "calendario",
    "comparar_estrategias",
    "comparar_modelos",
    "comprobar_ausencia_de_fuga",
    "construir_matriz",
    "contraste_diebold_mariano",
    "correlacion_informacion",
    "coste_de_equilibrio",
    "crear_modelo",
    "curva_beneficio",
    "ejecutar_estrategia",
    "error_absoluto_medio",
    "error_escalado_medio",
    "importancia_variables",
    "medias_moviles",
    "momento",
    "objetivo_rendimiento_futuro",
    "objetivo_volatilidad",
    "prediccion_ingenua",
    "prediccion_media_movil",
    "raiz_error_cuadratico_medio",
    "resumen_estrategia",
    "retardos",
    "senal_desde_prediccion",
    "tabla_evaluacion",
    "validacion_walk_forward",
    "volatilidad_realizada",
]
