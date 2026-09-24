"""Aprendizaje automatico para prediccion de precios y estrategias de trading.

El orden de los modulos es el orden en que se usan:

``caracteristicas``
    De una serie de precios a una matriz de variables explicativas, con la
    garantia de que ninguna usa informacion del futuro.
``validacion``
    Como partir una serie temporal sin hacer trampas: hacia delante, nunca al
    azar, y con un margen entre entrenamiento y prueba.
``modelos``
    Los estimadores: regresion regularizada como referencia honesta y arboles
    potenciados como modelo principal.
``redes``
    Redes recurrentes y convolucionales temporales. Requiere PyTorch, que se
    importa solo si se usa.
``evaluacion``
    Si la prediccion es buena, y si lo es de forma estadisticamente
    significativa frente a una referencia ingenua.
``estrategia``
    De la prediccion al dinero: senal, posicion, costes y cuenta de
    resultados, que es la unica metrica que decide de verdad.
"""
