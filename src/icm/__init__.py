"""icm - analisis de carteras y modelos de riesgo de mercado.

El paquete se organiza en tres capas:

``icm.datos``
    De donde vienen los precios. Un proveedor abstracto, la implementacion de
    Yahoo Finance, un cache en disco y un generador de datos simulados para
    trabajar sin conexion.
``icm.cartera``
    Que hacer con ellos: medir rentabilidad y riesgo, optimizar pesos y
    dibujar la frontera eficiente.
``icm.riesgo``
    Cuanto se puede perder: VaR y ES por varios metodos, volatilidad
    condicional, simulacion, copulas y contraste de los modelos.

Ejemplo minimo
--------------
>>> from icm.datos import cargar_precios, rendimientos
>>> from icm.riesgo import var_historico
>>> precios = cargar_precios(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
>>> var_historico(rendimientos(precios)["AAPL"], nivel=0.99)  # doctest: +SKIP
0.0512
"""

from icm.comun import (
    DIAS_HABILES_ANIO,
    ErrorDatos,
    ErrorDescarga,
    ErrorOptimizacion,
)

__version__ = "0.1.0"

__all__ = [
    "DIAS_HABILES_ANIO",
    "ErrorDatos",
    "ErrorDescarga",
    "ErrorOptimizacion",
    "__version__",
]
