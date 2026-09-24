"""Mercados de energia: catalogo de activos y simulador de precios.

La energia no se comporta como una accion. Sus precios revierten a una media
que se mueve con las estaciones, dan saltos bruscos cuando el sistema se
tensiona y dependen de variables fisicas observables (el frio que hace, cuanto
viento sopla, cuanto gas queda almacenado). Ese ultimo punto es el que hace
interesante aplicar aprendizaje automatico aqui: a diferencia de una accion,
hay variables exogenas con poder predictivo real.
"""
