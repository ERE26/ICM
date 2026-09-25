# ICM — Carteras, riesgo de mercado y trading de energía

Librería en Python para tres cosas: construir carteras, medir su riesgo de
mercado y aplicar aprendizaje automático al trading de energía.

```
src/icm/
├── datos/      proveedores (Yahoo, tus ficheros), caché y simulador
├── cartera/    métricas, optimización y frontera eficiente
├── riesgo/     VaR, ES, volatilidad condicional, simulación, cópulas y backtesting
├── energia/    spreads de generación, curvas de futuros y estacionalidad
├── ml/         variables, validación temporal, modelos, redes y estrategias
└── graficos.py gráficos para los notebooks (opcional, requiere matplotlib)
```

Cada capa habla con la anterior a través de una interfaz estable, así que se
pueden usar por separado: el VaR no necesita PyTorch y la estrategia de trading
reutiliza tal cual las medidas de riesgo.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -e ".[notebooks,dev]"
```

Requiere Python 3.10 o superior.

## Uso en treinta segundos

```python
from icm.cartera import maximo_sharpe, resumen_metricas
from icm.datos import cargar_precios, matriz_covarianzas, rendimientos, rendimientos_cartera
from icm.riesgo import backtest_var, tabla_medidas

precios = cargar_precios(["AAPL", "MSFT", "JNJ", "XOM"], "2018-01-01", "2024-12-31")
# ...o con tus propios datos:
# from icm.datos import cargar_precios_fichero
# precios = cargar_precios_fichero("datos/precios.csv")
rend = rendimientos(precios, "log")

pesos = maximo_sharpe(rend.mean() * 252, matriz_covarianzas(rend, anualizar=True))
cartera = rendimientos_cartera(rend, pesos)

print(resumen_metricas(cartera))          # rentabilidad, Sharpe, máxima caída...
print(tabla_medidas(cartera))             # VaR y ES por cuatro métodos
print(backtest_var(cartera, 0.99).resumen)  # ¿el modelo de VaR acierta?
```

## Las tres capas

### `icm.datos` — de dónde vienen los precios

El resto de la librería nunca habla con Yahoo Finance: habla con la interfaz
`ProveedorDatos`. Cambiar de fuente es escribir una clase nueva, sin tocar el
análisis.

| Pieza | Para qué |
|---|---|
| `ProveedorYahoo` | descarga cierres ajustados vía `yfinance`, con reintentos |
| `ProveedorFichero` | lee **tus** datos de un CSV, un Excel o un Parquet |
| `ProveedorMarco` | usa un `DataFrame` que ya tienes en memoria |
| `ProveedorConCache` | guarda en Parquet lo descargado; Yahoo limita las peticiones |
| `ProveedorSimulado` | genera series sintéticas para trabajar sin conexión |
| `rendimientos` | precios → rendimientos, logarítmicos o simples |
| `matriz_covarianzas` | muestral, EWMA o con encogimiento de Ledoit-Wolf |

El simulador no genera un paseo aleatorio ingenuo: usa innovaciones t de Student
correlacionadas y un factor de volatilidad común con dinámica GARCH(1,1), de modo
que las series tienen colas gruesas y volatilidad agrupada, como las reales.

```python
from icm.datos import ProveedorSimulado, simular_precios

precios = simular_precios(["A", "B", "C"], n_periodos=1500, semilla=7)
```

#### Meter tus propios datos

```python
from icm.datos import cargar_precios_fichero, ProveedorMarco

# Una columna de fechas y una columna por activo
precios = cargar_precios_fichero("datos/precios.csv")

# CSV exportado de una hoja de cálculo española: ; como separador,
# coma decimal, punto de miles y fechas dd/mm/aaaa
precios = cargar_precios_fichero("datos/ibex.csv", dia_primero=True)

# Formato largo (fecha, ticker, valor), típico de una consulta SQL
precios = cargar_precios_fichero("datos/largo.csv", formato="largo")

# Si lo que tienes son rendimientos, no precios
precios = cargar_precios_fichero("datos/rend.csv", contenido="rendimientos")

# Un DataFrame que ya está en memoria
precios = ProveedorMarco(mi_dataframe).descargar(["AAA", "BBB"])
```

Se leen `.csv`, `.xlsx`, `.xls` y `.parquet`. La carga resuelve sola lo que
suele romper estos ficheros: localiza la columna de fechas, distingue el
decimal del separador de miles, entiende las fechas escritas como `AAAAMMDD`,
arrastra hacia delante el festivo de un mercado que no lo es de otro y se queda
con el último valor si una fecha viene repetida. Lo que no puede adivinar lo
dice con un error concreto en vez de devolver números silenciosamente malos.

Una vez cargados, **todo lo demás funciona igual**: los proveedores comparten
interfaz, así que tus datos pasan por la optimización, los modelos y los
contrastes sin cambiar una línea.

### `icm.cartera` — qué hacer con ellos

Métricas (`ratio_sharpe`, `ratio_sortino`, `ratio_calmar`, `beta`, `alfa_jensen`,
`error_seguimiento`, `contraste_jarque_bera`, `resumen_metricas`) y construcción
de carteras:

```python
from icm.cartera import (
    Restricciones, minima_varianza, maximo_sharpe, paridad_riesgo,
    frontera_eficiente, contribuciones_riesgo, comparar_carteras,
)

restricciones = Restricciones(permitir_cortos=False, peso_maximo=0.35)
pesos = maximo_sharpe(mu, cov, tasa_libre_riesgo=0.02, restricciones=restricciones)

contribuciones_riesgo(pesos, cov)   # quién aporta el riesgo, no sólo el capital
frontera_eficiente(mu, cov, n_puntos=40)
```

`mu` y `cov` deben venir en la misma frecuencia; lo habitual es anualizar ambas.

### `icm.riesgo` — cuánto se puede perder

Todas las medidas devuelven la pérdida como **número positivo**: un VaR al 99 % de
`0.031` significa que se espera perder más del 3,1 % sólo el 1 % de los días.

| Familia | Funciones |
|---|---|
| VaR y ES | `var_historico`, `var_parametrico` (normal y t), `var_cornish_fisher`, y sus `es_*` |
| Volatilidad condicional | `volatilidad_ewma`, `ajustar_garch` (GARCH, GJR, EGARCH), `var_garch` |
| Caídas | `serie_caidas`, `maxima_caida`, `tabla_caidas`, `ulcer_index` |
| Simulación | `simular_cartera`, `bootstrap_historico` (con bloques), `trayectorias_simuladas` |
| Cópulas | `ajustar_copula_gaussiana`, `ajustar_copula_t`, `var_copula`, `comparar_copulas` |
| Backtesting | `backtest_var`, `contraste_kupiec`, `contraste_christoffersen_*`, `zona_basilea` |

Dos cosas que conviene mirar en cualquier análisis serio:

**Las cópulas.** La matriz de correlación resume la dependencia media, pero no dice
nada de los extremos. La cópula gaussiana no tiene dependencia de cola: por mucha
correlación que se le ponga, la probabilidad de que dos activos sufran su peor día
a la vez tiende a cero. La cópula t sí la tiene.

```python
from icm.riesgo import comparar_copulas
comparar_copulas(rend, pesos, nivel=0.99)   # verosimilitud, AIC/BIC y VaR de cada una
```

**El backtesting.** Un modelo de riesgo no vale por lo elegante que sea, sino por
si acierta. Kupiec mira cuántas excepciones hay; Christoffersen, si están agrupadas.

```python
from icm.riesgo import comparar_metodos_var
comparar_metodos_var(cartera, nivel=0.99, ventana=250)
```

### `icm.energia` — el mercado que no se comporta como una bolsa

La energía tiene tres rasgos que ninguna acción tiene: revierte a una media que
se mueve con las estaciones, da saltos bruscos al alza cuando el sistema se
tensiona, y depende de variables físicas observables.

**`spreads`** — Lo que de verdad se negocia. Una central de ciclo combinado no
gana con el precio de la luz ni con el del gas, sino con la diferencia:

```python
from icm.energia import PLANTAS, spark_spread_limpio, valor_opcionalidad

planta = PLANTAS["ciclo_combinado"]
margen = spark_spread_limpio(precio_luz, precio_gas, precio_co2,
                             planta.eficiencia, planta.factor_emision)
valor_opcionalidad(margen)   # la planta puede parar: eso es una opción
```

Para producir 1 MWh eléctrico hacen falta `1/eficiencia` MWh térmicos, así que se
**divide** por la eficiencia. Multiplicar es el error clásico. Y valorar la planta
como si produjese siempre la infravalora: por Jensen, el derecho a parar cuando
el margen es negativo vale dinero.

**`curvas`** — El rolado de futuros, que casi siempre se ignora. Al vencer un
contrato y pasar al siguiente, el salto de precio **no es un rendimiento**, es un
artefacto. Medido en los tests: con un escalón del 10 % entre contratos,
encadenar sin ajustar inventa un rendimiento del 11,1 % el día del roll cuando la
posición rindió un 0,2 %.

```python
from icm.energia import serie_continua, rendimiento_roll, descomponer_rendimiento

continua = serie_continua(contratos, vencimientos, metodo="ratio", dias_antes=5)
descomponer_rendimiento(continua)   # separa el precio del carry
```

**`simulacion`** y **`estacionalidad`** — Escenarios sintéticos con señal
predecible real, para trabajar sin acceso a los mercados, y descomposición
estacional con contraste de Kruskal-Wallis.

### `icm.ml` — aprendizaje automático aplicado al trading

```python
from icm.ml import (construir_matriz, comprobar_ausencia_de_fuga,
                    ParticionWalkForward, comparar_modelos,
                    ejecutar_estrategia, coste_de_equilibrio)

datos = construir_matriz(precios, "TTF", horizonte=1, exogenas=exogenas)
comprobar_ausencia_de_fuga(construir_matriz, precios, "TTF", exogenas=exogenas)

tabla, predicciones = comparar_modelos(datos.X, datos.y,
                                       ["media_cero", "ridge", "arbol_potenciado"],
                                       ParticionWalkForward(n_particiones=5, embargo=1))
estrategia = ejecutar_estrategia(predicciones["ridge"], datos.y, coste_bps=2.0)
coste_de_equilibrio(predicciones["ridge"], datos.y)   # cuánta fricción aguanta
```

Cuatro decisiones de diseño que son el motivo de que el módulo exista:

**La garantía de no mirar al futuro es comprobable.**
`comprobar_ausencia_de_fuga` reconstruye la matriz alterando los datos
posteriores a una fecha y verifica que ninguna variable de esa fecha cambia. No
es un comentario prometiendo cuidado, es un contraste que falla si hay fuga.

**Las particiones van hacia delante, con embargo.** La validación cruzada al azar
entrena con el futuro. Con un objetivo a *h* días, las últimas *h* observaciones
del entrenamiento se solapan con la prueba, y descartarlas cuesta muy poco.

**Toda medida está anclada a una referencia.** `media_cero` predice siempre cero y
es un estimador más, para que entre en la misma validación. El MASE dice si un
modelo la bate, y el contraste de Diebold-Mariano, si esa diferencia es real o
ruido muestral.

**La convención de fechas viaja con los datos.** `objetivo_rendimiento_futuro`
devuelve una serie adelantada y lo marca en `attrs`; `ejecutar_estrategia` lo lee
y se niega a desplazarla dos veces. Medido: el doble desplazamiento convertía un
Sharpe de +2,6 en −0,5 sin dar ningún aviso.

Las redes (`icm.ml.redes`: LSTM, GRU y convolucional temporal) importan PyTorch
de forma diferida, así que el resto funciona sin instalarlo.

### `icm.graficos` — los gráficos (opcional)

Módulo aparte que sólo se usa desde los notebooks; el resto de la librería no lo
importa, así que `icm` funciona sin matplotlib instalado.

```python
from icm import graficos

graficos.aplicar_estilo()
graficos.frontera_eficiente(frontera, carteras=..., recta=recta)
graficos.contribuciones_riesgo(tabla)     # capital frente a riesgo
graficos.caidas(serie_caidas(cartera))    # gráfico "bajo el agua"
graficos.backtest(cartera, var_previsto)  # excepciones marcadas
```

La paleta está fijada y validada para daltonismo, y los colores se asignan en
orden fijo, nunca cíclicamente: un activo conserva su color aunque cambie el
número de series del gráfico. Los cuatro colores de estado (bien, aviso, grave,
crítico) están reservados y nunca se usan como una serie más.

## Notebooks

| Notebook | Contenido |
|---|---|
| `notebooks/01_analisis_carteras.ipynb` | métricas, optimización, frontera eficiente y contribuciones al riesgo |
| `notebooks/03_trading_energia_ml.ipynb` | spark spread, señal predecible, validación temporal, modelos y redes, y el paso del error al dinero |
| `notebooks/02_modelos_riesgo.ipynb` | VaR y ES comparados, GARCH, Monte Carlo, cópulas y backtesting |

Los dos primeros intentan descargar datos reales y, si no hay salida a internet,
**pasan automáticamente a datos simulados avisando de ello por pantalla**. El
tercero trabaja siempre sobre el escenario simulado de `icm.energia`, porque los
precios que necesita (TTF, MIBGAS, OMIE) no tienen feed gratuito.

> Los notebooks se guardaron ejecutados **con datos simulados**, porque el entorno
> donde se generaron no tenía acceso a Yahoo Finance. Las cifras que se ven sirven
> para seguir el razonamiento; vuelve a ejecutarlos con conexión para obtener las
> tuyas.

Y una advertencia que vale la pena repetir sobre el tercero: **el simulador tiene
la señal que se le puso**. Que un modelo la encuentre demuestra que el pipeline
funciona, no que el método gane dinero en el mercado.

```bash
jupyter lab notebooks/
```

## Tests

```bash
pytest
```

Ninguna prueba toca la red: todos los datos salen del simulador con semilla fija,
así que los resultados son idénticos en cualquier máquina. Donde existe una
solución analítica se contrasta contra ella (el VaR normal frente a su fórmula
cerrada, la cartera de mínima varianza con activos independientes frente a los
pesos inversamente proporcionales a la varianza, los umbrales del semáforo de
Basilea frente a los 0-4 / 5-9 / 10 excepciones del supervisor).

### Pasar la batería sobre tus datos

`tests/test_datos_propios.py` ejecuta el análisis completo sobre tu fichero en
lugar de sobre datos simulados. Se salta solo si no encuentra datos, de modo que
la suite sigue pasando en una máquina donde no están.

```bash
cp mis_precios.csv datos/precios.csv
pytest tests/test_datos_propios.py -v -s      # -s para ver lo que informa

# O apuntando a otro sitio
ICM_DATOS=~/tfg/ibex.xlsx pytest tests/test_datos_propios.py -s
ICM_DATOS_CONTENIDO=rendimientos ICM_DATOS_DIA_PRIMERO=1 pytest
```

No comprueba que tus datos «den bien», comprueba que el análisis se sostiene
sobre ellos, que es lo que interesa antes de fiarse de una cifra:

- **Sanidad**: sin huecos ni precios negativos, fechas ordenadas y sin repetir,
  ningún salto diario de más del 50 % (casi siempre un split no ajustado o una
  coma decimal mal puesta), ninguna serie congelada.
- **Estimación**: el optimizador converge, el GARCH da una persistencia
  estacionaria, la cópula t estima unos grados de libertad plausibles.
- **Contraste**: VaR y ES coherentes entre niveles y entre sí, y el backtest de
  cada método con Kupiec, Christoffersen y el semáforo de Basilea.

La distinción importante: que se rechace la normalidad o que un método de VaR no
supere el contraste **no es un fallo del código, es un resultado**, así que esas
pruebas informan por pantalla en vez de fallar. Sólo falla si *ningún* método
supera el contraste, porque entonces hay algo que revisar en los datos.

## Sobre las fuentes de datos

**Yahoo Finance** es la fuente por defecto, vía `yfinance`. No publica un contrato
de API estable: la descarga está aislada en `icm/datos/yahoo.py` y cualquier fallo
se traduce a `ErrorDescarga`. Limita las peticiones, de ahí el caché.

**TradingView** no tiene API pública de datos de mercado. Lo que circula son
endpoints no documentados que sus términos de servicio no contemplan y que se
rompen sin aviso, así que la librería no los usa. Si interesan sus gráficos, lo
razonable es su widget embebido o la librería `lightweight-charts`, alimentados con
los datos que produce este paquete.

**Mercados de energía europeos.** Ni TTF, ni MIBGAS, ni el pool eléctrico de OMIE
tienen feed gratuito en Yahoo. `icm.energia.activos_sin_yahoo()` los enumera. Se
descargan del propio MIBGAS, de ESIOS (el portal del operador del sistema
español) o de ENTSO-E, y se cargan con `cargar_precios_fichero`. Lo que sí está en
Yahoo son los futuros de crudo (`CL=F`, `BZ=F`), el gas Henry Hub (`NG=F`) y los
ETF sectoriales.

## Dos resultados que salieron de este proyecto

No son opiniones: están medidos, y los reproduce el tercer notebook.

**1. En energía, la señal está en las previsiones publicadas, no en el historial
de precios.** Las variables físicas afectan al precio de forma contemporánea, así
que el frío de hoy no anticipa el rendimiento de mañana:

| Relación | r |
|---|---|
| Cambio de temperatura de hoy vs rendimiento de hoy | +0,25 |
| Nivel de temperatura de hoy vs rendimiento de mañana | −0,03 (p=0,23) |
| **Cambio previsto para mañana** vs rendimiento de mañana | **+0,20** |

Y usar esa previsión no es mirar al futuro: el operador del sistema la publica
antes de que cierre el mercado diario. Es la diferencia estructural con la renta
variable, donde no existe un boletín oficial de lo que va a pasar mañana.

**2. El error de predicción y el dinero ordenan los modelos de forma opuesta.**

| Modelo | MASE (error) | Sharpe (dinero) | Coste de equilibrio |
|---|---|---|---|
| árbol potenciado | **1,06** (el mejor) | **−0,13** | 0 pb |
| ridge | 1,24 | +2,63 | 64 pb |
| lasso | 1,26 (el peor) | **+2,99** | 69 pb |

La razón es algebraica, no anecdótica. Como
`MSE(pred) = MSE(0) − 2·cov(pred, y) + var(pred)`, minimizar el error premia la
prudencia en la magnitud; y una posición no cobra por la magnitud, cobra por el
signo. Elegir modelo por el RMSE es elegir el que pierde dinero.

## Limitaciones conocidas

- **Cornish-Fisher tiene un dominio de validez estrecho.** Es una expansión
  asintótica: con asimetría y curtosis altas el polinomio deja de ser creciente
  y el resultado ya no es un cuantil (puede dispararse a decenas de sigmas).
  `var_cornish_fisher` avisa con un `UserWarning` cuando ocurre, y
  `dominio_valido_cornish_fisher(s, k)` permite comprobarlo antes. Con colas muy
  gruesas, el método histórico o la t de Student.
- El escalado de un VaR diario a horizontes largos usa la raíz del tiempo, que
  supone independencia entre días. Con volatilidad agrupada infravalora el riesgo
  más allá de unas dos semanas.
- **El simulador de energía tiene la señal que se le puso.** Que un modelo la
  encuentre valida el pipeline, no el método. Con datos reales los números serán
  peores, y ese es el punto de partida honesto.
- `valor_opcionalidad` modela el derecho a parar periodo a periodo, sin mínimos
  técnicos, rampas ni tiempos mínimos de arranque. Es una cota superior del valor
  real de una central; un modelo de despacho intertemporal es otra cosa.
- El backtest de estrategias supone que se puede operar al precio de cierre y que
  el tamaño de la posición no mueve el mercado. En un mercado de energía estrecho
  ninguna de las dos cosas es cierta, y de ahí que `coste_de_equilibrio` sea la
  cifra que conviene mirar antes que el Sharpe.
- Las series de futuros necesitan rolado (`icm.energia.curvas`). Encadenar
  contratos sin ajustar inventa rendimientos en cada vencimiento.
- Las rentabilidades esperadas históricas son un estimador muy ruidoso. Por eso
  `minima_varianza`, que no las usa, suele comportarse mejor fuera de muestra que
  `maximo_sharpe`.
- El backtest del VaR con GARCH reestima el modelo cada `frecuencia_reajuste` días,
  no a diario, por coste computacional.
- El diagnóstico del Expected Shortfall (`diagnostico_es`) no es un contraste
  formal con distribución conocida, sino una comparación con intervalo bootstrap.
