# ICM — Análisis de carteras y modelos de riesgo

Librería en Python para construir carteras y medir su riesgo de mercado, con los
datos de Yahoo Finance.

El paquete se organiza en tres capas independientes: de dónde vienen los datos,
qué se hace con ellos y cuánto se puede perder.

```
src/icm/
├── datos/      proveedores, caché en disco, simulador y transformaciones
├── cartera/    métricas, optimización y frontera eficiente
├── riesgo/     VaR, ES, volatilidad condicional, simulación, cópulas y backtesting
└── graficos.py gráficos para los notebooks (opcional, requiere matplotlib)
```

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
| `notebooks/02_modelos_riesgo.ipynb` | VaR y ES comparados, GARCH, Monte Carlo, cópulas y backtesting |

Ambos intentan descargar datos reales y, si no hay salida a internet, **pasan
automáticamente a datos simulados avisando de ello por pantalla**. Así se pueden
ejecutar en cualquier entorno, pero las conclusiones sólo valen con datos reales.

> Los notebooks se guardaron ejecutados **con datos simulados**, porque el entorno
> donde se generaron no tenía acceso a Yahoo Finance. Las cifras que se ven sirven
> para seguir el razonamiento; vuelve a ejecutarlos con conexión para obtener las
> tuyas.

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
- Las rentabilidades esperadas históricas son un estimador muy ruidoso. Por eso
  `minima_varianza`, que no las usa, suele comportarse mejor fuera de muestra que
  `maximo_sharpe`.
- El backtest del VaR con GARCH reestima el modelo cada `frecuencia_reajuste` días,
  no a diario, por coste computacional.
- El diagnóstico del Expected Shortfall (`diagnostico_es`) no es un contraste
  formal con distribución conocida, sino una comparación con intervalo bootstrap.
