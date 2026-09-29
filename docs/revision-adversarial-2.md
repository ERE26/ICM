# Segunda revisión adversarial: los módulos que la primera no tocó

Septiembre de 2026.

La primera revisión cubrió seis ficheros de `icm.energia` e `icm.ml` y encontró
28 defectos. Quedaban **22 ficheros sin auditar**: todo `riesgo/`, todo
`cartera/`, todo `datos/`, `energia/spreads.py`, `energia/estacionalidad.py`,
`energia/simulacion.py`, `ml/evaluacion.py` y `comun.py`.

Siete revisores independientes, mismas reglas que la vez anterior: encontrar
fallos y no aprobar, y ningún hallazgo válido sin un script que lo reproduzca
con cifras.

**107 hallazgos.** La suite tenía 944 pruebas en verde.

| Revisor | Ficheros | Hallazgos |
|---|---|---:|
| Evaluación | `ml/evaluacion.py`, `comun.py` | 11 |
| Carteras | `cartera/*.py` | 12 |
| Volatilidad y dependencia | `riesgo/volatilidad.py`, `caidas.py`, `montecarlo.py`, `copulas.py` | 19 |
| Datos | `datos/*.py` | 19 |
| VaR | `riesgo/var.py`, `backtesting.py` | 16 |
| Simulador | `energia/simulacion.py` | 17 |
| Spreads y estacionalidad | `energia/spreads.py`, `estacionalidad.py` | 13 |

---

## Lo más grave: la tesis central del proyecto es un artefacto

El README afirmaba dos resultados. El primero — *en energía la señal está en las
previsiones publicadas, no en el historial de precios* — **no sobrevive**.

**Es circular por construcción.** El precio se fija en la línea 648 de
`simulacion.py` sobre el valor *realizado* de la exógena, y la previsión se
fabrica en la línea 465 a partir de ese mismo valor futuro con ruido calibrado.
De ahí sale, algebraicamente:

```
r(prevision, rendimiento) = habilidad x r(contemporanea, rendimiento)
```

Barriendo el parámetro `habilidad` sobre TTF, el cociente entre las dos
correlaciones **es** la habilidad: 0,253 con habilidad 0,30; 0,459 con 0,50;
0,883 con 0,90. El «+0,20» de la tabla era 0,90 × 0,25. Los dos números de esa
tabla eran el mismo número, separados por una constante del fichero.

La hipótesis de mercado incorporada nunca se declaró: **el mercado simulado
ignora la información pública hasta que se materializa.** En OMIE la casación de
D+1 se hace *con* la previsión eólica del operador sobre la mesa, así que esa
previsión ya está en el precio; lo que lo mueve es el error de previsión.

**La mitad negativa también es falsa.** «Los rezagos no sirven» se cumple solo
para el par (TTF, HDD), que es el único que la prueba comprueba. El nivel de
viento de hoy predice el rendimiento de OMIE de mañana con **r = +0,21** (media
de 30 semillas), y una estrategia trivial `signo(viento − media móvil)`, sin
usar ninguna previsión, da **Sharpe 2,71**. Falla en cuatro de los cinco pares
probados. La causa es un número escrito a mano: la persistencia del viento es
0,60 (línea 524) y la del HDD 0,974.

**El 80 % del Sharpe del cuaderno es un almanaque.** La estacionalidad es un
coseno determinista idéntico cada año, y `calendario` emite exactamente la base
`sin_anio`/`cos_anio` que lo recupera. Con la tubería del propio proyecto sobre
TTF:

| configuración | ridge | lasso |
|---|---|---|
| como el cuaderno (todo) | 1,06 | 1,15 |
| sin previsiones | 0,40 | 0,57 |
| **solo calendario (8 columnas de fecha)** | **0,91** | **0,93** |

Ocho columnas de fecha, sin mirar el precio ni el tiempo, reproducen el 80 % del
resultado.

**Lo que sí sobrevive** es el segundo resultado, porque su justificación es
algebraica (`MSE = MSE(0) − 2cov + var`) y no depende del simulador. Los
*niveles* con los que se ilustra, no.

---

## Fugas de información futura

Cuatro, todas reproducidas:

- **EWMA** (`volatilidad.py:65-66`). Se inicializa con la varianza de las 30
  primeras observaciones, así que el valor en t=0 usa datos hasta t=29. Un
  choque metido en t=25 multiplica por **90,7** la volatilidad estimada en t=5.
  El docstring promete literalmente lo contrario. Con series de menos de 30
  observaciones la contaminación es total.
- **`desestacionalizar`** (`estacionalidad.py:538`). Estima el factor estacional
  con toda la muestra, futuro incluido, y la media móvil de tendencia es
  centrada. Triplicar los últimos 36 meses mueve el valor ya publicado de enero
  de 2005 de 51,03 a 44,82. **El 18,8 % de la varianza de la serie resultante es
  revisión que solo se conoce al final.** El módulo la recomienda en su cabecera
  como «la serie que hay que llevar a un modelo», y su razón de existir escrita
  es evitar que un modelo se puntúe leyendo el almanaque.
- **Índice desordenado** (`backtesting.py:58`). `serie_var_movil` construye las
  ventanas por posición sin comprobar que el índice sea creciente. Con una serie
  permutada, **947 de 950 ventanas contienen observaciones posteriores** a la
  fecha que estiman, con una media de 126,7 futuras de 250. El veredicto de
  Basilea cambia de verde a amarilla según la permutación.
- **Rama degenerada de `prevision_exogena`** (`simulacion.py:470`). La condición
  es `std(cambio) == 0`, que se cumple con cualquier rampa lineal, no solo con
  una serie constante. En esos casos devuelve el futuro exacto sea cual sea la
  habilidad pedida: con `habilidad=0.01` el error sigue siendo cero.

---

## Números silenciosamente equivocados

Por orden de daño:

- **`normalizar_pesos` invierte el signo** de toda cartera con exposición neta
  negativa (`comun.py:105`). `{IBEX: +0.3, SP500: −1.3}` sale como
  `{IBEX: −0.3, SP500: +1.3}`, sumando 1,0 y con aspecto impecable. De punta a
  punta, un acumulado real de **−1,17 % se publica como +1,03 %**. La
  volatilidad sí cuadra, porque es cuadrática en los pesos, así que la
  comprobación habitual no lo detecta. Son 13 puntos de llamada en optimización,
  precios, cópulas, Monte Carlo y VaR. **No existía `tests/test_comun.py`.**
- **`maxima_caida` devuelve 0,0** para una cartera que perdió el 50 % el primer
  día (`caidas.py:24`). `curva_valor` arranca *después* del primer rendimiento,
  así que el capital inicial nunca entra en el `cummax`. Afecta al 4,2 % de las
  series al azar y al 100 % de las que empiezan cayendo, que es justo cuando la
  caída máxima importa.
- **El VaR histórico sale un 3,1 % por debajo** (`var.py:33`). Usa la
  interpolación `linear` de numpy, la más optimista de las definiciones
  estándar, cuando el docstring promete «el cuantil empírico». El propio
  `backtest_var` del paquete lo rechaza: sobre 30 000 normales iid da 393
  excepciones frente a 297,5 esperadas, Kupiec p = 1,1e−07, **zona roja**. El
  usuario concluye que su modelo está roto cuando lo roto es el estimador.
- **`paridad_riesgo` no converge y scipy dice que sí** (`optimizacion.py:239`).
  El objetivo no es convexo; el optimizador se queda en caras del símplex con
  activos expulsados a peso cero y devuelve contribuciones perfectamente iguales
  entre los supervivientes, que es lo que uno miraría para validarlo. Falla en
  3/10 carteras con 3 activos y 6/10 con 10.
- **`volatilidad_cartera` depende del orden** en que se escriban los pesos
  (`optimizacion.py:85,91`): la misma cartera 90/10 da 0,190 o 0,540 según el
  orden del diccionario. Factor 2,85.
- **El MASE depende de las unidades** (`evaluacion.py:196`). Un umbral absoluto
  (`q99 < 1.0`) decide si una serie es de rendimientos. Los mismos datos en
  porcentaje dan MASE 0,69 a un modelo que predice cero; en tanto por uno, 1,00.
- **`periodo=252` para series diarias hábiles** (`estacionalidad.py:417`). Un
  año de `bdate_range` tiene 261 días, no 252: la fase se desfasa 9 días por
  año. Con estacionalidad pura de amplitud 40, la desestacionalización deja el
  **32 % de la varianza mensual dentro** y estima la amplitud en 32,4.
- **La volatilidad realizada de OMIE excede en un 52 %** la que el escenario
  declara como verdad (`simulacion.py:633`), porque los saltos y las exógenas no
  entran en el presupuesto de varianza. El sesgo es desigual entre activos
  (+3 % en XLE, +52 % en OMIE), así que las comparaciones entre activos están
  distorsionadas de forma no uniforme.
- **La raíz del tiempo sobre un VaR con media dentro** (`var.py:247`): el VaR
  anual sale un **35,7 % de más**, y el docstring advierte del error contrario.
- **Las cópulas alinean por posición** (`copulas.py:90,142`): la misma
  verosimilitud da +1614 o −2115 según el orden de las columnas.
- **El bootstrap por bloques no es circular** (`montecarlo.py:139`): las
  observaciones de los extremos entran 10 veces menos que las del centro. Como
  las series van en orden cronológico, penaliza las sesiones recientes; la misma
  crisis movida del centro al final pierde el 41 % de su probabilidad de cola.
  El `% n_observaciones` de la línea 143 **es código muerto** y delata que la
  intención era circular.
- **`tamano_efecto` no está corregido por grados de libertad**
  (`estacionalidad.py:604`): con ruido puro vale 0,26 sobre cuatro años diarios,
  «grande» en la escala del propio docstring.
- **Kruskal-Wallis sobre precios diarios rechaza el 100 % de las veces** sin
  estacionalidad, por autocorrelación. El docstring dedica seis líneas a
  justificar la elección del contraste y ninguna a la independencia.
- **Las autocovarianzas de Diebold-Mariano se dividen entre `n−k`** en vez de
  entre `n` (`evaluacion.py:495`), lo que sesga el estadístico al alza.
- **Los p-valores se calculan como `1 − chi2.cdf`** y colapsan a 0,0 exacto a
  partir de LR ≈ 75.

---

## La capa de datos no estaba lista para datos reales

Los cuatro formatos españoles fallaban, tres de ellos **en silencio**:

| Formato | Resultado |
|---|---|
| OMIE diario `dd/mm/aaaa` | 366 días entran, **144 salen** (39 %), sin aviso |
| OMIE horario `Fecha;Hora` | 48 filas entran, **2 salen**; `HORA` pasa a ser un activo |
| MIBGAS con miles `11.250` | se lee **11,25**; la misma columna mezcla dos escalas |
| ESIOS largo, id numérico detrás | marco revuelto: las columnas se llaman por los precios |
| CSV en latin-1 | `UnicodeDecodeError`, sin parámetro de codificación |

Y el rechazo de precios no positivos (`proveedor.py:78`) impide cargar el WTI
(−36,98 el 20 de abril de 2020) y el pool español al completo: de las 102 167
horas descargadas de OMIE, **2 315 están a cero y 3 553 en negativo**, el 5,74 %
repartido por 390 días.

Dos pruebas verdes **defendían el comportamiento equivocado**:
`test_las_fechas_duplicadas_se_resuelven_con_la_ultima` consagra que se borre
una hora real del año en el cambio de horario, y
`test_los_precios_no_positivos_se_rechazan` exige el rechazo que hace la capa
inservible para el mercado al que se dirige el proyecto.

El diagnóstico de fondo cabe en una frase: **esta capa nunca cuenta lo que
descarta.** Filas sin fecha interpretable, celdas coercionadas a `NaN`, fechas
deduplicadas, columnas enteras de `NaN`, huecos rellenados con `ffill`: todo se
resuelve solo y nada se reporta.

---

## Pruebas que consagran el fallo

Nueve, contra dos en la primera revisión:

- `test_los_precios_no_positivos_se_rechazan` — exige lo que rompe el caso de uso.
- `test_las_fechas_duplicadas_se_resuelven_con_la_ultima` — bendice perder una hora del cambio de horario.
- `test_con_cortos_se_pueden_tomar_posiciones_negativas` — solo comprueba que los pesos sumen 1; no hay ni un peso negativo.
- `test_el_mase_no_depende_de_las_unidades` — pasa `referencia` explícita, que es la única rama donde la invarianza se cumple.
- `test_no_se_avisa_con_datos_normales` (Cornish-Fisher) — pasa por suerte de la semilla; con 4 de las 8 primeras fallaría.
- `test_el_bootstrap_por_bloques_toma_tramos_consecutivos` — construye la referencia con `mode="wrap"` y afirma `<=`, así que pasa con implementación circular o no.
- `test_una_serie_constante_se_preve_sin_error` — usa ceros, donde la fuga es invisible.
- `test_sin_la_prevision_no_habria_senal_explotable` — comprueba el único par donde la tesis se cumple.
- `test_desestacionalizar_esta_definida_en_toda_la_muestra` — celebra como virtud que el factor se conozca en los extremos, que es la fuga.

Más dos que no pueden fallar nunca: las dos que comprueban que
`descomponer().reconstruir()` cuadra, cuando el residuo se define como
`valores − tendencia − estacional` y la identidad es una tautología.

---

## Lo que se verificó correcto

Para que conste lo que está cubierto y no hace falta volver a mirar: la
log-verosimilitud de la cópula t coincide **al bit** con
`scipy.multivariate_t.logpdf`; la mínima varianza coincide con la fórmula
cerrada con diferencia de 8,5e−7; la descomposición de Euler suma exactamente la
volatilidad; la corrección de Harvey-Leybourne-Newbold está transcrita carácter
a carácter y usa la t con n−1 grados de libertad; la línea del mercado de
capitales es tangente y no corta la frontera en las 12 combinaciones probadas;
la aritmética de `spreads.py` da error **0,000e+00** contra una reimplementación
independiente en 3 000 casos aleatorios, con la eficiencia dividiendo al
combustible y el factor de emisión por MWh térmico, como promete el docstring;
la descomposición clásica coincide con `statsmodels.seasonal_decompose` a 1e−15;
los umbrales de Basilea para 250 días reproducen la norma; la calibración de la
habilidad de previsión es exacta (0,9001 medido sobre 200 semillas para un
nominal de 0,90); `_ar1` cumple su contrato al milímetro; y no hay ni un
`bfill` ni un `interpolate` en toda la capa de datos.

El patrón es el mismo que en la primera ronda: **las fórmulas están bien**. Lo
que falla son los bordes, las convenciones y los casos que ninguna prueba tocó —
la primera observación, el orden de un diccionario, una serie corta, una cartera
neta corta, un día que el mercado cerró a cero.

---

## Estado: el simulador, cerrado y recalibrado

Septiembre de 2026. Los 17 hallazgos del simulador están cerrados, y el
módulo se rehízo contra datos del mercado español en vez de contra las
intuiciones de quien lo escribió.

### Lo que cambió

| | Antes | Ahora | Real |
|---|---:|---:|---:|
| Volatilidad de OMIE | 1,14 | 7,20 | 7,69 |
| Volatilidad de MIBGAS | 0,73 | 1,07 | 1,16 |
| Curtosis de OMIE | 0,5 | 39,4 | 15,0 |
| Curtosis de Henry Hub | 5,6 | 107,9 | 90,4 |
| Ciclo semanal (domingo) | no existía | −0,248 | −0,256 |
| Correlación Henry Hub-Brent | +0,312 | +0,010 | +0,026 |
| Correlación WTI-Brent | +0,519 | +0,628 | +0,687 |
| Mínimo de OMIE / su media | 0,54 | 0,006 | 0,002 |
| Perfil mensual repetido año a año | +1,00 | +0,13 | +0,08 |
| Asimetría de XLU | +1,81 | −1,42 | (negativa) |

La circularidad se rompió invirtiendo la causalidad: **la previsión se genera
primero y la realidad se deriva de ella**. Antes, el cociente entre la
correlación de la previsión y la contemporánea era *exactamente* el parámetro
`habilidad` (0,30 → 0,253; 0,90 → 0,883). Ahora ese parámetro gobierna lo que
dice gobernar —cuánto del clima es predecible— y ya no fija el resultado.

El reparto conserva la varianza del clima, y esto costó dos intentos: el
primero escalaba la sorpresa hacia arriba, con lo que **una previsión peor
hacía el clima más variable** y la correlación contemporánea *subía* al
empeorar la previsión. Una previsión mala no hace que llueva más.

### Donde el revisor se equivocó

El hallazgo *«TTF y MIBGAS correlacionan 0,22 cuando en el mundo real son casi
el mismo producto»* y su acompañante *«MIBGAS y OMIE deberían ir a 0,7-0,8»*
eran **la intuición del revisor, no un dato**. Medido sobre 1 652 días
comunes:

| Par | Real | Simulador de entonces |
|---|---:|---:|
| MIBGAS-OMIE | **+0,168** | +0,201 |
| Henry Hub-Brent | **+0,026** | +0,312 |
| WTI-Brent | **+0,687** | +0,519 |

El simulador acertaba en el par que el revisor criticaba y fallaba en los
otros dos, en dirección contraria: el defecto no era que los mercados
relacionados correlacionasen poco, sino que los **no relacionados**
correlacionaban demasiado, porque un único factor global los ataba a todos.

Es el recordatorio de que un revisor adversarial con buen criterio sigue
teniendo priores, y que la regla de «ningún hallazgo sin una reproducción con
cifras» no protege de un prior cuando lo que se afirma es sobre el mundo y no
sobre el código.

### Lo que sigue sin reproducir

El **agrupamiento de volatilidad**. La autocorrelación de los rendimientos en
valor absoluto sale entre 0,07 y 0,13 y en los datos va de 0,25 a 0,44, pese a
llevar el GARCH casi al límite IGARCH (α+β = 0,995), que es donde lo sitúa el
ajuste sobre las series reales. La causa es que las componentes exógena,
estacional y semanal no se agrupan, y ahora pesan más. Queda documentado en la
cabecera del módulo en lugar de disimulado.

---

## Estado: volatilidad y dependencia, cerradas

Septiembre de 2026. El bloque de `riesgo/volatilidad.py`, `caidas.py`,
`montecarlo.py` y `copulas.py`. Las cifras de cada fila están reproducidas con
un script; las pruebas nuevas se pasaron antes contra el código viejo para
comprobar que fallaban.

### Caídas

| | Antes | Ahora |
|---|---:|---:|
| `maxima_caida` de `[−50 %, +30 %, +30 %, +30 %]` | +0,0000 | **−0,5000** |
| Filas de `tabla_caidas` en esa serie | 0 | **1** |
| `periodos_hasta_suelo` con una meseta antes de la caída | 3 | **1** |
| `pico` en esa meseta | primer día | **último día** |

`curva_valor` arrancaba *después* del primer rendimiento, así que el capital
invertido nunca entraba en el máximo acumulado y la peor caída posible —la que
empieza el primer día— salía como cero. Ahora la curva arranca en el capital.
En los empates del máximo se toma **el último**, que es el pico desde el que
la caída se mide de verdad.

### Monte Carlo

- **El bootstrap por bloques ya es circular.** Las observaciones de los
  extremos entraban 10 veces menos que las del centro; medido sobre la
  cobertura de cada índice, la razón entre la más y la menos muestreada pasa
  de 10 a **1,0**. Como las series van en orden cronológico, el sesgo caía
  sobre las sesiones recientes.
- Un `tamano_bloque` mayor que la muestra ya da `ErrorDatos` en vez de
  devolver trayectorias silenciosamente degeneradas.
- Los rendimientos logarítmicos pasan por `expm1` **antes** de ponderarse: el
  rendimiento de una cartera no es la media ponderada de los logaritmos.
- Las etiquetas de nivel dejaron de colisionar: `99.5 %` y `99.9 %` se
  redondeaban las dos a `100%` y la segunda pisaba a la primera. Las claves
  son ahora `var_99%`, `var_99.5%`, `var_99.9%`.
- `resumen_simulacion` con una sola simulación da error en lugar de una
  desviación típica `NaN` presentada como resultado.

### Cópulas

| | Antes | Ahora |
|---|---|---|
| Log-verosimilitud con las columnas invertidas | **+788,19 / −3654,30** | idéntica, +788,19 |
| Correlación declarada vs. la que se simula | 0,9200 / 0,8894 | 0,8898 / 0,8894 |
| Matriz con autovalor −0,98 | se reparaba en silencio | `ErrorDatos` |
| `simular_rendimientos_copula` con columnas numéricas | `KeyError: 0` | funciona |
| `var_copula` con rendimientos logarítmicos | sumaba logaritmos | `expm1` primero |
| `CopulaT(..., grados_libertad=1.5)` | se aceptaba | `ErrorDatos` |

Las uniformes se cruzaban con la matriz de correlación **por posición** aunque
los dos objetos llevan etiquetas: bastaba un `reindex`, un `merge` o un
`groupby` para evaluar la verosimilitud de otro modelo. Medido sobre tres
activos, la gaussiana daba +788,19 con las columnas en su orden y −3654,30
con el inverso, sobre exactamente los mismos datos; la t, +786,98 y −2687,15.
Eso cambia el signo del AIC en `comparar_copulas`, que es la tabla con la que
se elige entre las dos. Ahora se alinean por nombre, y que falte una columna
es un error en vez de un resultado.

La proyección sobre las definidas positivas era la otra mitad del problema:
`simular_uniformes` y `log_verosimilitud` proyectaban, `dependencia_cola` no,
y el objeto **declaraba una dependencia de cola y generaba otra**. Con una
matriz cuyo menor autovalor es −0,043, la cópula guardaba ρ(A,C) = 0,92,
declaraba una dependencia de cola de 0,7040 y simulaba una muestra cuya
correlación por rangos es 0,8894. Ahora la matriz se valida y se proyecta
**una sola vez, al construir la cópula**, así que los tres métodos hablan de
la misma: ρ = 0,8898 guardado, 0,8894 simulado, dependencia de cola 0,6544.
(El revisor lo había medido sobre otra matriz, con 0,5945 declarado frente a
0,164 simulado; esa matriz ahora se rechaza directamente.)

Una desviación pequeña avisa con un `RuntimeWarning` —reparar mueve las
correlaciones que se han pedido, y todo lo que salga después describe la
matriz reparada— y una grande se rechaza.

Comprobado de punta a punta sobre siete series reales (OMIE, MIBGAS, Brent,
WTI, Henry Hub, VIX y EUR/USD, 2 118 días comunes): el menor autovalor de la
matriz de Kendall es +0,162, así que no hay nada que reparar ni aviso que dar,
y la t gana a la gaussiana por 451 puntos de AIC con 10,96 grados de libertad.
Su ES al 99 % sale un 7 % por encima, que es exactamente el riesgo de cola que
la gaussiana no ve.

### Volatilidad

Dos hallazgos propios, encontrados al repasar el módulo entero y no solo lo
que el revisor había nombrado:

- **`volatilidad_movil` mira el día que estima.** Su valor en `t` incluye
  `r_t`, mientras que `volatilidad_ewma` y el GARCH están desplazados. Con un
  rendimiento del 50 % metido en un solo día, la ventana de 21 pasa de 0,0098
  a 0,1095 **en la fecha del propio choque** (×11,2) y la EWMA de esa fecha no
  se mueve. No es un fallo en sí —es volatilidad realizada—, pero
  `objetivo_volatilidad` la recomendaba en su docstring para dimensionar la
  posición, que es la quinta fuga de futuro del proyecto. Medido con señal
  constante y objetivo del 10 % anual:

  | Serie | Volatilidad realizada | Peor día | Máxima caída |
  |---|---:|---:|---:|
  | MIBGAS, con fuga | 10,21 % | −2,53 % | −18,61 % |
  | MIBGAS, sin fuga | 11,26 % | −5,37 % | −18,54 % |
  | OMIE, con fuga | 10,19 % | −2,56 % | −6,74 % |
  | OMIE, sin fuga | **19,85 %** | −5,12 % | −9,37 % |
  | Brent, con fuga | 10,19 % | −2,49 % | −48,12 % |
  | Brent, sin fuga | 11,12 % | −4,46 % | −49,39 % |
  | WTI, con fuga | 10,12 % | −2,82 % | −34,78 % |
  | WTI, sin fuga | 11,24 % | **−11,74 %** | **−45,03 %** |

  La señal de que algo va mal es que la volatilidad realizada dé 10,1-10,2 %
  en las cuatro series: el objetivo clavado al decimal, porque el tamaño de la
  posición ya sabe cuánto se va a mover el mercado ese día. `volatilidad_movil`
  acepta ahora `prevision=True`, que la desplaza y la deja alineada con las
  otras dos.

- **El ES del GARCH integraba la cola por la derecha.** Tomaba el extremo
  derecho de cada tramo, que es el cuantil menos extremo, así que la integral
  salía corta **siempre hacia el lado inseguro**: con una t de 3 grados al
  99 %, 4,0176 frente al valor exacto 4,0432. Con la regla del punto medio el
  error baja un orden de magnitud. De paso, los cuantiles se piden
  vectorizados, que es lo que hace viable comprobarlo: integrar con 200 000
  tramos pasa de **74 segundos a 0,19**.

### Una prueba reescrita

`test_el_bootstrap_por_bloques_toma_tramos_consecutivos` construía la
referencia con `mode="wrap"` y comparaba con `<=`, así que la cumplían por
igual la implementación circular y la que no lo es: el conjunto envolvente
contiene al otro. Ahora separa las ventanas que cruzan el final y **exige que
salgan**, que es justo lo que el muestreo no circular no puede producir.

---

## Estado: VaR y backtesting, cerrados

Septiembre de 2026. `riesgo/var.py` y `backtesting.py`. Como en el bloque
anterior, cada cifra está reproducida con un script y las pruebas nuevas se
pasaron antes contra el código viejo: **24 de las 28 fallan**.

### El hallazgo que el revisor no vio, y que era el más grave

**El criterio de dominio de Cornish-Fisher rechazaba el 57 % de las muestras
normales.** El revisor había anotado que
`test_no_se_avisa_con_datos_normales` «pasa por suerte de la semilla; con 4 de
las 8 primeras fallaría». Es cierto, pero la conclusión no era que la prueba
fuese débil: era que **el código estaba mal**.

`dominio_valido_cornish_fisher` aplicaba el criterio de Maillard en su versión
global —que la derivada del polinomio sea positiva en *toda* la recta—, y esa
condición exige `a = k/8 − s²/6 ≥ 0`. Con asimetría cercana a cero eso se
reduce a exigir **curtosis de exceso no negativa**, y la mitad de las muestras
normales tienen curtosis muestral negativa. Medido sobre 2 000 muestras:

| Tamaño de la muestra | Muestras normales rechazadas |
|---|---:|
| 250 | **57,2 %** |
| 1 000 | 53,3 % |
| 2 000 | 54,2 % |

Y las rechazaba por nada. Con `s = 0` y `k = −0,02`, la monotonía se rompe en
`|z| = 20`; el VaR al 99,99 % vive en `z = −3,72` y el punto más extremo que
toca el ES integrado con 1 000 tramos está en `z = −4,42`. El aviso decía
literalmente «la expansión deja de ser monótona» de una expansión que era
monótona en todo el rango utilizable.

Ahora el criterio se evalúa **en el tramo que cada función usa**:
`var_cornish_fisher` comprueba de la mediana al cuantil de su nivel,
`es_cornish_fisher` hasta el extremo de la cola que integra, y
`_var_movil_cornish_fisher` lo mismo que el VaR. El criterio global sigue
disponible sin argumentos de rango, documentado como la pregunta matemática
que es. La tasa de falsas alarmas sobre muestras normales pasa de **57 % a
0,0 %**, y lo que hay que rechazar se sigue rechazando: `(−1,2, 29,0)` es
inválido, y `k = −0,5` también lo es dentro del rango del ES, porque ahí la
monotonía se rompe de verdad en `z = −4,1`.

### El VaR histórico: el revisor tenía razón, y también le faltaba una mitad

Confirmado de forma independiente. `np.quantile` interpola por omisión
(`linear`) y **siempre cae del lado menos extremo**: con 250 observaciones al
99 % el índice es `0,01 × 249 = 2,49`, un punto entre el tercer y el cuarto
peor día en lugar del tercero. Sobre 30 000 normales iid con ventana móvil de
250:

| Definición del cuantil | Excepciones | Tasa | Kupiec p | VaR medio |
|---|---:|---:|---:|---:|
| `linear` (la que había) | 401 | 1,348 % | 1,0e−08 | 0,022809 |
| `inverted_cdf` (la empírica) | 348 | 1,170 % | 0,0042 | 0,023488 |
| `higher` | 482 | 1,620 % | 5,9e−23 | 0,022102 |

El VaR salía un **2,9 %** por debajo (el revisor dijo 3,1 %, con otra
muestra). Corregido a `inverted_cdf`, que es la definición de la función de
cuantiles empírica: `Q(p) = inf{x : F(x) ≥ p}`.

**Pero el 0,0042 sigue rechazando, y eso ya no es un defecto del código.** Con
250 observaciones el cuantil al 99 % *es* el tercer peor día, cuya
probabilidad de ser superado es `3/251 = 1,1952 %`. La tasa medida es 1,170 %:
coincide. Ese exceso es una propiedad de la ventana, no del estimador, y se
quita alargando la ventana o bajando el nivel. El hallazgo del revisor
concluía «el usuario concluye que su modelo está roto cuando lo roto es el
estimador»; la mitad que faltaba es que, arreglado el estimador, **una parte
del rechazo se queda**, y confundir las dos cosas llevaría a seguir buscando
un fallo que no existe. Está escrito en el docstring.

### El resto, por orden de daño

| | Antes | Ahora |
|---|---|---|
| VaR anual por raíz del tiempo con deriva del 0,05 % diario | 0,3614 (**+48,5 %**) | 0,2433 |
| `es_parametrico(..., "t")` con t de 0,8 grados | ES = **−3,0654** con VaR = +0,6194 | `ErrorDatos` |
| p-valor con LR = 75 | **0,0 exacto** | 4,7e−18 |
| Christoffersen con el índice barajado | LR 76,38 → **0,37**, p 0,0000 → 0,5422 | `ErrorDatos` |
| Kupiec con una serie de rendimientos | «ninguna excepción» | `ErrorDatos` |
| Recargo de Basilea con 500 obs. y 12 excepciones | **0,85**, con aspecto regulatorio | `NaN` + `recargo_aplicable=False` |
| `_momentos` con 2 observaciones | asimetría 0,0 y curtosis −2,0 *siempre* | `ErrorDatos` |
| ES de Cornish-Fisher, 1 000 tramos | −0,043 % (integral por la derecha) | −0,003 % (punto medio) |

**La raíz del tiempo sobre la media.** Un VaR es `−(μ + σz)`. A T días la media
se acumula linealmente y la desviación típica con la raíz, así que el VaR
correcto es `−(Tμ + √T σz)`, no `√T` veces el diario. `escalar_horizonte`
acepta ahora `media_diaria`, aplica `√T (VaR₁ + μ) − Tμ` —que con `μ = 0` se
reduce a la regla de siempre— y `var_anualizado` le pasa la media de la
muestra. El error era del +4,99 % a 10 días y del +48,53 % a un año. El
docstring advertía del error contrario.

**El ES de la t sin grados de libertad.** La fórmula cerrada divide entre
`gl − 1` sin comprobarlo. Con una muestra de una t de 0,8 grados el ajuste da
0,832 y el ES sale **negativo** con un VaR positivo: un ES menor que el VaR es
imposible por definición, porque es una media de pérdidas que ya superan el
VaR. Por debajo de un grado de libertad la t no tiene media y el ES
simplemente no existe, así que ahora es un error; por debajo de dos no tiene
varianza, y eso es un aviso.

**Los p-valores.** `1 − chi2.cdf` agota la precisión del flotante y devuelve
0,0 exacto a partir de LR ≈ 75. Con `chi2.sf`, LR = 75 da 4,7e−18 y LR = 200
da 2,1e−45. Un cero exacto no es un p-valor, y además impide distinguir un
rechazo claro de uno aplastante.

**El orden en Christoffersen.** La corrección anterior puso `excepciones` a
devolver en orden cronológico, pero nada impedía llamar al contraste
directamente con un marco sin ordenar: el contraste cuenta transiciones de un
día al siguiente sobre la secuencia que recibe, así que **el orden es el
dato**. Con las mismas 12 excepciones agrupadas en una racha, barajar el
índice llevaba el estadístico de 76,38 (rechaza) a 0,37 (no rechaza). Ahora se
exige índice monótono, y de paso se comprueba que los valores sean 0 o 1:
`asarray(dtype=int)` truncaba hacia cero, así que una serie de rendimientos
pasada por error se convertía en un vector de ceros y colaba como «ninguna
excepción», con su Kupiec y todo.

### Donde el revisor volvió a equivocarse

**El ES histórico no tenía el problema del cuantil.** El hallazgo era sobre
`var.py:33`, pero merecía comprobar si arrastraba a `es_historico`. No: con
30 000 observaciones al 99 %, `linear` e `inverted_cdf` dan umbrales distintos
(−0,023611 y −0,023660) pero **la misma cola de 300 observaciones y el mismo
ES hasta el sexto decimal**, porque el umbral cae entre dos observaciones y el
corte `<= umbral` no cambia de lado. Se ha corregido de todas formas por
coherencia, pero no era un hallazgo.

**`var_incremental` tampoco lo era.** Renormaliza los pesos restantes, lo que
en principio podría hacer que quitar un activo pequeño *subiera* el VaR por
reapalancamiento. Medido con `{A: 0,90, B: 0,05, C: 0,05}` sobre 1 000
sesiones, no pasa: quitar B da −0,00054, es decir baja el VaR. La
renormalización es la convención habitual —comparar dos carteras invertidas al
100 %— y lo que hacía falta era decirlo en el docstring y apuntar a
`contribuciones_riesgo` para quien quiera la descomposición de Euler.

### Comprobado con datos reales

`comparar_metodos_var` sobre las 2 500 últimas sesiones de MIBGAS y de Brent,
ventana de 250 al 99 % (esperadas 22,5 excepciones):

| Serie | Método | Excepciones | Kupiec p | Independencia p | Zona |
|---|---|---:|---:|---:|---|
| MIBGAS | histórico | 30 | **0,1305** | 2,8e−05 | verde |
| MIBGAS | paramétrico normal | 46 | 1,3e−05 | 2,1e−06 | roja |
| MIBGAS | Cornish-Fisher | 25 | **0,6028** | 1,0e−04 | verde |
| MIBGAS | EWMA | 42 | 2,3e−04 | 0,8117 | roja |
| Brent | histórico | 34 | 0,0235 | 0,0144 | amarilla |
| Brent | paramétrico normal | 58 | 3,4e−10 | 0,0189 | roja |
| Brent | Cornish-Fisher | 31 | 0,0885 | 0,0082 | amarilla |
| Brent | EWMA | 47 | 5,9e−06 | 0,3552 | roja |

Tres cosas que se leen bien ahí, y ninguna se podía leer antes:

- El **paramétrico normal está en zona roja en las dos series**, con más del
  doble de excepciones de las esperadas. Es el resultado correcto: suponer
  normalidad en materias primas energéticas no se sostiene.
- El **histórico y Cornish-Fisher aprueban Kupiec pero suspenden
  independencia**, y el EWMA justo al contrario (p = 0,8117 en MIBGAS). Es el
  reparto de papeles del manual: los estimadores de ventana aciertan el número
  de excepciones y no reaccionan al régimen; el EWMA reacciona y por eso las
  reparte bien, pero calibra bajo. Con el criterio de dominio viejo,
  Cornish-Fisher habría avisado en más de la mitad de las 2 250 ventanas y
  esta comparación no habría sido legible; ahora avisa en el 12-13 %, y son
  episodios de colas gruesas de verdad.
- Con el **histórico completo** en lugar de las últimas 2 500 sesiones el
  veredicto se endurece, y conviene saberlo antes de presumir de un aprobado:
  MIBGAS pasa de verde a amarilla (53 excepciones sobre 3 626 ventanas frente
  a 36,26 esperadas, Kupiec p = 0,0089) y Brent, de amarilla a roja (125 sobre
  8 841 frente a 88,41, p = 2,3e−04, pero independencia p = 0,3915). El
  `parametrico_t` da en MIBGAS exactamente las mismas 53 excepciones que el
  histórico. Ninguno de los cuatro métodos aprueba el contraste condicional en
  ninguna de las dos series, que es la conclusión honesta: **un VaR de ventana
  fija no modeliza bien una materia prima energética**, y la librería ahora lo
  dice en vez de taparlo.
- El VaR anual de MIBGAS sale **312,82 %** con la media escalada linealmente
  frente a 328,56 % con la raíz del tiempo ingenua. Los dos números son
  disparatados, y eso no es un fallo: son rendimientos logarítmicos y la regla
  de la raíz del tiempo a un año no vale, como dice el docstring. La
  diferencia entre ambos es exactamente la deriva.

---

## Estado: carteras, cerradas

Septiembre de 2026. `cartera/optimizacion.py`, `frontera.py` y `metricas.py`.
Cada cifra reproducida con un script; el módulo de pruebas pasa de 41 a 61
casos y **12 de los nuevos fallan contra el código viejo**.

### El orden de las claves de un diccionario decidía el resultado

Confirmado exacto. `volatilidad_cartera` y `rentabilidad_cartera` llamaban a
`normalizar_pesos(pesos)` **sin etiquetas**, así que el vector salía en el
orden en que viniera el diccionario, y luego se multiplicaba por una matriz
que estaba en el orden de *sus* columnas. Con dos activos de volatilidad 20 %
y 60 %:

| Pesos escritos así | Volatilidad | Rentabilidad |
|---|---:|---:|
| `{A: 0.9, B: 0.1}` | 0,1897 | 0,0600 |
| `{B: 0.1, A: 0.9}` | **0,5404** | **0,1400** |
| correcto | 0,1897 | 0,0600 |

Factor 2,85, exactamente lo que medía el revisor. Y lo más revelador:
`contribuciones_riesgo`, en el mismo fichero y doce líneas más abajo, **sí
alineaba** —pasaba `marco_cov.columns` a `normalizar_pesos`—, así que la
volatilidad total que sale de sumar las contribuciones y la que devolvía
`volatilidad_cartera` podían diferir por un factor de tres sin que nada
chirriara. Ahora las tres pasan por un único `_pesos_alineados`, que además
rechaza un vector de longitud equivocada en lugar de propagar el error.

### Paridad de riesgo: el objetivo no era convexo

El revisor tenía razón, y mi primer intento de reproducirlo **no encontró
nada**: con covarianzas de datos normales aleatorios, bien condicionadas,
salían 0 fallos de 10. El fallo aparece donde la paridad de riesgo es difícil
de verdad, con volatilidades muy dispares:

| Activos | ρ | Fallos / 20 | Peor dispersión | Activos a peso cero |
|---:|---:|---:|---:|---:|
| 3 | 0,00 | 1 | 0,500 | 1 |
| 10 | 0,00 | **6** | 0,250 | **9** |
| 10 | 0,50-0,97 | 0 | 0 | 0 |

(Volatilidades sorteadas entre el 0,3 % y el 50 %.) El objetivo era la suma de
cuadrados de las diferencias entre contribuciones relativas, que **no es
convexo**: SLSQP se quedaba en caras del símplex con activos expulsados a peso
cero y devolvía `success=True`, y lo que uno miraría para validarlo —que las
contribuciones de los supervivientes salgan iguales— también cuadraba.

Sustituido por la formulación logarítmica estándar (Maillard et al. 2010;
Spinu 2013):

```
minimizar  f(y) = ½ y'Σy − Σᵢ bᵢ log(yᵢ)     con y > 0
```

que es **estrictamente convexa** en el ortante positivo, tiene mínimo único y
cuyo punto crítico cumple `yᵢ(Σy)ᵢ = bᵢ` exactamente —la condición de paridad—,
con los pesos saliendo de reescalar `y`. Se resuelve con el gradiente
analítico `Σy − b/y`. Dos detalles que costaron una iteración: hay que
normalizar la matriz a diagonal media 1, porque si no los dos términos del
objetivo viven a cuatro órdenes de magnitud de distancia y la búsqueda lineal
aborta; y el indicador de `scipy` **no es el criterio**, porque con este
objetivo dice `ABNORMAL` cuando ya está en el óptimo. Quien juzga es
`_verificar_paridad`, que comprueba la condición de primer orden sobre el
resultado y levanta `ErrorOptimizacion` con los números si no se cumple. Esa
comprobación es la que faltaba: `success=True` no dice nada sobre si el óptimo
encontrado reparte el riesgo.

Resultado sobre las mismas 320 matrices: **0 fallos, ningún activo expulsado,
peor dispersión 3,1e−04** frente a 0,25.

### Restricciones que se contradecían en silencio

- `Restricciones(permitir_cortos=False, peso_minimo=-0.5)` devolvía
  `[(-0.5, 1.0), ...]`: una cartera con posiciones cortas pedida
  explícitamente **sin** cortos. Ahora es un error.
- Con `permitir_cortos=True` el tope quedaba en 1,0, así que un **130/30 —la
  razón de ser de permitir cortos— era infactible** sin que nada lo dijera. El
  tope pasa a ser el máximo que de verdad puede alcanzar un peso dado el suelo
  de los demás, `1 − (n−1)·inferior`.

### La frontera y la recta

- **`frontera_eficiente` descartaba puntos en silencio.** Con suelo del 5 % y
  techo del 25 % sobre ocho activos muy correlados salían **30 de los 40**
  puntos pedidos y nada lo decía: una frontera recortada y una completa se
  veían igual. Ahora avisa y el número queda en
  `attrs["puntos_descartados"]`.
- **Y presentaba como eficientes puntos dominados.** Con cortos permitidos
  sobre activos muy correlados, SLSQP se queda en mínimos locales: en 10 de
  unas 120 fronteras aparecían puntos con *menos* rentabilidad a *más*
  volatilidad, hasta **4,5 puntos porcentuales** perdidos. Ahora se marcan en
  la columna `eficiente` y se avisa.
- **`linea_mercado_capitales` dibujaba rectas con pendiente negativa.** Si
  ninguna cartera bate al activo sin riesgo, la recta baja —más riesgo, menos
  rentabilidad— y el objeto devolvía esa recta con su `peso_en_tangente` y
  todo. Con `mu = {A: −5 %, B: −2 %}` y tasa libre del 3 %, pendiente
  −0,0833. Ahora es un error que dice que lo óptimo con esas estimaciones es
  no invertir.

### Dos Sharpes distintos en el mismo paquete

`cartera.ratio_sharpe` usaba la tasa **geométrica** en el numerador y
`ml.estrategia` la **aritmética**, sin que nada lo dijera, así que comparar un
número de un módulo con uno del otro no era una comparación válida. La
diferencia es estructural: el numerador geométrico es el aritmético menos
`σ²/2`, que con volatilidad del 2 % diaria vale **5,04 puntos porcentuales**
sobre los 12,6 de la media aritmética. Medido con 200 000 observaciones,
+0,2488 frente a +0,3985 (−37,6 %, contra el −40,0 % que predice la fórmula).

`ratio_sharpe` acepta ahora `media="aritmetica"`, que reproduce exactamente
`ml.estrategia._sharpe_aritmetico`, y el docstring dice cuál usa cada módulo y
por qué. El valor por omisión no se ha tocado: cambiarlo movería todas las
cifras ya publicadas.

En muestras cortas hay que mirar la diferencia absoluta y no el porcentaje:
sobre ocho semillas de 2 000 observaciones el «hueco» relativo va del −87 % al
+317 %, porque es un cociente de dos números pequeños, mientras que la resta
en el numerador sigue siendo esos 5,04 puntos.

### Etiquetas numéricas

`_preparar` hacía `str()` de las columnas de la covarianza y luego
`mu.reindex(["0", "1"])`, que con índices numéricos daba todo NaN. El error
que salía —«hay rentabilidades esperadas ausentes»— mandaba a buscar un hueco
en los datos que no existía: lo que no encajaba era el tipo de la etiqueta.
Ahora se reindexa por las etiquetas originales y el mensaje distingue los dos
casos, además de nombrar qué activos faltan.

### Donde me equivoqué yo

Marqué como hallazgo que `frontera_eficiente` devolvía fronteras no monótonas
con techo del 20 %. Al medir la violación resultó ser de **1,4e−16**: ruido de
coma flotante, no un punto dominado. El detector se ha quedado —con tolerancia
de 1e−12, que absorbe el ruido— porque los puntos dominados de verdad existen,
pero aparecen con cortos permitidos, no con techo.

También sospeché que `ratio_sharpe` mezclaba unidades al dividir una tasa
compuesta de rendimientos simples por una volatilidad de logarítmicos. No: la
misma serie marcada como `log` y convertida a `simple` da 0,6746 y 0,6745. No
era un hallazgo.

### Comprobado con datos reales

Paridad de riesgo sobre seis series (OMIE, MIBGAS, Brent, Henry Hub, VIX y
EUR/USD, 2 122 sesiones comunes). Las contribuciones relativas salen
`0,166667 · 0,166659 · 0,166664 · 0,166667 · 0,166663 · 0,166680`, todas a
menos de 2e−05 de 1/6. Y los pesos son los que tienen que ser: OMIE, con una
volatilidad anualizada 176 veces mayor que la del EUR/USD, se lleva el **0,43 %**
de la cartera y el EUR/USD el **76,88 %**. Con las claves del diccionario en
orden inverso, la volatilidad de esa cartera sale 0,143735 en los dos casos.

---

## Estado: spreads y estacionalidad, cerrados

Septiembre de 2026. `energia/spreads.py` y `estacionalidad.py`. De los tres
hallazgos que el revisor nombraba, uno se confirma, uno es **mucho menor** de
lo que decía y el tercero es **mucho mayor**. Y la aritmética de los spreads,
que la revisión daba por verificada, lo estaba.

### El período por posición: confirmado, y peor de lo que parecía

`descomponer` toma la posición del ciclo como `arange(n) % periodo`, o sea
**por posición y no por calendario**, así que el `periodo` tiene que ser el
número real de observaciones por ciclo. El 252 que suele escribirse es el
número convencional de sesiones de bolsa, y ningún calendario real lo tiene:
`bdate_range` da 261 días al año.

Sobre nueve años de días hábiles con una estacionalidad anual pura de amplitud
40:

| `periodo` | Amplitud estimada | Varianza mensual en el residuo |
|---:|---:|---:|
| 252 | 34,10 | 6,1 % |
| 261 | **40,01** | **0,0 %** |

Pero el caso que de verdad duele es con datos reales. La serie de temperatura
de Madrid tiene **353 observaciones por año** —días naturales con huecos—, y
pedir 252 sobre ella no desplaza la fase: la **destruye**.

| `periodo` | Amplitud estacional | Enero | Julio |
|---:|---:|---:|---:|
| 353 (el del calendario) | **10,67 °C** | **−7,56** | **+8,66** |
| 252 (el de manual) | 3,16 °C | −0,14 | +0,23 |

El ciclo anual de Madrid prácticamente desaparece: enero pasa de −7,56 °C a
−0,14 °C. Y OMIE y MIBGAS tienen 365 y 362 observaciones por año, así que
escribir 252 sobre ellas sería un desfase de **110 observaciones al año**.

Añadida `periodo_del_indice(indice, ciclo)`, que lo mide del calendario, y una
detección que avisa cuando el `periodo` pedido está dentro de un factor 2 del
anual del índice pero no coincide. Esa banda la puse primero en ±25 % y
**dejaba escapar justo el caso peor**: 252 frente a 353 se quedaba fuera. El
factor 2 lo cubre sin molestar a quien pide de verdad el ciclo semanal.

### El tamaño del efecto: el revisor se equivocó de causa

El hallazgo decía que `tamano_efecto` «con ruido puro vale 0,26 sobre cuatro
años diarios, *grande* en la escala del propio docstring», y lo atribuía a que
`H / (n − 1)` no está corregido por grados de libertad.

La corrección es real, pero es **veinticinco veces más pequeña** que eso.
Medido con ruido iid puro sobre cuatro años de días hábiles agrupando por mes,
el efecto medio sale **0,0106**, que es exactamente el `(k − 1)/(n − 1) =
0,0105` que predice la fórmula: el sesgo y nada más. Es «pequeño» en esa misma
escala. Corregido a `(H − k + 1)/(n − k)`, que es gratis.

El 0,26 que vio el revisor **venía de otro sitio**, y de algo bastante más
grave.

### Kruskal-Wallis sobre una serie de precios no vale, y no hay arreglo barato

Kruskal-Wallis, como el ANOVA, supone **observaciones independientes**. Con un
AR(1) y sin ninguna estacionalidad:

| `φ` | Rechaza al 5 % | Tamaño del efecto medio |
|---:|---:|---:|
| 0,00 | 5,5 % (correcto) | 0,0109 |
| 0,50 | 81,0 % | 0,0288 |
| 0,90 | **100,0 %** | **0,1157** |
| 0,99 | 97,5 % | 0,1234 |

Ahí está el 0,26 del revisor: no en los grados de libertad, en la
autocorrelación. El docstring dedicaba seis líneas a justificar por qué los
rangos y ni una a la independencia.

**Se intentaron tres arreglos y los tres fallan.** Queda escrito para que nadie
los repita:

1. **Permutación por desplazamiento circular.** Conserva toda la dependencia y
   corrige el tamaño en series diarias: 5,5 %, 2,5 %, 1,5 % y 5,5 % para los
   cuatro `φ`. Pero cuando la serie se muestrea al mismo período con que se
   agrupa —una serie mensual agrupada por mes— un desplazamiento es una simple
   **reetiquetación cíclica de los grupos**, y H es invariante a cómo se llamen
   los grupos: el p-valor sale próximo a 1 pase lo que pase.
2. **Surrogates de fase aleatoria.** Tamaño correcto en todos los casos (3,5 %
   a 7,0 %), pero **sin potencia** contra lo que más interesa: un patrón
   sinusoidal con el período de la agrupación da los mismos doce grupos a
   cualquier fase. Sobre serie mensual con amplitud 1,2 σ detecta el 4 %.
3. **Wald conjunto con errores HAC sobre los rangos.** Sobre-rechaza: 12,8 %
   con datos iid diarios, 19,5 % mensuales y 82,8 % con `φ = 0,9`. El
   truncamiento de retardos de la regla habitual es cortísimo para esa
   persistencia y el Wald tiene sesgo de muestra pequeña con doce grupos.

Así que **no se finge un arreglo**. El p-valor se queda como estaba y lo que se
añade es el diagnóstico que faltaba: `autocorrelacion_1` y
`observaciones_efectivas = n(1−ρ)/(1+ρ)`, con un aviso cuando esta última baja
de diez veces el número de grupos. Es la cifra que lo explica de un golpe:

| Serie real | p-valor | Efecto | ρ₁ residual | Observaciones efectivas |
|---|---:|---:|---:|---:|
| OMIE diario | 1,4e−52 | 0,0833 | +0,941 | **97** de 3 185 |
| MIBGAS diario | 9,2e−12 | 0,0168 | +0,989 | **21** de 3 877 |
| Temperatura de Madrid | 0,0 | **0,8330** | +0,795 | 351 de 3 078 |

Las dos primeras avisan: un contraste de doce grupos sobre 97 o 21
observaciones efectivas no puede decir nada del calendario, por mucho que el
p-valor sea 1e−52. La tercera **no avisa**, y es la correcta: la
estacionalidad de la temperatura es real, el efecto es 0,83 y hay 351
observaciones efectivas. El diagnóstico distingue los dos casos.

**La autocorrelación se mide sobre los residuos dentro de cada grupo**, no
sobre la serie entera, y eso importa: una serie con estacionalidad limpia y
ruido iid tiene autocorrelación total altísima —enero se parece a febrero
porque los dos son invierno— y sin embargo el contraste está perfectamente
calibrado, porque las observaciones son independientes *dado el grupo*.
Midiéndola sobre la serie entera salían 135 avisos en 127 pruebas; sobre los
residuos del grupo, cero falsos.

### Lo que estaba bien

La revisión daba por verificada la aritmética de `spreads.py` y lo está. Se
deja fijado como prueba en lugar de como nota: **error 0,000e+00** en 3 000
casos aleatorios contra una reimplementación independiente, para el spark, el
dark limpio (con el CO₂) y el crack 3-2-1, incluyendo precios de electricidad
negativos. Y la desigualdad de Jensen que promete `valor_opcionalidad` se
cumple y crece con la volatilidad: con spread medio −5, la media de
`max(spread, 0)` vale 0,425 · 5,662 · 21,848 para volatilidades de 5, 20 y 60.
Una planta con spread medio negativo vale dinero, que es el caso del ciclo
abierto de punta.

La corrección 2×M de la media móvil centrada para períodos pares también está
bien: coincide con `statsmodels.seasonal_decompose` a **3,6e−15**.

Y cuatro sospechas mías que tampoco eran hallazgos, medidas y descartadas: los
percentiles de `resumen_spread` usan la interpolación de `numpy`, pero son
descriptivos y no una medida de riesgo; `spread_locacional` da +8 en un sentido
y −12 en el otro, que es correcto porque el transporte cuesta en los dos;
`horas_en_dinero` con huecos da 2 de 3 y no 2 de 4, que también es lo correcto
—una hora desconocida no estuvo en dinero— y el recuento queda visible en
`resumen_spread`; y el caso de un solo año en `indice_estacional` ya levantaba
el error adecuado.

Los módulos de prueba pasan de 67 y 60 casos a 73 y 62: **5 de los nuevos
fallan contra el código viejo** y los 2 de spreads pasan en las dos versiones a
propósito, porque fijan lo que estaba bien.

---

## Estado: evaluación, cerrada. Y con esto, los 107

Septiembre de 2026. `ml/evaluacion.py`. `comun.py` ya se cerró con la capa de
datos. Este es el último de los siete bloques.

### El MASE dependía de las unidades, y la causa era peor de lo que parecía

Confirmado exacto. La misma serie de 1 000 rendimientos de volatilidad 1,5 %,
el mismo modelo —uno que predice cero— y dos respuestas:

| Unidades | `quantile(\|x\|, 0,99)` | Se clasificaba como | MASE |
|---|---:|---|---:|
| tanto por uno | 0,0347 | rendimiento | **1,0000** |
| porcentaje | 3,4718 | precio | **0,7046** |

La causa es `_es_serie_de_rendimientos`, que decidía con un umbral **absoluto**:
rendimientos si la serie está centrada en cero y `quantile(|x|, 0,99) < 1`. En
porcentaje el cuantil pasa de 1, la serie se clasifica como precio, la
referencia ingenua deja de ser «cero» y pasa a ser «el último valor» —una
referencia pésima para una serie sin memoria— y cualquier modelo la bate. La
métrica que el docstring vende como adimensional cambiaba con las unidades.

**Y hay algo que el revisor no vio: la regla fallaba en el mercado para el que
está escrita la librería.** Los rendimientos diarios de OMIE tienen
`quantile(|x|, 0,99) = 2,55`, porque el pool pasa de 1 a 100 EUR/MWh con
normalidad. En tanto por uno y todo, **se clasificaban como precio**.

La regla nueva no mira la escala, mira la **persistencia**: el cociente entre
la desviación típica de la serie y la de su primera diferencia, que es
invariante a la escala porque la multiplica por la misma constante arriba y
abajo. Para un AR(1) de parámetro `φ` vale `1/√(2(1−φ))`.

| Serie | Cociente | ρ(1) |
|---|---:|---:|
| Ruido blanco (teórico) | 0,707 | 0 |
| OMIE rendimientos | 0,710 | +0,008 |
| **OMIE rendimientos en %** | **0,710** | +0,008 |
| Spark spread OMIE-MIBGAS | 2,140 | +0,891 |
| OMIE precio EUR/MWh | 2,978 | +0,944 |
| VIX nivel | 4,647 | +0,977 |
| EUR/USD precio | 23,180 | +0,999 |

Los dos grupos quedan separados y el cociente **no se mueve al cambiar de
unidades**, que era todo el problema. Entre 1,2 y 2,0 la función **no adivina**:
levanta `ErrorDatos` pidiendo la marca, porque ahí caen las series de verdad
ambiguas —un rendimiento con autocorrelación de 0,8 da 1,58— y elegir mal la
referencia cambia la conclusión entera.

### Una prueba más que pasaba por el motivo equivocado

`test_un_spread_centrado_en_cero_se_trata_como_nivel_y_no_como_rendimiento`
construía un «spark spread» con **ruido blanco** de escala 8 y exigía que se
tratase como nivel. La premisa es correcta —un spread es un nivel— pero los
datos no la cumplían: el ruido blanco tiene cociente 0,710 y para una serie sin
memoria la mejor predicción ingenua es la media, no el valor de ayer. La regla
vieja acertaba por la magnitud, no por la dinámica.

Un spark spread **de verdad** sí es un nivel, y lo es por persistencia: medido
sobre OMIE contra MIBGAS, 3 153 días, cociente 2,140 y ρ(1) = +0,891. La prueba
se reescribió con esa dinámica, y se añadió la mitad que faltaba: que el ruido
blanco medido en euros **no** es un nivel. Van once pruebas en este proyecto
que consagraban un fallo.

### Diebold-Mariano: dos fallos que iban juntos

**El divisor.** `np.mean` sobre el producto desplazado divide entre `n − k` en
vez de entre `n`, inflando cada autocovarianza por el factor exacto `n/(n−k)`.
El efecto típico es pequeño, pero la cola es larga y **cambia el veredicto**:

| `n` | `h` | Mediana del cociente | Percentil 90 | Máximo | Veredicto al 5 % cambia en |
|---:|---:|---:|---:|---:|---:|
| 60 | 22 | 1,027 | 1,569 | 21,85 | **3,7 %** de los casos |
| 120 | 22 | 1,016 | 1,246 | 19,59 | 2,7 % |
| 250 | 22 | 1,005 | 1,063 | 2,27 | 0,9 % |

(3 000 réplicas cada uno. La primera medición que hice daba «el estadístico se
multiplica por 2,28 y el p-valor pasa de 0,0452 a 0,3722»: era **un solo
sorteo**, y presentarlo como la magnitud típica habría sido el mismo error que
critiqué a los subagentes. La mediana es 1,027.)

**La varianza negativa.** Sumar las autocovarianzas sin ponderar no garantiza
una varianza positiva. Con `n = 60` y horizonte 22 salía **negativa el 13,5 %
de las veces**, y en esos casos la función devolvía estadístico 0 y p-valor 1:
se callaba sin avisar de que se callaba. Con pesos de Bartlett (Newey-West) es
no negativa por construcción y eso baja al **0,0 %**. `nucleo="rectangular"`
recupera el estimador truncado del artículo de 1995, ya con el divisor bien.

**El orden.** Las autocovarianzas se calculan **por posición** y nadie
comprobaba el índice. Sobre 200 errores solapados a horizonte 10, barajar
llevaba el estadístico de +1,5104 a +3,3244 y el p-valor de **0,1325** —no
rechaza— a **0,0011**: una ventaja predictiva inventada por el orden de las
filas. Es el mismo fallo que apareció en `serie_var_movil` y en Christoffersen,
y van tres. Ahora se exige índice monótono cuando el horizonte pasa de 1; con
horizonte 1 no hay autocovarianzas y el orden da igual.

### Lo que sigue sin estar bien, y queda escrito

**Con errores solapados el contraste rechaza más de lo que dice, y eso no lo
arregla ninguno de los tres cambios.** Medido con 2 000 réplicas de dos
predicciones de precisión idéntica:

| `n` | `h` | Rectangular | Bartlett |
|---:|---:|---:|---:|
| 60 | 1 | 5,4 % | 5,4 % |
| 60 | 10 | 10,4 % | 8,3 % |
| 60 | 22 | 12,1 % | 8,2 % |
| 120 | 22 | 10,9 % | 7,3 % |
| 250 | 22 | 8,2 % | 8,6 % |
| 500 | 22 | 6,7 % | 9,2 % |

Con horizonte 1 el tamaño es correcto. Con horizontes largos se va al 7-12 %
frente al 5 % nominal, en las dos variantes, y la corrección de
Harvey-Leybourne-Newbold —que está bien transcrita, eso lo verificó el
revisor— no llega a compensarlo. Así que un p-valor de 0,04 con horizonte 22
**no** es un contraste al 5 %, y el docstring lo dice ahora con la tabla
delante.

### El acierto direccional descartaba fechas en silencio

`_acierto_direccional` excluye las fechas en que el valor real o el previsto
son exactamente cero. Es correcto —ahí no hay dirección que acertar— pero
**cuántas** no se veía en ninguna parte. Sobre las diferencias del precio
horario de OMIE son **9 169 de 102 142, el 8,98 %**: el acierto se calculaba
sobre el 91 % de la muestra. Añadida la columna `fechas_sin_direccion` a
`tabla_evaluacion`.

### Comprobado con datos reales

`tabla_evaluacion` sobre los 3 162 días de rendimientos diarios de OMIE, con un
AR(1) ajustado en la primera mitad, una media móvil de 21 sesiones y la media
muestral:

| Modelo | MAE | MASE | Acierto dir. | Fechas sin dirección | p DM |
|---|---:|---:|---:|---:|---:|
| ingenua | 0,2825 | 1,0000 | NaN | **3 162** | 1,0000 |
| ar1 | 0,2825 | 1,0001 | 0,4984 | 0 | 0,3276 |
| media_muestral | 0,2894 | 1,0247 | 0,4820 | 0 | 0,0001 |
| media_movil_21 | 0,3298 | 1,1675 | 0,4687 | 0 | 0,0013 |

**Ninguno bate a la referencia.** El AR(1) empata porque el `φ` ajustado en la
primera mitad es −0,0016, o sea nada; los otros dos son significativamente
peores. Es el resultado honesto: los rendimientos diarios del pool no se
predicen, y la tabla lo dice en lugar de fabricar un ganador. La fila de la
ingenua muestra las 3 162 fechas sin dirección y su acierto como NaN, que es
exactamente lo que hay que decir de quien no apuesta nunca.

Y el MASE del AR(1) sale **1,000139 en tanto por uno y 1,000139 en
porcentaje**, que era el hallazgo.

### Lo que esto cambia en el cuaderno 03

El fallo no era teórico: el cuaderno 03 publicaba MASE de **0,656, 0,681 y
0,650** para sus tres configuraciones, o sea «los modelos baten a la referencia
por un 32-35 %». Esos números salían de una referencia mal elegida, porque los
rendimientos del pool caían del lado «precio» de aquel umbral de escala y la
ingenua pasaba a ser «lo mismo que ayer» en vez de «cero». Batir una referencia
mala era gratis.

Con la regla corregida son **0,951, 0,986 y 0,941**: baten por un 1-6 %. Y la
tabla de sensibilidad al factor de escala pasa de 0,673 / 0,649 / 0,810 / 3,504
a 0,976 / 0,941 / 1,174 / 5,079. **No cambia ninguna conclusión del cuaderno
—su tesis era justamente que ahí no hay señal cobrable— pero la refuerza**, y
queda anotado en el propio cuaderno con el antes y el después. El cambio a
`inverted_cdf` en el VaR histórico también se nota ahí: la fila `historico` del
backtest pasa de 55 a 52 excepciones.

---

## Balance de los 107

Los siete bloques están cerrados. Lo que queda escrito de todo el ejercicio:

**Las fórmulas estaban bien.** Lo dijo la propia revisión al terminar y se
confirma bloque a bloque: la log-verosimilitud de la cópula t coincide al bit
con `scipy`, la descomposición de Euler suma exactamente la volatilidad, la
aritmética de los spreads da error 0,000e+00 en 3 000 casos, la corrección
2×M coincide con `statsmodels` a 3,6e−15, la corrección de
Harvey-Leybourne-Newbold está transcrita carácter a carácter. **Lo que falla
son los bordes, las convenciones y lo que ninguna prueba tocaba**: la primera
observación, el orden de un diccionario, el orden de un índice, una serie
corta, las unidades, un umbral absoluto, una cartera neta corta, un día que el
mercado cerró a cero.

**Tres fallos aparecieron tres veces cada uno**, en módulos distintos y
escritos en momentos distintos, lo que dice que eran patrones de pensamiento y
no despistes:

1. **Cruzar dos objetos etiquetados por posición.** En las cópulas (+788 o
   −3 654 de verosimilitud según el orden de las columnas), en
   `volatilidad_cartera` (factor 2,85 según el orden de las claves de un
   diccionario) y en `_preparar` con etiquetas numéricas.
2. **Calcular algo posicional sobre un índice que nadie comprueba que esté
   ordenado.** En `serie_var_movil` (947 de 950 ventanas con futuro dentro),
   en Christoffersen (LR de 76,38 a 0,37) y en Diebold-Mariano (p-valor de
   0,1325 a 0,0011).
3. **Integrar una cola por el extremo derecho de cada tramo**, siempre corto y
   siempre hacia el lado inseguro: en `es_garch` y en `es_cornish_fisher`.

**Once pruebas consagraban un fallo**, y la lección es la del spread: una
prueba puede tener la premisa correcta y los datos equivocados. El «spark
spread» era ruido blanco, la referencia del bootstrap se construía con
`mode="wrap"`, el MASE se comprobaba pasando la referencia a mano —la única
rama donde la invarianza se cumplía sola—, el aviso de Cornish-Fisher pasaba
con una semilla de cada dos.

**Donde el revisor se equivocó**, cinco veces, y merece constar con la misma
tinta: las correlaciones del simulador (era su prior, no un dato), el tamaño
del efecto de Kruskal-Wallis (0,0106 y no 0,26; la causa era la
autocorrelación), `es_historico` (no arrastraba el problema del cuantil),
`var_incremental` (no infla el VaR al quitar un activo pequeño) y la prueba de
Cornish-Fisher (no era una prueba floja, era el código: rechazaba el 57 % de
las muestras normales).

**Donde me equivoqué yo**, y también consta: una frontera «no monótona» que
era ruido de coma flotante a 1,4e−16; una sospecha de mezcla de unidades en
`ratio_sharpe` que no existía (0,6746 frente a 0,6745); un primer intento de
reproducir la paridad de riesgo que no encontró nada porque usé matrices
demasiado fáciles; una banda de aviso del ±25 % que dejaba escapar justo el
caso peor; un arreglo de Kruskal-Wallis por desplazamiento circular que era
inválido por aliasing con la agrupación; y una magnitud del 2,28× en
Diebold-Mariano medida sobre **un solo sorteo** cuando la mediana es 1,027.

**Y tres cosas que no se arreglaron**, porque no tienen arreglo barato y
fingirlo habría sido peor que decirlo:

- El p-valor de Kruskal-Wallis sobre una serie con dependencia serial. Tres
  enfoques probados, tres modos de fallo distintos, documentados. Se añadió el
  diagnóstico —`observaciones_efectivas`— en lugar de un arreglo falso.
- El tamaño de Diebold-Mariano con errores solapados: 7-12 % frente al 5 %
  nominal a horizontes largos, en las dos variantes del núcleo.
- El agrupamiento de volatilidad del simulador, que sigue entre 0,07 y 0,13
  cuando los datos dan 0,25-0,44.

La suite pasa de las 944 pruebas con que empezó la revisión a **1 113**, y a
**1 183** con el apéndice de datos que cierra este documento.

---

# Apéndice: los datos que faltaban

La revisión y el optimizador dejaron tres huecos señalados explícitamente como
«no están en los datos de este proyecto»: el ingreso de capacidad, el de
servicios de ajuste y el impacto del propio activo en el precio. Este apéndice
los cierra —dos de ellos con datos, uno con la constatación de que el dato no
existe.

## 1. Mercado de capacidad: el dato no existe

No es que esté cerrado o de pago. **El mercado de capacidad español se creó el
15 de septiembre de 2026**, dos semanas antes de esta descarga, así que no hay
serie histórica de precios que buscar. Cualquier cifra sería inventada, y por
eso `ingreso_adicional_anual_por_mw` sigue valiendo cero por omisión y siendo
un parámetro obligatorio de rellenar a mano.

## 2. Servicios de ajuste: 34,28 €/MW y hora

Transcrito del Informe del Sistema Eléctrico de REE a
`datos/reales/ree_servicios_ajuste.csv`. ESIOS exige un token que se pide por
correo y `apidatos.ree.es` está tras protección anti-bot, así que no hubo
descarga automática.

La unidad no venía sin ambigüedad en el informe, y una unidad mal interpretada
aquí cambia el VAN por un factor de 8 760. Se resolvió aritméticamente: con
una banda media asignada de 922 MW, 34,28 × 922 × 8 760 = **277 M€**, el
10,4 % de los 2 668 M€ de servicios de ajuste de 2024. Es coherente con que la
banda secundaria sea una partida mediana del total, lo que confirma €/MW **y
por hora**. Si fuera por año daría 32 000 €, el 0,001 % del total, absurdo; si
fuera €/MWh no cuadraría con que la banda se retribuye por disponibilidad.

## 3. Impacto en el precio: el hallazgo que cambia la conclusión

Este era el hueco de verdad, porque afectaba a la validez de todo el módulo de
almacenamiento: el despacho se resuelve como precio aceptante. Se descargaron
21 sesiones de curvas agregadas de OMIE (144 MB, no versionadas) y se midió.

**El lector tuvo tres errores antes de funcionar**, los tres del tipo que da
números plausibles y falsos:

1. `skiprows=2` se comía la primera fila de datos en vez del preámbulo.
2. Se asumieron dos formatos cuando hay **tres**: `Hora`+`Energía` hasta
   2025-04-17, `Período`+`Potencia` con enteros hasta 2025-09-30, y
   `Período`+`Potencia` con etiquetas `HxQy` desde el paso a cuartos de hora.
3. El peor: en las variantes 2 y 3 la cantidad es **potencia, no energía**. Con
   períodos horarios coincide numéricamente, así que el error habría pasado
   inadvertido en cinco meses de datos y habría multiplicado por cuatro el
   volumen del tercer formato.

**La validación.** El precio publicado sale exacto de la pila en el 82,1 % de
los 864 períodos. Ese 17,9 % restante se investigó en vez de reportarlo como
hueco, y se descompone: el 15,5 % cae **dentro del cruce** con error mediano de
0,24 €/MWh —redondeo del fichero y casación parcial de la unidad marginal, no
un fallo de lectura— y solo el **2,4 % queda fuera**, concentrado en los días
con separación del mercado ibérico (acierto del 53,5 % en los períodos
marcados `ES,PT` frente al 83,6 % de los `MI`). Nada de ello afecta al
impacto, porque el impacto es una diferencia de dos puntos de la misma curva y
un sesgo de nivel se cancela.

**El resultado.** Añadir 500 MW de demanda sube el precio 1,93 €/MWh en la
mediana; añadir 500 MW de oferta lo baja 10,57. Un factor de **5,5**, por la
mayor inelasticidad de la demanda. Y ordenado por nivel de precio va en contra
de la batería: en las horas de más de 120 €/MWh —donde descarga— vender 500 MW
hunde el precio 20,01, mientras que en las de menos de 5 —donde carga—
comprarlos lo sube 0,48.

**Lo que eso cambia.** El VAN dejaba el capex de equilibrio en 80,8 €/kWh
frente a un mercado de 200-400, y se concluyó que el arbitraje se queda a un
factor 3-5. Con el impacto dentro la conclusión es más fuerte y distinta:

| Potencia | Merma | Capex de equilibrio |
|---:|---:|---:|
| 1 MW | 0,1 % | 81,2 €/kWh |
| 100 MW | 10,3 % | 71,2 €/kWh |
| 500 MW | 41,5 % | 40,5 €/kWh |

**No hay escala a la que el arbitraje solo funcione.** Crecer empeora el
negocio en vez de mejorarlo, que es lo contrario de la intuición de economías
de escala, y explica por qué el almacenamiento se financia con pagos por MW
disponible —capacidad, banda de regulación— y no con el *spread* horario: esos
se cobran por potencia y no se autodestruyen al crecer.

## 4. Un fallo latente encontrado de camino

`ParametrosAlmacenamiento.horas_por_periodo` es un escalar y
`datos/reales/omie_intradiario.csv` **no lo es**: es horario hasta el
2025-09-30 y cuarto-horario desde el 2025-10-01. Despachar el fichero entero
con los parámetros por omisión multiplicaba por cuatro la energía del último
tercio de la muestra —33,6 % de los períodos— sin que nada lo delatara: el
plan sigue siendo factible y el margen sale simplemente demasiado alto.

Es exactamente la clase de error que esta revisión busca, y estaba en el
camino por defecto de los datos propios del proyecto. `despacho_optimo` ahora
avisa con `RuntimeWarning` cuando la cadencia del índice no cuadra con la
declarada, distinguiendo el caso legítimo del cambio de hora —dos saltos raros
al año— de una cadencia realmente mixta. Por eso todas las cifras de este
apéndice se miden sobre los 67 871 períodos horarios homogéneos y no sobre los
102 143 del fichero completo.

## 5. Lo que los datos nuevos cambian en la conclusión del proyecto

Con las dos piezas juntas —el impacto medido y el precio de la banda— la
valoración de la batería de 100 MW queda así:

| Escenario | Capex de equilibrio | ¿Viable a 200-400 €/kWh? |
|---|---:|---|
| Arbitraje, precio aceptante | 81,2 €/kWh | No, por factor 3-5 |
| Arbitraje, con impacto propio | 71,2 €/kWh | No, peor aún |
| + banda el 10 % de las horas | 135,4 €/kWh | Todavía no |
| + banda el 25 % de las horas | **231,8 €/kWh** | **Sí, en el rango bajo** |
| + banda el 50 % de las horas | 392,5 €/kWh | Sí, y con VAN positivo |

Es la primera vez en todo el proyecto que una configuración sale viable, y el
mecanismo que la hace viable no es el que se optimiza: **el arbitraje paga la
operación, la banda paga la inversión**. La razón estructural es que la banda
se retribuye por MW disponible y el arbitraje por MWh movido, así que solo el
segundo se autodestruye al crecer.

Eso tiene una consecuencia para el resto del proyecto que conviene dejar
escrita: `valor_de_la_prevision` mide lo que una previsión captura **del
arbitraje**, que es la parte pequeña y la que no escala. Una previsión buena
sigue valiendo los 72 156 € de VAN que calcula el módulo, pero no es lo que
decide si la batería se construye. Quien quiera que este proyecto informe una
inversión real tiene que modelar la banda, y para eso hacen falta datos de
asignación de banda por unidad que ESIOS publica y este proyecto no tiene.
