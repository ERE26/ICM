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
