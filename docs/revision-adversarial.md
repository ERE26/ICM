# Revisión adversarial de `icm.energia` e `icm.ml`

Septiembre de 2026.

Seis revisores independientes, uno por módulo de riesgo alto, con la instrucción
explícita de **encontrar fallos y no aprobar el código**, y de no dar por válido
ningún hallazgo sin un script que lo reproduzca numéricamente. Después, un agente
de síntesis descartó duplicados y falsos positivos y ordenó el resto por daño real.

**30 hallazgos, 2 fusionados, 1 descartado: 28 defectos reales.** Ninguno lo
cubrían las pruebas existentes, que pasaban todas.

Lo que sigue es el informe tal como salió, sin recortar.

---

## Estado: cerrado

Los 28 defectos están arreglados, cada uno con la prueba que lo reproducía
convertida en prueba de regresión. La suite pasa: **944 pruebas, 19 omitidas**.

Dos apuntes sobre el cierre, porque el informe original los dejaba abiertos:

- **D3 no era un defecto.** `comprobar_ausencia_de_fuga` solo puede auditar lo
  que el constructor hace con los datos que recibe; un `interpolate()` aplicado
  *antes* de la llamada queda fuera de su alcance por construcción. Lo que sí
  faltaba era decirlo, y ahora lo dice el apartado «Alcance» de su docstring.

- **La sospecha del embargo sí se pudo medir.** El último hallazgo del informe
  quedó como «solo sospecha» porque con h=1 no se veía nada. No se veía porque
  con h=1 *no hay solape*: el objetivo de la última fila de entrenamiento se
  mide justo en el primer día de prueba, sin invadirlo. El efecto aparece al
  subir el horizonte y mirar donde vive la fuga, que son las primeras h−1 filas
  de cada tramo de prueba, no el tramo entero:

  | horizonte | ECM sin embargo | ECM con embargo | diferencia | p |
  |---|---|---|---|---|
  | h=1  | — | — | sin solape posible | — |
  | h=21 | 9.43e-03 | 9.52e-03 | +0.9 % | 0.48 |
  | h=63 | 7.54e-03 | 1.13e-02 | **+33.15 %** | 6.19e-06 |

  Sobre 250 semillas, con h=63 el embargo=0 «gana» en 144 de ellas: el error
  medido es un 33 % mejor de lo que el modelo merece, y esa mejora es futuro
  colado en el entrenamiento. Por eso `comparar_modelos` y
  `buscar_hiperparametros` ya no heredan `embargo=0`, sino que lo derivan del
  horizonte que `objetivo_rendimiento_futuro` deja marcado en `attrs`.

Además de los 28, se cerró un hueco que la revisión no llegó a listar: el
arreglo del hallazgo 15 (rechazar un panel con `MultiIndex`) se había aplicado
en `validacion.py` pero no en `redes.py`, donde el daño es mayor porque la
ventana se construye por posición y cada muestra mezclaba activos distintos.

---

## Síntesis y orden de arreglo

Los 30 hallazgos verificados contra el código y las reproducciones vueltas a
ejecutar. Este es el triaje.

---

# TRIAGE DE LOS 30 HALLAZGOS — 2 fusionados, 1 medio descartado, 28 a arreglar

## A. Descartados por ser EL MISMO fallo contado dos veces

**D1. Fusiono "Un solo periodo con 1+pnl <= 0 hace que coste_de_equilibrio devuelva 0.0 bps" dentro de "El Sharpe publicado no es invariante a la escala".**
Una sola causa: `src/icm/ml/estrategia.py:691` y `:781` reutilizan `icm.cartera.metricas.ratio_sharpe`, cuyo numerador es la CAGR geometrica (`metricas.py:44-49`), sobre un `pnl` que este modulo define como aditivo (`curva_beneficio`: "se acumula por suma y no por composicion, a proposito"). El caso `1+pnl <= 0` es solo el extremo de esa misma incoherencia: `rentabilidad_anualizada` trunca a `-1.0` y arrastra a `coste_de_equilibrio`. El propio hallazgo lo admite ("Con el Sharpe aritmetico del hallazgo anterior el problema desaparece"). Un cambio, dos sintomas.

**D2. Fusiono "rendimiento_total no es el rendimiento de la posicion" dentro de "los rendimientos pasados cambian cuando llega dato futuro".**
Una sola causa: con `metodo='diferencia'`, `descomponer_rendimiento` divide por el nivel AJUSTADO (`curvas.py:649`, `anterior = precio.shift(1)`) y el ajuste aditivo no se cancela en el cociente, mientras el multiplicativo si. De ahi salen las dos cosas a la vez: el nivel pasado depende de cuanto futuro haya (`_ajustes_hacia_atras` acumula todos los escalones posteriores) y la magnitud del rendimiento queda mal. El propio hallazgo dice "Mismo arreglo que el hallazgo de fuga de futuro".

## B. Descartado por ser comportamiento correcto (comprobado en el codigo)

**D3. Descarto la MITAD del hallazgo "Las fugas condicionadas a los datos pasan la auditoria": el tramo de la interpolacion PREVIA a la llamada.**
`comprobar_ausencia_de_fuga` (`caracteristicas.py:733-762`) recibe `constructor` y `precios`, y construye sus dos experimentos aplicando truncado y perturbacion **a la entrada que le dan**. No hay ningun mecanismo por el que pueda auditar una transformacion ya aplicada antes de la llamada, ni lo promete: su docstring dice "Audita que ninguna variable de `X` use informacion posterior a su fecha", y `X` la produce el constructor. Que `comprobar_ausencia_de_fuga(construir_matriz, precios.interpolate(), 'ALFA')` pase es correcto: la fuga esta en `precios`, no en `construir_matriz`. Es una limitacion de alcance, no un defecto; a lo sumo falta la advertencia en el docstring (queda recogida en el punto 23).
**La otra mitad SI sobrevive** (punto 23): con los tres cortes por omision de `caracteristicas.py:598`, un `bfill` **dentro** del constructor no se detecta, y el mensaje de error de `:684-685` anuncia "un relleno hacia atras" como uno de los fallos que busca.

No descarto ningun otro. Revise en el codigo los cuatro candidatos mas debiles y los tres son defectos reales:
- **`importancia_variables` dentro de muestra**: el docstring `modelos.py:580-588` SI documenta el ajuste automatico. Pero no hay aviso ni marca en la `Series` devuelta, y `notebooks/03_trading_energia_ml.ipynb` celda 21 usa esa via y titula el grafico "Qué mira el modelo". Sobrevive (con el arreglo ajustado: avisar y marcar, no "el docstring miente").
- **`_clonar` con deepcopy**: `validacion.py:111-128` promete "Copia sin ajustar del modelo" y la via de respaldo no la cumple. Verificado que `sklearn.base.clone` lanza `TypeError` con un objeto sin `get_params`, asi que la rama es alcanzable; y verificado que `RedSecuencial` SI tiene `get_params`, asi que nada de lo que trae el repositorio dispara el fallo. Promesa falsa real, probabilidad menor que la que le da el revisor.
- **`MediaCero` sin `n_features_in_`**: comprobado que **nada** en `src/`, `tests/` ni `notebooks/` lee `n_features_in_` ni `feature_names_in_`. El requisito es hipotetico. Sobrevive solo como cosmetico, al final.
- **MultiIndex en `_indice_temporal`**: el modulo no ofrece paneles, pero el control de `validacion.py:65` existe con la razon escrita ("una misma fecha puede acabar en entrenamiento y en prueba a la vez") y un MultiIndex la burla. Sobrevive; el arreglo es rechazar, no dar soporte.

---

# C. ORDEN DE ARREGLO

## BLOQUE 1 — Resultado equivocado sin avisar

### 1. El docstring de `estrategia.py` ordena exactamente la trampa que dedica 40 lineas a prohibir
`src/icm/ml/estrategia.py:20-33` (y su repeticion en `:519-527`)
Cambio: **borrar la receta `.shift(-1)`**. `ResultadoValidacion.predicciones` se indexa en `validacion.py:508` con `indice[pos_pru]`, o sea con las filas de X/y, y la `y` canonica de `construir_matriz` -> `objetivo_rendimiento_futuro` esta indexada en la fecha de decision (`caracteristicas.py:373-411`). Ya viene en la convencion de este modulo: se pasa tal cual (contra `md.y`, retardo 0 deducido, o contra rendimientos contemporaneos con `retardo=1`) y **nunca** con `.shift(-1)`. Reescribir las lineas 20-33 para decirlo, y limitar el aviso sobre `.shift(-1)` a series construidas sobre una `y` contemporanea. Nota: `icm.ml.evaluacion` no tiene convencion temporal propia — solo compara termino a termino lo que le den —, asi que la frase "no es la convencion de `icm.ml.evaluacion`" tambien hay que quitarla.
Confirmacion independiente de que el equivocado es el docstring: `tests/test_ml_convencion.py:131-166` (`test_las_dos_formas_correctas_de_emparejar_dan_la_misma_economia`) consagra como correctas las dos vias SIN `.shift(-1)`, y el notebook (celda 26 + `:1198`) tambien.
Prueba a anadir en `tests/test_ml_convencion.py`: `construir_matriz` -> `validacion_walk_forward` -> `ejecutar_estrategia`, comprobando que el pnl con `.shift(-1)` coincide con el de `retardo=0` (es decir, que la receta esta prohibida).

### 2. `attrs['convencion']` sobrevive a `.shift()`, asi que una serie ya realineada desactiva el unico aviso de fuga
`src/icm/ml/estrategia.py:582`
Cambio: `if retardo == 0 and not adelantada:` -> avisar **siempre** que el retardo resuelto sea 0, con dos textos: el actual para `not adelantada`, y para la via `adelantada` uno nuevo ("se ejecuta con retardo=0 porque la serie viene marcada como adelantada; compruebe que sigue estandolo"). Adaptar `tests/test_ml_convencion.py:108-117`, que hoy consagra la supresion.
Ademas, `estrategia.py:568-575` rechaza `retardo=1` sobre esa serie y el mensaje empuja a `retardo=0`, es decir, a la fuga.
Refuerzo recomendado: que `caracteristicas.py:408-410` guarde en `attrs` una huella del alineamiento (`attrs['fecha_inicial']` o un hash del indice) y que `estrategia.py:561` desconfie de la marca cuando la huella no cuadre. Verificado en pandas 3.0.6 que `attrs` sobrevive a shift, dropna, reindex, rename, astype, copy y aritmetica.

### 3. Con `attrs['horizonte'] > 1` el retardo deducido es 0 y el pnl se solapa: cada movimiento se cobra hasta h veces
`src/icm/ml/estrategia.py:562`
Cambio: `horizonte_serie` se lee y solo se usa para redactar el mensaje de `:570-574`. Anadir, junto a `:564-565`, el rechazo del caso por omision: `if adelantada and horizonte_serie != 1: raise ErrorDatos(...)` explicando que con horizonte h hay que quedarse con una decision cada h periodos (`serie.iloc[::h]`, reponiendo `attrs`) y anualizar con `periodos_anio/h`. Como minimo, `warnings.warn`.
Medido: comprar y mantener sobre un rendimiento log total de -0.0851 da suma de pnl -0.0851 con h=1, -0.6106 con h=5 (7.17x) y -3.2360 con h=21 (38.01x), con autocorrelacion del pnl de +0.950 y volatilidad anual inflada de 0.158 a 0.705.
El test existente `test_un_horizonte_mayor_tambien_se_rechaza` solo cubre el rechazo de `retardo=1`; el camino por omision no se prueba.

> Los puntos 1-3 tocan el mismo bloque (`estrategia.py:555-590`): arreglarlos en una sola pasada.

### 4. La auditoria de fuga no toca ningun argumento de datos que no se llame literalmente `exogenas`
`src/icm/ml/caracteristicas.py:741`
Cambio: `exogenas = kwargs.get("exogenas")` -> recorrer **todos** los `kwargs` y aplicar el mismo tratamiento a cada valor que sea `Series`/`DataFrame` con indice comparable con el de `precios`: truncar con `.loc[idx <= corte]` en el experimento 1 (`:747`) y `_perturbar_futuro` en el 2 (`:759-761`). Red de seguridad: si queda algun kwarg de datos sin tratar, `ErrorDatos` diciendo que la auditoria no puede garantizar nada sobre el, en lugar del visto bueno silencioso.
Medido: un constructor con `X['demanda_manana'] = demanda.shift(-1)` y kwarg `demanda=DEM` **pasa**; el mismo constructor con el kwarg renombrado a `exogenas` se detecta al instante. La unica diferencia es el nombre del parametro. El docstring (`:700-701`) invita expresamente a auditar "cualquier constructor que escriba el usuario, con la firma `(precios, objetivo, **kwargs)`", y el notebook presenta la funcion como "la garantía" (celda 13).

### 5. `_comparar` descarta en silencio las columnas que no estan en las dos matrices, y la perturbacion cambia el tipo de `exogenas`
`src/icm/ml/caracteristicas.py:650` y `:759-761`
Dos cambios que hay que hacer juntos:
- `:759-761`: perturbar la exogena **conservando su tipo original**. Hoy pasa por `_como_marco` antes de `_perturbar_futuro`, asi que el constructor recibe un `DataFrame` donde en la referencia recibio una `Series`, mientras el truncado de `:747` conserva la `Series`. Si la entrada era `Series`, devolver `_perturbar_futuro(frame, ...).iloc[:, 0]` con su nombre.
- `:650`: `columnas = [c for c in referencia.X.columns if c in alterada.X.columns]` solo protesta si la interseccion queda **vacia**. Comparar los conjuntos de columnas y lanzar `ErrorDatos` (o al menos enumerar las no auditadas) cuando no coincidan: hoy ese desajuste es indistinguible de "no hay fuga".
Medido: con `retardos(exogenas.shift(-1), (0,), prefijo='exo')`, la referencia da `exo_ret0` y la perturbada `exo_temperatura_ret0` (por `_nombre_base`, `:158-167`), comunes queda en `['rend_ret1']` y la auditoria **pasa**; con la exogena como `DataFrame` la misma fuga se detecta. No hay ninguna prueba que verifique que la auditoria DETECTA fugas en exogenas (`tests/test_ml_caracteristicas.py:431` solo comprueba que las acepta).

### 6. `acierto_direccional` cuenta como fallo toda fecha sin prediccion, y es la unica metrica que no sale NaN
`src/icm/ml/validacion.py:159`
Cambio: `utiles = reales != 0.0` -> `utiles = (reales != 0.0) & np.isfinite(reales) & np.isfinite(predicciones)`, devolver NaN si no queda ninguna, y anadir `n_validas` (o `n_ausentes`) al diccionario de `_metricas` y al resumen para que el hueco quede a la vista.
Medido: modelo que imita a `RedSecuencial` (NaN en las 9 primeras filas de cada tramo), 5 particiones de 50: `rmse`/`mae`/`sesgo`/`correlacion`/`r2` = NaN en las 5, mientras `acierto_direccional` informa 0.8080 cuando el real sobre las 205 filas con prediccion es 0.9854. El sesgo es exactamente la fraccion de NaN (-18.0%). Y no es hipotetico: es la combinacion que el propio `RedSecuencial` documenta en `redes.py:459-464`.

### 7. La prediccion etiquetada que devuelve el modelo se desetiqueta y se pega por posicion
`src/icm/ml/validacion.py:498`
Cambio: antes de `np.asarray(crudas, dtype=float).reshape(-1)`, si `crudas` llega con indice (`pd.Series`/`pd.DataFrame`), hacer `crudas = crudas.reindex(indice[pos_pru])` y lanzar `ErrorDatos` si aparece algun ausente. El unico control actual (`:500`, `crudas.size != pos_pru.size`) es de tamano y una reordenacion lo pasa sin ruido.
Medido: el mismo modelo con `.sort_values()` en la salida informa correlacion -0.088 en vez de +0.957, acierto 0.488 en vez de 0.904 y Sharpe -1.21 en vez de +18.02; el indice de salida y el conjunto de valores son identicos, asi que nada permite distinguirlos a posteriori. Extra medido: una prediccion de forma `(n/2, 2)` se acepta y se aplana en silencio, porque `reshape(-1)` hace cuadrar el tamano — el mismo `reindex` lo resolveria.

### 8. `importancia_variables` permuta DENTRO de muestra cuando el modelo llega sin ajustar
`src/icm/ml/modelos.py:620`
Cambio: invertir el valor por omision — exigir modelo ya ajustado y lanzar `ErrorDatos` si llega sin ajustar, obligando a un `dentro_de_muestra=True` explicito. Si se mantiene el ajuste automatico, marcar el resultado (`importancias.attrs['dentro_de_muestra'] = True`) y emitir `UserWarning`; el docstring de `:580-588` ya lo explica, pero la `Series` devuelta no lleva nada.
Y arreglar `notebooks/03_trading_energia_ml.ipynb` celda 21, que hace `importancia_variables(crear_modelo('arbol_potenciado'), md_datos.X, md_datos.y, n_repeticiones=5)` y titula el grafico "Qué mira el modelo".
Medido con `y` = ruido independiente: la via del notebook da `{'x4': 0.2996, 'x1': 0.2754, ...}` (afirma ~0.28 de R cuadrado por variable); la via honesta (ajustado 0:300, permutado 300:400) da ruido alrededor de cero, maximo +0.0374.

### 9. El guardian de alineacion solo actua si `y` es una `Series`
`src/icm/ml/modelos.py:164` y `src/icm/ml/validacion.py:429` (condicion identica en los dos)
Cambio: `isinstance(y, pd.Series)` -> `isinstance(y, (pd.Series, pd.DataFrame))` para la comparacion de indices, y rechazar un `y` con mas de una dimension util con un `ErrorDatos` que diga "seleccione una columna" (como ya hace `validacion.py:473-477`). Afecta a `ajustar_y_predecir`, `buscar_hiperparametros` e `importancia_variables`.
Medido: `y` de otro periodo como `Series` -> `ErrorDatos` correcto; **el mismo** `y` como `DataFrame` de una columna -> aceptado sin aviso, coeficientes `{'a': -0.0014, 'b': -0.0009, ...}` con la relacion real (0.02, -0.01) desaparecida. `buscar_hiperparametros` con `y` desplazado como `DataFrame` elige `alpha=1e+06` frente a `1e-06`. `comparar_modelos` y `validacion_walk_forward` se salvan por accidente, via `valores_y.ndim != 1`.

### 10. Con `metodo='diferencia'`, los rendimientos pasados cambian cuando llega dato futuro y no son el rendimiento de la posicion
`src/icm/energia/curvas.py:649` (`anterior = precio.shift(1)`), `:665`, `:668`
Cambio: para el ajuste aditivo, los rendimientos relativos no se pueden calcular sobre el nivel ajustado. Reconstruir el crudo (`precio - ajuste`) y dividir por el crudo anterior — `total_t = (crudo_t - crudo_{t-1}^{mismo contrato}) / crudo_{t-1}` — o devolver la descomposicion en unidades absolutas (EUR) cuando `metodo='diferencia'` y documentar que ahi no hay rendimiento porcentual bien definido. La causa esta en `_ajustes_hacia_atras` (`:346-373`): el ajuste de cada tramo es la suma acumulada de TODOS los escalones posteriores, y un sumando no se cancela en el cociente (un factor si).
Medido, cadena en contango de las propias pruebas: al truncar en 10-feb, `max|d total|` = 5.3e-05 y `max|d roll|` = 3.0e-03 con 'diferencia' frente a ~1e-16 con 'ratio'; el roll del 2024-01-26 vale -0.053654 con datos hasta marzo y -0.056604 con datos hasta el 10-feb (30 pb, 5,5 % relativo, sobre una fecha que ya era pasado). Y el acumulado reportado es 0.917595 frente al verdadero 0.913890: 37 pb de rentabilidad inventada en 3 meses.
Prueba a anadir, parametrizada por metodo: el rendimiento de la serie coincide fecha a fecha con el del contrato activo. Hoy `test_la_serie_ajustada_reproduce_el_rendimiento_verdadero_en_todas_las_fechas` solo lo comprueba con 'ratio'.

### 11. Los rendimientos se dividen por el nivel anterior sin comprobar que sea positivo: signo invertido
`src/icm/energia/curvas.py:649`
Cambio: `if (anterior.iloc[1:] == 0.0).any():` -> exigir `(anterior.iloc[1:] > 0).all()` con el mismo estilo de mensaje. Y en `serie_continua` con `metodo='ratio'`, validar la positividad de **toda** la serie activa, no solo de las fechas de empalme: `_ajustes_hacia_atras:355-361` solo comprueba el ancla. Para series que cruzan el cero, devolver la variacion absoluta.
Medido: spark spread de -6 a +2 lineal con `metodo='diferencia'` — el nivel sube TODOS los dias y `rendimiento_total` sale negativo en 5 de 19 fechas (hasta -1.3333), con +4.0000 el dia del cruce por cero y un roll de -2.92. Con `metodo='ratio'` y negativos solo fuera del ancla, `serie_continua` no protesta y los 4 primeros rendimientos son [-0.40, -0.667, -2.00, +0.50] sobre un precio que sube siempre. Es el caso para el que el docstring de `serie_continua` recomienda `metodo='diferencia'`.

> Los puntos 10 y 11 tocan el mismo bloque (`curvas.py:640-670`): una sola pasada.

### 12. `redes.py` no exige indice creciente ni sin duplicados: con la tabla ordenada al reves la ventana mira al futuro
`src/icm/ml/redes.py:540` (final de `_preparar`, antes del `return` de `:551`)
Cambio: replicar las dos comprobaciones de `validacion.py:65-79` sobre `marco.index` cuando la entrada sea pandas — `ErrorDatos` si `indice.has_duplicates` y si `not indice.is_monotonic_increasing`, con el mismo texto que ya usan `validacion.py` y `caracteristicas.py:143-155` ("Ordenelo con `.sort_index()`"). Hacerlo tambien en `_alinear_columnas`, porque `predecir` sufre lo mismo por si solo. Las ventanas se construyen por posicion (`_ventanas`, `:416`).
Medido: objetivo construido para depender SOLO del futuro (`y[t] = x[t+1]+...+x[t+5]`), ventana 5, lstm. Con indice creciente la red no aprende nada (RMSE 2.2585 frente a 2.2540 de predecir la media); con `X.iloc[::-1]` e `y.iloc[::-1]` el RMSE baja a 1.0090, `ajustar()` no se queja y `red.ajustada = True`. Y una tabla con el indice enteramente duplicado tambien se ajusta sin queja.
La cabecera del modulo presume de cerrar las tres fugas clasicas de las redes recurrentes; esta cuarta queda abierta y todos los datos de `tests/test_ml_redes.py` se generan con `pd.bdate_range` ya ordenado.

### 13. El Sharpe publicado no es invariante a la escala: `comparar_estrategias` premia el apalancamiento y `coste_de_equilibrio` cae a 0 bps *(fusion D1)*
`src/icm/ml/estrategia.py:691`, `:705`, `:779-781`, `:852`
Cambio: calcular en este modulo los dos Sharpe (y el de `_sharpe_con_coste`) con el estimador aritmetico invariante a escala, `pnl.mean()*n / (pnl.std(ddof=1)*sqrt(n))`, en lugar de reutilizar `ratio_sharpe`, cuyo numerador es la CAGR (`cartera/metricas.py:44-49`). Dejar la CAGR solo en la clave `rentabilidad_anualizada`. Y advertir en el docstring de `resumen_estrategia` (`:679-681`) que `coste_total_anualizado` es aritmetico (`:705`, `costes.mean() * n`) y no coincide con la diferencia de dos CAGR, donde hoy dice "se compara directamente con `rentabilidad_anualizada`".
Medido: misma senal a x1/x2/x3 -> `ratio_sharpe` del modulo 4.116 / 5.297 / 6.834, mientras el aritmetico es 3.217 en los tres; `comparar_estrategias` ordena `['x3','x2','x1']`. Barrido: el Sharpe sube monotonamente hasta 158.6 en x16 y se hunde a -0.298 en x20 (vol anual 3.36), y `coste_de_equilibrio` pasa de 13.08 bps a exactamente 0.0000 aunque la suma del pnl sea +70.66 y el equilibrio aritmetico siga siendo 21.27 bps. `coste_total_anualizado` con 2 bps: 0.0493 frente a una caida real de 0.0780 (-36.7 %).
**Sub-caso fusionado, mismo arreglo** (`:852`, `if not sin_costes > 0.0`): un unico periodo con `pnl <= -1` hace que `rentabilidad_anualizada` devuelva el `-1.0` truncado de `metricas.py:47` y que `coste_de_equilibrio` devuelva 0.0 bps con el mensaje "una estrategia que pierde dinero sin friccion...", cuando la suma del pnl es +1.5659 (sin ese dia: 19.6206 bps). Si se conserva la CAGR por algun motivo, detectar `(1+pnl).min() <= 0` en `resumen_estrategia` y en `coste_de_equilibrio` y devolver NaN en vez de 0.0.
Prueba a anadir: la misma senal a x1 y x2 debe dar el mismo `ratio_sharpe` y el mismo `coste_de_equilibrio`.

### 14. `buscar_hiperparametros` puntua la seleccion sobre los mismos tramos que luego se presentan como fuera de muestra
`src/icm/ml/modelos.py:761-767` (el `GridSearchCV(..., cv=ParticionSklearn(particion))` y el `busqueda.fit(X, y)` de `:768`)
Cambio, por orden de preferencia: (a) recortar internamente `X`, `y` a las fechas anteriores al primer `inicio_prueba` de la particion recibida y buscar con una particion interna sobre ese tramo; (b) exigir un tramo de seleccion explicito (`X_seleccion`, `y_seleccion`) documentando que no puede ser el de evaluacion; (c) seleccion anidada, un `GridSearchCV` por tramo de entrenamiento dentro de `validacion_walk_forward`. Como minimo, escribir en el docstring que los tramos de prueba de `particion` quedan quemados para el informe — hoy `:703-706` advierte de `refit=False` pero no de esto, que es el mismo problema en su forma clasica: `refit` solo afecta al ajuste final, la SELECCION ya esta contaminada.
Y corregir el encadenamiento que propone el propio docstring en `:719-721`.
Medido: alterando `y` SOLO en el tramo de prueba de la ultima particion (entrenamiento identico, verificado con `np.allclose`), la eleccion pasa de `alpha=1e-06` a `alpha=1e+06`; con 3 particiones de 60 alterando solo el ultimo bloque, igual. Optimismo con ruido puro y la rejilla de `arbol_potenciado`: MSE relativo 1.1295 eligiendo con la misma particion frente a 1.1722 eligiendo solo con fechas anteriores (mejor en 4 de 6 semillas, nunca peor).
Atenuante para el orden: `buscar_hiperparametros` no se usa en el notebook, asi que el dano hoy esta en la receta documentada, no en una cifra publicada.

### 15. Con indice MultiIndex la misma fecha cae a los dos lados del corte y el embargo se cuenta en filas
`src/icm/ml/validacion.py:65-71`
Cambio: rechazar explicitamente un `MultiIndex` (o cualquier indice cuyo primer nivel tenga fechas repetidas) con un `ErrorDatos` que diga que la particion es por posicion y no sabe de paneles, y que el panel hay que partirlo por fecha. El control de duplicados existe con su razon escrita ("una misma fecha puede acabar en entrenamiento y en prueba a la vez") y un MultiIndex la burla: los pares `(fecha, activo)` son unicos y el indice es creciente.
Medido: panel de 140 fechas x 5 activos = 700 filas, aceptado sin queja; con `n_particiones=4, tamano_prueba=48`, 1 fecha compartida entre entrenamiento y prueba en 4 de las 5 particiones (p.ej. 2018-05-22, A/B/C en entrenamiento y D/E en prueba). Y el embargo efectivo es `embargo/n_activos`: `embargo=5` compra 0 dias habiles de hueco, `embargo=25` compra 4 — justo lo que rompe la regla `embargo = h` que recomienda el docstring de `:220-225`. `describir()` imprime "embargo: 5" con toda naturalidad.

### 16. `rendimiento_roll` invierte el signo si el contrato lejano cotiza negativo, y contradice a `estado_curva`
`src/icm/energia/curvas.py:411`
Cambio: `/ precio_lejano` -> `/ precio_lejano.abs()`, igual que hace `estado_curva` en `:437` (`magnitud = precio_lejano.abs()`) precisamente para no invertir la clasificacion; o rechazar los lejanos negativos con `ErrorDatos` remitiendo al diferencial absoluto. La nota del docstring (`:604-607`) advierte de que la razon relativa "pierde sentido", pero el codigo no avisa ni corrige: devuelve un numero plausible con el signo cambiado.
Medido: cercano=-2, lejano=-8 -> `estado_curva` dice `backwardation` y `rendimiento_roll` da -9.1313, cuando el docstring afirma "Es positivo en backwardation"; con `|lejano|` sale +9.1312.
Ampliar `test_el_signo_del_carry_coincide_con_el_estado_de_la_curva` a un par con precios negativos.

### 17. `serie_continua` construye la serie con un contrato ya vencido si el calendario esta obsoleto
`src/icm/energia/curvas.py:296`
Cambio: tras `posicion = fechas_roll[:-1].searchsorted(...)`, comprobar que el contrato activo siga vivo — `ErrorDatos` si el ultimo vencimiento es anterior a la ultima fecha del indice, remitiendo a completar `vencimientos`; o devolver NaN en esas fechas, igual que hace `dias_a_vencimiento`. Hoy la cadena se decide solo con las fechas de roll y el ultimo contrato "aguanta hasta el final" (comentario de `:294-295`).
Medido: indice de 20 dias habiles de febrero de 2024 con `vencimientos={'F1':'2023-01-01','F2':'2023-02-01'}` -> `dias_a_vencimiento` devuelve 0 fechas no-NaN de 20 ("falta calendario"), mientras `serie_continua` devuelve 20 precios del contrato F2, vencido el 2023-02-01, con `ajuste=1.0` y sin error. La incoherencia esta dentro del propio modulo.
(Comprobado que el resto del calendario si es correcto: vencimiento en sabado -> se rueda el lunes siguiente; vencimientos posteriores al indice -> no se rueda nunca.)

### 18. El clon por `deepcopy` conserva el ajuste
`src/icm/ml/validacion.py:128`
Cambio: en la rama `except TypeError: return copy.deepcopy(modelo)`, rechazar o limpiar el estado ajustado en vez de copiarlo — comprobar antes del bucle si el modelo recibido ya esta ajustado y lanzar `ErrorDatos` pidiendo un modelo sin entrenar (`sklearn.utils.validation.check_is_fitted` para estimadores; para el resto, atributos que no estaban tras `__init__`, comparando con `type(modelo)()` cuando el constructor lo permita). Como minimo, dejar de prometer "Copia sin ajustar del modelo" en `:112` para esa via.
Medido: red de descenso de gradiente con interfaz `ajustar`/`predecir`, sin `get_params` y con ajuste de arranque caliente, `y` = ruido puro, 8 semillas: entregada sin ajustar, correlacion media -0.023 (correcto); entregada ya ajustada con las 600 fechas, +0.244.
Orden: va aqui abajo porque **nada de lo que trae el repositorio lo dispara** — verificado que `clone` lanza `TypeError` con objetos sin `get_params` (la rama es alcanzable) pero que `RedSecuencial` SI tiene `get_params`. Hace falta un modelo escrito por el usuario, preajustado y de arranque caliente.
Prueba a anadir: `test_el_modelo_recibido_no_queda_ajustado` (`tests/test_ml_validacion.py:263`) solo usa `LinearRegression`, es decir, la via de `clone`, que es la que si funciona.

### 19. Si ninguna epoca mejora, se restauran los pesos ALEATORIOS iniciales
`src/icm/ml/redes.py:641` (y el `load_state_dict` de `:689`)
Cambio: `mejores_pesos = copy.deepcopy(red.state_dict())` -> `mejores_pesos = None`, y tras el bucle `if mejores_pesos is not None: red.load_state_dict(mejores_pesos)`. Si ninguna epoca dio una perdida finita, `ErrorDatos` explicando que la perdida de validacion no fue finita en ninguna epoca (probable valor extremo en el tramo reservado de `y`) en lugar de devolver una red sin entrenar. Complementario: comprobar `np.isfinite(perdida_validacion)` al calcularla. La comparacion de `:677` (`perdida_validacion < mejor_perdida`) es False con `inf` y con NaN.
Medido: mismos datos que el fixture de las pruebas, cambiando UN valor (`y.iloc[-5] = 1e21`, dentro del 20 % de validacion): RMSE 2.2773 frente a 0.0764, peor que predecir la media (2.2486), mientras `perdida_entrenamiento` va de 0.4202 a 0.0019 en el historial y `perdida_validacion` vale `inf` en las 20 epocas. Comparado tensor a tensor: "pesos devueltos == red sin entrenar" = True. `ajustada` es True y `predecir` devuelve numeros de aspecto normal.
Detonante estrecho (descartado por medicion un segundo: con Adam y lr de 5e2 a 1e7 nunca aparece NaN), pero la ruta es incorrecta en si misma.

### 20. La TCN por defecto ignora 27 de las 30 fechas de la ventana
`src/icm/ml/redes.py:362` (en `_crear_red`) o en `ConfiguracionRed.__post_init__` (`:185-201`)
Cambio: si `arquitectura == 'tcn'` y `alcance_tcn < ventana`, `ErrorDatos` (encaja mejor con el estilo del modulo) o `warnings.warn` indicando el alcance efectivo, cuantas fechas se descartan y las capas necesarias para cubrirla: `ceil(log2(1 + (ventana-1)/(NUCLEO_TCN-1)))`. La propiedad `alcance_tcn` (`:204-211`) existe con el docstring "Sirve para detectar la incoherencia mas comun de esta arquitectura" y, comprobado por grep, **no la consulta nadie** en `src/` — solo `tests/test_ml_redes.py:585-587`, que verifica la aritmetica.
Con los valores por omision (`ventana=30`, `capas=1`) el alcance es `(3-1)*(2**1-1)+1 = 3`.
Medido: perturbando la fila t-k en +50, la prediccion en t cambia para t-0/t-1/t-2 y da `|delta| = 0.000e+00` EXACTAMENTE para t-3, t-5, t-10 y t-29. El coste es doble: memoria de 3 dias creyendo tener 30, y 29 filas de arranque sin prediccion por una historia que la red no mira.

### 21. El metodo de ajuste se adivina desde `.attrs`
`src/icm/energia/curvas.py:632`
Cambio: `metodo = marco.attrs.get("metodo", "ratio")` -> no adivinar. Guardar el metodo en una **columna** de la salida de `serie_continua` (junto a `precio`/`contrato`/`ajuste`, `:315-318`), o exigir el argumento `metodo` cuando `attrs` no lo traiga. `attrs` no sobrevive a un viaje por disco (`to_csv`/`read_csv`, parquet), y guardar la serie continua para descomponerla despues es un uso natural.
Medido: serie aditiva recortada a 2024-01-05..2024-02-10 (cruza el primer empalme, ambos ajustes > 0, asi que la validacion "el ajuste multiplicativo debe ser positivo" de `:659-663` no salta) y con `attrs` vaciado: el roll del dia del empalme vale -0.053654 con `metodo='diferencia'` explicito y -1.028464 al caer al defecto (19 veces mayor), sin ningun error. El total acumulado no cambia — solo se destroza el reparto entre precio y roll, que es justo para lo que sirve la funcion.
(Verificado que en pandas 3.0.6 `attrs` SI se propaga por `.loc`, `.copy()`, `pd.concat` y `.reindex`, asi que hace falta una perdida real de metadatos.)

### 22. Los valores por defecto de `ejecutar_estrategia` descartan en silencio el dimensionamiento de `objetivo_volatilidad`
`src/icm/ml/estrategia.py:592` (`senal = senal_desde_prediccion(prediccion, **opciones_senal)`)
Cambio: marcar la salida de `objetivo_volatilidad` con `attrs['tipo'] = 'posicion'` y, en `senal_desde_prediccion` / `ejecutar_estrategia`, avisar con `UserWarning` (o exigir `modo='proporcional'` con `escala=1.0`) cuando llegue una serie asi marcada. Alternativa: un parametro explicito `ya_son_posiciones=True` que salte la construccion de la senal. `coste_de_equilibrio` arrastra el mismo defecto por defecto.
Medido: posiciones pedidas con `objetivo_anual=0.10` (media `|pos|` = 0.629) -> con la receta del docstring (`:376-382`) la volatilidad anual del pnl es 0.1033; con los valores por omision (`modo='signo'`, `limite=1.0`) es 0.1589 (54 % por encima del objetivo) y la media de `|posicion|` pasa a 0.987.
Orden: bajo porque el docstring da la receta correcta en `:373-382` y el notebook (celda 44) la sigue. Es un footgun sin enforcement, no una cifra publicada mal.

### 23. Las fugas condicionadas a los datos pasan la auditoria con los cortes por omision *(mitad superviviente; ver D3)*
`src/icm/ml/caracteristicas.py:598`
Cambio: subir la potencia. Por omision, muchos mas cortes que los tres actuales (`{int(n*f) for f in (0.5, 0.7, 0.85)}`) — por ejemplo todos los de la segunda mitad; medido que 193 cortes consecutivos de `construir_matriz` tardan unos pocos segundos — y, si el indice de entrada trae NaN, incluir un corte en cada hueco y en el dia anterior. Y anadir al docstring de `comprobar_ausencia_de_fuga` que la auditoria solo cubre lo que el constructor hace con los datos que recibe: cualquier relleno, interpolacion o reindexacion previa queda fuera de su alcance (esto es lo correcto de D3, pero hay que decirlo).
Medido: con huecos en 120, 121, 200 y 300 y un constructor que hace `precios[objetivo].bfill()`, `rend_ret0` del 2015-06-18 pasa de 0.00506266 a 0.26742693 al multiplicar por 1.30 el precio de DOS DIAS DESPUES: fuga real. Con cortes por omision **pasa**; con `fechas=[indice[120]]` o `[indice[121]]` se detecta; con 118, 119 o 250 vuelve a pasar. El mensaje de error de `:684-685` anuncia "un relleno hacia atras" como uno de los fallos que busca.

### 24. `_perturbar_futuro` solo altera valores, nunca el indice
`src/icm/ml/caracteristicas.py:639` (dentro de `_perturbar_futuro`, `:626`)
Cambio: anadir al experimento 2 una variante que ademas de los valores altere la parte futura del **indice** (desplazando unos dias las fechas posteriores al corte, o eliminando algunas), de modo que una variable que dependa del calendario posterior cambie. Si se considera fuera de alcance, dejarlo escrito en el docstring junto a la limitacion del punto 23.
Medido: `X['dias_al_siguiente'] = -indice.to_series().diff(-1).dt.days` **pasa** la auditoria, mientras los seis intentos basados en valores (media centrada, tipificacion global, cuantiles globales, rango min-max, rango percentil y maximo acumulado invertido) se detectan todos. El experimento 1 no lo ve porque el `dropna` se lleva la fila del borde.
Poco frecuente, pero en series de energia con calendario irregular "cuando vuelve a haber dato" no es informacion neutra.

### 25. `_tramos` puede dejar el entrenamiento en UNA sola muestra
`src/icm/ml/redes.py:553` (`_tramos`) y el mensaje de `:603-606`
Cambio: elevar `ErrorDatos` en `ajustar` cuando `n_entrenamiento` sea menor que un minimo razonable (2, o mejor `tamano_lote` o un multiplo pequeno de `ventana`), diciendo cuantas muestras han quedado, cuantas se ha llevado el embargo y que se reduzca `ventana` o `proporcion_validacion`. La unica guarda actual, `n_muestras < 2` (`:602`), mira el total; despues `_tramos` resta validacion y embargo y solo garantiza `n_entrenamiento >= 1`. Y ajustar el mensaje de `:605`, que promete que con `ventana+1` filas se puede ajustar.
Medido: `ventana=30` con 40 filas -> `_tramos` devuelve `(1, 2)`, `_escala_y` colapsa a 1.0 (porque `_escala` sustituye la desviacion nula), el ajuste completa 3 epocas y predice sin queja. `proporcion_validacion=0.99` con 30 filas y `ventana=4` -> `(1, 26)`.

## BLOQUE 2 — Revienta (ninguna cifra sale mal; el problema es el mensaje)

### 26. Casos limite normales que salen por `ValueError` opaco en vez de `ErrorDatos`
`src/icm/ml/caracteristicas.py:559`, `:512`, `:557`
Tres cambios en `construir_matriz`:
- `:559` (`X = pd.concat(bloques, axis=1)`): si `bloques` queda vacio, `ErrorDatos` diciendo que `'cruzadas'` necesita al menos dos activos. El control de `:506-509` no lo cubre porque `incluir` no esta vacio. Hoy: `ValueError: No objects to concatenate`.
- `:512` (`if bool((serie <= 0).any())`): validar en `_como_marco` (o aqui) que las columnas de `precios` no esten repetidas, como ya se hace con las de `X` en `:560-565`. Hoy `marco[objetivo]` devuelve un `DataFrame` y sale `ValueError: The truth value of a Series is ambiguous`.
- `:557` (`exo.reindex(marco.index)`): llamar a `_validar_indice_temporal(exo.index, "exogenas")` antes del reindex; hoy solo se aplica a `precios` (`:492`). Hoy: `ValueError: cannot reindex on an axis with duplicate labels`.
(Comprobado que una exogena DESORDENADA si se alinea bien, porque `reindex` trabaja por etiqueta: ahi no hay fuga ni hay nada que arreglar. Igual que serie constante e `incluir=()` con exogenas, que funcionan.)

### 27. Los parametros no enteros se aceptan al construir y revientan con un `IndexError` de numpy a mitad de la validacion
`src/icm/ml/validacion.py:246` (`ParticionWalkForward.__post_init__`)
Cambio: exigir que `n_particiones`, `tamano_prueba`, `embargo` y `tamano_entrenamiento_minimo` sean enteros (`isinstance(v, (int, np.integer))` y no `bool`), con el mismo estilo de mensaje que los controles de signo que ya hay en `:247-266`.
Medido: `ParticionWalkForward(embargo=5.7).dividir(X)` devuelve 5 tramos con huecos `np.float64(5.0)`, y `validacion_walk_forward` con esa particion lanza `IndexError: arrays used as indices must be of integer or boolean type`, que no dice que parametro lo ha causado. Igual con `tamano_entrenamiento_minimo=100.5`. `tamano_prueba=50.9` no da error: `int()` lo trunca a 50 en silencio. `n_particiones=2.5` muere en `range()` con un `TypeError` igual de opaco.

## BLOQUE 3 — Cosmetico

### 28. `MediaCero` no fija `n_features_in_` ni `feature_names_in_` y no valida el numero de columnas al predecir
`src/icm/ml/modelos.py:93` (`fit`) y `:98` (`predict`)
Cambio: usar `sklearn.utils.validation.validate_data(self, X, y, ...)` en `fit` (o al menos fijar `self.n_features_in_` y `self.feature_names_in_` ademas de `n_variables_`) y `validate_data(self, X, reset=False)` en `predict`, que de paso hace que un vector de una dimension falle en lugar de devolver n predicciones.
Medido: tras `fit` con `X` de 3 columnas, ridge/lasso/elastic_net/arbol_potenciado/bosque dan `n_features_in_=3` y `feature_names_in_=['a','b','c']`; `media_cero` da `AttributeError` en los dos. `m.predict(DataFrame de 7 columnas)` devuelve `[0., 0.]` sin protestar. `check_estimator(MediaCero(), on_fail=None)` en sklearn 1.9.1: 52 comprobaciones, 16 fallidas, **todas** de validacion de entrada; ninguna de clonado ni de parametros.
Orden: el ultimo, y con margen. Comprobado por grep que **nada** en `src/`, `tests/` ni `notebooks/` lee `n_features_in_` ni `feature_names_in_`, asi que el "codigo generico que recorre los modelos de una tabla" no existe en el repositorio; y como `MediaCero` siempre predice cero, un desajuste de columnas no puede producir una cifra equivocada. Es incoherencia de contrato, no dano.

---

## Nota de ejecucion

Tres lotes que conviene tocar de una vez, porque comparten lineas:
- **`estrategia.py:555-590`** -> puntos 1, 2, 3 (y adaptar `tests/test_ml_convencion.py:102-117`).
- **`caracteristicas.py:626-762`** -> puntos 4, 5, 23, 24 (y anadir la prueba que falta: que la auditoria DETECTA una fuga en una exogena).
- **`curvas.py:640-670`** -> puntos 10 y 11.

Y dos correcciones que hay que hacer fuera de `src/`, porque el repositorio se equivoca en su propia documentacion: el docstring de `estrategia.py` (punto 1) y la celda 21 de `notebooks/03_trading_energia_ml.ipynb` (punto 8).

---

## Informes por módulo

### src/icm/ml/validacion.py

**Veredicto.** La geometria de la particion es solida: he recorrido unas 1.800 combinaciones de n_particiones, tamano_prueba, embargo, modo expansivo/rodante, minimo y longitud de serie, y en todas ellas entrenamiento y prueba no se solapan ni por posicion ni por fecha, el hueco es exactamente el embargo pedido, el modo expansivo arranca siempre en 0 y crece, el rodante mantiene el tamanio constante, los tramos de prueba son contiguos, terminan en la ultima observacion y no repiten ni pierden ni una fecha; describir() coincide con dividir() y el R cuadrado contra la media del entrenamiento (Campbell-Thompson) esta bien transcrito, igual que la correlacion de Pearson (coincide con scipy a 1e-12). Los fallos que quedan estan fuera de ese nucleo: la copia del modelo no borra el ajuste cuando cae a deepcopy (con una y de ruido puro, una red preentrenada con toda la serie informa una correlacion fuera de muestra de +0,24 en vez de 0, en 8 semillas), acierto_direccional cuenta los NaN como fallos y es la unica metrica que no avisa volviendose NaN (0,985 real informado como 0,808 con la propia RedSecuencial del repositorio), y la prediccion etiquetada que devuelve el modelo se desetiqueta y se pega por posicion (un Sharpe de +18 se convierte en -1,2 en silencio). Los dos primeros son fugas o sesgos reales; el tercero es la clase de desalineacion que el encargo senala como mas peligrosa.

#### [alto · fuga_futuro] El clon por deepcopy conserva el ajuste: un modelo preentrenado con toda la serie fabrica senal de la nada

`/home/user/ICM/src/icm/ml/validacion.py` línea 128 · reproducido

_clonar promete en su docstring una "copia sin ajustar del modelo" y esa promesa es la garantia central del modulo. Se cumple por la via de sklearn.base.clone, pero NO por la via de respaldo: copy.deepcopy copia el objeto entero, estado ajustado incluido. La via de respaldo se dispara exactamente para los modelos del convenio del repositorio que no exponen get_params (una red de torch a pelo, una clase propia, los ModeloPrimeraColumna/ModeloContador del propio fichero de pruebas; RedSecuencial se salva por casualidad, porque si tiene get_params). Si el usuario hace lo mas natural del mundo -- entrenar la red y despues "validarla" -- cada particion arranca desde unos pesos que ya han visto las 600 fechas, incluidas las de prueba. Y si el ajustar del modelo es de arranque caliente (no reinicializa, como cualquier envoltorio de torch que siga desde los pesos actuales), esa informacion sobrevive al reajuste y contamina las predicciones fuera de muestra. No salta ningun aviso: los numeros solo salen mejor. La prueba test_el_modelo_recibido_no_queda_ajustado solo comprueba LinearRegression, es decir, unicamente la via de sklearn.clone, que es la que si funciona.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/g5b.py: n=600, k=200 variables, y = RUIDO PURO (no hay nada que aprender). Red de descenso de gradiente con interfaz ajustar/predecir y sin get_params, ajustar de 25 pasos con arranque caliente. Correlacion fuera de muestra que informa resumen, en 8 semillas: red entregada sin ajustar [-0.073 -0.043 0.059 -0.045 -0.031 -0.050 0.002 -0.000], media -0.023 (correcto, no hay senal); red entregada ya ajustada con las 600 fechas [0.228 0.220 0.311 0.245 0.230 0.223 0.242 0.252], media +0.244. Tambien en g3_clonado.py: _clonar de una red preajustada devuelve pesos [0.5, -0.2996, 0.0003] en vez de ceros, y el bucle no toca el objeto original (por eso no hay contaminacion ENTRE particiones, solo desde el estado de entrada).

**Arreglo propuesto.** En _clonar, cuando se cae a copy.deepcopy, rechazar o limpiar el estado ajustado en vez de copiarlo: comprobar antes del bucle si el modelo recibido ya esta ajustado y lanzar ErrorDatos pidiendo un modelo sin entrenar (sklearn.utils.validation.check_is_fitted para estimadores, y para el resto la presencia de atributos que no estaban tras __init__, comparando con type(modelo)() cuando el constructor lo permita). Como minimo, dejar de prometer "copia sin ajustar" en la via de deepcopy y documentar que el modelo debe llegar sin entrenar. Anadir una prueba con un modelo sin get_params, preajustado y de arranque caliente.

#### [alto · estadistico] acierto_direccional cuenta como fallo toda fecha sin prediccion, y es la unica metrica que no sale NaN

`/home/user/ICM/src/icm/ml/validacion.py` línea 159 · reproducido

_acierto_direccional excluye las fechas de rendimiento real nulo, pero no las fechas sin prediccion. np.sign(nan) es nan, y nan == sign(real) es False, asi que cada NaN entra en el denominador como un fallo. El resultado es la proporcion de acierto multiplicada por la fraccion de filas validas, un numero finito y sesgado sistematicamente a la baja. Lo grave es el contraste con las demas metricas: rmse, mae, sesgo, correlacion y r2_fuera_muestra si se vuelven NaN y gritan que algo falta, mientras acierto_direccional devuelve una cifra de aspecto normal. Y no es un caso hipotetico: es precisamente la combinacion que documenta el repositorio. RedSecuencial.predecir devuelve NaN en las ventana-1 primeras filas de la tabla que recibe (redes.py, lineas 460-466), y cada tramo de prueba empieza con ese hueco porque el modulo le pasa solo las filas de prueba. Con ventana=10 y tramos de 50, el 18% de las filas son NaN y el acierto informado es un 18% menor que el real. Un modelo bueno se descarta por un numero que el modulo presenta como valido. Lo mismo ocurre con NaN en y (reproducido).

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/g4_nan.py: modelo que imita a RedSecuencial (ventana=10, NaN en las 9 primeras filas de cada tramo), 5 particiones de 50. por_particion: rmse/mae/sesgo/correlacion/r2 = NaN en las 5 particiones; acierto_direccional = 0.80, 0.80, 0.82, 0.82, 0.80. resumen['acierto_direccional'] = 0.8080. Acierto real calculado solo sobre las 205 filas con prediccion = 0.9854. Sesgo relativo = -18.0%, exactamente la fraccion de NaN (45/250). En g6_varios.py, con y con NaN en las 3 ultimas fechas: rmse = NaN pero acierto_direccional = 0.9667.

**Arreglo propuesto.** En _acierto_direccional, construir la mascara como utiles = (reales != 0.0) & np.isfinite(reales) & np.isfinite(predicciones), devolver NaN si no queda ninguna, y anadir al diccionario de _metricas (y a resumen) una columna n_validas o n_ausentes para que el hueco quede a la vista en vez de disolverse en el denominador. Prueba: un modelo que devuelva NaN en las k primeras filas de cada tramo debe dar acierto_direccional igual al calculado sobre las filas restantes.

#### [medio · alineacion] La prediccion etiquetada que devuelve el modelo se desetiqueta y se pega por posicion, sin comprobar el orden

`/home/user/ICM/src/icm/ml/validacion.py` línea 498 · reproducido

np.asarray(crudas, dtype=float).reshape(-1) descarta el indice que trae la prediccion, y despues pd.Series(crudas, index=fechas) la vuelve a etiquetar por posicion. Si el modelo devuelve una Series o un DataFrame cuyas filas no estan en el mismo orden que las de la tabla que recibio -- basta un groupby, un sort_index, un sort_values o un reindex interno --, cada prediccion queda pegada a la fecha de otra. El unico control es de tamanio (crudas.size != pos_pru.size), que una reordenacion pasa sin ruido. Es el fallo de la familia que el propio encargo senala como mas peligrosa: el resultado sigue pareciendo razonable. Y contradice lo que el docstring de ResultadoValidacion vende: una serie que se puede pasar a una evaluacion de estrategia "sin volver a pensar en el alineamiento". Aqui el modulo tiene el indice del modelo en la mano y lo tira, cuando podria usarlo para verificar.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/g8_orden.py: mismo modelo lineal, una version devuelve pd.Series(pred, index=X.index) y la otra le anade .sort_values(). Con 5 particiones de 50 sobre n=600, correlacion informada +0.957 frente a -0.088; acierto direccional 0.904 frente a 0.488; Sharpe anualizado de la estrategia de signo +18.02 frente a -1.21. El indice de salida es identico en los dos casos y el conjunto de valores predichos tambien, asi que nada permite distinguirlos a posteriori.

**Arreglo propuesto.** Si la prediccion llega con indice (pd.Series o pd.DataFrame), reindexarla a fechas antes de convertirla -- crudas = crudas.reindex(fechas) -- y lanzar ErrorDatos si aparece algun ausente, en vez de aplicar np.asarray directamente. Anadir una prueba con un modelo que devuelva la Series reordenada.

#### [medio · fuga_futuro] Con indice MultiIndex (panel) la misma fecha cae a los dos lados del corte y el embargo se cuenta en filas, no en fechas

`/home/user/ICM/src/icm/ml/validacion.py` línea 65 · reproducido

_indice_temporal acepta cualquier indice unico y creciente, y un MultiIndex (fecha, activo) de un panel lo es. El control de duplicados existe justo para impedir que "una misma fecha puede acabar en entrenamiento y en prueba a la vez", pero un MultiIndex lo esquiva: los pares (fecha, activo) son unicos aunque las fechas se repitan tantas veces como activos. Con eso caen las dos garantias del modulo. Primera: el corte por posicion parte una fecha por la mitad, dejando unos activos del dia t en entrenamiento y otros del dia t en prueba, es decir informacion contemporanea (los rendimientos de un mismo dia estan muy correlacionados entre activos por el factor de mercado). Segunda, y peor: el embargo se mide en observaciones, asi que con cinco activos embargo=5 -- lo que el docstring recomienda para un objetivo a 5 dias, "embargo = h" -- compra cero dias habiles de hueco, no cinco. El usuario cree haber purgado el solapamiento del objetivo y no ha purgado nada. Nada avisa: describir() imprime "embargo: 5" con toda naturalidad.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/g2_panel.py y g7.py: panel de 140 fechas x 5 activos = 700 filas, aceptado sin queja. Con n_particiones=4, tamano_prueba=48: embargo=0 -> 0 dias habiles de hueco y 1 fecha compartida entre entrenamiento y prueba en las 4 particiones (p.ej. 2018-05-22, activos A,B,C en entrenamiento y D,E en prueba); embargo=5 -> 0 dias habiles de hueco; embargo=25 -> 4 dias habiles de hueco. Es decir, el embargo efectivo es embargo/n_activos.

**Arreglo propuesto.** En _indice_temporal, rechazar explicitamente un MultiIndex (o cualquier indice cuyo primer nivel tenga fechas repetidas) con un ErrorDatos que diga que la particion es por posicion y no sabe de paneles, y que el panel hay que partirlo por fecha. Si se quiere dar soporte, calcular los cortes sobre las fechas unicas y traducirlos a posiciones, y medir el embargo en fechas.

#### [bajo · caso_limite] Los parametros no enteros se aceptan al construir y revientan con un IndexError de numpy a mitad de la validacion

`/home/user/ICM/src/icm/ml/validacion.py` línea 246 · reproducido

__post_init__ valida signos y minimos pero no el tipo. Con embargo=5.7 o tamano_entrenamiento_minimo=100.5 el calculo se propaga en coma flotante, np.arange devuelve posiciones float y dividir()/describir() funcionan sin protestar; el fallo aparece mucho despues, dentro de numpy, como "IndexError: arrays used as indices must be of integer or boolean type", que no dice ni que parametro lo ha causado. Con tamano_prueba=50.9 no hay error: int() lo trunca a 50 en silencio. Y n_particiones=2.5 muere en range() con un TypeError igual de opaco. Es solo calidad del mensaje -- ningun numero sale mal --, pero rompe el patron del resto de la clase, que se esmera en explicar cada parametro imposible.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/g7.py y g9.py: ParticionWalkForward(embargo=5.7).dividir(X) devuelve 5 tramos con huecos np.float64(5.0); validacion_walk_forward con esa particion lanza IndexError: arrays used as indices must be of integer or boolean type. Igual con tamano_entrenamiento_minimo=100.5. tamano_prueba=50.9 se acepta y da tramos de 50. n_particiones=2.5 -> TypeError: 'float' object cannot be interpreted as an integer.

**Arreglo propuesto.** Exigir en __post_init__ que n_particiones, tamano_prueba, embargo y tamano_entrenamiento_minimo sean enteros (isinstance(v, (int, np.integer)) y no bool), con el mismo estilo de mensaje que los demas controles.

### src/icm/ml/estrategia.py (con tests/test_ml_estrategia.py y tests/test_ml_convencion.py)

**Veredicto.** La mecanica central esta bien: el emparejamiento senal/posicion/rendimiento con retardo>=1 es correcto, la rotacion y los costes son exactos (+1 a -1 paga 2, sin cambio paga 0, verificado periodo a periodo), los huecos del rendimiento se tratan como dice la documentacion, objetivo_volatilidad usa la volatilidad de t y no la del periodo entero, y la biseccion de coste_de_equilibrio converge con un unico cruce por cero (200/200 casos). Los fallos estan todos alrededor de la deduccion del retardo a partir de attrs y de la metrica que se publica: la propia documentacion del modulo prescribe una receta (.shift(-1) sobre ResultadoValidacion.predicciones) que es aritmeticamente identica a retardo=0 y no emite ningun aviso, un horizonte mayor que 1 se acepta en silencio y multiplica el pnl por h, y el Sharpe que publica no es invariante a la escala de la posicion, lo que hace que comparar_estrategias premie el apalancamiento y que coste_de_equilibrio devuelva 0 bps por puro arrastre de volatilidad. Cuatro de los seis hallazgos estan reproducidos con numeros; ninguno lo cubren los 50 tests que pasan.

#### [critico · fuga_futuro] La receta que manda el docstring (.shift(-1) sobre ResultadoValidacion.predicciones) es exactamente retardo=0 y no avisa

`src/icm/ml/estrategia.py` línea 26 · reproducido

El docstring del modulo (lineas 20-33, repetido en 522-527) afirma que ResultadoValidacion.predicciones esta "indexada en la fecha del rendimiento que predice, no en la fecha en que se decide" y ordena corregirlo con `prediccion = resultado.predicciones.shift(-1)` dejando retardo=1. Es falso para la tuberia de la propia libreria: validacion.py:508 indexa las predicciones con `indice[pos_pru]`, es decir con las filas de X/y, y la y canonica (construir_matriz -> objetivo_rendimiento_futuro) esta indexada en la FECHA DE DECISION t con y[t]=r[t+1] (attrs convencion='futuro'). Por tanto predicciones ya viene en la convencion de este modulo, y aplicarle shift(-1) la adelanta un periodo: la posicion de t pasa a decidirse con informacion de t y a cobrar el rendimiento de t, que ya habia ocurrido. Es literalmente el retardo=0 contra el que el modulo dedica 40 lineas de docstring, pero como los rendimientos que se pasan son contemporaneos y el retardo es 1, NO salta ningun UserWarning. El notebook del proyecto (notebooks/03_trading_energia_ml.ipynb, celda de ejecutar_estrategia) hace lo correcto y pasa `predicciones` tal cual contra `objetivo`, lo que confirma que el equivocado es el docstring y no el notebook. Ningun test toca esta ruta.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/e7.py — proceso con momento (r[t]=0.12*r[t-1]+eps), construir_matriz + validacion_walk_forward(Ridge) y las tres formas de ejecutar:  A) p.shift(-1) con retardo=1 (receta del docstring): sharpe = +1.711, suma pnl = +1.3771.  B) p tal cual con retardo=1 (convencion real de la serie): sharpe = -0.352, suma pnl = -0.2475.  C) p tal cual con retardo=0 (la trampa declarada, que si avisa): sharpe = +1.704.  Y la comprobacion decisiva: `np.allclose(A.pnl, C.pnl.reindex(A.pnl.index))` -> True, A y C son la MISMA cuenta de resultados. Correlaciones: corr(pred[t], r[t+1]) = -0.038 (no hay capacidad predictiva) frente a corr(pred[t], r[t]) = +0.059 (correlacion espuria con el pasado ya conocido), que es de donde sale todo el +1.711.

**Arreglo propuesto.** Corregir el docstring: ResultadoValidacion.predicciones YA esta indexada en la fecha de decision cuando la y viene de objetivo_rendimiento_futuro/construir_matriz, asi que se pasa tal cual (contra md.y, con retardo=0 deducido, o contra los rendimientos contemporaneos con retardo=1) y NUNCA con .shift(-1). Limitar la advertencia sobre .shift(-1) a las series de icm.ml.evaluacion construidas sobre una y contemporanea, y anadir un test en tests/test_ml_convencion.py que ejecute construir_matriz -> validacion_walk_forward -> ejecutar_estrategia y compruebe que el pnl con .shift(-1) coincide con el de retardo=0 (es decir, que la receta esta prohibida).

#### [alto · estadistico] Con attrs['horizonte'] > 1 el retardo deducido es 0 y el pnl solapa: cada movimiento se cobra hasta h veces

`src/icm/ml/estrategia.py` línea 562 · reproducido

ejecutar_estrategia lee `horizonte_serie` de attrs (linea 562) pero solo lo usa para redactar el mensaje de error de la linea 569. Cuando se le pasa la salida de objetivo_rendimiento_futuro con horizonte=h>1, deduce retardo=0 sin error ni aviso y calcula pnl[t] = posicion[t] * r(t -> t+h) para TODAS las t. Los tramos se solapan h-1 periodos, de modo que cada movimiento diario del precio se cobra hasta h veces: la cuenta de resultados no es reproducible con una unidad de capital. Ademas el pnl queda fuertemente autocorrelacionado, con lo que la volatilidad anualizada se infla por ~sqrt(h) y TODAS las anualizaciones de resumen_estrategia (rentabilidad, volatilidad, Sharpe, VaR, coste anualizado) dejan de significar nada; los costes, en cambio, se cobran una vez por periodo aunque la posicion se mantenga h. test_ml_convencion.py::test_un_horizonte_mayor_tambien_se_rechaza solo comprueba que se rechaza retardo=1 con h=5; el camino por defecto (retardo deducido = 0) no se prueba.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/e2.py — comprar y mantener (prediccion constante 1) sobre 1260 dias cuyo rendimiento log total es -0.0851:  h=1  -> suma pnl = -0.0851 (1.00x el rendimiento real), autocorr(pnl,1) = -0.03, vol_anual = 0.158.  h=5  -> suma pnl = -0.6106 (7.17x), autocorr(pnl,1) = +0.783, vol_anual = 0.344.  h=21 -> suma pnl = -3.2360 (38.01x), autocorr(pnl,1) = +0.950, vol_anual = 0.705. Una posicion fija de 1 no puede producir 38 veces el rendimiento del activo.

**Arreglo propuesto.** En ejecutar_estrategia, tras leer `horizonte_serie`, rechazar con ErrorDatos (o como minimo avisar con UserWarning) cuando `adelantada and horizonte_serie != 1`, explicando que con horizonte h hay que quedarse con una decision cada h periodos (`serie.iloc[::h]`, reponiendo attrs) y anualizar con periodos_anio/h. Anadir el test correspondiente.

#### [alto · fuga_futuro] attrs['convencion'] sobrevive a .shift(), asi que una serie ya realineada desactiva el unico aviso de fuga

`src/icm/ml/estrategia.py` línea 582 · reproducido

La marca convencion='futuro' es un attr de pandas y se propaga intacta por shift, dropna, reindex, rename, astype, copy y aritmetica (verificado en pandas 3.0.6). Es decir: la marca NO describe el alineamiento actual de la serie, solo su origen. Quien convierte la serie adelantada en contemporanea con `.shift(1)` (la operacion mas natural del mundo, y la que la propia libreria describe como equivalente) se queda con una serie contemporanea que sigue diciendo 'futuro': ejecutar_estrategia deduce retardo=0 y, por la condicion `retardo == 0 and not adelantada` de la linea 582, SUPRIME deliberadamente el aviso de informacion futura. El resultado es la fuga completa, en silencio. Y encima, si ese usuario pide el retardo correcto (retardo=1), la linea 568 lo rechaza con un ErrorDatos que le manda usar retardo=0, es decir, le empuja a la fuga. El test test_con_serie_adelantada_no_se_avisa_de_retardo_cero consagra precisamente la supresion del aviso.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/e3.py y e4.py. e3 muestra que attrs sobrevive a shift/dropna/reindex/rename/astype/copy/aritmetica. e4: y = objetivo_rendimiento_futuro(precios,1); contemp = y.shift(1) (comprobado identico al rendimiento ordinario, allclose=True) pero contemp.attrs sigue siendo {'convencion':'futuro',...}. Con prediccion = el rendimiento del propio periodo t: ejecutar_estrategia(pred, contemp) -> avisos emitidos: [] (ninguno), sharpe = 68.39, suma pnl = 12.1416, que coincide EXACTAMENTE con la suma de |r| (12.1416), la huella del retardo=0. Con la marca borrada y retardo=1: sharpe = -0.02, suma pnl = 0.0606. Y ejecutar_estrategia(pred, contemp, retardo=1) lanza ErrorDatos diciendo 'Pase retardo=0 con esta serie'.

**Arreglo propuesto.** Emitir el UserWarning de retardo=0 SIEMPRE que el retardo resuelto sea 0, tambien por la via `adelantada` (cambiando el texto para el caso legitimo: 'se ejecuta con retardo=0 porque la serie viene marcada como adelantada; compruebe que sigue estandolo'), y adaptar test_con_serie_adelantada_no_se_avisa_de_retardo_cero. Mas robusto: que objetivo_rendimiento_futuro guarde en attrs una huella del alineamiento (p. ej. attrs['fecha_inicial'] o un hash del indice) y que ejecutar_estrategia desconfie de la marca cuando la huella no cuadre.

#### [medio · estadistico] El Sharpe publicado no es invariante a la escala de la posicion: comparar_estrategias premia el apalancamiento y coste_de_equilibrio cae a 0 bps por arrastre de volatilidad

`src/icm/ml/estrategia.py` línea 691 · reproducido

resumen_estrategia (691-701) y _sharpe_con_coste (779-781) usan icm.cartera.metricas.ratio_sharpe, cuyo numerador es la CAGR geometrica ((1+pnl).prod()^(n/len)-1) mientras el denominador es lineal en la escala. Pero este modulo define el pnl como aditivo por unidad de capital y curva_beneficio lo dice explicitamente ('se acumula por suma y no por composicion, a proposito'). Consecuencias, todas silenciosas: (a) la misma senal apalancada da Sharpes distintos, asi que comparar_estrategias (que ordena por ratio_sharpe y se presenta como 'el criterio con el que se reparte capital') premia el apalancamiento puro, y la afirmacion de objetivo_volatilidad de que tras normalizar 'lo que queda de diferencia es senal' no se sostiene; (b) por encima de cierta volatilidad el arrastre vuelve negativa la CAGR y coste_de_equilibrio devuelve exactamente 0.0 bps, afirmando que la senal no aguanta ninguna friccion, cuando la senal es la misma y gana dinero; (c) coste_total_anualizado se calcula aritmeticamente (media*n, linea 705) y el docstring dice que 'se compara directamente con rentabilidad_anualizada', que es geometrica: discrepan entre un 22 % y un 37 %.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/e10.py: misma senal, mismo coste (2 bps), modo proporcional, apalancamiento x1/x2/x3 -> ratio_sharpe del modulo = 4.116 / 5.297 / 6.834, mientras el Sharpe estandar (media/desviacion * sqrt(252)) es 3.217 en los tres casos; comparar_estrategias ordena ['x3','x2','x1']. e9.py: barrido de apalancamiento, el Sharpe sube monotonamente hasta 158.6 en x16 (vol anual 2.68) y se hunde a -0.298 en x20 (vol anual 3.36), y coste_de_equilibrio pasa de 13.08 bps a exactamente 0.0000 bps aunque la suma del pnl sea +70.66 y el equilibrio aritmetico (suma pnl bruto / rotacion total) siga siendo 21.27 bps en todos los casos. e12.py: con 2 bps, coste_total_anualizado = 0.0493 frente a una caida real de la rentabilidad anualizada de 0.0780 (error -36.7 %); con 20 bps, -21.9 %.

**Arreglo propuesto.** Calcular en este modulo los dos Sharpe (y el de _sharpe_con_coste) con el estimador aritmetico invariante a escala, `pnl.mean()*n / (pnl.std(ddof=1)*sqrt(n))`, en lugar de reutilizar ratio_sharpe con numerador geometrico; dejar la CAGR solo en la clave rentabilidad_anualizada y advertir en el docstring de resumen_estrategia que coste_total_anualizado es aritmetico y no coincide con la diferencia de dos CAGR. Anadir un test: la misma senal a x1 y x2 debe dar el mismo ratio_sharpe y el mismo coste_de_equilibrio.

#### [medio · caso_limite] Un solo periodo con 1+pnl <= 0 hace que coste_de_equilibrio devuelva 0.0 bps en silencio

`src/icm/ml/estrategia.py` línea 852 · reproducido

Si algun periodo tiene pnl <= -1 (perfectamente posible en energia con precios negativos, o con la posicion apalancada que el propio modulo admite via limite>1), rentabilidad_anualizada de icm.cartera.metricas detecta crecimiento<=0 y devuelve el valor fijo -1.0. Entonces ratio_sharpe pasa a ser -1/volatilidad, un numero sin contenido, y en coste_de_equilibrio la comprobacion `if not sin_costes > 0.0` de la linea 852 se cumple y la funcion devuelve 0.0 con el mensaje conceptual 'una estrategia que pierde dinero sin friccion no tiene ninguna friccion que aguantar'. Pero la estrategia no pierde dinero: la suma del pnl es positiva. Ni el modulo ni resumen_estrategia dan ninguna senial de que la cifra este truncada.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/e9.py, segunda parte: se pone un rendimiento de -1.4 en una fecha en la que la posicion es +1. Resultado: pnl de ese dia = -1.400, suma pnl = +1.5659 (la estrategia gana en agregado), rentabilidad_anualizada = -1.0000 exacto (valor truncado), ratio_sharpe = -1.6752, coste_de_equilibrio = 0.0000 bps. Sin ese unico dia, coste_de_equilibrio = 19.6206 bps.

**Arreglo propuesto.** Con el Sharpe aritmetico del hallazgo anterior el problema desaparece. Si se quiere conservar la CAGR, detectar en resumen_estrategia y en coste_de_equilibrio el caso `(1+pnl).min() <= 0` y avisar (o devolver NaN en lugar de 0.0), distinguiendo 'la estrategia no aguanta friccion' de 'la cartera se ha ido a cero y la cifra no esta definida'.

#### [bajo · caso_limite] Los valores por defecto de ejecutar_estrategia descartan en silencio el dimensionamiento de objetivo_volatilidad

`src/icm/ml/estrategia.py` línea 592 · reproducido

objetivo_volatilidad devuelve posiciones, no una prediccion, y solo se ejecutan correctamente con modo='proporcional', escala=1.0 y un limite no menor. Con los valores por defecto (modo='signo', limite=1.0) ejecutar_estrategia vuelve a construir la senal y reduce todo a +-1, destruyendo el escalado sin error ni aviso; el docstring lo explica pero nada lo comprueba, y coste_de_equilibrio arrastra el mismo defecto por defecto. El resultado sigue pareciendo razonable, que es exactamente el modo de fallo que este modulo dice querer evitar.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/e8.py, apartado 1: posiciones pedidas con objetivo_anual=0.10 (media |pos| = 0.629). Con la receta del docstring (modo proporcional, escala 1.0) la volatilidad anual del pnl es 0.1033, practicamente el objetivo. Con los valores por defecto es 0.1589 (54 % por encima del objetivo) y la media de |posicion| pasa a 0.987: el escalado se ha perdido por completo.

**Arreglo propuesto.** Marcar la salida de objetivo_volatilidad con attrs['tipo'] = 'posicion' y, en senal_desde_prediccion / ejecutar_estrategia, avisar con UserWarning (o exigir modo='proporcional' con escala=1.0) cuando se recibe una serie asi marcada; alternativamente ofrecer un parametro explicito `ya_son_posiciones=True` que salte la construccion de la senal.

### src/icm/ml/caracteristicas.py

**Veredicto.** construir_matriz esta limpio: una auditoria exhaustiva con un corte en CADA fecha (193 cortes consecutivos por configuracion, con y sin huecos en los precios, con y sin exogenas, horizonte 1 y 5) no encuentra ninguna dependencia del futuro, la alineacion de objetivo_rendimiento_futuro es correcta a mano para h=1,2,3 en las dos convenciones, el calendario depende solo del indice y emite siempre sus 8 columnas, los retardos cruzados valen log(otro[t-k]/otro[t-k-1]) empezando en k=1 y la exogena con retardo 0 es exactamente exo[t]. El problema esta en la auditoria, y es de falsos negativos, no de falsas alarmas: aguanta bien todo lo que se calcula con la muestra entera (detecta media centrada, tipificacion global, cuantiles globales discretizados, rango min-max, rango percentil, maximo acumulado invertido, winsorizacion global y escala redondeada), pero da el visto bueno a un shift(-1) puro en cuanto la fuga entra por un kwarg que no se llama 'exogenas' (hallazgo 1) o por una exogena pasada como Series (hallazgo 2), y no ve los rellenos hacia atras ni las interpolaciones que su propio mensaje de error dice buscar (hallazgo 3). Mientras no se arreglen los dos primeros, la frase del docstring de que la garantia 'se somete a contraste' vale para construir_matriz pero no para los constructores de usuario, que son precisamente los que la auditoria existe para vigilar.

#### [alto · fuga_futuro] La auditoria de fuga no trunca ni perturba ningun argumento de datos que no se llame 'exogenas'

`/home/user/ICM/src/icm/ml/caracteristicas.py` línea 741 · reproducido

comprobar_ausencia_de_fuga solo somete a los dos experimentos 'precios' y kwargs['exogenas'] (lineas 741-762). Cualquier otra Series/DataFrame que el constructor reciba por kwargs (demanda, volumen, produccion eolica, una segunda exogena con otro nombre, o simplemente 'exogena' en singular por un error de tecleo) se pasa IDENTICA a las tres llamadas: a la de referencia, a la truncada y a la perturbada. Por construccion las tres matrices coinciden en esas columnas y _comparar no puede ver nada. Eso deja pasar la fuga mas descarada que existe, un shift(-1) puro, en la funcion que el propio modulo presenta como la garantia de causalidad de todo lo demas. Es especialmente grave porque el docstring invita a usar la auditoria sobre 'cualquier constructor que escriba el usuario' con la firma (precios, objetivo, **kwargs), es decir, justo con kwargs arbitrarios.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/kwargs_y_previo.py. Constructor con X['demanda_manana'] = demanda.shift(-1) y kwarg demanda=DEM: comprobar_ausencia_de_fuga(ctor, P, 'ALFA', demanda=DEM) PASA sin error. El MISMO constructor con el kwarg renombrado a 'exogenas' se detecta al instante: "Fuga de informacion del futuro en la variable 'demanda_manana': en la fecha 2015-10-08 vale 29599.2 ...". La unica diferencia entre pasar y fallar es el nombre del parametro.

**Arreglo propuesto.** Recorrer todos los kwargs y aplicar el mismo tratamiento a cada valor que sea Series o DataFrame con indice comparable con el de precios (truncar con .loc[idx <= corte] en el experimento 1 y _perturbar_futuro en el experimento 2), en vez de buscar la clave literal 'exogenas'. Como red de seguridad, si queda algun kwarg de tipo Series/DataFrame sin tratar, lanzar ErrorDatos explicando que la auditoria no puede garantizar nada sobre el, en lugar de devolver un visto bueno silencioso.

#### [alto · fuga_futuro] _comparar descarta en silencio las columnas que no estan en las dos matrices, y _perturbar_futuro cambia el tipo de 'exogenas' de Series a DataFrame

`/home/user/ICM/src/icm/ml/caracteristicas.py` línea 650 · reproducido

Son dos defectos que se combinan. (a) _comparar (linea 650) interseca los nombres de columna y solo protesta si la interseccion queda VACIA; si coincide una sola columna, todas las demas quedan sin auditar y no se dice nada. (b) En el experimento de perturbacion (lineas 758-761) la exogena se pasa por _como_marco antes de _perturbar_futuro, asi que el constructor recibe un DataFrame donde en la llamada de referencia recibio una Series; en el experimento de truncado (linea 747) se conserva la Series. Un constructor que use retardos() sobre la exogena tal como la recibe nombra sus columnas segun era_serie (linea 167), asi que en la referencia salen 'exo_ret0' y en la perturbacion 'exo_temperatura_ret0': ninguna columna exogena entra en la comparacion. Resultado: una fuga en la exogena no se ve ni en el truncado (la fila del corte desaparece por el dropna, que es justo el punto ciego que el experimento 2 deberia cubrir) ni en la perturbacion (columna ignorada). No hay ninguna prueba que verifique que la auditoria DETECTA fugas en exogenas; test_la_auditoria_acepta_una_exogena_prevista_con_retardo_cero solo comprueba que las acepta.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/exogenas.py. Constructor que usa retardos(exogenas.shift(-1), (0,), prefijo='exo'). Con exogenas=EXO (Series): columnas ['rend_ret1','exo_ret0'] en la referencia y ['rend_ret1','exo_temperatura_ret0'] en la perturbada; comunes = ['rend_ret1']; la auditoria PASA. Con exogenas=EXO.to_frame() la misma fuga se detecta: "Fuga ... en la variable 'exo_temperatura_ret0': en la fecha 2016-02-25 vale 21.3969 ...".

**Arreglo propuesto.** En comprobar_ausencia_de_fuga, perturbar la exogena conservando su tipo original (si la entrada era Series, devolver _perturbar_futuro(frame, ...).iloc[:, 0] con su nombre), de forma que los dos experimentos entreguen exactamente el mismo tipo que la llamada de referencia. Y en _comparar, comparar los conjuntos de columnas y lanzar ErrorDatos (o como minimo enumerar las columnas no auditadas) cuando no coincidan, en lugar de intersecar callando: hoy ese desajuste es indistinguible de 'no hay fuga'.

#### [medio · fuga_futuro] Las fugas condicionadas a los datos (interpolate/bfill de huecos) pasan la auditoria con los cortes por omision

`/home/user/ICM/src/icm/ml/caracteristicas.py` línea 598 · reproducido

Los dos experimentos solo revelan una fuga si su efecto alcanza a alguna fila anterior a uno de los cortes. Un relleno hacia atras o una interpolacion solo actua en las fechas con hueco: para una fila t sin hueco cercano los tres calculos (completo, truncado, perturbado) dan el mismo numero, porque el valor que se arrastra es anterior al corte. Con los tres cortes por omision de _fechas_de_corte (50/70/85 %) basta con que ningun hueco caiga pegado a un corte para que la fuga sea invisible, y sin embargo el mensaje de error del propio modulo (lineas 684-685) anuncia 'un relleno hacia atras' como uno de los fallos que busca. Peor aun: el caso habitual es interpolar ANTES de llamar, y entonces la fuga ya viene cocinada en 'precios' y no hay eleccion de cortes que la detecte, porque la auditoria audita al constructor, no a la entrada. Esto no es un matiz teorico: en series de energia los huecos por festivos son la norma y interpolar es el reflejo automatico.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/huecos.py: con huecos en las posiciones 120, 121, 200 y 300 y un constructor que hace precios[objetivo].bfill(), rend_ret0 del 2015-06-18 pasa de 0.00506266 a 0.26742693 al multiplicar por 1.30 el precio de DOS DIAS DESPUES (fila 122): fuga real. La auditoria con cortes por omision PASA; con fechas=[indice[120]] o [indice[121]] la detecta; con [indice[118]], [indice[119]] o [indice[250]] vuelve a pasar. Y en /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/kwargs_y_previo.py: comprobar_ausencia_de_fuga(construir_matriz, precios.interpolate(), 'ALFA') PASA mientras rend_ret1 del 2015-05-22 pasa de -0.006773 a 0.072223 al cambiar un precio dos dias posterior.

**Arreglo propuesto.** Dos cosas. Primero, subir la potencia: por omision usar muchos mas cortes (por ejemplo todos los de la segunda mitad, que en 500 fechas cuesta segundos; el script /tmp/.../exhaustivo.py hace 193 cortes consecutivos de construir_matriz en unos pocos segundos) y, si el indice de entrada trae NaN, incluir ademas un corte en cada hueco y en el dia anterior. Segundo, decir en el docstring, con las mismas letras que el resto de las advertencias del modulo, que la auditoria solo cubre lo que el constructor hace con los datos que recibe: cualquier relleno, interpolacion o reindexacion aplicada antes de la llamada queda fuera de su alcance.

#### [bajo · fuga_futuro] _perturbar_futuro solo altera valores, nunca el indice: las variables que leen el calendario futuro de la muestra son invisibles

`/home/user/ICM/src/icm/ml/caracteristicas.py` línea 626 · reproducido

El experimento 2 reescribe valores dejando el indice intacto, y el experimento 1 pierde por el dropna la unica fila que delataria un adelanto de un periodo. Por eso una variable que use la POSICION de las observaciones futuras en lugar de sus valores (dias hasta la siguiente observacion, si manana hay cotizacion, cuantas fechas quedan en la muestra) atraviesa las dos pruebas sin dejar rastro. Es menos frecuente que los casos anteriores, pero en series de energia con calendario irregular 'cuando vuelve a haber dato' no es informacion neutra.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/enganio.py, constructor c_indice_futuro con X['dias_al_siguiente'] = -indice.to_series().diff(-1).dt.days: la auditoria PASA. En el mismo script los seis intentos basados en valores (media centrada, tipificacion global, cuantiles globales discretizados, rango min-max global, rango percentil global y maximo acumulado invertido) se detectan todos.

**Arreglo propuesto.** Anadir al experimento 2 una variante que ademas de los valores altere la parte futura del indice (por ejemplo desplazando las fechas posteriores al corte unos dias, o eliminando algunas), de modo que una variable que dependa del calendario posterior cambie. Si se considera fuera de alcance, dejarlo escrito en el docstring de comprobar_ausencia_de_fuga junto a la limitacion del preprocesado previo.

#### [bajo · caso_limite] Casos limite normales que salen por ValueError opaco en vez de ErrorDatos

`/home/user/ICM/src/icm/ml/caracteristicas.py` línea 559 · reproducido

Tres entradas perfectamente plausibles rompen con una excepcion de pandas que no dice que hacer, mientras el resto del modulo se esfuerza en dar mensajes accionables. (a) Un solo activo con incluir=('cruzadas',): 'otros' queda vacio, no se agrega ningun bloque y pd.concat(bloques) de la linea 559 revienta con 'No objects to concatenate'; el control de la linea 506 no lo cubre porque 'incluir' no esta vacio. (b) 'precios' con nombres de columna repetidos: marco[objetivo] devuelve un DataFrame y bool((serie <= 0).any()) de la linea 512 lanza 'The truth value of a Series is ambiguous'. (c) 'exogenas' con indice duplicado: exo.reindex(marco.index) de la linea 557 lanza 'cannot reindex on an axis with duplicate labels'; _validar_indice_temporal se aplica a 'precios' (linea 492) pero nunca a 'exogenas'. Comprobado aparte que un indice de exogenas DESORDENADO si se alinea bien, porque reindex trabaja por etiqueta, asi que ahi no hay fuga.

**Reproducción.** /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/bordes.py, bloque 5: [ValueError] un solo activo con incluir=('cruzadas',): No objects to concatenate; [ValueError] precios con columnas duplicadas: The truth value of a Series is ambiguous; [ValueError] exogena con indice duplicado: cannot reindex on an axis with duplicate labels. En contraste, serie constante (336, 8), exogena desordenada (489, 7) e incluir=() con exogenas (499, 2) funcionan.

**Arreglo propuesto.** En construir_matriz: si bloques queda vacio, lanzar ErrorDatos diciendo que 'cruzadas' necesita al menos dos activos; validar en _como_marco (o en construir_matriz) que las columnas de 'precios' no esten repetidas, como ya se hace con las de X en la linea 560; y llamar a _validar_indice_temporal(exo.index, 'exogenas') antes del reindex.

### src/icm/energia/curvas.py

**Veredicto.** El camino por defecto (metodo='ratio') esta solido: he verificado con un fuzz de 400 configuraciones (indices con huecos, dias habiles, vencimientos en fin de semana, antes y despues del indice, dias_antes de 0 a 11) que el rendimiento de la serie coincide exactamente con el del contrato activo, que la serie termina en el precio de mercado, que la cadena nunca retrocede de vencimiento, que total = precio + roll y que el roll es cero fuera de los empalmes salvo ruido de 1e-16; pendiente_curva tambien es correcta con contratos ausentes, con plazos repetidos y con filas vacias. Los fallos estan concentrados en el ajuste aditivo y en el tratamiento de los precios negativos: con metodo='diferencia' los rendimientos relativos se calculan sobre un nivel desplazado por los rolls futuros, lo que introduce fuga de informacion del futuro (los rendimientos de una fecha pasada cambian al llegar dato nuevo) y un sesgo al alza de 37 pb en 3 meses, y ni descomponer_rendimiento ni serie_continua protegen el divisor frente a niveles negativos, que es precisamente el caso para el que existe ese metodo. A ello se suman el signo invertido de rendimiento_roll con el lejano negativo -- que contradice a estado_curva sobre los mismos datos -- y que serie_continua fabrica una serie completa con un contrato ya vencido si el calendario esta obsoleto.

#### [alto · fuga_futuro] Con metodo='diferencia', los rendimientos pasados cambian cuando llega dato futuro

`src/icm/energia/curvas.py` línea 665 · reproducido

El ajuste aditivo de cada tramo es la suma acumulada de TODOS los escalones posteriores, asi que el nivel de una fecha pasada depende de cuantos rolls futuros haya. Con el ajuste 'ratio' eso es inocuo porque un factor multiplicativo constante se cancela en el cociente precio_t/precio_{t-1}; con el aditivo NO se cancela: aparece en el denominador. Resultado: descomponer_rendimiento devuelve, para la MISMA fecha historica, un rendimiento distinto segun cuanto futuro se le haya dado. Un backtest ejecutado hoy no reproduce los rendimientos que habria visto en su momento, y los reprocesados mejoran: es fuga de informacion del futuro en la serie de rendimientos, no solo en el nivel. Las pruebas no lo detectan porque test_el_rendimiento_total_es_la_suma_del_precio_y_del_roll solo comprueba la identidad total=precio+roll, que se cumple para cualquier ajuste.

**Reproducción.** /tmp/rev/e_futuro.py con la cadena en contango de las propias pruebas (M1/M2/M3, VENCIMIENTOS_CADENA, dias_antes=5). Se descompone la serie completa (hasta 31-mar) y la truncada en 10-feb (antes del 2o roll) y se comparan las fechas comunes:
  ratio       max|d total|=1.8e-16  max|d precio|=1.8e-16  max|d roll|=1.3e-16
  diferencia  max|d total|=5.3e-05  max|d precio|=2.9e-03  max|d roll|=3.0e-03
El roll del 2024-01-26 (dia de empalme) vale -0.053654 con datos hasta marzo y -0.056604 con datos hasta el 10-feb: 30 pb de diferencia (un 5,5 % relativo) sobre una fecha que ya era pasado en ambos casos.

**Arreglo propuesto.** Para el ajuste aditivo, los rendimientos relativos no se pueden calcular sobre el nivel ajustado. Calcular la descomposicion sobre el precio SIN ajustar del contrato activo: total_t = (crudo_t - crudo_{t-1}^{mismo contrato}) / crudo_{t-1}, es decir reconstruir el crudo como precio - ajuste y dividir por crudo_{t-1} (no por precio_{t-1}); o bien devolver la descomposicion en unidades absolutas (EUR) cuando metodo='diferencia' y documentar que ahi no hay rendimiento porcentual bien definido.

#### [alto · caso_limite] Los rendimientos se dividen por el nivel anterior sin comprobar que sea positivo: signo invertido

`src/icm/energia/curvas.py` línea 649 · reproducido

descomponer_rendimiento solo protege el caso anterior == 0 y despues divide por 'anterior' tal cual. Si el nivel de la vispera es negativo -- el caso para el que existe metodo='diferencia' segun el propio docstring de serie_continua (spark spread, precio electrico negativo) -- el cociente sale con el signo invertido: una serie que sube todos los dias devuelve rendimientos negativos, y en el cruce por cero devuelve valores de cientos por cien. No es exclusivo del ajuste aditivo: serie_continua con metodo='ratio' solo exige precios positivos EN LAS FECHAS DE EMPALME (_ajustes_hacia_atras, lineas 355-361), asi que un precio negativo en medio de un tramo pasa la validacion y llega igualmente al divisor. Cualquier agregacion posterior (suma, producto de 1+r, Sharpe) queda con el signo invertido sin ningun aviso.

**Reproducción.** /tmp/rev/c_spread.py: spark spread S1 de -6 a +2 lineal, S2 = S1+4, metodo='diferencia'. El nivel sube TODOS los dias y rendimiento_total sale negativo en 5 de 19 fechas (-0.2105, -0.2667, -0.3636, -0.5714, -1.3333) y +4.0000 el dia del cruce por cero; el roll del empalme sale -2.92 (-292 %).
/tmp/rev/h_final.py caso C: con metodo='ratio' y precios negativos solo fuera del ancla, serie_continua no protesta y los 4 primeros rendimientos son [-0.40, -0.667, -2.00, +0.50] sobre un precio que sube siempre (3 de 19 fechas con precio al alza y rendimiento negativo).

**Arreglo propuesto.** Exigir (anterior > 0).all() en descomponer_rendimiento con el mismo mensaje que ya usa para los ceros, y en serie_continua con metodo='ratio' validar la positividad de toda la serie activa (no solo de las fechas de empalme). Para series que cruzan el cero, devolver la descomposicion en variacion absoluta en lugar de en tanto por uno.

#### [medio · estadistico] rendimiento_roll invierte el signo si el contrato lejano cotiza negativo, y contradice a estado_curva

`src/icm/energia/curvas.py` línea 411 · reproducido

El carry se calcula como (cercano - lejano)/lejano, con el lejano CON SIGNO en el denominador. estado_curva, en cambio, divide por precio_lejano.abs() (linea 437) precisamente para no invertir la clasificacion. Con precios electricos negativos las dos funciones dicen lo contrario sobre los mismos datos: estado_curva clasifica backwardation (el cercano es el caro) y rendimiento_roll devuelve carry negativo, cuando el docstring afirma 'Es positivo en backwardation'. La prueba test_el_signo_del_carry_coincide_con_el_estado_de_la_curva asegura esa coherencia, pero solo con precios positivos. La nota del docstring advierte de que la razon relativa 'pierde sentido' con el lejano negativo, pero el codigo no avisa ni corrige: devuelve un numero plausible con el signo cambiado.

**Reproducción.** /tmp/rev/h_final.py caso A, cercano=-2, lejano=-8, 30 dias: estado_curva='backwardation' y rendimiento_roll=-9.1313; con |lejano| en el denominador sale +9.1312. Con la serie completa (/tmp/rev/a_signos.py) los tres dias se contradicen: estados ['backwardation','contango','backwardation'] frente a carries [-9.13, +48.70, -24.35].

**Arreglo propuesto.** Usar precio_lejano.abs() en el denominador, igual que estado_curva, o rechazar explicitamente los lejanos negativos con ErrorDatos remitiendo al diferencial absoluto. Ampliar la prueba de coherencia carry/estado a un par con precios negativos.

#### [medio · estadistico] Con metodo='diferencia', rendimiento_total no es el rendimiento de la posicion y rendimiento_precio no es el encadenado sin ajustar

`src/icm/energia/curvas.py` línea 668 · reproducido

El docstring promete que rendimiento_total es 'lo que realmente gana la posicion' y rendimiento_precio 'la variacion del precio cotizado encadenado sin ajustar'. Con el ajuste aditivo ninguna de las dos cosas es cierta, porque el denominador es el nivel desplazado y no el precio crudo del contrato. En contango el desplazamiento infla los niveles pasados, encoge la magnitud de los rendimientos antiguos y deja un sesgo sistematico al alza en la rentabilidad acumulada. Es el mismo origen que el hallazgo de fuga de futuro, pero se mide aparte: aqui no hace falta reprocesar nada, la cifra ya es incorrecta. Ninguna prueba lo cubre: la propiedad 'pct_change == rendimiento del contrato activo' solo se comprueba con metodo='ratio' (test_la_serie_ajustada_reproduce_el_rendimiento_verdadero_en_todas_las_fechas), y para 'diferencia' solo se comprueban variaciones absolutas.

**Reproducción.** /tmp/rev/e_futuro.py y /tmp/rev/b_retornos.py con la cadena en contango de las pruebas (decaimiento 0,999 diario, 90 dias):
  acumulado reportado (1+total).prod() = 0.917595
  acumulado verdadero  (0.999**90)     = 0.913890  -> 37 pb de rentabilidad inventada en 3 meses (~1,5 % anual)
  max|pct_change - rendimiento del contrato activo| = 1.06e-04 diario (con 'ratio': 1.1e-16)
  max|rendimiento_precio - encadenado sin ajustar| = 6.2e-03 (con 'ratio': 1.5e-16)

**Arreglo propuesto.** Mismo arreglo que el hallazgo de fuga de futuro: para metodo='diferencia' dividir por el precio crudo del contrato activo (precio - ajuste) en lugar de por el nivel ajustado, o devolver la descomposicion en unidades absolutas. Anadir a las pruebas, parametrizada por metodo, la comprobacion de que el rendimiento de la serie coincide fecha a fecha con el del contrato activo.

#### [medio · caso_limite] serie_continua construye la serie con un contrato ya vencido si el calendario esta obsoleto

`src/icm/energia/curvas.py` línea 296 · reproducido

La cadena se decide solo con las fechas de roll (fechas_roll[:-1].searchsorted), y nunca se comprueba que el contrato activo siga vivo en cada fecha. Si al diccionario de vencimientos le faltan los contratos nuevos -- el olvido tipico al refrescar un calendario -- el ultimo de la cadena 'aguanta hasta el final' aunque venciera hace meses, y serie_continua devuelve una serie completa, con ajuste neutro y sin ninguna advertencia, construida sobre cotizaciones de un contrato inexistente (tipicamente el ultimo precio arrastrado por el proveedor). La incoherencia es visible dentro del propio modulo: dias_a_vencimiento sobre ese mismo calendario devuelve NaN en todas las fechas ('falta calendario, no datos'), mientras serie_continua no se queja.

**Reproducción.** /tmp/rev/h_final.py caso B: indice de 20 dias habiles de febrero de 2024 y vencimientos {'F1':'2023-01-01','F2':'2023-02-01'}. dias_a_vencimiento devuelve 0 fechas no-NaN de 20; serie_continua devuelve 20 precios del contrato F2, vencido el 2023-02-01, con ajuste = 1.0 y sin error. Caso 2 de /tmp/rev/f_bordes.py igual. (El resto de los casos de calendario si es correcto: vencimiento en sabado -> se rueda el lunes siguiente; vencimientos posteriores al indice -> no se rueda nunca; ambos verificados.)

**Arreglo propuesto.** Avisar o fallar cuando la fecha supere el vencimiento del contrato activo por mas de un margen (por ejemplo, ErrorDatos si el ultimo vencimiento es anterior a la ultima fecha del indice), remitiendo a completar 'vencimientos'. Alternativamente devolver NaN en esas fechas, igual que hace dias_a_vencimiento.

#### [bajo · alineacion] El metodo de ajuste se adivina desde .attrs: si se pierde la marca, el reparto precio/roll sale 19 veces mayor sin error

`src/icm/energia/curvas.py` línea 632 · reproducido

descomponer_rendimiento lee el metodo de marco.attrs y, a falta de marca, supone 'ratio'. attrs no sobrevive a un viaje por disco (to_csv/read_csv, parquet) ni a varias operaciones de pandas, y guardar la serie continua para descomponerla despues es un uso natural. Con una serie aditiva en contango cuyo tramo final no este incluido, todos los ajustes son positivos, asi que la validacion 'el ajuste multiplicativo debe ser positivo' no salta y la descomposicion se calcula con la formula equivocada en silencio. El total acumulado no cambia (solo depende de 'precio'), pero el reparto entre precio y roll -- que es justo para lo que sirve la funcion -- queda destrozado.

**Reproducción.** /tmp/rev/g_attrs.py: serie aditiva de la cadena en contango recortada a 2024-01-05..2024-02-10 (cruza el primer empalme, ambos ajustes > 0) y con attrs vaciado. rendimiento_roll del dia del empalme: -0.053654 con metodo='diferencia' explicito frente a -1.028464 al caer al defecto 'ratio' (19 veces mayor), sin ningun error. Nota: con pandas 3.0.6 attrs si se propaga por .loc, .copy(), pd.concat y .reindex, asi que hace falta una perdida real de los metadatos.

**Arreglo propuesto.** No adivinar: guardar el metodo en una columna de la propia salida de serie_continua (o exigir el argumento metodo cuando attrs no lo traiga) en lugar de suponer 'ratio'.

### src/icm/ml/redes.py (pruebas: tests/test_ml_redes.py)

**Veredicto.** Las seis propiedades del encargo se cumplen y las he verificado numericamente una por una: las ventanas solo miran atras (perturbar la fila k cambia exactamente las predicciones k..k+ventana-1 y ninguna anterior), media y desviacion se estiman solo con las filas del tramo de entrenamiento (media_X identica a la de las filas [0, filas_entrenamiento) y distinta de la de toda la serie), predecir reutiliza esa normalizacion sin reajustarla, la validacion interna es el ultimo tramo temporal sin solape de filas con el entrenamiento (ultima fila de entrenamiento 156, primera del primer bloque de validacion 157), los pesos restaurados son los de la mejor epoca (perdida recalculada = 1.1056629 = minimo del historial, frente a 1.1160113 de la ultima) y las predicciones son identicas al bit con la misma semilla, incluso variando el numero de hilos de torch y entre procesos. El diseno antifuga del nucleo es solido. Los fallos que quedan estan en los bordes: falta la comprobacion de indice creciente que si hacen sus modulos hermanos (via de fuga real y reproducida), la restauracion de pesos puede tirar el entrenamiento completo en silencio cuando ninguna epoca mejora, y la configuracion TCN por defecto ignora 27 de las 30 fechas de la ventana sin avisar.

#### [alto · fuga_futuro] redes.py no exige indice creciente ni sin duplicados; con la tabla ordenada al reves la ventana mira al futuro

`/home/user/ICM/src/icm/ml/redes.py` línea 540 · reproducido

_preparar solo comprueba longitudes, finitud y que X e y compartan indice; nunca comprueba que el indice sea monotono creciente ni que no tenga fechas repetidas. Las ventanas se construyen por POSICION (_ventanas, linea 416), asi que si el llamante pasa la tabla de mas reciente a mas antiguo -lo habitual al bajar datos de casi cualquier proveedor- la ventana 'hacia atras' en posicion es hacia DELANTE en el calendario y el modelo predice la fecha t con datos de t+1..t+ventana-1. Esto contradice la convencion explicita del propio repositorio: icm/ml/validacion.py:66-79 e icm/ml/caracteristicas.py:143-155 rechazan con ErrorDatos tanto el indice duplicado como el no creciente, y sus mensajes hablan justamente de fuga del futuro ('con las filas desordenadas un retardo puede traer un valor posterior en el tiempo'). La cabecera del modulo presume de cerrar las tres fugas clasicas de las redes recurrentes; esta cuarta queda abierta y ninguna prueba la cubre, porque todos los datos sinteticos de tests/test_ml_redes.py se generan con pd.bdate_range ya ordenado.

**Reproducción.** Script /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/s10.py: objetivo construido para depender SOLO del futuro, y[t] = x[t+1]+...+x[t+5], con n=500, ventana=5, lstm, 20 epocas, semilla 7. Con el indice creciente (correcto) el objetivo es incausable y la red no aprende nada: RMSE 2.2585 frente a 2.2540 de predecir la media. Pasando exactamente los mismos datos con X.iloc[::-1] e y.iloc[::-1] (indice decreciente, is_monotonic_increasing = False) ajustar() no se queja, red.ajustada = True, y el RMSE baja a 1.0090: la red aprende una relacion que solo existe hacia el futuro. Ademas, script s6.py caso 5: una tabla con el indice ENTERAMENTE duplicado (todas las filas con la misma fecha) tambien se ajusta sin ninguna queja.

**Arreglo propuesto.** En _preparar (antes de devolver marco y objetivo), replicar las dos comprobaciones de icm/ml/validacion.py:66-79 sobre marco.index cuando la entrada sea pandas: rechazar con ErrorDatos si indice.has_duplicates y si not indice.is_monotonic_increasing, con el mismo texto que ya usan validacion.py y caracteristicas.py ('Ordenelo con .sort_index()'). Conviene hacerlo tambien en _alinear_columnas, porque predecir sufre el mismo problema por si solo. Y anadir dos pruebas: una tabla con indice descendente y otra con fechas repetidas deben elevar ErrorDatos.

#### [medio · caso_limite] Si ninguna epoca mejora, se restauran los pesos ALEATORIOS iniciales y se pierde todo el entrenamiento sin aviso

`/home/user/ICM/src/icm/ml/redes.py` línea 641 · reproducido

mejor_perdida arranca en inf y mejores_pesos se inicializa con copy.deepcopy del state_dict de la red RECIEN CREADA, es decir, con los pesos aleatorios sin entrenar. La actualizacion es 'if perdida_validacion < mejor_perdida' (linea 677), comparacion que es False cuando perdida_validacion es inf o NaN. Si eso ocurre en todas las epocas, mejores_pesos nunca se actualiza y el load_state_dict de la linea 689 devuelve la red a su inicializacion aleatoria, tirando el entrenamiento entero. No se eleva ningun error ni se emite ningun aviso: 'ajustada' es True, predecir devuelve numeros finitos de aspecto normal, y la columna perdida_entrenamiento del historial muestra una convergencia impecable. La proteccion de _como_marco/_preparar no sirve aqui porque solo rechaza ausentes e infinitos, y el detonante es un valor finito pero extremo en el tramo reservado de y: al estandarizarse con la escala del entrenamiento y elevarse al cuadrado en float32 desborda a inf. El detonante es estrecho (hace falta |z| del orden de 1.8e19 en la cola), pero la ruta de codigo es incorrecta en si misma: 'ninguna epoca mejoro' deberia ser un error o, como minimo, conservar los ultimos pesos, nunca los de la inicializacion.

**Reproducción.** Script /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/s7.py, con los mismos datos y la misma configuracion que el fixture de las pruebas (suma de la ventana, n=500, ventana=5, lstm, 16 unidades, 20 epocas, lr 0.05, semilla 7). Con y limpia: RMSE 0.0764. Cambiando UN solo valor, y.iloc[-5] = 1e21 (dentro del 20 % reservado a validacion): RMSE 2.2773, peor que predecir la media (2.2486), mientras perdida_entrenamiento va de 0.4202 a 0.0019 en el historial y perdida_validacion vale inf en las 20 epocas. Comparando estado por estado contra una red creada con _crear_red y la misma semilla sin entrenar: 'pesos devueltos == red sin entrenar' = True (igualdad exacta tensor a tensor). Descartado por medicion un segundo detonante que sospechaba (divergencia del optimizador): script s11.py con lr de 5e2 a 1e7 nunca produce NaN con Adam, las perdidas se quedan en 5.5e5..2.2e14 y los pesos si se actualizan.

**Arreglo propuesto.** Inicializar mejores_pesos a None y, tras el bucle, hacer 'if mejores_pesos is not None: red.load_state_dict(mejores_pesos)'; y si ninguna epoca dio una perdida finita, elevar ErrorDatos explicando que la perdida de validacion no fue finita en ninguna epoca (probable valor extremo en el tramo reservado de y) en lugar de devolver una red sin entrenar. Alternativa complementaria: comprobar np.isfinite(perdida_validacion) al calcularla y abortar con mensaje. Prueba de regresion: ajustar con un valor extremo en la cola de y debe fallar con ErrorDatos, no devolver pesos iguales a los de _crear_red.

#### [medio · caso_limite] La TCN por defecto ignora 27 de las 30 fechas de la ventana y nadie comprueba alcance_tcn

`/home/user/ICM/src/icm/ml/redes.py` línea 362 · reproducido

ConfiguracionRed por defecto trae ventana=30 y capas=1, asi que ConfiguracionRed(arquitectura='tcn') tiene alcance_tcn = (3-1)*(2**1-1)+1 = 3: de las 30 fechas de cada muestra solo las 3 ultimas pueden influir en la prediccion. La propiedad alcance_tcn existe con el docstring 'Sirve para detectar la incoherencia mas comun de esta arquitectura', pero ni __post_init__ ni _crear_red la consultan nunca, asi que la incoherencia no se detecta: el modelo se ajusta, predice, y sigue marcando con NaN las primeras 29 filas como si de verdad usara 30 fechas de historia. El coste es doble: el usuario cree tener una memoria de 30 dias y tiene 3, y encima paga 29 filas de arranque sin prediccion por una historia que la red no mira. La prueba test_el_alcance_de_la_tcn_crece_de_forma_exponencial_con_las_capas solo comprueba la aritmetica de la propiedad, no que se use para nada.

**Reproducción.** Script /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/s9.py: arquitectura tcn, ventana=30, capas=1, n=300, semilla 2. Perturbando la fila t-k en +50 y midiendo el cambio de la prediccion en t: t-0 |delta|=1.676e+01, t-1 |delta|=4.431e+01, t-2 |delta|=1.488e+01, y a partir de ahi t-3, t-5, t-10 y t-29 dan |delta| = 0.000e+00 EXACTAMENTE. Es decir, 27 de las 30 columnas de cada muestra no influyen en nada, y el ajuste no emite aviso alguno.

**Arreglo propuesto.** En __post_init__ (o al principio de ajustar), si arquitectura == 'tcn' y alcance_tcn < ventana, emitir un warnings.warn -o elevar ErrorDatos, que encaja mejor con el estilo del resto del modulo- indicando el alcance efectivo, cuantas fechas de la ventana se descartan y las capas necesarias para cubrirla (ceil(log2(1 + (ventana-1)/(NUCLEO_TCN-1)))). Prueba de regresion: ConfiguracionRed(arquitectura='tcn', ventana=30, capas=1) debe avisar, y capas=5 (alcance 63) no.

#### [bajo · caso_limite] _tramos puede dejar el entrenamiento en UNA sola muestra sin avisar, y _escala_y degenera a 1

`/home/user/ICM/src/icm/ml/redes.py` línea 553 · reproducido

La unica guarda de tamano es n_muestras < 2 (linea 602), que mira el total. Despues _tramos resta validacion y embargo y, por construccion, solo garantiza n_entrenamiento >= 1. Con una ventana grande respecto a la serie, o con proporcion_validacion alta, el tramo de entrenamiento se queda en una sola ventana: la red entrena con un unico ejemplo, _media_y es ese unico objetivo y _escala_y colapsa a 1.0 porque _escala sustituye la desviacion nula (correcto en si mismo, pero aqui deja la escala del objetivo sin estimar). Todo ocurre en silencio: ajustada = True, el historial tiene sus epocas y predecir devuelve numeros. El mensaje de error de la linea 603 ('Hacen falta al menos ventana+1 filas') promete que con ventana+1 filas se puede ajustar, cuando lo que sale de ahi es un ajuste degenerado.

**Reproducción.** Script /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/s6.py. Caso 1: ventana=30 con 40 filas (n_muestras=11) -> _tramos devuelve (1, 2), _escala_y = 1.0, el ajuste completa 3 epocas y predice sin ninguna queja. Caso 2: proporcion_validacion=0.99 con 30 filas y ventana=4 (n_muestras=27) -> _tramos devuelve (1, 26): una muestra de entrenamiento y 26 de validacion, tambien sin aviso.

**Arreglo propuesto.** Elevar ErrorDatos en ajustar cuando n_entrenamiento sea menor que un minimo razonable (por ejemplo 2, o mejor tamano_lote o un multiplo pequeno de ventana), diciendo cuantas muestras de entrenamiento han quedado, cuantas se ha llevado el embargo y que se reduzca ventana o proporcion_validacion. Y ajustar el mensaje de la linea 603 para que no prometa que ventana+1 filas bastan.

#### [bajo · alineacion] Con proporcion_validacion=0, la perdida que elige la mejor epoca se mide con los pesos de antes de los pasos del optimizador

`/home/user/ICM/src/icm/ml/redes.py` línea 660 · solo sospecha

perdida_entrenamiento es la media ponderada de las perdidas de los minilotes, y cada una se calcula ANTES de su optimizador.step() (linea 656-658): describe los pesos que la red tenia DURANTE la epoca, no los que tiene al terminarla. Con proporcion_validacion > 0 esto no afecta a la parada temprana, porque perdida_validacion se recalcula despues de la epoca con los mismos pesos que se guardan (verificado: coincide al bit con el minimo del historial). Pero con proporcion_validacion = 0 la linea 667 usa esa perdida de entrenamiento como criterio, de modo que los pesos guardados en la epoca k se acreditan con una perdida que corresponde a un estado anterior: hay un desfase de una epoca entre el numero que decide y los pesos que se conservan. Se suma que esa perdida se mide en modo train, con el abandono activo, asi que con abandono > 0 el criterio es ademas ruidoso. No he conseguido construir un caso en el que el desfase cambie el resultado, porque en este modo la perdida de entrenamiento suele decrecer de forma monotona y la parada temprana casi nunca se dispara; lo dejo como SOSPECHA.

**Reproducción.** solo sospecha. Script /tmp/claude-0/-home-user-ICM/0e2d14cf-20b5-5e4d-8c38-84278d2af30e/scratchpad/s8.py: con proporcion_validacion=0, abandono=0, 30 epocas y datos sin senial, comparo epoca a epoca la perdida registrada con la perdida REAL en modo eval de los pesos de esa epoca (reentrenando truncado). Las dos series difieren sistematicamente (epoca 1: 1.036897 registrada frente a 0.990781 real; epoca 30: 0.137365 frente a 0.113718), lo que confirma el desfase, pero en este caso los dos argmin coinciden (epoca 30) y la perdida real de los pesos restaurados (0.113718) es la mejor posible, asi que NO hay dano demostrado.

**Arreglo propuesto.** Cuando n_validacion == 0, recalcular la perdida de entrenamiento tras la epoca con red.eval() y torch.no_grad() sobre X_entrenamiento completo, y usar ese valor como criterio de la parada temprana (dejando la media de minilotes solo como cifra informativa del historial, o anadiendola como tercera columna). Asi el numero que decide y los pesos que se guardan corresponden al mismo estado, y el criterio deja de depender del abandono.

### src/icm/ml/modelos.py (y tests/test_ml_modelos.py)

**Veredicto.** De los cinco puntos de la lista, dos salen limpios y verificados con instrumentacion, no por lectura: el escalado esta realmente dentro del Pipeline y se reajusta en cada particion usando solo el tramo de entrenamiento (espia sobre StandardScaler: medias 219/279/339 filas, -0.0756 / -0.1038 / 5.6083, nunca la media global 12.35, tanto en buscar_hiperparametros como en comparar_modelos), y ParticionSklearn no es un KFold disfrazado (los tamanios de entrenamiento que llegan a GridSearchCV son 219/279/339, no los 320 que daria un KFold de 5). Los otros tres fallan: buscar_hiperparametros elige los hiperparametros puntuando sobre los mismos tramos de prueba que despues se reportan como fuera de muestra (demostrado alterando y solo en el bloque de prueba: la eleccion salta de alpha 1e-6 a 1e+6 con el entrenamiento intacto), importancia_variables permuta dentro de muestra en silencio cuando el modelo llega sin ajustar y eso fabrica importancias de 0.28 de R cuadrado sobre ruido puro (que es exactamente la llamada del notebook 03), y el guardian de alineacion de _alinear se desactiva si 'y' es un DataFrame de una columna, con lo que se puede ajustar con un objetivo de otro periodo sin ningun aviso. MediaCero cumple lo que se le pedia (clone, get_params, set_params, GridSearchCV) pero no fija n_features_in_, lo que la deja como el unico modelo del catalogo sin esos atributos. Las 56 pruebas del modulo pasan; los tres fallos estan en huecos que sus pruebas no cubren.

#### [alto · fuga_futuro] buscar_hiperparametros puntua la seleccion sobre los mismos tramos de prueba que luego se presentan como fuera de muestra

`src/icm/ml/modelos.py` línea 761 · reproducido

La funcion recibe (X, y) completos y construye GridSearchCV con cv=ParticionSklearn(particion). La particion walk forward pega sus tramos de prueba al FINAL de la muestra, es decir, exactamente las mismas fechas que comparar_modelos / validacion_walk_forward presentan despues como resultado fuera de muestra. La combinacion elegida es por tanto la que minimiza el error EN el tramo de prueba. El encadenamiento que propone el propio docstring (mejores = buscar_hiperparametros('ridge', X, y); modelo = crear_modelo('ridge', **mejores)) y despues validar con la misma particion es, literalmente, elegir hiperparametros mirando el conjunto de prueba. El docstring advierte de refit=False pero no de esto, que es el mismo problema en su forma mas clasica: refit solo afectaria al ajuste final, mientras que la SELECCION ya se ha contaminado. No existe ninguna reserva de un tramo final ni particion anidada.

**Reproducción.** Script e2b.py: X = ruido, y = 0.02*a + ruido, n=400. Se altera y SOLO en el tramo de prueba de la ultima particion (posiciones que no entran en ningun entrenamiento; se comprueba np.allclose sobre el resto: entrenamiento identico = True). Con n_particiones=1/tamano_prueba=100: y base -> {'alpha': 1e-06}; y alterada -> {'alpha': 1000000.0}. Con n_particiones=3/tamano_prueba=60 y alterando solo el ultimo bloque (340..399): igual, 1e-06 -> 1e+06. La eleccion la decide dato que ninguna particion usa para entrenar. Script e2d.py cuantifica el optimismo con ruido puro y la rejilla por defecto de arbol_potenciado sobre 6 semillas: MSE/MSE(media_cero) = 1.1295 eligiendo con la misma particion frente a 1.1722 eligiendo solo con fechas anteriores al primer tramo de prueba (mejor en 4 de 6 semillas, nunca peor).

**Arreglo propuesto.** Que buscar_hiperparametros no vea el tramo que luego se reporta: (a) recortar internamente X, y a las fechas anteriores al primer inicio_prueba de la particion recibida y buscar con una particion interna sobre ese tramo, o (b) exigir que la busqueda reciba explicitamente el tramo de seleccion (X_seleccion, y_seleccion) y documentar que no puede ser el mismo que el de evaluacion, o (c) hacer la busqueda dentro de cada particion de validacion_walk_forward (seleccion anidada, un GridSearchCV por tramo de entrenamiento). Como minimo, dejar escrito en el docstring que los tramos de prueba de 'particion' quedan quemados para el informe.

#### [alto · fuga_futuro] importancia_variables hace la permutacion DENTRO de muestra en silencio cuando el modelo llega sin ajustar, y eso fabrica importancias enormes sobre ruido puro

`src/icm/ml/modelos.py` línea 620 · reproducido

El bloque check_is_fitted / except NotFittedError -> modelo.fit(X, y) ajusta sobre (X, y) y a continuacion permuta sobre ESE MISMO (X, y). La cifra resultante no es importancia predictiva, es cuanto memorizo el modelo. No hay ningun aviso, ninguna marca en la Series devuelta y ninguna comprobacion de orden temporal (a diferencia de ajustar_y_predecir, que si tiene _contrastar_orden_temporal). Ademas tampoco hay defensa en el otro sentido: con un modelo ya ajustado, nada impide pasarle un (X, y) que incluya su propio tramo de entrenamiento. La asimetria importa porque el uso real del repositorio es precisamente el inseguro: notebooks/03_trading_energia_ml.ipynb, celda 21, hace importancia_variables(crear_modelo('arbol_potenciado'), md_datos.X, md_datos.y, n_repeticiones=5) y titula el grafico 'Que mira el modelo'. La prueba test_la_importancia_senala_la_unica_variable_relevante usa la misma via, asi que el fichero de pruebas no distingue las dos cosas.

**Reproducción.** Script e4_importancia.py: X = 6 columnas de ruido, y = ruido independiente (no hay nada que explicar), n=400. Via del notebook (modelo sin ajustar, X e y completos), arbol_potenciado: {'x4': 0.2996, 'x1': 0.2754, 'x2': 0.2741, 'x5': 0.2737, 'x0': 0.2589, 'x3': 0.1983} -> afirma que cada variable aporta ~0.28 de R cuadrado. Via honesta (ajustado con 0:300, permutado en 300:400): {'x5': 0.0374, 'x4': 0.0197, 'x0': -0.027, 'x2': -0.0287, 'x1': -0.0405, 'x3': -0.0432}, es decir, ruido alrededor de cero como corresponde. Con bosque: 0.0659 maximo dentro de muestra frente a 0.0205 fuera.

**Arreglo propuesto.** Invertir el valor por defecto: exigir un modelo ya ajustado y lanzar ErrorDatos si llega sin ajustar, obligando a pasar dentro_de_muestra=True de forma explicita. Si se mantiene el ajuste automatico, marcarlo en el resultado (importancias.attrs['dentro_de_muestra'] = True) y emitir un aviso. Adicionalmente, cuando el modelo llega ajustado y expone feature_names_in_ o un indice de entrenamiento, comprobar que el (X, y) recibido no se solapa con el tramo de ajuste, igual que hace _contrastar_orden_temporal.

#### [alto · alineacion] El guardian de alineacion de _alinear solo actua si 'y' es una Series: con y = df[['col']] se ajusta cruzando fechas sin ningun aviso

`src/icm/ml/modelos.py` línea 164 · reproducido

_alinear compara indices solo dentro de 'if isinstance(X, (pd.DataFrame, pd.Series)) and isinstance(y, pd.Series)'. Un DataFrame de una columna (el error mas comun de todos: df[['rend_futuro_1']] en lugar de df['rend_futuro_1']) supera la comprobacion de longitud, no entra en la comparacion de indices y se empareja por posicion. La comprobacion es justo lo que el docstring de ajustar_y_predecir vende como su aportacion principal ('que X e y esten alineados de verdad y no solo tengan la misma longitud'). El agujero esta en las tres funciones que usan este _alinear: ajustar_y_predecir, buscar_hiperparametros e importancia_variables. comparar_modelos y validacion_walk_forward si quedan protegidos, pero por accidente: por la comprobacion valores_y.ndim != 1 de validacion.py, no por el alineado.

**Reproducción.** Script e6_alineacion.py, n=400, y = 0.02*a - 0.01*b + ruido. (A) y correcto como Series: coeficientes destipificados {'a': 0.0202, 'b': -0.0099}. (B) y de otro periodo como Series: ErrorDatos ('misma longitud pero indices distintos'), correcto. (D) EL MISMO y de otro periodo pero como DataFrame de una columna (y.iloc[50:350].to_frame()): aceptado sin aviso, coeficientes {'a': -0.0014, 'b': -0.0009, 'c': -0.0004, 'd': -0.0017}; la relacion real ha desaparecido y el modelo devuelve predicciones de aspecto normal. (E) buscar_hiperparametros con y desplazado 20 dias como DataFrame -> {'alpha': 1000000.0}; con el y correcto como Series -> {'alpha': 1e-06}.

**Arreglo propuesto.** En _alinear, aceptar tambien DataFrame en 'y': cambiar la condicion a isinstance(y, (pd.Series, pd.DataFrame)) para la comparacion de indices, y ademas rechazar un 'y' con mas de una dimension util con un ErrorDatos que diga 'seleccione una columna' (como ya hace validacion.py). Lo mismo en icm/ml/validacion.py::_alinear, que tiene la condicion identica.

#### [bajo · caso_limite] MediaCero no fija n_features_in_ ni feature_names_in_: es el unico modelo del catalogo sin ellos y no valida el numero de columnas al predecir

`src/icm/ml/modelos.py` línea 93 · reproducido

Lo que pedia la revision funciona: get_params() -> {}, clone() reproduce la instancia sin arrastrar n_variables_, set_params() devuelve self, is_regressor() es True, repr correcto y el Pipeline se clona bien, asi que MediaCero entra sin problemas en GridSearchCV y en validacion_walk_forward. Lo que falla es el contrato de atributos: fit guarda n_variables_ en lugar de n_features_in_, de modo que el Pipeline de 'media_cero' es el unico de los seis cuyo n_features_in_ y feature_names_in_ lanzan AttributeError tras ajustar. Cualquier codigo generico que recorra los modelos de una tabla y lea esos atributos (o que compare columnas antes de predecir) se rompe solo en la fila de la referencia. Como corolario, predict no comprueba que X tenga las columnas con las que se ajusto: siempre devuelve ceros, y sobre un vector de una dimension devuelve tantos ceros como elementos en vez de una prediccion.

**Reproducción.** Script e8.py: tras fit(X, y) con X de 3 columnas, ridge/lasso/elastic_net/arbol_potenciado/bosque dan n_features_in_=3 y feature_names_in_=['a','b','c']; media_cero da AttributeError en los dos. Script e3_mediacero.py: m.fit(X de 3 columnas) y despues m.predict(DataFrame de 7 columnas) devuelve [0., 0.] sin protestar. check_estimator(MediaCero(), on_fail=None) de sklearn 1.9.1: 52 comprobaciones, 16 fallidas, todas de validacion de entrada (check_n_features_in_after_fitting, check_fit2d_predict1d, check_estimators_nan_inf, check_regressors_train por X e y de longitudes distintas, etc.); ninguna de clonado ni de parametros.

**Arreglo propuesto.** En MediaCero.fit usar sklearn.utils.validation.validate_data(self, X, y, ...) (o al menos fijar self.n_features_in_ y self.feature_names_in_ ademas de n_variables_) y en predict llamar a validate_data(self, X, reset=False), que de paso hace que un vector de una dimension falle en lugar de devolver n predicciones.

#### [bajo · fuga_futuro] comparar_modelos y buscar_hiperparametros heredan en silencio embargo=0 cuando no se pasa 'particion'

`src/icm/ml/modelos.py` línea 824 · solo sospecha

Las dos funciones publicas hacen 'particion = particion if particion is not None else ParticionWalkForward()', y el valor por defecto de la clase es embargo=0. Con un objetivo adelantado (rend_futuro_h, que es lo que produce construir_matriz) la ultima fila de entrenamiento de cada particion tiene su objetivo medido ya dentro del tramo de prueba, que es precisamente el solapamiento que el docstring de ParticionWalkForward describe y para el que recomienda 'embargo = h'. Es decir, la via corta comparar_modelos(X, y) valida con el solapamiento que el modulo dice evitar; el notebook 03 lo hace bien porque pasa ParticionWalkForward(n_particiones=5, embargo=1) a mano.

**Reproducción.** Solo sospecha estructural. Verificado que el defecto es embargo=0 y que con el fin_entrenamiento de la particion 0 es el dia inmediatamente anterior a inicio_prueba (script e7_embargo.py). NO he conseguido convertirlo en una cifra: con y[t]=r[t+1], una variable muy autocorrelacionada y 20-40 particiones de 2-5 observaciones, el MASE del bosque frente a media_cero es 1.1993 con embargo=0 y 1.2033 con embargo=1 (e7b.py, incluso forzando min_samples_leaf=1 para que pueda memorizar): la regularizacion conservadora del catalogo no explota la fila solapada. Lo dejo como advertencia sobre el valor por defecto, no como fallo demostrado.

**Arreglo propuesto.** No usar ParticionWalkForward() sin argumentos como valor por defecto en estas dos funciones: o exigir 'particion' explicitamente, o derivar el embargo del horizonte del objetivo (por ejemplo leyendo y.name cuando sigue el patron rend_futuro_h) y documentar que la via corta usa embargo=1 como minimo.
