# Datos reales

Quince series **no simuladas**, 159 339 filas, descargadas el 25 de septiembre
de 2026. Hasta ahora todo el proyecto se había construido y validado contra
`icm.energia.simulacion`, así que esta carpeta es su primer contacto con datos
que no ha generado él mismo.

Las tres primeras son las que de verdad importan al proyecto, y vienen
directamente de los operadores del mercado español:

| Fichero | Contenido | Filas | Desde | Hasta |
|---|---|---:|---|---|
| `omie_intradiario.csv` | Precio marginal por **período de casación** del pool ibérico, €/MWh (ES y PT) | 102 143 | 2018-01-01 | 2026-09-25 |
| `omie_diario.csv` | Media aritmética diaria del pool ibérico, €/MWh | 3 185 | 2018-01-01 | 2026-09-25 |
| `mibgas_diario.csv` | Gas spot español `GDAES_D+1` en el PVB, €/MWh | 3 877 | 2016-01-06 | 2026-09-26 |
| `meteo_madrid.csv` | Temperatura diaria de Madrid: media, máxima y mínima, °C | 3 078 | 2018-01-01 | 2026-09-19 |

**El período no siempre dura una hora.** El 1 de octubre de 2025 OMIE pasó de
casar 24 períodos horarios al día a casar 96 cuarto-horarios, así que la misma
serie trae los dos regímenes: 2 813 días con 24 períodos y 355 con 96, más los
del cambio de hora (23 o 25 horarios, 92 o 100 cuarto-horarios). Por eso el
fichero no se llama «horario». El cargador lo deduce solo de cuántos períodos
tiene cada día:

```python
precios = cargar_precios_fichero(
    "datos/reales/omie_intradiario.csv",
    columna_hora="periodo", huso="Europe/Madrid",
)
```

El `huso` hace falta para el último domingo de octubre, que tiene 25 períodos:
sin él, las dos veces que dan las 02:00 comparten etiqueta y una desaparece.
Con él son dos instantes distintos y sobreviven las dos, que es lo correcto —
en 2018 costaron 52,40 y 50,12 €/MWh.

Comprobaciones de autenticidad que pasan: el máximo del pool es **700 €/MWh el
8 de marzo de 2022** (récord histórico tras la invasión de Ucrania) y la media
de ese día, 544,98 €/MWh, coincide al céntimo con la cifra que publicó OMIE —
que es además cómo se determinó que la sexta columna del fichero
`marginalpdbc` es España y la quinta Portugal. El máximo de MIBGAS es
240 €/MWh el 30 de agosto de 2022, el pico de la crisis del gas.

Los ficheros están **tal y como vienen del origen**, sin reformatear. Si alguno
no entra bien por el cargador, el arreglo va en `icm.datos.ficheros`, no en el
CSV: el objetivo es que la librería trague lo que hay en el mundo.

## Qué hay

| Fichero | Contenido | Filas | Desde | Hasta |
|---|---|---:|---|---|
| `brent_diario.csv` | Brent contado, USD/barril | 9 092 | 1987-05-20 | 2026-09-22 |
| `wti_diario.csv` | WTI contado, USD/barril | 9 507 | 1986-01-02 | 2026-09-22 |
| `henryhub_diario.csv` | Henry Hub, USD/MMBtu | 5 693 | 1997-01-07 | 2026-09-22 |
| `brent_mensual.csv` | Brent, media mensual | 472 | 1987-05 | 2026-08 |
| `wti_mensual.csv` | WTI, media mensual | 488 | 1986-01 | 2026-08 |
| `henryhub_mensual.csv` | Henry Hub, media mensual | 356 | 1997-01 | 2026-08 |
| `vix_diario.csv` | VIX, OHLC | 9 278 | 1990-01-02 | 2026-09-22 |
| `sp500_mensual.csv` | S&P 500 con dividendos, BPA, IPC y PER10 (Shiller) | 1 868 | 1871-01 | 2026-08 |
| `oro_mensual.csv` | Oro, USD/onza | 2 324 | 1833-01 | 2026-08 |
| `bono10a_mensual.csv` | Bono EEUU 10 años, rendimiento | 880 | 1953-04 | 2026-07 |
| `eurusd_diario.csv` | USD por EUR | 7 230 | 1999-01-04 | 2026-09-18 |
| `tiempo_diario.csv` | Temperatura máx/mín y viento, Seattle y Nueva York | 2 922 | 2012-01-01 | 2015-12-31 |

`manifiesto.json` guarda de cada uno el repositorio de origen, la ruta exacta y
un SHA-256 truncado, para poder comprobar que no han cambiado.

## Procedencia y licencia

- Las once primeras vienen de los paquetes de datos de **Frictionless/datahub**
  (`github.com/datasets/...`), que republican series públicas de la EIA, la
  Reserva Federal de San Luis, el CBOE y Robert Shiller. Dominio público
  (ODC-PDDL).
- `tiempo_diario.csv` viene de **vega-datasets** (BSD-3), a su vez de la NOAA.

Todas son de acceso público y sin registro. No hay nada de pago ni con licencia
restrictiva.

## Cómo se cargan

Las de una sola serie entran directas:

```python
from icm.datos.ficheros import cargar_precios_fichero

brent = cargar_precios_fichero("datos/reales/brent_diario.csv")
```

El S&P 500 trae nueve columnas, así que hay que elegir:

```python
sp = cargar_precios_fichero("datos/reales/sp500_mensual.csv", tickers=["SP500"])
```

El fichero de temperatura está en formato largo y con columnas de texto, así que
se le dice qué columna es cuál:

```python
from icm.datos.ficheros import leer_tabla, preparar_precios

tmax = preparar_precios(
    leer_tabla("datos/reales/tiempo_diario.csv"),
    formato="largo", columna_fecha="date",
    columna_ticker="location", columna_valor="temp_max",
)   # -> columnas NEW YORK y SEATTLE
```

## Lo que estos datos ya han roto

**`wti_diario.csv` no se puede cargar.** `validar_marco_precios` rechaza la serie
entera con «precios nulos o negativos». El culpable es una sola fila:

```
2020-04-20   -36.98
```

Es correcta. Es el día que el WTI cerró en negativo por primera vez en la
historia, cuando los contratos de mayo vencían sin sitio donde almacenar el
crudo. La validación da por sentado que un precio no puede ser negativo, y en
materias primas energéticas eso es falso: además del episodio del WTI, los
precios negativos son **rutina** en el mercado eléctrico europeo siempre que
coinciden mucha renovable y poca demanda.

Dicho de otro modo: la regla que protege los datos es la que impide analizar el
día más interesante de la serie, y haría lo mismo con cualquier descarga de
OMIE. Está anotado para arreglar.

## Los precios negativos del pool

De los 102 143 períodos del fichero, **2 315 casaron a cero exacto y 3 553 en
negativo**, con un mínimo de −15 €/MWh: el 5,74 % del mercado, repartido por
**390 días distintos**. No es una anomalía ni un error de descarga, es cómo
funciona el pool cuando coinciden mucha renovable y poca demanda.

Durante un tiempo la librería no pudo cargar este fichero: `validar_marco_precios`
rechazaba la serie entera en cuanto encontraba un precio no positivo, igual que
hacía con el WTI del 20 de abril de 2020. Ya no. La protección vive ahora donde
de verdad hace falta, en `rendimientos(tipo="log")`, que es la única operación
que no admite un cero; para una serie eléctrica lo correcto es trabajar con la
diferencia de precios, y el mensaje de error lo dice.

## Sobre la temperatura

Viene de la estación NOAA `SP000003195` (Madrid). Dos cautelas que importan para
los grados-día:

- La columna `TAVG` solo está en 1 626 de los 3 078 días. El resto de la media
  es `(máxima + mínima) / 2`, que es la convención con la que el sector publica
  los grados-día pero queda algo por encima de la media real del día, y por
  tanto **subestima ligeramente los HDD**.
- Faltan unos 11 días al año. Con `grados_dia` eso no desplaza nada porque la
  transformación es puntual, pero cualquier acumulado móvil sí lo nota.

Los grados-día que salen (base 15,5 / 22 °C) van de 1 341 HDD en 2018 a 924 en
2024, con los CDD subiendo de 404 a 468. La tendencia es real, pero parte de la
caída de 2025-2026 es que la muestra termina en septiembre y se queda sin los
meses fríos.

## El sistema eléctrico, y la previsión publicada

`sistema_electrico_diario.csv` y `sistema_electrico_horario.csv` traen, para
2015-2020, la demanda española realizada, la **previsión de demanda que el
operador publica antes de que cierre la subasta**, y la generación eólica y
solar. Vienen de Open Power System Data, que republica ENTSO-E Transparency.

Esa columna de previsión es la pieza que permite contrastar la tesis central
del proyecto, y costó encontrarla. La intuición inicial —buscar previsiones
meteorológicas— era la equivocada: **lo que entra en la subasta de OMIE no es
el tiempo, es la previsión de demanda del operador**. La previsión es buena de
verdad: 0,84 % de error absoluto medio y correlación de 0,9939 con la demanda
realizada.

Solapa 1 004 días con OMIE (2018-01-01 a 2020-09-30). OPSD dejó de
actualizarse en octubre de 2020, así que para años posteriores haría falta
ENTSO-E directamente, que pide un token gratuito.

## Lo que no se pudo conseguir

- **ESIOS** necesita un token personal gratuito que se pide por correo a
  `consultasios@ree.es`. El dominio es alcanzable: devuelve 403 por falta de
  credencial, no por la red. La API `apidatos.ree.es` está detrás de protección
  anti-bot y no se intentó rodear.
- **`open-meteo`** tiene un archivo de previsiones meteorológicas pasadas y no
  pide clave, pero devuelve 429 de forma persistente: el límite va por IP de
  salida y el contenedor la comparte. No se buscó forma de esquivarlo, y al
  final no hizo falta.
- **TTF y derechos de emisión EUA.** El primero lo publica ICE bajo licencia y
  el segundo EEX; ninguno tiene descarga abierta comparable. Por eso el CO2
  entra en el cuaderno como análisis de sensibilidad y no como serie.
