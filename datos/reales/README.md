# Datos reales

Quince series **no simuladas**, 159 339 filas, descargadas el 25 de septiembre
de 2026. Hasta ahora todo el proyecto se había construido y validado contra
`icm.energia.simulacion`, así que esta carpeta es su primer contacto con datos
que no ha generado él mismo.

Las tres primeras son las que de verdad importan al proyecto, y vienen
directamente de los operadores del mercado español:

| Fichero | Contenido | Filas | Desde | Hasta |
|---|---|---:|---|---|
| `omie_horario.csv` | Precio marginal **horario** del pool ibérico, €/MWh (ES y PT) | 102 167 | 2018-01-01 | 2026-09-25 |
| `omie_diario.csv` | Media aritmética diaria del pool ibérico, €/MWh | 3 185 | 2018-01-01 | 2026-09-25 |
| `mibgas_diario.csv` | Gas spot español `GDAES_D+1` en el PVB, €/MWh | 3 877 | 2016-01-06 | 2026-09-26 |

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

## El pool español no entra por la puerta

`omie_horario.csv` **no se puede cargar**, por la misma regla que tumba el WTI:

```
ErrorDatos: Se han encontrado precios nulos o negativos
```

En las 102 167 horas del fichero hay **2 315 exactamente a cero y 3 553
negativas**, con un mínimo de −15 €/MWh: el 5,74 % del mercado, repartido por
**390 días distintos**. No es una anomalía ni un error de descarga, es cómo
funciona el pool cuando coinciden mucha renovable y poca demanda.

`omie_diario.csv` sí carga, pero solo porque la media aritmética del día nunca
llega a bajar de cero (mínimo 0,18 €/MWh). Es decir: la librería admite el
mercado eléctrico español únicamente si se le borra antes la granularidad en la
que ocurre lo interesante.

## Lo que sigue faltando

**TTF** (el gas de referencia europeo) y los derechos de emisión **EUA**. El
primero lo publica ICE bajo licencia y el segundo EEX; ninguno tiene descarga
abierta comparable. ESIOS (generación por tecnología, previsión eólica, demanda)
necesita un token personal gratuito que hay que pedir por correo a
`consultasios@ree.es`; con ese token entraría, porque el dominio ya es
alcanzable — hoy devuelve 403 por falta de credencial, no por la red.

Para el spark spread, MIBGAS y OMIE bastan: gas y electricidad españoles, ambos
en €/MWh, con más de ocho años de solapamiento.
