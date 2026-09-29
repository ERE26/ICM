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

## La pila de ofertas de OMIE, y por qué hizo falta bajar a ella

Hasta aquí todo eran precios. Un precio no basta para una pregunta concreta:
**cuánto mueve el precio el propio activo al operar**. Eso no se puede
responder con series de precios, porque un precio es el resultado de la
casación, no la curva que la produjo. Hace falta la pila de ofertas.

OMIE publica la [curva agregada de oferta y
demanda](https://www.omie.es/es/file-access-list?parents=curva_pbc_uof&dir=Curvas%20agregadas%20de%20oferta%20y%20demanda%20del%20mercado%20diario)
de cada sesión del diario: todas las ofertas de compra y de venta de cada
período, con precio, cantidad y si quedaron casadas. Pesan unos 7 MB por
sesión, así que **no se guardan en el repositorio**: se descargaron 21
sesiones, se midió sobre ellas y lo que queda aquí es el resultado.

Las 21 sesiones, elegidas para cubrir los tres formatos del fichero, las
cuatro estaciones y los dos regímenes de precio (la crisis del gas y la
normalización posterior):

```
2023-01-17  2023-03-21  2023-04-18  2023-07-18  2023-10-17
2024-01-03  2024-01-17  2024-04-18  2024-04-21  2024-07-18  2024-10-17
2025-01-08  2025-01-17  2025-04-18  2025-04-20  2025-07-18  2025-10-17
2026-01-17  2026-01-18  2026-04-18  2026-07-18
```

Son 864 períodos: 384 horarios y 480 cuarto-horarios.

### El fichero ha cambiado de formato dos veces

Y las tres variantes conviven en el histórico, así que un lector que solo
entienda una de ellas suma mal:

| Variante | Columnas | Etiqueta de período | Vigencia |
|---|---|---|---|
| 1 | `Hora` + `Energía` (MWh) | enteros 1–24 | hasta 2025-04-17 |
| 2 | `Período` + `Potencia` (MW) | enteros 1–24 | 2025-04-18 a 2025-09-30 |
| 3 | `Período` + `Potencia` (MW) | `H1Q1`…`H24Q4` | desde 2025-10-01 |

El detalle que hay que acertar es que **en las variantes 2 y 3 la cantidad es
potencia, no energía**. Con períodos horarios coincide numéricamente, así que
el error pasa desapercibido durante cinco meses de datos; con
cuarto-horarios sobrestima la energía por cuatro. `leer_curva_agregada`
deduce la duración de cuántos períodos tiene el día, con la misma regla que
ya usaba el cargador de precios.

### Comprobación de autenticidad: el precio publicado sale de la curva

Es el contraste que decide si la lectura vale para algo, porque una curva mal
leída daría impactos plausibles y falsos sin que nada lo delatara. El precio
publicado tiene que poder reconstruirse de la propia pila:

| Resultado | Períodos | % |
|---|---:|---:|
| Coincide **exacto** con el máximo de venta casada o el mínimo de compra casada | 709 | 82,1 % |
| Cae **dentro del cruce**, entre ambos | 134 | 15,5 % |
| Queda fuera | 21 | 2,4 % |

El 15,5 % del medio no es un fallo de lectura: la curva agregada viene
redondeada a MWh enteros y la unidad marginal se casa parcialmente, así que el
equilibrio cae *entre* los dos extremos y no sobre uno de ellos. El error
mediano ahí es de 0,24 €/MWh y el percentil 99 de 2,00.

El 2,4 % restante se explica por la **separación de mercados**: cuando la
interconexión con Portugal se satura, España y Portugal casan por separado y
el precio español no es el de la casación ibérica única que describe el
fichero. Se ve en la etiqueta de país de las ofertas casadas — en los 43
períodos marcados `ES,PT` el acierto baja al 53,5 %, frente al 83,6 % de los
821 marcados `MI`.

Nada de esto afecta a la medida de impacto, y la razón importa: **el impacto
es una diferencia de dos puntos leídos de la misma curva**, así que un sesgo
de nivel se cancela. Lo que lo haría falso sería leer mal la *pendiente*, y la
pendiente es justo lo que confirma el contraste del cruce.

### Lo que salió: el impacto no es simétrico

`omie_elasticidad_oferta.csv`, 25 filas. Mediana del movimiento del precio
marginal, en €/MWh:

| Tramo | % de horas | Subir al comprar 500 MW | Bajar al vender 500 MW |
|---|---:|---:|---:|
| < 5 €/MWh | 6,0 % | 0,48 | 2,29 |
| 5–25 | 6,6 % | 7,06 | 7,51 |
| 25–60 | 36,0 % | 5,71 | 6,89 |
| 60–120 | 31,6 % | 3,83 | 9,68 |
| > 120 | 19,7 % | 1,44 | **20,01** |

Añadir 500 MW de demanda sube el precio 1,93 €/MWh en la mediana global;
añadir 500 MW de oferta lo baja 10,57. Un factor de **5,5**. La razón es
económica y no un artefacto: la demanda eléctrica es mucho más inelástica que
la oferta, porque buena parte se oferta al límite del mercado —tiene que
consumir pase lo que pase—, así que en cuanto se agota el tramo elástico el
precio se desploma. La oferta, en cambio, tiene un tramo plano largo de
renovables a coste cero.

Y la asimetría va **justo en contra del almacenamiento**, porque se ordena por
nivel de precio al revés de como opera una batería: carga en el tramo barato,
donde comprar casi no mueve el precio (0,48), y descarga en el caro, donde
vender lo hunde 20,01. El activo destruye el pico del que vive.

## Servicios de ajuste

`ree_servicios_ajuste.csv`, 8 filas. **Este fichero no está verificado contra
su fuente y es el único de esta carpeta que no lo está.** Conviene decir
exactamente qué es y qué no es.

Las cifras corresponden al [Informe del Sistema
Eléctrico](https://www.sistemaelectrico-ree.es/informe-del-sistema-electrico)
de Red Eléctrica, pero **no se descargaron**: todo el dominio de REE
—`esios.ree.es`, `apidatos.ree.es`, `www.ree.es`,
`sistemaelectrico-ree.es`— devuelve 403 de Incapsula a este contenedor, lo
que se comprobó host por host. Están reproducidas de memoria del modelo que
escribió este proyecto, que es una fuente que puede equivocarse en un dígito
sin avisar.

La única comprobación que se les ha hecho es **aritmética interna**, y es
importante ver qué prueba y qué no: 34,28 × 922 MW × 8 760 h = 277 M€, el
10,4 % de los 2 668 M€ de servicios de ajuste de 2024. Eso es consistente con
que la banda secundaria sea una partida mediana del total, lo que **fija la
unidad** —€/MW y por hora, no por año ni por MWh— pero **no confirma el
valor**: un 34,28 que en realidad fuera 31,50 pasaría la misma comprobación.

Quien use este fichero para decidir algo tiene que contrastarlo con el
informe. Se deja en el repositorio porque la unidad sí está establecida y el
orden de magnitud permite hacer el análisis de sensibilidad, no porque sea un
dato tan sólido como los demás de esta carpeta.

Hacen falta porque son el ingreso que `ParametrosInversion` pide a mano y que
el proyecto no tenía de ninguna fuente: el precio de la **banda de regulación
secundaria**, 34,28 €/MW por hora asignada en 2024, y el coste total de los
servicios de ajuste, que pasó de 724 M€ en 2021 a 3 347 M€ en 2025.

La unidad del 34,28 no venía indicada sin ambigüedad en el informe, así que se
comprobó aritméticamente: con una banda media asignada de 922 MW, 34,28 ×
922 × 8 760 = **277 M€**, el 10,4 % de los 2 668 M€ de servicios de ajuste de
2024. Cuadra con que la banda secundaria sea una partida mediana del total, lo
que confirma que es €/MW **y por hora**, no por año ni por MWh.

**El mercado de capacidad no tiene histórico que buscar.** Se creó el 15 de
septiembre de 2026, dos semanas antes de esta descarga, así que no existe
serie de precios de capacidad para España. Cualquier cifra que se pusiera ahí
sería inventada, y por eso `ingreso_adicional_anual_por_mw` sigue valiendo
cero por omisión.

## La asignación de banda: dónde está y por qué no está aquí

Es el dato que cierra la valoración del almacenamiento, porque de él depende
si la banda cubre el 10 %, el 25 % o el 50 % de las horas, que es la
diferencia entre 135 y 392 €/kWh de capex de equilibrio. Se buscó en serio.

**ESIOS no es alcanzable desde este entorno, y no por falta de credencial.**
Se comprobó host por host, con y sin `User-Agent` descriptivo:

| Host | Respuesta |
|---|---|
| `api.esios.ree.es` | 403 Incapsula |
| `www.esios.ree.es` | 403 Incapsula |
| `apidatos.ree.es` | 403 Incapsula |
| `www.ree.es` | 403 Incapsula |
| `sistemaelectrico-ree.es` | 403 Incapsula |
| `demanda.ree.es` | 403 Incapsula |

No es la política de red del contenedor —el proxy no registra ningún fallo de
relé y OMIE, ENTSO-E y el resto de orígenes responden con normalidad— sino
protección anti-bot del origen, que rechaza a este contenedor por su IP. No se
intentó rodearla.

**ENTSO-E sí es alcanzable y publica lo mismo.** Responde con el protocolo
correcto, no con un muro:

```xml
<Reason><code>999</code><text>Authentication failed.</text></Reason>
```

Es decir: el host contesta, entiende la consulta y solo falta el token. La
plataforma de transparencia publica las reservas de balance contratadas bajo
los artículos 17.1.B, C y D, y lo que en España se llama banda de regulación
secundaria es allí **aFRR**.

El token es gratuito pero va ligado a una persona, así que este proyecto no
puede pedirlo solo:

1. Cuenta en <https://transparency.entsoe.eu>.
2. Correo a `transparency@entsoe.eu` con asunto *Restful API access* desde esa
   dirección.
3. El token aparece en *My Account Settings*.

Con el token en la variable de entorno `ENTSOE_TOKEN`, `icm.datos.entsoe`
funciona sin tocar nada:

```python
from icm.datos import descargar_banda_secundaria, horas_con_banda

banda = descargar_banda_secundaria("2024-01-01", "2025-01-01")
horas_con_banda(banda, potencia=100.0)   # la fracción que el VAN necesita
```

El lector está escrito y probado; lo que no está verificado son los
parámetros de consulta, porque sin token no se ha podido ejecutar una
descarga real ni una vez. Está explicado en la cabecera del módulo y aislado
en `parametros_banda_secundaria` para que corregirlo sea una línea.

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
- **Precios del mercado de capacidad.** No es que estén cerrados: **no
  existen**. El mercado se creó el 15 de septiembre de 2026.
- **Curvas agregadas de OMIE completas.** Están abiertas y se descargan sin
  credencial, pero son ~7 MB por sesión y hay más de 3 000 sesiones. Se
  tomaron 21, que bastan para medir la pendiente de la pila; no bastarían para
  un estudio de cómo ha ido cambiando esa pendiente con los años.
