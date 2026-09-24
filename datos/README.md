# Tus datos

Deja aquí tu fichero de precios como `precios.csv` (también vale `.xlsx`,
`.xls` o `.parquet`) y el proyecto lo usará automáticamente:

```bash
pytest tests/test_datos_propios.py -v -s
```

```python
from icm.datos import cargar_precios_fichero

precios = cargar_precios_fichero("datos/precios.csv")
```

## Formato

Lo normal es **formato ancho**: una columna de fechas y una columna por activo.

```csv
fecha,IBE.MC,SAN.MC,ITX.MC
2024-01-02,11.82,3.91,39.55
2024-01-03,11.76,3.88,39.21
```

También se admite **formato largo** (`formato="largo"`), con tres columnas:
fecha, identificador del activo y valor.

```csv
fecha,ticker,cierre
2024-01-02,IBE.MC,11.82
2024-01-02,SAN.MC,3.91
```

Y **rendimientos** en lugar de precios, con `contenido="rendimientos"`. Ojo con
la escala: `0.025` es un 2,5 %; si tu fichero pone `2.5`, divídelo antes por 100.

## Si tu fichero viene de una hoja de cálculo española

Pasa `dia_primero=True` para las fechas en `dd/mm/aaaa`. El punto y coma como
separador, la coma decimal y el punto de miles se reconocen solos.

```python
cargar_precios_fichero("datos/precios.csv", dia_primero=True)
```

## Ajustes por variable de entorno

Para la batería de pruebas, sin tocar código:

| Variable | Para qué | Ejemplo |
|---|---|---|
| `ICM_DATOS` | ruta del fichero | `~/tfg/ibex.xlsx` |
| `ICM_DATOS_FORMATO` | `ancho` o `largo` | `largo` |
| `ICM_DATOS_CONTENIDO` | `precios` o `rendimientos` | `rendimientos` |
| `ICM_DATOS_DIA_PRIMERO` | fechas `dd/mm/aaaa` | `1` |
| `ICM_DATOS_COLUMNA_FECHA` | nombre de la columna de fechas | `Fecha` |

## Sobre el control de versiones

Los ficheros de esta carpeta **sí se suben al repositorio**, que suele ser lo
que interesa en un trabajo académico: sin los datos, el análisis no es
reproducible. Si tu fichero es muy grande o no lo puedes redistribuir, añade a
`.gitignore`:

```
datos/*
!datos/README.md
```
