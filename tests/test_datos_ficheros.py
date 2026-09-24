"""Pruebas de la carga de datos propios: ficheros y marcos en memoria."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.datos import (
    ProveedorFichero,
    ProveedorMarco,
    cargar_precios_fichero,
    preparar_precios,
    rendimientos,
)

FECHAS = pd.bdate_range("2024-01-01", periods=10)
PRECIOS = pd.DataFrame(
    {
        "AAA": np.linspace(100.0, 109.0, 10),
        "BBB": np.linspace(50.0, 45.5, 10),
    },
    index=FECHAS,
)


@pytest.fixture
def csv_ancho(tmp_path):
    ruta = tmp_path / "precios.csv"
    PRECIOS.rename_axis("fecha").to_csv(ruta)
    return ruta


# --- formato ancho -------------------------------------------------------------


def test_se_lee_un_csv_en_formato_ancho(csv_ancho):
    precios = cargar_precios_fichero(csv_ancho)
    assert list(precios.columns) == ["AAA", "BBB"]
    assert len(precios) == 10
    np.testing.assert_allclose(precios["AAA"].to_numpy(), PRECIOS["AAA"].to_numpy())


def test_se_puede_pedir_un_subconjunto_de_activos(csv_ancho):
    precios = cargar_precios_fichero(csv_ancho, ["AAA"])
    assert list(precios.columns) == ["AAA"]


def test_se_puede_recortar_la_ventana(csv_ancho):
    precios = cargar_precios_fichero(csv_ancho, inicio="2024-01-05", fin="2024-01-10")
    assert precios.index.min() >= pd.Timestamp("2024-01-05")
    assert precios.index.max() <= pd.Timestamp("2024-01-10")


def test_pedir_un_activo_inexistente_da_un_error_util(csv_ancho):
    with pytest.raises(ErrorDatos, match="Disponibles"):
        cargar_precios_fichero(csv_ancho, ["ZZZ"])


def test_una_ventana_vacia_da_error(csv_ancho):
    with pytest.raises(ErrorDatos, match="ninguna fecha"):
        cargar_precios_fichero(csv_ancho, inicio="2030-01-01")


# --- formato largo -------------------------------------------------------------


def test_se_lee_un_csv_en_formato_largo(tmp_path):
    largo = PRECIOS.rename_axis("fecha").reset_index().melt(
        id_vars="fecha", var_name="ticker", value_name="cierre"
    )
    ruta = tmp_path / "largo.csv"
    largo.to_csv(ruta, index=False)

    precios = cargar_precios_fichero(ruta, formato="largo")
    assert list(precios.columns) == ["AAA", "BBB"]
    np.testing.assert_allclose(precios["BBB"].to_numpy(), PRECIOS["BBB"].to_numpy())


def test_el_formato_largo_admite_columnas_con_nombre_propio(tmp_path):
    largo = pd.DataFrame(
        {
            "dia": list(FECHAS) * 2,
            "simbolo": ["AAA"] * 10 + ["BBB"] * 10,
            "ultimo": list(PRECIOS["AAA"]) + list(PRECIOS["BBB"]),
            "volumen": 1,
        }
    )
    ruta = tmp_path / "largo2.csv"
    largo.to_csv(ruta, index=False)

    precios = cargar_precios_fichero(
        ruta, formato="largo", columna_fecha="dia",
        columna_ticker="simbolo", columna_valor="ultimo",
    )
    assert list(precios.columns) == ["AAA", "BBB"]


def test_el_formato_largo_ambiguo_avisa(tmp_path):
    largo = pd.DataFrame(
        {"fecha": list(FECHAS) * 2, "a": 1, "b": 2, "c": 3}
    )
    ruta = tmp_path / "ambiguo.csv"
    largo.to_csv(ruta, index=False)
    with pytest.raises(ErrorDatos, match="columna_ticker"):
        cargar_precios_fichero(ruta, formato="largo")


# --- particularidades de los ficheros reales -----------------------------------


def test_se_entiende_un_csv_espaniol_con_punto_y_coma_y_coma_decimal(tmp_path):
    ruta = tmp_path / "espaniol.csv"
    ruta.write_text(
        "Fecha;AAA;BBB\n"
        "02/01/2024;1.234,50;50,25\n"
        "03/01/2024;1.240,75;49,80\n"
        "04/01/2024;1.228,00;51,10\n",
        encoding="utf-8",
    )
    precios = cargar_precios_fichero(ruta, dia_primero=True)
    assert precios.loc["2024-01-02", "AAA"] == pytest.approx(1234.50)
    assert precios.loc["2024-01-03", "BBB"] == pytest.approx(49.80)
    assert list(precios.index.day) == [2, 3, 4]


def test_las_fechas_dia_primero_no_se_confunden(tmp_path):
    ruta = tmp_path / "fechas.csv"
    ruta.write_text("fecha,AAA\n05/03/2024,100\n06/03/2024,101\n", encoding="utf-8")
    precios = cargar_precios_fichero(ruta, dia_primero=True)
    assert precios.index[0] == pd.Timestamp("2024-03-05")


def test_los_nombres_de_columna_se_normalizan(tmp_path):
    ruta = tmp_path / "espacios.csv"
    ruta.write_text("  fecha , aapl , msft \n2024-01-02,100,200\n2024-01-03,101,201\n")
    precios = cargar_precios_fichero(ruta)
    assert list(precios.columns) == ["AAPL", "MSFT"]


def test_los_huecos_se_arrastran_hacia_delante(tmp_path):
    ruta = tmp_path / "huecos.csv"
    ruta.write_text("fecha,AAA,BBB\n2024-01-02,100,50\n2024-01-03,,51\n2024-01-04,102,52\n")
    precios = cargar_precios_fichero(ruta)
    # El festivo de un mercado que no lo es de otro no debe tirar la fila.
    assert len(precios) == 3
    assert precios.loc["2024-01-03", "AAA"] == pytest.approx(100.0)


def test_las_fechas_duplicadas_se_resuelven_con_la_ultima(tmp_path):
    ruta = tmp_path / "duplicadas.csv"
    ruta.write_text("fecha,AAA\n2024-01-02,100\n2024-01-02,105\n2024-01-03,101\n")
    precios = cargar_precios_fichero(ruta)
    assert len(precios) == 2
    assert precios.loc["2024-01-02", "AAA"] == pytest.approx(105.0)


def test_una_columna_sin_numeros_se_denuncia(tmp_path):
    ruta = tmp_path / "texto.csv"
    ruta.write_text("fecha,AAA,NOTAS\n2024-01-02,100,revisar\n2024-01-03,101,ok\n")
    with pytest.raises(ErrorDatos, match="no contienen numeros"):
        cargar_precios_fichero(ruta)


def test_un_fichero_inexistente_da_error(tmp_path):
    with pytest.raises(ErrorDatos, match="No existe el fichero"):
        cargar_precios_fichero(tmp_path / "fantasma.csv")


def test_una_extension_desconocida_da_error(tmp_path):
    ruta = tmp_path / "datos.docx"
    ruta.write_text("nada")
    with pytest.raises(ErrorDatos, match="no reconocida"):
        cargar_precios_fichero(ruta)


def test_sin_columna_de_fechas_reconocible_se_avisa(tmp_path):
    ruta = tmp_path / "sinfecha.csv"
    ruta.write_text("AAA,BBB\n100,50\n101,51\n")
    with pytest.raises(ErrorDatos, match="columna de fechas"):
        cargar_precios_fichero(ruta)


def test_se_entienden_las_fechas_escritas_como_aaaammdd(tmp_path):
    ruta = tmp_path / "enteras.csv"
    ruta.write_text("fecha,AAA\n20240102,100\n20240103,101\n")
    precios = cargar_precios_fichero(ruta)
    assert list(precios.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_una_primera_columna_numerica_no_pasa_por_fecha(tmp_path):
    # Sin este cuidado, pandas leeria los enteros como nanosegundos desde 1970
    # y aceptaria como fechas una columna que no lo es.
    ruta = tmp_path / "indices.csv"
    ruta.write_text("id,AAA\n1,100\n2,101\n")
    with pytest.raises(ErrorDatos, match="columna de fechas"):
        cargar_precios_fichero(ruta)


def test_una_columna_de_fechas_inexistente_se_denuncia(csv_ancho):
    with pytest.raises(ErrorDatos, match="no esta en el fichero"):
        cargar_precios_fichero(csv_ancho, columna_fecha="jornada")


def test_se_lee_un_excel(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "precios.xlsx"
    PRECIOS.rename_axis("fecha").to_excel(ruta)
    precios = cargar_precios_fichero(ruta)
    assert list(precios.columns) == ["AAA", "BBB"]


def test_se_lee_un_parquet(tmp_path):
    ruta = tmp_path / "precios.parquet"
    PRECIOS.rename_axis("fecha").reset_index().to_parquet(ruta)
    precios = cargar_precios_fichero(ruta)
    assert len(precios) == 10


# --- rendimientos en lugar de precios ------------------------------------------


def test_se_admiten_rendimientos_y_se_reconstruye_el_precio():
    tabla = pd.DataFrame({"AAA": [0.01, -0.02, 0.03]}, index=FECHAS[:3])
    precios = ProveedorMarco(tabla, contenido="rendimientos").descargar(["AAA"])
    esperado = 100.0 * np.cumprod([1.01, 0.98, 1.03])
    np.testing.assert_allclose(precios["AAA"].to_numpy(), esperado)


def test_el_ciclo_rendimientos_precios_rendimientos_es_exacto():
    originales = pd.DataFrame(
        {"AAA": [0.01, -0.02, 0.03, 0.005], "BBB": [-0.01, 0.02, 0.0, 0.01]}, index=FECHAS[:4]
    )
    precios = ProveedorMarco(originales, contenido="rendimientos").descargar(["AAA", "BBB"])
    recuperados = rendimientos(precios, "simple")
    np.testing.assert_allclose(
        recuperados.to_numpy(), originales.iloc[1:].to_numpy(), atol=1e-12
    )


def test_se_admiten_rendimientos_logaritmicos():
    tabla = pd.DataFrame({"AAA": [0.01, -0.02]}, index=FECHAS[:2])
    precios = ProveedorMarco(
        tabla, contenido="rendimientos", tipo_rendimiento="log"
    ).descargar(["AAA"])
    np.testing.assert_allclose(
        precios["AAA"].to_numpy(), 100.0 * np.cumprod(np.exp([0.01, -0.02]))
    )


def test_unos_rendimientos_en_porcentaje_se_denuncian():
    # 2.5 queriendo decir 2,5 %: como rendimiento simple seria un +250 %.
    tabla = pd.DataFrame({"AAA": [2.5, -150.0]}, index=FECHAS[:2])
    with pytest.raises(ErrorDatos, match="dividalos antes por 100"):
        ProveedorMarco(tabla, contenido="rendimientos")


def test_un_hueco_en_los_rendimientos_no_se_arrastra():
    # Repetir el rendimiento anterior inventaria un movimiento que no ocurrio,
    # asi que la fecha incompleta se descarta.
    tabla = pd.DataFrame(
        {"AAA": [0.01, np.nan, 0.03], "BBB": [0.02, 0.01, -0.01]}, index=FECHAS[:3]
    )
    precios = ProveedorMarco(tabla, contenido="rendimientos").descargar(["AAA", "BBB"])
    assert len(precios) == 2
    np.testing.assert_allclose(precios["AAA"].to_numpy(), 100.0 * np.cumprod([1.01, 1.03]))


def test_un_contenido_desconocido_da_error():
    with pytest.raises(ErrorDatos, match="Contenido"):
        preparar_precios(PRECIOS, contenido="dividendos")


def test_un_formato_desconocido_da_error():
    with pytest.raises(ErrorDatos, match="Formato"):
        preparar_precios(PRECIOS, formato="diagonal")


# --- integracion con el resto de la libreria -----------------------------------


def test_el_proveedor_de_fichero_encaja_con_el_cache(tmp_path, csv_ancho):
    from icm.datos import CacheParquet, ProveedorConCache

    con_cache = ProveedorConCache(ProveedorFichero(csv_ancho), CacheParquet(tmp_path))
    primera = con_cache.descargar(["AAA", "BBB"])
    segunda = con_cache.descargar(["AAA", "BBB"])
    pd.testing.assert_frame_equal(primera, segunda)


def test_el_proveedor_de_marco_encaja_con_cargar_precios():
    from icm.datos import cargar_precios

    precios = cargar_precios(["AAA"], proveedor=ProveedorMarco(PRECIOS))
    assert list(precios.columns) == ["AAA"]


def test_los_precios_no_positivos_se_rechazan():
    tabla = pd.DataFrame({"AAA": [100.0, -5.0]}, index=FECHAS[:2])
    with pytest.raises(ErrorDatos, match="nulos o negativos"):
        ProveedorMarco(tabla).descargar(["AAA"])
