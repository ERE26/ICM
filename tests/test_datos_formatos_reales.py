"""Los formatos con los que de verdad se descargan los datos espanioles.

Cada prueba de aqui nacio de un fichero real que la libreria leia mal. Tres de
los cuatro formatos fallaban **en silencio**: un anio de OMIE entraba con 366
dias y salia con 144, un volumen de MIBGAS se dividia por mil, y un fichero de
ESIOS salia con las columnas llamadas por sus precios. El proposito de este
modulo es que ninguno de esos cuatro vuelva a pasar desapercibido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.datos.ficheros import (
    InformeCarga,
    ProveedorFichero,
    cargar_precios_fichero,
    convertir_fechas,
    resumen_carga,
)


# --------------------------------------------------------------------------
# El orden de dia y mes
# --------------------------------------------------------------------------


def test_un_anio_de_omie_no_pierde_222_dias_en_silencio(tmp_path):
    """El fallo mas caro de la capa: 366 dias entraban y salian 144.

    Con ``dia_primero=False`` por omision, pandas deducia ``%m/%d/%Y``: los
    dias 13 o mayores se volvian NaT y un dropna los tiraba sin avisar, y los
    12 primeros de cada mes salian con el dia y el mes intercambiados. La
    volatilidad anualizada de la serie resultante era 6,7 veces la real.
    """
    lineas = ["Fecha;MD_ES"]
    dia = pd.Timestamp("2024-01-01")
    while dia.year == 2024:
        lineas.append(f"{dia:%d/%m/%Y};{45 + dia.dayofyear % 30},50")
        dia += pd.Timedelta(days=1)
    ruta = tmp_path / "omie.csv"
    ruta.write_text("\n".join(lineas))

    # Un anio entero trae dias del 13 al 31, que mes/dia/anio no puede leer,
    # asi que la deteccion resuelve sola y sin preguntar nada.
    precios = cargar_precios_fichero(ruta, separador=";", decimal=",")
    assert len(precios) == 366
    assert precios.index.min() == pd.Timestamp("2024-01-01")
    assert precios.index.max() == pd.Timestamp("2024-12-31")
    assert "dia/mes/anio" in precios.attrs["informe_carga"]["interpretacion_fechas"]

    # Y decirlo a mano da exactamente lo mismo.
    explicito = cargar_precios_fichero(ruta, separador=";", decimal=",", dia_primero=True)
    pd.testing.assert_frame_equal(precios, explicito, check_like=True)


def test_una_fecha_iso_no_se_declara_ambigua():
    # 2024-01-02 solo admite una lectura, pero pandas aplica 'dayfirst'
    # igualmente y la reinterpreta: sin tratarla aparte, la deteccion la
    # declaraba ambigua y se negaba a leer un CSV perfectamente normal.
    serie = pd.Series(["2024-01-02", "2024-01-03", "2024-03-15"])
    fechas = convertir_fechas(serie)
    assert list(fechas) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-03-15"),
    ]


def test_cuando_solo_una_lectura_interpreta_todo_se_elige_sola():
    # Con un 31 por medio, mes/dia/anio no puede ser: no hay que preguntar.
    informe = InformeCarga()
    fechas = convertir_fechas(pd.Series(["01/02/2024", "31/12/2024"]), None, informe)
    assert list(fechas) == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-12-31")]
    assert "dia/mes/anio" in informe.interpretacion_fechas


def test_sin_un_dia_mayor_que_doce_se_pregunta():
    with pytest.raises(ErrorDatos, match="ambiguo"):
        convertir_fechas(pd.Series(["01/02/2024", "03/04/2024"]))


# --------------------------------------------------------------------------
# Los numeros
# --------------------------------------------------------------------------


def test_el_separador_de_miles_no_divide_por_mil(tmp_path):
    """"11.250" son once mil doscientos cincuenta, no 11,25.

    La misma columna mezclaba dos escalas con un factor 1000 entre ellas segun
    la celda llevase decimales o no, de modo que una serie que se movio un
    0,5 % producia un +100.000 % y un -99,9 %.
    """
    ruta = tmp_path / "mibgas.csv"
    ruta.write_text("Fecha;VOLUMEN\n02/01/2024;11.250\n31/01/2024;11.310,50\n01/02/2024;11.180\n")
    precios = cargar_precios_fichero(ruta, separador=";", decimal=",")
    np.testing.assert_allclose(precios["VOLUMEN"].to_numpy(), [11250.0, 11310.5, 11180.0])
    assert "VOLUMEN" in precios.attrs["informe_carga"]["celdas_con_miles"]


def test_un_decimal_de_dos_cifras_no_se_confunde_con_miles(tmp_path):
    ruta = tmp_path / "precios.csv"
    ruta.write_text("fecha,AAA\n2024-01-02,45.67\n2024-01-03,46.10\n")
    precios = cargar_precios_fichero(ruta)
    np.testing.assert_allclose(precios["AAA"].to_numpy(), [45.67, 46.10])


def test_el_porcentaje_se_divide_en_vez_de_borrarse(tmp_path):
    # Quitar el simbolo y dejar el numero convertia un 1,5 % en un 1,5, que
    # como rendimiento es un +150 % diario.
    ruta = tmp_path / "rend.csv"
    ruta.write_text("fecha;AAA\n02/01/2024;1,5%\n31/01/2024;-0,8%\n")
    precios = cargar_precios_fichero(
        ruta, separador=";", decimal=",", contenido="rendimientos"
    )
    np.testing.assert_allclose(
        precios["AAA"].to_numpy(), [100.0, 101.5, 101.5 * 0.992], rtol=1e-9
    )


# --------------------------------------------------------------------------
# El formato largo
# --------------------------------------------------------------------------


def test_esios_publica_el_identificador_detras_y_no_se_pivota_al_reves(tmp_path):
    """ESIOS da ``(datetime, value, indicator)``, con el id numerico al final.

    Asignar las dos columnas por posicion pivotaba la tabla del reves: las
    columnas del marco pasaban a llamarse por los precios y los valores eran
    los codigos de indicador. Como todo era numerico, nada protestaba.
    """
    ruta = tmp_path / "esios.csv"
    ruta.write_text(
        "datetime,value,indicator\n"
        "2024-01-02,45.67,600\n2024-01-03,46.10,600\n2024-01-04,47.00,600\n"
        "2024-01-02,31.25,1900\n2024-01-03,30.80,1900\n2024-01-04,29.95,1900\n"
    )
    precios = cargar_precios_fichero(ruta, formato="largo")
    assert sorted(precios.columns) == ["1900", "600"]
    np.testing.assert_allclose(precios["600"].to_numpy(), [45.67, 46.10, 47.00])
    np.testing.assert_allclose(precios["1900"].to_numpy(), [31.25, 30.80, 29.95])


def test_si_la_cardinalidad_no_discrimina_se_pregunta(tmp_path):
    # Dos columnas con la misma variedad de valores no permiten deducir cual
    # es el activo: adivinar seria volver al fallo anterior.
    ruta = tmp_path / "ambiguo.csv"
    ruta.write_text("fecha,a,b\n2024-01-02,1,10\n2024-01-03,2,20\n")
    with pytest.raises(ErrorDatos, match="columna_ticker"):
        cargar_precios_fichero(ruta, formato="largo")


# --------------------------------------------------------------------------
# Las horas
# --------------------------------------------------------------------------


def _fichero_omie_horario(ruta, dias, periodos):
    lineas = ["Fecha;Hora;MD_ES"]
    for dia in dias:
        for p in range(1, periodos + 1):
            lineas.append(f"{dia:%d/%m/%Y};{p};{40 + p},50")
    ruta.write_text("\n".join(lineas))


def test_el_formato_horario_de_omie_no_colapsa_el_dia(tmp_path):
    # Sin 'columna_hora', las 24 filas de cada dia compartian fecha y se
    # quedaba con la de la hora 24 como si fuera el precio del dia, dejando
    # ademas la columna HORA como si fuera un activo mas.
    ruta = tmp_path / "omieh.csv"
    _fichero_omie_horario(ruta, pd.date_range("2024-02-01", periods=2), 24)

    with pytest.raises(ErrorDatos, match="fechas repetidas"):
        cargar_precios_fichero(ruta, separador=";", decimal=",", dia_primero=True)

    precios = cargar_precios_fichero(
        ruta, separador=";", decimal=",", dia_primero=True, columna_hora="Hora"
    )
    assert list(precios.columns) == ["MD_ES"]
    assert len(precios) == 48
    assert precios.index[0] == pd.Timestamp("2024-02-01 00:00")
    assert precios.index[1] == pd.Timestamp("2024-02-01 01:00")


def test_un_dia_cuartohorario_no_se_estira_cuatro_dias(tmp_path):
    # El 1 de octubre de 2025 OMIE paso de 24 periodos horarios a 96
    # cuarto-horarios. Tomar el numero de periodo como horas convertia el
    # periodo 96 en cuatro dias despues.
    ruta = tmp_path / "omie96.csv"
    _fichero_omie_horario(ruta, pd.date_range("2025-10-01", periods=1), 96)
    precios = cargar_precios_fichero(
        ruta, separador=";", decimal=",", dia_primero=True, columna_hora="Hora"
    )
    assert len(precios) == 96
    assert precios.index[1] - precios.index[0] == pd.Timedelta(minutes=15)
    assert precios.index[-1] == pd.Timestamp("2025-10-01 23:45")


def test_con_huso_el_dia_de_veinticinco_horas_conserva_las_dos_dos(tmp_path):
    """El ultimo domingo de octubre el pool casa 25 periodos.

    Sin huso, el periodo 25 cae en la medianoche del dia siguiente y una de
    las dos veces que dieron las 02:00 desaparecia. Con huso son dos instantes
    distintos y las dos sobreviven, que es lo correcto: son dos horas reales
    con dos precios distintos.
    """
    lineas = ["Fecha;Hora;MD_ES"]
    for p in range(1, 26):
        lineas.append(f"27/10/2024;{p};{40 + p},00")
    for p in range(1, 25):
        lineas.append(f"28/10/2024;{p};{60 + p},00")
    ruta = tmp_path / "dst.csv"
    ruta.write_text("\n".join(lineas))

    precios = cargar_precios_fichero(
        ruta, separador=";", decimal=",", dia_primero=True,
        columna_hora="Hora", huso="Europe/Madrid",
    )
    assert len(precios) == 49
    del_domingo = precios.index[precios.index.date == pd.Timestamp("2024-10-27").date()]
    dos = del_domingo[del_domingo.strftime("%H:%M") == "02:00"]
    assert len(dos) == 2  # una en CEST y otra en CET
    assert dos[0].utcoffset() != dos[1].utcoffset()
    assert precios.loc[dos[0], "MD_ES"] != precios.loc[dos[1], "MD_ES"]


# --------------------------------------------------------------------------
# La codificacion y el preambulo
# --------------------------------------------------------------------------


def test_un_csv_en_latin1_se_puede_leer(tmp_path):
    # Excel en Espania exporta en cp1252. Antes no habia ningun parametro de
    # codificacion, asi que una tilde en la cabecera lo hacia ilegible.
    ruta = tmp_path / "latin1.csv"
    ruta.write_bytes(b"D\xeda;Precio\n02/01/2024;45,67\n31/01/2024;46,10\n")
    precios = cargar_precios_fichero(ruta, separador=";", decimal=",")
    assert len(precios) == 2
    np.testing.assert_allclose(precios["PRECIO"].to_numpy(), [45.67, 46.10])


def test_un_preambulo_antes_de_la_tabla_se_salta_solo(tmp_path):
    """Los informes de MIBGAS abren con una linea de fecha de emision.

    La deteccion cuenta separadores: la linea de preambulo trae uno y la
    cabecera real, siete. Sin saltarla, la cabecera se leia como datos.
    """
    ruta = tmp_path / "mibgas.csv"
    ruta.write_text(
        "Fecha Emision :31/12/2024 - 21:33;\n"
        "Trading day;Product;Place of delivery;Area;First Day Delivery;"
        "Last Day Delivery;MIBGAS Daily Price;\n"
        "02/01/2024;GDAES_D+1;PVB;ES;02/01/2024;02/01/2024;28,50;\n"
        "31/01/2024;GDAES_D+1;PVB;ES;31/01/2024;31/01/2024;29,10;\n"
    )
    precios = cargar_precios_fichero(
        ruta, separador=";", decimal=",",
        formato="largo", columna_fecha="Trading day",
        columna_ticker="Product", columna_valor="MIBGAS Daily Price",
    )
    assert list(precios.columns) == ["GDAES_D+1"]
    assert len(precios) == 2
    assert precios.attrs["informe_carga"]["filas_saltadas"] == 1
    np.testing.assert_allclose(precios["GDAES_D+1"].to_numpy(), [28.50, 29.10])


# --------------------------------------------------------------------------
# El informe de carga
# --------------------------------------------------------------------------


def test_las_celdas_ilegibles_se_cuentan_en_vez_de_taparse(tmp_path):
    # pd.to_numeric(errors="coerce") las volvia NaN y el ffill las tapaba con
    # el precio de ayer, indistinguible de un dato real.
    lineas = ["fecha,AAA"] + [f"2024-{m:02d}-01,{100 + m}" for m in range(1, 13)]
    lineas[3] = "2024-03-01,45.67 EUR/MWh"
    ruta = tmp_path / "sucio.csv"
    ruta.write_text("\n".join(lineas))
    precios = cargar_precios_fichero(ruta)
    informe = precios.attrs["informe_carga"]
    assert informe["celdas_ilegibles"]["AAA"] == 1
    assert "45.67 EUR/MWh" in informe["ejemplos_ilegibles"]["AAA"][0]


def test_demasiadas_celdas_ilegibles_detienen_la_carga(tmp_path):
    lineas = ["fecha,AAA"] + [
        f"2024-{m:02d}-01," + ("n.d." if m % 3 else str(100 + m)) for m in range(1, 13)
    ]
    ruta = tmp_path / "muysucio.csv"
    ruta.write_text("\n".join(lineas))
    with pytest.raises(ErrorDatos, match="no se han podido leer como numero"):
        cargar_precios_fichero(ruta)


def test_el_informe_sobrevive_al_viaje_por_parquet(tmp_path):
    # Un dataclass en attrs no sobrevive: pandas serializa attrs a JSON y, al
    # no poder, descarta el diccionario entero con un simple aviso.
    ruta = tmp_path / "p.csv"
    ruta.write_text("fecha,AAA\n2024-01-02,100\n2024-01-03,101\n")
    precios = cargar_precios_fichero(ruta)
    destino = tmp_path / "p.parquet"
    precios.to_parquet(destino)
    recuperado = pd.read_parquet(destino)
    assert recuperado.attrs["informe_carga"]["filas_leidas"] == 2
    assert "Filas leidas" in resumen_carga(recuperado)


# --------------------------------------------------------------------------
# El cache y el recorte por activos
# --------------------------------------------------------------------------


def test_dos_ficheros_distintos_no_comparten_entrada_de_cache(tmp_path):
    # 'nombre' valia "fichero" para todos, asi que la clave del cache no
    # dependia del fichero y el segundo recibia los datos del primero.
    a = tmp_path / "ibex.csv"
    b = tmp_path / "dax.csv"
    a.write_text("fecha,ACT\n2024-01-02,100\n2024-01-03,101\n")
    b.write_text("fecha,ACT\n2024-01-02,900\n2024-01-03,901\n")
    assert ProveedorFichero(a).nombre != ProveedorFichero(b).nombre


def test_modificar_el_fichero_cambia_la_huella(tmp_path):
    # El cache solo miraba su propia antiguedad, nunca la del origen: con
    # caducidad_horas=None servia los datos viejos para siempre.
    ruta = tmp_path / "serie.csv"
    ruta.write_text("fecha,ACT\n2024-01-02,100\n2024-01-03,101\n")
    antes = ProveedorFichero(ruta).nombre
    ruta.write_text("fecha,ACT\n2024-01-02,100\n2024-01-03,101\n2024-01-04,102\n")
    assert ProveedorFichero(ruta).nombre != antes


def test_pedir_un_activo_no_lo_recorta_por_culpa_de_otro(tmp_path):
    # La limpieza se hacia sobre todas las columnas del fichero antes del
    # recorte, asi que una serie que empieza mas tarde borraba filas de las
    # demas aunque nadie la hubiese pedido.
    ruta = tmp_path / "dos.csv"
    ruta.write_text(
        "fecha;ES;FR\n02/01/2024;45,00;\n31/01/2024;46,00;\n"
        "01/02/2024;47,00;80,00\n05/02/2024;48,00;81,00\n"
    )
    solo_es = cargar_precios_fichero(ruta, ["ES"], separador=";", decimal=",")
    assert len(solo_es) == 4


def test_una_columna_repetida_al_normalizar_se_denuncia(tmp_path):
    # 'es' y 'ES' colapsan al pasar a mayusculas, y entonces marco[columna]
    # devolvia un DataFrame y el camino de texto reventaba con AttributeError.
    ruta = tmp_path / "rep.csv"
    ruta.write_text("fecha,es,ES\n2024-01-02,100,200\n2024-01-03,101,201\n")
    with pytest.raises(ErrorDatos, match="mismo nombre"):
        cargar_precios_fichero(ruta)
