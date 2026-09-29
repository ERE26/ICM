"""Pruebas del lector de documentos de ENTSO-E.

Sin token no se puede ejecutar ni una descarga real, asi que estas pruebas
cubren lo que si es verificable y que resulta ser la parte donde estan los
errores de verdad: **la interpretacion del documento**. El formato es un
esquema publicado, la estructura no depende del indicador, y los casos que
rompen a un lector ingenuo se pueden construir a mano.

Los tres que importan: varias series en un documento, huecos en las
posiciones --el esquema permite omitir un punto que repite el valor anterior,
asi que la posicion **no** es el numero de fila-- y resoluciones distintas de
la horaria.
"""

from __future__ import annotations

import pandas as pd
import pytest

from icm.comun import ErrorDatos, ErrorDescarga
from icm.datos.entsoe import (
    VARIABLE_TOKEN,
    ZONA_ESPANA,
    consultar,
    descargar_banda_secundaria,
    horas_con_banda,
    leer_documento,
    parametros_banda_secundaria,
    token_disponible,
)


ESPACIO = "urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:0"


def documento(series) -> str:
    """Arma un documento segun el esquema, con las series que se le pasen.

    Cada serie es ``(mRID, direccion, inicio, resolucion, [(posicion, valor)])``.
    Se escribe con el espacio de nombres real para que las pruebas ejerciten
    la resolucion por comodin y no una version simplificada del formato.
    """
    partes = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<Balancing_MarketDocument xmlns="{ESPACIO}">',
        "<mRID>prueba</mRID>",
    ]
    for mrid, direccion, inicio, resolucion, puntos in series:
        partes.append("<TimeSeries>")
        partes.append(f"<mRID>{mrid}</mRID>")
        if direccion:
            partes.append(f"<flowDirection.direction>{direccion}</flowDirection.direction>")
        partes.append("<Period>")
        partes.append(f"<timeInterval><start>{inicio}</start></timeInterval>")
        partes.append(f"<resolution>{resolucion}</resolution>")
        for posicion, valor in puntos:
            partes.append(
                f"<Point><position>{posicion}</position>"
                f"<quantity>{valor}</quantity></Point>"
            )
        partes.append("</Period></TimeSeries>")
    partes.append("</Balancing_MarketDocument>")
    return "".join(partes)


def test_lee_una_serie_horaria_sencilla():
    xml = documento([("subir", "A01", "2024-01-15T00:00Z", "PT60M",
                      [(1, 500.0), (2, 520.0), (3, 480.0)])])
    tabla = leer_documento(xml)
    assert len(tabla) == 3
    assert list(tabla["valor"]) == [500.0, 520.0, 480.0]
    assert tabla["momento"].iloc[0] == pd.Timestamp("2024-01-15T00:00Z")
    assert tabla["momento"].iloc[2] == pd.Timestamp("2024-01-15T02:00Z")
    assert set(tabla["serie"]) == {"subir"}
    assert set(tabla["direccion"]) == {"A01"}


def test_un_hueco_en_las_posiciones_no_desplaza_la_serie():
    """El error que este lector existe para no cometer.

    El esquema permite omitir un punto cuando repite el valor del anterior.
    Un lector que use el orden de aparicion como indice desplazaria **toda**
    la serie a partir del primer hueco, y el resultado seguiria pareciendo una
    serie perfectamente razonable.
    """
    xml = documento([("subir", None, "2024-01-15T00:00Z", "PT60M",
                      [(1, 500.0), (2, 520.0), (5, 600.0)])])
    tabla = leer_documento(xml)
    assert len(tabla) == 3
    # El tercer punto es la posicion 5, o sea las 04:00, no las 02:00.
    assert tabla["momento"].iloc[2] == pd.Timestamp("2024-01-15T04:00Z")
    assert tabla["valor"].iloc[2] == 600.0


def test_separa_las_series_del_mismo_documento():
    """Subir y bajar vienen en el mismo documento y no se pueden mezclar."""
    xml = documento([
        ("subir", "A01", "2024-01-15T00:00Z", "PT60M", [(1, 500.0), (2, 520.0)]),
        ("bajar", "A02", "2024-01-15T00:00Z", "PT60M", [(1, 400.0), (2, 410.0)]),
    ])
    tabla = leer_documento(xml)
    assert len(tabla) == 4
    assert set(tabla["serie"]) == {"subir", "bajar"}
    subir = tabla[tabla["serie"] == "subir"]
    assert list(subir["valor"]) == [500.0, 520.0]


def test_entiende_las_resoluciones_de_quince_minutos_y_de_un_dia():
    cuartos = documento([("s", None, "2024-01-15T00:00Z", "PT15M",
                          [(1, 10.0), (2, 20.0), (5, 50.0)])])
    tabla = leer_documento(cuartos)
    assert tabla["momento"].iloc[1] == pd.Timestamp("2024-01-15T00:15Z")
    assert tabla["momento"].iloc[2] == pd.Timestamp("2024-01-15T01:00Z")

    diaria = documento([("s", None, "2024-01-15T00:00Z", "P1D", [(1, 10.0), (3, 30.0)])])
    tabla = leer_documento(diaria)
    assert tabla["momento"].iloc[1] == pd.Timestamp("2024-01-17T00:00Z")


def test_una_resolucion_desconocida_se_rechaza_en_vez_de_adivinarse():
    xml = documento([("s", None, "2024-01-15T00:00Z", "PT7M", [(1, 10.0)])])
    with pytest.raises(ErrorDatos, match="Resolucion"):
        leer_documento(xml)


def test_un_acuse_de_recibo_se_distingue_de_un_documento_con_datos():
    """Es lo que devuelve la API sin token, y confundirlo daria una tabla vacia."""
    acuse = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:'
        'acknowledgementdocument:7:0">'
        "<Reason><code>999</code><text>Authentication failed.</text></Reason>"
        "</Acknowledgement_MarketDocument>"
    )
    with pytest.raises(ErrorDatos, match="acuse de recibo"):
        leer_documento(acuse)


def test_un_xml_roto_lo_dice_en_vez_de_devolver_nada():
    with pytest.raises(ErrorDatos, match="no es XML valido"):
        leer_documento("<Balancing_MarketDocument><sin cerrar>")


def test_pedir_precios_a_un_documento_de_volumenes_avisa_del_nombre():
    xml = documento([("s", None, "2024-01-15T00:00Z", "PT60M", [(1, 10.0)])])
    with pytest.raises(ErrorDatos, match="price.amount"):
        leer_documento(xml, valor="price.amount")


def test_sin_token_no_se_intenta_la_descarga_y_se_explica_como_pedirlo(monkeypatch):
    """Falla antes de tocar la red y con la instruccion completa.

    Importa que falle **antes**: un error de red tardio no dice que lo que
    falta es una gestion administrativa de tres pasos.
    """
    monkeypatch.delenv(VARIABLE_TOKEN, raising=False)
    with pytest.raises(ErrorDescarga, match="transparency@entsoe.eu"):
        consultar({"documentType": "A81"})
    with pytest.raises(ErrorDescarga, match=VARIABLE_TOKEN):
        descargar_banda_secundaria("2024-01-01", "2024-01-02")


def test_token_disponible_no_miente_en_ninguno_de_los_dos_sentidos(monkeypatch):
    monkeypatch.delenv(VARIABLE_TOKEN, raising=False)
    assert not token_disponible()
    assert token_disponible("un-token")
    monkeypatch.setenv(VARIABLE_TOKEN, "  ")
    assert not token_disponible(), "solo espacios no es un token"
    monkeypatch.setenv(VARIABLE_TOKEN, "abc123")
    assert token_disponible()


def test_los_parametros_llevan_la_zona_y_el_formato_de_fecha_del_protocolo():
    p = parametros_banda_secundaria("2024-03-01", "2024-04-01")
    assert p["controlArea_Domain"] == ZONA_ESPANA
    assert p["periodStart"] == "202403010000"
    assert p["periodEnd"] == "202404010000"
    assert p["processType"] == "A51", "A51 es aFRR, la secundaria europea"
    assert p["documentType"] == "A81"
    assert parametros_banda_secundaria("2024-03-01", "2024-04-01", precios=True)[
        "documentType"] == "A89"


def test_una_fecha_con_huso_se_pasa_a_utc_antes_de_formatear():
    """El protocolo solo entiende UTC, y mandar hora local desplaza el periodo."""
    p = parametros_banda_secundaria(
        pd.Timestamp("2024-03-01 00:00", tz="Europe/Madrid"),
        pd.Timestamp("2024-03-02 00:00", tz="Europe/Madrid"),
    )
    assert p["periodStart"] == "202402292300"


def test_un_proceso_inventado_se_rechaza():
    with pytest.raises(ErrorDatos, match="desconocido"):
        parametros_banda_secundaria("2024-01-01", "2024-01-02", proceso="terciaria")


def test_las_horas_con_banda_son_una_cota_superior():
    asignacion = pd.DataFrame({"valor": [50.0, 150.0, 300.0, 80.0]})
    assert horas_con_banda(asignacion, 100.0) == 0.5
    assert horas_con_banda(asignacion, 10.0) == 1.0
    assert horas_con_banda(asignacion, 1000.0) == 0.0


def test_las_horas_con_banda_rechazan_entradas_sin_sentido():
    asignacion = pd.DataFrame({"valor": [50.0, 150.0]})
    with pytest.raises(ErrorDatos, match="positiva"):
        horas_con_banda(asignacion, 0.0)
    with pytest.raises(ErrorDatos, match="vacia"):
        horas_con_banda(pd.DataFrame({"valor": []}), 100.0)
    with pytest.raises(ErrorDatos, match="leer_documento"):
        horas_con_banda(pd.DataFrame({"otra": [1.0]}), 100.0)
