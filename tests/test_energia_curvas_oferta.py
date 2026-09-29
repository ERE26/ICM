"""Pruebas del lector de curvas de oferta de OMIE y del impacto en el precio.

Lo que hay que asegurar aqui son dos cosas distintas. La primera es que el
lector reconoce las tres variantes de formato del fichero **y convierte la
potencia a energia solo cuando toca**: ese es el error que no se nota con
ficheros horarios y sobrestima por cuatro con los cuarto-horarios.

La segunda es que el impacto tiene el signo y la forma correctos. Anadir
demanda no puede bajar el precio y anadir oferta no puede subirlo, y el
impacto tiene que crecer con la potencia. Sobre curvas construidas a mano el
valor exacto se puede calcular, asi que se comprueba el numero y no solo el
sentido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.energia.curvas_oferta import (
    ELASTICIDAD_OMIE,
    ETIQUETAS_TRAMO,
    POTENCIAS_MEDIDAS,
    casacion,
    contrastar_casacion,
    impacto_estimado,
    impacto_precio,
    leer_curva_agregada,
    margen_con_impacto,
    tramo_de_precio,
)


CABECERA = "OMIE - Curva agregada de oferta y demanda\nUnidades\nPeriodo;Fecha;Pais;Unidad;Tipo;Energia;Precio;Ofertada (O)/Casada (C)\n"


def escribir_curva(ruta, filas, etiquetas=None) -> None:
    """Deja en disco un fichero con el formato exacto que publica OMIE.

    Las filas son ``(periodo, tipo, cantidad, precio, estado)``. Se escribe con
    punto y coma, coma decimal y latin-1 porque es justo lo que hay que saber
    interpretar, y un lector que solo funcione con ficheros bien formados no
    sirve para estos datos.
    """
    lineas = [CABECERA]
    for periodo, tipo, cantidad, precio, estado in filas:
        cant = f"{cantidad:.1f}".replace(".", ",")
        prec = f"{precio:.2f}".replace(".", ",")
        lineas.append(f"{periodo};17/01/2024;MI;UNI;{tipo};{cant};{prec};{estado};\n")
    ruta.write_text("".join(lineas), encoding="latin-1")


def curva_simple(tmp_path, n_periodos=24, etiqueta_hxqy=False):
    """Curva con una pila de venta escalonada conocida y demanda inelastica.

    La venta ofrece 100 MWh a 10, 100 a 20, 100 a 30 y 100 a 100 EUR/MWh; la
    compra pide 150 MWh a 200. Asi la casacion queda en 20 y el impacto de
    anadir demanda es calculable a mano.
    """
    filas = []
    for i in range(n_periodos):
        if etiqueta_hxqy:
            periodo = f"H{i // 4 + 1}Q{i % 4 + 1}"
        else:
            periodo = str(i + 1)
        for precio, estado in ((10.0, "C"), (20.0, "C"), (30.0, "O"), (100.0, "O")):
            filas.append((periodo, "V", 100.0, precio, estado))
        filas.append((periodo, "C", 150.0, 200.0, "C"))
        filas.append((periodo, "C", 100.0, 5.0, "O"))
    ruta = tmp_path / "curva.txt"
    escribir_curva(ruta, filas)
    return ruta


def test_lee_el_formato_de_horas_enteras(tmp_path):
    curva = leer_curva_agregada(curva_simple(tmp_path))
    assert curva.attrs["n_periodos"] == 24
    assert curva.attrs["horas_por_periodo"] == 1.0
    assert curva.attrs["fecha"] == "2024-01-17"
    assert set(curva["hora"]) == set(range(1, 25))
    assert (curva["cuarto"] == 1).all()


def test_con_periodos_horarios_la_energia_coincide_con_la_cantidad(tmp_path):
    curva = leer_curva_agregada(curva_simple(tmp_path))
    assert np.allclose(curva["energia"], curva["cantidad"])


def test_con_periodos_cuartohorarios_la_energia_es_la_cuarta_parte(tmp_path):
    """El error que este modulo existe para no cometer.

    En el formato nuevo la cantidad es potencia en MW. Sumarla como si fuera
    energia multiplicaria por cuatro el volumen del mercado, y el punto de
    partida del impacto saldria cuatro veces mas a la derecha de la pila.
    """
    curva = leer_curva_agregada(curva_simple(tmp_path, n_periodos=96, etiqueta_hxqy=True))
    assert curva.attrs["n_periodos"] == 96
    assert curva.attrs["horas_por_periodo"] == 0.25
    assert np.allclose(curva["energia"], curva["cantidad"] * 0.25)
    assert set(curva["cuarto"]) == {1, 2, 3, 4}
    assert set(curva["hora"]) == set(range(1, 25))


def test_acepta_los_dias_de_cambio_de_hora(tmp_path):
    """23 y 25 periodos son dias legitimos, no ficheros corruptos."""
    for n in (23, 25):
        curva = leer_curva_agregada(curva_simple(tmp_path, n_periodos=n))
        assert curva.attrs["n_periodos"] == n
        assert curva.attrs["horas_por_periodo"] == 1.0


def test_un_numero_de_periodos_imposible_se_rechaza(tmp_path):
    ruta = curva_simple(tmp_path, n_periodos=7)
    with pytest.raises(ErrorDatos, match="no se puede deducir su"):
        leer_curva_agregada(ruta)


def test_un_fichero_que_no_existe_lo_dice(tmp_path):
    with pytest.raises(ErrorDatos, match="No existe"):
        leer_curva_agregada(tmp_path / "no_esta.txt")


def test_la_casacion_sale_de_las_ofertas_casadas(tmp_path):
    """El maximo de venta casada es 20 y el minimo de compra casada 200."""
    curva = leer_curva_agregada(curva_simple(tmp_path))
    cas = casacion(curva)
    assert len(cas) == 24
    assert np.allclose(cas["venta_maxima"], 20.0)
    assert np.allclose(cas["compra_minima"], 200.0)
    assert np.allclose(cas["energia"], 200.0)


def test_el_contraste_reconoce_el_precio_que_sale_de_la_pila(tmp_path):
    curva = leer_curva_agregada(curva_simple(tmp_path))
    publicados = pd.Series([20.0] * 24)
    informe = contrastar_casacion(curva, publicados)
    assert informe["exactos"] == 24.0
    assert informe["proporcion_exactos"] == 1.0
    assert informe["fuera"] == 0.0


def test_el_contraste_distingue_estar_en_el_cruce_de_estar_fuera(tmp_path):
    """Un precio entre los dos extremos no es un fallo de lectura.

    Es lo que produce el redondeo del fichero y la casacion parcial de la
    unidad marginal, y por eso se cuenta aparte: sobre datos reales son el
    15,5 % de los periodos frente a un 2,4 % que de verdad queda fuera.
    """
    curva = leer_curva_agregada(curva_simple(tmp_path))
    dentro = pd.Series([100.0] * 24)
    assert contrastar_casacion(curva, dentro)["en_el_cruce"] == 24.0
    fuera = pd.Series([500.0] * 24)
    assert contrastar_casacion(curva, fuera)["fuera"] == 24.0


def test_el_impacto_de_comprar_camina_la_pila_de_venta(tmp_path):
    """Con la pila de la fixture el valor exacto es calculable.

    Casa en 20 con 200 MWh de venta acumulada. Anadir 100 MW lleva la
    acumulada a 300, que es donde termina el tramo de 30 EUR/MWh, asi que el
    precio marginal pasa a 30 y el impacto es exactamente 10.
    """
    curva = leer_curva_agregada(curva_simple(tmp_path))
    imp = impacto_precio(curva, potencias=(100.0,), precios_publicados=pd.Series([20.0] * 24))
    assert np.allclose(imp["oferta_mw"], 200.0)
    assert np.allclose(imp["sube_al_comprar_100"], 10.0)


def test_el_impacto_crece_con_la_potencia(tmp_path):
    curva = leer_curva_agregada(curva_simple(tmp_path))
    imp = impacto_precio(curva, precios_publicados=pd.Series([20.0] * 24))
    columnas = [f"sube_al_comprar_{p:g}" for p in POTENCIAS_MEDIDAS]
    medianas = [imp[c].median() for c in columnas]
    assert all(b >= a for a, b in zip(medianas, medianas[1:]))


def test_el_impacto_nunca_tiene_el_signo_equivocado(tmp_path):
    """Anadir demanda no puede abaratar y anadir oferta no puede encarecer."""
    curva = leer_curva_agregada(curva_simple(tmp_path))
    imp = impacto_precio(curva, precios_publicados=pd.Series([20.0] * 24))
    for p in POTENCIAS_MEDIDAS:
        assert (imp[f"sube_al_comprar_{p:g}"] >= -1e-9).all()
        assert (imp[f"baja_al_vender_{p:g}"] >= -1e-9).all()


def test_pocos_precios_publicados_para_los_periodos_se_rechaza(tmp_path):
    curva = leer_curva_agregada(curva_simple(tmp_path))
    with pytest.raises(ErrorDatos, match="precios publicados"):
        impacto_precio(curva, precios_publicados=pd.Series([20.0] * 3))


def test_una_curva_que_no_viene_del_lector_se_rechaza():
    with pytest.raises(ErrorDatos, match="columnas"):
        casacion(pd.DataFrame({"hora": [1], "precio": [10.0]}))
    with pytest.raises(ErrorDatos, match="DataFrame"):
        casacion([1, 2, 3])


def test_los_tramos_de_precio_cubren_toda_la_recta():
    """Incluidos los precios negativos, que en OMIE son rutina."""
    precios = pd.Series([-50.0, 0.0, 3.0, 10.0, 40.0, 90.0, 300.0])
    tramos = tramo_de_precio(precios)
    assert not tramos.isna().any()
    assert list(tramos) == ["<5", "<5", "<5", "5-25", "25-60", "60-120", ">120"]


def test_la_tabla_de_elasticidad_esta_completa_y_es_creciente():
    """Una tabla con un hueco o no monotona daria impactos sin sentido."""
    for etiqueta in ETIQUETAS_TRAMO:
        assert etiqueta in ELASTICIDAD_OMIE
        for direccion in ("compra", "venta"):
            valores = ELASTICIDAD_OMIE[etiqueta][direccion]
            assert len(valores) == len(POTENCIAS_MEDIDAS)
            assert all(v > 0.0 for v in valores)
            assert all(b >= a for a, b in zip(valores, valores[1:]))


def test_la_asimetria_medida_esta_en_la_tabla():
    """El hallazgo central del modulo, fijado como prueba.

    En el tramo caro, que es donde una bateria descarga, vender hunde el
    precio mucho mas de lo que comprar lo sube. Si algun dia se rehace la
    tabla con mas sesiones y esta relacion se invierte, la conclusion
    economica del modulo cambia y hay que reescribir la cabecera, no
    silenciar la prueba.
    """
    caro = ELASTICIDAD_OMIE[">120"]
    assert caro["venta"][3] > 5.0 * caro["compra"][3]
    barato = ELASTICIDAD_OMIE["<5"]
    assert barato["compra"][3] < 1.0


def test_el_impacto_estimado_interpola_en_potencia():
    precios = pd.Series([150.0] * 4)
    tabla = ELASTICIDAD_OMIE[">120"]["compra"]
    exacto = impacto_estimado(precios, 500.0, "compra")
    assert np.allclose(exacto, tabla[3])
    medio = impacto_estimado(precios, 750.0, "compra")
    assert np.allclose(medio, (tabla[3] + tabla[4]) / 2.0)


def test_el_impacto_estimado_de_un_megavatio_es_casi_nulo():
    """Que es lo que justifica tratar 1 MW como precio aceptante."""
    precios = pd.Series([150.0] * 4)
    assert (impacto_estimado(precios, 1.0, "compra") < 0.01).all()


def test_el_impacto_estimado_de_potencia_nula_es_nulo():
    precios = pd.Series([150.0] * 4)
    assert np.allclose(impacto_estimado(precios, 0.0, "venta"), 0.0)


def test_una_direccion_inventada_se_rechaza():
    with pytest.raises(ErrorDatos, match="compra' o 'venta"):
        impacto_estimado(pd.Series([10.0]), 100.0, "arbitraje")


def test_una_potencia_negativa_se_rechaza():
    with pytest.raises(ErrorDatos, match="no puede ser negativa"):
        impacto_estimado(pd.Series([10.0]), -5.0, "compra")


# --- El margen con impacto, que es donde esto se aplica ------------------


def perfil_barato_y_caro(dias=20):
    """Precios con seis horas a 2 EUR/MWh y seis a 150, que es el caso claro.

    Se elige asi porque pone las horas de carga en el tramo ``<5`` y las de
    descarga en el ``>120``, que son los dos extremos de la asimetria medida.
    """
    horas = np.tile(
        np.concatenate([np.full(6, 2.0), np.full(6, 40.0), np.full(6, 150.0), np.full(6, 80.0)]),
        dias,
    )
    indice = pd.date_range("2024-01-01", periods=len(horas), freq="h")
    return pd.Series(horas, index=indice, name="precio")


def test_el_impacto_reduce_el_margen_y_nunca_lo_aumenta():
    from icm.energia.almacenamiento import ParametrosAlmacenamiento, despacho_optimo

    precios = perfil_barato_y_caro()
    ficha = ParametrosAlmacenamiento(nombre="100 MW", capacidad=400.0, potencia=100.0)
    informe = margen_con_impacto(despacho_optimo(precios, ficha), precios)
    assert informe["coste_impacto"] > 0.0
    assert informe["margen_con_impacto"] < informe["margen_bruto"]
    assert np.isclose(
        informe["coste_impacto"],
        informe["coste_impacto_carga"] + informe["coste_impacto_descarga"],
    )


def test_la_descarga_aporta_mas_coste_de_impacto_que_la_carga():
    """La consecuencia practica de la asimetria de las curvas.

    Con este perfil la bateria mueve la misma energia al cargar que al
    descargar --menos el rendimiento-- y aun asi el coste de descargar es
    mayor, porque la curva de demanda es mas inelastica que la de oferta.
    """
    from icm.energia.almacenamiento import ParametrosAlmacenamiento, despacho_optimo

    precios = perfil_barato_y_caro()
    ficha = ParametrosAlmacenamiento(nombre="100 MW", capacidad=400.0, potencia=100.0)
    informe = margen_con_impacto(despacho_optimo(precios, ficha), precios)
    assert informe["coste_impacto_descarga"] > informe["coste_impacto_carga"]


def test_la_merma_crece_con_la_escala_del_activo():
    """Que es la razon por la que el VAN no escala con el tamanio."""
    from icm.energia.almacenamiento import ParametrosAlmacenamiento, despacho_optimo

    precios = perfil_barato_y_caro()
    mermas = []
    for potencia in (1.0, 10.0, 100.0, 500.0):
        ficha = ParametrosAlmacenamiento(
            nombre=f"{potencia:g} MW", capacidad=4.0 * potencia, potencia=potencia
        )
        informe = margen_con_impacto(despacho_optimo(precios, ficha), precios)
        mermas.append(informe["merma_por_ciento"])
    assert all(b > a for a, b in zip(mermas, mermas[1:]))
    assert mermas[0] < 0.5


def test_unos_precios_de_otra_longitud_se_rechazan():
    """Liquidar un plan con precios que no son los suyos daria un numero falso."""
    from icm.energia.almacenamiento import ParametrosAlmacenamiento, despacho_optimo

    precios = perfil_barato_y_caro()
    ficha = ParametrosAlmacenamiento(nombre="100 MW", capacidad=400.0, potencia=100.0)
    despacho = despacho_optimo(precios, ficha)
    with pytest.raises(ErrorDatos, match="periodos y se han dado"):
        margen_con_impacto(despacho, precios.head(10))


def test_algo_que_no_es_un_despacho_se_rechaza():
    with pytest.raises(ErrorDatos, match="Despacho"):
        margen_con_impacto(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))


def test_el_van_con_merma_es_menor_que_sin_ella():
    """La conexion con la valoracion: el impacto tiene que llegar al VAN."""
    from icm.energia.almacenamiento import ParametrosAlmacenamiento, van_almacenamiento

    precios = perfil_barato_y_caro(dias=60)
    ficha = ParametrosAlmacenamiento(nombre="100 MW", capacidad=400.0, potencia=100.0)
    sin_merma = van_almacenamiento(precios, ficha)
    con_merma = van_almacenamiento(precios, ficha, merma_por_impacto=0.087)
    assert con_merma["van"] < sin_merma["van"]
    assert con_merma["merma_por_impacto"] == 0.087
    assert (
        con_merma["capex_equilibrio_por_kwh"] < sin_merma["capex_equilibrio_por_kwh"]
    )


def test_una_merma_fuera_de_rango_se_rechaza():
    from icm.energia.almacenamiento import van_almacenamiento

    precios = perfil_barato_y_caro(dias=20)
    for mala in (-0.1, 1.0, 1.5):
        with pytest.raises(ErrorDatos, match="merma_por_impacto"):
            van_almacenamiento(precios, "bateria_4h", merma_por_impacto=mala)
