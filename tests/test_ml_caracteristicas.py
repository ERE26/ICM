"""Pruebas de la ingenieria de variables y de la auditoria de fuga temporal.

El bloque que de verdad importa es el ultimo: no basta con que
``comprobar_ausencia_de_fuga`` acepte la matriz que construye el modulo, hay
que ver que rechaza fugas puestas a proposito. Una comprobacion que nunca
falla no comprueba nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.datos import simular_precios
from icm.ml.caracteristicas import (
    FAMILIAS,
    MatrizDatos,
    _comparar,
    _perturbar_futuro,
    calendario,
    comprobar_ausencia_de_fuga,
    construir_matriz,
    medias_moviles,
    momento,
    objetivo_rendimiento_futuro,
    retardos,
    volatilidad_realizada,
)

VENTANA_ANUAL = 252


@pytest.fixture(scope="module")
def precios_cortos() -> pd.DataFrame:
    """Serie corta para las pruebas que reconstruyen la matriz muchas veces."""
    return simular_precios(["ALFA", "BETA", "GAMA"], n_periodos=500, semilla=3)


@pytest.fixture
def serie_precios() -> pd.Series:
    """Cinco precios que duplican cada dia: los rendimientos son log 2 exactos."""
    indice = pd.bdate_range("2021-01-04", periods=5, name="fecha")
    return pd.Series([1.0, 2.0, 4.0, 8.0, 16.0], index=indice, name="ALFA")


@pytest.fixture
def exogena() -> pd.Series:
    """Prevision meteorologica sintetica alineada con el calendario de precios."""
    indice = pd.bdate_range("2015-01-01", periods=500, name="fecha")
    estacional = 12.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(500) / VENTANA_ANUAL)
    return pd.Series(estacional, index=indice, name="temperatura")


# --- retardos ------------------------------------------------------------------


def test_los_retardos_traen_el_pasado_y_no_el_presente(serie_precios):
    salida = retardos(serie_precios, retardos=(1, 2))

    assert list(salida.columns) == ["ALFA_ret1", "ALFA_ret2"]
    assert salida["ALFA_ret1"].tolist()[1:] == [1.0, 2.0, 4.0, 8.0]
    assert np.isnan(salida["ALFA_ret1"].iloc[0])
    assert salida["ALFA_ret2"].tolist()[2:] == [1.0, 2.0, 4.0]


def test_el_retardo_cero_es_el_valor_del_propio_dia(serie_precios):
    salida = retardos(serie_precios, retardos=(0,))
    pd.testing.assert_series_equal(
        salida["ALFA_ret0"], serie_precios, check_names=False
    )


def test_un_retardo_negativo_se_rechaza_por_mirar_al_futuro(serie_precios):
    with pytest.raises(ErrorDatos, match="futuro"):
        retardos(serie_precios, retardos=(-1,))


def test_los_retardos_de_un_marco_conservan_el_activo_de_origen(precios_cortos):
    salida = retardos(precios_cortos, retardos=(1, 3), prefijo="cruz")

    assert list(salida.columns) == [
        "cruz_ALFA_ret1",
        "cruz_ALFA_ret3",
        "cruz_BETA_ret1",
        "cruz_BETA_ret3",
        "cruz_GAMA_ret1",
        "cruz_GAMA_ret3",
    ]
    assert salida.index.equals(precios_cortos.index)


# --- medias moviles ------------------------------------------------------------


def test_la_media_movil_relativa_tiene_el_valor_analitico(serie_precios):
    salida = medias_moviles(serie_precios, ventanas=(3,), relativa=True)

    # Media de (1, 2, 4) = 7/3; el precio de esa fecha es 4.
    assert salida["ALFA_mm3_rel"].iloc[2] == pytest.approx(4.0 / (7.0 / 3.0) - 1.0)
    assert salida["ALFA_mm3_rel"].iloc[:2].isna().all()


def test_la_media_movil_relativa_no_depende_de_la_escala_del_activo(precios_cortos):
    """La razon de ser de relativa=True: un activo de 20 y otro de 200 comparables."""
    serie = precios_cortos["ALFA"]
    original = medias_moviles(serie, relativa=True)
    reescalada = medias_moviles(serie * 137.0, relativa=True)

    pd.testing.assert_frame_equal(original, reescalada)

    en_niveles = medias_moviles(serie, ventanas=(5,), relativa=False)
    en_niveles_reescalada = medias_moviles(serie * 137.0, ventanas=(5,), relativa=False)
    assert not np.allclose(
        en_niveles.dropna().to_numpy(), en_niveles_reescalada.dropna().to_numpy()
    )


def test_la_media_movil_solo_usa_la_ventana_que_termina_en_la_fecha(precios_cortos):
    serie = precios_cortos["ALFA"]
    salida = medias_moviles(serie, ventanas=(10,), relativa=False)

    posicion = 120
    esperado = serie.iloc[posicion - 9 : posicion + 1].mean()
    assert salida["ALFA_mm10"].iloc[posicion] == pytest.approx(esperado)


def test_la_media_movil_relativa_rechaza_series_no_positivas():
    serie = pd.Series([1.0, 0.0, -1.0], index=pd.bdate_range("2021-01-04", periods=3))
    with pytest.raises(ErrorDatos, match="estrictamente positiva"):
        medias_moviles(serie, ventanas=(2,), relativa=True)


# --- volatilidad y momento -----------------------------------------------------


def test_la_volatilidad_de_una_ventana_de_dos_tiene_formula_cerrada():
    indice = pd.bdate_range("2021-01-04", periods=4)
    rend = pd.Series([0.0, 0.02, 0.0, 0.02], index=indice, name="rend")

    salida = volatilidad_realizada(rend, ventanas=(2,))

    # Con dos datos, la cuasidesviacion tipica es |x - y| / sqrt(2).
    esperado = 0.02 / np.sqrt(2.0) * np.sqrt(VENTANA_ANUAL)
    assert salida["rend_vol2"].iloc[1:].tolist() == pytest.approx([esperado] * 3)
    assert np.isnan(salida["rend_vol2"].iloc[0])


def test_la_volatilidad_realizada_recupera_la_volatilidad_del_proceso():
    generador = np.random.default_rng(20240101)
    sigma_diaria = 0.015
    rend = pd.Series(
        generador.standard_normal(4000) * sigma_diaria,
        index=pd.bdate_range("2000-01-03", periods=4000),
        name="rend",
    )

    salida = volatilidad_realizada(rend, ventanas=(63,))

    esperado = sigma_diaria * np.sqrt(VENTANA_ANUAL)
    assert salida["rend_vol63"].mean() == pytest.approx(esperado, rel=0.02)


def test_el_momento_es_el_rendimiento_acumulado_de_la_ventana(precios_cortos):
    serie = precios_cortos["ALFA"]
    rend = np.log(serie).diff().rename("rend")

    salida = momento(rend, ventanas=(21,))

    posicion = 200
    esperado = np.log(serie.iloc[posicion] / serie.iloc[posicion - 21])
    assert salida["rend_mom21"].iloc[posicion] == pytest.approx(esperado)
    assert salida["rend_mom21"].iloc[:20].isna().all()


@pytest.mark.parametrize("funcion", [volatilidad_realizada, momento])
def test_las_ventanas_moviles_dan_una_columna_por_ventana_y_serie(funcion, precios_cortos):
    rend = np.log(precios_cortos).diff()
    salida = funcion(rend, ventanas=(5, 21))

    assert salida.shape == (len(precios_cortos), 6)
    assert salida.index.equals(precios_cortos.index)


# --- calendario ----------------------------------------------------------------


def test_el_calendario_deja_fuera_una_categoria_de_dia_de_la_semana():
    indice = pd.date_range("2022-01-03", periods=14, freq="D")  # empieza en lunes
    salida = calendario(indice)

    assert list(salida.columns) == [
        "dia_martes",
        "dia_miercoles",
        "dia_jueves",
        "dia_viernes",
        "dia_sabado",
        "dia_domingo",
        "sin_anio",
        "cos_anio",
    ]
    indicadores = salida.filter(like="dia_")
    # El lunes es la referencia: se reconoce porque no enciende ningun indicador.
    assert indicadores.sum(axis=1).isin({0.0, 1.0}).all()
    assert (indicadores.iloc[0] == 0.0).all()
    assert salida.loc[indice[1], "dia_martes"] == 1.0


def test_el_calendario_exige_un_indice_de_fechas():
    with pytest.raises(ErrorDatos, match="DatetimeIndex"):
        calendario(pd.Index([1, 2, 3]))


def test_el_fin_de_anio_y_el_principio_quedan_contiguos_en_el_circulo():
    fechas = pd.DatetimeIndex(["2021-12-31", "2022-01-01", "2022-02-01"])
    ciclico = calendario(fechas)[["sin_anio", "cos_anio"]].to_numpy()

    def distancia(a, b):
        return float(np.linalg.norm(a - b))

    fin_de_anio = distancia(ciclico[0], ciclico[1])
    un_mes = distancia(ciclico[1], ciclico[2])

    # En el circulo, dos dias consecutivos estan mucho mas cerca que dos fechas
    # separadas por un mes, aunque una de ellas cambie de anio.
    assert fin_de_anio == pytest.approx(2.0 * np.sin(np.pi / 365.0), rel=1e-3)
    assert fin_de_anio < 0.05
    assert fin_de_anio < un_mes / 10.0

    # Con un numero de mes el orden se invierte: diciembre y enero pasarian a ser
    # el par mas alejado del calendario, que es justo lo que no queremos ensenar.
    meses = fechas.month.to_numpy(dtype=float)
    assert abs(meses[0] - meses[1]) == 11.0
    assert abs(meses[1] - meses[2]) == 1.0
    assert abs(meses[0] - meses[1]) > abs(meses[1] - meses[2])


# --- variable objetivo ---------------------------------------------------------


def test_el_objetivo_se_indexa_por_la_fecha_en_que_se_decide(serie_precios):
    y = objetivo_rendimiento_futuro(serie_precios, horizonte=1, tipo="log")

    # Precios 1, 2, 4, 8, 16: el rendimiento de cada dia al siguiente es log 2,
    # anotado en el dia en que se toma la decision, no en el siguiente.
    assert y.iloc[:4].tolist() == pytest.approx([np.log(2.0)] * 4)
    assert np.isnan(y.iloc[-1])
    assert y.index.equals(serie_precios.index)


def test_el_objetivo_a_dos_dias_acumula_dos_periodos(serie_precios):
    y = objetivo_rendimiento_futuro(serie_precios, horizonte=2, tipo="log")

    assert y.iloc[0] == pytest.approx(np.log(4.0))
    assert y.iloc[2] == pytest.approx(np.log(4.0))
    assert y.iloc[-2:].isna().all()


def test_el_objetivo_simple_es_la_variacion_porcentual(serie_precios):
    y = objetivo_rendimiento_futuro(serie_precios, horizonte=1, tipo="simple")
    assert y.iloc[0] == pytest.approx(1.0)


def test_el_objetivo_rechaza_horizontes_no_positivos_y_tipos_desconocidos(serie_precios):
    with pytest.raises(ErrorDatos, match="horizonte"):
        objetivo_rendimiento_futuro(serie_precios, horizonte=0)
    with pytest.raises(ErrorDatos, match="desconocido"):
        objetivo_rendimiento_futuro(serie_precios, tipo="geometrico")


# --- matriz completa -----------------------------------------------------------


def test_la_matriz_comparte_indice_y_no_deja_huecos(precios_cortos):
    datos = construir_matriz(precios_cortos, "ALFA", horizonte=3)

    assert datos.X.index.equals(datos.y.index)
    assert not datos.X.isna().any().any()
    assert not datos.y.isna().any()
    assert len(datos) == len(datos.X)
    # Las tres ultimas fechas no pueden tener objetivo: su futuro no ha ocurrido.
    assert datos.X.index[-1] == precios_cortos.index[-4]


def test_los_grupos_reparten_exactamente_las_columnas(precios_cortos):
    datos = construir_matriz(precios_cortos, "ALFA")

    reunidas = [c for columnas in datos.nombres_grupo.values() for c in columnas]
    assert sorted(reunidas) == sorted(datos.X.columns)
    assert len(reunidas) == len(set(reunidas))
    assert set(datos.nombres_grupo) == set(FAMILIAS)


@pytest.mark.parametrize(
    ("familia", "n_columnas", "muestra"),
    [
        ("retardos", 5, "rend_ret10"),
        ("medias", 3, "precio_mm21_rel"),
        ("volatilidad", 3, "rend_vol63"),
        ("momento", 3, "rend_mom63"),
        ("calendario", 8, "sin_anio"),
        ("cruzadas", 6, "cruz_BETA_ret1"),
    ],
)
def test_cada_familia_aporta_sus_propias_columnas(precios_cortos, familia, n_columnas, muestra):
    datos = construir_matriz(precios_cortos, "ALFA", incluir=(familia,))

    assert list(datos.nombres_grupo) == [familia]
    assert datos.X.shape[1] == n_columnas
    assert muestra in datos.X.columns


def test_las_cruzadas_traen_los_demas_activos_y_nunca_el_propio(precios_cortos):
    datos = construir_matriz(precios_cortos, "ALFA", incluir=("cruzadas",))

    assert all("ALFA" not in columna for columna in datos.X.columns)
    assert any("BETA" in columna for columna in datos.X.columns)
    assert any("GAMA" in columna for columna in datos.X.columns)


def test_las_exogenas_entran_con_los_retardos_pedidos(precios_cortos, exogena):
    datos = construir_matriz(
        precios_cortos,
        "ALFA",
        exogenas=exogena,
        retardos_exogenas=(0, 1),
        incluir=("retardos",),
    )

    assert datos.nombres_grupo["exogenas"] == ["exo_temperatura_ret0", "exo_temperatura_ret1"]
    fecha = datos.X.index[50]
    assert datos.X.loc[fecha, "exo_temperatura_ret0"] == pytest.approx(exogena.loc[fecha])


def test_una_exogena_sin_fechas_en_comun_da_error(precios_cortos):
    ajena = pd.Series(
        1.0, index=pd.bdate_range("1990-01-01", periods=100), name="temperatura"
    )
    with pytest.raises(ErrorDatos, match="no se solapa"):
        construir_matriz(precios_cortos, "ALFA", exogenas=ajena)


def test_un_objetivo_inexistente_dice_que_columnas_hay(precios_cortos):
    with pytest.raises(ErrorDatos, match="ALFA, BETA, GAMA"):
        construir_matriz(precios_cortos, "DELTA")


def test_una_familia_desconocida_enumera_las_disponibles(precios_cortos):
    with pytest.raises(ErrorDatos, match="desconocidas"):
        construir_matriz(precios_cortos, "ALFA", incluir=("astrologia",))


def test_una_serie_demasiado_corta_lo_dice_en_vez_de_devolver_una_matriz_vacia(precios_cortos):
    with pytest.raises(ErrorDatos, match="ninguna fila completa"):
        construir_matriz(precios_cortos.iloc[:30], "ALFA")


def test_un_indice_desordenado_se_rechaza_en_vez_de_ordenarse_en_silencio(precios_cortos):
    """Con las filas desordenadas, un retardo puede traer un valor posterior."""
    desordenados = precios_cortos.sample(frac=1.0, random_state=0)
    with pytest.raises(ErrorDatos, match="sort_index"):
        construir_matriz(desordenados, "ALFA")


def test_un_indice_con_fechas_repetidas_se_rechaza(precios_cortos):
    duplicados = pd.concat([precios_cortos, precios_cortos.iloc[[100]]]).sort_index()
    with pytest.raises(ErrorDatos, match="repetidas"):
        construir_matriz(duplicados, "ALFA")


def test_matriz_datos_rechaza_x_e_y_desalineados(precios_cortos):
    datos = construir_matriz(precios_cortos, "ALFA", incluir=("retardos",))
    with pytest.raises(ErrorDatos, match="no comparten indice"):
        MatrizDatos(X=datos.X, y=datos.y.iloc[:-5], nombres_grupo={})


# --- auditoria de fuga de informacion ------------------------------------------


def _matriz_desde(X: pd.DataFrame, y: pd.Series, familia: str) -> MatrizDatos:
    """Alinea X e y con un unico dropna, como hace construir_matriz."""
    completo = pd.concat([X, y.rename("__y__")], axis=1).dropna(how="any")
    return MatrizDatos(
        X=completo[X.columns],
        y=completo["__y__"],
        nombres_grupo={familia: list(X.columns)},
    )


def constructor_con_fuga_de_un_dia(precios, objetivo, horizonte=1, **_):
    """Constructor tramposo: mete el precio de maniana entre las explicativas.

    Es la fuga clasica y la mas dificil de ver, porque no rompe nada: la matriz
    tiene el mismo tamanio, el mismo indice y ni un solo NaN.
    """
    serie = precios[objetivo]
    X = pd.DataFrame(
        {
            "rend_ret1": np.log(serie).diff().shift(1),
            "precio_de_maniana": serie.shift(-1),
        }
    )
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "fuga")


def constructor_con_tipificacion_global(precios, objetivo, horizonte=1, **_):
    """Constructor tramposo: tipifica con la media y la desviacion de todo el historico."""
    serie = precios[objetivo]
    rend = np.log(serie).diff()
    X = pd.DataFrame({"rend_tipificado": (rend - rend.mean()) / rend.std()})
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "fuga")


def constructor_con_ventana_centrada(precios, objetivo, horizonte=1, **_):
    """Constructor tramposo: media movil centrada, que promedia pasado y futuro."""
    serie = precios[objetivo]
    X = pd.DataFrame({"media_centrada": serie.rolling(11, center=True).mean() / serie - 1.0})
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "fuga")


def constructor_con_fuga_por_otro_kwarg(precios, objetivo, demanda=None, horizonte=1, **_):
    """Constructor tramposo: la fuga entra por un kwarg que no se llama 'exogenas'.

    El mismo ``shift(-1)`` es fuga se llame el argumento ``demanda``, ``volumen``
    o ``exogena`` en singular por un error de tecleo. Si la auditoria se fia del
    nombre del parametro, el veredicto depende de como se haya bautizado la
    firma del constructor y no de lo que el constructor hace.
    """
    serie = precios[objetivo]
    X = pd.DataFrame(
        {
            "rend_ret1": np.log(serie).diff().shift(1),
            "demanda_manana": demanda.reindex(serie.index).shift(-1),
        }
    )
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "fuga")


def constructor_con_fuga_en_la_exogena(precios, objetivo, exogenas=None, horizonte=1, **_):
    """Constructor tramposo: usa la exogena adelantada un dia, tal como la recibe.

    Al llamar a :func:`retardos` con la exogena sin convertir, el nombre de la
    columna depende de si llego una Series o un DataFrame: es el caso que se
    escapaba cuando los dos experimentos no entregaban el mismo tipo.
    """
    serie = precios[objetivo]
    exo = retardos(exogenas.shift(-1), (0,), prefijo="exo")
    X = pd.concat(
        [pd.DataFrame({"rend_ret1": np.log(serie).diff().shift(1)}), exo.reindex(serie.index)],
        axis=1,
    )
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "fuga")


def constructor_sin_exogenas(precios, objetivo, horizonte=1, **_):
    """Constructor limpio que ignora cualquier argumento de mas."""
    serie = precios[objetivo]
    X = pd.DataFrame({"rend_ret1": np.log(serie).diff().shift(1)})
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "retardos")


def constructor_con_columnas_cambiantes(precios, objetivo, horizonte=1, **_):
    """Constructor cuyo conjunto de columnas depende de los datos que recibe."""
    serie = precios[objetivo]
    X = pd.DataFrame(
        {
            "rend_ret1": np.log(serie).diff().shift(1),
            f"rend_mm{len(serie)}": np.log(serie).diff().rolling(5).mean().shift(1),
        }
    )
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "fuga")


def constructor_con_relleno_hacia_atras(precios, objetivo, horizonte=1, **_):
    """Constructor tramposo: rellena los huecos con el PRIMER dato posterior.

    La fuga solo existe en las fechas con hueco, asi que solo se ve si algun
    corte cae pegado a una de ellas.
    """
    serie = precios[objetivo].bfill()
    X = pd.DataFrame({"rend_ret0": np.log(serie).diff()})
    return _matriz_desde(X, objetivo_rendimiento_futuro(serie, horizonte), "fuga")


@pytest.fixture(scope="module")
def precios_con_huecos(precios_cortos) -> pd.DataFrame:
    """Los mismos precios con cuatro sesiones sin dato, como un puente de festivos."""
    con_huecos = precios_cortos.copy()
    columna = con_huecos.columns.get_loc("ALFA")
    for posicion in (120, 121, 200, 300):
        con_huecos.iloc[posicion, columna] = np.nan
    return con_huecos


def test_la_auditoria_acepta_la_matriz_del_modulo(precios_cortos):
    comprobar_ausencia_de_fuga(construir_matriz, precios_cortos, "ALFA")


def test_la_auditoria_acepta_un_horizonte_largo(precios_cortos):
    """Que el objetivo mire cinco dias hacia delante no es fuga: es la prediccion."""
    comprobar_ausencia_de_fuga(construir_matriz, precios_cortos, "ALFA", horizonte=5)


def test_la_auditoria_acepta_una_exogena_prevista_con_retardo_cero(precios_cortos, exogena):
    comprobar_ausencia_de_fuga(
        construir_matriz,
        precios_cortos,
        "ALFA",
        exogenas=exogena,
        retardos_exogenas=(0, 1),
        incluir=("retardos", "calendario"),
    )


def test_la_auditoria_detecta_una_variable_que_mira_un_dia_al_futuro(precios_cortos):
    with pytest.raises(ErrorDatos, match="precio_de_maniana"):
        comprobar_ausencia_de_fuga(constructor_con_fuga_de_un_dia, precios_cortos, "ALFA")


def test_la_auditoria_detecta_una_tipificacion_con_toda_la_muestra(precios_cortos):
    with pytest.raises(ErrorDatos, match="rend_tipificado"):
        comprobar_ausencia_de_fuga(constructor_con_tipificacion_global, precios_cortos, "ALFA")


def test_la_auditoria_detecta_una_ventana_centrada(precios_cortos):
    with pytest.raises(ErrorDatos, match="media_centrada"):
        comprobar_ausencia_de_fuga(constructor_con_ventana_centrada, precios_cortos, "ALFA")


def test_la_auditoria_senala_la_columna_y_la_fecha(precios_cortos):
    fecha = precios_cortos.index[300]
    with pytest.raises(ErrorDatos) as excepcion:
        comprobar_ausencia_de_fuga(
            constructor_con_fuga_de_un_dia, precios_cortos, "ALFA", fechas=[fecha]
        )

    mensaje = str(excepcion.value)
    assert "precio_de_maniana" in mensaje
    assert fecha.date().isoformat() in mensaje


def test_truncar_por_si_solo_no_veria_la_fuga_de_un_dia(precios_cortos):
    """Justifica que la auditoria haga dos experimentos y no solo el de truncar.

    Una variable con ``shift(-1)`` da exactamente los mismos valores en la serie
    truncada: la unica fila que la delataria, la del propio corte, desaparece al
    eliminar los NaN. El punto ciego esta siempre en el borde, y solo se ve
    reescribiendo el futuro en lugar de quitarlo.
    """
    corte = precios_cortos.index[300]
    referencia = constructor_con_fuga_de_un_dia(precios_cortos, "ALFA")

    truncada = constructor_con_fuga_de_un_dia(precios_cortos.loc[:corte], "ALFA")
    _comparar(referencia, truncada, corte, "al truncar la serie en")

    perturbada = constructor_con_fuga_de_un_dia(
        _perturbar_futuro(precios_cortos, corte), "ALFA"
    )
    with pytest.raises(ErrorDatos, match="precio_de_maniana"):
        _comparar(referencia, perturbada, corte, "al reescribir lo posterior a")


def test_la_auditoria_rechaza_una_fecha_de_corte_fuera_del_tramo(precios_cortos):
    with pytest.raises(ErrorDatos, match="fuera del tramo"):
        comprobar_ausencia_de_fuga(
            construir_matriz, precios_cortos, "ALFA", fechas=["2099-01-01"]
        )


def test_la_auditoria_detecta_la_fuga_aunque_el_kwarg_no_se_llame_exogenas(
    precios_cortos, exogena
):
    """El veredicto no puede depender de como se llame el parametro.

    Antes solo se truncaba y se perturbaba ``kwargs['exogenas']``: cualquier
    otro argumento de datos llegaba identico a las tres llamadas, asi que las
    columnas que salian de el coincidian por construccion y la comparacion no
    veia nada.
    """
    with pytest.raises(ErrorDatos, match="demanda_manana"):
        comprobar_ausencia_de_fuga(
            constructor_con_fuga_por_otro_kwarg, precios_cortos, "ALFA", demanda=exogena
        )


def test_la_auditoria_detecta_la_fuga_de_una_exogena_dada_como_serie(precios_cortos, exogena):
    """Los dos experimentos tienen que entregar el mismo tipo que la referencia.

    Con la exogena convertida a DataFrame solo en el experimento de
    perturbacion, un constructor que bautice sus columnas segun el tipo devolvia
    'exo_ret0' en la referencia y 'exo_temperatura_ret0' en la perturbada, y
    ninguna columna de la exogena entraba en la comparacion.
    """
    with pytest.raises(ErrorDatos, match="exo_ret0"):
        comprobar_ausencia_de_fuga(
            constructor_con_fuga_en_la_exogena, precios_cortos, "ALFA", exogenas=exogena
        )


def test_la_auditoria_detecta_la_misma_fuga_con_la_exogena_como_marco(precios_cortos, exogena):
    with pytest.raises(ErrorDatos, match="exo_temperatura_ret0"):
        comprobar_ausencia_de_fuga(
            constructor_con_fuga_en_la_exogena,
            precios_cortos,
            "ALFA",
            exogenas=exogena.to_frame(),
        )


def test_la_auditoria_aborta_si_el_constructor_no_devuelve_siempre_las_mismas_columnas(
    precios_cortos,
):
    """Intersecar los nombres en silencio haria pasar por 'no hay fuga' un desajuste."""
    with pytest.raises(ErrorDatos, match="columnas distintas"):
        comprobar_ausencia_de_fuga(
            constructor_con_columnas_cambiantes, precios_cortos, "ALFA"
        )


@pytest.mark.parametrize(
    ("extra", "motivo"),
    [
        (pd.Series(np.arange(10.0)), "no se puede comparar"),
        (
            pd.Series(["a", "b", "c"], index=pd.bdate_range("2015-01-01", periods=3)),
            "no sabe reescribir",
        ),
    ],
)
def test_la_auditoria_rechaza_un_argumento_de_datos_que_no_sabe_alterar(
    precios_cortos, extra, motivo
):
    """Un 'pasa' que no significa nada es peor que un error.

    Si un argumento de datos no se puede truncar ni perturbar, sus columnas
    salen iguales en las tres matrices y la auditoria no esta comprobando nada
    sobre ellas: hay que decirlo, no dar el visto bueno.
    """
    with pytest.raises(ErrorDatos) as excepcion:
        comprobar_ausencia_de_fuga(
            constructor_sin_exogenas, precios_cortos, "ALFA", demanda=extra
        )

    mensaje = str(excepcion.value)
    assert "no puede garantizar nada" in mensaje
    assert "'demanda'" in mensaje
    assert motivo in mensaje


def test_la_auditoria_detecta_un_relleno_hacia_atras_en_un_hueco(precios_con_huecos):
    """Los cortes por omision tienen que caer tambien junto a los ausentes.

    Con solo tres cortes al 50, 70 y 85 % de la muestra, un relleno hacia atras
    pasaba desapercibido salvo que un hueco cayese pegado a uno de ellos, y es
    el fallo que el propio mensaje de error dice buscar.
    """
    with pytest.raises(ErrorDatos, match="rend_ret0"):
        comprobar_ausencia_de_fuga(
            constructor_con_relleno_hacia_atras, precios_con_huecos, "ALFA"
        )


def test_la_auditoria_sigue_aceptando_la_matriz_del_modulo_con_huecos(precios_con_huecos):
    """Los cortes nuevos no pueden convertirse en falsas alarmas."""
    comprobar_ausencia_de_fuga(
        construir_matriz, precios_con_huecos, "ALFA", incluir=("retardos", "momento")
    )


def test_la_auditoria_exige_que_el_constructor_devuelva_una_matriz_de_datos(precios_cortos):
    with pytest.raises(ErrorDatos, match="MatrizDatos"):
        comprobar_ausencia_de_fuga(
            lambda precios, objetivo, **kwargs: precios, precios_cortos, "ALFA"
        )
