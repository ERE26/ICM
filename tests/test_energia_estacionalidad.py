"""Pruebas de la estacionalidad de las series de energia.

Casi todo se contrasta contra series con solucion analitica conocida (un
sinusoide puro de periodo dado, una serie cuyo valor es el numero de mes, unos
factores multiplicativos fijados a mano) o contra propiedades que deben
cumplirse siempre (el indice aditivo suma cero, la reconstruccion cuadra, el
acumulado de grados dia nunca es negativo).

Los dos contrastes que de verdad demuestran que ``contraste_estacionalidad``
sirve van juntos: uno mide su **potencia** (detecta una estacionalidad
inyectada a mano) y otro su **tamanio** (no la detecta en ruido puro). Ninguno
depende de una semilla afortunada: en lugar de mirar una sola muestra, se
repiten doscientas replicas independientes y se comprueba la tasa de rechazo,
que es la propiedad que define un contraste bien calibrado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.energia.estacionalidad import (
    AGRUPACIONES,
    DIAS_SEMANA,
    MESES,
    Descomposicion,
    contraste_estacionalidad,
    descomponer,
    desestacionalizar,
    etiqueta_grupo,
    grados_dia_acumulados,
    indice_estacional,
    perfil_estacional,
    tabla_perfiles,
)

N_MESES = 240
N_REPLICAS = 200

#: Factores mensuales tipicos de un mercado de gas: invierno caro, verano
#: barato. Promedian aproximadamente uno, pero no exactamente, para que las
#: pruebas de normalizacion no salgan bien por casualidad.
FACTORES_MES = np.array([1.24, 1.18, 1.06, 0.94, 0.88, 0.82, 0.86, 0.90, 0.96, 1.04, 1.14, 1.22])


@pytest.fixture(scope="module")
def indice_mensual() -> pd.DatetimeIndex:
    """Veinte anios de datos mensuales: veinte observaciones por mes."""
    return pd.date_range("2005-01-31", periods=N_MESES, freq="ME", name="fecha")


@pytest.fixture(scope="module")
def indice_diario() -> pd.DatetimeIndex:
    """Cuatro anios de dias naturales, para las pruebas de calendario fino."""
    return pd.date_range("2020-01-01", periods=4 * 365 + 1, freq="D", name="fecha")


@pytest.fixture(scope="module")
def serie_mes(indice_mensual: pd.DatetimeIndex) -> pd.Series:
    """Serie cuyo valor es el numero de mes: el perfil se conoce de antemano."""
    return pd.Series(indice_mensual.month.to_numpy(dtype=float), index=indice_mensual)


@pytest.fixture(scope="module")
def serie_estacional(indice_mensual: pd.DatetimeIndex) -> pd.Series:
    """Precio mensual con factor estacional conocido y ruido moderado."""
    generador = np.random.default_rng(20240501)
    base = 50.0 * FACTORES_MES[indice_mensual.month - 1]
    return pd.Series(base + generador.normal(scale=1.5, size=N_MESES), index=indice_mensual)


@pytest.fixture(scope="module")
def sinusoide() -> pd.Series:
    """Nivel constante mas un ciclo puro de periodo 12, sin ruido.

    Un ciclo exacto es el caso con solucion analitica: la media movil centrada
    de una ventana igual al periodo promedia el ciclo completo y lo cancela
    exactamente, asi que la tendencia debe salir plana y el residuo nulo hasta
    la precision de la maquina.
    """
    t = np.arange(240)
    valores = 100.0 + 15.0 * np.sin(2.0 * np.pi * t / 12.0)
    return pd.Series(valores, index=pd.RangeIndex(240), name="sinusoide")


def _varianza_explicada_por_mes(serie: pd.Series) -> float:
    """Fraccion de la varianza que explica el mes (eta cuadrado de un ANOVA)."""
    medias = serie.groupby(serie.index.month).transform("mean")
    total = float(((serie - serie.mean()) ** 2).sum())
    entre = float(((medias - serie.mean()) ** 2).sum())
    return entre / total


# --- perfiles de calendario -----------------------------------------------------


def test_el_perfil_por_mes_recupera_el_valor_exacto_de_cada_mes(serie_mes):
    perfil = perfil_estacional(serie_mes, por="mes")
    assert list(perfil.index) == list(range(1, 13))
    assert np.allclose(perfil.to_numpy(), np.arange(1.0, 13.0))
    assert perfil.index.name == "mes"
    assert perfil.name == "media_por_mes"


def test_el_perfil_por_dia_de_la_semana_recupera_el_dia(indice_diario):
    serie = pd.Series(indice_diario.dayofweek.to_numpy(dtype=float), index=indice_diario)
    perfil = perfil_estacional(serie, por="dia_semana")
    assert list(perfil.index) == list(range(7))
    assert np.allclose(perfil.to_numpy(), np.arange(7.0))


def test_el_perfil_por_trimestre_agrega_los_meses_del_trimestre(serie_mes):
    perfil = perfil_estacional(serie_mes, por="trimestre")
    assert list(perfil.index) == [1, 2, 3, 4]
    # Cada trimestre es la media de sus tres meses: (1+2+3)/3 = 2, etc.
    assert np.allclose(perfil.to_numpy(), [2.0, 5.0, 8.0, 11.0])


def test_el_perfil_por_dia_del_anio_tiene_un_grupo_por_dia(indice_diario):
    serie = pd.Series(1.0, index=indice_diario)
    perfil = perfil_estacional(serie, por="dia_anio")
    # La muestra incluye un 29 de febrero, asi que aparece el dia 366.
    assert len(perfil) == 366
    assert perfil.index.max() == 366


def test_la_mediana_resiste_un_pico_que_arrastra_a_la_media(indice_mensual):
    """Un solo dia de ola de frio no debe redefinir el perfil de enero."""
    serie = pd.Series(50.0, index=indice_mensual)
    serie.iloc[0] = 3000.0

    media = perfil_estacional(serie, por="mes", estadistico="media")
    mediana = perfil_estacional(serie, por="mes", estadistico="mediana")

    assert mediana.loc[1] == pytest.approx(50.0)
    assert media.loc[1] > 190.0
    assert np.allclose(mediana.to_numpy(), 50.0)


def test_el_perfil_descarta_los_valores_ausentes_sin_desplazar_el_calendario(serie_mes):
    con_huecos = serie_mes.copy()
    con_huecos.iloc[::24] = np.nan  # se pierde el enero de uno de cada dos anios
    perfil = perfil_estacional(con_huecos, por="mes")
    assert np.allclose(perfil.to_numpy(), np.arange(1.0, 13.0))
    assert int(tabla_perfiles(con_huecos).loc[("mes", 1), "observaciones"]) == 10


@pytest.mark.parametrize("por", AGRUPACIONES)
def test_todas_las_agrupaciones_documentadas_funcionan(serie_estacional, por):
    perfil = perfil_estacional(serie_estacional, por=por)
    assert len(perfil) >= 4
    assert perfil.notna().all()


def test_una_agrupacion_desconocida_da_error(serie_mes):
    with pytest.raises(ErrorDatos, match="por"):
        perfil_estacional(serie_mes, por="semana_del_anio")


def test_un_estadistico_desconocido_da_error(serie_mes):
    with pytest.raises(ErrorDatos, match="estadistico"):
        perfil_estacional(serie_mes, estadistico="moda")


def test_sin_indice_de_fechas_no_hay_calendario():
    serie = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ErrorDatos, match="DatetimeIndex"):
        perfil_estacional(serie)
    with pytest.raises(ErrorDatos, match="DatetimeIndex"):
        tabla_perfiles(serie)
    with pytest.raises(ErrorDatos, match="DatetimeIndex"):
        contraste_estacionalidad(serie)


def test_las_etiquetas_de_grupo_son_legibles():
    assert etiqueta_grupo("mes", 1) == "enero"
    assert etiqueta_grupo("mes", 12) == "diciembre"
    assert etiqueta_grupo("dia_semana", 0) == "lunes"
    assert etiqueta_grupo("dia_semana", 6) == "domingo"
    assert etiqueta_grupo("trimestre", 3) == "T3"
    assert etiqueta_grupo("dia_anio", 200) == "dia 200"
    assert len(MESES) == 12 and len(DIAS_SEMANA) == 7


# --- tabla de perfiles ----------------------------------------------------------


def test_la_tabla_reune_los_doce_meses_y_los_siete_dias(serie_estacional):
    tabla = tabla_perfiles(serie_estacional)
    assert len(tabla) == 19
    assert list(tabla.loc["mes"].index) == list(range(1, 13))
    assert list(tabla.loc["dia_semana"].index) == list(range(7))
    assert list(tabla.loc["mes", "etiqueta"]) == list(MESES)


def test_la_tabla_coincide_con_los_perfiles_por_separado(serie_estacional):
    tabla = tabla_perfiles(serie_estacional)
    medias = perfil_estacional(serie_estacional, "mes", "media")
    medianas = perfil_estacional(serie_estacional, "mes", "mediana")
    assert np.allclose(tabla.loc["mes", "media"].to_numpy(), medias.to_numpy())
    assert np.allclose(tabla.loc["mes", "mediana"].to_numpy(), medianas.to_numpy())
    assert tabla.loc[("mes", 1), "observaciones"] == 20


def test_el_indice_aditivo_de_la_tabla_es_la_desviacion_sobre_la_media_global(serie_estacional):
    tabla = tabla_perfiles(serie_estacional)
    global_ = tabla.attrs["media_global"]
    assert global_ == pytest.approx(float(serie_estacional.mean()))
    assert np.allclose(
        tabla["indice_aditivo"].to_numpy(),
        tabla["media"].to_numpy() - global_,
    )
    # Con grupos del mismo tamanio las desviaciones mensuales se compensan.
    assert float(tabla.loc["mes", "indice_aditivo"].sum()) == pytest.approx(0.0, abs=1e-9)


# --- descomposicion clasica -----------------------------------------------------


def test_la_descomposicion_recupera_un_ciclo_puro_y_deja_residuo_nulo(sinusoide):
    piezas = descomponer(sinusoide, periodo=12, modelo="aditivo")
    verdad = sinusoide - 100.0

    central = piezas.residuo.notna()
    assert np.max(np.abs(piezas.estacional - verdad)) < 1e-9
    assert np.max(np.abs(piezas.residuo[central])) < 1e-9
    assert np.allclose(piezas.tendencia[central], 100.0)


def test_la_tendencia_lineal_se_recupera_exactamente_bajo_el_ciclo():
    """La media movil centrada es exacta sobre una recta: queda solo el ciclo."""
    t = np.arange(240)
    recta = 40.0 + 0.05 * t
    serie = pd.Series(recta + 8.0 * np.sin(2.0 * np.pi * t / 12.0))
    piezas = descomponer(serie, periodo=12)

    central = piezas.tendencia.notna()
    assert np.max(np.abs(piezas.tendencia[central] - recta[central.to_numpy()])) < 1e-9
    assert np.max(np.abs(piezas.residuo[central])) < 1e-9


def test_la_correccion_de_ventana_par_evita_desplazar_la_tendencia():
    """Una media movil par sin corregir centra mal: se desplaza medio periodo.

    Con doce observaciones no hay centro, asi que ``rolling(12, center=True)``
    etiqueta la media medio paso corrida. Sobre una recta de pendiente 0,05 eso
    se traduce en un sesgo constante de exactamente 0,025 unidades, mientras
    que la correccion 2x12 (pesos 1/2, 1, ..., 1, 1/2) reproduce la recta.
    """
    t = np.arange(240)
    pendiente = 0.05
    recta = 40.0 + pendiente * t
    serie = pd.Series(recta + 8.0 * np.sin(2.0 * np.pi * t / 12.0))

    corregida = descomponer(serie, periodo=12).tendencia
    ingenua = serie.rolling(12, center=True).mean()
    central = corregida.notna() & ingenua.notna()
    verdad = recta[central.to_numpy()]

    assert np.max(np.abs(corregida[central] - verdad)) < 1e-9
    assert np.allclose(ingenua[central] - verdad, -pendiente / 2.0)


def test_la_reconstruccion_devuelve_la_serie_original_en_la_parte_central(sinusoide):
    piezas = descomponer(sinusoide, periodo=12)
    reconstruida = piezas.reconstruir()

    central = reconstruida.notna()
    assert np.allclose(reconstruida[central], sinusoide[central])
    # La media movil 2x12 usa 13 puntos: pierde 6 por cada extremo.
    assert int((~central).sum()) == 12
    assert bool(central.iloc[6]) and not bool(central.iloc[5])


def test_la_reconstruccion_tambien_cuadra_en_el_modelo_multiplicativo(serie_estacional):
    piezas = descomponer(serie_estacional, periodo=12, modelo="multiplicativo")
    reconstruida = piezas.reconstruir()
    central = reconstruida.notna()
    assert np.allclose(reconstruida[central], serie_estacional[central])


def test_la_reconstruccion_cuadra_con_ventana_impar(sinusoide):
    piezas = descomponer(sinusoide, periodo=12, ventana_tendencia=13)
    reconstruida = piezas.reconstruir()
    central = reconstruida.notna()
    assert np.allclose(reconstruida[central], sinusoide[central])
    assert int((~central).sum()) == 12


def test_el_componente_estacional_esta_centrado(serie_estacional):
    aditiva = descomponer(serie_estacional, periodo=12, modelo="aditivo")
    multiplicativa = descomponer(serie_estacional, periodo=12, modelo="multiplicativo")

    ciclo_aditivo = aditiva.estacional.iloc[:12]
    ciclo_multiplicativo = multiplicativa.estacional.iloc[:12]
    assert float(ciclo_aditivo.sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(ciclo_multiplicativo.mean()) == pytest.approx(1.0)


def test_el_modelo_multiplicativo_recupera_los_factores_conocidos():
    """Amplitud que crece con el nivel: es el caso que pide el multiplicativo."""
    t = np.arange(360)
    tendencia = 30.0 + 0.1 * t
    serie = pd.Series(tendencia * FACTORES_MES[t % 12])
    piezas = descomponer(serie, periodo=12, modelo="multiplicativo")

    estimados = piezas.estacional.iloc[:12].to_numpy()
    verdaderos = FACTORES_MES / FACTORES_MES.mean()
    assert np.max(np.abs(estimados - verdaderos)) < 0.01


def test_el_multiplicativo_exige_serie_estrictamente_positiva(indice_mensual):
    con_cero = pd.Series(np.linspace(0.0, 50.0, N_MESES), index=indice_mensual)
    con_negativos = pd.Series(np.linspace(-20.0, 50.0, N_MESES), index=indice_mensual)

    for serie in (con_cero, con_negativos):
        with pytest.raises(ErrorDatos, match="estrictamente positiva"):
            descomponer(serie, periodo=12, modelo="multiplicativo")
        with pytest.raises(ErrorDatos, match="estrictamente positiva"):
            desestacionalizar(serie, periodo=12, modelo="multiplicativo")
    # El aditivo si admite precios negativos, que en el pool existen.
    assert descomponer(con_negativos, periodo=12, modelo="aditivo").periodo == 12


def test_un_periodo_mayor_que_la_serie_da_error(serie_estacional):
    with pytest.raises(ErrorDatos, match="dos ciclos completos"):
        descomponer(serie_estacional, periodo=N_MESES + 1)
    with pytest.raises(ErrorDatos, match="dos ciclos completos"):
        descomponer(serie_estacional.iloc[:100], periodo=252)
    with pytest.raises(ErrorDatos, match="dos ciclos completos"):
        desestacionalizar(serie_estacional.iloc[:100], periodo=252)


def test_un_periodo_de_menos_de_dos_observaciones_no_es_un_ciclo(serie_estacional):
    for periodo in (0, 1, -12):
        with pytest.raises(ErrorDatos, match="periodo"):
            descomponer(serie_estacional, periodo=periodo)
    with pytest.raises(ErrorDatos, match="periodo"):
        descomponer(serie_estacional, periodo=12.5)


def test_un_modelo_desconocido_da_error(serie_estacional):
    with pytest.raises(ErrorDatos, match="modelo"):
        descomponer(serie_estacional, periodo=12, modelo="logaritmico")
    with pytest.raises(ErrorDatos, match="modelo"):
        desestacionalizar(serie_estacional, periodo=12, modelo="logaritmico")
    with pytest.raises(ErrorDatos, match="modelo"):
        indice_estacional(serie_estacional, modelo="logaritmico")


def test_una_ventana_de_tendencia_que_no_cabe_da_error(serie_estacional):
    with pytest.raises(ErrorDatos, match="ventana"):
        descomponer(serie_estacional, periodo=12, ventana_tendencia=N_MESES + 5)
    with pytest.raises(ErrorDatos, match="ventana_tendencia"):
        descomponer(serie_estacional, periodo=12, ventana_tendencia=1)


def test_una_ventana_mas_ancha_da_una_tendencia_mas_lisa(serie_estacional):
    estrecha = descomponer(serie_estacional, periodo=12, ventana_tendencia=12)
    ancha = descomponer(serie_estacional, periodo=12, ventana_tendencia=48)
    assert float(ancha.tendencia.diff().std()) < float(estrecha.tendencia.diff().std())


def test_la_descomposicion_se_puede_volcar_a_un_marco(serie_estacional):
    piezas = descomponer(serie_estacional, periodo=12)
    marco = piezas.a_marco()
    assert list(marco.columns) == ["tendencia", "estacional", "residuo"]
    assert len(marco) == N_MESES


def test_la_fuerza_estacional_separa_una_serie_estacional_de_un_ruido(serie_estacional):
    generador = np.random.default_rng(11)
    ruido = pd.Series(generador.standard_normal(N_MESES) * 5.0 + 50.0)
    assert descomponer(serie_estacional, periodo=12).fuerza_estacional() > 0.8
    assert descomponer(ruido, periodo=12).fuerza_estacional() < 0.3


def test_una_descomposicion_con_modelo_invalido_no_se_puede_construir(sinusoide):
    with pytest.raises(ErrorDatos, match="modelo"):
        Descomposicion(sinusoide, sinusoide, sinusoide, 12, "raro")


# --- desestacionalizacion -------------------------------------------------------


def test_desestacionalizar_reduce_la_varianza_explicada_por_el_mes(serie_estacional):
    antes = _varianza_explicada_por_mes(serie_estacional)
    despues = _varianza_explicada_por_mes(desestacionalizar(serie_estacional, periodo=12))
    assert antes > 0.9
    assert despues < 0.05
    assert despues < antes / 10.0


def test_desestacionalizar_esta_definida_en_toda_la_muestra(serie_estacional):
    """A diferencia de la tendencia, el factor estacional se conoce en cada fecha."""
    ajustada = desestacionalizar(serie_estacional, periodo=12)
    assert len(ajustada) == len(serie_estacional)
    assert ajustada.notna().all()
    assert descomponer(serie_estacional, periodo=12).tendencia.isna().any()


def test_desestacionalizar_es_restar_el_componente_estacional(serie_estacional):
    piezas = descomponer(serie_estacional, periodo=12, modelo="aditivo")
    ajustada = desestacionalizar(serie_estacional, periodo=12, modelo="aditivo")
    assert np.allclose(ajustada.to_numpy(), (serie_estacional - piezas.estacional).to_numpy())


def test_desestacionalizar_en_multiplicativo_aplana_el_perfil_mensual():
    t = np.arange(360)
    indice = pd.date_range("1995-01-31", periods=360, freq="ME")
    serie = pd.Series((30.0 + 0.1 * t) * FACTORES_MES[indice.month - 1], index=indice)

    perfil_antes = indice_estacional(serie, "mes", "multiplicativo")
    perfil_despues = indice_estacional(
        desestacionalizar(serie, periodo=12, modelo="multiplicativo"), "mes", "multiplicativo"
    )
    assert float(np.max(np.abs(perfil_antes - 1.0))) > 0.15
    assert float(np.max(np.abs(perfil_despues - 1.0))) < 0.02


# --- indice estacional ----------------------------------------------------------


def test_el_indice_aditivo_suma_cero(serie_estacional):
    indice = indice_estacional(serie_estacional, por="mes", modelo="aditivo")
    assert float(indice.sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(indice.mean()) == pytest.approx(0.0, abs=1e-12)


def test_el_indice_multiplicativo_promedia_uno(serie_estacional):
    indice = indice_estacional(serie_estacional, por="mes", modelo="multiplicativo")
    assert float(indice.mean()) == pytest.approx(1.0)
    assert (indice > 0.0).all()


@pytest.mark.parametrize("por", AGRUPACIONES)
def test_la_normalizacion_se_cumple_en_cualquier_agrupacion(serie_estacional, por):
    assert float(indice_estacional(serie_estacional, por, "aditivo").sum()) == pytest.approx(
        0.0, abs=1e-8
    )
    assert float(indice_estacional(serie_estacional, por, "multiplicativo").mean()) == (
        pytest.approx(1.0)
    )


def test_el_indice_multiplicativo_recupera_los_factores_publicados(indice_mensual):
    serie = pd.Series(80.0 * FACTORES_MES[indice_mensual.month - 1], index=indice_mensual)
    indice = indice_estacional(serie, por="mes", modelo="multiplicativo")
    assert np.allclose(indice.to_numpy(), FACTORES_MES / FACTORES_MES.mean())


def test_el_indice_multiplicativo_no_depende_de_la_escala(serie_estacional):
    """Un indice publicado tiene que poder compararse entre mercados."""
    original = indice_estacional(serie_estacional, modelo="multiplicativo")
    reescalada = indice_estacional(serie_estacional * 37.0, modelo="multiplicativo")
    assert np.allclose(original.to_numpy(), reescalada.to_numpy())


def test_el_indice_multiplicativo_no_admite_valores_no_positivos(indice_mensual):
    con_negativos = pd.Series(np.linspace(-10.0, 40.0, N_MESES), index=indice_mensual)
    with pytest.raises(ErrorDatos, match="estrictamente positiva"):
        indice_estacional(con_negativos, modelo="multiplicativo")
    # El aditivo si: es la unica lectura posible de un pool con precios negativos.
    assert float(indice_estacional(con_negativos, modelo="aditivo").sum()) == pytest.approx(
        0.0, abs=1e-9
    )


# --- contraste de estacionalidad ------------------------------------------------


def test_el_contraste_detecta_una_estacionalidad_fuerte(serie_estacional):
    resultado = contraste_estacionalidad(serie_estacional, por="mes")
    assert resultado["rechaza_al_5pct"] is True
    assert resultado["p_valor"] < 1e-10
    assert resultado["tamano_efecto"] > 0.5
    assert resultado["n_grupos"] == 12
    assert resultado["grados_libertad"] == 11
    assert resultado["observaciones"] == N_MESES
    assert resultado["por"] == "mes"


def test_el_contraste_tiene_potencia_frente_a_una_estacionalidad_moderada(indice_mensual):
    """Potencia: con estacionalidad inyectada debe rechazar casi siempre.

    Se repiten doscientas replicas independientes en lugar de mirar una sola
    muestra, para que el resultado sea una propiedad del contraste y no de la
    semilla que haya tocado.
    """
    generador = np.random.default_rng(31415)
    efecto = 1.2 * np.sin(2.0 * np.pi * (indice_mensual.month.to_numpy() - 1) / 12.0)

    rechazos = 0
    for _ in range(N_REPLICAS):
        muestra = pd.Series(
            efecto + generador.standard_normal(N_MESES), index=indice_mensual
        )
        rechazos += contraste_estacionalidad(muestra, por="mes")["rechaza_al_5pct"]
    assert rechazos / N_REPLICAS > 0.95


def test_el_contraste_no_detecta_estacionalidad_en_ruido_puro(indice_mensual):
    """Tamanio: sin estacionalidad la tasa de rechazo debe rondar el 5 % nominal.

    Es la otra mitad de la demostracion. Un contraste que rechazase siempre
    tendria potencia perfecta y no valdria para nada; lo que lo hace util es
    que con ruido puro se calla. Con doscientas replicas la tasa observada se
    distribuye como una binomial de media 10 y desviacion 3,1 rechazos, asi
    que la cota del 12 % esta a mas de seis desviaciones: no puede fallar por
    mala suerte, solo si el contraste esta mal calibrado.
    """
    generador = np.random.default_rng(2718)

    rechazos = 0
    for _ in range(N_REPLICAS):
        ruido = pd.Series(generador.standard_normal(N_MESES), index=indice_mensual)
        rechazos += contraste_estacionalidad(ruido, por="mes")["rechaza_al_5pct"]
    tasa = rechazos / N_REPLICAS
    assert 0.005 < tasa < 0.12


def test_el_p_valor_cae_al_crecer_la_amplitud_estacional(indice_mensual):
    generador = np.random.default_rng(99)
    ruido = generador.standard_normal(N_MESES)
    forma = np.sin(2.0 * np.pi * (indice_mensual.month.to_numpy() - 1) / 12.0)

    p_valores = []
    for amplitud in (0.0, 0.5, 1.0, 2.0):
        muestra = pd.Series(amplitud * forma + ruido, index=indice_mensual)
        p_valores.append(contraste_estacionalidad(muestra)["p_valor"])
    assert p_valores == sorted(p_valores, reverse=True)
    assert p_valores[0] > 0.05 > p_valores[-1]


def test_el_tamano_del_efecto_crece_con_la_amplitud(indice_mensual):
    generador = np.random.default_rng(1234)
    ruido = generador.standard_normal(N_MESES)
    forma = np.sin(2.0 * np.pi * (indice_mensual.month.to_numpy() - 1) / 12.0)

    efectos = [
        contraste_estacionalidad(pd.Series(a * forma + ruido, index=indice_mensual))[
            "tamano_efecto"
        ]
        for a in (0.0, 1.0, 4.0)
    ]
    assert efectos == sorted(efectos)
    assert 0.0 <= efectos[0] <= efectos[-1] <= 1.0


def test_tras_desestacionalizar_el_contraste_ya_no_rechaza(serie_estacional):
    """La comprobacion de que el ajuste ha hecho su trabajo."""
    assert contraste_estacionalidad(serie_estacional)["rechaza_al_5pct"] is True
    ajustada = desestacionalizar(serie_estacional, periodo=12)
    assert contraste_estacionalidad(ajustada)["rechaza_al_5pct"] is False


def test_el_contraste_con_un_solo_grupo_no_tiene_nada_que_comparar():
    enero = pd.date_range("2024-01-01", periods=25, freq="D")
    serie = pd.Series(np.arange(25.0), index=enero)
    with pytest.raises(ErrorDatos, match="un grupo"):
        contraste_estacionalidad(serie, por="mes")


def test_el_contraste_con_una_observacion_por_grupo_es_vacio():
    indice = pd.date_range("2024-01-31", periods=12, freq="ME")
    serie = pd.Series(np.arange(12.0), index=indice)
    with pytest.raises(ErrorDatos, match="vacio"):
        contraste_estacionalidad(serie, por="mes")


def test_el_contraste_de_una_serie_constante_no_esta_definido(indice_mensual):
    with pytest.raises(ErrorDatos, match="identicos"):
        contraste_estacionalidad(pd.Series(42.0, index=indice_mensual))


def test_el_contraste_rechaza_una_agrupacion_desconocida(serie_estacional):
    with pytest.raises(ErrorDatos, match="por"):
        contraste_estacionalidad(serie_estacional, por="luna_llena")


# --- grados dia acumulados ------------------------------------------------------


def test_el_acumulado_de_grados_dia_da_el_numero_exacto(indice_diario):
    """A 10,5 C con base 15,5 son 5 grados dia diarios: 30 dias suman 150."""
    temperatura = pd.Series(10.5, index=indice_diario)
    acumulado = grados_dia_acumulados(temperatura, base=15.5, ventana=30)
    assert acumulado.iloc[-1] == pytest.approx(150.0)
    assert acumulado.iloc[29] == pytest.approx(150.0)


def test_el_acumulado_deja_en_blanco_las_primeras_observaciones(indice_diario):
    """Un acumulado de 7 dias no es comparable con uno de 30: no se rellena."""
    acumulado = grados_dia_acumulados(pd.Series(0.0, index=indice_diario), ventana=30)
    assert int(acumulado.isna().sum()) == 29
    assert bool(acumulado.iloc[:29].isna().all())
    assert acumulado.notna().iloc[29]


def test_por_encima_de_la_base_no_hay_grados_dia_de_calefaccion(indice_diario):
    """En verano nadie enciende la caldera: el acumulado es cero, no negativo."""
    acumulado = grados_dia_acumulados(pd.Series(28.0, index=indice_diario))
    assert (acumulado.dropna() == 0.0).all()


def test_el_acumulado_nunca_es_negativo_y_crece_al_bajar_la_temperatura(indice_diario):
    generador = np.random.default_rng(5)
    templada = pd.Series(
        12.0 + 10.0 * np.sin(np.arange(len(indice_diario)) / 58.0)
        + generador.standard_normal(len(indice_diario)),
        index=indice_diario,
    )
    acumulado = grados_dia_acumulados(templada)
    mas_frio = grados_dia_acumulados(templada - 5.0)

    assert (acumulado.dropna() >= 0.0).all()
    assert (mas_frio.dropna() >= acumulado.dropna()).all()
    assert float(mas_frio.sum()) > float(acumulado.sum())


def test_una_base_mas_alta_nunca_da_menos_grados_dia(indice_diario):
    generador = np.random.default_rng(6)
    temperatura = pd.Series(
        14.0 + 4.0 * generador.standard_normal(len(indice_diario)), index=indice_diario
    )
    suave = grados_dia_acumulados(temperatura, base=15.5)
    exigente = grados_dia_acumulados(temperatura, base=18.0)
    assert (exigente.dropna() >= suave.dropna()).all()


def test_el_acumulado_conserva_el_indice_y_documenta_sus_parametros(indice_diario):
    temperatura = pd.Series(5.0, index=indice_diario)
    acumulado = grados_dia_acumulados(temperatura, base=15.5, ventana=7)
    assert acumulado.name == "hdd_acumulado_7"
    assert list(acumulado.index) == list(indice_diario)
    assert acumulado.attrs["base"] == pytest.approx(15.5)
    assert acumulado.attrs["ventana"] == 7


def test_el_acumulado_de_un_dia_es_el_grado_dia_del_dia():
    temperatura = pd.Series([15.5, 14.5, 10.5, 20.0])
    acumulado = grados_dia_acumulados(temperatura, base=15.5, ventana=1)
    assert list(acumulado) == [0.0, 1.0, 5.0, 0.0]


def test_una_ventana_invalida_da_error(indice_diario):
    temperatura = pd.Series(10.0, index=indice_diario)
    with pytest.raises(ErrorDatos, match="ventana"):
        grados_dia_acumulados(temperatura, ventana=0)
    with pytest.raises(ErrorDatos, match="ventana"):
        grados_dia_acumulados(temperatura, ventana=-3)
    with pytest.raises(ErrorDatos, match="ventana"):
        grados_dia_acumulados(temperatura, ventana=len(indice_diario) + 1)


def test_una_base_no_numerica_da_error(indice_diario):
    temperatura = pd.Series(10.0, index=indice_diario)
    with pytest.raises(ErrorDatos, match="base"):
        grados_dia_acumulados(temperatura, base="templado")
    with pytest.raises(ErrorDatos, match="base"):
        grados_dia_acumulados(temperatura, base=float("nan"))


def test_una_serie_de_temperatura_vacia_da_error():
    with pytest.raises(ErrorDatos):
        grados_dia_acumulados(pd.Series([np.nan, np.nan]))
