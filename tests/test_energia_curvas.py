"""Pruebas de la estructura temporal de los futuros de energia.

El bloque que de verdad importa es el primero: con dos contratos y un escalon
de precio entre ellos, encadenar sin ajustar fabrica un rendimiento del 11 % el
dia del roll que nadie ha ganado. Todo lo demas de este modulo existe para que
ese numero no llegue nunca a un backtest.

Los precios estan construidos a mano (progresiones geometricas y niveles
redondos) para que los valores esperados se puedan comprobar con lapiz; solo la
prueba de propiedad general usa un generador con semilla fija.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.energia.curvas import (
    DIAS_ANIO_NATURAL,
    descomponer_rendimiento,
    dias_a_vencimiento,
    estado_curva,
    interpolar_curva,
    pendiente_curva,
    rendimiento_roll,
    serie_continua,
)

#: Dos contratos mensuales consecutivos.
VENCIMIENTOS = {"F1": "2024-02-15", "F2": "2024-03-15"}

#: Con cinco dias de antelacion, el roll de F1 a F2 cae el 10 de febrero.
FECHA_ROLL = pd.Timestamp("2024-02-10")

#: Cadena de tres vencimientos mensuales para las pruebas de descomposicion.
VENCIMIENTOS_CADENA = {"M1": "2024-01-31", "M2": "2024-02-29", "M3": "2024-03-31"}

#: Perdida diaria de valor dentro de cada contrato de la cadena.
DECAIMIENTO = 0.999


@pytest.fixture
def contratos() -> pd.DataFrame:
    """Dos contratos que suben suavemente con un escalon del 10 % entre ellos.

    F1 rinde un 0,1 % diario y F2 un 0,2 %, asi que cualquier rendimiento que no
    sea uno de esos dos numeros lo ha inventado el empalme. Las columnas van
    deliberadamente en orden inverso al de vencimiento: la cadena se ordena por
    la fecha de vencimiento, no por como venga el DataFrame.
    """
    fechas = pd.date_range("2024-02-01", periods=20, freq="D")
    pasos = np.arange(len(fechas))
    return pd.DataFrame(
        {"F2": 110.0 * 1.002**pasos, "F1": 100.0 * 1.001**pasos},
        index=fechas,
    )


def _cadena(niveles: dict[str, float]) -> pd.DataFrame:
    """Tres contratos planos en niveles distintos que pierden un 0,1 % al dia."""
    fechas = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    factor = DECAIMIENTO ** np.arange(len(fechas))
    return pd.DataFrame({c: nivel * factor for c, nivel in niveles.items()}, index=fechas)


@pytest.fixture
def cadena_contango() -> pd.DataFrame:
    """Curva ascendente: cada contrato siguiente cotiza mas caro que el anterior."""
    return _cadena({"M1": 100.0, "M2": 106.0, "M3": 112.0})


@pytest.fixture
def cadena_backwardation() -> pd.DataFrame:
    """Curva descendente: el contrato cercano es el mas caro."""
    return _cadena({"M1": 112.0, "M2": 106.0, "M3": 100.0})


# --- dias a vencimiento --------------------------------------------------------


def test_los_dias_a_vencimiento_cuentan_hasta_el_proximo_contrato(contratos):
    dias = dias_a_vencimiento(contratos.index, VENCIMIENTOS)
    # Del 1 al 15 de febrero hay 14 dias naturales.
    assert dias.loc["2024-02-01"] == 14.0
    # El dia del vencimiento quedan cero, y al siguiente se mira ya al 15 de
    # marzo: 13 dias hasta fin de febrero (2024 es bisiesto) mas 15 de marzo.
    assert dias.loc["2024-02-15"] == 0.0
    assert dias.loc["2024-02-16"] == 28.0


def test_los_dias_a_vencimiento_bajan_de_uno_en_uno_dentro_del_contrato(contratos):
    dias = dias_a_vencimiento(contratos.index, VENCIMIENTOS)
    vigencia_f1 = dias.loc[:"2024-02-15"]
    assert (vigencia_f1.diff().dropna() == -1.0).all()


def test_despues_del_ultimo_vencimiento_no_hay_dias_que_contar():
    indice = pd.date_range("2024-03-14", periods=3, freq="D")
    dias = dias_a_vencimiento(indice, VENCIMIENTOS)
    assert dias.loc["2024-03-15"] == 0.0
    assert np.isnan(dias.loc["2024-03-16"])


def test_los_vencimientos_admiten_series_ademas_de_diccionarios(contratos):
    como_serie = pd.Series(VENCIMIENTOS)
    esperado = dias_a_vencimiento(contratos.index, VENCIMIENTOS)
    assert dias_a_vencimiento(contratos.index, como_serie).equals(esperado)


def test_unos_vencimientos_mal_construidos_dan_error(contratos):
    with pytest.raises(ErrorDatos, match="dict"):
        dias_a_vencimiento(contratos.index, ["2024-02-15"])
    with pytest.raises(ErrorDatos, match="vacio"):
        dias_a_vencimiento(contratos.index, {})
    with pytest.raises(ErrorDatos, match="fechas"):
        dias_a_vencimiento(contratos.index, {"F1": "no soy una fecha"})


# --- el problema del roll: la prueba central -----------------------------------


def test_sin_ajuste_el_roll_inventa_un_rendimiento_y_con_ratio_no(contratos):
    """Con un escalon del 10 % entre contratos, encadenar sin ajustar miente.

    El 10 de febrero la posicion pasa de F1 a F2 y no gana nada: sigue teniendo
    un contrato que ese dia se ha revalorizado un 0,2 %. La serie sin ajustar,
    en cambio, muestra un salto de dos digitos.
    """
    sin_ajustar = serie_continua(contratos, VENCIMIENTOS, metodo="ninguno", dias_antes=5)
    ajustada = serie_continua(contratos, VENCIMIENTOS, metodo="ratio", dias_antes=5)

    falso = sin_ajustar["precio"].pct_change().loc[FECHA_ROLL]
    verdadero = ajustada["precio"].pct_change().loc[FECHA_ROLL]

    # El artefacto compara el precio de F2 el dia 9 con el de F1 el dia 8:
    # 110*1.002^9 / (100*1.001^8) - 1, en torno al 11 %.
    assert falso == pytest.approx(110 * 1.002**9 / (100 * 1.001**8) - 1.0)
    assert falso > 0.10
    # El rendimiento verdadero es el del contrato que se mantiene: 0,2 %.
    assert verdadero == pytest.approx(0.002)
    assert abs(verdadero) < 0.01


def test_el_ajuste_por_ratio_conserva_los_rendimientos_dentro_de_cada_contrato(contratos):
    ajustada = serie_continua(contratos, VENCIMIENTOS, metodo="ratio", dias_antes=5)
    rendimientos = ajustada["precio"].pct_change()
    mismo_contrato = ajustada["contrato"] == ajustada["contrato"].shift(1)

    dentro_de_f1 = rendimientos[mismo_contrato & (ajustada["contrato"] == "F1")]
    dentro_de_f2 = rendimientos[mismo_contrato & (ajustada["contrato"] == "F2")]
    assert len(dentro_de_f1) == 8
    assert len(dentro_de_f2) == 10
    assert np.allclose(dentro_de_f1, 0.001)
    assert np.allclose(dentro_de_f2, 0.002)


@pytest.mark.parametrize("metodo", ["ratio", "diferencia", "ninguno"])
def test_la_serie_continua_termina_en_el_precio_del_contrato_activo(contratos, metodo):
    """El ajuste va hacia atras: el ultimo valor es el precio de mercado de hoy."""
    ajustada = serie_continua(contratos, VENCIMIENTOS, metodo=metodo)
    ultima_fecha = ajustada.index[-1]
    contrato_hoy = ajustada["contrato"].iloc[-1]
    assert contrato_hoy == "F2"
    assert ajustada["precio"].iloc[-1] == pytest.approx(contratos.at[ultima_fecha, contrato_hoy])


@pytest.mark.parametrize(("metodo", "neutro"), [("ratio", 1.0), ("diferencia", 0.0)])
def test_el_ajuste_deja_intacto_el_ultimo_tramo_y_mueve_el_pasado(contratos, metodo, neutro):
    ajustada = serie_continua(contratos, VENCIMIENTOS, metodo=metodo)
    tramo_actual = ajustada["contrato"] == "F2"
    assert (ajustada.loc[tramo_actual, "ajuste"] == neutro).all()
    assert (ajustada.loc[~tramo_actual, "ajuste"] != neutro).all()
    # El escalon del empalme se mide el ultimo dia de vida de F1.
    ancla = pd.Timestamp("2024-02-09")
    esperado = (
        contratos.at[ancla, "F2"] / contratos.at[ancla, "F1"]
        if metodo == "ratio"
        else contratos.at[ancla, "F2"] - contratos.at[ancla, "F1"]
    )
    assert ajustada.at[ancla, "ajuste"] == pytest.approx(esperado)


@pytest.mark.parametrize("dias_antes", [0, 1, 5, 10])
def test_se_rueda_exactamente_dias_antes_del_vencimiento(contratos, dias_antes):
    ajustada = serie_continua(contratos, VENCIMIENTOS, dias_antes=dias_antes)
    fecha_roll = pd.Timestamp(VENCIMIENTOS["F1"]) - pd.Timedelta(days=dias_antes)
    assert ajustada.at[fecha_roll, "contrato"] == "F2"
    assert ajustada.at[fecha_roll - pd.Timedelta(days=1), "contrato"] == "F1"


def test_si_la_fecha_de_roll_no_cotiza_se_rueda_en_la_siguiente_sesion():
    """El 10 de febrero de 2024 es sabado: la cadena cambia el lunes 12."""
    fechas = pd.bdate_range("2024-02-01", periods=15)
    pasos = np.arange(len(fechas))
    contratos = pd.DataFrame(
        {"F1": 100.0 * 1.001**pasos, "F2": 110.0 * 1.002**pasos},
        index=fechas,
    )
    ajustada = serie_continua(contratos, VENCIMIENTOS, dias_antes=5)
    assert ajustada.at[pd.Timestamp("2024-02-09"), "contrato"] == "F1"
    assert ajustada.at[pd.Timestamp("2024-02-12"), "contrato"] == "F2"


def test_la_cadena_encadena_los_dos_rolls_con_tres_contratos(cadena_contango):
    ajustada = serie_continua(cadena_contango, VENCIMIENTOS_CADENA, dias_antes=5)
    assert list(pd.unique(ajustada["contrato"])) == ["M1", "M2", "M3"]
    # Los tres contratos pierden lo mismo cada dia, asi que la serie ajustada
    # decae de forma uniforme y ningun roll deja rastro.
    rendimientos = ajustada["precio"].pct_change().dropna()
    assert np.allclose(rendimientos, DECAIMIENTO - 1.0)
    # El ajuste acumulado del primer tramo es el producto de los dos escalones.
    assert ajustada["ajuste"].iloc[0] == pytest.approx(112.0 / 100.0)


def test_la_serie_ajustada_reproduce_el_rendimiento_verdadero_en_todas_las_fechas():
    """Propiedad general: cada dia se gana lo que gana el contrato que se tiene.

    Con precios aleatorios (semilla fija) y dos rolls, el rendimiento de la
    serie continua debe coincidir, fecha a fecha, con el del contrato activo
    ese dia medido contra su propio precio de la vispera. Es la definicion de
    lo que gana la posicion, y no hay ningun dia en que no se cumpla.
    """
    generador = np.random.default_rng(20240301)
    fechas = pd.date_range("2023-06-01", periods=180, freq="D")
    choques = generador.normal(0.0, 0.012, size=(len(fechas), 3))
    niveles = np.array([100.0, 104.0, 108.0])
    valores = niveles * np.exp(np.cumsum(choques, axis=0))
    contratos = pd.DataFrame(valores, index=fechas, columns=["C1", "C2", "C3"])
    vencimientos = {"C1": "2023-08-15", "C2": "2023-09-15", "C3": "2023-11-30"}

    ajustada = serie_continua(contratos, vencimientos, metodo="ratio", dias_antes=4)
    columna = np.array([contratos.columns.get_loc(c) for c in ajustada["contrato"]])
    crudos = contratos.to_numpy()
    filas = np.arange(len(fechas))
    esperado = crudos[filas[1:], columna[1:]] / crudos[filas[:-1], columna[1:]] - 1.0

    obtenido = ajustada["precio"].pct_change().to_numpy()[1:]
    assert np.allclose(obtenido, esperado)
    # Y el escalon sin ajustar era grande de verdad: la prueba no es vacia.
    sin_ajustar = serie_continua(contratos, vencimientos, metodo="ninguno", dias_antes=4)
    ingenuo = sin_ajustar["precio"].pct_change().to_numpy()[1:]
    assert np.abs(ingenuo - esperado).max() > 0.02


def test_el_ajuste_aditivo_reproduce_las_variaciones_absolutas(cadena_contango):
    ajustada = serie_continua(cadena_contango, VENCIMIENTOS_CADENA, metodo="diferencia")
    columna = np.array([cadena_contango.columns.get_loc(c) for c in ajustada["contrato"]])
    crudos = cadena_contango.to_numpy()
    filas = np.arange(len(cadena_contango))
    esperado = crudos[filas[1:], columna[1:]] - crudos[filas[:-1], columna[1:]]
    assert np.allclose(ajustada["precio"].diff().to_numpy()[1:], esperado)


def test_el_ratio_rechaza_precios_negativos_y_la_diferencia_los_admite():
    """Un spark spread cruza el cero: ahi el ajuste multiplicativo no existe."""
    fechas = pd.date_range("2024-02-01", periods=20, freq="D")
    spread = pd.DataFrame(
        {"F1": np.full(len(fechas), -4.0), "F2": np.full(len(fechas), 3.0)},
        index=fechas,
    )
    with pytest.raises(ErrorDatos, match="diferencia"):
        serie_continua(spread, VENCIMIENTOS, metodo="ratio")

    ajustada = serie_continua(spread, VENCIMIENTOS, metodo="diferencia")
    assert ajustada["precio"].iloc[-1] == pytest.approx(3.0)
    # Ambos contratos estan planos: tras el ajuste no puede quedar ni un salto.
    assert np.allclose(ajustada["precio"].diff().dropna(), 0.0)


# --- errores de la serie continua ----------------------------------------------


def test_unos_vencimientos_que_no_cubren_todos_los_contratos_dan_error(contratos):
    with pytest.raises(ErrorDatos, match="F2"):
        serie_continua(contratos, {"F1": "2024-02-15"})


def test_un_metodo_de_ajuste_desconocido_da_error(contratos):
    with pytest.raises(ErrorDatos, match="desconocido"):
        serie_continua(contratos, VENCIMIENTOS, metodo="promediando_un_poco")


def test_unos_dias_antes_negativos_dan_error(contratos):
    with pytest.raises(ErrorDatos, match="negativo"):
        serie_continua(contratos, VENCIMIENTOS, dias_antes=-3)


def test_un_hueco_en_el_contrato_activo_da_error(contratos):
    con_hueco = contratos.copy()
    con_hueco.loc["2024-02-05", "F1"] = np.nan
    with pytest.raises(ErrorDatos, match="2024-02-05"):
        serie_continua(con_hueco, VENCIMIENTOS)


def test_un_hueco_en_el_contrato_nuevo_el_dia_del_empalme_da_error(contratos):
    con_hueco = contratos.copy()
    con_hueco.loc["2024-02-09", "F2"] = np.nan
    with pytest.raises(ErrorDatos, match="empalme"):
        serie_continua(con_hueco, VENCIMIENTOS)


def test_las_fechas_repetidas_dan_error(contratos):
    repetido = pd.concat([contratos, contratos.iloc[[3]]])
    with pytest.raises(ErrorDatos, match="repetidas"):
        serie_continua(repetido, VENCIMIENTOS)


def test_dos_contratos_con_el_mismo_vencimiento_dan_error(contratos):
    with pytest.raises(ErrorDatos, match="misma fecha"):
        serie_continua(contratos, {"F1": "2024-02-15", "F2": "2024-02-15"})


def test_una_entrada_que_no_es_un_marco_de_contratos_da_error():
    with pytest.raises(ErrorDatos, match="DataFrame"):
        serie_continua(pd.Series([1.0, 2.0]), VENCIMIENTOS)


# --- carry y estado de la curva ------------------------------------------------


def _par(cercano: float, lejano: float) -> tuple[pd.Series, pd.Series]:
    fechas = pd.date_range("2024-01-01", periods=2, freq="D")
    return pd.Series([cercano] * 2, index=fechas), pd.Series([lejano] * 2, index=fechas)


def test_el_carry_es_positivo_en_backwardation_y_negativo_en_contango():
    # A mano: (105-100)/100 = 0,05 en 30 dias, o sea 0,05 * 365,25/30 = 0,60875.
    cercano, lejano = _par(105.0, 100.0)
    assert rendimiento_roll(cercano, lejano, 30).iloc[0] == pytest.approx(0.60875)

    cercano, lejano = _par(95.0, 100.0)
    assert rendimiento_roll(cercano, lejano, 30).iloc[0] == pytest.approx(-0.60875)

    cercano, lejano = _par(100.0, 100.0)
    assert rendimiento_roll(cercano, lejano, 30).iloc[0] == 0.0


def test_el_carry_se_anualiza_con_la_distancia_entre_vencimientos():
    cercano, lejano = _par(105.0, 100.0)
    mensual = rendimiento_roll(cercano, lejano, 30).iloc[0]
    trimestral = rendimiento_roll(cercano, lejano, 90).iloc[0]
    # El mismo diferencial repartido en el triple de tiempo es un tercio de tasa.
    assert trimestral == pytest.approx(mensual / 3.0)
    assert mensual == pytest.approx(0.05 * DIAS_ANIO_NATURAL / 30.0)


def test_un_plazo_entre_vencimientos_no_positivo_da_error():
    cercano, lejano = _par(105.0, 100.0)
    with pytest.raises(ErrorDatos, match="positivo"):
        rendimiento_roll(cercano, lejano, 0)


def test_un_contrato_lejano_nulo_da_error():
    cercano, lejano = _par(105.0, 0.0)
    with pytest.raises(ErrorDatos, match="nulos"):
        rendimiento_roll(cercano, lejano)


def test_dos_series_sin_fechas_comunes_dan_error():
    cercano = pd.Series([100.0], index=pd.date_range("2024-01-01", periods=1))
    lejano = pd.Series([101.0], index=pd.date_range("2024-06-01", periods=1))
    with pytest.raises(ErrorDatos, match="ninguna fecha"):
        rendimiento_roll(cercano, lejano)


def test_estado_curva_clasifica_los_tres_casos():
    fechas = pd.date_range("2024-01-01", periods=3, freq="D")
    cercano = pd.Series([105.0, 95.0, 100.0], index=fechas)
    lejano = pd.Series([100.0, 100.0, 100.0], index=fechas)
    estados = estado_curva(cercano, lejano)
    assert list(estados) == ["backwardation", "contango", "plana"]


def test_la_tolerancia_aplana_las_diferencias_pequenias():
    fechas = pd.date_range("2024-01-01", periods=2, freq="D")
    cercano = pd.Series([100.5, 99.5], index=fechas)
    lejano = pd.Series([100.0, 100.0], index=fechas)
    assert list(estado_curva(cercano, lejano)) == ["backwardation", "contango"]
    # Media diferencia por cien no es una senal de mercado, es ruido.
    assert list(estado_curva(cercano, lejano, tolerancia=0.01)) == ["plana", "plana"]


def test_una_tolerancia_negativa_da_error():
    cercano, lejano = _par(105.0, 100.0)
    with pytest.raises(ErrorDatos, match="cero o positiva"):
        estado_curva(cercano, lejano, tolerancia=-0.01)


def test_el_signo_del_carry_coincide_con_el_estado_de_la_curva(cadena_contango):
    cercano, lejano = cadena_contango["M1"], cadena_contango["M2"]
    estados = estado_curva(cercano, lejano)
    carry = rendimiento_roll(cercano, lejano, 29)
    assert (estados == "contango").all()
    assert (carry < 0).all()

    invertido = estado_curva(lejano, cercano)
    assert (invertido == "backwardation").all()
    assert (rendimiento_roll(lejano, cercano, 29) > 0).all()


# --- pendiente de la curva -----------------------------------------------------


def test_la_pendiente_recupera_la_tasa_de_una_curva_log_lineal():
    """Con precios exp(a + b*plazo), la regresion devuelve b exactamente."""
    plazos = {"M1": 1.0, "M3": 3.0, "M6": 6.0, "M12": 12.0}
    tasa = 0.012
    fechas = pd.date_range("2024-01-01", periods=3, freq="D")
    niveles = np.array([100.0, 101.0, 99.0])
    curva = pd.DataFrame(
        {c: niveles * np.exp(tasa * m) for c, m in plazos.items()},
        index=fechas,
    )
    pendiente = pendiente_curva(curva, plazos)
    assert np.allclose(pendiente, tasa)


def test_la_pendiente_cambia_de_signo_con_la_forma_de_la_curva():
    plazos = [1.0, 3.0, 6.0]
    fechas = pd.date_range("2024-01-01", periods=2, freq="D")
    contango = pd.DataFrame([[100.0, 103.0, 108.0]] * 2, index=fechas, columns=["a", "b", "c"])
    backwardation = contango.iloc[:, ::-1]
    backwardation.columns = ["a", "b", "c"]
    assert (pendiente_curva(contango, plazos) > 0).all()
    assert (pendiente_curva(backwardation, plazos) < 0).all()


def test_una_fecha_con_un_solo_contrato_cotizado_no_tiene_pendiente():
    plazos = [1.0, 3.0, 6.0]
    fechas = pd.date_range("2024-01-01", periods=2, freq="D")
    curva = pd.DataFrame(
        [[100.0, 103.0, 108.0], [100.0, np.nan, np.nan]],
        index=fechas,
        columns=["a", "b", "c"],
    )
    pendiente = pendiente_curva(curva, plazos)
    assert np.isfinite(pendiente.iloc[0])
    assert np.isnan(pendiente.iloc[1])


def test_la_pendiente_usa_los_contratos_disponibles_aunque_falte_alguno():
    plazos = {"a": 1.0, "b": 3.0, "c": 6.0}
    fechas = pd.date_range("2024-01-01", periods=1, freq="D")
    tasa = 0.02
    completa = pd.DataFrame({c: [100.0 * np.exp(tasa * m)] for c, m in plazos.items()}, index=fechas)
    con_hueco = completa.copy()
    con_hueco.loc[fechas[0], "b"] = np.nan
    assert pendiente_curva(con_hueco, plazos).iloc[0] == pytest.approx(tasa)


def test_la_pendiente_exige_precios_positivos_y_plazos_completos():
    plazos = {"a": 1.0, "b": 3.0}
    fechas = pd.date_range("2024-01-01", periods=1, freq="D")
    curva = pd.DataFrame({"a": [-5.0], "b": [3.0]}, index=fechas)
    with pytest.raises(ErrorDatos, match="log-precio"):
        pendiente_curva(curva, plazos)
    with pytest.raises(ErrorDatos, match="Faltan plazos"):
        pendiente_curva(curva.abs(), {"a": 1.0})
    with pytest.raises(ErrorDatos, match="dos contratos"):
        pendiente_curva(curva[["a"]], {"a": 1.0})
    with pytest.raises(ErrorDatos, match="plazos distintos"):
        pendiente_curva(curva.abs(), {"a": 2.0, "b": 2.0})


# --- interpolacion a plazo constante -------------------------------------------


def test_la_interpolacion_lineal_pasa_por_los_puntos_y_por_el_punto_medio():
    curva = pd.Series([100.0, 104.0, 112.0], index=[1.0, 3.0, 7.0])
    # A mano: entre 1 y 3, 100 + 4*(1/2) = 102; entre 3 y 7, 104 + 8*(2/4) = 108.
    interpolada = interpolar_curva(curva, [1.0, 2.0, 5.0, 7.0])
    assert list(interpolada) == [100.0, 102.0, 108.0, 112.0]
    assert list(interpolada.index) == [1.0, 2.0, 5.0, 7.0]


def test_la_interpolacion_log_lineal_es_exacta_en_una_curva_geometrica():
    # 100 y 121 a plazos 0 y 2: la media geometrica es 110, no 110,5.
    curva = pd.Series([100.0, 121.0], index=[0.0, 2.0])
    assert interpolar_curva(curva, 1.0, metodo="log_lineal").iloc[0] == pytest.approx(110.0)
    assert interpolar_curva(curva, 1.0, metodo="lineal").iloc[0] == pytest.approx(110.5)


def test_la_interpolacion_cubica_reproduce_un_polinomio_de_grado_tres():
    plazos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    valores = 100.0 + 2.0 * plazos - 0.5 * plazos**2 + 0.05 * plazos**3
    curva = pd.Series(valores, index=plazos)
    objetivo = 2.5
    esperado = 100.0 + 2.0 * objetivo - 0.5 * objetivo**2 + 0.05 * objetivo**3
    obtenido = interpolar_curva(curva, objetivo, metodo="cubica").iloc[0]
    assert obtenido == pytest.approx(esperado)


def test_el_plazo_constante_permite_comparar_dos_dias_distintos():
    """Los contratos envejecen un dia cada dia; el plazo objetivo no.

    Dos curvas identicas en forma pero cotizadas con los contratos un dia mas
    cerca del vencimiento dan el mismo precio a tres meses si se interpola, y
    precios distintos si se compara contrato con contrato.
    """
    tasa = 0.01
    hoy = pd.Series({1.0: 100.0 * np.exp(tasa), 4.0: 100.0 * np.exp(4 * tasa)})
    manana = pd.Series({0.9: 100.0 * np.exp(0.9 * tasa), 3.9: 100.0 * np.exp(3.9 * tasa)})
    a_tres_meses_hoy = interpolar_curva(hoy, 3.0, metodo="log_lineal").iloc[0]
    a_tres_meses_manana = interpolar_curva(manana, 3.0, metodo="log_lineal").iloc[0]
    assert a_tres_meses_hoy == pytest.approx(a_tres_meses_manana)
    # El primer contrato, en cambio, ha cambiado de precio sin que el mercado
    # se haya movido: solo por acercarse el vencimiento.
    assert hoy.iloc[0] != pytest.approx(manana.iloc[0])


def test_la_curva_no_se_extrapola():
    curva = pd.Series([100.0, 104.0, 112.0], index=[1.0, 3.0, 7.0])
    with pytest.raises(ErrorDatos, match="fuera del rango"):
        interpolar_curva(curva, [5.0, 12.0])
    with pytest.raises(ErrorDatos, match="fuera del rango"):
        interpolar_curva(curva, 0.5)


def test_los_errores_de_interpolacion_explican_que_hacer():
    curva = pd.Series([100.0, 104.0, 112.0], index=[1.0, 3.0, 7.0])
    with pytest.raises(ErrorDatos, match="desconocido"):
        interpolar_curva(curva, 2.0, metodo="a_ojo")
    with pytest.raises(ErrorDatos, match="lineal"):
        interpolar_curva(curva, 2.0, metodo="cubica")
    with pytest.raises(ErrorDatos, match="repetidos"):
        interpolar_curva(pd.Series([100.0, 101.0], index=[1.0, 1.0]), 1.0)
    with pytest.raises(ErrorDatos, match="positivos"):
        interpolar_curva(pd.Series([-1.0, 2.0], index=[1.0, 2.0]), 1.5, metodo="log_lineal")
    with pytest.raises(ErrorDatos, match="Series"):
        interpolar_curva([100.0, 104.0], 1.0)


# --- descomposicion del rendimiento --------------------------------------------


@pytest.mark.parametrize("metodo", ["ratio", "diferencia"])
def test_el_rendimiento_total_es_la_suma_del_precio_y_del_roll(cadena_contango, metodo):
    partes = descomponer_rendimiento(serie_continua(cadena_contango, VENCIMIENTOS_CADENA, metodo))
    assert np.allclose(
        partes["rendimiento_total"],
        partes["rendimiento_precio"] + partes["rendimiento_roll"],
    )


def test_una_materia_prima_puede_subir_de_precio_y_hacer_perder_dinero(cadena_contango):
    """La paradoja del contango, con numeros de cabeza.

    El precio de portada pasa de 100 a 112 (dos rolls de +6 % y +5,66 %) mientras
    cada contrato pierde un 0,1 % diario durante 90 dias. La serie que se ve en
    pantalla sube un 2,4 %; la posicion pierde un 8,6 %. La diferencia entera es
    el roll.
    """
    partes = descomponer_rendimiento(
        serie_continua(cadena_contango, VENCIMIENTOS_CADENA, dias_antes=5)
    )
    total = (1.0 + partes["rendimiento_total"]).prod()
    precio = (1.0 + partes["rendimiento_precio"]).prod()

    assert total == pytest.approx(DECAIMIENTO**90)
    assert precio == pytest.approx(1.12 * DECAIMIENTO**90)
    assert precio > 1.0 > total
    assert partes["rendimiento_roll"].sum() < -0.11


def test_el_roll_solo_aparece_en_las_fechas_de_roll(cadena_contango):
    partes = descomponer_rendimiento(
        serie_continua(cadena_contango, VENCIMIENTOS_CADENA, dias_antes=5)
    )
    con_roll = partes.index[partes["rendimiento_roll"].abs() > 1e-12]
    assert list(con_roll) == [pd.Timestamp("2024-01-26"), pd.Timestamp("2024-02-24")]
    # A mano: 0,999 * (1 - 1,06) = -0,05994 el dia del primer empalme.
    assert partes.at[con_roll[0], "rendimiento_roll"] == pytest.approx(DECAIMIENTO * (1.0 - 1.06))


def test_el_roll_es_negativo_en_contango_y_positivo_en_backwardation(
    cadena_contango, cadena_backwardation
):
    en_contango = descomponer_rendimiento(serie_continua(cadena_contango, VENCIMIENTOS_CADENA))
    en_backwardation = descomponer_rendimiento(
        serie_continua(cadena_backwardation, VENCIMIENTOS_CADENA)
    )
    assert en_contango["rendimiento_roll"].sum() < 0.0
    assert en_backwardation["rendimiento_roll"].sum() > 0.0
    # Mismo decaimiento dentro de cada contrato: lo unico que cambia es el carry.
    assert en_backwardation["rendimiento_total"].sum() == pytest.approx(
        en_contango["rendimiento_total"].sum()
    )


def test_una_serie_sin_ajustar_no_se_puede_descomponer(cadena_contango):
    sin_ajustar = serie_continua(cadena_contango, VENCIMIENTOS_CADENA, metodo="ninguno")
    with pytest.raises(ErrorDatos, match="sin ajustar"):
        descomponer_rendimiento(sin_ajustar)


def test_descomponer_exige_la_salida_de_serie_continua():
    with pytest.raises(ErrorDatos, match="serie_continua"):
        descomponer_rendimiento(pd.DataFrame({"precio": [1.0, 2.0]}))
    with pytest.raises(ErrorDatos, match="dos fechas"):
        descomponer_rendimiento(pd.DataFrame({"precio": [1.0], "ajuste": [1.0]}))
    with pytest.raises(ErrorDatos, match="desconocido"):
        descomponer_rendimiento(
            pd.DataFrame({"precio": [1.0, 2.0], "ajuste": [1.0, 1.0]}), metodo="inventado"
        )
