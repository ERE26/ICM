"""Pruebas de Monte Carlo, bootstrap y copulas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icm.comun import ErrorDatos
from icm.datos import matriz_covarianzas
from icm.riesgo import (
    CopulaGaussiana,
    CopulaT,
    ajustar_copula_gaussiana,
    ajustar_copula_t,
    bootstrap_historico,
    comparar_copulas,
    dependencia_cola_empirica,
    es_montecarlo,
    resumen_simulacion,
    simular_cartera,
    simular_rendimientos,
    simular_rendimientos_copula,
    trayectorias_simuladas,
    var_copula,
    var_montecarlo,
)
from icm.riesgo.copulas import correlacion_por_kendall, pseudo_observaciones

MU = pd.Series({"X": 0.0005, "Y": 0.0003})
COV = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0009]], index=["X", "Y"], columns=["X", "Y"])
PESOS = pd.Series({"X": 0.5, "Y": 0.5})


# --- Monte Carlo ---------------------------------------------------------------


def test_la_simulacion_es_reproducible():
    primera = simular_cartera(MU, COV, PESOS, 5000, semilla=1)
    segunda = simular_cartera(MU, COV, PESOS, 5000, semilla=1)
    pd.testing.assert_series_equal(primera, segunda)


def test_la_simulacion_recupera_la_media_y_la_varianza_teoricas():
    simulaciones = simular_cartera(MU, COV, PESOS, 200_000, semilla=2)
    media_teorica = float(PESOS @ MU)
    varianza_teorica = float(PESOS @ COV @ PESOS)
    # Se compara en logaritmos, que es como se genera el proceso.
    logaritmos = np.log1p(simulaciones)
    assert logaritmos.mean() == pytest.approx(media_teorica, abs=0.0003)
    assert logaritmos.var(ddof=1) == pytest.approx(varianza_teorica, rel=0.02)


def test_la_varianza_crece_linealmente_con_el_horizonte():
    a_un_dia = np.log1p(simular_cartera(MU, COV, PESOS, 100_000, 1, semilla=3)).var(ddof=1)
    a_diez_dias = np.log1p(simular_cartera(MU, COV, PESOS, 100_000, 10, semilla=3)).var(ddof=1)
    assert a_diez_dias == pytest.approx(10 * a_un_dia, rel=0.05)


def test_la_simulacion_conserva_la_correlacion():
    simulaciones = simular_rendimientos(MU, COV, 100_000, semilla=4)
    correlacion_teorica = COV.iloc[0, 1] / np.sqrt(COV.iloc[0, 0] * COV.iloc[1, 1])
    observada = np.corrcoef(simulaciones.T)[0, 1]
    assert observada == pytest.approx(correlacion_teorica, abs=0.02)


def test_la_t_multivariante_engorda_las_colas():
    normal = simular_cartera(MU, COV, PESOS, 100_000, distribucion="normal", semilla=5)
    t_student = simular_cartera(
        MU, COV, PESOS, 100_000, distribucion="t", grados_libertad=4.0, semilla=5
    )
    assert var_montecarlo(t_student, 0.99) > var_montecarlo(normal, 0.99)
    # Pero con la misma varianza: la t se reescala para no cambiar la escala.
    assert t_student.std(ddof=1) == pytest.approx(normal.std(ddof=1), rel=0.10)


def test_el_es_simulado_supera_al_var_simulado():
    simulaciones = simular_cartera(MU, COV, PESOS, 50_000, semilla=6)
    assert es_montecarlo(simulaciones, 0.99) > var_montecarlo(simulaciones, 0.99)


def test_una_distribucion_desconocida_da_error():
    with pytest.raises(ErrorDatos, match="Distribucion"):
        simular_rendimientos(MU, COV, 100, distribucion="logistica")


def test_una_matriz_casi_singular_no_rompe_la_simulacion():
    # Dos activos practicamente identicos: la Cholesky directa puede fallar.
    casi_singular = pd.DataFrame(
        [[0.0004, 0.0004], [0.0004, 0.0004]], index=["X", "Y"], columns=["X", "Y"]
    )
    simulaciones = simular_rendimientos(MU, casi_singular, 1000, semilla=7)
    assert np.isfinite(simulaciones).all()


def test_el_resumen_recoge_las_probabilidades_de_perdida():
    simulaciones = simular_cartera(MU, COV, PESOS, 50_000, horizonte=20, semilla=8)
    resumen = resumen_simulacion(simulaciones, valor_cartera=1_000_000)
    assert 0.0 <= resumen["prob_perdida"] <= 1.0
    assert resumen["prob_perder_mas_5%"] >= resumen["prob_perder_mas_10%"]
    assert resumen["var_99%_monetario"] == pytest.approx(resumen["var_99%"] * 1_000_000)


def test_las_trayectorias_arrancan_todas_del_mismo_valor():
    trayectorias = trayectorias_simuladas(MU, COV, PESOS, 60, 50, valor_inicial=100.0)
    assert trayectorias.shape == (61, 50)
    np.testing.assert_allclose(trayectorias.iloc[0].to_numpy(), 100.0)
    assert (trayectorias > 0).all().all()


# --- bootstrap -----------------------------------------------------------------


def test_el_bootstrap_reproduce_el_riesgo_historico(rend, pesos_iguales):
    simulaciones = bootstrap_historico(rend, pesos_iguales, 50_000, horizonte=1, semilla=9)
    historico = rend.mul(pesos_iguales, axis=1).sum(axis=1)
    assert var_montecarlo(simulaciones, 0.95) == pytest.approx(
        -np.quantile(historico, 0.05), rel=0.10
    )


def test_el_bootstrap_por_bloques_toma_tramos_consecutivos(pesos_iguales):
    # Con el bloque igual al horizonte, cada trayectoria simulada tiene que ser
    # literalmente una ventana consecutiva de la historia. La version anterior
    # de esta prueba construia la referencia con mode="wrap" y comparaba con
    # <=, asi que la cumplian por igual la implementacion circular y la que no
    # lo es: el conjunto envolvente contiene al otro. Aqui se separan las
    # ventanas que cruzan el final y se exige que salgan, que es justo lo que
    # el muestreo no circular no puede producir.
    generador = np.random.default_rng(17)
    historia = pd.DataFrame(generador.standard_normal((60, 3)) * 0.01, columns=pesos_iguales.index)
    simulaciones = bootstrap_historico(historia, pesos_iguales, 200, horizonte=5, tamano_bloque=5, semilla=18)

    cartera = historia.mul(pesos_iguales, axis=1).sum(axis=1).to_numpy()
    n = len(cartera)

    def acumulado(inicio):
        tramo = np.take(cartera, range(inicio, inicio + 5), mode="wrap")
        return round(float(np.prod(1.0 + tramo) - 1.0), 10)

    interiores = {acumulado(i) for i in range(n - 4)}
    envolventes = {acumulado(i) for i in range(n - 4, n)}
    obtenidas = {round(float(v), 10) for v in simulaciones}

    assert obtenidas <= interiores | envolventes
    assert obtenidas & envolventes, "el muestreo no esta llegando al final de la serie"


def test_el_bootstrap_por_bloques_puede_repetir_una_crisis_entera(pesos_iguales):
    # Historia tranquila con un unico episodio de diez sesiones muy malas
    # seguidas. Muestreando dias sueltos casi nunca se encadenan varias de
    # ellas; muestreando bloques de diez, el episodio completo sale con la
    # frecuencia con que aparece en la historia, y la cola lo refleja.
    activos = list(pesos_iguales.index)
    diario = np.full((250, len(activos)), 0.0005)
    diario[100:110] = -0.03
    historia = pd.DataFrame(diario, columns=activos)

    sueltos = bootstrap_historico(historia, pesos_iguales, 50_000, 10, None, semilla=19)
    bloques = bootstrap_historico(historia, pesos_iguales, 50_000, 10, 10, semilla=19)

    assert var_montecarlo(bloques, 0.99) > 0.20
    assert var_montecarlo(sueltos, 0.99) < 0.15


def test_el_bootstrap_rechaza_un_bloque_invalido(rend, pesos_iguales):
    with pytest.raises(ErrorDatos, match="tamanio de bloque"):
        bootstrap_historico(rend, pesos_iguales, 100, 5, tamano_bloque=0)


# --- copulas -------------------------------------------------------------------


def test_las_pseudo_observaciones_caen_dentro_del_intervalo_abierto(rend):
    uniformes = pseudo_observaciones(rend)
    assert (uniformes > 0).all().all()
    assert (uniformes < 1).all().all()


def test_las_pseudo_observaciones_son_casi_uniformes(rend):
    uniformes = pseudo_observaciones(rend)
    assert uniformes.mean().mean() == pytest.approx(0.5, abs=0.01)


def test_la_correlacion_por_kendall_es_una_matriz_de_correlacion_valida(rend):
    matriz = correlacion_por_kendall(rend)
    np.testing.assert_allclose(np.diag(matriz.to_numpy()), 1.0)
    np.testing.assert_allclose(matriz.to_numpy(), matriz.to_numpy().T)
    assert np.linalg.eigvalsh(matriz.to_numpy()).min() > 0


def test_la_correlacion_por_kendall_se_acerca_a_pearson_con_datos_normales():
    generador = np.random.default_rng(11)
    factor = generador.standard_normal(20_000)
    marco = pd.DataFrame(
        {
            "X": factor * 0.7 + generador.standard_normal(20_000) * np.sqrt(1 - 0.49),
            "Y": factor * 0.7 + generador.standard_normal(20_000) * np.sqrt(1 - 0.49),
        }
    )
    kendall = correlacion_por_kendall(marco).iloc[0, 1]
    assert kendall == pytest.approx(marco.corr().iloc[0, 1], abs=0.02)


def test_la_copula_gaussiana_no_tiene_dependencia_de_cola(rend):
    assert ajustar_copula_gaussiana(rend).dependencia_cola() == 0.0


def test_la_copula_t_si_tiene_dependencia_de_cola(rend):
    copula = ajustar_copula_t(rend)
    assert 0.0 < copula.dependencia_cola() < 1.0


def test_menos_grados_de_libertad_implican_mas_dependencia_de_cola():
    correlacion = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    gruesa = CopulaT(correlacion, 3.0).dependencia_cola()
    fina = CopulaT(correlacion, 30.0).dependencia_cola()
    assert gruesa > fina


def test_la_copula_t_ajusta_unos_grados_de_libertad_razonables(rend):
    copula = ajustar_copula_t(rend)
    assert 2.0 < copula.grados_libertad < 60.0


def test_la_copula_t_gana_a_la_gaussiana_con_datos_de_colas_gruesas(rend):
    uniformes = pseudo_observaciones(rend)
    gaussiana = ajustar_copula_gaussiana(rend).log_verosimilitud(uniformes)
    t_student = ajustar_copula_t(rend).log_verosimilitud(uniformes)
    assert t_student > gaussiana


def test_las_simulaciones_de_la_copula_son_uniformes(rend):
    uniformes = ajustar_copula_t(rend).simular_uniformes(20_000, semilla=12)
    assert uniformes.mean().mean() == pytest.approx(0.5, abs=0.01)
    assert uniformes.min().min() > 0.0
    assert uniformes.max().max() < 1.0


def test_la_copula_conserva_la_correlacion_por_rangos(rend):
    copula = ajustar_copula_gaussiana(rend)
    simuladas = copula.simular_uniformes(50_000, semilla=13)
    objetivo = copula.correlacion.iloc[0, 1]
    obtenida = correlacion_por_kendall(simuladas).iloc[0, 1]
    assert obtenida == pytest.approx(objetivo, abs=0.03)


def test_las_marginales_simuladas_reproducen_las_reales(rend):
    simulados = simular_rendimientos_copula(rend, n_simulaciones=50_000, tipo="t", semilla=14)
    for activo in rend.columns:
        assert simulados[activo].mean() == pytest.approx(rend[activo].mean(), abs=0.001)
        assert simulados[activo].std(ddof=1) == pytest.approx(rend[activo].std(ddof=1), rel=0.10)


def test_la_dependencia_de_cola_empirica_es_simetrica(rend):
    matriz = dependencia_cola_empirica(rend, 0.05)
    np.testing.assert_allclose(matriz.to_numpy(), matriz.to_numpy().T, equal_nan=True)
    np.testing.assert_allclose(np.diag(matriz.to_numpy()), 1.0)


def test_la_copula_gaussiana_infravalora_el_riesgo_de_cola(rend, pesos_iguales):
    con_t = var_copula(rend, pesos_iguales, 0.99, "t", 50_000, semilla=15)
    con_gaussiana = var_copula(rend, pesos_iguales, 0.99, "gaussiana", 50_000, semilla=15)
    assert con_t["es"] > con_gaussiana["es"]


def test_un_tipo_de_copula_desconocido_da_error(rend):
    with pytest.raises(ErrorDatos, match="desconocido"):
        simular_rendimientos_copula(rend, tipo="arquimediana")


def test_la_comparacion_de_copulas_devuelve_las_dos_filas(rend, pesos_iguales):
    tabla = comparar_copulas(rend, pesos_iguales, 0.99, 20_000)
    assert set(tabla.index) == {"gaussiana", "t"}
    assert tabla.loc["t", "aic"] < tabla.loc["gaussiana", "aic"]
    assert tabla.loc["gaussiana", "dependencia_cola"] == 0.0


def test_la_copula_gaussiana_es_el_limite_de_la_t_con_muchos_grados():
    correlacion = pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    uniformes = CopulaGaussiana(correlacion).simular_uniformes(30_000, semilla=16)
    log_gaussiana = CopulaGaussiana(correlacion).log_verosimilitud(uniformes)
    log_t = CopulaT(correlacion, 200.0).log_verosimilitud(uniformes)
    assert log_t == pytest.approx(log_gaussiana, rel=0.05)


def test_la_verosimilitud_no_depende_del_orden_de_las_columnas():
    # Las uniformes se cruzaban con la matriz de correlacion por POSICION
    # aunque ambas llevan etiquetas: bastaba reordenar las columnas -- un
    # reindex, un merge, un groupby -- para evaluar la verosimilitud de otro
    # modelo sin que nada avisara. Medido sobre estos mismos tres activos, la
    # gaussiana daba +788,19 con el orden natural y -3654,30 con el inverso.
    generador = np.random.default_rng(7)
    ruido = generador.standard_normal((600, 3))
    datos = pd.DataFrame(
        {
            "A": ruido[:, 0],
            "B": 0.9 * ruido[:, 0] + 0.44 * ruido[:, 1],
            "C": -0.8 * ruido[:, 0] + 0.6 * ruido[:, 2],
        }
    )
    uniformes = pseudo_observaciones(datos)
    invertidas = uniformes[["C", "B", "A"]]
    for copula in (ajustar_copula_gaussiana(datos), ajustar_copula_t(datos)):
        assert copula.log_verosimilitud(invertidas) == pytest.approx(
            copula.log_verosimilitud(uniformes)
        )


def test_la_verosimilitud_exige_los_activos_de_la_copula(rend):
    copula = ajustar_copula_t(rend)
    with pytest.raises(ErrorDatos, match="Faltan columnas"):
        copula.log_verosimilitud(pseudo_observaciones(rend[[rend.columns[0]]]))


def test_la_dependencia_de_cola_declarada_es_la_que_se_simula():
    # simular_uniformes proyectaba la matriz sobre las definidas positivas y
    # dependencia_cola leia la original, asi que el objeto declaraba una cosa
    # y generaba otra. Aqui la matriz pedida no es definida positiva (menor
    # autovalor -0,043) y las dos cifras tienen que salir de la reparada.
    pedida = pd.DataFrame(
        [[1.0, 0.92, 0.92], [0.92, 1.0, 0.58], [0.92, 0.58, 1.0]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    with pytest.warns(RuntimeWarning, match="definidas positivas"):
        copula = CopulaT(pedida, 3.0)

    guardada = float(copula.correlacion.loc["A", "C"])
    assert guardada < 0.92  # la reparacion ha movido la correlacion pedida

    simuladas = copula.simular_uniformes(200_000, semilla=1)
    # La correlacion que de verdad se simula, leida de los rangos: tiene que
    # ser la guardada (0,8898), no la pedida (0,9200). Es la comprobacion
    # limpia; la lambda empirica converge muy despacio y no sirve de arbitro.
    tau = simuladas[["A", "C"]].corr(method="kendall").iloc[0, 1]
    assert float(np.sin(np.pi * tau / 2.0)) == pytest.approx(guardada, abs=0.01)

    # Y la dependencia de cola declarada sale de esa misma matriz.
    declarada = copula.dependencia_cola(("A", "C"))
    condicionante = simuladas["C"] <= 0.01
    observada = float(((simuladas["A"] <= 0.01) & condicionante).sum() / condicionante.sum())
    assert observada == pytest.approx(declarada, abs=0.05)


def test_una_correlacion_groseramente_imposible_se_rechaza():
    # A y C casi perfectamente opuestos, y los dos casi identicos a B: no hay
    # ningun conjunto de activos con esas correlaciones. Repararla en silencio
    # devolvia cifras de riesgo de una matriz distinta de la pedida.
    imposible = pd.DataFrame(
        [[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    with pytest.raises(ErrorDatos, match="semidefinida positiva"):
        CopulaGaussiana(imposible)


@pytest.mark.parametrize("grados", [1.5, 2.0, -3.0, float("nan")])
def test_la_copula_t_rechaza_grados_de_libertad_imposibles(grados):
    correlacion = pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    with pytest.raises(ErrorDatos, match="mayores que 2"):
        CopulaT(correlacion, grados)


def test_se_puede_simular_con_columnas_no_textuales(rend):
    # 'activos' devuelve str(c), asi que con columnas numericas el simulador
    # indexaba las uniformes con un nombre que no existia: KeyError: 0.
    numerico = rend.copy()
    numerico.columns = range(rend.shape[1])
    simulados = simular_rendimientos_copula(numerico, n_simulaciones=2_000, semilla=21)
    assert list(simulados.columns) == list(numerico.columns)
    for posicion in numerico.columns:
        assert simulados[posicion].mean() == pytest.approx(numerico[posicion].mean(), abs=0.002)


def test_el_var_de_la_copula_no_suma_rendimientos_logaritmicos(rend, pesos_iguales):
    # El rendimiento de una cartera no es la media ponderada de los
    # logaritmos. Con la marca puesta hay que pasar por expm1 antes de
    # ponderar, igual que hace el bootstrap.
    logaritmicos = rend.copy()
    logaritmicos.attrs["tipo_rendimiento"] = "log"
    simples = rend.copy()
    simples.attrs["tipo_rendimiento"] = "simple"

    con_log = var_copula(logaritmicos, pesos_iguales, 0.99, "gaussiana", 20_000, semilla=22)
    con_simple = var_copula(simples, pesos_iguales, 0.99, "gaussiana", 20_000, semilla=22)
    assert con_log["var"] != pytest.approx(con_simple["var"], rel=1e-9)
    # La concavidad del logaritmo hace que la perdida verdadera sea menor.
    assert con_log["var"] < con_simple["var"]


def test_simular_con_una_copula_de_otros_activos_da_error(rend):
    copula = ajustar_copula_t(rend)
    otros = rend.rename(columns={c: f"otro_{c}" for c in rend.columns})
    with pytest.raises(ErrorDatos, match="alinee las columnas"):
        simular_rendimientos_copula(otros, copula, 100)
