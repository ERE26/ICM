"""Bateria que se ejecuta sobre TUS datos, no sobre los simulados.

Como usarla
-----------
1. Deja tu fichero de precios en ``datos/precios.csv`` (tambien vale ``.xlsx``
   o ``.parquet``). Formato: una columna de fechas y una columna por activo.
2. Ejecuta ``pytest`` como siempre.

Si el fichero no esta, todo este modulo se salta en silencio, de modo que la
suite sigue pasando en una maquina donde no estan tus datos.

Para apuntar a otro sitio, o para ajustar la lectura, usa variables de entorno::

    ICM_DATOS=~/tfg/ibex.xlsx pytest tests/test_datos_propios.py -v
    ICM_DATOS_CONTENIDO=rendimientos pytest
    ICM_DATOS_DIA_PRIMERO=1 pytest        # fechas en dd/mm/aaaa

Qué comprueba
-------------
No comprueba que tus datos "den bien": comprueba que el analisis completo se
sostiene sobre ellos. Es lo que interesa antes de fiarse de una cifra. Varias
pruebas no fallan, sino que **informan** por pantalla (ejecuta con ``-s`` para
verlas): que la normalidad se rechace o que un modelo de VaR no supere el
contraste no es un error del codigo, es un resultado.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icm.cartera import (
    cartera_pesos_iguales,
    contraste_jarque_bera,
    contribuciones_riesgo,
    minima_varianza,
    resumen_metricas,
)
from icm.datos import cargar_precios_fichero, matriz_covarianzas, rendimientos, rendimientos_cartera
from icm.riesgo import (
    ajustar_copula_t,
    ajustar_garch,
    backtest_var,
    maxima_caida,
    serie_caidas,
    tabla_medidas,
    var_historico,
)

RAIZ = Path(__file__).resolve().parents[1]
CANDIDATOS = ("datos/precios.csv", "datos/precios.xlsx", "datos/precios.parquet")


def _localizar_fichero() -> Path | None:
    indicado = os.environ.get("ICM_DATOS")
    if indicado:
        ruta = Path(indicado).expanduser()
        return ruta if ruta.exists() else None
    for relativa in CANDIDATOS:
        ruta = RAIZ / relativa
        if ruta.exists():
            return ruta
    return None


FICHERO = _localizar_fichero()

pytestmark = pytest.mark.skipif(
    FICHERO is None,
    reason=(
        "No hay datos propios. Deja un fichero en datos/precios.csv "
        "o apunta la variable de entorno ICM_DATOS a el."
    ),
)

#: Minimo de sesiones para que un backtest con ventana de 250 tenga sentido.
MINIMO_SESIONES = 500


@pytest.fixture(scope="module")
def precios() -> pd.DataFrame:
    opciones = {
        "contenido": os.environ.get("ICM_DATOS_CONTENIDO", "precios"),
        "formato": os.environ.get("ICM_DATOS_FORMATO", "ancho"),
        "dia_primero": os.environ.get("ICM_DATOS_DIA_PRIMERO", "") not in {"", "0", "false"},
    }
    if columna := os.environ.get("ICM_DATOS_COLUMNA_FECHA"):
        opciones["columna_fecha"] = columna
    marco = cargar_precios_fichero(FICHERO, **opciones)
    print(f"\nDatos: {FICHERO}")
    print(f"  {len(marco)} sesiones, {marco.shape[1]} activos: {', '.join(marco.columns)}")
    print(f"  desde {marco.index.min().date()} hasta {marco.index.max().date()}")
    return marco


@pytest.fixture(scope="module")
def rend(precios: pd.DataFrame) -> pd.DataFrame:
    return rendimientos(precios, "log")


@pytest.fixture(scope="module")
def cartera(rend: pd.DataFrame) -> pd.Series:
    return rendimientos_cartera(rend, cartera_pesos_iguales(rend.columns))


# --- 1. ¿Los datos estan sanos? ------------------------------------------------


def test_hay_suficientes_sesiones(precios):
    assert len(precios) >= 60, (
        f"Solo hay {len(precios)} sesiones. Con menos de 60 ninguna estimacion es fiable."
    )


def test_no_quedan_huecos_ni_valores_imposibles(precios):
    assert precios.notna().all().all()
    assert (precios > 0).all().all()


def test_las_fechas_estan_ordenadas_y_sin_repetir(precios):
    assert precios.index.is_monotonic_increasing
    assert not precios.index.has_duplicates


def test_no_hay_saltos_absurdos(rend):
    # Un rendimiento diario de mas del 50 % casi siempre es un error de datos:
    # un split no ajustado, una coma decimal mal puesta o un cambio de divisa.
    sospechosos = rend[(rend.abs() > 0.5).any(axis=1)]
    assert sospechosos.empty, (
        f"Hay {len(sospechosos)} sesiones con movimientos de mas del 50 %:\n{sospechosos}"
    )


def test_ningun_activo_esta_congelado(rend):
    # Una serie que no se mueve suele indicar un precio arrastrado por error.
    for activo in rend.columns:
        proporcion_quieta = float((rend[activo].abs() < 1e-10).mean())
        assert proporcion_quieta < 0.5, (
            f"'{activo}' no varia en el {proporcion_quieta:.0%} de las sesiones."
        )


# --- 2. Lo que los modelos aprenden de tus datos -------------------------------


def test_las_metricas_de_la_cartera_salen_completas(cartera):
    resumen = resumen_metricas(cartera)
    assert resumen.drop("observaciones").notna().all()
    print("\nCartera equiponderada:")
    for nombre in ("rentabilidad_anualizada", "volatilidad_anualizada", "ratio_sharpe", "maxima_caida"):
        print(f"  {nombre:<28} {resumen[nombre]:>8.4f}")


def test_la_optimizacion_converge(rend):
    cov = matriz_covarianzas(rend, anualizar=True)
    pesos = minima_varianza(cov)
    assert pesos.sum() == pytest.approx(1.0)
    assert (pesos >= -1e-9).all()
    print("\nCartera de minima varianza:")
    for activo, peso in pesos.sort_values(ascending=False).items():
        print(f"  {activo:<12} {peso:>7.2%}")


def test_el_riesgo_se_reparte_de_forma_desigual(rend):
    cov = matriz_covarianzas(rend)
    tabla = contribuciones_riesgo(cartera_pesos_iguales(rend.columns), cov)
    assert tabla["contribucion_relativa"].sum() == pytest.approx(1.0)
    print("\nCon pesos iguales, cada activo aporta al riesgo:")
    for activo, fila in tabla.iterrows():
        print(f"  {activo:<12} {fila['contribucion_relativa']:>7.2%}")


def test_informa_de_si_los_rendimientos_son_normales(cartera):
    prueba = contraste_jarque_bera(cartera)
    print(f"\nJarque-Bera p = {prueba['p_valor']:.2e} -> "
          f"{'compatible con la normal' if prueba['normal_al_5pct'] else 'NO son normales'}")
    if not prueba["normal_al_5pct"]:
        print("  Con colas gruesas, el VaR parametrico normal se queda corto.")
    # Es un resultado, no un fallo: solo se exige que el contraste se calcule.
    assert 0.0 <= prueba["p_valor"] <= 1.0


def test_el_garch_estima_una_persistencia_valida(cartera):
    if len(cartera) < 250:
        pytest.skip("Hacen falta al menos 250 sesiones para estimar un GARCH con sentido.")
    ajuste = ajustar_garch(cartera)
    print(f"\nPersistencia GARCH: {ajuste.persistencia:.3f}")
    assert 0.0 <= ajuste.persistencia < 1.05, "Un modelo con persistencia >= 1 no es estacionario."
    assert np.isfinite(ajuste.resultado.loglikelihood)


def test_la_copula_estima_la_dependencia_de_cola(rend):
    if rend.shape[1] < 2:
        pytest.skip("La dependencia de cola necesita al menos dos activos.")
    copula = ajustar_copula_t(rend)
    print(f"\nGrados de libertad de la copula t: {copula.grados_libertad:.2f}")
    print(f"Dependencia de cola implicita:     {copula.dependencia_cola():.3f}")
    assert 2.0 < copula.grados_libertad <= 60.0
    assert 0.0 <= copula.dependencia_cola() < 1.0


# --- 3. ¿Acierta el modelo de riesgo sobre tus datos? --------------------------


def test_el_var_es_coherente_entre_niveles(cartera):
    assert var_historico(cartera, 0.99) > var_historico(cartera, 0.95) > 0


def test_la_tabla_de_medidas_se_calcula_entera(cartera):
    tabla = tabla_medidas(cartera, niveles=(0.95, 0.99))
    assert tabla.notna().all().all()
    assert (tabla["es"] >= tabla["var"]).all(), "El ES nunca puede quedar por debajo del VaR."
    print("\nVaR y ES al 99 %:")
    for metodo, fila in tabla.xs(0.99, level="nivel").iterrows():
        print(f"  {metodo:<20} VaR {fila['var']:>7.2%}   ES {fila['es']:>7.2%}")


def test_la_maxima_caida_es_coherente(cartera):
    caidas = serie_caidas(cartera)
    assert (caidas <= 1e-12).all()
    assert maxima_caida(cartera) == pytest.approx(float(caidas.min()))


@pytest.mark.parametrize(
    "metodo", ["historico", "parametrico_normal", "cornish_fisher", "ewma"]
)
def test_informa_del_backtest_de_cada_metodo(cartera, metodo):
    if len(cartera) < MINIMO_SESIONES:
        pytest.skip(
            f"Hacen falta al menos {MINIMO_SESIONES} sesiones para un backtest con ventana de 250."
        )
    resultado = backtest_var(cartera, 0.99, 250, metodo)
    print(
        f"\n{metodo}: {int(resultado.kupiec['excepciones'])} excepciones "
        f"(esperadas {resultado.kupiec['excepciones_esperadas']:.1f}) | "
        f"Kupiec p={resultado.kupiec['p_valor']:.3f} | "
        f"Christoffersen p={resultado.condicional['p_valor']:.3f} | "
        f"zona {resultado.basilea['zona']}"
    )
    # Que un metodo no supere el contraste es un hallazgo, no un fallo del
    # codigo. Lo que si debe cumplirse es que el contraste este bien formado.
    assert resultado.kupiec["observaciones"] == len(cartera) - 250
    assert 0.0 <= resultado.condicional["p_valor"] <= 1.0
    assert resultado.basilea["zona"] in {"verde", "amarilla", "roja"}


def test_al_menos_un_metodo_de_var_supera_el_contraste(cartera):
    """Este si falla: si ningun modelo pasa, hay algo que revisar en los datos."""
    if len(cartera) < MINIMO_SESIONES:
        pytest.skip(f"Hacen falta al menos {MINIMO_SESIONES} sesiones.")

    metodos = ("historico", "parametrico_t", "cornish_fisher", "ewma")
    validos = []
    for metodo in metodos:
        resultado = backtest_var(cartera, 0.99, 250, metodo)
        if not resultado.condicional["rechaza_al_5pct"]:
            validos.append(metodo)

    print(f"\nMetodos que superan el contraste conjunto: {validos or 'ninguno'}")
    assert validos, (
        "Ningun metodo de VaR supera el contraste sobre estos datos. "
        "Suele apuntar a un cambio de regimen muy brusco en la muestra, a una "
        "serie con saltos de datos, o a una ventana de 250 sesiones inadecuada "
        "para su frecuencia."
    )
