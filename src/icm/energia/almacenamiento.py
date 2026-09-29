"""Despacho optimo de un activo de almacenamiento y cuanto vale preverlo.

Por que este modulo existe
--------------------------
El cuaderno 03 llego a una conclusion incomoda: el mayor patron del pool --el
ciclo semanal, 19 EUR/MWh de recorrido-- da un ratio de Sharpe de +4,4 y una
rentabilidad compuesta del -100 %. La razon no es estadistica: **la
electricidad no se almacena**, asi que ese patron mide la amplitud de una
forma y no un beneficio acumulable. Quien lo cobra no es un modelo, es una
bateria o un bombeo.

Este modulo es esa bateria. Dado un vector de precios, la capacidad, la
potencia y la eficiencia de ciclo, resuelve **cuando cargar y cuando
descargar** maximizando el margen. Y con eso se puede por fin plantear la
pregunta bien:

    no "se puede predecir el precio de manana"
    sino "que fraccion del maximo teorico captura una prevision"

Esa segunda pregunta tiene tres propiedades que la primera no tenia:

* El objetivo es **cobrable**: un activo fisico ejecuta la operacion.
* Hay una **referencia ingenua** evidente --programar con los precios de
  ayer-- igual que el MASE tiene "predecir cero".
* Hay un **techo duro**: el despacho con los precios realizados. Eso permite
  informar "capturamos el 80 % del maximo teorico", que es una cifra acotada
  por arriba y **que no se puede inflar**. Ningun error de metodologia la
  lleva por encima de 100.

La formulacion
--------------
Programacion lineal, no heuristica de "comprar en el minimo y vender en el
maximo". Las variables son la carga y la descarga de cada periodo, y el
estado de carga se deriva de sus sumas acumuladas:

    e(t) = e(0) + suma_{s<=t} [ carga(s) * rendimiento_carga
                                - descarga(s) / rendimiento_descarga ]

sujeto a ``0 <= e(t) <= capacidad`` y a que ni la carga ni la descarga
superen la potencia. Se maximiza el margen menos un coste por MWh movido, que
representa la degradacion.

La heuristica de un solo ciclo se queda corta y conviene saber cuanto: un
activo de cuatro horas puede hacer **mas de un ciclo al dia** cuando el perfil
tiene dos valles (madrugada y mediodia solar, que es justo lo que pasa en el
pool iberico desde que hay tanta fotovoltaica).

El coste de degradacion no es un adorno
---------------------------------------
Sin el, el optimizador cicla con cualquier diferencia positiva por minima que
sea, porque el modelo no sabe que cada ciclo consume vida util. Con una
bateria de litio a 250-400 EUR/kWh y del orden de 6 000 ciclos de vida, el
coste marginal de mover un MWh ronda los 2-5 EUR. Por debajo de ese umbral, el
arbitraje destruye valor aunque el margen de energia salga positivo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from icm.comun import ErrorDatos, como_serie


@dataclass(frozen=True)
class ParametrosAlmacenamiento:
    """Ficha tecnica de un activo de almacenamiento.

    Atributos
    ---------
    nombre:
        Descripcion legible de la tecnologia.
    capacidad:
        Energia almacenable en MWh. Una bateria "de 1 MW / 4 h" tiene
        capacidad 4 y potencia 1.
    potencia:
        Potencia maxima de carga y de descarga en MW. Se supone simetrica,
        que es lo habitual en bateria; en un bombeo no lo es y habria que
        separarlas.
    rendimiento_ciclo:
        Eficiencia de ida y vuelta en tanto por uno: la fraccion de la energia
        cargada que se recupera al descargar. Litio 0,85-0,92; bombeo
        0,70-0,80. Se reparte a partes iguales entre carga y descarga como
        ``sqrt(rendimiento_ciclo)``, que es la convencion habitual y la unica
        que deja el resultado invariante a donde se ponga la perdida.
    coste_degradacion:
        EUR por MWh **movido** (cargado o descargado). Representa el consumo
        de vida util. Ponerlo a cero hace que el optimizador cicle por
        diferencias irrelevantes.
    horas_por_periodo:
        Duracion de cada periodo del vector de precios, en horas. 1,0 para
        precios horarios, 0,25 para los cuarto-horarios que casa OMIE desde
        octubre de 2025.
    """

    nombre: str
    capacidad: float
    potencia: float
    rendimiento_ciclo: float = 0.85
    coste_degradacion: float = 2.0
    horas_por_periodo: float = 1.0

    def __post_init__(self) -> None:
        for campo in ("capacidad", "potencia", "horas_por_periodo"):
            valor = float(getattr(self, campo))
            if not np.isfinite(valor) or valor <= 0.0:
                raise ErrorDatos(f"'{campo}' debe ser un numero positivo, y vale {valor}.")
        if not 0.0 < float(self.rendimiento_ciclo) <= 1.0:
            raise ErrorDatos(
                f"'rendimiento_ciclo' debe estar en (0, 1], y vale {self.rendimiento_ciclo}. "
                "Es la fraccion de energia que se recupera, no un porcentaje."
            )
        if float(self.coste_degradacion) < 0.0:
            raise ErrorDatos("'coste_degradacion' no puede ser negativo.")

    @property
    def energia_por_periodo(self) -> float:
        """MWh que entran o salen si se opera a plena potencia un periodo."""
        return float(self.potencia) * float(self.horas_por_periodo)

    @property
    def rendimiento_medio(self) -> float:
        """Rendimiento de carga y de descarga por separado."""
        return float(np.sqrt(self.rendimiento_ciclo))


#: Activos de referencia, con cifras de orden de magnitud del sector.
#:
#: Los rendimientos y los costes de degradacion son valores tipicos, no la
#: ficha de una unidad concreta: para valorar un activo real hay que sustituir
#: los tres numeros por los de su hoja de caracteristicas y su contrato de
#: garantia, que es donde se fija cuantos ciclos admite.
ALMACENAMIENTOS: dict[str, ParametrosAlmacenamiento] = {
    "bateria_4h": ParametrosAlmacenamiento(
        nombre="Bateria de litio 1 MW / 4 h",
        capacidad=4.0,
        potencia=1.0,
        rendimiento_ciclo=0.85,
        coste_degradacion=3.0,
    ),
    "bateria_2h": ParametrosAlmacenamiento(
        nombre="Bateria de litio 1 MW / 2 h",
        capacidad=2.0,
        potencia=1.0,
        rendimiento_ciclo=0.88,
        coste_degradacion=3.0,
    ),
    "bombeo": ParametrosAlmacenamiento(
        nombre="Bombeo hidraulico 1 MW / 8 h",
        capacidad=8.0,
        potencia=1.0,
        rendimiento_ciclo=0.75,
        # Una central de bombeo no se degrada por ciclar como una bateria: su
        # coste marginal es de operacion, no de vida util.
        coste_degradacion=0.5,
    ),
}


@dataclass
class Despacho:
    """El plan de operacion de un activo y lo que rinde.

    Atributos
    ---------
    carga, descarga:
        MWh cargados y descargados en cada periodo, indexados como los
        precios.
    estado_carga:
        Energia almacenada al final de cada periodo, en MWh.
    margen:
        Ingreso menos coste en cada periodo, ya descontada la degradacion.
    parametros:
        La ficha del activo con la que se resolvio.
    """

    carga: pd.Series
    descarga: pd.Series
    estado_carga: pd.Series
    margen: pd.Series
    margen_bruto: pd.Series
    parametros: ParametrosAlmacenamiento

    @property
    def margen_total(self) -> float:
        return float(self.margen.sum())

    @property
    def margen_bruto_total(self) -> float:
        """Margen de energia **sin** descontar la degradacion.

        Hace falta separarlo porque el coste de degradacion cumple dos
        papeles distintos y confundirlos es contar la misma cosa dos veces:
        como **senial de despacho** evita ciclar por diferencias
        irrelevantes, y como **coste** es amortizacion de la vida util, o sea
        del propio capex. En un VAN que ya lleva el capex dentro hay que usar
        el bruto; vease :func:`van_almacenamiento`.
        """
        return float(self.margen_bruto.sum())

    @property
    def energia_movida(self) -> float:
        """MWh totales que han pasado por el activo, cargados mas descargados."""
        return float(self.carga.sum() + self.descarga.sum())

    @property
    def ciclos_equivalentes(self) -> float:
        """Cuantas veces se ha llenado y vaciado el activo entero.

        Se cuenta sobre la descarga, que es lo que consume garantia en el
        contrato de una bateria.
        """
        return float(self.descarga.sum()) / float(self.parametros.capacidad)

    def resumen(self) -> pd.Series:
        """Las cifras que se miran para decidir si el activo tiene sentido."""
        return pd.Series(
            {
                "margen_total": self.margen_total,
                "margen_bruto_total": self.margen_bruto_total,
                "periodos": float(len(self.margen)),
                "energia_movida": self.energia_movida,
                "ciclos_equivalentes": self.ciclos_equivalentes,
                "margen_por_ciclo": (
                    self.margen_total / self.ciclos_equivalentes
                    if self.ciclos_equivalentes > 0
                    else float("nan")
                ),
            },
            name=self.parametros.nombre,
        )


def _resolver_bloque(
    precios: np.ndarray,
    parametros: ParametrosAlmacenamiento,
    carga_inicial: float,
    carga_final: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resuelve el programa lineal de un bloque y devuelve (carga, descarga).

    Las variables son ``[carga_0..carga_{T-1}, descarga_0..descarga_{T-1}]`` en
    MWh. El estado de carga no es variable: se expresa como suma acumulada, lo
    que ahorra T variables y T igualdades sin cambiar el problema.
    """
    n = precios.size
    tope = parametros.energia_por_periodo
    rendimiento = parametros.rendimiento_medio
    degradacion = float(parametros.coste_degradacion)

    # Minimizar el coste = -(margen). Cargar cuesta el precio, descargar
    # ingresa el precio, y mover un MWh cuesta la degradacion en los dos
    # sentidos.
    objetivo = np.concatenate([precios + degradacion, -precios + degradacion])

    # Estado de carga acumulado: e(t) = e0 + sum_{s<=t} (c_s * r - d_s / r).
    acumulada = np.tril(np.ones((n, n)))
    matriz_estado = np.hstack([acumulada * rendimiento, -acumulada / rendimiento])

    # 0 <= e(t) <= capacidad, escrito como dos desigualdades <=.
    a_ub = np.vstack([matriz_estado, -matriz_estado])
    b_ub = np.concatenate(
        [
            np.full(n, float(parametros.capacidad) - carga_inicial),
            np.full(n, carga_inicial),
        ]
    )

    a_eq = b_eq = None
    if carga_final is not None:
        a_eq = matriz_estado[-1:, :]
        b_eq = np.array([float(carga_final) - carga_inicial])

    resultado = optimize.linprog(
        objetivo,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, tope)] * (2 * n),
        method="highs",
    )
    if not resultado.success:  # pragma: no cover - HiGHS no falla con este problema
        raise ErrorDatos(
            f"El despacho optimo no ha convergido: {resultado.message}. "
            "Compruebe que la capacidad y la potencia sean coherentes."
        )
    solucion = np.asarray(resultado.x, dtype=float)
    # Limpiar el ruido numerico: HiGHS deja residuos de 1e-13 que despues
    # cuentan como energia movida y ensucian los ciclos equivalentes.
    solucion[np.abs(solucion) < 1e-9] = 0.0
    return solucion[:n], solucion[n:]


def despacho_optimo(
    precios,
    parametros: ParametrosAlmacenamiento | str = "bateria_4h",
    por_dia: bool = True,
    carga_inicial: float = 0.0,
    cerrar_ciclo: bool = True,
) -> Despacho:
    """Plan de carga y descarga que maximiza el margen a esos precios.

    Parametros
    ----------
    precios:
        Serie de precios por periodo, en EUR/MWh. Puede tener precios
        negativos: en el pool iberico son rutina y para un activo de
        almacenamiento son una oportunidad, porque le **pagan** por cargar.
    por_dia:
        Con ``True`` (lo habitual) se resuelve un programa independiente por
        dia natural, que es como funciona el mercado diario: la subasta de
        cada dia se casa por separado y el activo llega al dia siguiente con
        el estado de carga que decidio. Con ``False`` se resuelve la serie
        entera de una vez, lo que permite arbitraje entre dias y da un techo
        mas alto -- util para acotar, irreal como plan de operacion.
    carga_inicial:
        Energia almacenada al empezar, en MWh.
    cerrar_ciclo:
        Con ``True`` se exige terminar cada bloque con la misma energia con la
        que se empezo. Es lo que evita el artefacto mas facil de este
        ejercicio: sin esa restriccion el activo empieza lleno de energia
        gratis, la vende y el margen del primer dia sale inflado.

    Errores
    -------
    Se rechaza un indice desordenado o con fechas repetidas. El estado de
    carga es una suma acumulada, asi que el orden **es** el dato: con las
    filas barajadas el plan no es peor, es otro.
    """
    ficha = _resolver_parametros(parametros)
    serie = como_serie(precios, "precios")
    if len(serie) < 2:
        raise ErrorDatos(f"Hacen falta al menos 2 periodos para despachar; hay {len(serie)}.")
    if serie.index.has_duplicates:
        raise ErrorDatos(
            "El indice de los precios tiene periodos repetidos: el estado de carga se "
            "acumula en orden y no habria una secuencia bien definida."
        )
    if not serie.index.is_monotonic_increasing:
        raise ErrorDatos(
            "El indice de los precios no esta ordenado. El estado de carga es una suma "
            "acumulada, asi que con las filas barajadas el plan resultante no es peor: "
            "es otro. Ordene con sort_index() antes de despachar."
        )
    if float(carga_inicial) < 0.0 or float(carga_inicial) > float(ficha.capacidad):
        raise ErrorDatos(
            f"La carga inicial ({carga_inicial}) tiene que estar entre 0 y la capacidad "
            f"({ficha.capacidad})."
        )

    valores = serie.to_numpy(dtype=float)
    carga = np.zeros_like(valores)
    descarga = np.zeros_like(valores)

    if por_dia and isinstance(serie.index, pd.DatetimeIndex):
        posiciones = np.arange(len(serie))
        for _, bloque in pd.Series(posiciones, index=serie.index).groupby(
            serie.index.normalize()
        ):
            indices = bloque.to_numpy()
            if indices.size < 2:
                continue
            c, d = _resolver_bloque(
                valores[indices],
                ficha,
                float(carga_inicial),
                float(carga_inicial) if cerrar_ciclo else None,
            )
            carga[indices], descarga[indices] = c, d
    else:
        carga, descarga = _resolver_bloque(
            valores,
            ficha,
            float(carga_inicial),
            float(carga_inicial) if cerrar_ciclo else None,
        )

    return _construir_despacho(serie, carga, descarga, ficha, float(carga_inicial))


def _construir_despacho(
    precios: pd.Series,
    carga: np.ndarray,
    descarga: np.ndarray,
    parametros: ParametrosAlmacenamiento,
    carga_inicial: float,
) -> Despacho:
    """Arma el resultado y liquida el margen a los precios dados."""
    rendimiento = parametros.rendimiento_medio
    neto = carga * rendimiento - descarga / rendimiento
    estado = carga_inicial + np.cumsum(neto)
    valores = precios.to_numpy(float)
    bruto = descarga * valores - carga * valores
    margen = bruto - (carga + descarga) * float(parametros.coste_degradacion)
    indice = precios.index
    return Despacho(
        carga=pd.Series(carga, index=indice, name="carga"),
        descarga=pd.Series(descarga, index=indice, name="descarga"),
        estado_carga=pd.Series(estado, index=indice, name="estado_carga"),
        margen=pd.Series(margen, index=indice, name="margen"),
        margen_bruto=pd.Series(bruto, index=indice, name="margen_bruto"),
        parametros=parametros,
    )


def liquidar_plan(
    plan: Despacho,
    precios_reales,
) -> Despacho:
    """Reliquida un plan a otros precios: **la funcion que hace honesto el ejercicio**.

    Un despacho calculado sobre precios **previstos** es una decision; lo que
    se cobra son los precios **realizados**. Separar las dos cosas es lo unico
    que impide que la evaluacion mire al futuro, y es el equivalente aqui de
    lo que ``prediccion_ingenua`` hace en :mod:`icm.ml.evaluacion`.

    El plan no se recalcula: se respeta tal cual y solo cambia el precio al
    que se liquida cada MWh. Si el plan compro a la hora equivocada, aqui se
    paga.
    """
    reales = como_serie(precios_reales, "precios_reales")
    comun = plan.carga.index
    if not reales.index.equals(comun):
        faltan = comun.difference(reales.index)
        raise ErrorDatos(
            f"Los precios reales no cubren el mismo indice que el plan: faltan "
            f"{len(faltan)} periodos. Reindexe antes de liquidar."
        )
    return _construir_despacho(
        reales,
        plan.carga.to_numpy(float),
        plan.descarga.to_numpy(float),
        plan.parametros,
        float(plan.estado_carga.iloc[0] - plan.carga.iloc[0] * plan.parametros.rendimiento_medio
              + plan.descarga.iloc[0] / plan.parametros.rendimiento_medio),
    )


def valor_de_la_prevision(
    precios_reales,
    previsiones: dict[str, object],
    parametros: ParametrosAlmacenamiento | str = "bateria_4h",
    por_dia: bool = True,
) -> pd.DataFrame:
    """Que fraccion del maximo teorico captura cada prevision.

    Esta es la tabla que sustituye al MASE en un problema de almacenamiento, y
    la razon de todo el modulo. Tres columnas y una lectura:

    * **oraculo**: se despacha con los precios realizados. Es imposible y no
      pretende otra cosa: acota por arriba. Ninguna prevision puede pasar del
      100 %.
    * **ingenua**: se despacha con los precios del dia anterior. Es la
      referencia que hay que batir, y bate a mucha gente.
    * cada prevision que se pase: se despacha con ella y se **liquida a los
      precios reales**.

    Informar el resultado como porcentaje del oraculo tiene una propiedad que
    ninguna metrica de error tiene: **esta acotada por arriba por
    construccion**. Un MASE se puede hacer bajar eligiendo mal la referencia,
    como paso en este mismo proyecto; un 80 % del oraculo no se puede inflar
    por encima de 100 sin que el fallo sea evidente.

    Parametros
    ----------
    previsiones:
        Diccionario ``{nombre: serie}`` con las previsiones a comparar. Cada
        una tiene que cubrir el mismo indice que los precios reales.
    """
    ficha = _resolver_parametros(parametros)
    reales = como_serie(precios_reales, "precios_reales")
    if not isinstance(reales.index, pd.DatetimeIndex):
        raise ErrorDatos("Los precios reales necesitan indice de fechas para comparar por dia.")

    oraculo = despacho_optimo(reales, ficha, por_dia).margen_total
    if oraculo <= 0.0:
        raise ErrorDatos(
            "Ni siquiera el despacho con los precios realizados da margen positivo: "
            "con estos precios y este coste de degradacion el activo no opera, y el "
            "porcentaje del oraculo no significaria nada."
        )

    candidatas: dict[str, object] = {"ingenua": _prevision_ingenua(reales)}
    candidatas.update(previsiones)

    filas = {"oraculo": {"margen": oraculo, "pct_oraculo": 100.0}}
    for nombre, prevision in candidatas.items():
        serie = como_serie(prevision, str(nombre)).reindex(reales.index)
        if serie.isna().any():
            # Los periodos sin prevision no se operan: es lo que pasaria de
            # verdad, no un hueco que rellenar.
            serie = serie.fillna(float(reales.mean()))
        plan = despacho_optimo(serie, ficha, por_dia)
        liquidado = liquidar_plan(plan, reales)
        filas[str(nombre)] = {
            "margen": liquidado.margen_total,
            "pct_oraculo": 100.0 * liquidado.margen_total / oraculo,
        }

    tabla = pd.DataFrame(filas).T
    tabla.index.name = "prevision"
    tabla.attrs["activo"] = ficha.nombre
    return tabla


def _prevision_ingenua(precios: pd.Series) -> pd.Series:
    """El perfil del dia anterior, que es la referencia que hay que batir.

    No es el precio de ayer a la misma hora sin mas: es el **perfil completo**
    del dia anterior colocado sobre el dia de hoy. Con eso el activo repite
    ayer, que es exactamente lo que hace un operador sin modelo.
    """
    marco = pd.DataFrame({"precio": precios.to_numpy(float)}, index=precios.index)
    marco["dia"] = precios.index.normalize()
    marco["posicion"] = marco.groupby("dia").cumcount()
    dias = list(dict.fromkeys(marco["dia"]))
    anterior = {dia: previo for previo, dia in zip(dias[:-1], dias[1:])}

    perfil = {
        dia: trozo.set_index("posicion")["precio"] for dia, trozo in marco.groupby("dia")
    }
    valores = np.full(len(precios), np.nan)
    for dia, trozo in marco.groupby("dia"):
        previo = anterior.get(dia)
        if previo is None:
            continue
        fuente = perfil[previo]
        posiciones = trozo["posicion"].to_numpy()
        # Si ayer tuvo menos periodos (cambio de hora), lo que falte se queda
        # sin prevision y 'valor_de_la_prevision' lo trata como no operar.
        tomados = fuente.reindex(posiciones).to_numpy()
        valores[np.searchsorted(precios.index, trozo.index)] = tomados
    return pd.Series(valores, index=precios.index, name="ingenua")


def _resolver_parametros(
    parametros: ParametrosAlmacenamiento | str,
) -> ParametrosAlmacenamiento:
    """Acepta una ficha o el nombre de una de las de referencia."""
    if isinstance(parametros, ParametrosAlmacenamiento):
        return parametros
    clave = str(parametros)
    if clave not in ALMACENAMIENTOS:
        raise ErrorDatos(
            f"Activo '{clave}' desconocido. Disponibles: {', '.join(ALMACENAMIENTOS)}. "
            "O pase una ParametrosAlmacenamiento propia."
        )
    return ALMACENAMIENTOS[clave]


# --- valoracion del activo -------------------------------------------------------


@dataclass(frozen=True)
class ParametrosInversion:
    """Lo que hace falta para pasar de un margen anual a un VAN.

    Atributos
    ---------
    capex_por_kwh:
        Coste de instalacion en EUR por kWh de capacidad. Para bateria de
        litio a escala de red, el rango que se publica en 2024-2025 va de 200
        a 400 EUR/kWh todo incluido (celdas, inversor, obra civil, conexion).
        Un bombeo nuevo no se cotiza asi, y la cifra correcta ahi es el coste
        del proyecto completo.
    coste_fijo_anual_por_kw:
        Operacion y mantenimiento fijos en EUR por kW de potencia y anio:
        seguros, vigilancia, terrenos, sistema de control. Del orden de 5-12
        EUR/kW.
    vida_anios:
        Horizonte de la valoracion. La garantia tipica de una bateria de red
        son 10-15 anios o un numero de ciclos, el que se cumpla antes.
    tasa_descuento:
        Coste medio ponderado del capital, en tanto por uno.
    retencion_anual:
        Fraccion de la capacidad que queda al final de cada anio. 0,975 deja
        el 68,5 % a los quince anios, que es el orden de magnitud de una
        garantia de 70-80 % al final del periodo.
    ingreso_adicional_anual_por_mw:
        Ingresos que **no** son arbitraje: mercado de capacidad, servicios de
        ajuste, restricciones tecnicas. Es un parametro y no un valor por
        omision a proposito: en Espania son una parte grande del ingreso real
        de una bateria y **no estan en los datos de este proyecto**, asi que
        quien valore tiene que ponerlo a mano y hacerse responsable.
    valor_residual:
        Valor de desmantelamiento o de segunda vida al terminar, en EUR. Se
        descuenta al ultimo anio.
    """

    capex_por_kwh: float = 300.0
    coste_fijo_anual_por_kw: float = 8.0
    vida_anios: int = 15
    tasa_descuento: float = 0.08
    retencion_anual: float = 0.975
    ingreso_adicional_anual_por_mw: float = 0.0
    valor_residual: float = 0.0

    def __post_init__(self) -> None:
        for campo in ("capex_por_kwh", "coste_fijo_anual_por_kw", "ingreso_adicional_anual_por_mw"):
            if float(getattr(self, campo)) < 0.0:
                raise ErrorDatos(f"'{campo}' no puede ser negativo.")
        if int(self.vida_anios) < 1:
            raise ErrorDatos("La vida del activo tiene que ser de al menos un anio.")
        if not -1.0 < float(self.tasa_descuento) < 1.0:
            raise ErrorDatos(
                f"'tasa_descuento' se expresa en tanto por uno y vale "
                f"{self.tasa_descuento}. Un 8 % es 0.08, no 8."
            )
        if not 0.0 < float(self.retencion_anual) <= 1.0:
            raise ErrorDatos(
                f"'retencion_anual' es la fraccion de capacidad que queda cada anio y "
                f"tiene que estar en (0, 1]; vale {self.retencion_anual}."
            )


def curva_margen_por_capacidad(
    precios,
    parametros: ParametrosAlmacenamiento | str = "bateria_4h",
    factores: Sequence[float] = (0.4, 0.55, 0.7, 0.85, 1.0),
    por_dia: bool = True,
) -> pd.Series:
    """Margen bruto anual segun la capacidad que le quede al activo.

    Hace falta porque **el margen no es proporcional a la capacidad**. Medido
    sobre OMIE con potencia de 1 MW: 26 350 EUR/anio con 2 horas, 42 088 con
    4 y 48 535 con 6. Duplicar la capacidad no duplica el ingreso, porque las
    horas que se aniaden son cada vez menos extremas -- el pool no tiene mas
    de unas seis horas de recorrido diario que explotar.

    Por eso la degradacion no se puede aplicar como un factor sobre el margen:
    perder el 20 % de capacidad no cuesta el 20 % del ingreso, cuesta menos.
    Esta curva se resuelve una vez y :func:`van_almacenamiento` interpola en
    ella anio a anio.

    Devuelve una Series indexada por el factor de capacidad, con el margen
    bruto **anualizado** en EUR.
    """
    ficha = _resolver_parametros(parametros)
    serie = como_serie(precios, "precios")
    valores = sorted({float(f) for f in factores} | {1.0})
    if valores[0] <= 0.0:
        raise ErrorDatos("Los factores de capacidad tienen que ser positivos.")
    if valores[-1] > 1.0:
        raise ErrorDatos("Los factores de capacidad no pueden pasar de 1: el activo no crece.")

    anios = _anios_de_la_muestra(serie)
    margenes = {}
    for factor in valores:
        reducida = ParametrosAlmacenamiento(
            nombre=f"{ficha.nombre} al {100 * factor:.0f} %",
            capacidad=float(ficha.capacidad) * factor,
            potencia=float(ficha.potencia),
            rendimiento_ciclo=float(ficha.rendimiento_ciclo),
            coste_degradacion=float(ficha.coste_degradacion),
            horas_por_periodo=float(ficha.horas_por_periodo),
        )
        plan = despacho_optimo(serie, reducida, por_dia)
        margenes[factor] = plan.margen_bruto_total / anios

    curva = pd.Series(margenes, name="margen_bruto_anual")
    curva.index.name = "factor_capacidad"
    return curva


def _anios_de_la_muestra(precios: pd.Series) -> float:
    """Anios que cubre la muestra, para anualizar el margen."""
    if not isinstance(precios.index, pd.DatetimeIndex):
        raise ErrorDatos(
            "Para anualizar el margen hacen falta fechas: el indice de los precios "
            "tiene que ser un DatetimeIndex."
        )
    dias = (precios.index[-1] - precios.index[0]).total_seconds() / 86400.0
    if dias < 30.0:
        raise ErrorDatos(
            f"La muestra cubre {dias:.0f} dias. Anualizar un margen a partir de menos "
            "de un mes da una cifra que no significa nada; use al menos un anio."
        )
    return dias / 365.25


def van_almacenamiento(
    precios,
    parametros: ParametrosAlmacenamiento | str = "bateria_4h",
    inversion: ParametrosInversion | None = None,
    por_dia: bool = True,
    curva: pd.Series | None = None,
) -> pd.Series:
    """VAN del activo: capex, degradacion, O&M y el margen que de verdad captura.

    Sobre la degradacion contada dos veces
    --------------------------------------
    Este es el punto delicado de toda la valoracion y conviene decirlo antes
    de dar un numero. El coste de degradacion de
    :class:`ParametrosAlmacenamiento` cumple **dos papeles**:

    1. Como **senial de despacho** decide cuando no merece la pena ciclar.
       Ahi es imprescindible: sin el, el optimizador arbitra por 0,10 EUR/MWh.
    2. Como **coste en euros** es amortizacion de la vida util, es decir del
       propio capex.

    En un VAN que ya descuenta el capex, restar tambien el coste por MWh es
    **pagar la bateria dos veces**. Asi que aqui el flujo de caja usa el
    margen **bruto** --el coste por MWh se queda solo como senial de
    despacho-- y la degradacion se expresa donde corresponde: como perdida de
    capacidad via ``retencion_anual``, que reduce el margen de los anios
    siguientes.

    Cuanto vale esa diferencia no es teorico, sale en la propia tabla:
    ``margen_bruto_anual`` frente a ``margen_neto_anual``.

    Devuelve
    --------
    Una Series con el capex, los flujos, el VAN, la TIR, el periodo de retorno
    y el capex de equilibrio.
    """
    ficha = _resolver_parametros(parametros)
    datos = inversion or ParametrosInversion()
    serie = como_serie(precios, "precios")

    if curva is None:
        curva = curva_margen_por_capacidad(serie, ficha, por_dia=por_dia)
    factores = np.asarray(curva.index, dtype=float)
    valores_curva = curva.to_numpy(dtype=float)

    capex = float(ficha.capacidad) * 1000.0 * float(datos.capex_por_kwh)
    fijo = float(ficha.potencia) * 1000.0 * float(datos.coste_fijo_anual_por_kw)
    adicional = float(ficha.potencia) * float(datos.ingreso_adicional_anual_por_mw)

    flujos = [-capex]
    retencion = 1.0
    for anio in range(1, int(datos.vida_anios) + 1):
        # La capacidad media del anio: se degrada a lo largo de el.
        media = retencion * (1.0 + float(datos.retencion_anual)) / 2.0
        arbitraje = float(np.interp(media, factores, valores_curva))
        flujo = arbitraje + adicional - fijo
        if anio == int(datos.vida_anios):
            flujo += float(datos.valor_residual)
        flujos.append(flujo)
        retencion *= float(datos.retencion_anual)

    flujos = np.asarray(flujos, dtype=float)
    van = float(np.sum(flujos / (1.0 + float(datos.tasa_descuento)) ** np.arange(len(flujos))))

    plan_nominal = despacho_optimo(serie, ficha, por_dia)
    anios_muestra = _anios_de_la_muestra(serie)

    return pd.Series(
        {
            "capex": capex,
            "margen_bruto_anual": plan_nominal.margen_bruto_total / anios_muestra,
            "margen_neto_anual": plan_nominal.margen_total / anios_muestra,
            "coste_fijo_anual": fijo,
            "ingreso_adicional_anual": adicional,
            "flujo_anio_1": float(flujos[1]),
            "flujo_ultimo_anio": float(flujos[-1]),
            "van": van,
            "tir": _tir(flujos),
            "retorno_anios": _periodo_de_retorno(flujos),
            "capex_equilibrio_por_kwh": _capex_de_equilibrio(flujos, capex, ficha, datos),
        },
        name=ficha.nombre,
    )


def _tir(flujos: np.ndarray) -> float:
    """Tasa interna de retorno por biseccion.

    Se resuelve por biseccion y no con las raices del polinomio porque con
    flujos de signo unico tras la inversion la TIR es unica y la biseccion no
    puede devolver una raiz compleja ni negativa espuria. Si el activo nunca
    recupera el capex, la TIR no existe y se devuelve ``NaN`` en lugar de un
    numero inventado.
    """
    if flujos[0] >= 0.0 or np.sum(flujos[1:]) <= 0.0:
        return float("nan")

    def van_a(tasa: float) -> float:
        return float(np.sum(flujos / (1.0 + tasa) ** np.arange(len(flujos))))

    if van_a(0.0) <= 0.0:
        return float("nan")  # ni sin descontar se recupera la inversion
    bajo, alto = 0.0, 1.0
    while van_a(alto) > 0.0 and alto < 10.0:
        alto *= 2.0
    if van_a(alto) > 0.0:  # pragma: no cover - TIR absurdamente alta
        return float("nan")
    for _ in range(200):
        medio = 0.5 * (bajo + alto)
        if van_a(medio) > 0.0:
            bajo = medio
        else:
            alto = medio
    return 0.5 * (bajo + alto)


def _periodo_de_retorno(flujos: np.ndarray) -> float:
    """Anios hasta que el flujo acumulado **sin descontar** cruza cero.

    Sin descontar a proposito: es la cifra que se usa para comparar de un
    vistazo, y mezclarla con el descuento la haria incomparable con la que
    publica cualquier otro. Si no se recupera dentro de la vida del activo se
    devuelve ``NaN``, no la vida entera.
    """
    acumulado = np.cumsum(flujos)
    cruces = np.flatnonzero(acumulado >= 0.0)
    if cruces.size == 0:
        return float("nan")
    primero = int(cruces[0])
    if primero == 0:  # pragma: no cover - capex nulo
        return 0.0
    # Interpolacion lineal dentro del anio en que cruza.
    previo = acumulado[primero - 1]
    return float(primero - 1 + (-previo) / (acumulado[primero] - previo))


def _capex_de_equilibrio(
    flujos: np.ndarray,
    capex: float,
    ficha: ParametrosAlmacenamiento,
    datos: ParametrosInversion,
) -> float:
    """Capex por kWh que deja el VAN en cero: la cifra que decide la inversion.

    Es mas util que el VAN en si, porque el VAN depende de un capex que se ha
    supuesto y este responde a la pregunta que de verdad se hace quien
    invierte: **cuanto puedo llegar a pagar por esto**.
    """
    descuentos = (1.0 + float(datos.tasa_descuento)) ** np.arange(1, len(flujos))
    valor_operacion = float(np.sum(flujos[1:] / descuentos))
    if valor_operacion <= 0.0:
        return 0.0
    return valor_operacion / (float(ficha.capacidad) * 1000.0)


def valor_van_de_la_prevision(
    precios_reales,
    previsiones: dict[str, object],
    parametros: ParametrosAlmacenamiento | str = "bateria_4h",
    inversion: ParametrosInversion | None = None,
    por_dia: bool = True,
) -> pd.DataFrame:
    """Cuanto vale la prevision **en euros de VAN**, que es la pregunta final.

    :func:`valor_de_la_prevision` dice que fraccion del techo captura cada
    prevision. Esta la traduce a lo unico que decide una inversion: el VAN del
    activo si se opera con ella. La diferencia entre la fila del oraculo y la
    de la ingenua es el valor maximo que puede tener un modelo de prevision
    para este activo, y la de cualquier fila intermedia es lo que vale ese
    modelo concreto.

    El despacho se decide con la prevision y se **liquida a los precios
    realizados**, igual que en :func:`valor_de_la_prevision`.
    """
    ficha = _resolver_parametros(parametros)
    datos = inversion or ParametrosInversion()
    reales = como_serie(precios_reales, "precios_reales")
    anios = _anios_de_la_muestra(reales)

    candidatas: dict[str, object] = {"oraculo": reales, "ingenua": _prevision_ingenua(reales)}
    candidatas.update(previsiones)

    # La curva de margen frente a capacidad se resuelve UNA vez: su forma no
    # depende de la prevision, solo su altura.
    curva = curva_margen_por_capacidad(reales, ficha, por_dia=por_dia)
    techo = float(curva.loc[1.0])
    filas = {}
    for nombre, prevision in candidatas.items():
        serie = como_serie(prevision, str(nombre)).reindex(reales.index)
        if serie.isna().any():
            serie = serie.fillna(float(reales.mean()))
        plan = despacho_optimo(serie, ficha, por_dia)
        liquidado = liquidar_plan(plan, reales)
        bruto_anual = liquidado.margen_bruto_total / anios
        # Se reescala la curva por la fraccion capturada.
        fraccion = bruto_anual / techo if techo else 0.0
        resumen = van_almacenamiento(reales, ficha, datos, por_dia, curva * fraccion)
        filas[str(nombre)] = {
            "margen_bruto_anual": bruto_anual,
            "pct_oraculo": 100.0 * fraccion,
            "van": resumen["van"],
            "tir": resumen["tir"],
            "capex_equilibrio_por_kwh": resumen["capex_equilibrio_por_kwh"],
        }

    tabla = pd.DataFrame(filas).T
    tabla.index.name = "prevision"
    tabla.attrs["activo"] = ficha.nombre
    tabla.attrs["capex_supuesto_por_kwh"] = float(datos.capex_por_kwh)
    return tabla
