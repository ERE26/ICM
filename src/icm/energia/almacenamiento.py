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
    parametros: ParametrosAlmacenamiento

    @property
    def margen_total(self) -> float:
        return float(self.margen.sum())

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
    margen = (
        descarga * precios.to_numpy(float)
        - carga * precios.to_numpy(float)
        - (carga + descarga) * float(parametros.coste_degradacion)
    )
    indice = precios.index
    return Despacho(
        carga=pd.Series(carga, index=indice, name="carga"),
        descarga=pd.Series(descarga, index=indice, name="descarga"),
        estado_carga=pd.Series(estado, index=indice, name="estado_carga"),
        margen=pd.Series(margen, index=indice, name="margen"),
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
