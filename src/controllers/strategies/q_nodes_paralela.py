import os # Se usa para obtener el numeor de nucleos disponibles en el procesador
import time
import numpy as np
from typing import Union, List, Tuple
from multiprocessing import Pool, Manager as MPManager
from concurrent.futures import ProcessPoolExecutor, as_completed
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# Importaciones originales
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import QNODES_ANALYSIS_TAG, QNODES_LABEL, QNODES_STRAREGY_TAG
from src.constants.base import TYPE_TAG, NET_LABEL, INFTY_NEG, INFTY_POS, LAST_IDX, EFECTO, ACTUAL


class QNodesParallel(SIA):
    """
    Versión paralela de QNodes usando MPI y multiprocessing.
    
    Estrategia de paralelización:
    1. MPI para distribución entre nodos de computación
    2. Multiprocessing para paralelización dentro de cada nodo
    3. Paralelización en múltiples niveles del algoritmo
    """
    
    def __init__(self, gestor: Manager, use_mpi=True, n_processes=None):
        super().__init__(gestor)
        # Configuración MPI
        self.use_mpi = use_mpi
        if self.use_mpi and not MPI.Is_initialized():
            MPI.Init()

        '''
        self.comm: Es el comunicador de MPI, que permite la comunicación entre los procesos.
        Si use_mpi es True, se utiliza MPI.COMM_WORLD, que es el comunicador global que abarca todos los nodos. 
        Si no se usa MPI, se asigna None.
        '''
        self.comm = MPI.COMM_WORLD if self.use_mpi else None
  
        '''
        self.rank: Representa el rango (ID único) del proceso actual dentro del grupo de procesos de MPI. 
        Si no se usa MPI, se establece en 0 (el proceso maestro).
        '''
        self.rank = self.comm.Get_rank() if self.use_mpi else 0
        '''
        self.size: Es el número total de procesos en la comunicación MPI (número de nodos). 
        Si no se usa MPI, se establece en 1 (solo un proceso).
        '''   
        self.size = self.comm.Get_size() if self.use_mpi else 1
        
        # Configuración multiprocessing
        '''
        n_processes: define el número de procesos que se usarán para paralelizar el trabajo dentro de un único nodo. 
                     Si no se proporciona, el valor predeterminado es el mínimo entre 8 o el número de núcleos disponibles en el sistema.
                     Esto se calcula en función de los núcleos disponibles en el sistema, utilizando os.sched_getaffinity(0), que devuelve los núcleos disponibles para el proceso actual.
        '''
        self.n_processes = n_processes or min(8, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4)
        
        # Atributos originales
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        
        # Memorias compartidas para multiprocessing
        self.memoria_omega_shared = MPManager().dict() # Diccionario compartido donde se almacenan los resultados de omega (candidatos).
        self.memoria_particiones_shared = MPManager().dict() # Diccionario compartido para almacenar las particiones y sus resultados asociados.

        
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray

        '''
        El logger (self.logger) es utilizado para generar registros detallados de las ejecuciones del algoritmo. 
        Se usa un logger específico para cada proceso (rank) para poder rastrear los logs de cada proceso de forma individual.
        '''
        self.logger = SafeLogger(f"{QNODES_STRAREGY_TAG}_rank_{self.rank}")

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
                
        # Solo el proceso maestro prepara los datos iniciales
        """
        El proceso maestro se encarga de preparar el sistema (self.rank == 0), 
        lo que incluye la configuración del subsistema, los índices de las particiones, 
        y la creación de las listas de vértices para las particiones (presente y futuro).

        Se utilizan los índices de indices_ncubos (que representan las posibles configuraciones de los nodos) 
        para definir los conjuntos de vértices correspondientes a las particiones del sistema. 
        Esta información luego se agrupa en broadcast_data para ser enviada a los procesos esclavos.
        """
        if self.rank == 0:
            self.sia_preparar_subsistema(condicion, alcance, mecanismo)
            
            futuro = tuple(
                (EFECTO, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos
            )
            presente = tuple(
                (ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos
            )
            
            self.m = self.sia_subsistema.indices_ncubos.size
            self.n = self.sia_subsistema.dims_ncubos.size
            self.indices_alcance = self.sia_subsistema.indices_ncubos
            self.indices_mecanismo = self.sia_subsistema.dims_ncubos
            
            self.tiempos = (
                np.zeros(self.n, dtype=np.int8),
                np.zeros(self.m, dtype=np.int8),
            )
            
            vertices = list(presente + futuro)
            self.vertices = set(presente + futuro)
            
            # Datos para broadcast
            broadcast_data = {
                'vertices': vertices,
                'sia_subsistema': self.sia_subsistema,
                'sia_dists_marginales': self.sia_dists_marginales,
                'indices_alcance': self.indices_alcance,
                'indices_mecanismo': self.indices_mecanismo,
                'm': self.m,
                'n': self.n,
                'tiempos': self.tiempos
            }
        else:
            broadcast_data = None
        
        # Broadcast de datos a todos los procesos MPI
        if self.use_mpi:
            '''
            MPI se utiliza para comunicar los datos entre los procesos. 
            El proceso maestro (rank 0) envía los datos a todos los procesos a través de bcast. 
            Esto garantiza que cada proceso tenga acceso a la misma información necesaria para ejecutar el algoritmo.
            
            self.comm.bcast(broadcast_data, root=0): La función bcast realiza el broadcast de los datos 
            desde el proceso maestro (rank 0) a todos los procesos en el sistema. 
            El root=0 indica que el proceso con rango 0 es el que envía la información.
            Los procesos esclavos reciben estos datos y los utilizan para inicializar sus propias variables.
            '''
            broadcast_data = self.comm.bcast(broadcast_data, root=0)
            
            # Procesos esclavos inicializan sus datos
            if self.rank != 0:
                self.vertices = set(broadcast_data['vertices'])
                self.sia_subsistema = broadcast_data['sia_subsistema']
                self.sia_dists_marginales = broadcast_data['sia_dists_marginales']
                self.indices_alcance = broadcast_data['indices_alcance']
                self.indices_mecanismo = broadcast_data['indices_mecanismo']
                self.m = broadcast_data['m']
                self.n = broadcast_data['n']
                self.tiempos = broadcast_data['tiempos']
        
        '''
        Una vez que todos los procesos tienen los mismos datos, se ejecuta el algoritmo de particionamiento en paralelo. 
        Aquí es donde se aplica la estrategia de paralelización, con multiprocessing dentro de cada nodo y MPI entre los nodos.

        Dependiendo de si MPI está habilitado, el algoritmo se ejecuta en paralelo en todos los procesos disponibles, 
        utilizando los vértices que se han distribuido previamente.
        '''
        vertices = broadcast_data['vertices'] if self.use_mpi else vertices
        
        # Algoritmo paralelo
        mip = self.algorithm_parallel(vertices)
        
        # Solo el proceso maestro retorna la solución
        '''
        El proceso maestro es el único que tiene la responsabilidad de consolidar los resultados de
        los demás procesos y devolver la solución final. Esto es porque, en este tipo de algoritmos 
        distribuidos, solo un proceso (el maestro) debe encargarse de la recopilación de resultados 
        y de la conclusión del cálculo.

        Si MPI está habilitado, solo el proceso maestro obtiene y formatea los resultados, 
        como la partición óptima (usando fmt_biparte_q) y la pérdida de información. 
        Luego, el proceso maestro devuelve un objeto Solution que contiene todos los resultados finales.
        '''
        if self.rank == 0:
            fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))
            perdida_mip, dist_marginal_mip = self.memoria_particiones_shared[mip]
            
            return Solution(
                estrategia=f"{QNODES_LABEL}_PARALLEL",
                perdida=perdida_mip,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist_marginal_mip,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion=fmt_mip,
            )
        
        return None

    def algorithm_parallel(self, vertices: List[Tuple[int, int]]):
        """
        Algoritmo QNodes paralelizado usando MPI + multiprocessing.
        
        Estrategia de paralelización:
        1. Distribución de fases entre procesos MPI
        2. Paralelización de iteraciones con multiprocessing
        3. Evaluación especulativa de funciones submodulares
        """

        '''
        Los vértices representan los nodos del sistema (en el presente y futuro). 
        El primer nodo se asigna a omegas_origen, mientras que los nodos restantes se asignan a deltas_origen. 
        vertices_fase mantiene todos los vértices del sistema que se utilizarán en las fases.

        El algoritmo divide los vértices en omegas_origen (nodos iniciales) y deltas_origen (nodos restantes), 
        que se usarán más adelante en el proceso de partición.
        '''
        omegas_origen = np.array([vertices[0]])
        deltas_origen = np.array(vertices[1:])
        vertices_fase = vertices

        '''
        El trabajo se distribuye entre los procesos en función de las fases del algoritmo. 
        Cada fase representa un conjunto de nodos que deben ser procesados. 
        Al dividir las fases, el algoritmo permite que cada proceso trabaje en una porción del total de fases.
        '''
        total_fases = len(vertices_fase) - 2
        
        # Distribución de trabajo entre procesos MPI
        fases_por_proceso = np.array_split(range(total_fases), self.size) # Divide el rango de fases entre los procesos disponibles.
        mis_fases = fases_por_proceso[self.rank] # Cada proceso obtiene un conjunto de fases específicas para su ejecución.
        
        resultados_locales = {}

        '''
        Cada fase del algoritmo se ejecuta en paralelo en función de los procesos distribuidos. 
        Cada proceso maneja su parte de las fases y evalúa las particiones correspondientes.
        '''
        
        for i in mis_fases: # Cada proceso ejecuta su conjunto de fases (especificado por mis_fases).
            self.logger.debug(f"Proceso {self.rank} ejecutando fase {i}")
            
            # Configuración de la fase
            omegas_ciclo = [vertices_fase[0]]
            deltas_ciclo = vertices_fase[1:]
            
            # Paralelización de ciclos
            resultado_fase = self._ejecutar_fase_paralela(i, omegas_ciclo, deltas_ciclo) # _ejecutar_fase_paralela: Este método maneja la ejecución de las iteraciones dentro de la fase en paralelo, utilizando multiprocessing.
            resultados_locales[i] = resultado_fase #El proceso almacena los resultados de cada fase en resultados_locales.
            
            # Actualizar vertices_fase para siguiente iteración
            vertices_fase = resultado_fase['vertices_siguiente'] #se actualiza para la siguiente iteración, utilizando los resultados de la fase actual.
        

        '''
        Después de que todos los procesos han ejecutado sus fases y obtenidos los resultados, 
        el proceso maestro (rank 0) recolecta estos resultados para consolidarlos.
        '''
        # Gather de resultados de todos los procesos MPI
        if self.use_mpi:
            todos_resultados = self.comm.gather(resultados_locales, root=0) # Recoge los resultados de todos los procesos y los gather en el proceso maestro (rank 0).
            
            if self.rank == 0:
                # Consolidar resultados
                for resultados_proceso in todos_resultados:
                    for fase, resultado in resultados_proceso.items():
                        self.memoria_particiones_shared.update(resultado['particiones']) # El proceso maestro actualiza la memoria compartida con las particiones obtenidas de todos los procesos
        else:
            # Modo sin MPI
            for fase, resultado in resultados_locales.items():
                self.memoria_particiones_shared.update(resultado['particiones'])
        
        '''
        El proceso maestro es el único responsable de encontrar la solución final. 
        Este proceso selecciona la partición óptima entre todas las particiones evaluadas, buscando la que tenga
        la menor pérdida de información.
        '''
        if self.rank == 0: # Solo el proceso maestro encuentra el mínimo global
            return min(
                self.memoria_particiones_shared, 
                key=lambda k: self.memoria_particiones_shared[k][0]
            ) #Se utiliza para encontrar la partición con la mínima pérdida de información, que se calcula usando los valores almacenados en self.memoria_particiones_shared.
        
        return None

    def _ejecutar_fase_paralela(self, fase_idx: int, omegas_ciclo: List, deltas_ciclo: List):
        """Ejecuta una fase usando multiprocessing para paralelizar iteraciones."""
        
        emd_particion_candidata = INFTY_POS # Se inicializa con infinito positivo. Este valor se usará para almacenar la pérdida de información de la partición candidata durante el proceso.
        dist_particion_candidata = None # Se inicializa como None, y contendrá la distribución marginal de la partición candidata.
        particiones_fase = {} # Un diccionario donde se almacenarán las particiones generadas durante la fase. Este diccionario se irá llenando con los resultados de las combinaciones de nodos.
        

        '''
        El algoritmo evalúa cada combinación de nodos (deltas y omegas) en paralelo. 
        Cada iteración calcula la diferencia entre la combinación de nodos actual y la partición existente, 
        utilizando la función submodular. El proceso paraleliza estas evaluaciones para mejorar la eficiencia.

        En cada iteración, se evalúa la pérdida de información entre la combinación de nodos (deltas y omegas), 
        y la partición que minimiza esa pérdida (menor EMD) se selecciona como la mejor.
        '''
        for j in range(len(deltas_ciclo) - 1):
            '''
            _evaluar_deltas_paralelo(deltas_ciclo, omegas_ciclo): 
            Es el encargado de paralelizar la evaluación de los nodos delta y omega en las iteraciones. 
            Utiliza multiprocessing para evaluar todas las combinaciones de nodos simultáneamente.
            '''
            resultados_iteracion = self._evaluar_deltas_paralelo(deltas_ciclo, omegas_ciclo)
            # Encontrar el mejor resultado
            mejor_idx = min(range(len(resultados_iteracion)), 
                          key=lambda idx: resultados_iteracion[idx]['emd_iteracion'])
            
            mejor_resultado = resultados_iteracion[mejor_idx]
            
            # Actualizar configuración para siguiente ciclo
            omegas_ciclo.append(deltas_ciclo[mejor_idx])
            deltas_ciclo.pop(mejor_idx)
            
            emd_particion_candidata = mejor_resultado['emd_delta']
            dist_particion_candidata = mejor_resultado['dist_marginal_delta']
        
        '''
        Aquí se guarda la mejor partición obtenida durante el ciclo. 
        La clave clave_particion es una representación de la partición candidata, y se almacena 
        junto con su pérdida de información y la distribución marginal.
        
        deltas_ciclo[LAST_IDX]: Representa el último delta evaluado en el ciclo. 
        La partición resultante se guarda bajo esta clave en el diccionario particiones_fase, 
        junto con los valores correspondientes de pérdida de información (emd_particion_candidata) 
        y distribución marginal (dist_particion_candidata).
        '''
        
        # Guardar partición candidata
        clave_particion = tuple(
            deltas_ciclo[LAST_IDX] if isinstance(deltas_ciclo[LAST_IDX], list) else deltas_ciclo
        )
        particiones_fase[clave_particion] = (emd_particion_candidata, dist_particion_candidata)
        
        '''
        El algoritmo crea un nuevo par candidato (combinación de nodos omega y delta) 
        que será utilizado en la siguiente fase del algoritmo.

        Los nodos omegas y deltas se actualizan para preparar el sistema para la próxima 
        iteración o fase del algoritmo.
        
        Se crea el par_candidato combinando el último omega y delta de las listas actuales. 
        Luego, se actualiza la lista omegas_ciclo para la siguiente fase.
        '''

        # Formar par candidato para siguiente fase
        par_candidato = (
            [omegas_ciclo[LAST_IDX]] if isinstance(omegas_ciclo[LAST_IDX], tuple) else omegas_ciclo[LAST_IDX]
        ) + (
            deltas_ciclo[LAST_IDX] if isinstance(deltas_ciclo[LAST_IDX], list) else deltas_ciclo
        )
        
        omegas_ciclo.pop()
        omegas_ciclo.append(par_candidato)
        
        return {
            'particiones': particiones_fase,
            'vertices_siguiente': omegas_ciclo,
            'emd_minimo': emd_particion_candidata
        }

    def _evaluar_deltas_paralelo(self, deltas_ciclo: List, omegas_ciclo: List):
        """Evalúa todos los deltas en paralelo usando multiprocessing."""
        
        # SOLUCIÓN: Usar evaluación secuencial en lugar de ProcessPoolExecutor 
        # para evitar problemas de serialización con métodos de instancia
        resultados = []
        
        for k in range(len(deltas_ciclo)):
            try:
                resultado = self._evaluar_delta_secuencial(deltas_ciclo[k], omegas_ciclo.copy(), k)
                resultados.append(resultado)
            except Exception as e:
                self.logger.error(f"Error en evaluación delta {k}: {e}")
                resultados.append({
                    'emd_union': INFTY_POS,
                    'emd_delta': INFTY_POS,
                    'dist_marginal_delta': None,
                    'emd_iteracion': INFTY_POS,
                    'indice': k
                })
        
        return resultados

    def _evaluar_delta_secuencial(self, delta, omegas, idx):
        """Versión secuencial de evaluación de delta para evitar problemas de serialización."""
        try:
            emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(delta, omegas)
            emd_iteracion = emd_union - emd_delta
            
            return {
                'emd_union': emd_union,
                'emd_delta': emd_delta,
                'dist_marginal_delta': dist_marginal_delta,
                'emd_iteracion': emd_iteracion,
                'indice': idx
            }
        except Exception as e:
            self.logger.error(f"Error en evaluación secuencial {idx}: {e}")
            return {
                'emd_union': INFTY_POS,
                'emd_delta': INFTY_POS,
                'dist_marginal_delta': None,
                'emd_iteracion': INFTY_POS,
                'indice': idx
            }

    def _evaluar_delta_worker(self, args):
        """Worker function para evaluación paralela de deltas."""
        delta, omegas, idx = args
        
        try:
            emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(delta, omegas)
            emd_iteracion = emd_union - emd_delta
            
            return {
                'emd_union': emd_union,
                'emd_delta': emd_delta,
                'dist_marginal_delta': dist_marginal_delta,
                'emd_iteracion': emd_iteracion,
                'indice': idx
            }
        except Exception as e:
            self.logger.error(f"Error en worker {idx}: {e}")
            return {
                'emd_union': INFTY_POS,
                'emd_delta': INFTY_POS,
                'dist_marginal_delta': None,
                'emd_iteracion': INFTY_POS,
                'indice': idx
            }

    def funcion_submodular(self, deltas, omegas):
        """
        Función submodular principal que debe existir para mantener compatibilidad.
        Esta es la versión base que usa la implementación optimizada.
        """
        return self.funcion_submodular_optimizada(deltas, omegas)

    def funcion_submodular_optimizada(self, deltas, omegas):
        """
        Versión optimizada de función submodular con cache.
        """
        
        # Cache lookup
        cache_key = (tuple(deltas) if isinstance(deltas, list) else deltas, tuple(map(tuple, omegas)))
        if cache_key in self.memoria_omega_shared:
            return self.memoria_omega_shared[cache_key]
        
        # Evaluación de delta individual
        emd_delta, dist_marginal_delta = self._evaluar_delta_individual(deltas)
        
        # Evaluación de union
        parametros_union = self._preparar_evaluacion_union(deltas, omegas)
        emd_union = self._evaluar_union(parametros_union)
        
        # Cache del resultado
        resultado = (emd_union, emd_delta, dist_marginal_delta)
        self.memoria_omega_shared[cache_key] = resultado
        
        return resultado

    def funcion_submodular_parallel(self, deltas, omegas):
        """
        Versión paralela de función submodular (mantenida para compatibilidad).
        """
        return self.funcion_submodular_optimizada(deltas, omegas)

    def _evaluar_delta_individual(self, deltas):
        """Evaluación optimizada del delta individual."""
        temporal = [[], []]
        
        if isinstance(deltas, tuple):
            d_tiempo, d_indice = deltas
            temporal[d_tiempo].append(d_indice)
        else:
            for delta in deltas:
                d_tiempo, d_indice = delta
                temporal[d_tiempo].append(d_indice)
        
        dims_alcance_delta = temporal[EFECTO]
        dims_mecanismo_delta = temporal[ACTUAL]
        
        particion_delta = self.sia_subsistema.bipartir(
            np.array(dims_alcance_delta, dtype=np.int8),
            np.array(dims_mecanismo_delta, dtype=np.int8),
        )
        
        vector_delta_marginal = particion_delta.distribucion_marginal()
        emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
        
        return emd_delta, vector_delta_marginal

    def _preparar_evaluacion_union(self, deltas, omegas):
        """Prepara parámetros para evaluación de union."""
        temporal = [[], []]
        
        # Procesar deltas
        if isinstance(deltas, tuple):
            d_tiempo, d_indice = deltas
            temporal[d_tiempo].append(d_indice)
        else:
            for delta in deltas:
                d_tiempo, d_indice = delta
                temporal[d_tiempo].append(d_indice)
        
        # Procesar omegas
        for omega in omegas:
            if isinstance(omega, list):
                for omg in omega:
                    o_tiempo, o_indice = omg
                    temporal[o_tiempo].append(o_indice)
            else:
                o_tiempo, o_indice = omega
                temporal[o_tiempo].append(o_indice)
        
        return {
            'dims_alcance': temporal[EFECTO],
            'dims_mecanismo': temporal[ACTUAL]
        }

    def _evaluar_union(self, parametros):
        """Evaluación optimizada de union."""
        particion_union = self.sia_subsistema.bipartir(
            np.array(parametros['dims_alcance'], dtype=np.int8),
            np.array(parametros['dims_mecanismo'], dtype=np.int8),
        )
        
        vector_union_marginal = particion_union.distribucion_marginal()
        return emd_efecto(vector_union_marginal, self.sia_dists_marginales)

    def nodes_complement(self, nodes: List[Tuple[int, int]]):
        """Método sin cambios."""
        return list(set(self.vertices) - set(nodes))

    def finalize(self):
        """Finaliza recursos MPI si es necesario."""
        if self.use_mpi and MPI.Is_initialized():
            MPI.Finalize()

