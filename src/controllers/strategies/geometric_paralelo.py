import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Dict, Tuple, List, Set, Optional
from functools import partial

from src.middlewares.slogger import SafeLogger
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.middlewares.profile import profiler_manager, profile
from src.models.core.solution import Solution
from src.funcs.base import ABECEDARY, emd_efecto
from src.funcs.format import fmt_biparte_q
from src.constants.base import EFECTO, ACTUAL

from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    INFTY_NEG,
    INFTY_POS,
    LAST_IDX,
    EFECTO,
    ACTUAL,
)


def _calcular_costos_worker(args):
    """
    Función worker que calcula costos de transición para un subconjunto de variables.
    Ejecuta en proceso separado con cache local.
    
    Args:
        args: Tupla con (variables_subset, estado_inicial, estado_destino, tensores_data, n_bits)
        
    Returns:
        List[Tuple]: Lista de (var_idx, costo, var_letra) para las variables procesadas
    """
    variables_subset, estado_inicial, estado_destino, tensores_data, n_bits = args
    
    # Cache local para este worker
    cache_hamming_local = {}
    cache_costos_local = {}
    
    def _calcular_distancia_hamming_local(estado_i: int, estado_j: int) -> int:
        """Versión local de cálculo de distancia Hamming con cache."""
        clave = (estado_i, estado_j)
        if clave in cache_hamming_local:
            return cache_hamming_local[clave]
        
        xor_result = estado_i ^ estado_j
        distancia = bin(xor_result).count('1')
        
        cache_hamming_local[clave] = distancia
        return distancia
    
    def _obtener_vecinos_hipercubo_local(estado: int, n_bits: int) -> List[int]:
        """Versión local de obtención de vecinos."""
        vecinos = []
        for bit_pos in range(n_bits):
            vecino = estado ^ (1 << bit_pos)
            vecinos.append(vecino)
        return vecinos
    
    def _calcular_costo_transicion_recursivo_local(estado_i: int, estado_j: int, tensor_data: np.ndarray, n_bits: int) -> float:
        """
        Versión local del cálculo recursivo de costo de transición.
        Implementa la misma lógica que el método original pero con cache local.
        """
        clave_cache = (estado_i, estado_j)
        if clave_cache in cache_costos_local:
            return cache_costos_local[clave_cache]
        
        distancia = _calcular_distancia_hamming_local(estado_i, estado_j)
        gamma = 2 ** (-distancia)
        
        coords_i = [(estado_i >> bit) & 1 for bit in range(n_bits)]
        coords_j = [(estado_j >> bit) & 1 for bit in range(n_bits)]
        
        try:
            if tensor_data is not None and tensor_data.size > 0:
                prob_i = tensor_data[tuple(coords_i)]
                prob_j = tensor_data[tuple(coords_j)]
                contribucion_directa = abs(prob_i - prob_j)
            else:
                contribucion_directa = 0.1
        except (IndexError, TypeError):
            contribucion_directa = 0.1
        
        if distancia <= 1:
            resultado = gamma * contribucion_directa
        else:
            vecinos = _obtener_vecinos_hipercubo_local(estado_i, n_bits)
            vecinos_validos = []
            for vecino in vecinos:
                dist_vecino_destino = _calcular_distancia_hamming_local(vecino, estado_j)
                if dist_vecino_destino < distancia:
                    vecinos_validos.append(vecino)
            
            costo_recursivo = 0.0
            for vecino in vecinos_validos:
                costo_vecino = _calcular_costo_transicion_recursivo_local(vecino, estado_j, tensor_data, n_bits)
                costo_recursivo += costo_vecino
            
            suma_total_dentro_parentesis = contribucion_directa + costo_recursivo
            resultado = gamma * suma_total_dentro_parentesis
        
        cache_costos_local[clave_cache] = resultado
        return resultado
    
    # Procesar cada variable en el subconjunto
    resultados = []
    
    for var_info in variables_subset:
        var_idx, tensor_idx = var_info
        var_letra = chr(65 + var_idx)
        
        # Obtener datos del tensor para esta variable
        tensor_data = tensores_data.get(tensor_idx, None)
        
        try:
            costo = _calcular_costo_transicion_recursivo_local(
                estado_inicial, estado_destino, tensor_data, n_bits
            )
            
            if costo < float('inf'):
                resultados.append((var_idx, costo, var_letra))
            else:
                resultados.append((var_idx, float('inf'), var_letra))
                
        except Exception as e:
            # En caso de error, asignar costo infinito
            resultados.append((var_idx, float('inf'), var_letra))
    
    return resultados


class GeometricSiaParallel(SIA):
    """
    Estrategia GeometricSia que implementa análisis geométrico de hipercubos
    con representación n-dimensional y cálculo paralelo de costos de transición.
    """
    
    def __init__(self, config: Manager):
        super().__init__(config)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(config.estado_inicial)}{config.pagina}"
        )
        self.logger = SafeLogger("geometric_sia")
        self.tensores = []
        self.cache_hamming = {}
        self.cache_costos = {}
        self.pool = None
        self._max_workers = min(8, mp.cpu_count())

    def __del__(self):
        """Cleanup del pool de procesos al destruir la instancia."""
        self._cleanup_pool()
    
    def _cleanup_pool(self):
        """Limpia el pool de procesos si existe."""
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
                self.pool = None
            except Exception as e:
                self.logger.warn(f"Error cerrando pool de procesos: {e}")

    def _get_process_pool(self) -> mp.Pool:
        """Obtiene o crea el pool de procesos reutilizable."""
        if self.pool is None:
            try:
                self.pool = mp.Pool(processes=self._max_workers)
            except Exception as e:
                self.logger.warn(f"Error creando pool de procesos: {e}. Usando cálculo secuencial.")
                return None
        return self.pool

    @profile(context={TYPE_TAG: "geometric_sia"})
    def aplicar_estrategia(self, conditions, purview, mechanism):
        """
        Método principal que ejecuta la estrategia GeometricSia.
        """
        try:
            self.sia_preparar_subsistema(conditions, purview, mechanism)
            self._construir_representacion_ndimensional()
            biparticion_optima = self._identificar_biparticion_optima()
            return self._formatear_resultado(biparticion_optima=biparticion_optima)
        finally:
            # Limpia el pool al final del procesamiento
            self._cleanup_pool()
    
    def _construir_representacion_ndimensional(self):
        """
        Construye la representación n-dimensional del sistema como tensores.
        Cada tensor representa las probabilidades condicionales de una variable.
        IMPORTANTE: Reordena los datos considerando el formato little-endian.
        """
        self.tensores = []
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_bits = len(self.sia_subsistema.dims_ncubos)
        
        self.logger.info(f"Construyendo representación n-dimensional...")
        self.logger.info(f"Número de variables: {n_vars}")
        self.logger.info(f"Número de bits: {n_bits}")
        
        for i, ncube in enumerate(self.sia_subsistema.ncubos):
            tensor_original = ncube.data.copy()
            tensor_reordenado = self._reordenar_tensor_little_endian(tensor_original, n_bits)
            self.tensores.append(tensor_reordenado)
            
            self.logger.debug(f"Tensor {i} creado con forma: {tensor_reordenado.shape}")
    
    def _preparar_datos_tensor_worker(self, tensor_indices: List[int]) -> Dict[int, np.ndarray]:
        """
        Prepara datos mínimos de tensores para el worker.
        Solo envía los tensores necesarios para minimizar overhead.
        
        Args:
            tensor_indices: Lista de índices de tensores necesarios
            
        Returns:
            Dict con {tensor_idx: tensor_data}
        """
        tensores_data = {}
        
        for tensor_idx in tensor_indices:
            if tensor_idx < len(self.tensores):
                # Crear copia para evitar problemas de concurrencia
                tensores_data[tensor_idx] = self.tensores[tensor_idx].copy()
            else:
                tensores_data[tensor_idx] = None
                
        return tensores_data
    
    def _calcular_costos_paralelo(self, variables_info: List[Tuple[int, int]], estado_inicial: int, estado_destino: int, n_bits: int) -> List[Tuple[int, float, str]]:
        """
        Coordinador que distribuye el cálculo de costos entre procesos paralelos.
        
        Args:
            variables_info: Lista de tuplas (var_idx, tensor_idx)
            estado_inicial: Estado inicial
            estado_destino: Estado destino
            n_bits: Número de bits del sistema
            
        Returns:
            Lista de tuplas (var_idx, costo, var_letra)
        """
        if not variables_info:
            return []
        
        pool = self._get_process_pool()
        
        # Si no se puede crear el pool, usar cálculo secuencial
        if pool is None:
            return self._calcular_costos_secuencial(variables_info, estado_inicial, estado_destino, n_bits)
        
        try:
            # Dividir trabajo entre workers
            chunk_size = max(1, len(variables_info) // self._max_workers)
            chunks = [variables_info[i:i + chunk_size] for i in range(0, len(variables_info), chunk_size)]
            
            # Preparar datos de tensores necesarios
            tensor_indices = list(set(tensor_idx for _, tensor_idx in variables_info))
            tensores_data = self._preparar_datos_tensor_worker(tensor_indices)
            
            # Preparar argumentos para workers
            args_list = []
            for chunk in chunks:
                if chunk:  # Solo procesar chunks no vacíos
                    args_list.append((chunk, estado_inicial, estado_destino, tensores_data, n_bits))
            
            # Ejecutar en paralelo sin timeout (usar get() para obtener resultados)
            async_result = pool.map_async(_calcular_costos_worker, args_list)
            
            # Esperar resultados con timeout manual
            try:
                resultados_chunks = async_result.get(timeout=300)
            except mp.TimeoutError:
                self.logger.warn("Timeout en cálculo paralelo, usando secuencial")
                return self._calcular_costos_secuencial(variables_info, estado_inicial, estado_destino, n_bits)
            
            # Combinar resultados
            resultados_finales = []
            for chunk_resultado in resultados_chunks:
                resultados_finales.extend(chunk_resultado)
            
            return resultados_finales
            
        except Exception as e:
            self.logger.warn(f"Error en cálculo paralelo: {e}. Usando cálculo secuencial.")
            return self._calcular_costos_secuencial(variables_info, estado_inicial, estado_destino, n_bits)
    
    def _calcular_costos_secuencial(self, variables_info: List[Tuple[int, int]], estado_inicial: int, estado_destino: int, n_bits: int) -> List[Tuple[int, float, str]]:
        """
        Fallback para cálculo secuencial cuando el paralelismo falla.
        
        Args:
            variables_info: Lista de tuplas (var_idx, tensor_idx)
            estado_inicial: Estado inicial
            estado_destino: Estado destino
            n_bits: Número de bits del sistema
            
        Returns:
            Lista de tuplas (var_idx, costo, var_letra)
        """
        resultados = []
        
        for var_idx, tensor_idx in variables_info:
            var_letra = chr(65 + var_idx)
            costo = self._obtener_costo_transicion(var_idx, estado_inicial, estado_destino)
            
            if costo < float('inf'):
                resultados.append((var_idx, costo, var_letra))
            else:
                resultados.append((var_idx, float('inf'), var_letra))
        
        return resultados
    
    def _reordenar_tensor_little_endian(self, tensor_original: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Reordena un tensor de formato big-endian a little-endian.
        
        Args:
            tensor_original: Tensor en formato big-endian (orden tradicional)
            n_bits: Número de bits del sistema
            
        Returns:
            np.ndarray: Tensor reordenado en formato little-endian
        """
        tensor_reordenado = np.zeros_like(tensor_original)
        n_estados = 2 ** n_bits
        
        for estado_decimal in range(n_estados):
            coords_big_endian = [(estado_decimal >> bit) & 1 for bit in range(n_bits)]
            coords_little_endian = coords_big_endian[::-1]
            estado_little_decimal = sum(bit * (2 ** pos) for pos, bit in enumerate(coords_little_endian))
            coords_little_indexing = [(estado_little_decimal >> bit) & 1 for bit in range(n_bits)]
            
            if tensor_original.ndim == n_bits:
                tensor_reordenado[tuple(coords_big_endian)] = tensor_original[tuple(coords_little_indexing)]
            else:
                tensor_reordenado[tuple(coords_big_endian)] = tensor_original[tuple(coords_little_indexing)]
        
        return tensor_reordenado
    
    def _decimal_to_little_endian_binary(self, decimal: int, n_bits: int) -> str:
        """
        Convierte un número decimal a su representación binaria little-endian.
        """
        binary = format(decimal, f'0{n_bits}b')
        return binary[::-1]
    
    def _obtener_vecinos_hipercubo(self, estado: int, n_bits: int) -> List[int]:
        """
        Obtiene los vecinos inmediatos de un estado en el hipercubo (distancia Hamming = 1).
        Considera la representación little-endian.
        """
        vecinos = []
        
        for bit_pos in range(n_bits):
            vecino = estado ^ (1 << bit_pos)
            vecinos.append(vecino)
            
        return vecinos
    
    def _calcular_distancia_hamming(self, estado_i: int, estado_j: int) -> int:
        """
        Calcula la distancia de Hamming entre dos estados.
        """
        clave = (estado_i, estado_j)
        if clave in self.cache_hamming:
            return self.cache_hamming[clave]
        
        xor_result = estado_i ^ estado_j
        distancia = bin(xor_result).count('1')
        
        self.cache_hamming[clave] = distancia
        return distancia
    
    def _calcular_costo_transicion_recursivo(self, estado_i: int, estado_j: int, tensor_idx: int, n_bits: int) -> float:
        """
        Calcula el costo de transición entre dos estados de forma recursiva con cache.
        
        Implementa la función CORREGIDA:
        t(i, j) = γ · (|X[i] - X[j]| + Σ_{k∈N(i)} {t(k, j)}) si d(i,j) > 1
        t(i, j) = γ · |X[i] - X[j]| si d(i,j) ≤ 1
        
        donde γ = 2^(-d(i,j)) y d(i,j) es la distancia de Hamming
        """
        clave_cache = (tensor_idx, estado_i, estado_j)
        if clave_cache in self.cache_costos:
            return self.cache_costos[clave_cache]
        
        distancia = self._calcular_distancia_hamming(estado_i, estado_j)
        gamma = 2 ** (-distancia)
        
        coords_i = [(estado_i >> bit) & 1 for bit in range(n_bits)]
        coords_j = [(estado_j >> bit) & 1 for bit in range(n_bits)]
        
        try:
            if tensor_idx < len(self.tensores):
                prob_i = self.tensores[tensor_idx][tuple(coords_i)]
                prob_j = self.tensores[tensor_idx][tuple(coords_j)]
                contribucion_directa = abs(prob_i - prob_j)
            else:
                prob_i, prob_j = 0.1, 0.1
                contribucion_directa = 0.1
        except (IndexError, TypeError):
            prob_i, prob_j = 0.1, 0.1
            contribucion_directa = 0.1
        
        if distancia <= 1:
            resultado = gamma * contribucion_directa
        else:
            vecinos = self._obtener_vecinos_hipercubo(estado_i, n_bits)
            vecinos_validos = []
            for vecino in vecinos:
                dist_vecino_destino = self._calcular_distancia_hamming(vecino, estado_j)
                if dist_vecino_destino < distancia:
                    vecinos_validos.append(vecino)
            
            costo_recursivo = 0.0
            
            for vecino in vecinos_validos:
                costo_vecino = self._calcular_costo_transicion_recursivo(vecino, estado_j, tensor_idx, n_bits)
                costo_recursivo += costo_vecino
            
            suma_total_dentro_parentesis = contribucion_directa + costo_recursivo
            resultado = gamma * suma_total_dentro_parentesis
        
        self.cache_costos[clave_cache] = resultado
        return resultado
    
    def _calcular_complemento_estado(self, estado_inicial: int, estado_destino: int, n_bits: int) -> int:
        """
        Calcula el estado complementario:
        - Identifica qué bits cambiaron en la transición original
        - El complementario cambia los bits que NO cambiaron en la original
        """
        bits_cambiados = estado_inicial ^ estado_destino
        complementario = estado_inicial
        
        for bit_pos in range(n_bits):
            bit_cambio_en_original = (bits_cambiados >> bit_pos) & 1
            
            if not bit_cambio_en_original:
                complementario ^= (1 << bit_pos)
        
        return complementario
    
    def _obtener_costo_transicion(self, var_idx, estado_i, estado_j):
        """
        Obtiene el costo de transición calculándolo bajo demanda.
        """
        indices_ncubos = self.sia_subsistema.indices_ncubos
        n_bits = len(self.sia_subsistema.dims_ncubos)
        
        # Convertir var_idx al índice correcto en la lista de tensores
        tensor_idx = None
        for i, indice in enumerate(indices_ncubos):
            if indice == var_idx:
                tensor_idx = i
                break
        
        if tensor_idx is None:
            self.logger.warn(f"Variable {var_idx} no encontrada en indices_ncubos")
            return float('inf')
        
        # Calcular el costo bajo demanda
        try:
            costo = self._calcular_costo_transicion_recursivo(estado_i, estado_j, tensor_idx, n_bits)
            return costo
        except Exception as e:
            self.logger.error(f"Error calculando costo para var {var_idx}, estados {estado_i}->{estado_j}: {e}")
            return float('inf')
    
    def _identificar_biparticion_optima(self) -> Tuple[Set[int], Set[int]]:
        """
        ENFOQUE MODULAR MODIFICADO con paralelización: Analiza los casos especiales y transiciones estándar
        utilizando cálculo paralelo de costos donde sea posible.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_bits = len(self.sia_subsistema.dims_ncubos)
        estado_inicial = 0
        
        print(f"\n=== INICIANDO ANÁLISIS DE BIPARTICIÓN ÓPTIMA (PARALELO) ===")
        
        # PASO 1: Analizar caso especial 000→111
        print(f"PASO 1: Analizando caso especial 000→111...")
        particiones_especiales, particion_optima_especial = self._analizar_caso_especial_todos_unos(n_vars, n_bits, estado_inicial)
        
        # PASO 2: Analizar transiciones con solo un cero
        print(f"PASO 2: Analizando transiciones con solo un cero...")
        particiones_un_cero, particion_optima_un_cero = self._analizar_transiciones_solo_un_cero(
            n_vars, n_bits, estado_inicial
        )
        
        # PASO 3: Comparar casos especiales y decidir si retornar
        print(f"\nPASO 3: COMPARANDO CASOS ESPECIALES...")
        
        todas_particiones_especiales = particiones_especiales + particiones_un_cero
        
        if todas_particiones_especiales:
            todas_particiones_especiales.sort(key=lambda x: x['emd'])
            mejor_caso_especial = todas_particiones_especiales[0]
            
            print(f"Mejor caso especial encontrado:")
            print(f"  - Transición: {mejor_caso_especial['transicion_original_bin']}")
            print(f"  - EMD (pérdida): {mejor_caso_especial['emd']:.6f}")
            print(f"  - Tipo: {mejor_caso_especial['info_debug'].get('caso_especial', 'N/A')}")
            
            if mejor_caso_especial['emd'] < 1e-2:
                print(f"  - EMD < 1e-2, retornando solución óptima")
                
                conjunto_presente = mejor_caso_especial['conjunto_presente']
                conjunto_futuro = mejor_caso_especial['conjunto_futuro']
                
                return (conjunto_presente, conjunto_futuro)
            else:
                print(f"  - EMD >= 1e-2, continuando con análisis completo")
        
        # PASO 4: Verificar soluciones óptimas individuales
        if particion_optima_especial is not None:
            print(f"PASO 4: Retornando solución óptima del caso especial 000→111")
            return particion_optima_especial
        
        if particion_optima_un_cero is not None:
            print(f"PASO 4: Retornando solución óptima de transiciones con un cero")
            return particion_optima_un_cero
        
        # PASO 5: Analizar transiciones estándar
        print(f"PASO 5: Analizando transiciones estándar...")
        particiones_estandar, particion_optima_estandar = self._analizar_transiciones_estandar(n_vars, n_bits, estado_inicial)
        
        if particion_optima_estandar is not None:
            print("PASO 5: Solución óptima encontrada en transiciones estándar")
            return particion_optima_estandar
        
        # PASO 6: Combinar todas las particiones candidatas
        print(f"PASO 6: Combinando todas las particiones candidatas...")
        particiones_candidatas = todas_particiones_especiales + particiones_estandar
        
        if particiones_candidatas:
            particiones_candidatas.sort(key=lambda x: x['emd'])
            mejor_particion = particiones_candidatas[0]
            
            print(f"MEJOR PARTICIÓN GLOBAL:")
            print(f"  - EMD: {mejor_particion['emd']:.6f}")
            print(f"  - Basada en: {mejor_particion['transicion_original_bin']}")
            
            return (mejor_particion['conjunto_presente'], mejor_particion['conjunto_futuro'])
        else:
            print("ERROR: No se encontraron particiones candidatas válidas")
            self.logger.error("No se encontraron particiones candidatas válidas")
            return None, None
    
    def _analizar_transiciones_estandar(self, n_vars, n_bits, estado_inicial) -> Tuple[List[dict], Optional[Tuple[Set[int], Set[int]]]]:
        """
        Analiza las transiciones estándar con paralelización en el cálculo de costos.
        """
        particiones_candidatas = []
        estado_todos_unos = 2**n_bits - 1

        estados_un_cero = set()
        for bit_pos in range(n_bits):
            estado = (2**n_bits - 1) ^ (1 << bit_pos)
            estados_un_cero.add(estado)
        
        variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
        variables_disponibles_mecanismo = set(self.sia_subsistema.dims_ncubos)
        
        promedios_transiciones = []
        
        # Calcular costos para los estados que vamos a analizar
        for estado_destino in range(1, 2**n_bits):
            if estado_destino == estado_todos_unos:
                continue
            if estado_destino in estados_un_cero:
                continue
                
            estado_destino_bin = self._decimal_to_little_endian_binary(estado_destino, n_bits)
            
            # PARALELIZACIÓN: Preparar información de variables
            variables_info = []
            indices_ncubos = self.sia_subsistema.indices_ncubos
            
            for var_idx in sorted(variables_disponibles_alcance):
                # Encontrar tensor_idx correspondiente
                tensor_idx = None
                for i, indice in enumerate(indices_ncubos):
                    if indice == var_idx:
                        tensor_idx = i
                        break
                
                if tensor_idx is not None:
                    variables_info.append((var_idx, tensor_idx))
            
            # Cálculo paralelo de costos
            if variables_info:
                print(f"  Calculando costos para t(000, {estado_destino_bin}) - {len(variables_info)} variables en paralelo")
                costos_resultado = self._calcular_costos_paralelo(variables_info, estado_inicial, estado_destino, n_bits)
                
                costos_alcance = [(var_idx, costo, var_letra) for var_idx, costo, var_letra in costos_resultado if costo < float('inf')]
                
                if costos_alcance:
                    suma_costos = sum(costo for _, costo, _ in costos_alcance)
                    promedio_transicion = suma_costos / len(costos_alcance)
                    promedios_transiciones.append({
                        'estado_destino': estado_destino,
                        'estado_destino_bin': estado_destino_bin,
                        'costos_alcance': costos_alcance,
                        'promedio': promedio_transicion
                    })
        
        if promedios_transiciones:
            suma_promedios = sum(t['promedio'] for t in promedios_transiciones)
            promedio_global = suma_promedios / len(promedios_transiciones)
            
            transiciones_candidatas = [t for t in promedios_transiciones if t['promedio'] <= promedio_global]
        else:
            return particiones_candidatas, None
        
        # Procesar transiciones candidatas
        for transicion in transiciones_candidatas:
            estado_destino = transicion['estado_destino']
            estado_destino_bin = transicion['estado_destino_bin']
            costos_alcance = transicion['costos_alcance']
            
            # Calcular estado complementario y sus costos en paralelo
            estado_complementario = self._calcular_complemento_estado(estado_inicial, estado_destino, n_bits)
            
            # Preparar variables para cálculo paralelo del complementario
            variables_info_comp = [(var_idx, tensor_idx) for var_idx, _, _ in costos_alcance 
                                   for tensor_idx, indice in enumerate(self.sia_subsistema.indices_ncubos) 
                                   if indice == var_idx]
            
            costos_complementario_resultado = self._calcular_costos_paralelo(
                variables_info_comp, estado_inicial, estado_complementario, n_bits
            )
            
            costos_complementario = [(var_idx, costo, var_letra) for var_idx, costo, var_letra in costos_complementario_resultado]
            
            conjunto_presente = set()
            conjunto_futuro = set()
            
            for i, (var_idx, costo_original, var_letra) in enumerate(costos_alcance):
                if i < len(costos_complementario):
                    _, costo_comp, _ = costos_complementario[i]
                    
                    if costo_original < costo_comp:
                        if var_idx in variables_disponibles_mecanismo:
                            conjunto_presente.add(var_idx)
                        else:
                            conjunto_futuro.add(var_idx)
                    else:
                        conjunto_futuro.add(var_idx)
            
            bits_que_se_mantienen = self._identificar_bits_que_se_mantienen(estado_inicial, estado_destino, n_bits)
            
            emd_valor = self._calcular_emd_biparticion_completa(bits_que_se_mantienen, conjunto_presente)
                
            if emd_valor < float('inf'):
                particiones_candidatas.append({
                    'transicion_original': (estado_inicial, estado_destino),
                    'transicion_original_bin': f"t(000, {estado_destino_bin})",
                    'promedio_original': transicion['promedio'],
                    'conjunto_presente': bits_que_se_mantienen,
                    'conjunto_futuro': conjunto_presente,
                    'emd': emd_valor,
                    'info_debug': {
                        'costos_originales': [costo for _, costo, _ in costos_alcance],
                        'costos_complemento': [costo for _, costo, _ in costos_complementario],
                        'bits_que_se_mantienen': bits_que_se_mantienen,
                        'caso_especial': False
                    }
                })
                    
                if emd_valor < 1e-4:
                    return particiones_candidatas, (bits_que_se_mantienen, conjunto_presente)
        
        # Log de estadísticas de caché
        self.logger.info(f"Costos calculados: {len(self.cache_costos)}")
        self.logger.info(f"Distancias Hamming calculadas: {len(self.cache_hamming)}")
        
        return particiones_candidatas, None
        
    def _analizar_caso_especial_todos_unos(self, n_vars, n_bits, estado_inicial):
        """
        Analiza específicamente la transición 000→111 con paralelización.
        """
        particiones_candidatas = []
        estado_todos_unos = 2**n_bits - 1
        
        print(f"\n=== CASO ESPECIAL: EVALUANDO TRANSICIÓN 000→111 (PARALELO) ===")
        
        variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
        print(f"Variables disponibles para alcance: {[chr(65 + i) for i in sorted(variables_disponibles_alcance)]}")
        
        # Preparar información de variables para cálculo paralelo
        variables_info = []
        indices_ncubos = self.sia_subsistema.indices_ncubos
        
        for var_idx in sorted(variables_disponibles_alcance):
            tensor_idx = None
            for i, indice in enumerate(indices_ncubos):
                if indice == var_idx:
                    tensor_idx = i
                    break
            
            if tensor_idx is not None:
                variables_info.append((var_idx, tensor_idx))

        if not variables_info:
            print("No se encontraron variables disponibles con costos válidos")
            return particiones_candidatas, None
        
        print(f"Calculando costos para {len(variables_info)} variables en paralelo...")
        
        # Cálculo paralelo de costos
        costos_resultado = self._calcular_costos_paralelo(variables_info, estado_inicial, estado_todos_unos, n_bits)
        
        variables_validas = [(var_idx, costo, var_letra) for var_idx, costo, var_letra in costos_resultado if costo < float('inf')]
        
        if not variables_validas:
            print("No se encontraron variables con costos válidos después del cálculo paralelo")
            return particiones_candidatas, None
        
        print(f"Evaluando {len(variables_validas)} variables válidas con costos calculados en paralelo")
        
        # Ordenar y evaluar todas las variables válidas
        variables_validas.sort(key=lambda x: x[1])
        
        for var_idx, costo, var_letra in variables_validas:
            conjunto_futuro = {var_idx}
            conjunto_presente = set()
            
            print(f"Evaluando variable {var_letra} como único elemento en FUTURO, mecanismo vacío")
            
            emd_valor = self._calcular_emd_biparticion_completa(conjunto_presente, conjunto_futuro)
            
            if emd_valor < float('inf'):
                particiones_candidatas.append({
                    'transicion_original': (estado_inicial, estado_todos_unos),
                    'transicion_original_bin': f"t(000, {self._decimal_to_little_endian_binary(estado_todos_unos, n_bits)})",
                    'promedio_original': costo,
                    'conjunto_presente': conjunto_presente,
                    'conjunto_futuro': conjunto_futuro,
                    'emd': emd_valor,
                    'info_debug': {
                        'costos_originales': [costo],
                        'costos_complemento': [0],
                        'bits_que_se_mantienen': [],
                        'caso_especial': True
                    }
                })
                
                print(f"  EMD calculado: {emd_valor:.6f}")
                
                if emd_valor < 1e-3:
                    print(f"  ¡EMD perfecto con variable {var_letra}! Retornando solución óptima.")
                    return particiones_candidatas, (conjunto_presente, conjunto_futuro)

        return particiones_candidatas, None
    
    def _analizar_transiciones_solo_un_cero(self, n_vars, n_bits, estado_inicial) -> Tuple[List[dict], Optional[Tuple[Set[int], Set[int]]]]:
        """
        Analiza las transiciones con solo un cero usando paralelización.
        """
        particiones_candidatas = []
        
        print(f"\n=== TRANSICIONES SOLO UN CERO: EVALUANDO (PARALELO) ===")
        
        variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
        variables_disponibles_mecanismo = set(self.sia_subsistema.dims_ncubos)
        
        # Generar estados con solo un cero
        estados_un_cero = []
        for bit_pos in range(n_bits):
            estado = (2**n_bits - 1) ^ (1 << bit_pos)
            estados_un_cero.append(estado)
        
        print(f"Estados con solo un cero a evaluar: {len(estados_un_cero)}")
        
        promedios_transiciones = []
        
        # Evaluar cada transición con paralelización
        for estado_destino in estados_un_cero:
            estado_destino_bin = self._decimal_to_little_endian_binary(estado_destino, n_bits)
            print(f"\n--- Evaluando transición t(000, {estado_destino_bin}) ---")
            
            # Preparar variables para cálculo paralelo
            variables_info = []
            indices_ncubos = self.sia_subsistema.indices_ncubos
            
            for var_idx in sorted(variables_disponibles_alcance):
                tensor_idx = None
                for i, indice in enumerate(indices_ncubos):
                    if indice == var_idx:
                        tensor_idx = i
                        break
                
                if tensor_idx is not None:
                    variables_info.append((var_idx, tensor_idx))
            
            # Cálculo paralelo de costos
            if variables_info:
                print(f"  Calculando costos para {len(variables_info)} variables en paralelo...")
                costos_resultado = self._calcular_costos_paralelo(variables_info, estado_inicial, estado_destino, n_bits)
                
                costos_alcance = [(var_idx, costo, var_letra) for var_idx, costo, var_letra in costos_resultado if costo < float('inf')]
                
                if costos_alcance:
                    suma_costos = sum(costo for _, costo, _ in costos_alcance)
                    promedio_transicion = suma_costos / len(costos_alcance)
                    promedios_transiciones.append({
                        'estado_destino': estado_destino,
                        'estado_destino_bin': estado_destino_bin,
                        'costos_alcance': costos_alcance,
                        'promedio': promedio_transicion
                    })
                    print(f"  Promedio calculado: {promedio_transicion:.4f}")
        
        if promedios_transiciones:
            suma_promedios = sum(t['promedio'] for t in promedios_transiciones)
            promedio_global = suma_promedios / len(promedios_transiciones)
            print(f"\n=== PROMEDIO GLOBAL TRANSICIONES UN CERO: {promedio_global:.4f} ===")
            
            transiciones_candidatas = [t for t in promedios_transiciones if t['promedio'] <= promedio_global]
            print(f"Transiciones candidatas: {len(transiciones_candidatas)}/{len(promedios_transiciones)}")
        else:
            print("ERROR: No se encontraron transiciones válidas con solo un cero")
            return particiones_candidatas, None
        
        # Analizar cada transición candidata
        for transicion in transiciones_candidatas:
            estado_destino = transicion['estado_destino']
            estado_destino_bin = transicion['estado_destino_bin']
            costos_alcance = transicion['costos_alcance']
            
            print(f"\n--- ANALIZANDO CANDIDATA UN CERO: t(000, {estado_destino_bin}) ---")
            
            # Calcular estado complementario
            estado_complementario = self._calcular_complemento_estado(estado_inicial, estado_destino, n_bits)
            
            # Preparar variables para cálculo paralelo del complementario
            variables_info_comp = []
            indices_ncubos = self.sia_subsistema.indices_ncubos
            
            for var_idx, _, _ in costos_alcance:
                tensor_idx = None
                for i, indice in enumerate(indices_ncubos):
                    if indice == var_idx:
                        tensor_idx = i
                        break
                
                if tensor_idx is not None:
                    variables_info_comp.append((var_idx, tensor_idx))
            
            # Cálculo paralelo de costos del complementario
            print(f"  Calculando costos complementarios para {len(variables_info_comp)} variables en paralelo...")
            costos_complementario_resultado = self._calcular_costos_paralelo(
                variables_info_comp, estado_inicial, estado_complementario, n_bits
            )
            
            costos_complementario = [(var_idx, costo, var_letra) for var_idx, costo, var_letra in costos_complementario_resultado]
            
            # Determinar partición
            conjunto_presente = set()
            conjunto_futuro = set()
            
            for i, (var_idx, costo_original, var_letra) in enumerate(costos_alcance):
                if i < len(costos_complementario):
                    _, costo_comp, _ = costos_complementario[i]
                    
                    if costo_original < costo_comp:
                        if var_idx in variables_disponibles_mecanismo:
                            conjunto_presente.add(var_idx)
                            print(f"    {var_letra}: Original mejor → PRESENTE")
                        else:
                            conjunto_futuro.add(var_idx)
                            print(f"    {var_letra}: Original mejor pero NO disponible en mecanismo → FUTURO")
                    else:
                        conjunto_futuro.add(var_idx)
                        print(f"    {var_letra}: Complementario mejor → FUTURO")
            
            bits_que_se_mantienen = self._identificar_bits_que_se_mantienen(estado_inicial, estado_destino, n_bits)
            emd_valor = self._calcular_emd_biparticion_completa(bits_que_se_mantienen, conjunto_presente)
            print(f"    EMD calculado: {emd_valor:.6f}")
                
            if emd_valor < float('inf'):
                particiones_candidatas.append({
                    'transicion_original': (estado_inicial, estado_destino),
                    'transicion_original_bin': f"t(000, {estado_destino_bin})",
                    'promedio_original': transicion['promedio'],
                    'conjunto_presente': bits_que_se_mantienen,
                    'conjunto_futuro': conjunto_presente,
                    'emd': emd_valor,
                    'info_debug': {
                        'costos_originales': [costo for _, costo, _ in costos_alcance],
                        'costos_complemento': [costo for _, costo, _ in costos_complementario],
                        'bits_que_se_mantienen': bits_que_se_mantienen,
                        'caso_especial': 'solo_un_cero'
                    }
                })
                    
                if emd_valor < 1e-3:
                    print(f"    ¡EMD perfecto encontrado! Retornando solución óptima.")
                    return particiones_candidatas, (bits_que_se_mantienen, conjunto_presente)
        
        print(f"\nTotal de particiones candidatas con solo un cero: {len(particiones_candidatas)}")
        return particiones_candidatas, None

    def _identificar_bits_que_se_mantienen(self, estado_inicial: int, estado_destino: int, n_bits: int) -> List[int]:
        """
        Identifica qué bits se MANTIENEN (no cambian) entre dos estados.
        
        Args:
            estado_inicial: Estado inicial
            estado_destino: Estado destino
            n_bits: Número total de bits
            
        Returns:
            List[int]: Lista de variables del mecanismo correspondientes a los bits que se mantienen
        """
        diferencia = estado_inicial ^ estado_destino
        print(f"Diferencia entre estados: {diferencia:0{n_bits}b} (en binario)")
        
        bits_indice = []
        variables_mecanismo = sorted(self.sia_subsistema.dims_ncubos)  # Ordenar para mapeo consistente
        
        # Encontrar posiciones de bits que NO cambian (donde diferencia es 0)
        for bit_pos in range(n_bits):
            if not ((diferencia >> bit_pos) & 1):
                bits_indice.append(bit_pos)
        # Mapear índices de bits a variables del mecanismo
        bits_que_se_mantienen = []
        for i, bit_idx in enumerate(bits_indice):
            if i < len(variables_mecanismo):
                variable_correspondiente = variables_mecanismo[i]
                bits_que_se_mantienen.append(variable_correspondiente)
                print(f"Bit índice {bit_idx} -> Variable {variable_correspondiente}")
        
        return bits_que_se_mantienen

    def _calcular_emd_biparticion_completa(self, conjunto_presente: Set[int], conjunto_futuro: Set[int]) -> float:
        """
        Calcula la EMD entre el subsistema original y una bipartición.
        """
        try:
            indices_ncubos_futuro = np.array(list(conjunto_futuro), dtype=np.int8)
            dimensiones_presente = np.array(list(conjunto_presente), dtype=np.int8)
            
            particion = self.sia_subsistema.bipartir(indices_ncubos_futuro, dimensiones_presente)

            dist_original = self.sia_dists_marginales
            dist_particion = particion.distribucion_marginal()
            
            emd = emd_efecto(dist_particion, dist_original)
            return emd
            
        except Exception as e:
            self.logger.error(f"Error en _calcular_emd_biparticion_completa: {e}")
            return float('inf')

    def _formatear_resultado(self, biparticion_optima: Tuple[Set[int], Set[int]]) -> Solution:
        """
        Formatea el resultado respetando las variables disponibles para cada componente.
        """
        if biparticion_optima[0] is None or biparticion_optima[1] is None:
            self.logger.error("No se pudo encontrar una bipartición válida")
            return None
        
        conjunto_presente, conjunto_futuro = biparticion_optima
        
        perdida = self._calcular_emd_biparticion_completa(conjunto_presente, conjunto_futuro)
        
        try:
            indices_mecanismo = np.array(list(conjunto_presente), dtype=np.int8)
            indices_alcance = np.array(list(conjunto_futuro), dtype=np.int8)
            
            particion = self.sia_subsistema.bipartir(indices_alcance, indices_mecanismo)
            distribucion_particion = particion.distribucion_marginal()
            variables_disponibles_mecanismo = set(self.sia_subsistema.dims_ncubos)
            
            variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)

            complemento_presente = variables_disponibles_mecanismo - set(conjunto_presente)
            complemento_futuro = variables_disponibles_alcance - set(conjunto_futuro)
            
            nodos_mecanismo = []
            nodos_alcance = []
            
            futuro_disponible = conjunto_futuro
            for var in futuro_disponible:
                nodos_mecanismo.append((EFECTO, var))
            
            presente_disponible = conjunto_presente
            for var in presente_disponible:
                nodos_mecanismo.append((ACTUAL, var))
            
            complemento_futuro_disponible = complemento_futuro.intersection(variables_disponibles_alcance)
            for var in complemento_futuro_disponible:
                nodos_alcance.append((EFECTO, var))
            
            complemento_presente_disponible = complemento_presente.intersection(variables_disponibles_mecanismo)
            for var in complemento_presente_disponible:
                nodos_alcance.append((ACTUAL, var))
            
            fmt_particion = fmt_biparte_q(nodos_mecanismo, nodos_alcance)
            
        except Exception as e:
            self.logger.error(f"Error en formateo: {e}")
            return None
            
        return Solution(
            estrategia="Geometric-SIA",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_particion,
        )