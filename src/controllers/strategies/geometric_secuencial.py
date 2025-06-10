import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Set, Optional

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

class GeometricSia(SIA):
    """
    Estrategia GeometricSia que implementa análisis geométrico de hipercubos
    con representación n-dimensional y cálculo de costos de transición bajo demanda.
    OPTIMIZADA con heurística de distancia Hamming adaptativa.
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

    @profile(context={TYPE_TAG: "geometric_sia"})
    def aplicar_estrategia(self, conditions, purview, mechanism):
        """
        Método principal que ejecuta la estrategia GeometricSia.
        """
        self.sia_preparar_subsistema(conditions, purview, mechanism)
        self._construir_representacion_ndimensional()
        biparticion_optima = self._identificar_biparticion_optima()
        return self._formatear_resultado(biparticion_optima=biparticion_optima)
    
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
        
        Args:
            estado: Estado actual en decimal
            n_bits: Número de bits del sistema
            
        Returns:
            List[int]: Lista de estados vecinos
        """
        vecinos = []
        
        for bit_pos in range(n_bits):
            vecino = estado ^ (1 << bit_pos)
            vecinos.append(vecino)
            
        return vecinos
    
    def _calcular_distancia_hamming(self, estado_i: int, estado_j: int) -> int:
        """
        Calcula la distancia de Hamming entre dos estados.
        
        Args:
            estado_i: Estado inicial
            estado_j: Estado final
            
        Returns:
            int: Distancia de Hamming
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
        
        Ejemplo: t(000, 100) -> bit 0 cambió
        Complementario: cambiar bits 1 y 2 -> 011
        
        Args:
            estado_inicial: Estado inicial (ej: 000 = 0)
            estado_destino: Estado destino (ej: 100 = 1) 
            n_bits: Número total de bits
            
        Returns:
            int: Estado complementario (ej: 011 = 6)
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
        
        Args:
            var_idx: Índice de la variable
            estado_i: Estado inicial
            estado_j: Estado final
            
        Returns:
            float: Costo de la transición
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
            self.logger.warning(f"Variable {var_idx} no encontrada en indices_ncubos")
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
        ENFOQUE MODULAR MODIFICADO: Analiza los casos especiales (000→111 y transiciones con solo un cero)
        y si el mejor tiene pérdida < 1e-2, lo retorna inmediatamente. Si no, continúa con transiciones estándar.
        
        Returns:
            Tuple con los dos conjuntos de variables de la bipartición óptima
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_bits = len(self.sia_subsistema.dims_ncubos)
        estado_inicial = 0
        
        print(f"\n=== INICIANDO ANÁLISIS DE BIPARTICIÓN ÓPTIMA ===")
        
        # PASO 1: Analizar caso especial 000→111
        print(f"PASO 1: Analizando caso especial 000→111...")
        particiones_especiales, particion_optima_especial = self._analizar_caso_especial_todos_unos(n_vars, n_bits, estado_inicial)
        
        # PASO 2: Analizar transiciones con solo un cero
        print(f"PASO 2: Analizando transiciones con solo un cero...")
        particiones_un_cero, particion_optima_un_cero = self._analizar_transiciones_solo_un_cero(
            n_vars, n_bits, estado_inicial
        )
        
        # PASO 3: NUEVO - Comparar casos especiales y decidir si retornar
        print(f"\nPASO 3: COMPARANDO CASOS ESPECIALES...")
        
        # Combinar todas las particiones de casos especiales
        todas_particiones_especiales = particiones_especiales + particiones_un_cero
        
        if todas_particiones_especiales:
            # Ordenar por EMD ascendente
            todas_particiones_especiales.sort(key=lambda x: x['emd'])
            mejor_caso_especial = todas_particiones_especiales[0]
            
            print(f"Mejor caso especial encontrado:")
            print(f"  - Transición: {mejor_caso_especial['transicion_original_bin']}")
            print(f"  - EMD (pérdida): {mejor_caso_especial['emd']:.6f}")
            print(f"  - Tipo: {mejor_caso_especial['info_debug'].get('caso_especial', 'N/A')}")
            
            # Si la pérdida es menor a 1e-2, considerarla como cero y retornar
            if mejor_caso_especial['emd'] < 1e-2:
                print(f"  - EMD < 1e-2 ({mejor_caso_especial['emd']:.6f}), considerando como cero y retornando solución óptima")
                
                conjunto_presente = mejor_caso_especial['conjunto_presente']
                conjunto_futuro = mejor_caso_especial['conjunto_futuro']
                
                # Log de la solución seleccionada
                vars_presente = [chr(65 + i) for i in sorted(conjunto_presente)] if conjunto_presente else []
                vars_futuro = [chr(65 + i) for i in sorted(conjunto_futuro)] if conjunto_futuro else []
                
                print(f"  SOLUCIÓN SELECCIONADA:")
                print(f"    Presente (mecanismo): {vars_presente}")
                print(f"    Futuro (alcance): {vars_futuro}")
                print(f"    Pérdida final: 0.0 (aproximada desde {mejor_caso_especial['emd']:.6f})")
                
                return (conjunto_presente, conjunto_futuro)
            else:
                print(f"  - EMD >= 1e-2 ({mejor_caso_especial['emd']:.6f}), continuando con análisis completo")
        else:
            print(f"  - No se encontraron casos especiales válidos, continuando con análisis completo")
        
        # PASO 4: Si llegamos aquí, verificar si ya teníamos soluciones óptimas individuales
        if particion_optima_especial is not None:
            print(f"PASO 4: Retornando solución óptima del caso especial 000→111")
            return particion_optima_especial
        
        if particion_optima_un_cero is not None:
            print(f"PASO 4: Retornando solución óptima de transiciones con un cero")
            return particion_optima_un_cero
        
        # PASO 5: Analizar transiciones estándar (código optimizado con filtro Hamming)
        print(f"PASO 5: Analizando transiciones estándar con filtro Hamming adaptativo...")
        particiones_estandar, particion_optima_estandar = self._analizar_transiciones_estandar(n_vars, n_bits, estado_inicial)
        
        if particion_optima_estandar is not None:
            print("PASO 5: Solución óptima encontrada en transiciones estándar")
            return particion_optima_estandar
        
        # PASO 6: Combinar TODAS las particiones candidatas
        print(f"PASO 6: Combinando todas las particiones candidatas...")
        particiones_candidatas = todas_particiones_especiales + particiones_estandar
        
        if particiones_candidatas:
            # Ordenar por EMD ascendente y tomar la mejor
            particiones_candidatas.sort(key=lambda x: x['emd'])
            mejor_particion = particiones_candidatas[0]
            
            print(f"MEJOR PARTICIÓN GLOBAL:")
            print(f"  - EMD: {mejor_particion['emd']:.6f}")
            print(f"  - Basada en: {mejor_particion['transicion_original_bin']}")
            print(f"  - Caso especial: {mejor_particion['info_debug']['caso_especial']}")
            
            return (mejor_particion['conjunto_presente'], mejor_particion['conjunto_futuro'])
        else:
            print("ERROR: No se encontraron particiones candidatas válidas")
            self.logger.error("No se encontraron particiones candidatas válidas")
            return None, None

    # ===== FUNCIONES DE OPTIMIZACIÓN HEURÍSTICA CON FILTRO HAMMING =====
    
    def _determinar_distancia_hamming_adaptativa(self, n_bits: int) -> int:
        """
        Calcula la distancia Hamming máxima adaptativa usando la fórmula:
        distancia_max = max(2, min(n_bits//3, 4))
        
        Args:
            n_bits: Número total de bits del sistema
            
        Returns:
            int: Distancia Hamming máxima a evaluar
            
        Examples:
            - n_bits=3  → max(2, min(1, 4)) = max(2, 1) = 2
            - n_bits=6  → max(2, min(2, 4)) = max(2, 2) = 2  
            - n_bits=9  → max(2, min(3, 4)) = max(2, 3) = 3
            - n_bits=15 → max(2, min(5, 4)) = max(2, 4) = 4
        """
        distancia_max = max(2, min(n_bits // 3, 4))
        return distancia_max

    def _obtener_estados_filtrados_por_hamming(self, n_bits: int, distancia_max: int) -> List[int]:
        """
        Genera lista de estados canónicos filtrados por distancia Hamming máxima.
        
        CRITERIOS DE FILTRADO:
        1. Distancia Hamming desde estado_inicial (0) <= distancia_max
        2. Estado debe ser canónico (usar _es_estado_canonico existente)
        3. Excluir casos especiales ya manejados:
           - Estado todos unos (2^n_bits - 1)
           - Estados con solo un cero (bin(estado).count('0') == 1)
           - Estados con solo un uno (bin(estado).count('1') == 1)
        
        Args:
            n_bits: Número de bits del sistema
            distancia_max: Distancia Hamming máxima permitida
            
        Returns:
            List[int]: Estados válidos ordenados por distancia Hamming ascendente
        """
        estados_filtrados = []
        estado_inicial = 0
        estado_todos_unos = 2**n_bits - 1
        
        # Crear lista de candidatos con sus distancias Hamming
        candidatos_con_distancia = []
        
        for estado in range(1, 2**n_bits):
            # Excluir casos especiales ya manejados
            if estado == estado_todos_unos:
                continue
            if bin(estado).count('0') == 1:  # Solo un cero
                continue
            if bin(estado).count('1') == 1:  # Solo un uno
                continue
            
            # Verificar distancia Hamming
            distancia = self._calcular_distancia_hamming(estado_inicial, estado)
            if distancia <= distancia_max:
                # Solo incluir estados canónicos
                if self._es_estado_canonico(estado, n_bits):
                    candidatos_con_distancia.append((estado, distancia))
        
        # Ordenar por distancia Hamming ascendente
        candidatos_con_distancia.sort(key=lambda x: x[1])
        estados_filtrados = [estado for estado, _ in candidatos_con_distancia]
        
        return estados_filtrados

    def _validar_filtro_hamming(self, estados_filtrados: List[int], n_bits: int, distancia_max: int):
        """
        Valida que el filtro Hamming funcione correctamente.
        
        VERIFICACIONES:
        1. Todos los estados están dentro de la distancia máxima
        2. No hay estados duplicados
        3. Se mantienen estados canónicos válidos
        4. Se excluyen correctamente los casos especiales
        
        Args:
            estados_filtrados: Lista de estados filtrados
            n_bits: Número de bits del sistema
            distancia_max: Distancia Hamming máxima aplicada
        """
        estado_inicial = 0
        estado_todos_unos = 2**n_bits - 1
        
        print(f"\n=== VALIDACIÓN DEL FILTRO HAMMING ===")
        
        # Verificación 1: Distancias dentro del límite
        distancias_invalidas = 0
        for estado in estados_filtrados:
            distancia = self._calcular_distancia_hamming(estado_inicial, estado)
            if distancia > distancia_max:
                distancias_invalidas += 1
        
        print(f"Estados con distancia > {distancia_max}: {distancias_invalidas}")
        
        # Verificación 2: Estados únicos
        duplicados = len(estados_filtrados) - len(set(estados_filtrados))
        print(f"Estados duplicados encontrados: {duplicados}")
        
        # Verificación 3: Casos especiales excluidos correctamente
        casos_especiales_incluidos = 0
        for estado in estados_filtrados:
            if estado == estado_todos_unos:
                casos_especiales_incluidos += 1
            elif bin(estado).count('0') == 1:
                casos_especiales_incluidos += 1
            elif bin(estado).count('1') == 1:
                casos_especiales_incluidos += 1
        
        print(f"Casos especiales incorrectamente incluidos: {casos_especiales_incluidos}")
        
        # Verificación 4: Estados canónicos
        no_canonicos = 0
        for estado in estados_filtrados:
            if not self._es_estado_canonico(estado, n_bits):
                no_canonicos += 1
        
        print(f"Estados no canónicos incluidos: {no_canonicos}")
        
        # Resultado de validación
        errores_totales = distancias_invalidas + duplicados + casos_especiales_incluidos + no_canonicos
        if errores_totales == 0:
            print(" VALIDACIÓN EXITOSA: El filtro Hamming funciona correctamente")
        else:
            print(f" VALIDACIÓN FALLIDA: {errores_totales} errores encontrados")
        
        print(f"=== FIN VALIDACIÓN ===")

    def _es_estado_canonico(self, estado: int, n_bits: int) -> bool:
        """
        Determina si un estado es canónico (representante de su clase de equivalencia).
        Criterio: el estado con menor valor decimal entre él y su complemento.
        
        Args:
            estado: Estado a evaluar
            n_bits: Número de bits del sistema
            
        Returns:
            bool: True si el estado es canónico
        """
        complemento = self._calcular_complemento_estado(0, estado, n_bits)
        return estado <= complemento

    def _log_estadisticas_optimizacion(self, n_bits: int, distancia_max_aplicada: int, 
                                     estados_filtrados_hamming: List[int]):
        """
        Registra estadísticas de la optimización aplicada, incluyendo filtro Hamming.
        
        Args:
            n_bits: Número de bits del sistema
            distancia_max_aplicada: Distancia Hamming máxima usada
            estados_filtrados_hamming: Lista de estados tras filtro Hamming
        """
        total_estados = 2**n_bits - 1
        estados_tras_hamming = len(estados_filtrados_hamming)
        
        # Calcular estados excluidos por categoría
        estados_excluidos_todos_unos = 1
        estados_excluidos_solo_un_cero = n_bits
        estados_excluidos_solo_un_uno = n_bits
        
        # Obtener estados canónicos sin filtro Hamming para comparación
        estados_canonicos_sin_filtro = self._obtener_estados_canonicos_sin_filtro(n_bits)
        estados_canonicos_sin_hamming = len(estados_canonicos_sin_filtro)
        
        estados_excluidos_no_canonicos = total_estados - estados_excluidos_todos_unos - estados_excluidos_solo_un_cero - estados_excluidos_solo_un_uno - estados_canonicos_sin_hamming
        estados_finales_evaluados = estados_tras_hamming
        
        # Calcular reducciones
        reduccion_hamming = (1 - estados_tras_hamming / estados_canonicos_sin_hamming) * 100 if estados_canonicos_sin_hamming > 0 else 0
        reduccion_total = (1 - estados_finales_evaluados / total_estados) * 100
        
        print(f"=== ESTADÍSTICAS DE OPTIMIZACIÓN HEURÍSTICA ===")
        print(f"Distancia Hamming máxima aplicada: {distancia_max_aplicada}")
        print(f"Total de estados posibles: {total_estados}")
        print(f"Estados tras filtro Hamming: {estados_tras_hamming}")
        print(f"Estados excluidos (caso especial 000→111): {estados_excluidos_todos_unos}")
        print(f"Estados excluidos (solo un cero): {estados_excluidos_solo_un_cero}")
        print(f"Estados excluidos (solo un uno): {estados_excluidos_solo_un_uno}")
        print(f"Estados excluidos (no canónicos): {estados_excluidos_no_canonicos}")
        print(f"Estados finales evaluados: {estados_finales_evaluados}")
        print(f"Reducción por filtro Hamming: {reduccion_hamming:.1f}%")
        print(f"Reducción total del espacio muestral: {reduccion_total:.1f}%")
        print(f"=== FIN ESTADÍSTICAS OPTIMIZACIÓN ===")

    def _obtener_estados_canonicos_sin_filtro(self, n_bits: int) -> List[int]:
        """
        Genera estados canónicos SIN filtro Hamming para comparación estadística.
        
        Args:
            n_bits: Número de bits del sistema
            
        Returns:
            List[int]: Lista de estados canónicos sin filtro Hamming
        """
        estados_canonicos = []
        estado_todos_unos = 2**n_bits - 1
        
        for estado in range(1, 2**n_bits):
            # Excluir casos especiales ya manejados
            if estado == estado_todos_unos:
                continue
            if bin(estado).count('0') == 1:  # Solo un cero
                continue
            if bin(estado).count('1') == 1:  # Solo un uno
                continue
                
            # Solo incluir estados canónicos
            if self._es_estado_canonico(estado, n_bits):
                estados_canonicos.append(estado)
        
        return estados_canonicos

    def _analizar_transiciones_estandar(self, n_vars, n_bits, estado_inicial) -> Tuple[List[dict], Optional[Tuple[Set[int], Set[int]]]]:
        """
        VERSIÓN OPTIMIZADA CON FILTRO HAMMING: Analiza las transiciones estándar usando 
        heurísticas para reducir el espacio muestral. Aplica filtro de distancia Hamming adaptativa.
        
        Args:
            n_vars: Número total de variables
            n_bits: Número de bits del sistema
            estado_inicial: Estado inicial (normalmente 0 = 000)
            
        Returns:
            Tuple con (lista de particiones candidatas, partición óptima si EMD=0)
        """
        particiones_candidatas = []
        
        print(f"\nPASO 5: ANÁLISIS OPTIMIZADO DE TRANSICIONES ESTÁNDAR CON FILTRO HAMMING")
        
        # PASO 1: Obtener estados filtrados por distancia Hamming adaptativa
        distancia_max = self._determinar_distancia_hamming_adaptativa(n_bits)
        estados_filtrados = self._obtener_estados_filtrados_por_hamming(n_bits, distancia_max)
        
        # Logging del filtro Hamming
        print(f"\n=== APLICANDO FILTRO HAMMING ADAPTATIVO ===")
        print(f"Número de bits: {n_bits}")
        print(f"Fórmula aplicada: max(2, min({n_bits}//3, 4))")
        print(f"Distancia Hamming máxima: {distancia_max}")
        print(f"Estados antes del filtro: {2**n_bits - 1}")
        print(f"Estados después del filtro: {len(estados_filtrados)}")
        
        if len(estados_filtrados) > 0:
            reduccion_inicial = (1 - len(estados_filtrados) / (2**n_bits - 1)) * 100
            print(f"Reducción inicial: {reduccion_inicial:.1f}%")
        
        # Log de estadísticas de optimización
        self._log_estadisticas_optimizacion(n_bits, distancia_max, estados_filtrados)
        
        # Validar filtro Hamming
        self._validar_filtro_hamming(estados_filtrados, n_bits, distancia_max)
        
        if not estados_filtrados:
            print("No hay estados válidos tras aplicar filtro Hamming para transiciones estándar")
            return particiones_candidatas, None
        
        # PASO 2: Obtener variables disponibles
        variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
        variables_disponibles_mecanismo = set(self.sia_subsistema.dims_ncubos)
        
        # PASO 3: Evaluar solo estados filtrados (ESPACIO ULTRA-REDUCIDO)
        promedios_transiciones = []
        
        print(f"\nEvaluando {len(estados_filtrados)} estados filtrados por Hamming...")
        
        for estado_destino in estados_filtrados:
            estado_destino_bin = self._decimal_to_little_endian_binary(estado_destino, n_bits)
            distancia_estado = self._calcular_distancia_hamming(estado_inicial, estado_destino)
            
            costos_alcance = []
            suma_costos = 0.0
            
            # Calcular costos bajo demanda solo para las variables necesarias
            for var_idx in sorted(variables_disponibles_alcance):
                costo = self._obtener_costo_transicion(var_idx, estado_inicial, estado_destino)
                var_letra = chr(65 + var_idx)
                
                if costo < float('inf'):
                    costos_alcance.append((var_idx, costo, var_letra))
                    suma_costos += costo
            
            if costos_alcance:
                promedio_transicion = suma_costos / len(costos_alcance)
                promedios_transiciones.append({
                    'estado_destino': estado_destino,
                    'estado_destino_bin': estado_destino_bin,
                    'distancia_hamming': distancia_estado,
                    'costos_alcance': costos_alcance,
                    'promedio': promedio_transicion
                })
                
                print(f"  Estado {estado_destino_bin} (d={distancia_estado}): promedio = {promedio_transicion:.4f}")

        # PASO 4: Calcular promedio global y filtrar candidatas
        if promedios_transiciones:
            suma_promedios = sum(t['promedio'] for t in promedios_transiciones)
            promedio_global = suma_promedios / len(promedios_transiciones)
            
            print(f"\nPromedio global de transiciones estándar filtradas: {promedio_global:.4f}")
            
            transiciones_candidatas = [t for t in promedios_transiciones if t['promedio'] <= promedio_global]
            print(f"Transiciones candidatas filtradas: {len(transiciones_candidatas)}/{len(promedios_transiciones)}")
        else:
            print("No se encontraron transiciones válidas en estados filtrados por Hamming")
            return particiones_candidatas, None

        # PASO 5: Procesar transiciones candidatas (LÓGICA ORIGINAL PRESERVADA)
        for transicion in transiciones_candidatas:
            estado_destino = transicion['estado_destino']
            estado_destino_bin = transicion['estado_destino_bin']
            distancia_hamming = transicion['distancia_hamming']
            costos_alcance = transicion['costos_alcance']
            
            print(f"\n--- Procesando candidata estándar: t(000, {estado_destino_bin}) [d={distancia_hamming}] ---")
            
            # Calcular estado complementario y sus costos bajo demanda
            estado_complementario = self._calcular_complemento_estado(estado_inicial, estado_destino, n_bits)
            estado_complementario_bin = self._decimal_to_little_endian_binary(estado_complementario, n_bits)
            
            costos_complementario = []
            
            # Calcular costos del complementario solo para las variables ya evaluadas
            for var_idx, costo_original, var_letra in costos_alcance:
                costo_comp = self._obtener_costo_transicion(var_idx, estado_inicial, estado_complementario)
                costos_complementario.append((var_idx, costo_comp, var_letra))
                print(f"  Variable {var_letra}: Original={costo_original:.4f}, Complementario={costo_comp:.4f}")
            
            # Determinar partición comparando costos
            conjunto_presente = set()
            conjunto_futuro = set()
            
            for i, (var_idx, costo_original, var_letra) in enumerate(costos_alcance):
                _, costo_comp, _ = costos_complementario[i]
                
                if costo_original < costo_comp:
                    if var_idx in variables_disponibles_mecanismo:
                        conjunto_presente.add(var_idx)
                        print(f"    {var_letra}: Original mejor → PRESENTE")
                    else:
                        conjunto_futuro.add(var_idx)
                        print(f"    {var_letra}: Original mejor, no en mecanismo → FUTURO")
                else:
                    conjunto_futuro.add(var_idx)
                    print(f"    {var_letra}: Complementario mejor → FUTURO")
            
            # Identificar bits que se mantienen
            bits_que_se_mantienen = self._identificar_bits_que_se_mantienen(estado_inicial, estado_destino, n_bits)
            
            # Calcular EMD
            emd_valor = self._calcular_emd_biparticion_completa(bits_que_se_mantienen, conjunto_presente)
            print(f"  EMD calculado: {emd_valor:.6f}")
                
            if emd_valor < float('inf'):
                particiones_candidatas.append({
                    'transicion_original': (estado_inicial, estado_destino),
                    'transicion_original_bin': f"t(000, {estado_destino_bin})",
                    'promedio_original': transicion['promedio'],
                    'distancia_hamming': distancia_hamming,
                    'conjunto_presente': bits_que_se_mantienen,
                    'conjunto_futuro': conjunto_presente,
                    'emd': emd_valor,
                    'info_debug': {
                        'costos_originales': [costo for _, costo, _ in costos_alcance],
                        'costos_complemento': [costo for _, costo, _ in costos_complementario],
                        'bits_que_se_mantienen': bits_que_se_mantienen,
                        'distancia_hamming_aplicada': distancia_max,
                        'caso_especial': 'estandar_hamming_optimizado'
                    }
                })
                    
                if emd_valor < 1e-4:
                    print(f"  ¡EMD perfecto encontrado en transición estándar filtrada! Retornando solución óptima.")
                    return particiones_candidatas, (bits_que_se_mantienen, conjunto_presente)
        
        # Log de estadísticas finales de caché
        print(f"\n=== ESTADÍSTICAS DE CACHÉ DESPUÉS DE OPTIMIZACIÓN HAMMING ===")
        print(f"Costos de transición calculados: {len(self.cache_costos)}")
        print(f"Distancias Hamming calculadas: {len(self.cache_hamming)}")
        print(f"Distancia Hamming máxima aplicada: {distancia_max}")
        print(f"Estados evaluados tras filtro: {len(estados_filtrados)}")
        
        return particiones_candidatas, None
    
    def _analizar_caso_especial_todos_unos(self, n_vars, n_bits, estado_inicial):
        """
        Analiza específicamente la transición desde el estado todos ceros (000) 
        al estado todos unos (111).
        OPTIMIZADO: Evalúa todas las variables disponibles en el alcance.
        """
        particiones_candidatas = []
        estado_todos_unos = 2**n_bits - 1  # Por ejemplo, 111 para 3 bits
        
        print(f"\n=== CASO ESPECIAL: EVALUANDO TRANSICIÓN 000→111 ===")
        
        # PASO 1: Obtener variables disponibles en el alcance
        variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
        print(f"Variables disponibles para alcance: {[chr(65 + i) for i in sorted(variables_disponibles_alcance)]}")
        
        # PASO 2: Calcular costos directos SOLO para variables disponibles en el alcance
        variables_validas = []
        
        # Solo evaluar las variables que están disponibles en el alcance
        for var_idx in sorted(variables_disponibles_alcance):
            costo = self._obtener_costo_transicion(var_idx, estado_inicial, estado_todos_unos)
            var_letra = chr(65 + var_idx)
            
            if costo < float('inf'):
                variables_validas.append((var_idx, costo, var_letra))
                print(f"Variable {var_letra}: costo directo = {costo:.4f}")
            else:
                print(f"Variable {var_letra}: costo infinito (descartada)")

        # PASO 3: Verificar si hay variables válidas
        if not variables_validas:
            print("No se encontraron variables disponibles con costos válidos")
            return particiones_candidatas, None
        
        print(f"Evaluando todas las {len(variables_validas)} variables disponibles con costos válidos")
        
        # PASO 4: Ordenar y evaluar todas las variables válidas
        variables_validas.sort(key=lambda x: x[1])  # Ordenar por costo ascendente
        
        for var_idx, costo, var_letra in variables_validas:
            # Crear bipartición: variable en futuro (alcance) y nada en presente (mecanismo)
            conjunto_futuro = {var_idx}
            conjunto_presente = set()  # Mecanismo vacío
            
            print(f"Evaluando variable {var_letra} como único elemento en FUTURO, mecanismo vacío")
            print(f"  Presente (mecanismo): []")
            print(f"  Futuro (alcance): [{var_letra}]")
            
            # Calcular EMD para esta bipartición
            emd_valor = self._calcular_emd_biparticion_completa(conjunto_presente, conjunto_futuro)
            
            # Evaluar resultado
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
                
                # Si EMD es prácticamente cero, encontramos solución óptima
                if emd_valor < 1e-3:
                    print(f"  ¡EMD perfecto (0) con variable {var_letra}! Retornando solución óptima.")
                    return particiones_candidatas, (conjunto_presente, conjunto_futuro)
            else:
                print(f"  EMD inválido para variable {var_letra}, partición descartada")

        # Si llegamos aquí, no encontramos EMD=0
        return particiones_candidatas, None
    
    def _analizar_transiciones_solo_un_cero(self, n_vars, n_bits, estado_inicial) -> Tuple[List[dict], Optional[Tuple[Set[int], Set[int]]]]:
            """
            Analiza las transiciones desde el estado inicial (000) hasta estados que tienen solo un cero.
            Por ejemplo, para 3 bits: 000→011, 000→101, 000→110
            SIMILAR AL MÉTODO ESTÁNDAR pero solo evalúa estados con un cero.
            
            Args:
                n_vars: Número total de variables
                n_bits: Número de bits del sistema
                estado_inicial: Estado inicial (normalmente 0 = 000)
                
            Returns:
                Tuple con (lista de particiones candidatas, partición óptima si EMD=0)
            """
            particiones_candidatas = []
            
            print(f"\n=== ANALIZANDO TRANSICIONES CON SOLO UN CERO ===")
            
            # PASO 1: Generar todos los estados que tienen exactamente un cero
            estados_un_cero = []
            estado_todos_unos = 2**n_bits - 1  # 111 para 3 bits
            
            for bit_pos in range(n_bits):
                # Crear estado con un cero en la posición bit_pos
                estado = estado_todos_unos ^ (1 << bit_pos)  # Voltear el bit en esa posición
                estado_bin = self._decimal_to_little_endian_binary(estado, n_bits)
                estados_un_cero.append((estado, estado_bin))
                print(f"Estado con cero en bit {bit_pos}: {estado} → {estado_bin}")
            
            # PASO 2: Obtener variables disponibles (CRÍTICO)
            variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
            variables_disponibles_mecanismo = set(self.sia_subsistema.dims_ncubos)
            
            print(f"Variables disponibles para alcance: {[chr(65 + i) for i in sorted(variables_disponibles_alcance)]}")
            print(f"Variables disponibles para mecanismo: {[chr(65 + i) for i in sorted(variables_disponibles_mecanismo)]}")
            
            print(variables_disponibles_mecanismo)
            
            # PASO 3: Calcular promedios de transiciones para filtrado
            promedios_transiciones = []
            
            for estado_destino, estado_destino_bin in estados_un_cero:
                print(f"\nEvaluando transición 000 → {estado_destino_bin}")
                
                costos_alcance = []
                suma_costos = 0.0
                
                # Calcular costos SOLO para variables disponibles en el alcance
                for var_idx in sorted(variables_disponibles_alcance):
                    costo = self._obtener_costo_transicion(var_idx, estado_inicial, estado_destino)
                    var_letra = chr(65 + var_idx)
                    
                    if costo < float('inf'):
                        costos_alcance.append((var_idx, costo, var_letra))
                        suma_costos += costo
                        print(f"  Variable {var_letra}: costo = {costo:.6f}")
                    else:
                        print(f"  Variable {var_letra}: costo infinito (descartada)")
                
                if costos_alcance:
                    promedio_transicion = suma_costos / len(costos_alcance)
                    promedios_transiciones.append({
                        'estado_destino': estado_destino,
                        'estado_destino_bin': estado_destino_bin,
                        'costos_alcance': costos_alcance,
                        'promedio': promedio_transicion
                    })
                    print(f"  Promedio de costos: {promedio_transicion:.6f}")
                else:
                    print(f"  No hay variables válidas para esta transición")
            
            # PASO 4: Filtrar transiciones candidatas por promedio global
            if promedios_transiciones:
                suma_promedios = sum(t['promedio'] for t in promedios_transiciones)
                promedio_global = suma_promedios / len(promedios_transiciones)
                print(f"\nPromedio global de todas las transiciones: {promedio_global:.6f}")
                
                transiciones_candidatas = [t for t in promedios_transiciones if t['promedio'] <= promedio_global]
                print(f"Transiciones candidatas (promedio <= global): {len(transiciones_candidatas)}")
            else:
                print("No se encontraron transiciones válidas")
                return particiones_candidatas, None
            
            # PASO 5: Procesar cada transición candidata
            for transicion in transiciones_candidatas:
                estado_destino = transicion['estado_destino']
                estado_destino_bin = transicion['estado_destino_bin']
                costos_alcance = transicion['costos_alcance']
                
                print(f"\nProcesando transición candidata: 000 → {estado_destino_bin}")
                
                # Calcular estado complementario y sus costos
                estado_complementario = self._calcular_complemento_estado(estado_inicial, estado_destino, n_bits)
                estado_complementario_bin = self._decimal_to_little_endian_binary(estado_complementario, n_bits)
                print(f"Estado complementario calculado: {estado_complementario_bin}")
                
                costos_complementario = []
                
                # Calcular costos del complementario para las mismas variables
                for var_idx, costo_original, var_letra in costos_alcance:
                    costo_comp = self._obtener_costo_transicion(var_idx, estado_inicial, estado_complementario)
                    costos_complementario.append((var_idx, costo_comp, var_letra))
                    print(f"  Variable {var_letra}: original={costo_original:.6f}, complemento={costo_comp:.6f}")
                
                # PASO 6: Asignar variables a conjuntos según costos y disponibilidad
                conjunto_presente = set()  # Variables para el mecanismo
                conjunto_futuro = set()    # Variables para el alcance
                
                for i, (var_idx, costo_original, var_letra) in enumerate(costos_alcance):
                    _, costo_comp, _ = costos_complementario[i]
                    
                    # Decisión basada en el menor costo
                    if costo_original < costo_comp:
                        # Preferir el costo original
                        if var_idx in variables_disponibles_mecanismo:
                            conjunto_presente.add(var_idx)
                            print(f"    {var_letra} → PRESENTE (mecanismo) - costo original menor")
                        else:
                            conjunto_futuro.add(var_idx)
                            print(f"    {var_letra} → FUTURO (alcance) - costo original menor, pero no disponible en mecanismo")
                    else:
                        # Preferir el complemento (va al futuro)
                        conjunto_futuro.add(var_idx)
                        print(f"    {var_letra} → FUTURO (alcance) - costo complemento menor")
                
                # PASO 7: Identificar bits que se mantienen para el cálculo de EMD
                print(f"Identificando bits que se mantienen entre {estado_inicial} y {estado_destino}...")

                bits_que_se_mantienen = self._identificar_bits_que_se_mantienen(estado_inicial, estado_destino, n_bits)
                print(f"Bits que se mantienen: {bits_que_se_mantienen}")
                print(f"Conjunto presente (mecanismo): {[chr(65 + i) for i in sorted(conjunto_presente)]}")
                
                # PASO 8: Calcular EMD para esta bipartición
                # IMPORTANTE: Usar la lógica correcta según la estructura del sistema
                emd_valor = self._calcular_emd_biparticion_completa(bits_que_se_mantienen, conjunto_presente)
                
                print(f"EMD calculado: {emd_valor:.6f}")
                
                # PASO 9: Agregar a candidatas si es válido
                if emd_valor < float('inf'):
                    particiones_candidatas.append({
                        'transicion_original': (estado_inicial, estado_destino),
                        'transicion_original_bin': f"t(000, {estado_destino_bin})",
                        'promedio_original': transicion['promedio'],
                        'conjunto_presente': bits_que_se_mantienen,  # Para consistencia con la estructura
                        'conjunto_futuro': conjunto_presente,        # Variables asignadas al mecanismo
                        'emd': emd_valor,
                        'info_debug': {
                            'costos_originales': [costo for _, costo, _ in costos_alcance],
                            'costos_complemento': [costo for _, costo, _ in costos_complementario],
                            'bits_que_se_mantienen': bits_que_se_mantienen,
                            'caso_especial': 'transicion_un_cero'
                        }
                    })
                    
                    # Si encontramos EMD prácticamente cero, retornar inmediatamente
                    if emd_valor < 1e-4:
                        print(f"¡EMD óptimo encontrado ({emd_valor:.6f})! Retornando solución.")
                        return particiones_candidatas, (bits_que_se_mantienen, conjunto_presente)
                else:
                    print(f"EMD inválido, partición descartada")
            
            # PASO 10: Log de estadísticas
            print(f"\nESTADÍSTICAS FINALES:")
            print(f"- Particiones candidatas encontradas: {len(particiones_candidatas)}")
            print(f"- Costos calculados en caché: {len(self.cache_costos)}")
            print(f"- Distancias Hamming en caché: {len(self.cache_hamming)}")
            
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
        CORRIGIDO: Usar la lógica correcta de alcance/mecanismo según el método bipartir.
        
        Args:
            conjunto_presente: Índices de variables que representan el presente
            conjunto_futuro: Índices de variables que representan el futuro
            
        Returns:
            float: Valor EMD calculado
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
            estrategia="Geometric-SIA-Secuencial",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_particion,
        )


