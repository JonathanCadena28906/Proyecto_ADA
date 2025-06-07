import time
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Set, Optional

from src.middlewares.slogger import SafeLogger
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_biparte_q
from src.constants.base import EFECTO, ACTUAL


class GeometricSIA(SIA):
    """
    Implementación del enfoque geométrico para el análisis SIA basado en hipercubos n-dimensionales.
    
    Esta clase utiliza representaciones geométricas del sistema como hipercubos donde cada vértice
    representa un estado posible. La función de costo se basa en la distancia de Hamming y 
    exploración recursiva del hipercubo para encontrar biparticiones óptimas.
    
    Attributes:
        tabla_transiciones: Diccionario que almacena los costos de transición entre estados
        cache_hamming: Cache para distancias de Hamming calculadas
        cache_costos: Cache para costos calculados recursivamente
        memoria_biparticiones: Almacena las biparticiones evaluadas y sus costos
        tensores: Representación tensorial del subsistema
        logger: Logger para el proceso geométrico
    """
    
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.tabla_transiciones: Dict[Tuple[int, int, int], float] = {}
        self.cache_hamming: Dict[Tuple[int, int], int] = {}
        self.cache_costos: Dict[Tuple[int, int, int], float] = {}
        self.memoria_biparticiones: Dict[Tuple, Tuple[float, np.ndarray]] = {}
        self.tensores: List[np.ndarray] = []
        self.logger = SafeLogger("geometric_strategy")
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        """
        Implementa el algoritmo geométrico para encontrar la bipartición óptima
        utilizando el enfoque topológico basado en hipercubos.
        
        Args:
            condiciones: String binario para condiciones de fondo
            alcance: String binario para variables de alcance  
            mecanismo: String binario para variables de mecanismo
            
        Returns:
            Solution: Objeto con la solución encontrada
        """
        # Preparar el subsistema
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        
        # 1. Construir la representación n-dimensional del sistema
        self._construir_representacion_ndimensional()
        
        # 2. Calcular la tabla de costos (T) para cada variable
        self._calcular_tabla_costos()
        
        # 3. Identificar la bipartición óptima usando análisis de transiciones desde estado inicial
        biparticion_optima = self._identificar_biparticion_optima()
        
        # 4. Formatear y retornar el resultado
        return self._formatear_resultado(biparticion_optima)
    
    def _decimal_to_little_endian_binary(self, decimal: int, n_bits: int) -> str:
        """
        Convierte un número decimal a su representación binaria little-endian.
        En little-endian, el bit menos significativo está a la izquierda.
        
        Args:
            decimal: Número decimal a convertir
            n_bits: Número de bits para la representación
            
        Returns:
            str: Representación binaria en formato little-endian
        """
        # Convertir a binario big-endian tradicional
        big_endian = format(decimal, f'0{n_bits}b')
        # Invertir para obtener little-endian
        little_endian = big_endian[::-1]
        return little_endian
    
    def _little_endian_to_decimal(self, little_endian_str: str) -> int:
        """
        Convierte una cadena binaria little-endian a decimal.
        
        Args:
            little_endian_str: Cadena binaria en formato little-endian
            
        Returns:
            int: Valor decimal correspondiente
        """
        # Invertir la cadena para obtener big-endian y convertir
        big_endian = little_endian_str[::-1]
        return int(big_endian, 2)
    
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
        
        # Para cada bit, alternar su valor para obtener vecinos
        for bit_pos in range(n_bits):
            vecino = estado ^ (1 << bit_pos)
            vecinos.append(vecino)
            
        return vecinos
    
    def _construir_representacion_ndimensional(self):
        """
        Construye la representación n-dimensional del sistema como tensores.
        Cada tensor representa las probabilidades condicionales de una variable.
        IMPORTANTE: Reordena los datos considerando el formato little-endian.
        """
        self.tensores = []
        n_bits = len(self.sia_subsistema.dims_ncubos)
        
        # Descomponer en tensores elementales
        for i, ncube in enumerate(self.sia_subsistema.ncubos):
            # Extraer el tensor de probabilidades condicionales
            tensor_original = ncube.data.copy()

            
            # Reordenar tensor para formato little-endian
            tensor_reordenado = self._reordenar_tensor_little_endian(tensor_original, n_bits)
            self.tensores.append(tensor_reordenado)
            
        
    def _reordenar_tensor_little_endian(self, tensor_original: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Reordena un tensor de formato big-endian a little-endian.
        
        Args:
            tensor_original: Tensor en formato big-endian (orden tradicional)
            n_bits: Número de bits del sistema
            
        Returns:
            np.ndarray: Tensor reordenado en formato little-endian
        """
        # Crear tensor de salida con la misma forma
        tensor_reordenado = np.zeros_like(tensor_original)
        
        # Crear mapeo de índices big-endian a little-endian
        n_estados = 2 ** n_bits
        
        # Iterar sobre todas las posibles coordenadas
        for estado_decimal in range(n_estados):
            # Obtener coordenadas big-endian (tradicionales)
            coords_big_endian = [(estado_decimal >> bit) & 1 for bit in range(n_bits)]
            
            # Convertir a little-endian: invertir el orden de los bits
            coords_little_endian = coords_big_endian[::-1]
            
            # Convertir coordenadas little-endian de vuelta a decimal para indexar
            estado_little_decimal = sum(bit * (2 ** pos) for pos, bit in enumerate(coords_little_endian))
            coords_little_indexing = [(estado_little_decimal >> bit) & 1 for bit in range(n_bits)]
            
            # Si el tensor tiene múltiples dimensiones, iterar sobre ellas
            if tensor_original.ndim == n_bits:
                # Tensor simple n-dimensional
                tensor_reordenado[tuple(coords_big_endian)] = tensor_original[tuple(coords_little_indexing)]
            else:
                # Tensor con dimensiones adicionales - copiar todos los valores
                tensor_reordenado[tuple(coords_big_endian)] = tensor_original[tuple(coords_little_indexing)]
        
        return tensor_reordenado
    
    def _calcular_distancia_hamming(self, estado_i: int, estado_j: int, n_bits: int) -> int:
        """
        Calcula la distancia de Hamming entre dos estados.
        
        Args:
            estado_i: Estado inicial
            estado_j: Estado final
            n_bits: Número de bits del sistema
            
        Returns:
            int: Distancia de Hamming
        """
        clave = (estado_i, estado_j)
        if clave in self.cache_hamming:
            return self.cache_hamming[clave]
        
        # XOR y contar bits activados
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
        # Verificar cache
        clave_cache = (estado_i, estado_j, tensor_idx)
        if clave_cache in self.cache_costos:
            return self.cache_costos[clave_cache]
        
        # Calcular distancia de Hamming
        distancia = self._calcular_distancia_hamming(estado_i, estado_j, n_bits)
        
        # Factor de decrecimiento exponencial
        gamma = 2 ** (-distancia)
        
        # Convertir estados a coordenadas para acceder al tensor
        coords_i = [(estado_i >> bit) & 1 for bit in range(n_bits)]
        coords_j = [(estado_j >> bit) & 1 for bit in range(n_bits)]
        
        # Obtener valores de probabilidad del tensor
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
        
        # Si son vecinos inmediatos o el mismo estado, solo contribución directa
        if distancia <= 1:
            resultado = gamma * contribucion_directa
        else:
            # Para distancias mayores, SUMAR PRIMERO dentro del paréntesis, LUEGO multiplicar por γ
            vecinos = self._obtener_vecinos_hipercubo(estado_i, n_bits)
            # Filtrar vecinos que nos acercan al destino
            vecinos_validos = []
            for vecino in vecinos:
                dist_vecino_destino = self._calcular_distancia_hamming(vecino, estado_j, n_bits)
                if dist_vecino_destino < distancia:
                    vecinos_validos.append(vecino)
            
            costo_recursivo = 0.0
            
            for vecino in vecinos_validos:
                costo_vecino = self._calcular_costo_transicion_recursivo(vecino, estado_j, tensor_idx, n_bits)
                costo_recursivo += costo_vecino
            
            # CORRECCIÓN CLAVE: Sumar PRIMERO, luego multiplicar por γ
            suma_total_dentro_parentesis = contribucion_directa + costo_recursivo
            resultado = gamma * suma_total_dentro_parentesis
        
        # Guardar en cache
        self.cache_costos[clave_cache] = resultado
        return resultado
    
    def _calcular_tabla_costos(self):
        """
        Calcula la tabla de costos T para todas las combinaciones de estados y variables.
        Ordena por distancia de Hamming (menor primero) y exporta en formato little-endian.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_bits = len(self.sia_subsistema.dims_ncubos)
        n_estados = 2 ** n_bits

        self.tabla_transiciones = {}

        # Crear lista de todas las combinaciones estado_i, estado_j ordenadas por distancia Hamming
        combinaciones = []
        for i in range(n_estados):
            for j in range(n_estados):
                distancia = self._calcular_distancia_hamming(i, j, n_bits)
                combinaciones.append((i, j, distancia))
        
        # Ordenar por distancia Hamming (menor primero)
        combinaciones.sort(key=lambda x: x[2])

        for var_idx in range(n_vars):
            for i, j, distancia in combinaciones:
                clave = (var_idx, i, j)
                costo = self._calcular_costo_transicion_recursivo(i, j, var_idx, n_bits)
                self.tabla_transiciones[clave] = costo

        # Crear DataFrame con formato little-endian para Excel
        self._exportar_tabla_excel(n_vars, n_estados, n_bits)

    def _exportar_tabla_excel(self, n_vars: int, n_estados: int, n_bits: int):
        """
        Exporta la tabla de costos a Excel con formato little-endian.
        """
        # Crear hojas separadas para cada variable
        with pd.ExcelWriter(f"tabla_costos_geometric_.xlsx") as writer:
            
            for var_idx in range(n_vars):
                # Crear matriz de costos para esta variable
                matriz_costos = np.zeros((n_estados, n_estados))
                
                for i in range(n_estados):
                    for j in range(n_estados):
                        clave = (var_idx, i, j)
                        matriz_costos[i, j] = self.tabla_transiciones.get(clave, 0.0)
                
                # Crear DataFrame con índices y columnas en formato little-endian
                indices_little = [f"{i} ({self._decimal_to_little_endian_binary(i, n_bits)})" for i in range(n_estados)]
                columnas_little = [f"{j} ({self._decimal_to_little_endian_binary(j, n_bits)})" for j in range(n_estados)]
                
                df = pd.DataFrame(matriz_costos, index=indices_little, columns=columnas_little)
                
                # Guardar en hoja específica
                nombre_hoja = f"Variable_{var_idx}"
                df.to_excel(writer, sheet_name=nombre_hoja)

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
        # Encontrar qué bits cambiaron en la transición original
        bits_cambiados = estado_inicial ^ estado_destino
        
        # El complementario parte del estado inicial
        complementario = estado_inicial
        
        # Para cada bit que NO cambió en la transición original, cambiarlo
        for bit_pos in range(n_bits):
            bit_cambio_en_original = (bits_cambiados >> bit_pos) & 1
            
            if not bit_cambio_en_original:  # Si este bit NO cambió en la transición original
                # Cambiar este bit en el complementario
                complementario ^= (1 << bit_pos)
        
        return complementario

    def _identificar_biparticion_optima(self) -> Tuple[Set[int], Set[int]]:
        """
        ENFOQUE MODULAR: Analiza el caso especial 000→111 y las transiciones estándar 
        en funciones separadas, combinando los resultados para encontrar la bipartición óptima.
        
        Returns:
            Tuple con los dos conjuntos de variables de la bipartición óptima
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_bits = len(self.sia_subsistema.dims_ncubos)
        estado_inicial = 0  # Estado inicial (000...0)
        
        # PASO 1: Analizar el caso especial 000→111
        particiones_caso_especial, particion_optima = self._analizar_caso_especial_todos_unos(
            n_vars, n_bits, estado_inicial
        )
        
        # Si encontramos una partición óptima (EMD=0) en el caso especial, retornarla inmediatamente
        if particion_optima is not None:
            return particion_optima
        
        # PASO 2: Analizar las transiciones estándar
        particiones_estandar, particion_optima = self._analizar_transiciones_estandar(
            n_vars, n_bits, estado_inicial
        )
        
        # Si encontramos una partición óptima (EMD=0) en las transiciones estándar, retornarla inmediatamente
        if particion_optima is not None:
            return particion_optima
        
        # PASO 3: Combinar todas las particiones candidatas
        particiones_candidatas = particiones_caso_especial + particiones_estandar
        
        # PASO 4: Seleccionar la mejor partición basada en EMD mínimo
        if particiones_candidatas:
            mejor_particion = min(particiones_candidatas, key=lambda x: x['emd'])
            
            print(f"\n=== MEJOR PARTICIÓN ENCONTRADA ===")
            print(f"Transición base: {mejor_particion['transicion_original_bin']}")
            print(f"Promedio de costos: {mejor_particion['promedio_original']:.6f}")
            print(f"Presente: {[chr(65 + i) for i in sorted(mejor_particion['conjunto_presente'])]}")
            print(f"Futuro: {[chr(65 + i) for i in sorted(mejor_particion['conjunto_futuro'])]}")
            print(f"EMD final: {mejor_particion['emd']:.6f}")
            
            # Debug adicional
            print(f"\nDETALLES DE LA PARTICIÓN GANADORA:")
            print(f"Costos originales: {[f'{chr(65+i)}={c:.4f}' for i, c in enumerate(mejor_particion['info_debug']['costos_originales'])]}")
            print(f"Costos complemento: {[f'{chr(65+i)}={c:.4f}' for i, c in enumerate(mejor_particion['info_debug']['costos_complemento'])]}")
            print(f"Bits que se mantienen: {mejor_particion['info_debug']['bits_que_se_mantienen']}")
            
            return (mejor_particion['conjunto_presente'], mejor_particion['conjunto_futuro'])
        else:
            print("ERROR: No se encontraron particiones válidas")
            return None, None
    
    def _analizar_transiciones_estandar(self, n_vars, n_bits, estado_inicial) -> Tuple[List[dict], Optional[Tuple[Set[int], Set[int]]]]:
        """
        Analiza las transiciones estándar desde el estado inicial (000) a todos los otros estados
        excepto el caso especial todos unos (111).
        
        Args:
            n_vars: Número de variables en el sistema
            n_bits: Número de bits del sistema
            estado_inicial: Estado inicial (generalmente 0)
            
        Returns:
            Tuple con:
            - Lista de particiones candidatas encontradas
            - Opcionalmente, partición óptima si se encontró una con EMD=0
        """
        particiones_candidatas = []
        n_estados = 2 ** n_bits
        
        print(f"\n=== ANÁLISIS DE TRANSICIONES DESDE ESTADO INICIAL {self._decimal_to_little_endian_binary(estado_inicial, n_bits)} ===")
        
        # PASO 1: Analizar todas las transiciones y calcular promedios
        transiciones_con_promedio = []
        
        for estado_destino in range(1, n_estados):  # t(000,001) hasta t(000,111)
            # Excluir el estado todos unos (ya evaluado en caso especial)
            if estado_destino == 2**n_bits - 1:
                continue
                
            estado_destino_bin = self._decimal_to_little_endian_binary(estado_destino, n_bits)
            print(f"\n--- ANALIZANDO TRANSICIÓN t(000, {estado_destino_bin}) ---")
            
            # Obtener costos para cada variable en esta transición
            costos_variables = []
            for var_idx in range(n_vars):
                clave = (var_idx, estado_inicial, estado_destino)
                costo = self.tabla_transiciones.get(clave, float('inf'))
                var_letra = chr(65 + var_idx)  # A, B, C...
                costos_variables.append(costo)
                print(f"Variable {var_letra}: costo = {costo:.4f}")
            
            # Calcular promedio de costos
            if costos_variables and all(c < float('inf') for c in costos_variables):
                promedio = sum(costos_variables) / len(costos_variables)
                print(f"Promedio de costos: {promedio:.4f}")
                
                transiciones_con_promedio.append({
                    'estado_destino': estado_destino,
                    'estado_destino_bin': estado_destino_bin,
                    'costos_variables': costos_variables,
                    'promedio': promedio
                })
            else:
                print("Transición descartada por costos infinitos")
        
        # PASO 2: Seleccionar transiciones con mejor promedio (menor costo)
        if not transiciones_con_promedio:
            print("ERROR: No se encontraron transiciones válidas")
            return [], None
        
        # Ordenar por promedio ascendente (menor costo primero)
        transiciones_con_promedio.sort(key=lambda x: x['promedio'])

        print(f"\n=== TRANSICIONES ORDENADAS POR PROMEDIO (MEJOR PRIMERO) ===")
        for i, trans in enumerate(transiciones_con_promedio[:5]):  # Mostrar top 5
            print(f"{i+1}. t(000, {trans['estado_destino_bin']}) - Promedio: {trans['promedio']:.4f}")
        
        # Calcular el promedio de los promedios como metrica de selección
        promedios = [trans['promedio'] for trans in transiciones_con_promedio]
        promedio_general = sum(promedios) / len(promedios)
        print(f"\nPromedio general de todas las transiciones: {promedio_general:.4f}")

        # Seleccionar transiciones con promedio menor o igual al promedio general
        mejores_transiciones = []
        for trans in transiciones_con_promedio:
            if trans['promedio'] <= promedio_general:
                mejores_transiciones.append(trans)
            else:
                # Ya encontramos el primer valor mayor al promedio, podemos detener la búsqueda
                # (la lista está ordenada por promedio ascendente)
                break

        print(f"\n=== ANALIZANDO {len(mejores_transiciones)} TRANSICIONES CON PROMEDIO <= {promedio_general:.4f} ===")
        
        # PASO 3: Para cada transición seleccionada, analizar con su complemento
        for trans in mejores_transiciones:
            estado_destino = trans['estado_destino']
            costos_originales = trans['costos_variables']
            
            print(f"\n--- ANALIZANDO TRANSICIÓN t(000, {trans['estado_destino_bin']}) ---")
            print(f"Costos originales: {[f'{chr(65+i)}={c:.4f}' for i, c in enumerate(costos_originales)]}")
            
            # PASO 3.1: Calcular estado complemento
            estado_complemento = self._calcular_complemento_estado(estado_inicial, estado_destino, n_bits)
            estado_complemento_bin = self._decimal_to_little_endian_binary(estado_complemento, n_bits)
            
            print(f"Estado complemento: t(000, {estado_complemento_bin})")
            
            # PASO 3.2: Obtener costos del complemento
            costos_complemento = []
            for var_idx in range(n_vars):
                clave = (var_idx, estado_inicial, estado_complemento)
                costo = self.tabla_transiciones.get(clave, float('inf'))
                costos_complemento.append(costo)
            
            print(f"Costos complemento: {[f'{chr(65+i)}={c:.4f}' for i, c in enumerate(costos_complemento)]}")
            
            # PASO 3.3: Comparar variable por variable y seleccionar las de menor costo
            variables_presente = []  # Variables que van al presente
            variables_futuro = []    # Variables que van al futuro
            
            print("Comparación variable por variable:")
            for var_idx in range(n_vars):
                costo_original = costos_originales[var_idx]
                costo_complemento = costos_complemento[var_idx]
                var_letra = chr(65 + var_idx)
                
                if costo_original <= costo_complemento:
                    variables_presente.append(var_idx)
                    print(f"Variable {var_letra}: PRESENTE (costo {costo_original:.4f} <= {costo_complemento:.4f})")
                else:
                    variables_futuro.append(var_idx)
                    print(f"Variable {var_letra}: FUTURO (costo {costo_complemento:.4f} < {costo_original:.4f})")
            
            # PASO 3.4: Analizar los bits que se MANTIENEN 
            bits_que_se_mantienen = self._identificar_bits_que_se_mantienen(estado_inicial, estado_destino, n_bits)
            print(f"Bits que se MANTIENEN en la transición: {bits_que_se_mantienen}")
            
            if bits_que_se_mantienen:
                print("APLICANDO REGLA DE BITS QUE SE MANTIENEN:")
                # Las variables correspondientes a bits que se mantienen van al PRESENTE
                for bit_pos in bits_que_se_mantienen:
                    var_correspondiente = bit_pos  # Variable correspondiente al bit
                    var_letra = chr(65 + var_correspondiente)
                    
                    if var_correspondiente < n_vars:  # Verificar que la variable existe
                        if var_correspondiente not in variables_presente:
                            print(f"  Forzando variable {var_letra} al PRESENTE (bit {bit_pos} se mantiene)")
                            if var_correspondiente in variables_futuro:
                                variables_futuro.remove(var_correspondiente)
                            variables_presente.append(var_correspondiente)
                        else:
                            print(f"  Variable {var_letra} ya estaba en PRESENTE (bit {bit_pos} se mantiene)")
            
            # PASO 3.5: Verificar que ambos conjuntos tengan elementos
            if not variables_presente and variables_futuro:
                # Si el presente está vacío, mover una variable del futuro
                var_movida = variables_futuro.pop(0)
                variables_presente.append(var_movida)
                print(f"AJUSTE: Moviendo variable {chr(65 + var_movida)} a presente (conjunto vacío)")
            
            if not variables_futuro and variables_presente:
                # Si el futuro está vacío, mover una variable del presente
                if len(variables_presente) > 1:
                    var_movida = variables_presente.pop(-1)
                    variables_futuro.append(var_movida)
                    print(f"AJUSTE: Moviendo variable {chr(65 + var_movida)} a futuro (conjunto vacío)")
            
            # PASO 3.6: Formar la partición
            print([variables_presente])
            print([bits_que_se_mantienen])
            conjunto_futuro = set(variables_presente)
            conjunto_presente = set(bits_que_se_mantienen)
            print(f"PARTICIÓN RESULTANTE:")
            print(f"  Presente: {[chr(65 + i) for i in sorted(conjunto_presente)]}")
            print(f"  Futuro: {[chr(65 + i) for i in sorted(conjunto_futuro)]}")
            
            # PASO 3.7: Calcular EMD para esta partición
            emd_valor = self._calcular_emd_biparticion_completa(conjunto_presente, conjunto_futuro)

            # Almacenar la partición si es válida (siempre mantener registro)
            if emd_valor < float('inf'):
                particiones_candidatas.append({
                    'transicion_original': (estado_inicial, estado_destino),
                    'transicion_original_bin': f"t(000, {trans['estado_destino_bin']})",
                    'promedio_original': trans['promedio'],
                    'conjunto_presente': conjunto_presente,
                    'conjunto_futuro': conjunto_futuro,
                    'emd': emd_valor,
                    'info_debug': {
                        'costos_originales': costos_originales,
                        'costos_complemento': costos_complemento,
                        'bits_que_se_mantienen': bits_que_se_mantienen
                    }
                })
                
                print(f"  EMD calculado: {emd_valor:.6f}")
                
                # VALIDACIÓN: Si EMD es prácticamente cero, hemos encontrado la solución óptima
                # Retornar esta partición pero manteniendo el registro completo
                if emd_valor < 1e-6:  # Umbral cercano a cero para manejar imprecisiones de punto flotante
                    print(f"\n¡PARTICIÓN ÓPTIMA ENCONTRADA! EMD = {emd_valor:.8f}")
                    print(f"Presente: {[chr(65 + i) for i in sorted(conjunto_presente)]}")
                    print(f"Futuro: {[chr(65 + i) for i in sorted(conjunto_futuro)]}")
                    print("Terminando búsqueda anticipadamente.")
                    return particiones_candidatas, (conjunto_presente, conjunto_futuro)  # Retorno inmediato
            else:
                print("  EMD inválido, partición descartada")
        
        # Si llegamos aquí, no encontramos EMD=0 en las transiciones estándar
        return particiones_candidatas, None

    def _identificar_bits_que_se_mantienen(self, estado_inicial: int, estado_destino: int, n_bits: int) -> List[int]:
        """
        Identifica qué bits se MANTIENEN (no cambian) entre dos estados.
        
        Args:
            estado_inicial: Estado inicial
            estado_destino: Estado destino
            n_bits: Número total de bits
            
        Returns:
            List[int]: Lista de posiciones de bits que se mantienen (no cambian)
        """
        diferencia = estado_inicial ^ estado_destino  # XOR para encontrar diferencias
        bits_que_se_mantienen = []
        
        for bit_pos in range(n_bits):
            # Si el bit NO está en la diferencia, significa que se mantiene
            if not ((diferencia >> bit_pos) & 1):
                bits_que_se_mantienen.append(bit_pos)
        
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
            # ANÁLISIS DEL MÉTODO BIPARTIR:
            # - alcance: Índices de n-cubos que se marginalizan EN las dimensiones excluidas del mecanismo
            # - mecanismo: Dimensiones que se conservan para los n-cubos del alcance
            #
            # Para lograr la separación presente/futuro:
            # - Variables del FUTURO (alcance): Sus n-cubos se marginalizan eliminando dimensiones del presente
            # - Variables del PRESENTE: Sus n-cubos se marginalizan eliminando dimensiones del futuro
            
            # PASO 1: Convertir índices de variables a índices de n-cubos y dimensiones
            # En tu sistema, parece que los índices de variables corresponden directamente a índices de n-cubos
            indices_ncubos_futuro = np.array(list(conjunto_futuro), dtype=np.int8)  # alcance
            
            # Para el mecanismo, necesitamos las dimensiones que queremos CONSERVAR
            # Si queremos conservar el presente, las dimensiones del mecanismo son las del presente
            dimensiones_presente = np.array(list(conjunto_presente), dtype=np.int8)  # mecanismo
            
            # PASO 2: Aplicar bipartición
            particion = self.sia_subsistema.bipartir(indices_ncubos_futuro, dimensiones_presente)

            # PASO 3: Calcular distribuciones marginales
            dist_original = self.sia_dists_marginales
            dist_particion = particion.distribucion_marginal()
            
            emd = emd_efecto(dist_particion, dist_original)
            print(f"  EMD calculado: {emd}")
            return emd
            
        except Exception as e:
            print(f"ERROR en _calcular_emd_biparticion_completa: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

    def _formatear_resultado(self, biparticion_optima: Tuple[Set[int], Set[int]]) -> Solution:
        """
        Formatea el resultado respetando las variables disponibles para cada componente
        (mecanismo y alcance) de la bipartición.
        """
        if biparticion_optima[0] is None or biparticion_optima[1] is None:
            print("ERROR: No se pudo encontrar una bipartición válida")
            return None
        
        conjunto_presente, conjunto_futuro = biparticion_optima
        
        print(f"\n=== FORMATEANDO RESULTADO FINAL ===")
        print(f"Conjunto presente (mecanismo): {[chr(65 + i) for i in sorted(conjunto_presente)]}")
        print(f"Conjunto futuro (alcance): {[chr(65 + i) for i in sorted(conjunto_futuro)]}")
        
        # Calcular EMD usando el método corregido
        perdida = self._calcular_emd_biparticion_completa(conjunto_presente, conjunto_futuro)
        
        try:
            # Preparar índices para bipartición
            indices_mecanismo = np.array(list(conjunto_presente), dtype=np.int8)
            indices_alcance = np.array(list(conjunto_futuro), dtype=np.int8)
            
            # Realizar bipartición para obtener distribución
            particion = self.sia_subsistema.bipartir(indices_alcance, indices_mecanismo)
            distribucion_particion = particion.distribucion_marginal()
            
            print(f"EMD final calculado: {perdida:.6f}")
            
            # CORRECCIÓN: Reorganizar nodos para el formato visual correcto
            nodos_mecanismo = []
            nodos_alcance = []
            
            # Identificar variables disponibles en el sistema
            variables_disponibles = set(range(len(self.sia_subsistema.indices_ncubos)))
            print(f"Variables disponibles en el sistema: {[chr(65 + i) for i in sorted(variables_disponibles)]}")
            
            # Validar que los conjuntos contengan solo variables disponibles
            conjunto_presente_valido = conjunto_presente.intersection(variables_disponibles)
            conjunto_futuro_valido = conjunto_futuro.intersection(variables_disponibles)
            
            if len(conjunto_presente_valido) != len(conjunto_presente) or len(conjunto_futuro_valido) != len(conjunto_futuro):
                print("ADVERTENCIA: Algunos índices en los conjuntos no corresponden a variables disponibles.")
                conjunto_presente = conjunto_presente_valido
                conjunto_futuro = conjunto_futuro_valido
            
            # Calcular complementos de los conjuntos
            complemento_presente = variables_disponibles - conjunto_presente
            complemento_futuro = variables_disponibles - conjunto_futuro
            
            # Validar disponibilidad de variables en sus respectivos contextos
            # Obtenemos las variables disponibles según los índices de ncubos
            variables_disponibles_mecanismo = set(self.sia_subsistema.dims_ncubos)
            variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
            
            print(f"Variables disponibles en mecanismo: {[chr(65 + i) for i in sorted(variables_disponibles_mecanismo)]}")
            print(f"Variables disponibles en alcance: {[chr(65 + i) for i in sorted(variables_disponibles_alcance)]}")
            
            # PRIMER PARÉNTESIS
            # Arriba (alcance): Variables EFECTO del conjunto futuro (solo las disponibles en alcance)
            futuro_disponible = conjunto_futuro.intersection(variables_disponibles_alcance)
            for var in futuro_disponible:
                nodos_mecanismo.append((EFECTO, var))
            
            # Abajo (mecanismo): Variables ACTUAL del conjunto presente (solo las disponibles en mecanismo)
            presente_disponible = conjunto_presente.intersection(variables_disponibles_mecanismo)
            for var in presente_disponible:
                nodos_mecanismo.append((ACTUAL, var))
            
            # SEGUNDO PARÉNTESIS
            # Arriba (alcance): Variables EFECTO del complemento futuro (solo las disponibles en alcance)
            complemento_futuro_disponible = complemento_futuro.intersection(variables_disponibles_alcance)
            for var in complemento_futuro_disponible:
                nodos_alcance.append((EFECTO, var))
            
            # Abajo (mecanismo): Variables ACTUAL del complemento presente (solo las disponibles en mecanismo)
            complemento_presente_disponible = complemento_presente.intersection(variables_disponibles_mecanismo)
            for var in complemento_presente_disponible:
                nodos_alcance.append((ACTUAL, var))
            
            # Formatear partición con los nodos correctos
            fmt_particion = fmt_biparte_q(nodos_mecanismo, nodos_alcance)
            
            # Debug para verificar
            print(f"Nodos mecanismo (1er paréntesis): {nodos_mecanismo}")
            print(f"Nodos alcance (2do paréntesis): {nodos_alcance}")
            print(f"Partición formateada: {fmt_particion}")
            
        except Exception as e:
            print(f"Error en formateo: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Verificación final
        variables_presente = [chr(65 + idx) for idx in sorted(conjunto_presente)]
        variables_futuro = [chr(65 + idx) for idx in sorted(conjunto_futuro)]
        print(f"RESULTADO FINAL:")
        print(f"  Variables en presente (mecanismo): {variables_presente}")
        print(f"  Variables en futuro (alcance): {variables_futuro}")
        print(f"  EMD: {perdida:.6f}")
        
        return Solution(
            estrategia="Geometric-SIA-Corregido",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_particion,
        )
    
    def _analizar_caso_especial_todos_unos(self, n_vars, n_bits, estado_inicial):
        """
        Analiza específicamente la transición desde el estado todos ceros (000) 
        al estado todos unos (111).
        OPTIMIZADO: Evalúa solo las variables con costo directo <= promedio.
        CORREGIDO: Alcance con una variable, mecanismo vacío.
        """
        particiones_candidatas = []
        estado_todos_unos = 2**n_bits - 1  # Por ejemplo, 111 para 3 bits
        
        print(f"\n=== CASO ESPECIAL: EVALUANDO TRANSICIÓN 000→111 ===")
        
        # OBTENER INFORMACIÓN DE VARIABLES DISPONIBLES
        variables_disponibles_alcance = set(self.sia_subsistema.indices_ncubos)
        print(f"Variables disponibles para alcance: {[chr(65 + i) for i in sorted(variables_disponibles_alcance)]}")
        
        # Obtener costos directos para cada variable en la transición 000→111
        costos_directos = []
        costos_totales = 0.0
        variables_validas = 0
        
        # Recopilar los costos directos de cada variable
        for var_idx in range(n_vars):
            clave = (var_idx, estado_inicial, estado_todos_unos)
            costo = self.tabla_transiciones.get(clave, float('inf'))
            var_letra = chr(65 + var_idx)
            
            if costo < float('inf') and var_idx in variables_disponibles_alcance:
                costos_totales += costo
                variables_validas += 1
            
            costos_directos.append((var_idx, costo, var_letra))
            print(f"Variable {var_letra}: costo directo = {costo:.4f}")
        
        # Calcular el promedio de los costos directos
        if variables_validas > 0:
            promedio_costos_directos = costos_totales / variables_validas
            print(f"Promedio de costos directos: {promedio_costos_directos:.4f}")
            
            # Filtrar variables con costo directo menor o igual al promedio Y disponibles en el alcance
            variables_seleccionadas = [(idx, costo, letra) for idx, costo, letra in costos_directos 
                                     if costo <= promedio_costos_directos and idx in variables_disponibles_alcance]
            
            print(f"Evaluando {len(variables_seleccionadas)}/{len(costos_directos)} variables con costo <= promedio")
        else:
            # Si no hay variables válidas, evaluar todas las disponibles en el alcance
            variables_seleccionadas = [(idx, costo, letra) for idx, costo, letra in costos_directos 
                                     if idx in variables_disponibles_alcance]
            print("No se encontraron variables con costos válidos, evaluando todas las disponibles en alcance")
        
        # Ordenar por costo (menor primero) para evaluar primero las más prometedoras
        variables_seleccionadas.sort(key=lambda x: x[1])
        
        # Para este caso especial, evaluamos solo las variables seleccionadas
        for var_idx, costo, var_letra in variables_seleccionadas:
            # CORRECCIÓN: Crear bipartición con esta variable en futuro (alcance) y nada en presente (mecanismo)
            conjunto_futuro = {var_idx}
            conjunto_presente = set()  # Mecanismo vacío
            
            print(f"Evaluando variable {var_letra} como único elemento en FUTURO, mecanismo vacío")
            print(f"  Presente (mecanismo): []")
            print(f"  Futuro (alcance): {[chr(65 + i) for i in sorted(conjunto_futuro)]}")
            
            # Calcular EMD para esta bipartición
            emd_valor = self._calcular_emd_biparticion_completa(conjunto_presente, conjunto_futuro)
            
            # Almacenar la partición si es válida
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
                
                # Si encontramos EMD=0, retornar inmediatamente esta solución óptima
                if emd_valor < 1e-6:
                    print(f"  ¡EMD perfecto (0) con variable {var_letra} en FUTURO! Retornando solución óptima.")
                    return particiones_candidatas, (conjunto_presente, conjunto_futuro)
            else:
                print(f"  EMD inválido para variable {var_letra}, partición descartada")

        # Si llegamos aquí, no encontramos EMD=0 en el caso especial
        return particiones_candidatas, None