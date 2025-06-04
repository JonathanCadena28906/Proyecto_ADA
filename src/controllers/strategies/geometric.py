import time
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Set, Optional
try:
    import networkx as nx
except ImportError:
    print("NetworkX not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
    import networkx as nx

from src.middlewares.slogger import SafeLogger
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_biparte_q
from src.constants.base import EFECTO, ACTUAL


class GeometricSIA(SIA):
    """
    Implementación optimizada del enfoque geométrico para el análisis SIA basado en hipercubos n-dimensionales.
    
    Esta clase utiliza representaciones geométricas del sistema como hipercubos donde cada vértice
    representa un estado posible. La estrategia principal es identificar patrones de complementariedad
    causal mediante análisis de transiciones de costo cero y agrupamiento de variables.
    
    Attributes:
        tabla_transiciones: Diccionario que almacena los costos de transición entre estados
        cache_hamming: Cache para distancias de Hamming calculadas
        cache_costos: Cache para costos calculados recursivamente
        memoria_biparticiones: Almacena las biparticiones evaluadas y sus costos
        tensores: Representación tensorial del subsistema
        logger: Logger para el proceso geométrico
        grupos_complementarios: Grupos de variables con patrones de complementariedad
    """
    
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.tabla_transiciones: Dict[Tuple[int, int, int], float] = {}
        self.cache_hamming: Dict[Tuple[int, int], int] = {}
        self.cache_costos: Dict[Tuple[int, int, int], float] = {}
        self.memoria_biparticiones: Dict[Tuple, Tuple[float, np.ndarray]] = {}
        self.tensores: List[np.ndarray] = []
        self.logger = SafeLogger("geometric_strategy")
        self.grupos_complementarios: List[Set[int]] = []
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        """
        Implementa el algoritmo geométrico optimizado para encontrar la bipartición óptima
        utilizando análisis de complementariedad causal y transiciones de costo cero.
        
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
        
        # 3. Descubrir biparticiones por análisis de costo cero y complementariedad
        biparticion_optima = self._descubrir_biparticiones_por_costo_cero()
        
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
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_bits = len(self.sia_subsistema.dims_ncubos)
        
        # Descomponer en tensores elementales
        for i, ncube in enumerate(self.sia_subsistema.ncubos):
            # Extraer el tensor de probabilidades condicionales
            tensor_original = ncube.data.copy()
            
            # Reordenar tensor para formato little-endian
            tensor_reordenado = self._reordenar_tensor_little_endian(tensor_original, n_bits)
            self.tensores.append(tensor_reordenado)
            
        print(f"Construida representación {n_vars}-dimensional con {len(self.tensores)} tensores (little-endian)")
    
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
    
    def _calcular_costo_transicion_recursivo(self, estado_i: int, estado_j: int, tensor_idx: int, n_bits: int, debug_prints: bool = False) -> float:
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
                costo_vecino = self._calcular_costo_transicion_recursivo(vecino, estado_j, tensor_idx, n_bits, False)
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
        
        print(f"Calculando tabla de costos para {n_vars} variables y {n_estados} estados")

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
                costo = self._calcular_costo_transicion_recursivo(i, j, var_idx, n_bits, False)
                self.tabla_transiciones[clave] = costo

        print(f"Tabla de costos calculada: {len(self.tabla_transiciones)} entradas")

    def _descubrir_biparticiones_por_costo_cero(self) -> Tuple[Set[int], Set[int]]:
        """
        Estrategia principal: Descubre biparticiones óptimas mediante análisis de 
        transiciones de costo cero y patrones de complementariedad causal.
        
        Returns:
            Tuple[Set[int], Set[int]]: La bipartición óptima encontrada
        """
        print("="*60)
        print("INICIANDO ANÁLISIS DE COMPLEMENTARIEDAD CAUSAL")
        print("="*60)
        
        # Paso 1: Detectar transiciones de costo cero
        transiciones_cero = self._detectar_transiciones_costo_cero()
        
        # Paso 2: Construir matriz de interacción entre variables
        matriz_interaccion = self._construir_matriz_interaccion(transiciones_cero)
        
        # Paso 3: Identificar grupos complementarios
        grupos_complementarios = self._identificar_grupos_complementarios(matriz_interaccion)
        
        # Paso 4: Generar biparticiones candidatas basadas en complementariedad
        candidatas = self._generar_candidatas_por_complementariedad(grupos_complementarios)
        
        # Paso 5: Evaluar candidatas con EMD y seleccionar la óptima
        biparticion_optima = self._evaluar_candidatas_con_emd(candidatas)
        
        return biparticion_optima
    
    def _detectar_transiciones_costo_cero(self, umbral_cero: float = 1e-10) -> Dict[int, List[Tuple[int, int]]]:
        """
        Detecta todas las transiciones (i, j) tales que t(i, j) ≈ 0 para cada variable.
        
        Args:
            umbral_cero: Umbral para considerar un costo como cero
            
        Returns:
            Dict[int, List[Tuple[int, int]]]: Diccionario var_idx -> lista de transiciones (i,j) con costo ≈ 0
        """
        transiciones_cero = defaultdict(list)
        n_vars = len(self.sia_subsistema.indices_ncubos)
        
        print("Detectando transiciones de costo cero...")
        
        for var_idx in range(n_vars):
            costo_cero_count = 0
            for clave, costo in self.tabla_transiciones.items():
                if clave[0] == var_idx and abs(costo) <= umbral_cero:
                    estado_i, estado_j = clave[1], clave[2]
                    transiciones_cero[var_idx].append((estado_i, estado_j))
                    costo_cero_count += 1
            
            print(f"Variable {var_idx}: {costo_cero_count} transiciones de costo cero")
        
        return dict(transiciones_cero)
    
    def _construir_matriz_interaccion(self, transiciones_cero: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
        """
        Construye una matriz de interacción entre variables basada en transiciones compartidas de costo cero.
        
        Args:
            transiciones_cero: Diccionario de transiciones de costo cero por variable
            
        Returns:
            np.ndarray: Matriz de interacción simétrica n_vars x n_vars
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        matriz_interaccion = np.zeros((n_vars, n_vars))
        
        print("Construyendo matriz de interacción...")
        
        # Para cada par de variables, calcular similaridad en transiciones de costo cero
        for var_i in range(n_vars):
            for var_j in range(var_i + 1, n_vars):
                transiciones_i = set(transiciones_cero.get(var_i, []))
                transiciones_j = set(transiciones_cero.get(var_j, []))
                
                # Calcular índice de Jaccard para similitud
                interseccion = len(transiciones_i & transiciones_j)
                union = len(transiciones_i | transiciones_j)
                
                if union > 0:
                    similaridad = interseccion / union
                else:
                    similaridad = 0.0
                
                # Complementariedad = 1 - similaridad (variables complementarias tienen pocas transiciones compartidas)
                complementariedad = 1.0 - similaridad
                
                matriz_interaccion[var_i, var_j] = complementariedad
                matriz_interaccion[var_j, var_i] = complementariedad
        
        print("Matriz de interacción construida:")
        print(matriz_interaccion)
        
        return matriz_interaccion
    
    def _identificar_grupos_complementarios(self, matriz_interaccion: np.ndarray) -> List[Set[int]]:
        """
        Identifica grupos de variables complementarias usando clustering espectral.
        
        Args:
            matriz_interaccion: Matriz de complementariedad entre variables
            
        Returns:
            List[Set[int]]: Lista de grupos complementarios
        """
        n_vars = matriz_interaccion.shape[0]
        
        if n_vars <= 2:
            return [{0}, {1}] if n_vars == 2 else [set(range(n_vars))]
        
        print("Identificando grupos complementarios...")
        
        # Usar clustering basado en umbral de complementariedad
        umbral_complementariedad = 0.6
        grupos = []
        variables_asignadas = set()
        
        # Construir grafo de complementariedad
        G = nx.Graph()
        for i in range(n_vars):
            G.add_node(i)
            for j in range(i + 1, n_vars):
                if matriz_interaccion[i, j] >= umbral_complementariedad:
                    G.add_edge(i, j, weight=matriz_interaccion[i, j])
        
        # Encontrar componentes conectadas como grupos
        componentes = list(nx.connected_components(G))
        
        if len(componentes) >= 2:
            # Si hay múltiples componentes, usar las dos más grandes
            componentes.sort(key=len, reverse=True)
            grupos = [componentes[0], componentes[1]]
        else:
            # Si solo hay una componente, dividir por métricas de centralidad
            if len(componentes[0]) > 1:
                centralidad = nx.eigenvector_centrality(G)
                nodos_ordenados = sorted(centralidad.items(), key=lambda x: x[1], reverse=True)
                
                mitad = len(nodos_ordenados) // 2
                grupo1 = {nodo for nodo, _ in nodos_ordenados[:mitad]}
                grupo2 = {nodo for nodo, _ in nodos_ordenados[mitad:]}
                grupos = [grupo1, grupo2]
            else:
                # Fallback: división simple
                mitad = n_vars // 2
                grupos = [set(range(mitad)), set(range(mitad, n_vars))]
        
        print(f"Grupos complementarios identificados: {grupos}")
        self.grupos_complementarios = grupos
        
        return grupos
    
    def _generar_candidatas_por_complementariedad(self, grupos_complementarios: List[Set[int]]) -> List[Tuple[Set[int], Set[int]]]:
        """
        Genera biparticiones candidatas basadas en grupos complementarios.
        
        Args:
            grupos_complementarios: Lista de grupos de variables complementarias
            
        Returns:
            List[Tuple[Set[int], Set[int]]]: Lista de biparticiones candidatas
        """
        candidatas = []
        n_vars = len(self.sia_subsistema.indices_ncubos)
        
        print("Generando candidatas por complementariedad...")
        
        # Candidata principal: usar los dos grupos más grandes
        if len(grupos_complementarios) >= 2:
            grupo1 = grupos_complementarios[0]
            grupo2 = grupos_complementarios[1]
            
            # Asegurar que todos los nodos estén asignados
            nodos_restantes = set(range(n_vars)) - grupo1 - grupo2
            if nodos_restantes:
                # Asignar nodos restantes al grupo más pequeño
                if len(grupo1) <= len(grupo2):
                    grupo1 = grupo1 | nodos_restantes
                else:
                    grupo2 = grupo2 | nodos_restantes
            
            candidatas.append((grupo1, grupo2))
        
        # Candidatas adicionales: variaciones de la principal
        if candidatas:
            grupo_base1, grupo_base2 = candidatas[0]
            
            # Intercambiar variables entre grupos
            for var in list(grupo_base1)[:min(2, len(grupo_base1))]:
                nuevo_grupo1 = grupo_base1 - {var}
                nuevo_grupo2 = grupo_base2 | {var}
                if len(nuevo_grupo1) > 0 and len(nuevo_grupo2) > 0:
                    candidatas.append((nuevo_grupo1, nuevo_grupo2))
            
            for var in list(grupo_base2)[:min(2, len(grupo_base2))]:
                nuevo_grupo1 = grupo_base1 | {var}
                nuevo_grupo2 = grupo_base2 - {var}
                if len(nuevo_grupo1) > 0 and len(nuevo_grupo2) > 0:
                    candidatas.append((nuevo_grupo1, nuevo_grupo2))
        
        # Fallback: candidatas balanceadas
        if not candidatas:
            mitad = n_vars // 2
            candidatas.append((set(range(mitad)), set(range(mitad, n_vars))))
        
        # Limitar número de candidatas para eficiencia
        candidatas = candidatas[:10]
        
        print(f"Generadas {len(candidatas)} candidatas")
        return candidatas
    
    def _evaluar_candidatas_con_emd(self, candidatas: List[Tuple[Set[int], Set[int]]]) -> Tuple[Set[int], Set[int]]:
        """
        Evalúa candidatas usando EMD (Earth Mover's Distance) y selecciona la óptima.
        
        Args:
            candidatas: Lista de biparticiones candidatas
            
        Returns:
            Tuple[Set[int], Set[int]]: Bipartición óptima
        """
        mejor_candidata = None
        mejor_emd = float('inf')
        
        print(f"Evaluando {len(candidatas)} candidatas con EMD...")
        print("="*50)
        
        for i, (conjunto1, conjunto2) in enumerate(candidatas):
            print(f"Candidata {i+1}: {conjunto1} | {conjunto2}")
            
            try:
                # Convertir conjuntos a arrays para bipartición
                indices_alcance = np.array(list(conjunto1), dtype=np.int8)
                indices_mecanismo = np.array(list(conjunto2), dtype=np.int8)
                
                # Realizar bipartición
                particion = self.sia_subsistema.bipartir(indices_alcance, indices_mecanismo)
                distribucion_marginal = particion.distribucion_marginal()
                
                # Calcular EMD
                emd = emd_efecto(distribucion_marginal, self.sia_dists_marginales)
                
                # Almacenar en memoria
                clave_candidata = (tuple(sorted(conjunto1)), tuple(sorted(conjunto2)))
                self.memoria_biparticiones[clave_candidata] = (emd, distribucion_marginal)
                
                print(f"  EMD = {emd:.8f}")
                
                # Actualizar mejor candidata
                if emd < mejor_emd:
                    mejor_emd = emd
                    mejor_candidata = (conjunto1, conjunto2)
                    print(f"  *** NUEVA MEJOR CANDIDATA ***")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if mejor_candidata is None:
            # Fallback
            n_vars = len(self.sia_subsistema.indices_ncubos)
            mitad = n_vars // 2
            mejor_candidata = (set(range(mitad)), set(range(mitad, n_vars)))
            mejor_emd = 1.0
        
        print("="*50)
        print(f"BIPARTICIÓN ÓPTIMA ENCONTRADA:")
        print(f"  Grupo 1: {mejor_candidata[0]}")
        print(f"  Grupo 2: {mejor_candidata[1]}")
        print(f"  EMD: {mejor_emd:.8f}")
        print("="*50)
        
        return mejor_candidata
    
    def _formatear_resultado(self, biparticion_optima: Tuple[Set[int], Set[int]]) -> Solution:
        """
        Formatea el resultado en el formato compatible con el sistema.
        """
        conjunto1, conjunto2 = biparticion_optima
        clave_candidata = (tuple(sorted(conjunto1)), tuple(sorted(conjunto2)))
        
        if clave_candidata in self.memoria_biparticiones:
            perdida, distribucion_particion = self.memoria_biparticiones[clave_candidata]
        else:
            perdida = 1.0
            distribucion_particion = self.sia_dists_marginales
        
        # Formatear partición para visualización
        nodos_conjunto1 = [(EFECTO, idx) for idx in conjunto1]
        nodos_conjunto2 = [(ACTUAL, idx) for idx in conjunto2]
        
        fmt_particion = fmt_biparte_q(nodos_conjunto1, nodos_conjunto2)
        
        return Solution(
            estrategia="Geometric-SIA-Optimized",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_particion,
        )
    
    def __str__(self) -> str:
        return f"GeometricSIA(variables={len(getattr(self.sia_subsistema, 'indices_ncubos', []))}, " \
               f"estados_calculados={len(self.tabla_transiciones)}, " \
               f"biparticiones_evaluadas={len(self.memoria_biparticiones)}, " \
               f"grupos_complementarios={len(self.grupos_complementarios)})"