import time
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Set

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
        self.tabla_transiciones: Dict[Tuple[int, int], float] = {}
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
        
        # 3. Identificar las biparticiones candidatas
        candidatas = self._identificar_biparticiones_candidatas()
        
        # 4. Evaluar y seleccionar la bipartición óptima
        biparticion_optima = self._evaluar_candidatos(candidatas)
        
        # 5. Formatear y retornar el resultado
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
            
            print(f"Tensor {i}: Reordenado de big-endian a little-endian")
            
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
        
        print(f"Tensor reordenado: {tensor_original.shape} -> {tensor_reordenado.shape}")
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
        
        if debug_prints and estado_i == 0 or estado_i == 1:  # Solo para estado 000
            little_i = self._decimal_to_little_endian_binary(estado_i, n_bits)
            little_j = self._decimal_to_little_endian_binary(estado_j, n_bits)
            
            print(f"\n{'='*60}")
            print(f"CÁLCULO: t({estado_i}, {estado_j}) = t({little_i}, {little_j})")
            print(f"{'='*60}")
            
            # Mostrar fórmula CORREGIDA
            if distancia <= 1:
                print(f"FÓRMULA: t(i,j) = γ · |X[i] - X[j]| (caso d ≤ 1)")
            else:
                print(f"FÓRMULA: t(i,j) = γ · (|X[i] - X[j]| + Σ{{t(k,j)}}) donde k∈N(i) (caso d > 1)")
            
            print(f"")
            print(f"PASO 1: Calcular distancia de Hamming d(i,j)")
            print(f"  d({estado_i}, {estado_j}) = d({little_i}, {little_j}) = {distancia}")
            print(f"")
            print(f"PASO 2: Calcular factor γ")
            print(f"  γ = 2^(-d(i,j)) = 2^(-{distancia}) = {gamma}")
            print(f"")
            print(f"PASO 3: Calcular |X[i] - X[j]|")
            print(f"  X[{estado_i}] = X[{little_i}] = {prob_i}")
            print(f"  X[{estado_j}] = X[{little_j}] = {prob_j}")
            print(f"  |X[i] - X[j]| = |{prob_i} - {prob_j}| = {contribucion_directa}")
        
        # Si son vecinos inmediatos o el mismo estado, solo contribución directa
        if distancia <= 1:
            resultado = gamma * contribucion_directa
            if debug_prints and estado_i == 0:
                print(f"")
                if distancia == 0:
                    print(f"CASO: d(i,j) = 0 (mismo estado)")
                else:
                    print(f"CASO: d(i,j) = 1 (vecinos inmediatos)")
                print(f"RESULTADO FINAL:")
                print(f"  t({estado_i}, {estado_j}) = γ · |X[i] - X[j]| = {gamma} · {contribucion_directa} = {resultado}")
                print(f"  t({little_i}, {little_j}) = {resultado}")
        else:
            # Para distancias mayores, SUMAR PRIMERO dentro del paréntesis, LUEGO multiplicar por γ
            vecinos = self._obtener_vecinos_hipercubo(estado_i, n_bits)
            # Filtrar vecinos que nos acercan al destino
            vecinos_validos = []
            for vecino in vecinos:
                dist_vecino_destino = self._calcular_distancia_hamming(vecino, estado_j, n_bits)
                if dist_vecino_destino < distancia:
                    vecinos_validos.append(vecino)
            
            if debug_prints and estado_i == 0 or estado_i == 1:
                print(f"")
                print(f"CASO: d(i,j) = {distancia} > 1 (requiere recursión)")
                print(f"PASO 4: Encontrar vecinos N(i) que se acercan a j")
                todos_vecinos = [self._decimal_to_little_endian_binary(v, n_bits) for v in vecinos]
                vecinos_validos_str = [self._decimal_to_little_endian_binary(v, n_bits) for v in vecinos_validos]
                print(f"  Todos los vecinos de {little_i}: {todos_vecinos}")
                print(f"  Vecinos válidos (se acercan a {little_j}): {vecinos_validos_str}")
            
            costo_recursivo = 0.0
            costo_recursivo_detalle = []
            
            for vecino in vecinos_validos:
                costo_vecino = self._calcular_costo_transicion_recursivo(vecino, estado_j, tensor_idx, n_bits, False)
                costo_recursivo += costo_vecino
                if debug_prints and estado_i == 0 or estado_i == 1:
                    vecino_little = self._decimal_to_little_endian_binary(vecino, n_bits)
                    costo_recursivo_detalle.append(f"t({vecino}, {estado_j}) = t({vecino_little}, {little_j}) = {costo_vecino}")
            
            # CORRECCIÓN CLAVE: Sumar PRIMERO, luego multiplicar por γ
            suma_total_dentro_parentesis = contribucion_directa + costo_recursivo
            resultado = gamma * suma_total_dentro_parentesis
            
            if debug_prints and estado_i == 0 or estado_i == 1:
                print(f"")
                print(f"PASO 5: Calcular Σ{{t(k,j)}} para k∈N(i)")
                for detalle in costo_recursivo_detalle:
                    print(f"  {detalle}")
                print(f"  Σ{{t(k,j)}} = {costo_recursivo}")
                print(f"")
                print(f"PASO 6: SUMAR dentro del paréntesis")
                print(f"  |X[i] - X[j]| + Σ{{t(k,j)}} = {contribucion_directa} + {costo_recursivo} = {suma_total_dentro_parentesis}")
                print(f"")
                print(f"RESULTADO FINAL:")
                print(f"  t({estado_i}, {estado_j}) = γ · (|X[i] - X[j]| + Σ{{t(k,j)}})")
                print(f"  t({estado_i}, {estado_j}) = {gamma} · ({contribucion_directa} + {costo_recursivo})")
                print(f"  t({estado_i}, {estado_j}) = {gamma} · {suma_total_dentro_parentesis} = {resultado}")
                print(f"  t({little_i}, {little_j}) = {resultado}")
        
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
        print("="*50)
        print("EJEMPLO DE CÁLCULO PARA ESTADO 000:")
        print("="*50)

        # Crear lista de todas las combinaciones estado_i, estado_j ordenadas por distancia Hamming
        combinaciones = []
        for i in range(n_estados):
            for j in range(n_estados):
                distancia = self._calcular_distancia_hamming(i, j, n_bits)
                combinaciones.append((i, j, distancia))
        
        # Ordenar por distancia Hamming (menor primero)
        combinaciones.sort(key=lambda x: x[2])

        for var_idx in range(n_vars):
            print(f"\nVariable {var_idx}:")
            
            for i, j, distancia in combinaciones:
                clave = (var_idx, i, j)
                
                # Imprimir detalles solo para estado 000 (i=0) de la primera variable
                debug_prints = (var_idx == 0 and i == 0)
                
                costo = self._calcular_costo_transicion_recursivo(i, j, var_idx, n_bits, debug_prints)
                self.tabla_transiciones[clave] = costo

        # Crear DataFrame con formato little-endian para Excel
        self._exportar_tabla_excel(n_vars, n_estados, n_bits)

        print(f"Tabla de costos calculada: {len(self.tabla_transiciones)} entradas")

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
                
                print(f"Variable {var_idx} exportada a hoja '{nombre_hoja}'")

    def _identificar_biparticiones_candidatas(self) -> List[Tuple[Set[int], Set[int]]]:
        """
        Identifica biparticiones candidatas basándose en patrones en la tabla de transiciones.
        
        Returns:
            Lista de tuplas (conjunto1, conjunto2) representando biparticiones candidatas
        """
        candidatas = []
        n_vars = len(self.sia_subsistema.indices_ncubos)
        variables = set(range(n_vars))
        
        # Generar todas las biparticiones posibles (excluyendo vacías y completas)
        for r in range(1, n_vars):
            for subset in combinations(variables, r):
                conjunto1 = set(subset)
                conjunto2 = variables - conjunto1
                
                if len(conjunto1) > 0 and len(conjunto2) > 0:
                    candidatas.append((conjunto1, conjunto2))
        
        # Filtrar candidatas prometedoras basándose en análisis de costos
        candidatas_filtradas = self._filtrar_candidatas_prometedoras(candidatas)
        
        print(f"Identificadas {len(candidatas_filtradas)} biparticiones candidatas")
        return candidatas_filtradas
    
    def _filtrar_candidatas_prometedoras(self, candidatas: List[Tuple[Set[int], Set[int]]]) -> List[Tuple[Set[int], Set[int]]]:
        """
        Filtra candidatas prometedoras basándose en análisis de complementariedad de costos.
        """
        candidatas_evaluadas = []
        
        for conjunto1, conjunto2 in candidatas:
            # Calcular métricas de complementariedad
            complementariedad = self._calcular_complementariedad(conjunto1, conjunto2)
            
            # Filtrar basándose en umbral de complementariedad
            if complementariedad > 0.1:  # Umbral ajustable
                candidatas_evaluadas.append((conjunto1, conjunto2))
        
        # Si tenemos demasiadas candidatas, mantener las mejores
        if len(candidatas_evaluadas) > 20:
            candidatas_evaluadas.sort(key=lambda x: self._calcular_complementariedad(x[0], x[1]), reverse=True)
            candidatas_evaluadas = candidatas_evaluadas[:20]
        
        return candidatas_evaluadas
    
    def _calcular_complementariedad(self, conjunto1: Set[int], conjunto2: Set[int]) -> float:
        """
        Calcula una métrica de complementariedad entre dos conjuntos de variables.
        """
        if not self.tabla_transiciones:
            return 0.5  # Valor neutro si no hay tabla
        
        complementariedad = 0.0
        n_comparaciones = 0
        n_bits = len(self.sia_subsistema.dims_ncubos)
        n_estados = min(16, 2 ** n_bits)  # Limitar para eficiencia
        
        for i in range(n_estados):
            for j in range(n_estados):
                costos1 = [self.tabla_transiciones.get((var, i, j), 0.0) for var in conjunto1]
                costos2 = [self.tabla_transiciones.get((var, i, j), 0.0) for var in conjunto2]
                
                if costos1 and costos2:
                    promedio1 = np.mean(costos1)
                    promedio2 = np.mean(costos2)
                    # Complementariedad como diferencia normalizada
                    if promedio1 + promedio2 > 0:
                        complementariedad += abs(promedio1 - promedio2) / (promedio1 + promedio2)
                        n_comparaciones += 1
        
        return complementariedad / max(1, n_comparaciones)
    
    def _evaluar_candidatos(self, candidatas: List[Tuple[Set[int], Set[int]]]) -> Tuple[Set[int], Set[int]]:
        """
        Evalúa las biparticiones candidatas y selecciona la óptima.
        """
        mejor_candidata = None
        mejor_emd = float('inf')
        
        print(f"Evaluando {len(candidatas)} candidatas")
        
        for i, (conjunto1, conjunto2) in enumerate(candidatas):
            # Convertir conjuntos a arrays para bipartición
            indices_alcance = np.array([idx for idx in conjunto1 if idx in self.sia_subsistema.indices_ncubos], dtype=np.int8)
            indices_mecanismo = np.array([idx for idx in conjunto2 if idx in self.sia_subsistema.dims_ncubos], dtype=np.int8)
            
            # Si algún conjunto está vacío, usar distribución por defecto
            if len(indices_alcance) == 0:
                indices_alcance = np.array([self.sia_subsistema.indices_ncubos[0]], dtype=np.int8)
            if len(indices_mecanismo) == 0:
                indices_mecanismo = np.array([self.sia_subsistema.dims_ncubos[0]], dtype=np.int8)
            
            try:
                # Realizar bipartición
                particion = self.sia_subsistema.bipartir(indices_alcance, indices_mecanismo)
                distribucion_marginal = particion.distribucion_marginal()
                
                # Calcular EMD
                emd = emd_efecto(distribucion_marginal, self.sia_dists_marginales)
                
                # Almacenar en memoria
                clave_candidata = (tuple(sorted(conjunto1)), tuple(sorted(conjunto2)))
                self.memoria_biparticiones[clave_candidata] = (emd, distribucion_marginal)
                
                # Actualizar mejor candidata
                if emd < mejor_emd:
                    mejor_emd = emd
                    mejor_candidata = (conjunto1, conjunto2)
                
                print(f"Candidata {i+1}: EMD = {emd:.6f}")
                
            except Exception as e:
                print(f"Error evaluando candidata {i+1}: {e}")
                continue
        
        if mejor_candidata is None:
            # Fallback: usar primera partición válida
            n_vars = len(self.sia_subsistema.indices_ncubos)
            mitad = n_vars // 2
            conjunto1 = set(range(mitad))
            conjunto2 = set(range(mitad, n_vars))
            mejor_candidata = (conjunto1, conjunto2)
            
            # Evaluar fallback
            try:
                indices_alcance = np.array(list(conjunto1), dtype=np.int8)
                indices_mecanismo = np.array(list(conjunto2), dtype=np.int8)
                particion = self.sia_subsistema.bipartir(indices_alcance, indices_mecanismo)
                distribucion_marginal = particion.distribucion_marginal()
                mejor_emd = emd_efecto(distribucion_marginal, self.sia_dists_marginales)
                
                clave_candidata = (tuple(sorted(conjunto1)), tuple(sorted(conjunto2)))
                self.memoria_biparticiones[clave_candidata] = (mejor_emd, distribucion_marginal)
            except:
                mejor_emd = 1.0
        
        print(f"Mejor bipartición encontrada con EMD = {mejor_emd:.6f}")
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
            estrategia="Geometric-SIA",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_particion,
        )
    
    def __str__(self) -> str:
        return f"GeometricSIA(variables={len(getattr(self.sia_subsistema, 'indices_ncubos', []))}, " \
               f"estados_calculados={len(self.tabla_transiciones)}, " \
               f"biparticiones_evaluadas={len(self.memoria_biparticiones)})"