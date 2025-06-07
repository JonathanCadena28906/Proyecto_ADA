import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from src.middlewares.slogger import SafeLogger
from src.funcs.base import ABECEDARY, emd_efecto
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import EFECTO, ACTUAL
from src.funcs.format import fmt_biparte_q

class GeometricSIA(SIA):  
    """
    Estrategia GeometricSIA que implementa análisis geométrico de hipercubos
    con representación n-dimensional y cálculo de costos de transición.
    """
    
    def __init__(self, config: Manager):  # Cambio aquí: _init_ → __init__
        super().__init__(config)  # Cambio aquí: _init_ → __init__
        self.logger = SafeLogger("geometric_sia")
        self.tensores = []
        self.tabla_transiciones = {}
        self.cache_hamming = {}
        self.cache_costos = {}
        
    def aplicar_estrategia(self, conditions, purview, mechanism):
        """
        Método principal que ejecuta la estrategia GeometricSIA.
        """
        # Llamar a la función que prepara el subsistema (heredada de SIA)
        self.sia_preparar_subsistema(conditions, purview, mechanism)
        
        # Construir representación n-dimensional (tensores)
        self._construir_representacion_ndimensional()
        
        # Mostrar información de los tensores y sus variables correspondientes
        #self._mostrar_tensores_y_variables()
        
        # Calcular tabla de costos
        self._calcular_tabla_costos()
        
        # Identificar bipartición óptima (nueva funcionalidad)
        biparticion_optima = self._identificar_biparticion_optima()
        
        # Formatear y retornar resultado
        return self._formatear_resultado(biparticion_optima)

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
        
        # Descomponer en tensores elementales
        for i, ncube in enumerate(self.sia_subsistema.ncubos):
            # Extraer el tensor de probabilidades condicionales
            tensor_original = ncube.data.copy()
            
            # Reordenar tensor para formato little-endian
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
    
    def _mostrar_tensores_y_variables(self):
        """
        Muestra información de los tensores y las variables a las que corresponden.
        """
        #print("\n" + "="*60)
        #print("TENSORES Y VARIABLES CORRESPONDIENTES")
        #print("="*60)
        
        # Obtener las variables correspondientes usando el abecedario
        variables = []
        for i, indice_ncube in enumerate(self.sia_subsistema.indices_ncubos):
            if indice_ncube < len(ABECEDARY):
                variable = ABECEDARY[indice_ncube].lower()
                variables.append(variable)
            else:
                variable = f"var_{indice_ncube}"
                variables.append(variable)
        
        #print(f"Número total de tensores: {len(self.tensores)}")
        #print(f"Variables del sistema: {variables}")
        #print()
        
        # Mostrar información detallada de cada tensor
        #for i, (tensor, variable) in enumerate(zip(self.tensores, variables)):
            #print(f"TENSOR {i} - Variable '{variable.upper()}':")
            #print(f"  Índice original del n-cubo: {self.sia_subsistema.indices_ncubos[i]}")
            #print(f"  Forma del tensor: {tensor.shape}")
            #print(f"  Dimensiones activas: {self.sia_subsistema.ncubos[i].dims}")
            #print(f"  Datos del tensor (little-endian):")
            
            # Mostrar los datos del tensor de forma estructurada
            #if tensor.ndim <= 3:
                #print(f"    {tensor}")
            #else:
                #print(f"    [Tensor de alta dimensionalidad - forma: {tensor.shape}]")
            #print()
        
        # Mostrar correspondencia con notación little-endian
        #n_bits = len(self.sia_subsistema.dims_ncubos)
        #print("CORRESPONDENCIA DE ESTADOS (Little-Endian):")
        #print("-" * 40)
        #for estado in range(min(8, 2**n_bits)):  # Mostrar solo los primeros 8 estados
            #binary_little = self._decimal_to_little_endian_binary(estado, n_bits)
            #print(f"  Estado {estado}: {binary_little}")
        #if 2**n_bits > 8:
            #print(f"  ... (y {2**n_bits - 8} estados más)")
        #print()
    
    def _decimal_to_little_endian_binary(self, decimal: int, n_bits: int) -> str:
        """
        Convierte un número decimal a su representación binaria little-endian.
        """
        # Obtener representación binaria estándar
        binary = format(decimal, f'0{n_bits}b')
        # Invertir para little-endian (bit menos significativo a la izquierda)
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
        
        # Para cada bit, alternar su valor para obtener vecinos
        for bit_pos in range(n_bits):
            vecino = estado ^ (1 << bit_pos)
            vecinos.append(vecino)
            
        return vecinos
    
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

        self.logger.info("Calculando tabla de costos...")
        self.logger.info(f"Estados totales: {n_estados}")
        self.logger.info(f"Variables: {n_vars}")

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
        Mapea correctamente las dimensiones activas a las letras del abecedario.
        """
        # Obtener nombres de variables y sus dimensiones activas
        nombres_variables = []
        dims_activas_por_variable = []
        
        # Paso 1: Recopilar información sobre las dimensiones activas de cada variable
        for i, ncube in enumerate(self.sia_subsistema.ncubos):
            # Obtener dimensiones activas para esta variable
            dims_activas = ncube.dims
            dims_activas_por_variable.append(dims_activas)
            
            # Obtener nombre de la variable según su índice original
            indice_ncube = self.sia_subsistema.indices_ncubos[i]
            if indice_ncube < len(ABECEDARY):
                variable = ABECEDARY[indice_ncube].upper()
            else:
                variable = f"VAR_{indice_ncube}"
            nombres_variables.append(variable)
        
        # Registrar dimensiones activas del mecanismo para depuración
        #print("\n" + "="*60)
        #print("DIMENSIONES ACTIVAS DEL MECANISMO")
        #print("="*60)
        #for i, (var, dims) in enumerate(zip(nombres_variables, dims_activas_por_variable)):
            #print(f"Variable {var}: Dimensiones activas {dims} (índice {i})")
        #print("="*60 + "\n")
        
        # Crear hojas separadas para cada variable
        filename = f"tabla_costos_geometric_sia.xlsx"
        with pd.ExcelWriter(filename) as writer:
            
            for var_idx in range(n_vars):
                # Crear matriz de costos para esta variable
                matriz_costos = np.zeros((n_estados, n_estados))
                
                for i in range(n_estados):
                    for j in range(n_estados):
                        clave = (var_idx, i, j)
                        matriz_costos[i, j] = self.tabla_transiciones.get(clave, 0.0)
                
                # Crear etiquetas con variables activas según las dimensiones activas
                indices_mejorados = []
                columnas_mejoradas = []
                
                # Dimensiones activas para esta variable específica
                dims_activas = dims_activas_por_variable[var_idx]
                
                for estado in range(n_estados):
                    # Obtener representación binaria
                    bits = [(estado >> bit) & 1 for bit in range(n_bits)]
                    
                    # Identificar qué letras están activas basado en dimensiones activas
                    letras_activas = []
                    
                    # Para cada posición activa en los bits, buscar la letra correspondiente
                    for pos_bit, valor_bit in enumerate(bits):
                        if valor_bit == 1 and pos_bit < len(dims_activas):
                            # Obtener la dimensión real a la que corresponde este bit
                            dim_real = dims_activas[pos_bit]
                            
                            # Convertir esta dimensión a una letra
                            if dim_real < len(ABECEDARY):
                                letra = ABECEDARY[dim_real].upper()
                                letras_activas.append(letra)
                            else:
                                letras_activas.append(f"VAR_{dim_real}")
                    
                    # Crear etiqueta con formato: número (letras_activas) [bits]
                    binary_rep = ''.join(str(b) for b in bits[:len(dims_activas)])  # Solo mostrar bits relevantes
                    etiqueta = f"{estado} ({','.join(letras_activas)}) [{binary_rep}]"
                    indices_mejorados.append(etiqueta)
                    columnas_mejoradas.append(etiqueta)
                
                # Crear DataFrame con las nuevas etiquetas
                df = pd.DataFrame(matriz_costos, index=indices_mejorados, columns=columnas_mejoradas)
                
                # Guardar en hoja específica con nombre de variable
                nombre_hoja = f"Variable_{nombres_variables[var_idx]}"
                df.to_excel(writer, sheet_name=nombre_hoja)
        
        self.logger.info(f"Tabla de costos exportada a: {filename}")
        
        # Mostrar resumen
        #print(f"\nTabla de costos calculada y exportada a: {filename}")
        #print(f"Formato: estado (letras_activas) [representación_binaria]")
        #print("Hojas creadas:")
        #for var_idx in range(n_vars):
            #print(f"  - Variable_{nombres_variables[var_idx]} (dimensiones activas: {dims_activas_por_variable[var_idx]})")
    
    def _calcular_complemento_estado(self, estado_inicial: int, estado_destino: int, n_bits: int) -> int:
        """
        Calcula el estado complementario:
        - Identifica qué bits cambiaron en la transición original
        - El complementario cambia los bits que NO cambiaron en la original
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

    def _identificar_bits_cambiados(self, estado_inicial: int, estado_destino: int, n_bits: int) -> List[int]:
        """
        Identifica qué bits cambiaron entre dos estados y retorna sus posiciones.
        """
        diferencia = estado_inicial ^ estado_destino
        bits_cambiados = []
        
        for bit_pos in range(n_bits):
            if (diferencia >> bit_pos) & 1:
                bits_cambiados.append(bit_pos)
        
        return bits_cambiados

    def _calcular_emd_biparticion(self, conjunto_presente: Set[int], conjunto_futuro: Set[int]) -> float:
        """
        Calcula el EMD para una bipartición específica.
        
        Args:
            conjunto_presente: Variables que van al estado presente (ACTUAL)
            conjunto_futuro: Variables que van al estado futuro (EFECTO)
            
        Returns:
            float: Valor del EMD
        """
        try:
            # Convertir a arrays numpy
            indices_presente = np.array(list(conjunto_presente), dtype=np.int8)
            indices_futuro = np.array(list(conjunto_futuro), dtype=np.int8)
            
            #print(f"    Debug EMD - Presente (ACTUAL): {[chr(65 + i) for i in sorted(conjunto_presente)]}")
            #print(f"    Debug EMD - Futuro (EFECTO): {[chr(65 + i) for i in sorted(conjunto_futuro)]}")
            
            # Asegurar que ambos conjuntos tengan al menos un elemento
            if len(indices_presente) == 0:
                indices_presente = np.array([0], dtype=np.int8)
            if len(indices_futuro) == 0:
                indices_futuro = np.array([1 if len(self.sia_subsistema.indices_ncubos) > 1 else 0], dtype=np.int8)
            
            # Realizar bipartición
            particion = self.sia_subsistema.bipartir(indices_futuro, indices_presente)
            distribucion_particion = particion.distribucion_marginal()
            
            # Calcular EMD
            emd = emd_efecto(distribucion_particion, self.sia_dists_marginales)
            #print(f"    Debug EMD - Valor calculado: {emd:.6f}")
            return emd
            
        except Exception as e:
            self.logger.error(f"Error calculando EMD: {e}")
            return float('inf')

    def _identificar_biparticion_optima(self) -> Tuple[Set[int], Set[int]]:
        """
        Identifica la bipartición óptima evaluando primero el caso especial 000→111
        y luego analizando otras transiciones y sus complementos.
        
        Returns:
            Tuple con los dos conjuntos de variables de la bipartición óptima
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_bits = len(self.sia_subsistema.dims_ncubos)
        n_estados = 2 ** n_bits
        
        # Estado inicial (000...0) y estado todos unos (111...1)
        estado_inicial = 0
        estado_todos_unos = 2**n_bits - 1
        
        # Lista para almacenar TODAS las particiones candidatas, incluida la especial
        particiones_candidatas = []
        
        # CASO ESPECIAL: Evaluar transición 000→111 directamente sin complemento
        print(f"\n=== CASO ESPECIAL: EVALUANDO TRANSICIÓN 000→111 ===")
        print(f"Evaluando t(000, {self._decimal_to_little_endian_binary(estado_todos_unos, n_bits)})")
        
        costos_variables_todos_unos = []
        
        for var_idx in range(n_vars):
            clave = (var_idx, estado_inicial, estado_todos_unos)
            costo = self.tabla_transiciones.get(clave, float('inf'))
            var_letra = chr(65 + var_idx)
            costos_variables_todos_unos.append((var_idx, costo, var_letra))
            print(f"Variable {var_letra}: costo = {costo:.4f}")
        
        if costos_variables_todos_unos and all(c[1] < float('inf') for c in costos_variables_todos_unos):
            # CAMBIO: Evaluar todas las variables como candidatas
            print("\n=== EVALUANDO TODAS LAS VARIABLES COMO CANDIDATAS PARA CASO ESPECIAL ===")
            for var_idx, costo, var_letra in costos_variables_todos_unos:
                print(f"\nEvaluando variable {var_letra} como candidata:")
                
                # Crear bipartición con esta variable en presente
                conjunto_presente = {var_idx}
                conjunto_futuro = set(range(n_vars)) - conjunto_presente
                
                print(f"Bipartición candidata:")
                print(f"  Presente: {[chr(65 + i) for i in sorted(conjunto_presente)]}")
                print(f"  Futuro: {[chr(65 + i) for i in sorted(conjunto_futuro)]}")
                
                # Calcular EMD para esta bipartición
                emd_valor = self._calcular_emd_biparticion(conjunto_presente, conjunto_futuro)
                print(f"  EMD calculado: {emd_valor:.6f}")
                if emd_valor < float('inf'):
                    particiones_candidatas.append({
                        'transicion_original': (estado_inicial, estado_todos_unos),
                        'transicion_original_bin': f"t(000, {self._decimal_to_little_endian_binary(estado_todos_unos, n_bits)})",
                        'promedio_original': costo,
                        'conjunto_presente': conjunto_presente,
                        'conjunto_futuro': conjunto_futuro,
                        'emd': emd_valor,
                        'info_debug': {
                            'caso_especial': True,
                            'variable_evaluada': var_letra,
                            'costo_variable': costo,
                            'costos_originales': [c[1] for c in costos_variables_todos_unos]
                        }
                    })
                    
                    # STOP TEMPRANO: Si EMD = 0, retornar inmediatamente
                    if emd_valor == 0:
                        #print(f"  ¡EMD perfecto (0) con variable {var_letra}! Retornando solución óptima.")
                        return (conjunto_presente, conjunto_futuro)
                else:
                    print(f"  EMD inválido para variable {var_letra}, descartada")
            
            print("  Todas las variables del caso especial evaluadas, continuando análisis...")
        else:
            print("  No se pudieron evaluar costos para 000→111, continuando con análisis estándar")
        
        # ANÁLISIS ESTÁNDAR (excluyendo 000 y 111)
        print(f"\n=== ANÁLISIS DE TRANSICIONES DESDE ESTADO INICIAL {self._decimal_to_little_endian_binary(estado_inicial, n_bits)} ===")
        
        # PASO 1: Analizar todas las transiciones (excluyendo 000 y 111)
        transiciones_con_promedio = []
        
        for estado_destino in range(1, n_estados):
            # CAMBIO: Excluir explícitamente el estado todos unos (111)
            if estado_destino == estado_todos_unos:
                #print(f"\nSaltando transición t(000, {self._decimal_to_little_endian_binary(estado_destino, n_bits)}) (ya evaluada en caso especial)")
                continue
                
            estado_destino_bin = self._decimal_to_little_endian_binary(estado_destino, n_bits)
            #print(f"\n--- ANALIZANDO TRANSICIÓN t(000, {estado_destino_bin}) ---")
            
            # Obtener costos para cada variable en esta transición
            costos_variables = []
            for var_idx in range(n_vars):
                clave = (var_idx, estado_inicial, estado_destino)
                costo = self.tabla_transiciones.get(clave, float('inf'))
                # Convertir variable index a letra para mostrar
                var_letra = chr(65 + var_idx)  # A, B, C...
                costos_variables.append(costo)
                #print(f"Variable {var_letra}: costo = {costo:.4f}")
            
             # Calcular promedio de costos
            if costos_variables and all(c < float('inf') for c in costos_variables):
                promedio = sum(costos_variables) / len(costos_variables)
                # print(f"Promedio de costos: {promedio:.4f}")
                
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
            #print("No se encontraron transiciones válidas")
            if particiones_candidatas:
                #print("Usando partición del caso especial")
                mejor_particion = min(particiones_candidatas, key=lambda x: x['emd'])
                return (mejor_particion['conjunto_presente'], mejor_particion['conjunto_futuro'])
            else:
                # Bipartición por defecto
                mitad = n_vars // 2
                return (set(range(mitad)), set(range(mitad, n_vars)))
        
        # Ordenar por promedio ascendente (menor costo primero)
        transiciones_con_promedio.sort(key=lambda x: x['promedio'])
        
        #print(f"\n=== TRANSICIONES ORDENADAS POR PROMEDIO (MEJOR PRIMERO) ===")
        #for i, trans in enumerate(transiciones_con_promedio[:5]):  # Mostrar top 5
           # print(f"{i+1}. t(000, {trans['estado_destino_bin']}) - Promedio: {trans['promedio']:.4f}")
        
        # Seleccionar las mejores transiciones (top 30% o mínimo 3)
        num_mejores = max(3, len(transiciones_con_promedio) // 3)
        mejores_transiciones = transiciones_con_promedio[:num_mejores]
        
        #print(f"\n=== ANALIZANDO {len(mejores_transiciones)} MEJORES TRANSICIONES PARA PARTICIONES ===")
        
        # PASO 3: Para cada transición seleccionada, analizar con su complemento
        for trans in mejores_transiciones:
            estado_destino = trans['estado_destino']
            costos_originales = trans['costos_variables']
            
           # print(f"\n--- ANALIZANDO TRANSICIÓN t(000, {trans['estado_destino_bin']}) ---")
            #print(f"Costos originales: {[f'{chr(65+i)}={c:.4f}' for i, c in enumerate(costos_originales)]}")
            
            # PASO 3.1: Calcular estado complemento
            estado_complemento = self._calcular_complemento_estado(estado_inicial, estado_destino, n_bits)
            estado_complemento_bin = self._decimal_to_little_endian_binary(estado_complemento, n_bits)
            
            #print(f"Estado complemento: t(000, {estado_complemento_bin})")
            
            # PASO 3.2: Obtener costos del complemento
            costos_complemento = []
            for var_idx in range(n_vars):
                clave = (var_idx, estado_inicial, estado_complemento)
                costo = self.tabla_transiciones.get(clave, float('inf'))
                costos_complemento.append(costo)
            
            #print(f"Costos complemento: {[f'{chr(65+i)}={c:.4f}' for i, c in enumerate(costos_complemento)]}")
            
            # PASO 3.3: Comparar variable por variable y seleccionar las de menor costo
            variables_presente = []  # Variables que van al presente
            variables_futuro = []    # Variables que van al futuro
            
            #print("Comparación variable por variable:")
            for var_idx in range(n_vars):
                costo_original = costos_originales[var_idx]
                costo_complemento = costos_complemento[var_idx]
                var_letra = chr(65 + var_idx)
                
                if costo_original <= costo_complemento:
                    variables_presente.append(var_idx)
                    #print(f"Variable {var_letra}: PRESENTE (costo {costo_original:.4f} <= {costo_complemento:.4f})")
                else:
                    variables_futuro.append(var_idx)
                    #print(f"Variable {var_letra}: FUTURO (costo {costo_complemento:.4f} < {costo_original:.4f})")
            
            # PASO 3.4: Analizar el bit cambiante
            bits_cambiados = self._identificar_bits_cambiados(estado_inicial, estado_destino, n_bits)
            #print(f"Bits que cambiaron en transición original: {bits_cambiados}")
            
            if bits_cambiados:
                # El primer bit que cambia determina qué variable debe estar en presente
                bit_principal = bits_cambiados[0]  # Bit más significativo que cambió
                var_bit_principal = bit_principal  # Variable correspondiente al bit
                var_letra_principal = chr(65 + var_bit_principal)
                
                #print(f"Variable principal que cambia: {var_letra_principal} (bit {bit_principal})")
                
                # PASO 3.5: Ajustar partición según regla del bit cambiante
                # La variable del bit que cambia DEBE estar en presente
                if var_bit_principal not in variables_presente:
                    #print(f"AJUSTE: Moviendo variable {var_letra_principal} de futuro a presente (regla del bit cambiante)")
                    if var_bit_principal in variables_futuro:
                        variables_futuro.remove(var_bit_principal)
                    variables_presente.append(var_bit_principal)
        
            # PASO 3.6: Asegurar que ambos conjuntos tengan elementos
            if not variables_presente:
                # Mover la primera variable del futuro al presente
                if variables_futuro:
                    var_movida = variables_futuro.pop(0)
                    variables_presente.append(var_movida)
                    #print(f"AJUSTE: Moviendo variable {chr(65 + var_movida)} a presente (conjunto vacío)")
    
            if not variables_futuro:
                # Mover la última variable del presente al futuro
                if len(variables_presente) > 1:
                    var_movida = variables_presente.pop(-1)
                    variables_futuro.append(var_movida)
                    #print(f"AJUSTE: Moviendo variable {chr(65 + var_movida)} a futuro (conjunto vacío)")
    
            # PASO 3.7: Formar la partición
            conjunto_presente = set(variables_presente)
            conjunto_futuro = set(variables_futuro)
            
            #print(f"PARTICIÓN RESULTANTE:")
            #print(f"  Presente: {[chr(65 + i) for i in sorted(conjunto_presente)]}")
            #print(f"  Futuro: {[chr(65 + i) for i in sorted(conjunto_futuro)]}")
            
            # PASO 3.8: Calcular EMD para esta partición
            emd_valor = self._calcular_emd_biparticion(conjunto_presente, conjunto_futuro)
            
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
                        'bits_cambiados': bits_cambiados
                    }
                })
                
                #print(f"  EMD calculado: {emd_valor:.6f}")
                
                # STOP TEMPRANO: Si EMD = 0, retornar inmediatamente
                if emd_valor == 0:
                    #print("  ¡EMD perfecto (0)! Retornando solución óptima.")
                    return (conjunto_presente, conjunto_futuro)
            else:
                print("  EMD inválido, partición descartada")

        # PASO 4: Seleccionar la mejor partición basada en EMD mínimo
        if particiones_candidatas:
            # CAMBIO: Puede elegir entre candidatas normales y la especial
            mejor_particion = min(particiones_candidatas, key=lambda x: x['emd'])
            
            # Identificar si es la partición del caso especial
            es_caso_especial = mejor_particion.get('info_debug', {}).get('caso_especial', False)
            
            #print(f"\n=== MEJOR PARTICIÓN ENCONTRADA ===")
            if es_caso_especial:
                print(f"TIPO: CASO ESPECIAL 000→111")
            else:
                print(f"TIPO: TRANSICIÓN ESTÁNDAR")
            print(f"Transición base: {mejor_particion['transicion_original_bin']}")
            print(f"Promedio de costos: {mejor_particion['promedio_original']:.6f}")
            print(f"Presente: {[chr(65 + i) for i in sorted(mejor_particion['conjunto_presente'])]}")
            print(f"Futuro: {[chr(65 + i) for i in sorted(mejor_particion['conjunto_futuro'])]}")
            print(f"EMD final: {mejor_particion['emd']:.6f}")
            
            return (mejor_particion['conjunto_presente'], mejor_particion['conjunto_futuro'])
        else:
            print("\n=== NO SE ENCONTRARON PARTICIONES VÁLIDAS - USANDO PARTICIÓN POR DEFECTO ===")
            # Bipartición por defecto
            mitad = n_vars // 2
            return (set(range(mitad)), set(range(mitad, n_vars)))

    def _formatear_resultado(self, biparticion_optima: Tuple[Set[int], Set[int]]) -> Solution:
        """
        Formatea el resultado en formato compatible con el sistema.
        
        Args:
            biparticion_optima: Tuple con los conjuntos presente y futuro
            
        Returns:
            Solution: Objeto solución formateado
        """
        conjunto_presente, conjunto_futuro = biparticion_optima
        
        try:
            # Convertir a arrays numpy
            indices_presente = np.array(list(conjunto_presente), dtype=np.int8)
            indices_futuro = np.array(list(conjunto_futuro), dtype=np.int8)
            
            # Asegurar que ambos conjuntos tengan al menos un elemento
            if len(indices_presente) == 0:
                indices_presente = np.array([0], dtype=np.int8)
            if len(indices_futuro) == 0:
                indices_futuro = np.array([1 if len(self.sia_subsistema.indices_ncubos) > 1 else 0], dtype=np.int8)
            
            # Realizar bipartición
            particion = self.sia_subsistema.bipartir(indices_presente, indices_futuro)
            distribucion_particion = particion.distribucion_marginal()
            
            # Calcular EMD
            perdida = emd_efecto(distribucion_particion, self.sia_dists_marginales)
            
            # Crear nodos en formato (tiempo, índice)
            nodos_seleccionados = []
            
            # Presente -> ACTUAL
            for idx in conjunto_presente:
                nodos_seleccionados.append((ACTUAL, idx))
            
            # Futuro -> EFECTO
            for idx in conjunto_futuro:
                nodos_seleccionados.append((EFECTO, idx))
            
            # Obtener complemento
            todos_nodos = set()
            n_vars = len(self.sia_subsistema.indices_ncubos)
            
            for i in range(n_vars):
                todos_nodos.add((EFECTO, i))
                todos_nodos.add((ACTUAL, i))
            
            nodos_complemento = list(todos_nodos - set(nodos_seleccionados))
            
            # Formatear partición
            fmt_particion = fmt_biparte_q(nodos_seleccionados, nodos_complemento)
            
        except Exception as e:
            self.logger.error(f"Error en formateo final: {e}")
            perdida = 1.0
            distribucion_particion = self.sia_dists_marginales
            fmt_particion = {"mecanismo": [], "alcance": []}
        
        return Solution(
            estrategia="GeometricSIA",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_particion
        )
