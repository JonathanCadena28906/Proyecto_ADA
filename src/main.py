import os
import json
import time
import pandas as pd
from openpyxl import load_workbook
from src.controllers.manager import Manager


from src.controllers.manager import Manager
from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.geometric_secuencial import GeometricSia
from src.controllers.strategies.geometric_paralelo import GeometricSiaParallel
from src.controllers.strategies.phi import Phi


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "100000000000000"  # Estado inicial del sistema
    condiciones =    "111111111111111"  # Condiciones del sistema
    alcance =        "111111111111111"  # Alcance de la solución
    mecanismo =      "111111111111111"  # Mecanismo de solución

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = GeometricSiaParallel(gestor_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)
    
