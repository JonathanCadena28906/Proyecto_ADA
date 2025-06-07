from src.controllers.manager import Manager
from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.geometric import GeometricSIA
#from src.controllers.strategies.geometric_prueba import GeometricSIA
from src.controllers.strategies.phi import Phi


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "10000"  # Estado inicial del sistema
    condiciones =    "11111"  # Condiciones del sistema
    alcance =        "11111"  # Alcance de la solución
    mecanismo =      "11111"  # Mecanismo de solución

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = GeometricSIA(gestor_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)
