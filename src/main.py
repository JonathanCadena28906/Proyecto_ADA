from src.controllers.manager import Manager

from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.q_nodes_paralela import QNodesParallel
from src.controllers.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "10000000000000000000"  # 1: activo, 0: inactivo
    condiciones =    "11111111111111111111"
    alcance =        "11111111111111111111"  # 1: activo, 0: inactivo
    mecanismo =      "11111111111111111111"

    gestor_sistema = Manager(estado_inicial)
    #Manager.generar_red(Manager, dimensiones=20)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = QNodesParallel(gestor_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)
