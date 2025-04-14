from src.controllers.manager import Manager
from src.controllers.strategies.phi import Phi

def iniciar():
    """Punto de entrada principal"""
                   # ABCD #
    estado_inicio = "1000"
    condiciones =   "1110"
    alcance =       "1010"
    mecanismo =     "0110"

    config_sistema = Manager(estado_inicial=estado_inicio)

    ### Ejemplo de solución mediante Pyphi ###
    analizador_fi = Phi(config_sistema)
    sia_dos = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
    print(sia_dos.)