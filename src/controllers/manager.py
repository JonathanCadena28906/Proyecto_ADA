from dataclasses import dataclass
from pathlib import Path
import time
import os

import numpy as np

from src.models.base.application import aplicacion
from src.constants.base import (
    ABC_START,
    CSV_EXTENSION,
    SAMPLES_PATH,
    RESOLVER_PATH,
)


@dataclass
class Manager:
    """El manejador es el encargado de en funcioón al tamaño del estado inicial y la página asociada, el traer el fichero de formato CSV con las TPM's almacenadas en `.samples/` para hacer una rápida depuración de los datos para la creación de sistemas.

    Args:
    ----
        - `estado_inicial` (str): Dado se manejan sistemas binarios es un número base dos de tamaño asociado a la red que se quiera cargar.
        - `pagina` (str): En la ruta de samples se tiene un literal asociado al tamaño de las redes por si se necesita añadir varias de un mismo tamaño.
        ruta_base (Path): Ruta donde se encuentran las muestras de TPMs en representación estado-nodo-on.

    Returns:
    -------
        None: Así mismo se encarga de asociar el directorio donde se mostrarán análisis sobre las ejecuciones, donde el programador haga uso del módulo de logging y profilling.
    """

    estado_inicial: str
    pagina: str = aplicacion.pagina_sample_network
    ruta_base: Path = Path(SAMPLES_PATH)
    

    @property
    def tpm_filename(self) -> Path:
        return (
            self.ruta_base / f"N{len(self.estado_inicial)}{self.pagina}.{CSV_EXTENSION}"
        )

    @property
    def output_dir(self) -> Path:
        return Path(
            f"{RESOLVER_PATH}/N{len(self.estado_inicial)}{self.pagina}/{''.join(self.estado_inicial)}"
        )

    def generar_red(self, dimensiones: int, datos_discretos: bool = True) -> str:
        np.random.seed(aplicacion.semilla_numpy) 
        
        if dimensiones < 1:
            raise ValueError("Las dimensiones deben ser positivas")

        # Calcular tamaño y tiempo estimado
        num_estados = 2**dimensiones
        total_size_gb = (num_estados * dimensiones) / (1024**3)
        estimated_time = total_size_gb * 2

        print(f"Tamaño estimado: {total_size_gb:.6f} GB")
        print(f"Tiempo estimado: {estimated_time:.1f} segundos")

        if total_size_gb > 1:
            if (
                input("El sistema ocupará más de 1GB. ¿Continuar? (s/n): ").lower()
                != "s"
            ):
                return None

        # Verificar archivos existentes y generar nuevo nombre
        base_path = Path("src/.samples")
        base_path.mkdir(parents=True, exist_ok=True)

        suffix = ABC_START
        while (base_path / f"N{dimensiones}{suffix}.csv").exists():
            if (
                input(
                    f"Ya existe N{dimensiones}{suffix}.csv. ¿Generar nueva red? (s/n): "
                ).lower()
                != "s"
            ):
                return f"N{dimensiones}{suffix}.{CSV_EXTENSION}"
            suffix = chr(ord(suffix) + 1)

        filename = f"N{dimensiones}{suffix}.csv"
        filepath = base_path / filename

        # Generar estados
        print("Generando estados...")
        start_time = time.time()

        if datos_discretos:
            states = np.random.randint(
                2, size=(num_estados, dimensiones), dtype=np.int8
            )
        else:
            states = np.random.random(size=(num_estados, dimensiones))

        print(f"Generación completada en {time.time() - start_time:.2f} segundos")

        # Guardar archivo
        print(f"Guardando en {filepath}...")
        start_time = time.time()
        np.savetxt(
            filepath, states, delimiter=",", fmt="%d" if datos_discretos else "%.6f"
        )

        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"Archivo guardado: {file_size_gb:.6f} GB")
        print(f"Tiempo de guardado: {time.time() - start_time:.2f} segundos")

        return filename
