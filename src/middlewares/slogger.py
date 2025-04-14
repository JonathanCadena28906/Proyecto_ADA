import sys
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Callable


class SafeLogger:
    """Logger seguro/robusto para manejar cualquier tipo de entrada y caracteres especiales."""

    def __init__(self, name: str):
        self._logger = self.__setup_logger(name)

    def _safe_str(self, obj: Any) -> str:
        """Convierte cualquier objeto a string de forma segura."""
        try:
            if isinstance(obj, (list, tuple, set, dict)):
                return str(obj)
            return str(obj).encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            return "[Objeto no representable]"

    def _safe_format(self, *args, **kwargs) -> str:
        """Formatea los argumentos de forma segura."""
        # Maneja args
        args_str = " ".join(self._safe_str(arg) for arg in args)

        # Maneja kwargs
        if kwargs:
            kwargs_str = " ".join(f"{k}={self._safe_str(v)}" for k, v in kwargs.items())
            return f"{args_str} {kwargs_str}"
        return args_str

    def __setup_logger(self, name: str) -> logging.Logger:
        """Configura el logger con manejo de encodings."""
        # Crear estructura de directorios
        base_log_dir = Path("logs")
        base_log_dir.mkdir(exist_ok=True)

        current_time = datetime.now()
        date_dir = base_log_dir / current_time.strftime("%d_%m_%Y")
        date_dir.mkdir(exist_ok=True)

        hour_dir = date_dir / f"{current_time.strftime('%H')}hrs"
        hour_dir.mkdir(exist_ok=True)

        log_file = hour_dir / f"{name}.log"

        # Configurar logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Handler para archivo con encoding UTF-8
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formato detallado
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log(self, level: int, *args, **kwargs) -> None:
        """Método genérico de logging."""
        message = self._safe_format(*args, **kwargs)
        self._logger.log(level, message)

    def debug(self, *args, **kwargs) -> None:
        """Log a nivel DEBUG."""
        self.log(logging.DEBUG, *args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        """Log a nivel INFO."""
        self.log(logging.INFO, *args, **kwargs)

    def warn(self, *args, **kwargs) -> None:
        """Log a nivel WARNING."""
        self.log(logging.WARNING, *args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        """Log a nivel ERROR."""
        self.log(logging.ERROR, *args, **kwargs)

    def critic(self, *args, **kwargs) -> None:
        """Log a nivel CRITICAL."""
        self.log(logging.CRITICAL, *args, **kwargs)


def get_logger(name: str) -> SafeLogger:
    """Función de conveniencia para obtener una instancia del logger."""
    return SafeLogger(name)


# Decorador opcional para logging automático
def log_execution(logger: SafeLogger):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"Iniciando {func.__name__}")
                result = func(*args, **kwargs)
                logger.debug(f"Completado {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {e}")
                raise

        return wrapper

    return decorator
