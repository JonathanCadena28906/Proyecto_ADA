from src.middlewares.profile import profiler_manager
from src.models.base.application import aplicacion
from src.main import start_up


def main():
    """Inicializar el aplicativo."""
    profiler_manager.enabled = True



    start_up()


if __name__ == "__main__":
    main()
