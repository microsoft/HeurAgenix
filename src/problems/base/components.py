class BaseSolution:
    """Base solution for CO problem."""
    def __init__(self, **kwargs):
        pass

    def __str__(self) -> str:
        pass


class BaseOperator:
    """Base operator that update the solution."""
    def __init__(self, **kwargs):
        pass

    def run(self, solution: BaseSolution) -> BaseSolution:
        pass

    def __str__(self) -> str:
        params = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"