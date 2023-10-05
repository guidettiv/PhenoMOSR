from abc import abstractmethod


class MOSRCallbackBase:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kwargs})"

    @abstractmethod
    def on_callback_set_init(self):
        pass

    @abstractmethod
    def on_initialization_start(self):
        pass

    @abstractmethod
    def on_initialization_end(self):
        pass

    @abstractmethod
    def on_generation_start(self):
        pass

    @abstractmethod
    def on_generation_end(self):
        pass

    @abstractmethod
    def on_offspring_generation_start(self):
        pass

    @abstractmethod
    def on_offspring_generation_end(self):
        pass

    @abstractmethod
    def on_refill_start(self):
        pass

    @abstractmethod
    def on_refill_end(self):
        pass

    @abstractmethod
    def on_stats_calculation_start(self):
        pass

    @abstractmethod
    def on_stats_calculation_end(self):
        pass

    @abstractmethod
    def on_convergence(self):
        pass

    @abstractmethod
    def on_training_completed(self):
        pass
