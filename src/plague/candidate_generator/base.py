import abc
from typing import Any, Dict, List


class CandidateGenerator(abc.ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

    @abc.abstractmethod
    def generate_candidates(self, jailbreak_method_instance, **kwargs) -> List[str]:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
