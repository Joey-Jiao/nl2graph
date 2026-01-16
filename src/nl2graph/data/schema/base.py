from abc import ABC, abstractmethod


class BaseSchema(ABC):
    @abstractmethod
    def to_prompt_string(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass
