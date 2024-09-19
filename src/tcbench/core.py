from __future__ import annotations

from enum import Enum

class StringEnum(Enum):
    @classmethod
    def from_str(cls, text):
        for member in cls.__members__.values():
            if member.value == text:
                return member
        return None

    @classmethod
    def values(cls):
        return [x.value for x in list(cls)]

    def __str__(self):
        return self.value

