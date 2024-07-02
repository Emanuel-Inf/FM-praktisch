"""Define available card types.

@authors: Katrin Kober, Emanuel Petrinovic, Max Weise
"""

from enum import Enum


class CardType(Enum):
    MONSTER = "Monster"
    Zauber = "Zauber"
    FALLEN = "Fallen"
    UNBEKANNT = "Unknown"

