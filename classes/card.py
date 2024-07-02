"""Define Cards as objects for the script

@authors: Katrin Kober, Emanuel Petrinovic, Max Weise
"""

import dataclasses
from dataclasses import fields
from typing import Any, Dict

from ENUMS.cardtype import CardType


@dataclasses.dataclass
class Card:
    """Define a Yu-Gi-Oh Card.

    Attributes:
        name (str): The name of the card.
        card_prices (dict[str, str]): Prices for the card.
        card_type (CardType): The type of the card.
        views (int): How popular the card is.
        archetype (str): What archetype the card belongs to.
    """

    name: str
    card_prices: dict[str, str]
    card_type: CardType
    views: int
    archetype: str

    def __init__(self, name, card_prices, card_type: CardType, views, archetype):
        self.name = name
        self.card_prices = card_prices
        self.card_type = card_type
        self.views = views
        self.archetype = archetype

    def get_name(self) -> str:
        return self.name

    def get_card_prices(self) -> Dict[str, str]:
        return self.card_prices

    def get_card_type(self) -> CardType:
        return self.card_type

    def get_views(self) -> int:
        return self.views

    def get_archetype(self) -> str:
        return self.archetype

    def get_attribute(self, attr_name: str) -> Any:
        if attr_name in {field.name for field in fields(self)}:
            return getattr(self, attr_name)
        else:
            raise AttributeError(f"'Card' object has no attribute '{attr_name}'")

