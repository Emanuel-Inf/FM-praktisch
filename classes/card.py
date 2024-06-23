import dataclasses
from classes.cardtype import CardType

@dataclasses.dataclass
class Card:
    """Definiton of a Pokemon. This class is immutabel and sortable."""

    name: str
    card_prices: dict[str,str]
    card_type: CardType

    def __init__(self, name, card_prices, card_type: CardType):
        self.name = name
        self.card_prices = card_prices
        self.card_type = card_type