import dataclasses

@dataclasses.dataclass
class Card:
    """Definiton of a Pokemon. This class is immutabel and sortable."""

    name: str
    card_prices: dict[str,str]

    def __init__(self, name, card_prices):
        self.name = name
        self.card_prices = card_prices