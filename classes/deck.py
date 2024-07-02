"""Define Decks as collections of cards.

@authors: Katrin Kober, Emanuel Petrinovic, Max Weise
"""

import dataclasses


@dataclasses.dataclass
class Deck:
    """Define a deck object. Decks are collections of cards.

    Attributes:
        deck_num (str): The ID of the deck.
        name (str): The name of the deck.
        main_deck (list[str]): All cards of the main deck.
        format (str): The format of the deck.
    """

    deck_num: str
    name: str
    main_deck: list[str]
    format: str

    def contains_card(self, card_id: str) -> int:
        """Search for a card in the deck."""
        counted_cards = self.main_deck.count(card_id)
        return counted_cards
