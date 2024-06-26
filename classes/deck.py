import dataclasses


@dataclasses.dataclass
class Deck:
    """Definiton of a Pokemon. This class is immutabel and sortable."""

    deck_num: str
    name: str
    format: str
    main_deck: list[str]

    def contains_card(self, card_id: str) -> int:
        """Search for a card in the deck."""
        counted_cards = self.main_deck.count(card_id)
        return counted_cards
