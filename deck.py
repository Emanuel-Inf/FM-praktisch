class Deck:
    """Definiton of a Pokemon. This class is immutabel and sortable."""
    deck_num: str
    name: str
    main_deck: list[str]

    def __init__(self, name: str, deck_num: str, main_deck: list[str]):
        self.name = name
        self.deck_num = deck_num
        self.main_deck = main_deck
    
    def __repr__(self):
        return (f"Deck(name={self.name!r}, deck_num={self.deck_num!r}, "
                f"main_deck={self.main_deck!r})")
    
    def __eq__(self, other):
        if isinstance(other, Deck):
            return (self.name == other.name and
                    self.deck_num == other.deck_num and
                    self.main_deck == other.main_deck)
        return False