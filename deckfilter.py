from classes.deck import Deck
import ygoAPI
from ENUMS.formatTypes import FormatType
import random
import ygoAPI

def deck_Price_Filter(decks: list[Deck]) -> list[Deck]:
    
    fitered_decks = []

    for deck in decks:
        if (ygoAPI.deckHasCompletePriceInfo(deck)):
            fitered_decks.append(deck)

    return fitered_decks

def deck_Format_Filter(decks: list[Deck], filterType: list[FormatType], isKomplementMenge: bool) -> list[Deck]:

    format_filtered_decks = []

    if(isKomplementMenge):
        for deck in decks:
            if (deck.format not in [ft.value for ft in filterType]):
                format_filtered_decks.append(deck)
    else:
        for deck in decks:
            if (deck.format in [ft.value for ft in filterType]):
                format_filtered_decks.append(deck)

    return format_filtered_decks

def deck_Random_Filter(decks: list[Deck], sample_size: int) -> list[Deck]:

    sample_size = min(sample_size, len(decks))
    
    return random.sample(decks, sample_size)

def deck_HasComplete_PriceInfo_Filter(decks: list[Deck]) -> list[Deck]:

    complete_decks= []
    
    for deck in decks:
        isValid = False
        for card_id in deck.main_deck:
            card = ygoAPI.getCardFromCache(card_id)
            if ygoAPI.has_valid_price_info(card):
                isValid = True
        if(isValid):
            complete_decks.append(deck)
    return complete_decks
