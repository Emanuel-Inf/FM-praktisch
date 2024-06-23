from classes.deck import Deck
from classes.card import Card
from classes.cardtype import CardType
import ygoprodeckAPI

def GetAllCardsInDeck(deck: Deck) -> list[Card]:
    
    list_of_cards_in_deck = []
    for card in deck.main_deck:
        list_of_cards_in_deck.append(ygoprodeckAPI.getCardFromLocal(card))
        
    return list_of_cards_in_deck
    
def getAnzahlKartenTypen(cards: list[Card]) -> dict[str, int]:

    counterMonster = 0
    counterZauber = 0
    counterFallen = 0
   
    for card in cards:
        if(card.card_type == CardType.MONSTER):
            counterMonster += 1
        elif(card.card_type == CardType.Zauber):
            counterZauber += 1
        else:
            counterFallen += 1
    dict_results = {
        "Monster": counterMonster,
        "Zauber": counterZauber,
        "Fallen": counterFallen
    }
    return dict_results