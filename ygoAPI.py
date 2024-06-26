import json
from classes.card import Card
from ENUMS.cardtype import CardType
from classes.deck import Deck

PATH_CARDS_JSON = "./daten/cards/allcards.json"
cards_cache = {}

def load_cards():
    with open(PATH_CARDS_JSON, "r") as file:
        data = json.load(file)
        global cards_cache
        cards_cache = {str(card["id"]): card for card in data["data"]}

def getCardFromCache(cardID: str) -> Card:
    card = cards_cache.get(cardID, {})
    return Card(
        name=card.get("name", ""),
        card_prices=card.get("card_prices", [{}]),
        card_type=checkTyping(card.get("type", ""))
    )

def has_valid_price_info(card: Card) -> bool:
    try:
        price_info = card.card_prices[0]
        return float(price_info.get('cardmarket_price', 0)) > 0
    except (ValueError, TypeError):
        return False

def getCardPriceSum(cardIDs: list[str]) -> float:
    return sum(float(getCardFromCache(card_id).card_prices[0].get("cardmarket_price", 0)) for card_id in cardIDs)

def checkTyping(type: str) -> CardType:
    type_mappings = {
        "Spell Card": CardType.Zauber,
        "Trap Card": CardType.FALLEN,
        "Effect Monster": CardType.MONSTER,
        "Tuner Monster": CardType.MONSTER,
        "Spirit Monster": CardType.MONSTER,
        "Normal Monster": CardType.MONSTER
    }
    return type_mappings.get(type, CardType.UNBEKANNT)

# Beim Start des Programms:
load_cards()
