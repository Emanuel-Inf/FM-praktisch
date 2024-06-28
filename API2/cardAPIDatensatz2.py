import json
from classes.card import Card
from ENUMS.cardtype import CardType
from classes.deck import Deck
import csv
cards_cache = {}
NEW_PATH_JSON = "./daten/csv/cards.csv"

def getCardFromCache(cardID: str) -> Card:
    card = cards_cache.get(cardID, {})
    
    new_card = Card(
        name=card.get("name", ""),
        card_prices=card.get("card_prices", [{}]),
        card_type=checkTyping(card.get("type", "")),
        views=card.get("views", "0"),  # Standardwert "0" falls `views` leer ist
        archetype=card.get("archetype")
    )
    
    return new_card

def load_cards():
    global cards_cache
    try:
        with open(NEW_PATH_JSON, "r", newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                card_id = row["id"]
                cards_cache[card_id] = {
                    "name": row["name"],
                    "type": row["type"],
                    "views": row["views"]
                }
            print("Karten erfolgreich geladen.")
    except FileNotFoundError:
        print(f"Fehler: Die Datei {NEW_PATH_JSON} wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

def has_valid_price_info(card: Card) -> bool:
    try:
        price_info = card.card_prices[0]
        return float(price_info.get('cardmarket_price', 0)) > 0
    except (ValueError, TypeError):
        return False

def getCardPriceSum(cardIDs: list[str]) -> float:
    return sum(float(getCardFromCache(card_id).views) for card_id in cardIDs)

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