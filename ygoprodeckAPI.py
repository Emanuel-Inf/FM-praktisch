import requests
from classes.card import Card
from classes.deck import Deck
import json
from ENUMS.cardtype import CardType
from jsonpath_ng import jsonpath, parse

PATH_CARDS_JSON = "./daten/cards/allcards.json"

def getCard(cardID: str) -> Card:
    api_url = "https://db.ygoprodeck.com/api/v7/cardinfo.php?id={0}".format(cardID)
    response = requests.get(api_url)
    response_json = response.json()

    response_card_prices = response_json["data"][0]["card_prices"]

    response_Card = Card(
        name=response_json["data"][0]["name"],
        card_prices=response_card_prices[0]
    )
    
    return response_Card

def deckHasCompletePriceInfo(deck: list[str]) -> bool:
    #Prüft, ob alle Karten in einem Deck gültige Preisinformationen haben.
    
    for card_id in deck:
        card = getCardFromLocal(card_id)
        if has_valid_price_info(card) == False:
            return False
    return True

def has_valid_price_info(card):
    if not card.card_prices or not isinstance(card.card_prices, list):
        return False
    if not isinstance(card.card_prices[0], dict):
        return False

    cardmarket_price = card.card_prices[0].get('cardmarket_price')
    if cardmarket_price: 
        try:
            if float(cardmarket_price) > 0:
                return True
        except ValueError:
            return False
    return False

def getCardPriceSum(cardIDs: list[str]) -> int:

    sum_card_prices = 0
    for id in cardIDs:
        card = getCardFromLocal(id)
        price_info = card.card_prices[0]
        if 'cardmarket_price' in price_info:
            sum_card_prices += float(price_info["cardmarket_price"])
    
    return sum_card_prices
    
def getCardFromLocal(cardID: str) -> Card:
    with open(PATH_CARDS_JSON, "r") as file:
     localJson = json.load(file)

    hit_card_name = ""
    hit_card_prices = dict[None,None]
    hit_card_types = None
    for card in localJson["data"]:
        if(str(card["id"]) == cardID):
            typing = checkTyping(card["type"])
            hit_card_name = card["name"]
            hit_card_prices=card.get("card_prices", [{}])
            hit_card_types=typing
            
    #karten_dict = {karte['id']: karte for karte in localJson['data']}
    #dict_card= karten_dict.get(int(cardID),None)

    #typing = checkTyping(dict_card["type"])
    return Card(
        name=hit_card_name,
        card_prices=hit_card_prices,
        card_type=hit_card_types
    )

def checkTyping(type: str)-> CardType:

    type_of_card = type
    if (type_of_card == "Spell Card"):
        type_of_card = CardType.Zauber
    elif (type_of_card == "Trap Card"):
        type_of_card = CardType.FALLEN
    elif(type_of_card == "Effect Monster" or "Tuner Monster" "Spirit Monster" or "Normal Monster"): 
        type_of_card = CardType.MONSTER
    else:
        print("what is going on")
    return type_of_card