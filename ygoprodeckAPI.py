import requests
from classes.card import Card
import json

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

def getCardPriceSum(cardIDs: list[str]) -> int:

    sum_card_prices = 0
    for id in cardIDs:
        card = getCard(id)
        sum_card_prices += float(card.card_prices["cardmarket_price"])
    
    return sum_card_prices
    
def getCardFromLocal(cardID: str) -> Card:
    with open(PATH_CARDS_JSON, "r") as file:
     localJson = json.load(file)
    
    karten_dict = {karte['id']: karte for karte in localJson['data']}
    dict_card= karten_dict.get(int(cardID),None)

    return Card(
        name=dict_card["name"],
        card_prices=dict_card["card_prices"]
    )
    

