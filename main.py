"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import functools
import os
import random
import ygoprodeckAPI
import data_reader
from classes.cardtype import CardType
import logRegression

CSV_DIRECTORY: str = "./daten/csv/"
SEARCH_CARDS = ["83764719", "47826112", "3643300", "53804307", "81843628", "36553319"]
SAMPLE_SIZE: int = 30


def main():
    """Executes the main function."""

    # Read all data from the data directory
    files = os.listdir(CSV_DIRECTORY)
    decks = []
    for f in files:
        csv_data = data_reader.read_csv(f"{CSV_DIRECTORY}/{f}")
        list_of_decks = data_reader.parse_data_to_list_with_deck_objects(csv_data)
        decks.append(list_of_decks)

    # Prepare the data
    flattened_decks = functools.reduce(lambda x, y: x + y, decks, [])
    random_sample = random.sample(
        flattened_decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(flattened_decks)
    )

    # Count the frequency of the specified cards in decks
    card_frequency = {}
    for card in SEARCH_CARDS:
        card_frequency[card] = 0
        count = 0
        for deck in random_sample:
            count += deck.contains_card(card)
        card_frequency[card] += count

    #print(card_frequency)

    # TODO: Calculate the confidence intervall

    #print(ygoprodeckAPI.getCardPriceSum(SEARCH_CARDS))
    #card = ygoprodeckAPI.getCardFromLocal("3643300")
    for deck in random_sample:
        list_of_cards = logRegression.GetAllCardsInDeck(deck)
        print(logRegression.getAnzahlKartenTypen(list_of_cards))
    
if __name__ == "__main__":
    main()
