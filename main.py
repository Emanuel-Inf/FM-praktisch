"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import functools
import os

import data_reader

CSV_DIRECTORY: str = "./daten/csv/"
SEARCHED_CARD: str = "83764718"


def main():
    """Executes the main function."""

    files = os.listdir(CSV_DIRECTORY)  # Get all files in directory
    decks = []
    for f in files:
        csv_data = data_reader.read_csv(f"{CSV_DIRECTORY}/{f}")
        list_of_decks = data_reader.parse_data_to_list_with_deck_objects(csv_data)
        print(f"Lade Datei mit {len(list_of_decks)} Decks")
        decks.append(list_of_decks)

    flattened_decks = functools.reduce(lambda x, y: x + y, decks, [])

    count = 0
    for deck in flattened_decks:
        count += deck.contains_card(SEARCHED_CARD)

    print(f"The card {SEARCHED_CARD} was counted {count} times")


if __name__ == "__main__":
    main()
