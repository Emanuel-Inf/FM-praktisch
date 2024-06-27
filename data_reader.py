"""This file contains functions to read and compile data.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import csv
import functools
import os
from typing import Any

from classes.deck import Deck

CSV_DIRECTORY: str = "./daten/csv/"


def get_All_Decks_Prepaired() -> list[Deck]:

    # Read all data from the data directory
    files = os.listdir(CSV_DIRECTORY)

    decks = []
    for f in files:
        csv_data = read_csv(f"{CSV_DIRECTORY}/{f}")
        list_of_decks = parse_data_to_list_with_deck_objects(csv_data)
        decks.append(list_of_decks)

    # Prepare the data
    flattened_decks = functools.reduce(lambda x, y: x + y, decks, [])
    flattened_decks = flattened_decks[
        1:
    ]  # Deck(deck_num='deck_num', name='deck_name', main_deck=['main_deck'])

    return flattened_decks


def read_csv(file_name: str) -> list[tuple[str, ...]]:
    """Read all values from file_name."""
    with open(file_name, "r", newline="", encoding="utf-8") as file_contents:
        csv_reader = csv.reader(file_contents, delimiter=",")
        return [tuple(item) for item in csv_reader]


def parse_data_to_list_with_deck_objects(data_vector: list[Any]) -> list[Deck]:
    """Get a list of data and parse it to a list with Decks.

    Args:
        data_vector: The list containing data representing a deck.

    Returns:
        list_of_decks: Representation of the dataset with Deck objects.
    """

    list_of_decks: list[Deck] = []
    for inner_list in data_vector:
        list_of_decks.append(
            construct_deck_object_from_dict(parse_datalist_to_dict(inner_list))
        )

    return list_of_decks


def parse_datalist_to_dict(data_vector: list[Any]) -> dict[str, Any]:
    """Get a list of data and parse it to a dictionary.

    Args:
        data_vector: The list containing data representing a deck.

    Returns:
        data_dict: Representation of a deck as a dictionary.
    """
    deck_num: str = str(data_vector[0])
    name: str = str(data_vector[2])
    format: str = str(data_vector[5])
    main_deck_string = data_vector[6]
    deck_format = data_vector[5]

    deck_str = main_deck_string.strip("[]")
    deck_liste = deck_str.split(",")
    deck_liste = [id.strip('"') for id in deck_liste]

    data_dict: dict[str, Any] = {
        "deck_num": deck_num,
        "name": name,
        "format": format,
        "main_deck": deck_liste,
        "format": deck_format,
    }

    return data_dict


def construct_deck_object_from_dict(deck_as_dict: dict[str, Any]) -> Deck:
    """Create a deckobject from a given dictionary of data.

    Args:
        deck_as_dict: The deck represented as a dictionary.

    Returns:
        deck: The Deckobject.
    """
    return Deck(
        deck_num=deck_as_dict["deck_num"],
        name=deck_as_dict["name"],
        format=deck_as_dict["format"],
        main_deck=deck_as_dict["main_deck"],
    )


def filterDecks(deck: list[Any], format: list[str]) -> int:

    return deck[5] in format
