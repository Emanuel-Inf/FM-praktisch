"""This file contains functions to read and compile data.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""


import numpy as np
import pandas as pd
from deck import Deck

from typing import Any

def read_xlsx(file_name: str) -> np.ndarray:
    """Read the values from [file_name] and convert them into an array of lists.

    Args:
        file_name: The file to be read.

    Returns:
        contents: The contents of the file as a two dimensional nparray.
    """
    excel_data = pd.read_excel(file_name)
    data = pd.DataFrame(excel_data)
    
    return data.to_numpy()

def _parse_data_to_list_with_deck_objects(data_vector: list[Any]) -> list[Deck]:
    """Get a list of data and parse it to a list with Decks.

    Args:
        data_vector: The list containing data representing a deck.

    Returns:
        list_of_decks: Representation of the dataset with Deck objects.
    """
     
    list_of_decks: list[Deck] = []

    for inner_list in data_vector:
        list_of_decks.append(_construct_deck_Object_from_dict(_parse_datalist_to_dict(inner_list)))
    
    return list_of_decks



def _parse_datalist_to_dict(data_vector: list[Any]) -> dict[str, Any]:
    """Get a list of data and parse it to a dictionary.

    Args:
        data_vector: The list containing data representing a deck.

    Returns:
        data_dict: Representation of a deck as a dictionary.
    """

    deck_num: str = str(data_vector[0])
    name: str = str(data_vector[2])
    main_deck: list[Any] = data_vector[6]

    data_dict: dict[str, Any] = {
            "deck_num": deck_num,
            "name": name,
            "main_deck": main_deck,
        }
    
    return data_dict

def _construct_deck_Object_from_dict(deck_as_dict: dict[str, Any]) -> Deck:
    """Create a deckobject from a given dictionary of data.

    Args:
        deck_as_dict: The deck represented as a dictionary.

    Returns:
        deck: The Deckobject.
    """
    return Deck(
        deck_as_dict["deck_num"],
        deck_as_dict["name"],
        deck_as_dict["main_deck"],
    )