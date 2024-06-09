"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import data_reader


def main():
    """Executes the main function."""

    excel_data = data_reader.read_xlsx("daten/xlsx/00000001.xlsx")

    list_of_decks = data_reader._parse_data_to_list_with_deck_objects(excel_data)
    
    #Beispiel um das letzte Deck innerhalb des Datensatzes zu bekommen
    print(list_of_decks[323])


main()