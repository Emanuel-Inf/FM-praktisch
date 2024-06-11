"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import data_reader


def main():
    """Executes the main function."""

    csv_data = data_reader.read_csv("daten/csv/00000001.csv")

    list_of_decks = data_reader.parse_data_to_list_with_deck_objects(csv_data)

    # # Beispiel um das letzte Deck innerhalb des Datensatzes zu bekommen
    print(list_of_decks)


if __name__ == "__main__":
    main()
