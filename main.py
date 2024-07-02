"""Main file for the code project.

@author: Katrin Kober, Emanuel Petrinovic, Max Weise
"""

import Clusteranalyse.clusteranalyse as clusteranalyse
import Regressionen.logisticRegressionViews as logisticRegressionViews
import Regressionen.logisticRegressionAnimeNonAnimePrices as logisticRegressionAnimeNonAnimePrices
import Regressionen.logisticRegressionPreise as logisticRegressionPreise

import functools
import os
import ygoAPI
import data_reader

CSV_DIRECTORY: str = "./daten/csv/"
SAMPLE_SIZE: int = 250


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
    flattened_decks = flattened_decks[1:] #Deck(deck_num='deck_num', name='deck_name', main_deck=['main_deck'])
    filtered_decks = [deck for deck in flattened_decks if ygoAPI.deckHasCompletePriceInfo(deck.main_deck)] 

    #Regressionen
    logisticRegressionAnimeNonAnimePrices.startRegression(flattened_decks, SAMPLE_SIZE)
    logisticRegressionPreise.startRegression(filtered_decks,SAMPLE_SIZE)

    logisticRegressionViews.startLogRegressionForViews()

    #Clusteranalyse
    clusteranalyse.startCluserAnalyse()


if __name__ == "__main__":
    main()

