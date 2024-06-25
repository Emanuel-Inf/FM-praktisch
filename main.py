"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import functools
import os
import random
import ygoAPI
import data_reader
from classes.cardtype import CardType
import logisticRegression

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, levene, probplot, mannwhitneyu
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

CSV_DIRECTORY: str = "./daten/csv/"
SEARCH_CARDS = ["83764719", "47826112", "3643300", "53804307", "81843628", "36553319"]
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
   # print(len(flattened_decks))
    flattened_decks = flattened_decks[1:] #Deck(deck_num='deck_num', name='deck_name', main_deck=['main_deck'])


    #only get deck where the contained cards have a price
    filtered_decks = [deck for deck in flattened_decks if ygoAPI.deckHasCompletePriceInfo(deck.main_deck)] 


    #world_championship_decks = [deck for deck in flattened_decks
    #                            if deck.format == "World Championship Decks"] '!! maybe also meta decks or tournament meta decks
    #print(len(world_championship_decks))

    #random_sample = random.sample(
    #    flattened_decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(flattened_decks)
    #)

    #complete_decks= []
    #for deck in world_championship_decks:
    #    if ygoprodeckAPI.deckHasCompletePriceInfo(deck.main_deck):
    #        print(f"Der Deck '{deck.name}' ist {ygoprodeckAPI.getCardPriceSum(deck.main_deck)} Wert.")
    #        complete_decks.append(deck)

   #print(len(complete_decks))

    anime_decks = [deck for deck in flattened_decks
                   if deck.format == "Anime Decks"]

    random_sample_anime = random.sample(
        anime_decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(flattened_decks)
    )

    prices_anime = [ygoAPI.getCardPriceSum(deck.main_deck) for deck in random_sample_anime]

    non_anime_decks = [deck for deck in flattened_decks
                   if deck.format != "Anime Decks"]

    random_sample_non_anime = random.sample(
        non_anime_decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(flattened_decks)
    )   

    prices_non_anime = [ygoAPI.getCardPriceSum(deck.main_deck) for deck in random_sample_non_anime
                        ]

    
    shapiro(prices_anime) # the shapiro test shows if the data is normally distributed
    shapiro(prices_non_anime)
        

    levene(prices_anime, prices_non_anime) # the levene test is used to check if the two groups(anime decks, non anime decks) have a similar variance (required to perform a t-test)

    data = pd.DataFrame({
    'Preis': prices_anime + prices_non_anime,
    'Kategorie': ['Anime'] * len(prices_anime) + ['Nicht-Anime'] * len(prices_non_anime)
    })

    sns.boxplot(x='Kategorie', y='Preis', data=data)
    plt.title('Vergleich der Varianz zwischen Anime und Nicht-Anime Decks')
    plt.show()

    probplot(prices_anime, dist="norm", plot=plt)
    plt.title('QQ-Plot für Anime-Deck Preise')
    plt.show()

    probplot(prices_non_anime, dist="norm", plot=plt)
    plt.title('QQ-Plot für Nicht-Anime-Deck Preise')
    plt.show()

#Whitney-U Test
    statistics, p_value = mannwhitneyu(prices_anime, prices_non_anime, alternative='greater')

    if p_value < 0.05:
        print("Es gibt signifikante Hinweise darauf, dass Anime-Decks im Durchschnitt teurer sind als Nicht-Anime-Decks.")
    else:
        print("Es gibt keine signifikanten Hinweise darauf, dass Anime-Decks im Durchschnitt teurer sind als Nicht-Anime-Decks.")
    


#LOGISTISCHE REGRESSION
    prices = np.concatenate([prices_anime, prices_non_anime])
    deck_type = np.array([1]*len(prices_anime) + [0]*len(prices_non_anime))

    X = sm.add_constant(prices)
    Y = deck_type

    model = sm.Logit(Y, X)
    results = model.fit()

    predicted_probs = results.predict(X)
    results.summary()

# Wahrscheinlichkeiten plotten
    plt.figure(figsize=(10, 6))
    plt.scatter(prices, Y, c=Y, cmap='coolwarm', label='Tatsächliche Werte', alpha=0.6)
    plt.scatter(prices, predicted_probs, color='black', label='Vorhergesagte Wahrscheinlichkeiten')

# Plot anpassen
    plt.title('Logistische Regression: Vorhersage des Deck-Typs basierend auf Preisen')
    plt.xlabel('Preis')
    plt.ylabel('Wahrscheinlichkeit für Anime-Deck')
    plt.legend()
    plt.grid(True)

    plt.show()

#Accuracy of model
    threshold = 0.5
    y_pred = (predicted_probs >= threshold).astype(int)
    cm = confusion_matrix(Y, y_pred)

    
# Konfusionsmatrix visualisieren
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nicht-Anime-Deck", "Anime-Deck"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Konfusionsmatrix für das logistische Regressionsmodell")
    plt.show()


#Tournament Decks correlate to a higer price
    prices_tournament = logisticRegression.getPrices(logisticRegression.getRandomTournamentDecks(filtered_decks))
    prices_non_tournament = logisticRegression.getPrices(logisticRegression.getRandomNonTournamentDecks(filtered_decks))
    
    logisticRegression.showBoxPlott(prices_tournament, prices_non_tournament)

    logisticRegression.show_QQ_plot(prices_tournament, "QQ-Plot für Tournament-Deck Preise")
    logisticRegression.show_QQ_plot(prices_non_tournament, "QQ-Plot für Nicht-Tournament-Deck Preise")

    logisticRegression.testsHypothesis(prices_tournament, prices_non_tournament)

    total_prices, y, predicted_x = logisticRegression.regression(filtered_decks)
    logisticRegression.showRegressionPlot(total_prices,y,predicted_x)

    cm_tournaments = logisticRegression.confusionMatrix(predicted_x, y)
    logisticRegression.showConfusionMatrix(cm_tournaments)

    

if __name__ == "__main__":
    main()


