"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""


import random
import ygoAPI
import data_reader
from ENUMS.cardtype import CardType
import logisticRegression

import deckfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, levene, probplot, mannwhitneyu
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
from ENUMS.formatTypes import FormatType
import ygoprodeckAPI

SEARCH_CARDS = ["83764719", "47826112", "3643300", "53804307", "81843628", "36553319"]
SAMPLE_SIZE: int = 250


def main():
    """Executes the main function."""
    
    #Decks aus dem Datensatz lesen
    prepaired_decks = (data_reader.get_All_Decks_Prepaired())

    #Filterung von Karten

    #WCC
    wcc_filtered_decks = deckfilter.deck_Format_Filter(prepaired_decks, [FormatType.WORLDCC], False)
    
    wcc_random_sample = deckfilter.deck_Random_Filter(wcc_filtered_decks, SAMPLE_SIZE)
    
    wcc_valid_price_decks = deckfilter.deck_HasComplete_PriceInfo_Filter(wcc_random_sample)
  
    #for deck in wcc_valid_price_decks:
    #    print(f"Der Deck '{deck.name}' ist {ygoAPI.getCardPriceSum(deck.main_deck)} Wert.")

    #--------------------------------------------------------------#

    #Anime
    anime_decks = deckfilter.deck_Format_Filter(prepaired_decks, [FormatType.ANIME], False)
    
    anime_random_sample_anime = deckfilter.deck_Random_Filter(anime_decks, SAMPLE_SIZE)

    anime_valid_price_decks = deckfilter.deck_HasComplete_PriceInfo_Filter(anime_random_sample_anime)

    anime_deck_prices = [ygoAPI.getCardPriceSum(deck.main_deck) for deck in anime_valid_price_decks]

    #for deck in anime_valid_price_decks:
        #print(f"Der Deck '{deck.name}' ist {ygoAPI.getCardPriceSum(deck.main_deck)} Wert.")

    non_animee_decks = deckfilter.deck_Format_Filter(prepaired_decks, [FormatType.ANIME], True)
    
    non_animee_random_sample_decks = deckfilter.deck_Random_Filter(non_animee_decks, SAMPLE_SIZE)

    non_animee_valid_prices_decks = deckfilter.deck_HasComplete_PriceInfo_Filter(non_animee_random_sample_decks)

    non_animee_deck_prices = [ygoAPI.getCardPriceSum(deck.main_deck) for deck in non_animee_valid_prices_decks]

    #for deck in non_animee_valid_prices_decks:
    #    print(f"Der Deck '{deck.name}' ist {ygoAPI.getCardPriceSum(deck.main_deck)} Wert.")

    #shapiro(anime_deck_prices) # the shapiro test shows if the data is normally distributed
    #shapiro(non_animee_deck_prices)
        
    #levene(anime_deck_prices, non_animee_deck_prices) # the levene test is used to check if the two groups(anime decks, non anime decks) have a similar variance (required to perform a t-test)


    data = pd.DataFrame({
    'Preis': anime_deck_prices + non_animee_deck_prices,
    'Kategorie': ['Anime'] * len(anime_deck_prices) + ['Nicht-Anime'] * len(non_animee_deck_prices)
    })

    sns.boxplot(x='Kategorie', y='Preis', data=data)
    plt.title('Vergleich der Varianz zwischen Anime und Nicht-Anime Decks')
    plt.show()

    probplot(anime_deck_prices, dist="norm", plot=plt)
    plt.title('QQ-Plot für Anime-Deck Preise')
    plt.show()

    probplot(non_animee_deck_prices, dist="norm", plot=plt)
    plt.title('QQ-Plot für Nicht-Anime-Deck Preise')
    plt.show()

"""""
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
"""""
    

if __name__ == "__main__":
    main()


