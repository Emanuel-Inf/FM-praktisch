"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import data_reader
import Regressionen.logisticRegressionTournamentDeckPreise as logisticRegressionTournamentDeckPreise
import Regressionen.logisticRegressionKarten as logisticRegressionKarten
from ENUMS.formatTypes import FormatType
import numpy as np
import statsmodels.api as sm


from classes.deck import Deck
import ygoAPI

import numpy as np
import random
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import shapiro, levene, mannwhitneyu, probplot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ENUMS.formatTypes import FormatType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
SEARCH_CARDS = [14558127]
SEARCH_THIS_CARD = "44968687"
SAMPLE_SIZE: int = 500

def predict_probability(X, results):
    return results.predict(X)

def main():
    """Executes the main function."""

    #Decks aus dem Datensatz lesen
    prepaired_decks = (data_reader.get_All_Decks_Prepaired())

    
    
    random_competitive_decks = logisticRegressionKarten.getRandomDecks(prepaired_decks, FormatType.COMPETITIVE, False, SAMPLE_SIZE)
    
    random_casual_decks = logisticRegressionKarten.getRandomDecks(prepaired_decks, FormatType.CASUAL, False, SAMPLE_SIZE)
    
    random_competitive_decks_card_hits = logisticRegressionKarten.getListOfHitsOnSpecificCard(random_competitive_decks, SEARCH_THIS_CARD)
    
    random_casual_decks_card_hits = logisticRegressionKarten.getListOfHitsOnSpecificCard(random_casual_decks, SEARCH_THIS_CARD)

    hits = np.concatenate([random_competitive_decks_card_hits, random_casual_decks_card_hits])
    deck_type = np.array([1]*len(random_competitive_decks_card_hits) + [0]*len(random_casual_decks_card_hits))

    X = sm.add_constant(hits)
    Y = deck_type

    model = sm.Logit(Y, X)
    results = model.fit()

    predicted_probs = results.predict(X)
    results.summary()

    # Wahrscheinlichkeiten plotten
    plt.figure(figsize=(10, 6))
    plt.scatter(hits, Y, c=Y, cmap='coolwarm', label='Tatsächliche Werte', alpha=0.6)
    plt.scatter(hits, predicted_probs, color='black', label='Vorhergesagte Wahrscheinlichkeiten')

    plt.title('Logistische Regression: Vorhersage des Deck-Typs basierend auf Preisen')
    plt.xlabel('Preis')
    plt.ylabel('Wahrscheinlichkeit für Anime-Deck')
    plt.legend()
    plt.grid(True)

    plt.show()

    threshold = 0.5
    y_pred = (predicted_probs >= threshold).astype(int)
    cm = confusion_matrix(Y, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nicht-Tournament-Deck", "Tournament-Deck"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Konfusionsmatrix für das logistische Regressionsmodell")
    plt.show()
    
    """""
#Tournament Decks correlate to a higer price
    prices_tournament = logisticRegressionTournamentDeckPreise.getPrices(logisticRegressionTournamentDeckPreise.getRandomDecks(prepaired_decks, FormatType.COMPETITIVE, False))
    prices_non_tournament = logisticRegressionTournamentDeckPreise.getPrices(logisticRegressionTournamentDeckPreise.getRandomDecks(prepaired_decks, FormatType.COMPETITIVE, True))
    
    logisticRegressionTournamentDeckPreise.showBoxPlott(prices_tournament, prices_non_tournament)

    logisticRegressionTournamentDeckPreise.show_QQ_plot(prices_tournament, "QQ-Plot für Tournament-Deck Preise")
    logisticRegressionTournamentDeckPreise.show_QQ_plot(prices_non_tournament, "QQ-Plot für Nicht-Tournament-Deck Preise")

    logisticRegressionTournamentDeckPreise.testsHypothesis(prices_tournament, prices_non_tournament)

    total_prices, y, predicted_x = logisticRegressionTournamentDeckPreise.regression(prepaired_decks, FormatType.COMPETITIVE)
    logisticRegressionTournamentDeckPreise.showRegressionPlot(total_prices,y,predicted_x)

    cm_tournaments = logisticRegressionTournamentDeckPreise.confusionMatrix(predicted_x, y)
    logisticRegressionTournamentDeckPreise.showConfusionMatrix(cm_tournaments)
"""""


if __name__ == "__main__":
    main()


