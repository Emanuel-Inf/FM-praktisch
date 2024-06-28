"""Main file for the Data Mining code project.

@author: Lisa Tochtermann, Emanuel Petrinovic, Max Weise
Date: 30.11.2022
"""

import data_reader
import Regressionen.logisticRegressionPreise as logisticRegressionPreise
#import Regressionen.logisticRegressionViews as logisticRegressionViews
from ENUMS.formatTypes import FormatType
import numpy as np
import statsmodels.api as sm

from sklearn.cluster import KMeans
from classes.deck import Deck
import API.cardAPIDatensatz1 as cardAPIDatensatz1

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
from sklearn.decomposition import PCA
import Clusteranalyse.clusteranalyse as  clusteranalyse

def predict_probability(X, results):
    return results.predict(X)

def main():
    """Executes the main function."""

    #Decks aus dem Datensatz lesen
    prepaired_decks = (data_reader.get_All_Decks_Prepaired())

    
    
    """""
    #Custeranalyse
    random_competitive_decks = logisticRegressionKarten.getRandomDecks(prepaired_decks, FormatType.META, False, SAMPLE_SIZE)
    ATTRIBUTE = "archetype"
    attribute_list, deck_dicts = clusteranalyse.getAttributeList(random_competitive_decks, ATTRIBUTE)

    matrix = clusteranalyse.prepairMatrix(attribute_list, deck_dicts, ATTRIBUTE)

    clusters, deck_matrix_pca, cluster_archetypes = clusteranalyse.calculateDominantArchetype(matrix,deck_dicts, ATTRIBUTE)

    clusteranalyse.plotCluster(clusters, deck_matrix_pca, cluster_archetypes)
    """""

    """""
    random_competitive_decks = logisticRegressionViews.getRandomDecks(prepaired_decks, FormatType.COMPETITIVE, False, )
    random_casual_decks = logisticRegressionViews.getRandomDecks(prepaired_decks, FormatType.CASUAL, False)

    views_competitive = logisticRegressionViews.getViews(random_competitive_decks)
    random_casual = logisticRegressionViews.getViews(random_casual_decks)

    logisticRegressionViews.showBoxPlott(views_competitive, random_casual)

    logisticRegressionViews.show_QQ_plot(views_competitive, "QQ-Plot für Tournament-Deck deck views")
    logisticRegressionViews.show_QQ_plot(random_casual, "QQ-Plot Casual Deck views")

    logisticRegressionViews.testsHypothesis(views_competitive, random_casual)

    total_prices, y, predicted_x = logisticRegressionViews.regression(prepaired_decks, FormatType.COMPETITIVE)
    logisticRegressionViews.showRegressionPlot(total_prices,y,predicted_x)

    cm_tournaments = logisticRegressionViews.confusionMatrix(predicted_x, y)
    logisticRegressionViews.showConfusionMatrix(cm_tournaments)
    
"""""
#Tournament Decks correlate to a higer price
    prices_tournament = logisticRegressionPreise.getPrices(logisticRegressionPreise.getRandomDecks(prepaired_decks, FormatType.COMPETITIVE, False))
    prices_non_tournament = logisticRegressionPreise.getPrices(logisticRegressionPreise.getRandomDecks(prepaired_decks, FormatType.COMPETITIVE, True))
    
    logisticRegressionPreise.showBoxPlott(prices_tournament, prices_non_tournament)

    logisticRegressionPreise.show_QQ_plot(prices_tournament, "QQ-Plot für Tournament-Deck Preise")
    logisticRegressionPreise.show_QQ_plot(prices_non_tournament, "QQ-Plot für Nicht-Tournament-Deck Preise")

    logisticRegressionPreise.testsHypothesis(prices_tournament, prices_non_tournament)

    total_prices, y, predicted_x = logisticRegressionPreise.regression(prepaired_decks, FormatType.COMPETITIVE)
    logisticRegressionPreise.showRegressionPlot(total_prices,y,predicted_x)

    cm_tournaments = logisticRegressionPreise.confusionMatrix(predicted_x, y)
    logisticRegressionPreise.showConfusionMatrix(cm_tournaments)



if __name__ == "__main__":
    main()