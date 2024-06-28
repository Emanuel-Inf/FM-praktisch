from classes.deck import Deck
import API2.cardAPIDatensatz2 as cardAPIDatensatz2

import numpy as np
import random
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import shapiro, levene, mannwhitneyu, probplot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ENUMS.formatTypes import FormatType
SAMPLE_SIZE: int = 250

def getAllDecksBasedOnFormat(list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool) -> list[Deck]:

    decks_based_on_Format = []
    if(isNotFormat):
        decks_based_on_Format = [deck for deck in list_of_decks
                            if deck.format not in formatType.value]
    else:
        decks_based_on_Format = [deck for deck in list_of_decks
                            if deck.format in formatType.value]
    return decks_based_on_Format

def getRandomDecks(list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool) -> list[Deck]:

    all_tournament_decks = getAllDecksBasedOnFormat(list_of_decks, formatType, isNotFormat)
    randomDecks = random.sample(all_tournament_decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(list_of_decks))

    return randomDecks

def getCardViewsSum(cardIDs: list[str]) -> float:

    return sum(int(cardAPIDatensatz2.getCardFromCache(card_id).views) for card_id in cardIDs)

def getViews(sample_decks: list[Deck]) -> list[int]:
    prices_tournament = [getCardViewsSum(deck.main_deck) for deck in sample_decks]
    return prices_tournament

def regression(prepaired_decks: list[Deck], formatType: FormatType):
    tournament_decks = getRandomDecks(prepaired_decks, formatType , False)
    prices_tournament = getViews(tournament_decks)

    non_tournament_decks = getRandomDecks(prepaired_decks, formatType, True)
    prices_non_tournament = getViews(non_tournament_decks)
    
    prices = np.concatenate([prices_tournament, prices_non_tournament])
    deck_type = np.array([1]*len(prices_tournament) + [0]*len(prices_non_tournament))
    
    X = sm.add_constant(prices)
    Y = deck_type

    model = sm.Logit(Y, X)
    results = model.fit()

    predicted_probs = results.predict(X)
    results.summary()

    return prices, Y, predicted_probs

def showBoxPlott(group_a, group_b):
    data = pd.DataFrame({
    'Preis': group_a + group_b,
    'Kategorie': ['Tournament'] * len(group_a) + ['Nicht-Tournament'] * len(group_b)
    })

    sns.boxplot(x='Kategorie', y='Preis', data=data)
    plt.title('Vergleich der Varianz zwischen Tournament und Nicht-Tournament Decks')
    plt.show()

def show_QQ_plot(group, title:str ):
    probplot(group, dist="norm", plot=plt)
    plt.title(title)
    plt.show() 

def showRegressionPlot(prices, Y, predicted_probs):
    # Wahrscheinlichkeiten plotten
    plt.figure(figsize=(10, 6))
    plt.scatter(prices, Y, c=Y, cmap='coolwarm', label='Tatsächliche Werte', alpha=0.6)
    plt.scatter(prices, predicted_probs, color='black', label='Vorhergesagte Wahrscheinlichkeiten')

    plt.title('Logistische Regression: Vorhersage des Deck-Typs basierend auf Preisen')
    plt.xlabel('Preis')
    plt.ylabel('Wahrscheinlichkeit für Anime-Deck')
    plt.legend()
    plt.grid(True)

    plt.show()

def confusionMatrix(predicted_probs, Y):
    threshold = 0.5
    y_pred = (predicted_probs >= threshold).astype(int)
    cm = confusion_matrix(Y, y_pred)

    return cm

def showConfusionMatrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nicht-Tournament-Deck", "Tournament-Deck"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Konfusionsmatrix für das logistische Regressionsmodell")
    plt.show()

def testsHypothesis(group_a, group_b):
    if check_if_t_test_vaiable(group_a, group_b):
        print("Ein t-Test muss ausgeführt werden.")

    else:
    #Whitney-U Test
        p_value = mannwhitneyu(group_a, group_b, alternative='greater').pvalue

    if p_value < 0.05:
        print("Es gibt signifikante Hinweise darauf, dass Tournament-Decks im Durchschnitt teurer sind als Nicht-Tournament-Decks.")
    else:
        print("Es gibt keine signifikanten Hinweise darauf, dass Tournament-Decks im Durchschnitt teurer sind als Nicht-Tournament-Decks.")

def check_if_t_test_vaiable(group_a, group_b, alpha=0.05) -> bool:
    # group a and group b would be in this case the prices of tournament cards and respectively the prices of non tournament cards
    
    # the shapiro test shows if the data is normally distributed
    shapiro_test_group_a = shapiro(group_a).pvalue >= alpha 
    shapiro_test_group_b = shapiro(group_b).pvalue >= alpha
        
    # the levene test is used to check if the two groups(anime decks, non anime decks) 
    # have a similar variance (required to perform a t-test)
    levene_test = levene(group_a, group_b).pvalue >= alpha

    return shapiro_test_group_a and shapiro_test_group_b and levene_test