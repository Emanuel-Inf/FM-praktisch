from classes.deck import Deck
import API.cardAPIDatensatz2 as cardAPIDatensatz2

import numpy as np
import random
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import shapiro, levene, mannwhitneyu, probplot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ENUMS.formatTypes import FormatType
import data_reader

def startLogRegressionForViews():
    sample_size = 250
    first_format = FormatType.ANIME
    second_format = FormatType.CASUAL

    prepaired_decks = (data_reader.get_All_Decks_Prepaired())

    random_competitive_decks = getRandomDecks(prepaired_decks, first_format, False, sample_size)
    random_casual_decks = getRandomDecks(prepaired_decks, second_format, False, sample_size)

    views_competitive = getViews(random_competitive_decks)
    views_casual = getViews(random_casual_decks)
    
    showBoxPlott(views_competitive, views_casual, first_format, second_format)

    show_QQ_plot(views_competitive, f"QQ-Plot für {first_format.value}-Deck deck views")
    show_QQ_plot(views_casual, f"QQ-Plot für {second_format.value}-Deck deck views")

    testsHypothesis(views_competitive, views_casual)

    total_prices, y, predicted_x = regression(views_competitive, views_casual)
    showRegressionPlot(total_prices,y,predicted_x, first_format)

    cm_tournaments = confusionMatrix(predicted_x, y)
    showConfusionMatrix(cm_tournaments, first_format, second_format)

def getAllDecksBasedOnFormat(list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool) -> list[Deck]:

    decks_based_on_Format = []
    if(isNotFormat):
        decks_based_on_Format = [deck for deck in list_of_decks
                            if deck.format not in formatType.value]
    else:
        decks_based_on_Format = [deck for deck in list_of_decks
                            if deck.format in formatType.value]
    return decks_based_on_Format

def getRandomDecks(list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool, sample_size) -> list[Deck]:

    all_tournament_decks = getAllDecksBasedOnFormat(list_of_decks, formatType, isNotFormat)
    randomDecks = random.sample(all_tournament_decks, sample_size if sample_size > 0 else len(list_of_decks))

    return randomDecks

def getCardViewsSum(cardIDs: list[str]) -> float:

    return sum(int(cardAPIDatensatz2.getCardFromCache(card_id).views) for card_id in cardIDs)

def getViews(sample_decks: list[Deck]) -> list[int]:
    prices_tournament = [getCardViewsSum(deck.main_deck) for deck in sample_decks]
    return prices_tournament

def regression(views_competitive , views_casual):
 
    prices = np.concatenate([views_competitive, views_casual])
    deck_type = np.array([1]*len(views_competitive) + [0]*len(views_casual))
    
    X = sm.add_constant(prices)
    Y = deck_type

    model = sm.Logit(Y, X)
    results = model.fit()

    predicted_probs = results.predict(X)
    results.summary()

    return prices, Y, predicted_probs

def showBoxPlott(group_a, group_b, format_one: FormatType, format_two: FormatType):
    data = pd.DataFrame({
    'Preis': group_a + group_b,
    'Kategorie': [f'{format_one}'] * len(group_a) + [f"{format_two}"] * len(group_b)
    })

    sns.boxplot(x='Kategorie', y='Preis', data=data)
    plt.title('Vergleich der Varianz zwischen Tournament und Nicht-Tournament Decks')
    plt.show()

def show_QQ_plot(group, title:str ):
    probplot(group, dist="norm", plot=plt)
    plt.title(title)
    plt.show() 

def showRegressionPlot(prices, Y, predicted_probs, format: FormatType):
    # Wahrscheinlichkeiten plotten
    plt.figure(figsize=(10, 6))
    plt.scatter(prices, Y, c=Y, cmap='coolwarm', label='Tatsächliche Werte', alpha=0.6)
    plt.scatter(prices, predicted_probs, color='black', label='Vorhergesagte Wahrscheinlichkeiten')

    plt.title('Logistische Regression: Vorhersage des Deck-Typs basierend auf Views')
    plt.xlabel('Views')
    plt.ylabel(f'Wahrscheinlichkeit für {format.value}')
    plt.legend()
    plt.grid(True)

    plt.show()

def confusionMatrix(predicted_probs, Y):
    threshold = 0.5
    y_pred = (predicted_probs >= threshold).astype(int)
    cm = confusion_matrix(Y, y_pred)

    return cm

def showConfusionMatrix(cm, format_one: FormatType, format_two: FormatType):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{format_one.value}", f"{format_two.value}"])
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