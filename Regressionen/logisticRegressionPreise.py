"""Run logistic regressions on the available decks.

@authors: Katrin Kober, Emanuel Petrinovic, Max Weise
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import levene, mannwhitneyu, probplot, shapiro
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import ygoAPI
from classes.deck import Deck

import random
import ygoAPI
import logisticRegression

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, levene, probplot, mannwhitneyu
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns


tournament_type = [
    "World Championship Decks",
    "Tournament Meta Decks",
    "Tournament Meta Decks OCG",
    "Meta Decks",
]


def startRegression(filtered_decks, SAMPLE_SIZE):
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

def getAllTournamentDecks(list_of_decks: list[Deck]) -> list[Deck]:
    tournament_decks = [
        deck for deck in list_of_decks if deck.format in tournament_type
    ]
    return tournament_decks


def getAllNonTournamentDecks(list_of_decks: list[Deck]) -> list[Deck]:
    non_tournament_decks = [
        deck for deck in list_of_decks if deck.format not in tournament_type
    ]
    return non_tournament_decks


def getRandomTournamentDecks(list_of_decks: list[Deck]) -> list[Deck]:
    all_tournament_decks = getAllTournamentDecks(list_of_decks)
    random_sample_tournament = random.sample(
        all_tournament_decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(list_of_decks)
    )
    return random_sample_tournament


def getRandomNonTournamentDecks(list_of_decks: list[Deck]) -> list[Deck]:
    all_non_tournament_decks = getAllNonTournamentDecks(list_of_decks)
    random_sample_non_tournament = random.sample(
        all_non_tournament_decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(list_of_decks)
    )
    return random_sample_non_tournament


def getPrices(sample_decks: list[Deck]) -> list[float]:
    prices_tournament = [
        ygoAPI.getCardPriceSum(deck.main_deck) for deck in sample_decks
    ]
    return prices_tournament


def regression(list_of_decks: list[Deck]):
    tournament_decks = getRandomTournamentDecks(list_of_decks)
    prices_tournament = getPrices(tournament_decks)

    non_tournament_decks = getRandomNonTournamentDecks(list_of_decks)
    prices_non_tournament = getPrices(non_tournament_decks)

    prices = np.concatenate([prices_tournament, prices_non_tournament])
    deck_type = np.array(
        [1] * len(prices_tournament) + [0] * len(prices_non_tournament)
    )

    X = sm.add_constant(prices)
    Y = deck_type

    model = sm.Logit(Y, X)
    results = model.fit()

    predicted_probs = results.predict(X)
    results.summary()

    return prices, Y, predicted_probs


def showRegressionPlot(prices, Y, predicted_probs):
    # Wahrscheinlichkeiten plotten
    plt.figure(figsize=(10, 6))
    plt.scatter(prices, Y, c=Y, cmap="coolwarm", label="Tatsächliche Werte", alpha=0.6)
    plt.scatter(
        prices,
        predicted_probs,
        color="black",
        label="Vorhergesagte Wahrscheinlichkeiten",
    )

    plt.title("Logistische Regression: Vorhersage des Deck-Typs basierend auf Preisen")
    plt.xlabel("Preis")
    plt.ylabel("Wahrscheinlichkeit für Anime-Deck")
    plt.legend()
    plt.grid(True)

    plt.show()


def confusionMatrix(predicted_probs, Y):
    threshold = 0.5
    y_pred = (predicted_probs >= threshold).astype(int)
    cm = confusion_matrix(Y, y_pred)

    return cm


def showConfusionMatrix(cm):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Nicht-Tournament-Deck", "Tournament-Deck"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Konfusionsmatrix für das logistische Regressionsmodell")
    plt.show()


def check_if_t_test_vaiable(group_a, group_b, alpha=0.05) -> bool:
    # group a and group b would be in this case the prices of tournament cards and respectively the prices of non tournament cards

    # the shapiro test shows if the data is normally distributed
    shapiro_test_group_a = shapiro(group_a).pvalue >= alpha
    shapiro_test_group_b = shapiro(group_b).pvalue >= alpha

    # the levene test is used to check if the two groups(anime decks, non anime decks)
    # have a similar variance (required to perform a t-test)
    levene_test = levene(group_a, group_b).pvalue >= alpha

    return shapiro_test_group_a and shapiro_test_group_b and levene_test


def showBoxPlott(group_a, group_b):
    data = pd.DataFrame(
        {
            "Preis": group_a + group_b,
            "Kategorie": ["Tournament"] * len(group_a)
            + ["Nicht-Tournament"] * len(group_b),
        }
    )

    sns.boxplot(x="Kategorie", y="Preis", data=data)
    plt.title("Vergleich der Varianz zwischen Tournament und Nicht-Tournament Decks")
    plt.show()


def show_QQ_plot(group, title: str):
    probplot(group, dist="norm", plot=plt)
    plt.title(title)
    plt.show()


def testsHypothesis(group_a, group_b):
    if check_if_t_test_vaiable(group_a, group_b):
        print("Ein t-Test muss ausgeführt werden.")

    else:
        # Whitney-U Test
        p_value = mannwhitneyu(group_a, group_b, alternative="greater").pvalue

    if p_value < 0.05:
        print(
            "Es gibt signifikante Hinweise darauf, dass Tournament-Decks im Durchschnitt teurer sind als Nicht-Tournament-Decks."
        )
    else:
        print(
            "Es gibt keine signifikanten Hinweise darauf, dass Tournament-Decks im Durchschnitt teurer sind als Nicht-Tournament-Decks."
        )

