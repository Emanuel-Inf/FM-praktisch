"""Define the views for the logistic regressions.

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

import API.cardAPIDatensatz2 as cardAPIDatensatz2
import data_reader
from classes.deck import Deck
from ENUMS.formatTypes import FormatType


def startLogRegressionForViews():
    sample_size = 1000
    first_format = FormatType.ANIME
    second_format = FormatType.ANIME
    isKomplämentär = True

    prepaired_decks = data_reader.get_All_Decks_Prepaired()

    random_competitive_decks = getRandomDecks(
        prepaired_decks, first_format, False, sample_size
    )
    random_casual_decks = getRandomDecks(
        prepaired_decks, second_format, isKomplämentär, sample_size
    )

    views_competitive = getViews(random_competitive_decks)
    views_casual = getViews(random_casual_decks)

    showBoxPlott(
        views_competitive, views_casual, first_format, second_format, isKomplämentär
    )

    show_QQ_plot(views_competitive, f"QQ-Plot für {first_format}-Deck deck views")

    if isKomplämentär:
        show_QQ_plot(views_casual, f"QQ-Plot für nicht {second_format}-Deck deck views")
    else:
        show_QQ_plot(views_casual, f"QQ-Plot für {second_format}-Deck deck views")

    testsHypothesis(views_competitive, views_casual)

    total_prices, y, predicted_x = regression(views_competitive, views_casual)
    showRegressionPlot(total_prices, y, predicted_x, first_format)

    cm_tournaments = confusionMatrix(predicted_x, y)
    showConfusionMatrix(cm_tournaments, first_format, second_format, isKomplämentär)


def getAllDecksBasedOnFormat(
    list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool
) -> list[Deck]:

    decks_based_on_Format = []
    if isNotFormat:
        decks_based_on_Format = [
            deck for deck in list_of_decks if deck.format not in formatType.value
        ]
    else:
        decks_based_on_Format = [
            deck for deck in list_of_decks if deck.format in formatType.value
        ]
    return decks_based_on_Format


def getRandomDecks(
    list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool, sample_size
) -> list[Deck]:

    all_tournament_decks = getAllDecksBasedOnFormat(
        list_of_decks, formatType, isNotFormat
    )
    randomDecks = random.sample(
        all_tournament_decks, sample_size if sample_size > 0 else len(list_of_decks)
    )

    return randomDecks


def getCardViewsSum(cardIDs: list[str]) -> float:

    return sum(
        int(cardAPIDatensatz2.getCardFromCache(card_id).views) for card_id in cardIDs
    )


def getViews(sample_decks: list[Deck]) -> list[int]:
    prices_tournament = [getCardViewsSum(deck.main_deck) for deck in sample_decks]
    return prices_tournament


def regression(views_competitive, views_casual):

    prices = np.concatenate([views_competitive, views_casual])
    deck_type = np.array([1] * len(views_competitive) + [0] * len(views_casual))

    X = sm.add_constant(prices)
    Y = deck_type

    model = sm.Logit(Y, X)
    results = model.fit()

    predicted_probs = results.predict(X)
    results.summary()

    return prices, Y, predicted_probs


def showBoxPlott(
    group_a, group_b, format_one: FormatType, format_two: FormatType, isKomplämentär
):

    kategorie_string = ""
    if isKomplämentär:
        kategorie_string = [f"{format_one}"] * len(group_a) + [
            f"nicht {format_two}"
        ] * len(group_b)
    else:
        kategorie_string = [f"{format_one}"] * len(group_a) + [f"{format_two}"] * len(
            group_b
        )
    data = pd.DataFrame({"Views": group_a + group_b, "Kategorie": kategorie_string})

    überschrift_string = ""
    if isKomplämentär:
        überschrift_string = f"Vergleich der Varianz zwischen '{format_one}' und nicht '{format_two}' Decks"
    else:
        überschrift_string = (
            f"Vergleich der Varianz zwischen '{format_one}' und '{format_two}' Decks"
        )

    sns.boxplot(x="Kategorie", y="Views", data=data)
    plt.title(überschrift_string)
    plt.show()


def show_QQ_plot(group, title: str):
    probplot(group, dist="norm", plot=plt)
    plt.title(title)
    plt.show()


def showRegressionPlot(prices, Y, predicted_probs, format: FormatType):
    # Wahrscheinlichkeiten plotten
    plt.figure(figsize=(10, 6))
    plt.scatter(prices, Y, c=Y, cmap="coolwarm", label="Tatsächliche Werte", alpha=0.6)
    plt.scatter(
        prices,
        predicted_probs,
        color="black",
        label="Vorhergesagte Wahrscheinlichkeiten",
    )

    plt.title("Logistische Regression: Vorhersage des Deck-Typs basierend auf Views")
    plt.xlabel("Views")
    plt.ylabel(f"Wahrscheinlichkeit für {format}")
    plt.legend()
    plt.grid(True)

    plt.show()


def confusionMatrix(predicted_probs, Y):
    threshold = 0.5
    y_pred = (predicted_probs >= threshold).astype(int)
    cm = confusion_matrix(Y, y_pred)

    return cm


def showConfusionMatrix(
    cm, format_one: FormatType, format_two: FormatType, isKomplämentär
):
    disp = None
    if isKomplämentär:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[f"{format_one}", f"nicht {format_two}"]
        )
    else:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[f"{format_one}", f"{format_two}"]
        )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Konfusionsmatrix für das logistische Regressionsmodell")
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


def check_if_t_test_vaiable(group_a, group_b, alpha=0.05) -> bool:
    # group a and group b would be in this case the prices of tournament cards and respectively the prices of non tournament cards

    # the shapiro test shows if the data is normally distributed
    shapiro_test_group_a = shapiro(group_a).pvalue >= alpha
    shapiro_test_group_b = shapiro(group_b).pvalue >= alpha

    # the levene test is used to check if the two groups(anime decks, non anime decks)
    # have a similar variance (required to perform a t-test)
    levene_test = levene(group_a, group_b).pvalue >= alpha

    return shapiro_test_group_a and shapiro_test_group_b and levene_test

