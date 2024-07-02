import functools
import os
import random
import ygoAPI
import data_reader
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, levene, probplot, mannwhitneyu
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns


CSV_DIRECTORY: str = "./daten/csv/"

def startRegression(flattened_decks, SAMPLE_SIZE): 


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