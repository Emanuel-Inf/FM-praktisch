from classes.deck import Deck
from classes.card import Card
from ENUMS.cardtype import CardType
import ygoprodeckAPI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def GetAllCardsInDeck(deck: Deck) -> list[Card]:
    
    list_of_cards_in_deck = []
    for card in deck.main_deck:
        list_of_cards_in_deck.append(ygoprodeckAPI.getCardFromLocal(card))
        
    return list_of_cards_in_deck
    
def getAnzahlKartenTypen(cards: list[Card]) -> dict[str, int]:

    counterMonster = 0
    counterZauber = 0
    counterFallen = 0
   
    for card in cards:
        if(card.card_type == CardType.MONSTER):
            counterMonster += 1
        elif(card.card_type == CardType.Zauber):
            counterZauber += 1
        else:
            counterFallen += 1
    dict_results = {
        "Monster": counterMonster,
        "Zauber": counterZauber,
        "Fallen": counterFallen
    }
    return dict_results

def getAnzahlKartenUndFormat(deck: Deck) -> dict[str, any]:

    list_of_cards = GetAllCardsInDeck(deck)

    dict_cards_zählung = getAnzahlKartenTypen(list_of_cards)

    dict_regression_daten = {
        "format": deck.format,
        "KartenZahlen": dict_cards_zählung
    }

    return dict_regression_daten

def logRegression(deck_data):
  
    data = []
    for deck in deck_data:
        format_binary = 1 if deck['format'] == 'Tournament Meta Decks' else 0
        data.append({
            'Monster': deck['KartenZahlen']['Monster'],
            'Zauber': deck['KartenZahlen']['Zauber'],
            'Fallen': deck['KartenZahlen']['Fallen'],
            'Format': format_binary
        })

    df = pd.DataFrame(data)

    # Features (unabhängige Variablen) und Zielvariable (abhängige Variable) definieren
    X = df[['Monster', 'Zauber', 'Fallen']]
    y = df['Format']

    # Daten in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistische Regression durchführen
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Vorhersagen auf Testdaten
    y_pred = model.predict(X_test)

    # Modellgenauigkeit berechnen
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Genauigkeit des Modells: {accuracy}")

    # Konfusionsmatrix anzeigen
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Konfusionsmatrix:\n{conf_matrix}")

    # Koeffizienten des Modells anzeigen
    coefficients = model.coef_[0]
    print(f"Koeffizienten: Monsters: {coefficients[0]}, Spells: {coefficients[1]}, Traps: {coefficients[2]}")

    # Visualisierung der Konfusionsmatrix
    plot_confusion_matrix(conf_matrix)

    # Visualisierung der Koeffizienten
    plot_coefficients(coefficients)

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Meta Decks", "Tournament Meta Decks"], yticklabels=["Meta Decks", "Tournament Meta Decks"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_coefficients(coefficients):
    plt.figure(figsize=(8, 6))
    feature_names = ['Monster', 'Zauber', 'Fallen']
    plt.bar(feature_names, coefficients)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Coefficients in Logistic Regression')
    plt.show()