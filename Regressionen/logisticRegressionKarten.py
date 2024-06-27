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

def getAllDecksBasedOnFormat(list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool) -> list[Deck]:

    decks_based_on_Format = []
    if(isNotFormat):
        decks_based_on_Format = [deck for deck in list_of_decks
                            if deck.format not in formatType.value]
    else:
        decks_based_on_Format = [deck for deck in list_of_decks
                            if deck.format in formatType.value]
    return decks_based_on_Format

def getRandomDecks(list_of_decks: list[Deck], formatType: FormatType, isNotFormat: bool, sample_size: int) -> list[Deck]:

    all_tournament_decks = getAllDecksBasedOnFormat(list_of_decks, formatType, isNotFormat)
    randomDecks = random.sample(all_tournament_decks, sample_size if sample_size > 0 else len(list_of_decks))

    return randomDecks

def getListOfHitsOnSpecificCard(decks: list[Deck], card_id: str) -> list[int]:

    
    
    decks_card_hit = []
    for deck in decks: 
        card_counter = deck.contains_card(card_id)
        if(card_counter != 0):
            decks_card_hit.append(1)
        else:
            decks_card_hit.append(0)
    
    return decks_card_hit 
"""""
    decks_card_hit = []
    for deck in decks: 
        decks_card_hit.append(deck.contains_card(card_id))
    return decks_card_hit
    
    """""