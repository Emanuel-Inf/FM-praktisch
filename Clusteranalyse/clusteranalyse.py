"""Run a cluster-analysis based on retrieved data

@authors: Katrin Kober, Emanuel Petrinovic, Max Weise
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import API.cardAPIDatensatz1 as cardAPIDatensatz1
import data_reader
import ENUMS.formatTypes as FormatType
import Regressionen.logisticRegressionViews as logisticRegressionViews
from classes.deck import Deck


def startCluserAnalyse():
    SAMPLE_SIZE: int = 10000
    ATTRIBUTE = "archetype"

    prepaired_decks = data_reader.get_All_Decks_Prepaired()
    random_competitive_decks = logisticRegressionViews.getRandomDecks(
        prepaired_decks, FormatType.FormatType.CASUAL, False, SAMPLE_SIZE
    )

    attribute_list, deck_dicts = getAttributeList(random_competitive_decks, ATTRIBUTE)

    matrix = prepairMatrix(attribute_list, deck_dicts, ATTRIBUTE)

    clusters, deck_matrix_pca, cluster_archetypes = calculateDominantArchetype(
        matrix, deck_dicts, ATTRIBUTE
    )
    plotCluster(clusters, deck_matrix_pca, cluster_archetypes)


def getAttributeList(decks: list[Deck], attribtue: str):
    attribtue_List = []
    deck_dicts = []

    for deck in decks:

        deck_info = {"deck_num": deck.deck_num, attribtue: []}

        for card in deck.main_deck:
            card = cardAPIDatensatz1.getCardFromCache(card)
            if card.get_attribute(attribtue) is not None:
                deck_info[attribtue].append(card.get_attribute(attribtue))
            if card.get_attribute(attribtue) not in attribtue_List:
                attribtue_List.append(card.get_attribute(attribtue))

        deck_dicts.append(deck_info)

    return attribtue_List, deck_dicts


def prepairMatrix(attribute_list, deck_dicts, attribute: str):

    num_decks = len(deck_dicts)
    num_archetypes = len(attribute_list)
    deck_matrix = np.zeros((num_decks, num_archetypes), dtype=int)

    # Fülle die Matrix basierend auf den Archetypen in jedem Deck
    for i, deck_info in enumerate(deck_dicts):
        attributes_in_deck = deck_info[attribute]
        for attribute_in_deck in attributes_in_deck:
            if attribute_in_deck in attribute_list:
                j = attribute_list.index(attribute_in_deck)
                deck_matrix[i, j] = 1

    return deck_matrix


def calculateDominantArchetype(deck_matrix, deck_dicts: list[Deck], attribute: str):

    # Anwendung von PCA zur Reduzierung der Dimensionalität auf 2 Dimensionen
    pca = PCA(n_components=2)
    deck_matrix_pca = pca.fit_transform(deck_matrix)

    # Anwendung des K-Means-Algorithmus auf den PCA-transformierten Daten
    kmeans = KMeans(n_clusters=4, random_state=0)
    clusters = kmeans.fit_predict(deck_matrix_pca)

    # Dominante Archetypen für jedes Cluster berechnen
    cluster_archetypes = {}
    for cluster_label in np.unique(clusters):
        cluster_archetypes[cluster_label] = []
        for i, deck_info in enumerate(deck_dicts):
            if clusters[i] == cluster_label:
                cluster_archetypes[cluster_label].extend(deck_info[attribute])
        # Finde den dominanten Archetyp für das Cluster
        dominant_archetype = max(
            set(cluster_archetypes[cluster_label]),
            key=cluster_archetypes[cluster_label].count,
        )
        cluster_name = (
            f"Cluster {cluster_label + 1}: Dominant Archetype '{dominant_archetype}'"
        )
        cluster_archetypes[cluster_label] = cluster_name

    return clusters, deck_matrix_pca, cluster_archetypes


def plotCluster(clusters, deck_matrix_pca, cluster_archetypes):

    plt.figure(figsize=(10, 8))
    for cluster_label in np.unique(clusters):
        plt.scatter(
            deck_matrix_pca[clusters == cluster_label, 0],
            deck_matrix_pca[clusters == cluster_label, 1],
            label=cluster_archetypes[cluster_label],
        )

    plt.title("Clusteranalyse der Decks basierend auf Archetypen")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.show()

