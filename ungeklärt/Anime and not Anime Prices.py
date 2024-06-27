"""""
    #Filterung von Karten

    #WCC
    wcc_filtered_decks = deckfilter.deck_Format_Filter(prepaired_decks, [FormatType.WORLDCC], False)
    
    wcc_random_sample = deckfilter.deck_Random_Filter(wcc_filtered_decks, SAMPLE_SIZE)
    
    wcc_valid_price_decks = deckfilter.deck_HasComplete_PriceInfo_Filter(wcc_random_sample)
  
    #for deck in wcc_valid_price_decks:
    #    print(f"Der Deck '{deck.name}' ist {ygoAPI.getCardPriceSum(deck.main_deck)} Wert.")

    #--------------------------------------------------------------#

    #Anime
    anime_decks = deckfilter.deck_Format_Filter(prepaired_decks, [FormatType.ANIME], False)
    
    anime_random_sample_anime = deckfilter.deck_Random_Filter(anime_decks, SAMPLE_SIZE)

    anime_valid_price_decks = deckfilter.deck_HasComplete_PriceInfo_Filter(anime_random_sample_anime)

    anime_deck_prices = [ygoAPI.getCardPriceSum(deck.main_deck) for deck in anime_valid_price_decks]

    #for deck in anime_valid_price_decks:
        #print(f"Der Deck '{deck.name}' ist {ygoAPI.getCardPriceSum(deck.main_deck)} Wert.")

    non_animee_decks = deckfilter.deck_Format_Filter(prepaired_decks, [FormatType.ANIME], True)
    
    non_animee_random_sample_decks = deckfilter.deck_Random_Filter(non_animee_decks, SAMPLE_SIZE)

    non_animee_valid_prices_decks = deckfilter.deck_HasComplete_PriceInfo_Filter(non_animee_random_sample_decks)

    non_animee_deck_prices = [ygoAPI.getCardPriceSum(deck.main_deck) for deck in non_animee_valid_prices_decks]

    #for deck in non_animee_valid_prices_decks:
    #    print(f"Der Deck '{deck.name}' ist {ygoAPI.getCardPriceSum(deck.main_deck)} Wert.")

    #shapiro(anime_deck_prices) # the shapiro test shows if the data is normally distributed
    #shapiro(non_animee_deck_prices)
        
    #levene(anime_deck_prices, non_animee_deck_prices) # the levene test is used to check if the two groups(anime decks, non anime decks) have a similar variance (required to perform a t-test)
    data = pd.DataFrame({
    'Preis': anime_deck_prices + non_animee_deck_prices,
    'Kategorie': ['Anime'] * len(anime_deck_prices) + ['Nicht-Anime'] * len(non_animee_deck_prices)
    })

    sns.boxplot(x='Kategorie', y='Preis', data=data)
    plt.title('Vergleich der Varianz zwischen Anime und Nicht-Anime Decks')
    plt.show()

    probplot(anime_deck_prices, dist="norm", plot=plt)
    plt.title('QQ-Plot für Anime-Deck Preise')
    plt.show()

    probplot(non_animee_deck_prices, dist="norm", plot=plt)
    plt.title('QQ-Plot für Nicht-Anime-Deck Preise')
    plt.show()


#Whitney-U Test
    statistics, p_value = mannwhitneyu(anime_deck_prices, non_animee_deck_prices, alternative='greater')

    if p_value < 0.05:
        print("Es gibt signifikante Hinweise darauf, dass Anime-Decks im Durchschnitt teurer sind als Nicht-Anime-Decks.")
    else:
        print("Es gibt keine signifikanten Hinweise darauf, dass Anime-Decks im Durchschnitt teurer sind als Nicht-Anime-Decks.")
    


#LOGISTISCHE REGRESSION
    prices = np.concatenate([anime_deck_prices, non_animee_deck_prices])
    deck_type = np.array([1]*len(anime_deck_prices) + [0]*len(non_animee_deck_prices))

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

"""""