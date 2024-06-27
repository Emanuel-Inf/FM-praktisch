"""Calculate the probability for an archetype to be present in a deck.

We use confidence intervalls to determine how sure we can be that a given
archetype is in a deck.

@authors: Katin Kober, Emanuel Petrinovic, Max Weise
"""

import random

import numpy as np

import data_reader
from classes.deck import Deck

SAMPLE_SIZE: int = 250


def _archetype_is_present(
    deck: Deck, archetype_cards: set[str], limit: int = 5
) -> bool:
    """Determine if the deck contains an archetype.

    An archetype is present, when enough cards of a predetermined set
    are present in a given deck.

    Args:
        deck: The deck to scan.
        archetype_cards: A set of cards that belong to an archetype.
        limit (optional): How many cards need to be present to determine
            the presence of an archetype. Defaults to 5.

    Returns:
        True if the archetype is present
        False otherwise.
    """
    main_deck = set(deck.main_deck)
    counter = 0

    for card in archetype_cards:
        if card in main_deck:
            counter += 1

    return counter >= limit


def _count_decks_with_archetype(decks: list[Deck], archetype_cards: set[str]) -> int:
    """Count how many decks contain a given archetype.

    Args:
        decks: List of all decks to be scanned.
        archetype_cards: Set of cards that belong to an archetype.

    Returns:
        An integer that counts how many decks contain the given archetype.
    """
    d = [d for d in decks if _archetype_is_present(d, archetype_cards)]
    return len(d)


def calculate_confidence_intervall(
    archetype_cards: set[str],
    random_sample: list[Deck],
    z_score: float = 1.96,
) -> tuple[float, float] | None:
    """Calculate the confidence intervall for a given card an sample set.

    Args:
        card_id (str): The id of the card from the dataset.
        random_sample (list[deck]): A list of decs to search card_id in.
        z_score (floa): The z-score corresponding to a desired cofidence
                intervall. Defaults to 1.96.

    Returns:
        A list of p-values used for plotting. If the given archetype is not
        present, return None.
    """
    sample_size = len(random_sample)

    counter = _count_decks_with_archetype(random_sample, archetype_cards)
    if counter == 0:
        print("Archetype not present.")
        return
    p = counter / sample_size
    standard_error = np.sqrt((p * (1 - p)) / sample_size)
    lower_bound = p - z_score * standard_error
    upper_bound = p + z_score * standard_error

    return (lower_bound if lower_bound > 0 else 0, upper_bound)


def main():
    """Count appearence of archetypes and calculate confidence intervall."""
    # === define card sets ===
    snake_eyes: set[str] = {
        "09674034",
        "12058741",
        "45663742",
        "27260347",
        "48452496",
        "90241276",
        "89023486",
        "24081957",
        "53639887",
        "26700718",
        "74906081",
    }
    horus: set[str] = {
        "66214679",
        "11335209",
        "47330808",
        "99307040",
        "75830094",
        "11224103",
        "48229808",
        "09264485",
        "84941194",
        "74725513",
        "26984177",
        "16528181",
        "01490690",
    }
    kashtira: set[str] = {
        "32909498",
        "94392192",
        "31149212",
        "68304193",
        "78534861",
        "04928565",
        "34447918",
        "69540484",
        "82286798",
        "08953369",
        "33925864",
        "21639276",
        "71832012",
    }

    # === Get Data ===
    decks = data_reader.get_All_Decks_Prepaired()
    random_sample = random.sample(decks, SAMPLE_SIZE if SAMPLE_SIZE > 0 else len(decks))

    # Calculate confidence intervalls
    # Round all results to 4 decimal points
    print("Snake Eyes:")
    snake_eye_intervall = calculate_confidence_intervall(snake_eyes, random_sample)
    if snake_eye_intervall:
        print(f"I = [{snake_eye_intervall[0]:.4f}; {snake_eye_intervall[1]:.4f})")

    print("Kashtira:")
    kashtira_intervall = calculate_confidence_intervall(kashtira, random_sample)
    if kashtira_intervall:
        print(f"I = [{kashtira_intervall[0]:.4f}; {kashtira_intervall[1]:.4f})")

    print("Horus:")
    horus_intervall = calculate_confidence_intervall(horus, random_sample)
    if horus_intervall:
        print(f"I = [{horus_intervall[0]:.4f}; {horus_intervall[1]:.4f})")


if __name__ == "__main__":
    main()
