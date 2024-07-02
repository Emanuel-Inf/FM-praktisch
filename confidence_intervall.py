"""Calculate the probability for an archetype to be present in a deck.

We use confidence intervalls to determine how sure we can be that a given
archetype is in a deck.

@authors: Katin Kober, Emanuel Petrinovic, Max Weise
"""

import random

import numpy as np

import archetypes
import data_reader
from classes.deck import Deck

SAMPLE_SIZE: int = 10000


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
    archetype_cards: archetypes.Archetypes,
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

    counter = _count_decks_with_archetype(random_sample, archetype_cards.value)
    if counter == 0:
        print("Archetype not present.")
        return
    p = counter / sample_size
    standard_error = np.sqrt((p * (1 - p)) / sample_size)
    lower_bound = p - z_score * standard_error
    upper_bound = p + z_score * standard_error

    return (lower_bound if lower_bound > 0 else 0, upper_bound)


def calc_confidence_intervall(decks: list[Deck], sample_size):
    """Count appearence of archetypes and calculate confidence intervall."""

    # === Get Data ===
    print(f"Using {(sample_size / len(decks)) * 100 :.2f} % of the decks as sample")
    random_sample = random.sample(decks, sample_size)

    print("\n=== Snake Eyes ===")
    snake_eye_intervall = calculate_confidence_intervall(
        archetypes.Archetypes.SNAKE_EYES, random_sample
    )
    if snake_eye_intervall:
        print(f"I = [{snake_eye_intervall[0]:.4f}; {snake_eye_intervall[1]:.4f})")

    print("\n=== of Greed ===")
    of_greed_intervall = calculate_confidence_intervall(
        archetypes.Archetypes.OF_GREED, random_sample
    )
    if of_greed_intervall:
        print(f"I = [{of_greed_intervall[0]:.4f}; {of_greed_intervall[1]:.4f})")

    print("\n=== Salamandgreat ===")
    salamand_great_interval = calculate_confidence_intervall(
        archetypes.Archetypes.SALAMANDGREAT, random_sample
    )
    if salamand_great_interval:
        print(
            f"I = [{salamand_great_interval[0]:.4f}; {salamand_great_interval[1]:.4f})"
        )

    print("\n=== Branded ===")
    branded_intervall = calculate_confidence_intervall(
        archetypes.Archetypes.BRANDED, random_sample
    )
    if branded_intervall:
        print(f"I = [{branded_intervall[0]:.4f}; {branded_intervall[1]:.4f})")

    print("\n=== Rescue-ACE ===")
    rescue_intervall = calculate_confidence_intervall(
        archetypes.Archetypes.RESCUE_ACE, random_sample
    )
    if rescue_intervall:
        print(f"I = [{rescue_intervall[0]:.4f}; {rescue_intervall[1]:.4f})")


if __name__ == "__main__":
    decks = data_reader.get_All_Decks_Prepaired()
    sample_size = SAMPLE_SIZE if 0 < SAMPLE_SIZE <= len(decks) else len(decks)
    calc_confidence_intervall(decks, sample_size)
