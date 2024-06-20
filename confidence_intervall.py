from typing import Any

import numpy as np


def _count_card_in_sample(card_id: str, samples: list[Any]) -> int:
    counter = 0
    for sample in samples:
        if card_id in sample.main_deck:
            counter += 1

    return counter


def calculate_confidence_intervall(
    card_id: str,
    random_sample: list[Any],
    z_score: float = 1.96,
) -> list[float]:
    """Calculate the confidence intervall for a given card an sample set.

    Args:
        card_id (str): The id of the card from the dataset.
        random_sample (list[deck]): A list of decs to search card_id in.
        z_score (floa): The z-score corresponding to a desired cofidence
                intervall. Defaults to 1.96.

    Returns:
        A list of p-values used for plotting.
    """

    # 95% Konfideninteravll für
    sample_size = len(random_sample)
    sample_iterations = 10
    results = []
    for _ in range(sample_iterations):
        counter = _count_card_in_sample(card_id, random_sample)
        p = counter / sample_size
        standard_error = np.sqrt((p * (1 - p)) / sample_size)
        lower_bound = p - z_score * standard_error
        upper_bound = p + z_score * standard_error

        print(
            f"Mit 95% Sicherheit ist die wahre Häufigkeit die Karte {card_id} in dem Intervall [{lower_bound};{upper_bound}] enthalten \n"
        )

        results.append(p)

    return results
