"""Main file for the code project.

@author: Katrin Kober, Emanuel Petrinovic, Max Weise
"""

import Clusteranalyse.clusteranalyse as clusteranalyse
import Regressionen.logisticRegressionViews as logisticRegressionViews

CSV_DIRECTORY: str = "./daten/csv/"
SEARCH_CARDS = ["83764719", "47826112", "3643300", "53804307", "81843628", "36553319"]
SAMPLE_SIZE: int = 250


def main():
    """Executes the main function."""

    clusteranalyse.startCluserAnalyse()
    logisticRegressionViews.startLogRegressionForViews()


if __name__ == "__main__":
    main()

