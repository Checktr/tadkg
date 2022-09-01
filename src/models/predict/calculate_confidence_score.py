from src import config
import math

criteria = config['confidence']['score_criteria']

"""
Confidence Score Legend

1 - No Merger / No Acquisition
2 - Potentially Merger or Acquisition
3 - Likely Merger or Acquisition
4 - Confident Merger or Acquisition
5 - High Confidence Merger or Acquisition
"""


def matches_pattern(pattern, row):
    if 'either' in pattern:
        for prop in pattern['either']:
            if matches_pattern(prop, row):
                return True
        return False
    else:
        prop = list(pattern.keys())[0]
        if 'equals' in pattern[prop] and (str(row[prop]) == str(pattern[prop]['equals'])):
            return True
        elif 'greater_than' in pattern[prop] and (float(row[prop]) > float(pattern[prop]['equals'])):
            return True
        elif 'less_than' in pattern[prop] and (float(row[prop]) < float(pattern[prop]['equals'])):
            return True
        else:
            return False


def calculate_sentence_confidence_score(row) -> int:
    """
    Calculate the confidence score of a processed sentence.

    Confidence scores are determined based on patterns defined within the "confidence" section
    in the config.yml file.

    Confidence scores should be graded out of the total number of patterns defined within the configuration.

    :return: int, confidence score
    """
    score = 0
    for pattern in criteria:
        if matches_pattern(pattern, row):
            score += 1
    return score


def calculate_abstracted_confidence_score(instances) -> int:
    """
    Calculate the confidence score of either a webpage or a company

    :param instances: dataframe of sentences or webpages
    :return: int, confidence score
    """
    # get average of confidence scores of instances
    score = math.floor(sum(list(instances["confidenceScore"])) / len(instances["confidenceScore"]))

    # if more than 3 instances detected, add 1 (MAX score 5)
    # if score < 5 and len(instances["confidenceScore"]) > 3:
    #     score += 1

    return score
