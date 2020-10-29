import json
import urllib
import unidecode

import pandas as pd
import numpy as np

BOOTSTRAP_STATIC_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
"""
URL to API containing player characteristics data
"""

VALUE_MULTIPLE = 10
"""
Scalar difference between player value in raw data and on website
"""

CHANCE_OF_PLAYING_NEXT_ROUND_NULL_IMPUTATION = 100
"""
If null assume 100% chance of playing
"""


def _get_fpl_json(url):
    """
    Get JSON from API URL

    :param url: URL containing JSON
    :return: JSON file
    """
    with urllib.request.urlopen(url) as open_url:
        json_file = json.loads(open_url.read())
    return json_file


def get_latest_fpl_cost_and_chance_of_playing():
    """
    Use official FPL API to get latest price for players and chance of playing next game. We need latest price before
    using our optimizer as a given transfer may no longer be possible if a player's price has changed since prediction.

    :return:
    """

    player_data_json = _get_fpl_json(BOOTSTRAP_STATIC_URL)

    players_raw = pd.json_normalize(player_data_json, 'elements')

    players_raw['name'] = players_raw['first_name'] + '_' + players_raw['second_name']
    players_raw['name'] = players_raw['name'].str.lower()

    # Processing step needed before using PuLP
    players_raw['name'] = players_raw['name'].str.replace(' ', '_')
    players_raw['name'] = players_raw['name'].str.replace('-', '_')

    # Apply same step used to enable join between FPL and FFS data
    players_raw['name'] = players_raw['name'].str.replace(' ', '_').apply(
        lambda string: unidecode.unidecode(string)
    )

    players_raw['chance_of_playing_next_round'].fillna(CHANCE_OF_PLAYING_NEXT_ROUND_NULL_IMPUTATION, inplace=True)

    players_raw['now_cost'] /= VALUE_MULTIPLE

    players_raw['chance_of_playing_next_round'] /= 100  # Convert from % to decimal

    players_raw['chance_of_playing_next_round'] = np.round(players_raw['chance_of_playing_next_round'], 1)

    return players_raw[['name', 'now_cost', 'chance_of_playing_next_round']]
