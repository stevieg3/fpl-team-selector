import json

import pandas as pd
import numpy as np
import requests

FPL_LOGIN_URL = 'https://users.premierleague.com/accounts/login/'
"""
URL for FPL authentication
"""

MY_TEAM_URL = 'https://fantasy.premierleague.com/api/my-team/{}/'
"""
URL for FPL team data
"""

BOOTSTRAP_STATIC_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
"""
URL to API containing player characteristics data
"""

VALUE_MULTIPLE = 10
"""
Scalar difference between player value in raw data and on website
"""


class TeamData:
    def __init__(self, fpl_team_id, fpl_email, fpl_password):
        """
        Class for interacting with official FPL API to get team specific data. Credit to
        https://medium.com/@bram.vanherle1/fantasy-premier-league-api-authentication-guide-2f7aeb2382e4 for details on
        how to get authorised access to the API.

        :param fpl_team_id: Team ID number (found in URL of FPL website after login)
        :param fpl_email: FPL log-in email
        :param fpl_password: FPL log-in password
        """

        # Create session
        self.session = requests.session()

        payload = {
            'password': fpl_password,
            'login': fpl_email,
            'redirect_uri': 'https://fantasy.premierleague.com/a/login',
            'app': 'plfpl-web'
        }

        self.session.post(
            url=FPL_LOGIN_URL,
            data=payload
        )

        team_data_response = self.session.get(
            MY_TEAM_URL.format(fpl_team_id)
        )

        assert team_data_response.status_code == 200, 'Invalid credentials - check log in details and team ID.'

        # Get team data
        self.team_data_json = json.loads(team_data_response.text)
        self.team_data = pd.json_normalize(self.team_data_json, 'picks')

        # Get player data to match IDs in team data with corresponding player names
        player_data_response = self.session.get(BOOTSTRAP_STATIC_URL)  # Doesn't require auth
        player_data_json = json.loads(player_data_response.text)
        self.player_data = pd.json_normalize(player_data_json, 'elements')

        # Money in the bank
        self.money_in_bank = np.round(
            self.team_data_json['transfers']['bank'] / VALUE_MULTIPLE,
            2
        )

    def get_previous_team_selection(self):
        """
        Get previously selected team (15 players).

        :return: DataFrame of players in previously selected team with selling and purchase prices.
        """
        previous_team_selection = _add_player_names(
            team_data_df=self.team_data,
            player_data_df=self.player_data
        )

        previous_team_selection['selling_price'] /= VALUE_MULTIPLE
        previous_team_selection['purchase_price'] /= VALUE_MULTIPLE

        return previous_team_selection[['selling_price', 'purchase_price', 'name']]

    def get_budget(self):
        """
        Calculate player budget.

        :return: Player budget in £m.
        """
        budget = self.team_data['selling_price'].sum()
        budget /= VALUE_MULTIPLE
        budget += self.money_in_bank
        budget = np.round(budget, 2)

        return budget

    def get_available_chips(self):
        """
        Find available chips out of 'wildcard', 'freehit', 'bboost', '3xc'

        :return: List of available chips.
        """
        chips_df = pd.json_normalize(self.team_data_json, 'chips')
        available_chips = list(
            chips_df[chips_df['status_for_entry'] == 'available']['name']
        )

        return available_chips

    def get_available_transfers(self):
        """
        Find the number of available free transfers.

        :return: Number of available transfers.
        """
        available_transfers = self.team_data_json['transfers']['limit']

        return available_transfers


def _add_player_names(team_data_df, player_data_df):
    """
    Add player names to raw team_data DataFrame generated from API.

    :param team_data_df: DataFrame containing 'element' player identifier column.
    :param player_data_df: DataFrame containing player 'ID's and first and last names.

    :return: team_data_df with formatted player names appended.
    """
    combined = team_data_df.merge(
        player_data_df[['id', 'first_name', 'second_name']],
        left_on='element',
        right_on='id'
    )

    combined['name'] = combined['first_name'] + '_' + combined['second_name']
    combined['name'] = combined['name'].str.lower()

    return combined
