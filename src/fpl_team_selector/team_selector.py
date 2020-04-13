import logging
from collections import namedtuple
import json

import pandas as pd
import numpy as np

from src.data.team_data import TeamData
from src.fpl_team_selector.solvers import \
    solve_starting_11_problem, \
    solve_fpl_team_selection_problem

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


LOW_VALUE_PLAYER_UPPER_LIMIT = 4.1
"""
Value below which a player is classified as low value.
"""

POINTS_PENALTY = 4
"""
Points deducted for each transfer in excess of free transfers.
"""

TeamSelectionCriteria = namedtuple(
    'TeamSelectionCriteria',
    ['max_permitted_transfers', 'include_top_3', 'include_low_value_player', 'min_spend']
)
"""
Named tuple for storing team selection parameters.
"""

TEAM_SELECTION_PERMUTATIONS = {

    '1 transfer': TeamSelectionCriteria(
        max_permitted_transfers=1,
        include_top_3=False,
        include_low_value_player=False,
        min_spend=0
    ),

    '2 transfer': TeamSelectionCriteria(
        max_permitted_transfers=2,
        include_top_3=False,
        include_low_value_player=False,
        min_spend=0
    ),

    '3 transfer': TeamSelectionCriteria(
        max_permitted_transfers=3,
        include_top_3=False,
        include_low_value_player=False,
        min_spend=0
    ),

    'Wildcard': TeamSelectionCriteria(
        max_permitted_transfers=15,
        include_top_3=False,
        include_low_value_player=False,
        min_spend=0
    )

}
"""
Dictionary of team selection permutations to iterate through.
"""

VARIABLES_FOR_TEAM_SELECTED_JSON = [
    'name',
    'position_DEF',
    'position_MID',
    'position_FWD',
    'position_GK',
    'team_name',
    'in_starting_11',
    'is_captain',
    'is_vice_captain',
    'bench_order',
    'GW_plus_1',
    'GW_plus_2',
    'GW_plus_3',
    'GW_plus_4',
    'GW_plus_5',
    'predictions'
]
"""
Variables to include in team selected JSON.
"""


def main(fpl_team_id, fpl_email, fpl_password, save_selection=False):
    # team_data = TeamData(
    #     fpl_team_id=fpl_team_id,
    #     fpl_email=fpl_email,
    #     fpl_password=fpl_password
    # )
    #
    # previous_team_selection = team_data.get_previous_team_selection()
    # budget = team_data.get_budget()
    # available_chips = team_data.get_available_chips()
    # available_transfers = team_data.get_available_transfers()

    # TODO Make it possible for manual assignment of parameters as well as live:
    previous_team_selection = pd.read_parquet('gw28_v3_lstm_team_selections.parquet')
    budget = 100.4
    available_chips = []
    available_transfers = 2

    current_predictions = generate_input_dataframe(previous_team_selection=previous_team_selection)

    results = {}

    permutations = _find_permutations(available_transfers, available_chips)

    for permutation in permutations:
        params = TEAM_SELECTION_PERMUTATIONS[permutation]

        selected_team_df, total_points = solve_fpl_team_selection_problem(
            current_predictions_df=current_predictions,
            budget_constraint=budget,
            max_permitted_transfers=params.max_permitted_transfers,
            include_top_3=params.include_top_3,
            include_low_value_player=params.include_low_value_player,
            min_spend=params.min_spend
        )

        # Penalty incurred if number of transfers exceeds available free transfers
        transfers_made = int(15 - selected_team_df['in_current_team'].sum())

        excess_transfers = int(max(0, transfers_made - available_transfers))

        total_points -= POINTS_PENALTY * excess_transfers

        results[permutation] = (selected_team_df, total_points)

    best_permutation = max(results, key=lambda k: results[k][1])  # Team which gets max total points

    # Find chips used:
    chips_used = []
    if best_permutation == 'Wildcard':
        chips_used = ['wildcard']

    # Get starting 11 for best team selection:
    best_selected_team_df = results[best_permutation][0]
    starting_11_names = solve_starting_11_problem(selected_team_df=best_selected_team_df)
    starting_11_df = pd.DataFrame({'name': starting_11_names})
    starting_11_df['in_starting_11'] = 1

    best_selected_team_df = best_selected_team_df.merge(starting_11_df, on='name', how='left')
    best_selected_team_df['in_starting_11'].fillna(0, inplace=True)

    # Record transfer data:
    players_out = list(
        set(previous_team_selection['name']) - set(best_selected_team_df['name'])
    )
    players_in = list(
        set(best_selected_team_df['name']) - set(previous_team_selection['name'])
    )

    _create_additional_variables_for_json(best_selected_team_df)

    best_selected_team_dict = best_selected_team_df[VARIABLES_FOR_TEAM_SELECTED_JSON].to_dict(orient='records')

    transfers_dict = {
        'players_in': players_in,
        'players_out': players_out,
        'chips_used': chips_used
    }

    output_dict = {
        "team_selected": best_selected_team_dict,
        "transfers": transfers_dict
    }

    if save_selection:
        # TODO Add function to save output (best_selected_team_df)
        pass

    with open('example_api_output.json', 'w') as json_file:
        json.dump(output_dict, json_file, ensure_ascii=False)

    return output_dict


def _create_additional_variables_for_json(best_selected_team_df):
    """
    Add 'is_captain', 'is_vice_captain' and 'bench_order' columns to `best_selected_team_df`.

    :param best_selected_team_df: DataFrame with selected team of 15 and starting 11 flag column.
    :return: None. Modifies DataFrame in-place.
    """
    # Captain and vice captain:
    best_selected_team_df['rank'] = best_selected_team_df.groupby('in_starting_11')['GW_plus_1'].rank(
        ascending=False,
        method='first'
    )

    best_selected_team_df['is_captain'] = np.where(
        (best_selected_team_df['rank'] == 1) & (best_selected_team_df['in_starting_11'] == 1),
        1,
        0
    )

    best_selected_team_df['is_vice_captain'] = np.where(
        (best_selected_team_df['rank'] == 2) & (best_selected_team_df['in_starting_11'] == 1),
        1,
        0
    )

    best_selected_team_df.drop('rank', axis=1, inplace=True)

    # Order of players on bench. Only outfield players can be ordered. All other players set to -1.
    best_selected_team_df['on_bench_not_gk'] = np.where(
        (best_selected_team_df['in_starting_11'] == 1) | (best_selected_team_df['position_GK'] == 1),
        0,
        1
    )
    best_selected_team_df['rank'] = best_selected_team_df.groupby('on_bench_not_gk')['GW_plus_1'].rank(
        ascending=False,
        method='first'
    )

    best_selected_team_df['bench_order'] = np.where(
        best_selected_team_df['on_bench_not_gk'] == 1,
        best_selected_team_df['rank'],
        -1
    )

    best_selected_team_df.drop(['rank', 'on_bench_not_gk'], axis=1, inplace=True)

    return best_selected_team_df


def _find_permutations(available_transfers, available_chips):
    # TODO Create test cases for each scenario
    if (available_transfers == 1) & ('wildcard' in available_chips):
        permutations = ['1 transfer', '2 transfer', 'Wildcard']

    elif (available_transfers == 1) & ('wildcard' not in available_chips):
        permutations = ['1 transfer', '2 transfer']

    elif (available_transfers == 2) & ('wildcard' in available_chips):
        permutations = ['1 transfer', '2 transfer', '3 transfer', 'Wildcard']

    elif (available_transfers == 2) & ('wildcard' not in available_chips):
        permutations = ['1 transfer', '2 transfer', '3 transfer']

    return permutations


# TODO Change so that it loads latest available or finds next gameweek by current date
# TODO Support loading from file and S3
def _load_player_predictions(prediction_filepath=None):
    """
    Load saved player points predictions as Pandas DataFrame
    :param prediction_filepath: Filepath to predictions parquet file
    :return: Pandas DataFrame
    """

    predictions = pd.read_parquet(prediction_filepath)
    if 'rank' in predictions.columns:
        predictions.drop('rank', inplace=True, axis=1)

    _format_player_predictions(predictions)

    return predictions


def _format_player_predictions(predictions_df):
    """
    Formats names in raw predictions DataFrame and sets 'predictions' column.
    :param predictions_df: Raw predictions DataFrame as loaded from parquet
    :return: None. Modifies DataFrame in-place
    """
    predictions_df.rename(columns={'sum': 'predictions'}, inplace=True)

    predictions_df['name'] = predictions_df['name'].str.replace(' ', '_')
    predictions_df['name'] = predictions_df['name'].str.replace('-', '_')


def _get_prev_predictions_for_missing_players_in_previous_team(previous_predictions, current_predictions_for_prev_team):
    """
    Returns DataFrame of previous predictions for players who are in the previously selected team but do not have points
    predictions for the current gameweek.

    :param previous_predictions: DataFrame of previous predictions
    :param current_predictions_for_prev_team: DataFrame of current predictions for previous team. Points predictions
    should be null for some players.
    :return: DataFrame
    """
    previous_predictions_missing_players = previous_predictions.merge(
        current_predictions_for_prev_team[current_predictions_for_prev_team['predictions'].isnull()][['name']],
        on='name',
        how='inner'
    )  # TODO Need to ensure they are not substituted out - could give arbitrary number of points to guarantee selection

    return previous_predictions_missing_players


def generate_input_dataframe(previous_team_selection):

    current_predictions = _load_player_predictions('gw29_v4_lstm_player_predictions.parquet')
    previous_predictions = _load_player_predictions('gw28_v4_lstm_player_predictions.parquet')

    previous_team_selection_names = previous_team_selection.copy()[['name', 'in_current_team']]

    current_predictions_for_prev_team = current_predictions.merge(previous_team_selection_names, on='name', how='inner')

    # Players without predictions will still be in current data but will have a null value
    if current_predictions_for_prev_team['predictions'].isnull().sum() != 0:
        logging.info('Some players missing')
        # Players not playing in next GW still appear but have null values for points predictions and next match value
        current_predictions.dropna(axis=0, how='any', inplace=True)

        previous_predictions_missing_players = _get_prev_predictions_for_missing_players_in_previous_team(
            previous_predictions=previous_predictions,
            current_predictions_for_prev_team=current_predictions_for_prev_team
        )

        # Append previous predictions for missing players who are in current team:
        current_predictions = current_predictions.append(previous_predictions_missing_players)

    current_predictions = current_predictions.merge(previous_team_selection, on='name', how='left')
    current_predictions['in_current_team'] = current_predictions['in_current_team'].fillna(0)
    assert current_predictions['in_current_team'].sum() == 15, 'Not all players in current team have points predictions'

    # Create top 3 flag:
    current_predictions.loc[0:2, 'in_top_3'] = 1
    current_predictions['in_top_3'].fillna(0, inplace=True)

    # Create low value player flag:
    current_predictions['low_value_player'] = np.where(
        current_predictions['next_match_value'] < LOW_VALUE_PLAYER_UPPER_LIMIT,
        1,
        0
    )

    return current_predictions
