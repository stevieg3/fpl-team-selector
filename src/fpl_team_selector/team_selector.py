import logging
from collections import namedtuple
import json

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from src.data.team_data import TeamData
from src.data.s3_utilities import \
    s3_filesystem, \
    S3_BUCKET_PATH, \
    GW_PREDICTIONS_SUFFIX, \
    write_dataframe_to_s3
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


def main(live, previous_gw, season, save_selection=False, **kwargs):
    """
    Function for making team selection.

    :param live: Boolean. True for live predictions, False for retro
    :param previous_gw: Gameweek prior to the one you want to make a selection for
    :param season: FPL season
    :param save_selection: Boolean. Whether or not to save team selection to S3
    :return: Dictionary containing selected team and transfer information
    """

    if live:
        fpl_team_id = kwargs['fpl_team_id']
        fpl_email = kwargs['fpl_email']
        fpl_password = kwargs['fpl_password']

        team_data = TeamData(
            fpl_team_id=fpl_team_id,
            fpl_email=fpl_email,
            fpl_password=fpl_password
        )

        previous_team_selection = team_data.get_previous_team_selection()
        budget = team_data.get_budget()
        available_chips = team_data.get_available_chips()

        # Project restart  # TODO Remove after gameweek 30+ (special wildcard for restart)
        available_chips = ['wildcard']

        available_transfers = team_data.get_available_transfers()

        logging.info(f"Budget: {budget}")
        logging.info(f"Available chips: {available_chips}")
        logging.info(f"Available transfers: {available_transfers}")

    else:  # retro run
        previous_team_selection_path = kwargs['previous_team_selection_path']
        budget = kwargs['budget']
        available_chips = kwargs['available_chips']
        available_transfers = kwargs['available_transfers']

        previous_team_selection = pq.read_table(
            previous_team_selection_path,
            filesystem=s3_filesystem
        ).to_pandas()

        logging.info(f"Budget: {budget}")
        logging.info(f"Available chips: {available_chips}")
        logging.info(f"Available transfers: {available_transfers}")

    current_predictions = generate_input_dataframe(
        previous_gw=previous_gw,
        season=season,
        previous_team_selection=previous_team_selection
    )

    results = {}

    permutations = _find_permutations(available_transfers, available_chips)

    logging.info(f'Possible permutations: {permutations}')

    for permutation in permutations:
        logging.info(f"Working on permutation {permutation}")
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
        logging.info(f"Transfers made: {transfers_made}")

        excess_transfers = int(max(0, transfers_made - available_transfers))
        if permutation == 'Wildcard':
            excess_transfers = 0
        logging.info(f"Excess transfers: {excess_transfers}")

        total_points -= POINTS_PENALTY * excess_transfers

        results[permutation] = (selected_team_df, total_points)

        logging.info(f"Total points using permutation {permutation}: {results[permutation][1]}")

    best_permutation = max(results, key=lambda k: results[k][1])  # Team which gets max total points
    logging.info(f"Best permutation: {best_permutation}")

    # TODO Control choice if points prediction is the same for 2 different permutations

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

    # TODO Why didn't this run when no transfers were made?
    if save_selection:
        best_selected_team_df['gw'] = previous_gw + 1
        best_selected_team_df['season'] = season
        write_dataframe_to_s3(
            best_selected_team_df,
            s3_root_path=S3_BUCKET_PATH + '/gw_team_selections',
            partition_cols=['season', 'gw']
        )
        logging.info("Saved team selection to S3")

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


def _load_player_predictions(previous_gw, season):
    """
    Load saved player points predictions as Pandas DataFrame
    :param prediction_filepath: S3 path to predictions parquet file
    :return: Pandas DataFrame
    """

    all_predictions = pq.read_table(
        S3_BUCKET_PATH + GW_PREDICTIONS_SUFFIX + f'/season={season}' + f'/gw={previous_gw}',
        filesystem=s3_filesystem
    ).to_pandas()

    all_predictions['season'] = season
    all_predictions['gw'] = previous_gw

    predictions = all_predictions.copy()

    logging.info(f'Using predictions from season {season}, GW {previous_gw}')

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
    )

    return previous_predictions_missing_players


def generate_input_dataframe(previous_gw, season, previous_team_selection):
    # TODO Will need to change for GW 1
    current_predictions = _load_player_predictions(previous_gw=previous_gw+1, season=season)
    logging.info("Top 5 predictions:")
    logging.info(current_predictions.head())
    previous_predictions = _load_player_predictions(previous_gw=previous_gw, season=season)

    previous_team_selection['in_current_team'] = 1
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
