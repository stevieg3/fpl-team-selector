import logging
from collections import namedtuple

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from src.data.team_data import TeamData
from src.data.s3_utilities import \
    s3_filesystem, \
    S3_BUCKET_PATH, \
    GW_PREDICTIONS_SUFFIX, \
    write_dataframe_to_s3
from src.data.latest_fpl_data import get_latest_fpl_cost_and_chance_of_playing
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
    ['max_permitted_transfers', 'include_top_3', 'number_of_low_value_players', 'min_spend']
)
"""
Named tuple for storing team selection parameters.
"""

TEAM_SELECTION_PERMUTATIONS = {

    '1 transfer': TeamSelectionCriteria(
        max_permitted_transfers=1,
        include_top_3=False,
        number_of_low_value_players=0,
        min_spend=0
    ),

    '2 transfer': TeamSelectionCriteria(
        max_permitted_transfers=2,
        include_top_3=False,
        number_of_low_value_players=0,
        min_spend=0
    ),

    '3 transfer': TeamSelectionCriteria(
        max_permitted_transfers=3,
        include_top_3=False,
        number_of_low_value_players=0,
        min_spend=0
    ),

    'wildcard': TeamSelectionCriteria(
        max_permitted_transfers=15,
        include_top_3=False,
        number_of_low_value_players=0,
        min_spend=0
    ),

    'freehit': TeamSelectionCriteria(
        max_permitted_transfers=15,
        include_top_3=False,
        number_of_low_value_players=0,
        min_spend=0
    ),

    'new_team': TeamSelectionCriteria(
        max_permitted_transfers=15,
        include_top_3=False,
        number_of_low_value_players=0,
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
    'predictions',
    'now_cost',
    'purchase_price',
    'gw_introduced_in',
    'in_current_team',
    'model'
]
"""
Variables to include in team selected JSON.
"""


SEASON_ORDER_DICT = {
    '2016-17': 1,
    '2017-18': 2,
    '2018-19': 3,
    '2019-20': 4,
    '2020-21': 5
}
"""
Order of seasons
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
        logging.info('Working on live run')
        fpl_team_id = kwargs['fpl_team_id']
        fpl_email = kwargs['fpl_email']
        fpl_password = kwargs['fpl_password']
        player_overwrites = kwargs['player_overwrites']
        team_prediction_scalars = kwargs['team_prediction_scalars']

        if previous_gw == 38:
            # New team selection
            previous_team_selection = pd.DataFrame({'name': []})
            budget = 100
            available_chips = ['new_team']
            available_transfers = 15
        else:
            team_data = TeamData(
                fpl_team_id=fpl_team_id,
                fpl_email=fpl_email,
                fpl_password=fpl_password
            )

            previous_team_selection = team_data.get_previous_team_selection()
            budget = team_data.get_budget()
            available_chips = team_data.get_available_chips()
            available_transfers = team_data.get_available_transfers()

        logging.info(f"Budget: {budget}")
        logging.info(f"Available chips: {available_chips}")
        logging.info(f"Available transfers: {available_transfers}")

    else:  # retro run
        logging.info('Working on retro run')
        previous_team_selection_path = kwargs['previous_team_selection_path']
        budget = kwargs['budget']
        available_chips = kwargs['available_chips']
        available_transfers = kwargs['available_transfers']
        player_overwrites = kwargs['player_overwrites']
        team_prediction_scalars = kwargs['team_prediction_scalars']

        if previous_gw == 38:
            previous_team_selection = pd.DataFrame({'name': []})  # Create empty DataFrame to enable transfers in/out
        else:
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

    # For gameweeks 35 onwards only need to look at 1-4 gameweek predictions in advance:
    if (previous_gw >= 34) and (previous_gw != 38):
        for i in range(38-previous_gw+1, 6):
            current_predictions['predictions'] -= current_predictions[f'GW_plus_{i}']

    # Apply manual player prediction overwrites:
    if player_overwrites:
        for player, prediction_overwrite in player_overwrites.items():
            logging.info(f'Overwriting prediction for {player} with {prediction_overwrite}')
            current_predictions.loc[current_predictions['name'] == player, 'predictions'] = prediction_overwrite

    # Apply manual team prediction scalars:
    if team_prediction_scalars:
        for team, prediction_scalar in team_prediction_scalars.items():
            current_predictions.loc[current_predictions['team_name'] == team, 'predictions'] *= prediction_scalar

    if live:
        # Get latest player prices and chance of playing next game from FPL API:
        latest_player_data = get_latest_fpl_cost_and_chance_of_playing()
        current_predictions.drop('_merge', axis=1, inplace=True)
        current_predictions = current_predictions.merge(
            latest_player_data,
            on='name',
            how='left',
            indicator=True
        )

        logging.info(current_predictions['_merge'].value_counts())
        players_without_latest_price = current_predictions[current_predictions['_merge'] == 'left_only']['name']

        logging.info(f'Players without latest price: {players_without_latest_price}')

        logging.info('Only keeping players with latest price')  # Likely due to being transferred
        current_predictions = current_predictions[current_predictions['_merge'] == 'both']
        current_predictions.drop('_merge', axis=1, inplace=True)

        assert current_predictions[['now_cost', 'chance_of_playing_next_round']].isnull().sum().sum() == 0, \
            'Latest price and chance of playing next game data missing for some players'

        logging.info('Players whose price changed since predictions made')
        logging.info(
            current_predictions[
                current_predictions['now_cost'] != current_predictions['next_match_value']
            ][
                ['name', 'next_match_value', 'now_cost']
            ]
        )

        # Scale next gameweek predictions by chance of playing
        current_predictions['GW_plus_1'] *= current_predictions['chance_of_playing_next_round']

        # For players in current team the cost of 'buying them back' is the selling price:
        current_predictions['now_cost'] = np.where(
            current_predictions['in_current_team'] == 1,
            current_predictions['selling_price'],
            current_predictions['now_cost']
        )
    else:  # retro run
        current_predictions['now_cost'] = current_predictions['next_match_value'].copy()

        # For players in current team the cost of 'buying them back' is the selling price:
        current_predictions['now_cost'] = np.where(
            current_predictions['in_current_team'] == 1,
            current_predictions['selling_price'],
            current_predictions['now_cost']
        )

    final_output = {}

    permutations = _find_permutations(available_chips)

    logging.info(f'Possible permutations: {permutations}')

    for permutation in permutations:
        logging.info(f"Working on permutation {permutation}")
        params = TEAM_SELECTION_PERMUTATIONS[permutation]

        selected_team_df, total_points = solve_fpl_team_selection_problem(
            current_predictions_df=current_predictions,
            budget_constraint=budget,
            max_permitted_transfers=params.max_permitted_transfers,
            include_top_3=params.include_top_3,
            number_of_low_value_players=params.number_of_low_value_players,
            min_spend=params.min_spend
        )

        # Penalty incurred if number of transfers exceeds available free transfers
        transfers_made = int(15 - selected_team_df['in_current_team'].sum())
        logging.info(f"Transfers made: {transfers_made}")

        excess_transfers = int(max(0, transfers_made - available_transfers))
        if (permutation == 'wildcard') | (permutation == 'freehit'):
            excess_transfers = 0
        logging.info(f"Excess transfers: {excess_transfers}")

        total_points -= POINTS_PENALTY * excess_transfers

        logging.info(f"Total points using permutation {permutation}: {total_points}")

        # Find chips used:
        chips_used = []
        if permutation == 'wildcard':
            chips_used = ['wildcard']
        if permutation == 'freehit':
            chips_used = ['freehit']

        starting_11_names = solve_starting_11_problem(selected_team_df=selected_team_df)
        starting_11_df = pd.DataFrame({'name': starting_11_names})
        starting_11_df['in_starting_11'] = 1

        selected_team_df = selected_team_df.merge(starting_11_df, on='name', how='left')
        selected_team_df['in_starting_11'].fillna(0, inplace=True)

        # Record transfer data:
        players_out = list(
            set(previous_team_selection['name']) - set(selected_team_df['name'])
        )
        players_in = list(
            set(selected_team_df['name']) - set(previous_team_selection['name'])
        )

        _create_additional_variables_for_json(selected_team_df)

        # Record purchase_price and gw_introduced_in

        if selected_team_df['in_current_team'].sum() == 0:
            selected_team_df['purchase_price'] = selected_team_df['now_cost'].copy()
            if previous_gw == 38:  # new team
                selected_team_df['gw_introduced_in'] = 1
            else:  # e.g. wildcard mid season which replaces full team
                selected_team_df['gw_introduced_in'] = previous_gw + 1
        else:
            selected_team_df.loc[
                selected_team_df['in_current_team'] == 0,
                'purchase_price'
            ] = selected_team_df['now_cost']

            selected_team_df.loc[
                selected_team_df['in_current_team'] == 0,
                'gw_introduced_in'
            ] = previous_gw + 1

        best_selected_team_dict = selected_team_df[VARIABLES_FOR_TEAM_SELECTED_JSON].to_dict(orient='records')

        transfers_dict = {
            'players_in': players_in,
            'players_out': players_out,
            'chips_used': chips_used
        }

        output_dict = {
            "team_selected": best_selected_team_dict,
            "transfers": transfers_dict,
            "total_points": total_points
        }

        final_output[permutation] = output_dict

    if save_selection:
        logging.info('Saving best team')
        best_permutation = max(
            final_output,
            key=lambda k: final_output[k]['total_points']
        )
        logging.info(f'Best permutation: {best_permutation}')

        if previous_gw == 38:
            # First gameweek of the season
            season_order = SEASON_ORDER_DICT[season]
            season = {v: k for k, v in SEASON_ORDER_DICT.items()}[season_order + 1]
            previous_gw = 0

        selected_team_df = pd.DataFrame(final_output[best_permutation]['team_selected'])
        selected_team_df['gw'] = previous_gw + 1
        selected_team_df['season'] = season
        write_dataframe_to_s3(
            selected_team_df,
            s3_root_path=S3_BUCKET_PATH + '/gw_team_selections',
            partition_cols=['season', 'gw'],
            partition_filename_cb=lambda x: f'{x[0]}-{x[1]}.parquet'
        )
        logging.info(selected_team_df)
        logging.info("Saved team selection to S3")

    return final_output


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


def _find_permutations(available_chips):
    """
    Find team selection permutations to iterate through

    :param available_chips: List of available FPL chips
    :return: List of permutations as defined in `TEAM_SELECTION_PERMUTATIONS`
    """

    if 'new_team' in available_chips:
        permutations = ['new_team']
    else:
        permutations = ['1 transfer', '2 transfer', '3 transfer'] + available_chips

    permutations.remove('bboost')
    permutations.remove('3xc')

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
    if previous_gw == 38:
        # First gameweek of the season
        season_order = SEASON_ORDER_DICT[season]
        next_season = {v: k for k, v in SEASON_ORDER_DICT.items()}[season_order + 1]

        current_predictions = _load_player_predictions(previous_gw=1, season=next_season)
        current_predictions['in_current_team'] = 0  # Needed for solver

        # Drop nulls due to players who left after end of season otherwise solver will error:
        current_predictions['drop_row'] = np.where(
            (current_predictions['gw'] == 1) & (current_predictions['predictions'].isnull()),
            1,
            0
        )
        current_predictions = current_predictions[current_predictions['drop_row'] == 0]
        current_predictions.drop('drop_row', axis=1, inplace=True)

    else:
        current_predictions = _load_player_predictions(previous_gw=previous_gw+1, season=season)
        logging.info("Top 5 predictions:")
        logging.info(current_predictions.head())
        previous_predictions = _load_player_predictions(previous_gw=previous_gw, season=season)

        previous_team_selection['in_current_team'] = 1

        if 'gw_introduced_in' in previous_team_selection.columns:  # Only needed for retro budget calculations. Will be
            # in team selection saved to S3
            previous_team_selection_names = previous_team_selection.copy()[
                ['name', 'in_current_team', 'purchase_price', 'gw_introduced_in', 'selling_price']
            ]
        else:
            previous_team_selection_names = previous_team_selection.copy()[
                ['name', 'in_current_team', 'purchase_price', 'selling_price']
            ]

        current_predictions_for_prev_team = current_predictions.merge(
            previous_team_selection_names,
            on='name',
            how='inner'
        )

        # Players without predictions will still be in current data but will have a null value
        if current_predictions_for_prev_team['predictions'].isnull().sum() != 0:
            logging.info('Some players missing')
            # Players not playing in next GW still appear but have null values for points predictions and next match
            # value
            current_predictions.dropna(axis=0, how='any', inplace=True)

            previous_predictions_missing_players = _get_prev_predictions_for_missing_players_in_previous_team(
                previous_predictions=previous_predictions,
                current_predictions_for_prev_team=current_predictions_for_prev_team
            )

            # Append previous predictions for missing players who are in current team:
            current_predictions = current_predictions.append(previous_predictions_missing_players)

        current_predictions = current_predictions.merge(previous_team_selection_names, on='name', how='left')
        current_predictions['in_current_team'] = current_predictions['in_current_team'].fillna(0)

        assert current_predictions['in_current_team'].sum() == 15, \
            'Not all players in current team have points predictions'
        # TODO Assertion will fail for players who leave mid season but are in current team. Need to accommodate. Not
        #  urgent because players selected are likely to be important to their respective team and less likely to be
        #  sold mid-season.

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
