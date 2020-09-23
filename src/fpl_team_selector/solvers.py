import pandas as pd
from pulp import \
    lpSum, \
    LpProblem, \
    LpMaximize, \
    LpVariable, \
    LpInteger


def solve_fpl_team_selection_problem(
        current_predictions_df,
        budget_constraint,
        max_permitted_transfers=1,
        include_top_3=False,
        number_of_low_value_players=0,
        min_spend=0
):
    """
    Use PuLP to select the best team of 15 players which maximises total predicted points subject to constraints.

    :param current_predictions_df: Predictions DataFrame
    :param budget_constraint: Total budget available
    :param max_permitted_transfers: Maximum number of transfers allowed
    :param include_top_3: Include top 3 players in final squad (identified using `in_top_3` column)
    :param number_of_low_value_players: Number of low value players to force into squad (identified using
    `low_value_player` column)
    :param min_spend: Minimum amount spent on final squad. Use to prevent value of team being too low

    :return: DataFrame of selected team
    """
    current_predictions = current_predictions_df.copy()

    team_names = current_predictions['team_name'].unique()
    current_predictions = pd.get_dummies(current_predictions, columns=['team_name'])
    players = list(current_predictions['name'])

    # CREATE NAME-VALUE DICTIONARIES FOR USE IN CONSTRAINTS

    team_dict = {}
    for team in team_names:
        team_dict[team] = dict(zip(current_predictions['name'], current_predictions[f'team_name_{team}']))

    # Use latest price not prediction price:
    costs = dict(zip(current_predictions['name'], current_predictions['now_cost']))

    predictions = dict(zip(current_predictions['name'], current_predictions['predictions']))

    DEF_flag = dict(zip(current_predictions['name'], current_predictions['position_DEF']))

    FWD_flag = dict(zip(current_predictions['name'], current_predictions['position_FWD']))

    GK_flag = dict(zip(current_predictions['name'], current_predictions['position_GK']))

    MID_flag = dict(zip(current_predictions['name'], current_predictions['position_MID']))

    current_team = dict(zip(current_predictions['name'], current_predictions['in_current_team']))

    in_top_3 = dict(zip(current_predictions['name'], current_predictions['in_top_3']))

    low_value_flag = dict(zip(current_predictions['name'], current_predictions['low_value_player']))

    # SET OBJECTIVE FUNCTION

    prob = LpProblem('FPL team selection', LpMaximize)
    player_vars = LpVariable.dicts('player', players, 0, 1, LpInteger)

    prob += lpSum([predictions[p] * player_vars[p] for p in players]), "Total predicted points"

    # DEFINE CONSTRAINTS

    # Rules-of-the-game constraints:

    # When max_permitted_transfers == 0, sale of all players may not raise enough funds to buy back same team due to
    # tax. Hence remove budget constraint so team remains unchanged i.e. no transfers made.
    if max_permitted_transfers != 0:
        prob += lpSum([costs[p] * player_vars[p] for p in players]) <= budget_constraint, "Total cost less than X"

    prob += lpSum(player_vars[p] for p in players) == 15, "Select 15 players"

    prob += lpSum(DEF_flag[p] * player_vars[p] for p in players) == 5, "5 defenders"

    prob += lpSum(GK_flag[p] * player_vars[p] for p in players) == 2, "2 goalkeepers"

    prob += lpSum(MID_flag[p] * player_vars[p] for p in players) == 5, "5 midfielders"

    prob += lpSum(FWD_flag[p] * player_vars[p] for p in players) == 3, "3 forwards"

    for team in team_dict.keys():
        prob += lpSum(team_dict[team][p] * player_vars[p] for p in players) <= 3, f"Max 3 players in the same {team}"

    # Additional constraints:

    if include_top_3:
        prob += lpSum(in_top_3[p] * player_vars[p] for p in players) == 3, "Top 3 must be included"

    if number_of_low_value_players > 0:
        prob += lpSum(low_value_flag[p] * player_vars[p] for p in players) >= number_of_low_value_players, \
            "Include X low value players"

    if min_spend > 0:
        prob += lpSum([costs[p] * player_vars[p] for p in players]) >= min_spend, "Total cost greater than X"

    prob += lpSum(current_team[p] * player_vars[p] for p in players) >= 15 - max_permitted_transfers, \
        "At least 15-`max_permitted_transfers` players from original team"

    # SOLVE OBJECTIVE FUNCTION SUBJECT TO CONSTRAINTS

    prob.solve()
    assert prob.status == 1, 'FPL team selection problem not solved!'

    # Get predictions data for chosen players
    chosen_players = []
    for v in prob.variables():
        if v.varValue == 0:
            continue
        else:
            chosen_players.append(v.name.replace('player_', ''))

    selected_team_df = current_predictions_df.merge(
        pd.DataFrame({'name': chosen_players}),
        on='name',
        how='inner'
    )

    total_predicted_points_next_5_gws = int(selected_team_df['predictions'].sum())

    return selected_team_df, total_predicted_points_next_5_gws


def solve_starting_11_problem(selected_team_df):
    """
    Use PuLP to select the starting 11 which maximises next fixture points subject to constraints.

    :param selected_team_df: DataFrame of players in selected team
    :return: List of starting 11 players
    """
    selected_team = selected_team_df.copy()

    players = list(selected_team['name'])

    # CREATE NAME-VALUE DICTIONARIES FOR USE IN CONSTRAINTS

    predictions = dict(zip(selected_team['name'], selected_team['GW_plus_1']))

    DEF_flag = dict(zip(selected_team['name'], selected_team['position_DEF']))

    FWD_flag = dict(zip(selected_team['name'], selected_team['position_FWD']))

    GK_flag = dict(zip(selected_team['name'], selected_team['position_GK']))

    MID_flag = dict(zip(selected_team['name'], selected_team['position_MID']))

    # SET OBJECTIVE FUNCTION

    prob = LpProblem('FPL team selection', LpMaximize)
    player_vars = LpVariable.dicts('player', players, 0, 1, LpInteger)

    prob += lpSum([predictions[p] * player_vars[p] for p in players]), "Total predicted points"

    # DEFINE CONSTRAINTS

    # Rules of the game constraints:

    prob += lpSum(player_vars[p] for p in players) == 11, "Select 11 players"

    prob += lpSum(DEF_flag[p] * player_vars[p] for p in players) >= 3, "At least 3 defenders"

    prob += lpSum(GK_flag[p] * player_vars[p] for p in players) == 1, "1 goalkeeper"

    prob += lpSum(FWD_flag[p] * player_vars[p] for p in players) >= 1, "At least 1 forward"

    # SOLVE OBJECTIVE FUNCTION SUBJECT TO CONSTRAINTS

    prob.solve()
    assert prob.status == 1, 'FPL team selection problem not solved!'

    # Return list of chosen player names:
    chosen_players = []
    for v in prob.variables():
        if v.varValue == 0:
            continue
        else:
            chosen_players.append(v.name.replace('player_', ''))

    return chosen_players
