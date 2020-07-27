import os

from flask import \
    Flask, \
    request, \
    jsonify

from src.fpl_team_selector import team_selector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

SG_TEAM_ID = int(os.environ['SG_TEAM_ID'])

NON_OPTIONAL_PAYLOAD_PARAMS_LIVE = ['previous_gw', 'season', 'fpl_team_id', 'fpl_email', 'fpl_password']

NON_OPTIONAL_PAYLOAD_PARAMS_RETRO = [
    'previous_gw', 'season', 'previous_team_selection_path', 'budget', 'available_chips', 'available_transfers'
]

# TODO Make API access private
@app.route('/api', methods=['GET'])
def api():
    content = request.get_json()

    # Non-optional parameters
    for param in NON_OPTIONAL_PAYLOAD_PARAMS_LIVE:
        assert param in content.keys(), f'{param} not provided in payload'

    previous_gw = content['previous_gw']
    season = content['season']
    fpl_team_id = content['fpl_team_id']
    fpl_email = content['fpl_email']
    fpl_password = content['fpl_password']

    # Optional parameters
    try:
        player_overwrites = content['player_overwrites']
    except KeyError:
        player_overwrites = None

    try:
        team_prediction_scalars = content['team_prediction_scalars']
    except KeyError:
        team_prediction_scalars = None

    # Only save team selection to S3 if it is my own team
    if fpl_team_id == SG_TEAM_ID:
        save_selection = True
    else:
        save_selection = False

    output_dict = team_selector.main(
        live=True,
        previous_gw=previous_gw,
        season=season,
        save_selection=save_selection,
        fpl_team_id=fpl_team_id,
        fpl_email=fpl_email,
        fpl_password=fpl_password,
        player_overwrites=player_overwrites,
        team_prediction_scalars=team_prediction_scalars
    )

    return jsonify(output_dict)


@app.route('/retro', methods=['GET'])
def retro():
    content = request.get_json()

    # Non-optional parameters
    for param in NON_OPTIONAL_PAYLOAD_PARAMS_RETRO:
        assert param in content.keys(), f'{param} not provided in payload'

    previous_gw = content['previous_gw']
    season = content['season']
    previous_team_selection_path = content['previous_team_selection_path']
    budget = content['budget']
    available_chips = content['available_chips']
    available_transfers = content['available_transfers']

    # Optional parameters
    try:
        save_selection = content['save_selection']
    except KeyError:
        save_selection = False

    try:
        player_overwrites = content['player_overwrites']
    except KeyError:
        player_overwrites = {}

    try:
        team_prediction_scalars = content['team_prediction_scalars']
    except KeyError:
        team_prediction_scalars = {}

    output_dict = team_selector.main(
        live=False,
        previous_gw=previous_gw,
        season=season,
        save_selection=save_selection,
        previous_team_selection_path=previous_team_selection_path,
        budget=budget,
        available_chips=available_chips,
        available_transfers=available_transfers,
        player_overwrites=player_overwrites,
        team_prediction_scalars=team_prediction_scalars
    )

    return jsonify(output_dict)


if __name__ == '__main__':
    # need debug and threaded parameters to prevent TensorFlow error
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
