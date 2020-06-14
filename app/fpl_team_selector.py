import os

from flask import \
    Flask, \
    request, \
    jsonify

from src.fpl_team_selector import team_selector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

SG_TEAM_ID = os.environ['SG_TEAM_ID']

# TODO Make API access private
@app.route('/api', methods=['GET'])
def api():
    content = request.get_json()

    previous_gw = int(content['previous_gw'])
    season = content['season']
    fpl_team_id = int(content['fpl_team_id'])
    fpl_email = content['fpl_email']
    fpl_password = content['fpl_password']

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
        fpl_password=fpl_password
    )

    return jsonify(output_dict)


@app.route('/retro', methods=['GET'])
def retro():
    content = request.get_json()

    previous_gw = int(content['previous_gw'])
    season = content['season']
    previous_team_selection_path = content['previous_team_selection_path']
    budget = float(content['budget'])
    available_chips = list(content['available_chips'])
    available_transfers = int(content['available_transfers'])

    output_dict = team_selector.main(
        live=False,
        previous_gw=previous_gw,
        season=season,
        save_selection=False,
        previous_team_selection_path=previous_team_selection_path,
        budget=budget,
        available_chips=available_chips,
        available_transfers=available_transfers
    )

    return jsonify(output_dict)


if __name__ == '__main__':
    # need debug and threaded parameters to prevent TensorFlow error
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
