from flask import Flask, request, jsonify

from src.fpl_team_selector import team_selector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/api', methods=['GET'])
def api():
    content = request.get_json()

    fpl_team_id = int(content['fpl_team_id'])
    fpl_email = content['fpl_email']
    fpl_password = content['fpl_password']

    output_dict = team_selector.main(fpl_team_id, fpl_email, fpl_password)

    return jsonify(output_dict)


if __name__ == '__main__':
    # need debug and threaded parameters to prevent TensorFlow error
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
