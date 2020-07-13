# fpl-team-selector

API to assist with team selection for the official Fantasy Premier League game (https://fantasy.premierleague.com/). Analyses existing team and recommends transfers for upcoming gameweek given a set of player predictions.

## User guide

There are two endpoints, one for live selections `/api` and one for retrospective selections `/retro`

### Run on local machine
1. Start application using one of:
    ```bash
    export $(xargs <.env)
    gunicorn app.fpl_team_selector:app --bind 0.0.0.0:5000
    ```
    ```bash
    docker-compose up
    ```
2. Live team selections
    
    Need to specify:
    - `previous_gw` - Gameweek prior to the one you want to make a selection for
    - `season` - Premier league season e.g. 2019-20
    - `fpl_team_id` - FPL team ID
    - `fpl_email` - Email for FPL account
    - `fpl_password` - Password for FPL account
    
    Optional:
    - `player_overwrites` - Dictionary of player names and _total_ predictions to use for them
    - `team_prediction_scalars` - Dictionary of team names and scalar to apply to _total_ predictions for all players in that team e.g. if you are worried about Pep's squad rotation set to {"Manchester City": 0.8}
    
        __Current teams__:
        'Manchester City', 'Liverpool', 'Arsenal', 'Wolverhampton Wanderers', 'Everton', 'Aston Villa', 'Leicester City', 
        'Manchester United', 'Southampton', 'Tottenham Hotspur', 'Chelsea', 'Burnley', 'West Ham United', 'Crystal Palace', 
        'Sheffield United', 'Watford', 'Norwich City', 'Bournemouth', 'Brighton & Hove Albion', 'Newcastle United'
   
    
    ```bash
    curl -X GET "http://0.0.0.0:5000/api" -H "Content-Type: application/json" --data \
    '{"previous_gw":"{previous_gw}", "season":"{season}", "fpl_team_id":"{fpl_team_id}", "fpl_email":"{fpl_email}", "fpl_password":"{fpl_password}"}'
    ```
3. Retro team selections

    Need to specify:
    - `previous_gw` - Gameweek prior to the one you want to make a selection for
    - `season` - Premier league season e.g. 2019-20
    - `previous_team_selection_path` - S3 path containing previous team selection parquet file
    - `budget`- Available budget including money in the bank
    - `available_chips` - List of available FPL chips e.g. "wildcard"
    - `available_transfers` - Number of free transfers available
    
    ```bash
    curl -X GET "http://0.0.0.0:5000/retro" -H "Content-Type: application/json" --data \
    '{"previous_gw":"{previous_gw}", "season":"{season}", "previous_team_selection_path":"{previous_team_selection_path}", "budget":"{budget}", "available_chips":"{available_chips}", "available_transfers":"{available_transfers}"}'
    ```
