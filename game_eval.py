import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ingame_winprob import compute_live_win_prob, compute_bracket_igwp, compute_log_prob

# Function to evaluate a single game and compute win probabilities for each possession
def game_eval(possessions_merged: pd.DataFrame, game_id: int) -> pd.DataFrame:
    game = possessions_merged[possessions_merged['game_id'] == game_id].sort_values('poss_num').reset_index(drop=True)

    if game.empty:
        raise ValueError(f"No data found for game_id {game_id}")

    # Compute CBB analytics first iteration win probability
    game['win_prob'] = game.apply(lambda row: compute_live_win_prob(
        current_lead=row['lead'],
        mins_remaining=max(0, row['time_rem']),
        net_rtg_spread=row['spread'],
        possession=1 if row['poss'] == 1 else -1,
    ), axis=1)

    # Compute CBB analytics bracket win probability
    game['bracket_prob'] = game.apply(lambda row: compute_bracket_igwp(
        current_lead=row['lead'],
        mins_remaining=max(0, row['time_rem']),
        net_rtg_spread=row['spread'],
        possession=1 if row['poss'] == 1 else -1,
    ), axis=1)

    # Compute trained model log probability
    game['log_prob'] = game.apply(lambda row: compute_log_prob(
        current_lead=row['lead'],
        mins_remaining=max(0, row['time_rem']),
        spread=row['spread'],
        poss=row['poss'],
        ortg=row['ortg'],
        ortg_agst=row['ortg_agst'],
        drtg=row['drtg'],
        drtg_agst=row['drtg_agst'],
        pace=row['pace'],
        pace_agst=row['pace_agst'],
    ), axis=1)

    return (
        game[[
            'game_id', 'game_date', 'team_id', 'team_id_agst', 'hex_color2', 'hex_color2_agst',
            'team_market', 'team_name', 'team_market_agst', 'team_name_agst',
            'poss_num', 'poss', 'time_rem', 'period', 'ortg', 'ortg_agst', 'drtg', 'drtg_agst', 'pace', 'pace_agst',
            'favorite_score', 'underdog_score', 'spread', 'lead', 'win_prob', 'bracket_prob', 'log_prob', 'result'
        ]]
    )
