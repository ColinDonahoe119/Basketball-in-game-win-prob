import pandas as pd
import numpy as np
import re

def camel_to_snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

def data_prep(possessions: pd.DataFrame, teams: pd.DataFrame, team_date_stats: pd.DataFrame) -> pd.DataFrame:
    # Ensure the dataframe is sorted by gameId and possession number for correct consecutive differences
    possessions = possessions.sort_values(['gameId', 'possNum']).reset_index(drop=True)

    df = possessions.merge(
       team_date_stats[['teamId', 'gameDate', 'netRtg', 'ortg', 'drtg', 'pace']],
        on=['teamId', 'gameDate'],
        how='left'
    ) 

    df = df.merge(
        team_date_stats[['teamId', 'gameDate', 'netRtg', 'ortg', 'drtg', 'pace']].rename(columns={
            'teamId': 'teamIdAgst',
            'netRtg': 'netRtgAgst',
            'ortg':   'ortgAgst',
            'drtg':   'drtgAgst',
            'pace':   'paceAgst'
        }),
        on=['teamIdAgst', 'gameDate'],
        how='left'
    )

    # Merge possession data with team names and markets
    df = df.merge(
        teams[['teamId', 'competitionId', 'teamMarket', 'teamName', 'hexColor2']],
        on=['teamId','competitionId'],
        how='left'
    )
    # Repeat for opposing team
    df = df.merge(
        teams[['teamId', 'competitionId', 'teamMarket', 'teamName','hexColor2']].rename(columns={
            'teamId':    'teamIdAgst',
            'teamMarket':'teamMarketAgst',
            'teamName':  'teamNameAgst',
            'hexColor2': 'hexColor2Agst'
        }),
        on=['teamIdAgst','competitionId'],
        how='left'
    )
    df.columns = [camel_to_snake(col) for col in df.columns]

    # Eliminate teams with no netRtg(first game of the season), and keep only columns necessary for the model
    df = df.dropna(subset=['net_rtg', 'net_rtg_agst'])
    df = df[['competition_id','game_id','game_date', 'poss_num', 'team_id', 'team_id_agst',
    'secs_in_end', 'period', 'score_end','net_rtg', 'net_rtg_agst', 'ortg', 'ortg_agst', 
    'drtg', 'drtg_agst', 'pace', 'pace_agst','team_market', 'team_name', 'team_market_agst', 
    'team_name_agst','hex_color2', 'hex_color2_agst'
    ]]

    df = df.sort_values(['game_id', 'poss_num']).reset_index(drop=True)
    # Convert the seconds in the game to time remaining, and adjust for overtime situations
    df['time_rem'] = (40 - df['secs_in_end'] / 60).round(2)
    df['time_rem'] = np.where(df['period'] == 'ot1', df['time_rem'] + 5, df['time_rem'])
    df['time_rem'] = np.where(df['period'] == 'ot2', df['time_rem'] + 10, df['time_rem'])
    df['time_rem'] = np.where(df['period'] == 'ot3', df['time_rem'] + 15, df['time_rem'])
    df['time_rem'] = np.where(df['period'] == 'ot4', df['time_rem'] + 20, df['time_rem'])
    df['time_rem'] = np.where(df['period'] == 'ot5', df['time_rem'] + 25, df['time_rem'])

    df['original_poss_team_id'] = df['team_id']

    # Identify rows where team_id_agst is the favorite (i.e., its net_rtg is higher)
    swap_mask = df['net_rtg'] < df['net_rtg_agst']

    # List of column pairs to swap if team_id_agst is the favorite
    cols_to_swap = [('team_id', 'team_id_agst'),
    ('net_rtg', 'net_rtg_agst'),
    ('ortg', 'ortg_agst'),
    ('drtg', 'drtg_agst'),
    ('pace', 'pace_agst'),
    ('team_market', 'team_market_agst'),
    ('team_name', 'team_name_agst'),
    ('hex_color2', 'hex_color2_agst')
    ]

    # Perform the swap for identified rows
    for col1, col2 in cols_to_swap:
        temp_col = df.loc[swap_mask, col1].copy()
        df.loc[swap_mask, col1] = df.loc[swap_mask, col2]
        df.loc[swap_mask, col2] = temp_col

    # Create the 'poss' variable: 1 if the favorite team (now in team_id) has possession, 0 otherwise
    df['poss'] = np.where(df['original_poss_team_id'] == df['team_id'], 1, 0)

    # Drop the temporary original_poss_team_id column
    df = df.drop(columns=['original_poss_team_id'])

    df = df.sort_values(['game_id', 'poss_num']).reset_index(drop=True)

    #Parse scoreEnd ("possessor_score-defender_score") into favorite/underdog scores
    scores = df['score_end'].str.split('-', expand=True).astype(int)
    df['favorite_score'] = scores[0].where(df['poss'] == 1, scores[1])
    df['underdog_score']  = scores[1].where(df['poss'] == 1, scores[0])

    # Identify net rating spread, favorite's lead, and game winner
    df['spread'] = df['net_rtg'] - df['net_rtg_agst']
    df['lead']   = df['favorite_score'] - df['underdog_score']

    df['result'] = (
        df.groupby('game_id')['favorite_score'].transform('last') >
        df.groupby('game_id')['underdog_score'].transform('last')).astype(int)
    # Split the data into Training and Testing sets
    game_ids = df['game_id'].unique()
    np.random.seed(42)
    np.random.shuffle(game_ids)

    split = int(len(game_ids) * 0.8)
    train_ids = game_ids[:split]
    test_ids = game_ids[split:]

    train = df[df['game_id'].isin(train_ids)]
    test = df[df['game_id'].isin(test_ids)]

    # Save processed data
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    return train, test

