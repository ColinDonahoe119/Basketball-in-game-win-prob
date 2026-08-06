import pandas as pd
import numpy as np

def comebacks(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure the dataframe is sorted by gameId and possession number for consistent results
    df_sorted = df.sort_values(by=['game_id', 'poss_num']).copy()

    # Calculate the probability of the actual winning team for each possession.
    # This is an intermediate step to find the lowest point of the winning team's probability.
    # If 'result' is 1 (favorite won), 'logProb' is the probability of the winning team.
    # If 'result' is 0 (favorite lost), '1 - logProb' is the probability of the winning team (underdog).
    df_sorted['actual_winner_prob_at_possession'] = np.where(
        df_sorted['result'] == 1, df_sorted['log_prob'], 1 - df_sorted['log_prob']
    )

    # Initialize a list to hold the selected rows (one per game)
    final_rows = []

    # Iterate through each game
    for game_id, game_group in df_sorted.groupby('game_id'):
        # Determine the lowest win probability for the actual winning team across the entire game
        lowest_game_prob = game_group['actual_winner_prob_at_possession'].min()

        # Find the row in the game group where this lowest probability occurred.
        # If multiple possessions have the same lowest probability, take the first one (based on possNum sort).
        comeback_moment_row = game_group[
            game_group['actual_winner_prob_at_possession'] == lowest_game_prob
        ].iloc[0].copy()

        # Add the 'lowest_prob' (as requested in the prompt) to this specific row.
        # This is the lowest probability the actual winning team had during the game.
        comeback_moment_row['lowest_prob'] = lowest_game_prob

        # Append this modified row to our list
        final_rows.append(comeback_moment_row)

    # Concatenate all collected rows into a new DataFrame
    comebacks_df = pd.DataFrame(final_rows)

    # Drop the temporary column used for calculation
    comebacks_df = comebacks_df.drop(columns=['actual_winner_prob_at_possession'])
    #comebacks_df = comebacks_df[comebacks_df['lowest_prob'] > 0]

    return comebacks_df


def swings(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure the dataframe is sorted by gameId and possession number for correct consecutive differences
    df_sorted = df.sort_values(by=['game_id', 'poss_num']).copy()

    # Calculate the absolute difference in 'logProb' between consecutive possessions within each game
    df_sorted['prob_swing'] = df_sorted.groupby('game_id')['log_prob'].diff().abs()

    # Initialize a list to hold the selected rows (one per game)
    final_rows = []

    # Iterate through each game
    for game_id, game_group in df_sorted.groupby('game_id'):
        # Find the maximum 'prob_swing' for the current game
        max_game_swing = game_group['prob_swing'].max()

        # If max_game_swing is NaN (e.g., game has only one possession), skip or handle as appropriate
        if pd.isna(max_game_swing):
            continue

        # Find the row(s) in the game group where this maximum swing occurred.
        # We take the first one if multiple possessions have the same maximum swing.
        swing_moment_row = game_group[
            game_group['prob_swing'] == max_game_swing
        ].iloc[0].copy()

        # Add the 'biggest_swing' to this specific row.
        swing_moment_row['biggest_swing'] = max_game_swing

        # Append this modified row to our list
        final_rows.append(swing_moment_row)

    # Concatenate all collected rows into a new DataFrame
    swings_df = pd.DataFrame(final_rows)

    # Drop the temporary column used for calculation ('prob_swing' is now 'biggest_swing')
    swings_df = swings_df.drop(columns=['prob_swing'])
    #swings_df = swings_df[swings_df['biggest_swing'] < 1]

    return swings_df
