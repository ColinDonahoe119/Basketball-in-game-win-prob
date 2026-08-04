
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math

def get_logo(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

from typing_extensions import final
def win_prob_chart(igwp_data: pd.DataFrame, comebacks_df: pd.DataFrame, swings_df: pd.DataFrame, game_id: int, model: int = 1) -> None:
    game = igwp_data[igwp_data['game_id'] == game_id].sort_values('poss_num').reset_index(drop=True)
    game['team_id'] = game['team_id'].astype(int)
    game['team_id_agst'] = game['team_id_agst'].astype(int)
    base_logo_url = "https://storage.googleapis.com/cbb-image-files/team-logo-images/"
    if game.empty:
        raise ValueError(f"No data found for game_id {game_id}")

    prob_col_name = 'bracket_prob' if model == 0 else 'log_prob'
    prob_values = game[prob_col_name].copy() # Make a copy to avoid SettingWithCopyWarning

    if game['result'].iloc[0] == 1:
        winner_team_id = game['team_id'].iloc[0]
        winner_market = game['team_market'].iloc[0]
        winner_name = game['team_name'].iloc[0]
        winner_logo_url = base_logo_url + str(winner_team_id) + ".webp"
        winner_logo = get_logo(winner_logo_url)
        winner_color = game['hex_color2'].iloc[0]
        final_winner_score = game['favorite_score'].iloc[-1]
        final_loser_score = game['underdog_score'].iloc[-1]

        loser_team_id = game['team_id_agst'].iloc[0]
        loser_market = game['team_market_agst'].iloc[0]
        loser_name = game['team_name_agst'].iloc[0]
        loser_logo_url = base_logo_url + str(loser_team_id) + ".webp"
        loser_logo = get_logo(loser_logo_url)
        loser_color = game['hex_color2_agst'].iloc[0]
    else:
        winner_team_id = game['team_id_agst'].iloc[0]
        winner_market = game['team_market_agst'].iloc[0]
        winner_name = game['team_name_agst'].iloc[0]
        winner_logo_url = base_logo_url + str(winner_team_id) + ".webp"
        winner_logo = get_logo(winner_logo_url)
        winner_color = game['hex_color_2_agst'].iloc[0]
        final_winner_score = game['underdog_score'].iloc[-1]
        final_loser_score = game['favorite_score'].iloc[-1]

        loser_team_id = game['team_id'].iloc[0]
        loser_market = game['team_market'].iloc[0]
        loser_name = game['team_name'].iloc[0]
        loser_logo_url = base_logo_url + str(loser_team_id) + ".webp"
        loser_logo = get_logo(loser_logo_url)
        prob_values = 1 - prob_values # Invert the prob_values Series
        loser_color = game['hex_color_2'].iloc[0]

    # Initialize OffsetImage objects after winner_logo and loser_logo are defined
    win_image = OffsetImage(winner_logo, zoom=0.25)
    lose_image = OffsetImage(loser_logo, zoom=0.25)

    # Check if there are overtime periods
    has_ot = (game['period'].str.contains('ot', case=False, na=False)).any()

    # Calculate 'minutes into game'
    # For H1/H2, minutes_into_game = 40 - time_rem
    game['minutes_into_game'] = game['time_rem'].apply(lambda x: 40 - x) # Default for regulation

    # Handle overtime periods: ot1, ot2, ot3, etc.
    ot_mask = game['period'].str.contains('ot', case=False, na=False)
    if ot_mask.any():
        # Extract overtime period number, e.g., 'ot1' -> 1, 'ot2' -> 2, 'ot3' -> 3
        # If 'ot' without a number, assume it's 'ot1'
        ot_period_numbers = game['period'][ot_mask].str.extract(r'ot(\d*)', expand=False).fillna('1').astype(int)

        # Calculate minutes into game for overtime periods
        # Each OT period is 5 minutes. OT1 starts at 40 mins, OT2 at 45, OT3 at 50, etc.
        # The `time_rem` for OT counts down from 5 to 0.
        # Formula: 40 + (OT_number - 1) * 5 + (5 - time_rem)
        game.loc[ot_mask, 'minutes_into_game'] = 40 + (ot_period_numbers - 1) * 5 + (5 - game.loc[ot_mask, 'time_rem'])

    fig, ax = plt.subplots(figsize=(14, 8)) # Increased height for logos/names
    fig.subplots_adjust(top=0.85, bottom=0.15)

    # Add faded team color area
    ax.fill_between(game['minutes_into_game'], prob_values, 0.5, where=prob_values >= 0.5,color=winner_color, alpha=0.15,interpolate=True
)

    ax.fill_between(game['minutes_into_game'], prob_values, 0.5, where=prob_values < 0.5, color=loser_color, alpha=0.15, interpolate=True
)

    # Plot the win probability line with dynamic colors
    for i in range(len(game) - 1):
      x1, y1 = game['minutes_into_game'].iloc[i], prob_values.iloc[i]
      x2, y2 = game['minutes_into_game'].iloc[i+1], prob_values.iloc[i+1]

      # No crossing
      if (y1 >= 0.5 and y2 >= 0.5):
        ax.plot([x1, x2], [y1, y2], color=winner_color, linewidth=2)

      elif (y1 < 0.5 and y2 < 0.5):
        ax.plot([x1, x2], [y1, y2], color=loser_color, linewidth=2)

      # Crossing 50%
      else:
        t = (0.5 - y1) / (y2 - y1)
        x_cross = x1 + t * (x2 - x1)

        if y1 < y2:
          ax.plot([x1, x_cross], [y1, 0.5],
            color=loser_color, linewidth=2)
          ax.plot([x_cross, x2], [0.5, y2],
            color=winner_color, linewidth=2)
        else:
          ax.plot([x1, x_cross], [y1, 0.5],
            color=winner_color, linewidth=2)
          ax.plot([x_cross, x2], [0.5, y2],
            color=loser_color, linewidth=2)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # X-axis configuration: Minutes into game
    max_minutes_into_game = game['minutes_into_game'].max()
    x_ticks_values = list(range(0, 41, 10))
    if has_ot:
        # Add ticks for OT if applicable, for every 5 minutes in OT
        ot_start_time = 40
        # num_ot_periods will now correctly reflect actual number of OT periods based on max_minutes_into_game
        num_ot_periods = math.ceil((max_minutes_into_game - ot_start_time) / 5)
        for i in range(1, num_ot_periods + 1):
            x_ticks_values.append(ot_start_time + 5 * i)

    ax.set_xlim(0, max_minutes_into_game + 1) # +1 to give a little padding at the end
    ax.set_xticks(x_ticks_values)
    ax.set_xlabel('Minutes Into Game')

    # Y-axis configuration: 0%, 50%, 100%
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0%', '50%', '100%'])
    ax.set_ylabel('Win Probability')

    # Grid lines only at axis markers
    ax.grid(True, which='major', axis='both', linestyle=':', alpha=0.4)
    ax.minorticks_off()

    # Halftime marker and score
    halftime_time_into_game = 20 # 20 minutes into the game
    halftime_data = game[game['minutes_into_game'] <= halftime_time_into_game].iloc[-1]
    if not halftime_data.empty:
        ax.axvline(halftime_time_into_game, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        halftime_winner_score = halftime_data['favorite_score'] if halftime_data['result'] == 1 else halftime_data['underdog_score']
        halftime_loser_score = halftime_data['underdog_score'] if halftime_data['result'] == 1 else halftime_data['favorite_score']

        ax.annotate(f"Halftime: {int(halftime_winner_score)}-{int(halftime_loser_score)}", xy=(halftime_time_into_game, 1.0),
                    xycoords=('data', 'axes fraction'),xytext=(5, 5), textcoords='offset points', ha='center',va='bottom',fontsize=14,color='black'
)
    if has_ot:
        end_reg = 40
        end_reg_data = game[game['minutes_into_game'] <= end_reg].iloc[-1]
        if not end_reg_data.empty:
            ax.axvline(end_reg, color='gray', linestyle='--', linewidth=1, alpha=0.8)
            end_reg_winner_score = end_reg_data['favorite_score'] if end_reg_data['result'] == 1 else end_reg_data['underdog_score']
            end_reg_loser_score = end_reg_data['underdog_score'] if end_reg_data['result'] == 1 else end_reg_data['favorite_score']
            ax.annotate(f"End of Regulation: {int(end_reg_winner_score)}-{int(end_reg_loser_score)}", xy=(end_reg, 1.0),
                        xycoords=('data', 'axes fraction'),xytext=(5, 5), textcoords='offset points', ha='center',va='bottom',fontsize=14,color='black'
)
    # Lowest WP marker (from previous code, adjusted x-coordinate)
    current_game_id = game['game_id'].iloc[0]
    comeback_data = comebacks_df[comebacks_df['game_id'] == current_game_id]
    if not comeback_data.empty:
        comeback_poss_num = comeback_data['poss_num'].iloc[0]
        comeback_game_row = game[game['poss_num'] == comeback_poss_num]
        if not comeback_game_row.empty:
            x_annotate = comeback_game_row['minutes_into_game'].iloc[0] # Use new minutes_into_game
            y_annotate = prob_values.loc[comeback_game_row.index].iloc[0]
            ax.scatter(
                x_annotate,
                y_annotate,
                s=80,
                color='gold',
                edgecolor='black',
                zorder=5
            )
            ax.annotate(
                f"{comeback_data['team_name'].iloc[0]} Lowest WP: {y_annotate * 100:.1f}%, {int(comeback_data['favorite_score'].iloc[0])}-{int(comeback_data['underdog_score'].iloc[0])}",
                (x_annotate, y_annotate),
                xytext=(-10, -15), # Adjusted xytext for better visibility
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->")
            )
    swing_data = swings_df[swings_df['game_id'] == current_game_id]
    if not swing_data.empty:
        swing_poss_num = swing_data['poss_num'].iloc[0]
        swing_game_row = game[game['poss_num'] == swing_poss_num]
        if not swing_game_row.empty:
            x_annotate = swing_game_row['minutes_into_game'].iloc[0] # Use new minutes_into_game
            y_annotate = prob_values.loc[swing_game_row.index].iloc[0]
            ax.scatter(
                x_annotate,
                y_annotate,
                s=80,
                color='gold',
                edgecolor='black',
                zorder=5
            )
            ax.annotate(
                f"{comeback_data['team_name'].iloc[0]} Lowest WP: {y_annotate * 100:.1f}%, {int(comeback_data['favorite_score'].iloc[0])}-{int(comeback_data['underdog_score'].iloc[0])}",
                (x_annotate, y_annotate),
                xytext=(10, 15), # Adjusted xytext for better visibility
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->")
            )

    # Main Title
    #ax.set_title(f"In Game Win Probability - {winner_name} {int(final_winner_score)}-{int(final_loser_score)}",
                 #fontsize=16, pad=30) # Increased pad for main title
    fig.suptitle(f"In Game Win Probability - {winner_name} vs {loser_name} {int(final_winner_score)}-{int(final_loser_score)}", fontsize=22)
    fig.subplots_adjust(top=0.82)
    ax.annotate(game['game_date'].iloc[1], xy=(40, 1),
                    xycoords=('data', 'axes fraction'),xytext=(5, 5), textcoords='offset points', ha='right',va='bottom',fontsize=14,color='black'
)

    # Team logos and names outside the main plot area
    # Winner above
    ab_winner = AnnotationBbox(win_image,(0.02, 1.03), xycoords=ax.transAxes, frameon=False, box_alignment=(0, 0.5)
)
    ax.add_artist(ab_winner)

    ax.text(0.08, 1.03, f"{winner_market} {winner_name}", transform=ax.transAxes, va="center", ha="left", fontsize=16, weight="bold"
)

    # Loser below
    ab_loser = AnnotationBbox(lose_image,(0.02, -0.08), xycoords=ax.transAxes, frameon=False, box_alignment=(0, 0.5)
)
    ax.add_artist(ab_loser)

    ax.text(0.08, -0.08, f"{loser_market} {loser_name}", transform=ax.transAxes, va="center", ha="left", fontsize=16, weight="bold"
)

    plt.tight_layout() # Removed rect to rely on default auto-adjustment
    plt.show()

