import math

_X1, _X2, _Y1, _Y2 = 0.412912, 2.082151, 0.344548, 0.716343


def compute_log_prob(
    current_lead,
    mins_remaining,
    spread,       # fav_netRtg - und_netRtg (pre-normalization)
    poss,         # 1 if favorite has ball, 0 otherwise
    ortg, ortg_agst, drtg, drtg_agst,
    pace, pace_agst,
):
    if mins_remaining == 0:
        return 1 if current_lead > 0 else (0 if current_lead < 0 else 0.5)
    t = max(mins_remaining, 1e-3)
    poss_feat = (ortg + drtg_agst) / 200 if poss == 1 else -(ortg_agst + drtg) / 200
    adj_spread = (pace + pace_agst) / 2 * spread / 100
    z = _X1 * (current_lead + poss_feat) / t ** _Y1 + _X2 * (t / 40) * adj_spread / t ** _Y2
    return math.exp(z) / (1 + math.exp(z))


def compute_bracket_igwp(
    current_lead,       # L(t) : current lead (points) for the favored team at time t
    mins_remaining,     # t : time remaining in the game
    net_rtg_spread, # S : netRtgAdj (favorite) - netRtgAdj (underdog)
    possession        # default to unknown possession
):
    # constants
    GAME_LENGTH = 40
    # derive pointSpread from adjNetRtgSpread (S : pregame point spread (betting line))
    # original paper uses pointSpread from betting lines
    point_spread = net_rtg_spread * (54 / 100)

    # Smooth transition in constant COEFFICIENT values between 2 mins and 1 min
    transition_start = 2   # start transition at 2 mins
    transition_end = 0.5   # end transition at 30 seconds
    transition_factor = min(1, max(0,
        (mins_remaining - transition_end) / (transition_start - transition_end)
    ))

    # LEAD_COEFFICIENT: How much current lead matters - higher means leads are more stable
    # 0.69 reflects that college leads are relatively stable but can change,
    # Leads matter less in final minutes since 3-pointers can change game quickly
    # 0.40 is about half of base to reflect this increased volatility
    BASE_LEAD_COEFFICIENT = 0.69
    END_LEAD_COEFFICIENT = 0.40
    LEAD_COEFFICIENT = END_LEAD_COEFFICIENT + (BASE_LEAD_COEFFICIENT - END_LEAD_COEFFICIENT) * transition_factor

    # How much pre-game spread matters - lower means less predictive
    # 2.0 reflects that college games have more variance than NBA
    # Pre-game spread barely matters in final minute
    # 0.8 is just 40% of base since actual game situation dominates (but favorites still more likely to comeback)
    BASE_SPREAD_COEFFICIENT = 2.0
    END_SPREAD_COEFFICIENT = 0.8
    SPREAD_COEFFICIENT = END_SPREAD_COEFFICIENT + (BASE_SPREAD_COEFFICIENT - END_SPREAD_COEFFICIENT) * transition_factor

    # How quickly lead impact decays with time - higher means leads matter more later
    # 0.54 means leads become less meaningful fairly quickly as time remains
    # Lead impact decays very quickly in final minute
    # 0.35 means even big leads can evaporate fast at the end
    BASE_LEAD_EXPONENT = 0.54
    END_LEAD_EXPONENT = 0.35
    LEAD_EXPONENT = END_LEAD_EXPONENT + (BASE_LEAD_EXPONENT - END_LEAD_EXPONENT) * transition_factor

    # How quickly spread impact decays - higher means spreads matter more later
    # 0.8 means pre-game expectations remain somewhat relevant throughout
    # Spread becomes nearly irrelevant in final minute
    # 0.3 means pre-game expectations have little impact on endgame
    BASE_SPREAD_EXPONENT = 0.8
    END_SPREAD_EXPONENT = 0.3
    SPREAD_EXPONENT = END_SPREAD_EXPONENT + (BASE_SPREAD_EXPONENT - END_SPREAD_EXPONENT) * transition_factor

    # Input validation
    if mins_remaining < 0 or mins_remaining > GAME_LENGTH:
        raise ValueError('Minutes remaining must be between 0 and 40')

    # Handle edge case when time is 0
    if mins_remaining == 0:
        return 1 if current_lead > 0 else (0 if current_lead < 0 else 0.5)

    # Calculate possession adjustment
    possession_adjustment = possession * 0.5

    # Calculate the two components of zz
    lead_component = (LEAD_COEFFICIENT * (current_lead + possession_adjustment)) / (mins_remaining ** LEAD_EXPONENT)
    spread_component = (SPREAD_COEFFICIENT * (mins_remaining / GAME_LENGTH) * point_spread) / (mins_remaining ** SPREAD_EXPONENT)

    # Calculate zz
    zz = lead_component + spread_component

    # Calculate final probability using logistic function
    win_prob = math.exp(zz) / (1 + math.exp(zz))

    # Clamp probability between 0 and 1 to handle extreme values
    return max(0, min(1, win_prob))


def compute_live_win_prob(
    current_lead,       # L(t) : current lead (points) for the favored team at time t
    mins_remaining,     # t : time remaining in the game
    net_rtg_spread, # S : netRtgAdj (favorite) - netRtgAdj (underdog)
    # point_spread,     # S : pregame point spread (betting line) - (NC: we can use netRtgAdj for both teams to estimate this)
    possession=0        # default to unknown possession
):
    """
    Returns win probability (0-1).
    """
    # constants
    GAME_LENGTH = 40
    LEAD_COEFFICIENT = 0.69   # increase to 0.85? leads are more stable in college
    SPREAD_COEFFICIENT = 3.49 # decrease to 2.9? spreads less predictive
    LEAD_EXPONENT = 0.54      # decrease to 0.50? more variance
    SPREAD_EXPONENT = 0.80    # Decrease to 0.75? more variance

    # pointSpread from ratingsSpread (S : pregame point spread (betting line))
    point_spread = net_rtg_spread * (68 / 100)

    # Input validation
    if mins_remaining < 0 or mins_remaining > GAME_LENGTH:
        raise ValueError('Minutes remaining must be between 0 and 48')

    # Handle edge case when time is 0
    if mins_remaining == 0:
        return 1 if current_lead > 0 else (0 if current_lead < 0 else 0.5)

    # Calculate possession adjustment
    possession_adjustment = possession * 0.5

    # Calculate the two components of zz
    lead_component = (LEAD_COEFFICIENT * (current_lead + possession_adjustment)) / (mins_remaining ** LEAD_EXPONENT)
    spread_component = (SPREAD_COEFFICIENT * (mins_remaining / GAME_LENGTH) * point_spread) / (mins_remaining ** SPREAD_EXPONENT)

    # Calculate zz
    zz = lead_component + spread_component

    # Final probability sigmoid function
    win_prob = math.exp(zz) / (1 + math.exp(zz))

    # Clamp probability between 0 and 1 to handle extreme values
    return max(0, min(1, win_prob))
