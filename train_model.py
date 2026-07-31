import numpy as np
import pandas as pd
import scipy.optimize


def log_loss(params: list, df: pd.DataFrame) -> float:
    x1, x2, y1, y2 = params
    t = np.clip(df['time_rem'].values, 1e-3, None)
    poss_feat = np.where(
        df['poss'].values == 1,
        (df['ortg'].values + df['drtg_agst'].values) / 200,
        -(df['ortg_agst'].values + df['drtg'].values) / 200,
    )
    adj_spread = (df['pace'].values + df['pace_agst'].values) / 2 * df['spread'].values / 100
    z = x1 * (df['lead'].values + poss_feat) / t ** y1 + x2 * (t / 40) * adj_spread / t ** y2
    p = np.clip(1 / (1 + np.exp(-z)), 1e-15, 1 - 1e-15)
    y = df['result'].values
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_model(df: pd.DataFrame):
    result = scipy.optimize.minimize(
        log_loss,
        x0=[1.0, 1.0, 0.5, 0.0],
        args=(df,),
        method='L-BFGS-B',
    )
    return result

