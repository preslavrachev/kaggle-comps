import numpy as np
import os
import time
import warnings
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import train_test_split

from kaggle.competitions import twosigmanews
# from pandasta import indicators as pdt
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def stochk(df, period):
    min_price = df['m_cap_low'].rolling(period).min()
    max_price = df['m_cap_high'].rolling(period).max()
    return (df['m_cap_close'] - min_price) / (max_price - min_price)


def asset_stochk(df, period):
    rolling = df.groupby(['assetCode']).rolling(period)
    min_price = rolling.agg({'close': 'min'}).reset_index().set_index('level_1')['close']
    max_price = rolling.agg({'close': 'max'}).reset_index().set_index('level_1')['close']

    return ((df['close'] - min_price) / (max_price - min_price)).round(decimals=2)


def asset_stochka(df, period):
    rolling = df.groupby(['assetCode']).rolling(period)
    mean_price = rolling.agg({'close': 'mean'}).reset_index().set_index('level_1')['close']
    min_price = rolling.agg({'close': 'min'}).reset_index().set_index('level_1')['close']
    max_price = rolling.agg({'close': 'max'}).reset_index().set_index('level_1')['close']

    return ((mean_price - min_price) / (max_price - min_price)).round(decimals=2)


def attach_market_level_indicators(df):
    market_agg_df = df.groupby(['time']).mean()
    market_agg_df['m_cap_open'] = market_agg_df['volume'] * market_agg_df['close']
    market_agg_df['m_cap_close'] = market_agg_df['volume'] * market_agg_df['close']
    market_agg_df['low'] = market_agg_df[['open', 'close']].min(axis=1)
    market_agg_df['high'] = market_agg_df[['open', 'close']].max(axis=1)
    market_agg_df['m_cap_low'] = market_agg_df[['m_cap_open', 'm_cap_close']].min(axis=1)
    market_agg_df['m_cap_high'] = market_agg_df[['m_cap_open', 'm_cap_close']].max(axis=1)

    # df = pdt.TaDataFrame(df.reset_index(), indicators=['stochk_14', 'stochk_30', 'atr_30', 'vol_14', 'vol_30'])
    market_agg_df['m_stochk_14'] = stochk(market_agg_df, 14)
    market_agg_df['m_stochk_30'] = stochk(market_agg_df, 30)
    market_agg_df['vol_14'] = market_agg_df['volume'] / market_agg_df['volume'].rolling(14).max()
    market_agg_df['vol_30'] = market_agg_df['volume'] / market_agg_df['volume'].rolling(30).max()
    # df['atr_30'] = df['atr_30'] / df['atr_30'].rolling(30).max()

    return pd.merge(df,
                    market_agg_df.reset_index()[['m_stochk_14', 'm_stochk_30', 'vol_14', 'vol_30', 'time']],
                    on=['time'])


def attach_asset_level_indicators(df):
    # df['stochk_7'] = asset_stochk(df, 7)
    df['stochk_2_365'] = asset_stochk(df, 2 * 365)
    df['stochka_14'] = asset_stochka(df, 14)
    # df['stochk_30'] = asset_stochk(df, 30)
    return df


def add_labels(df):
    df.loc[df['returnsOpenNextMktres10'] >= 0, 'label'] = 1
    df.loc[df['returnsOpenNextMktres10'] <= 0, 'label'] = -1

    return df


def build_prediction_model(market_df, features):
    tree = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=2, min_samples_leaf=100)

    print('Fitting the tree...')
    tree.fit(market_df.dropna()[features], market_df.dropna()[['label']])
    print('Feature Importances: {}'.format(tree.feature_importances_))
    print('Done fitting the tree...')
    return tree


def make_random_predictions(market_obs_df, estimator, predictions_df=None, calc_score=True):
    print("Predicting the confidence values...")
    x = market_obs_df[features].fillna(0)
    if calc_score:
        print(estimator.score(x, market_obs_df[['label']].dropna()))
    calc = (estimator.predict_proba(x) * 2)[:, 1] - 1
    if predictions_df is not None:
        predictions_df['confidenceValue'] = calc
    return calc


def fetch_test_data_at_once():
    print("Fetching the test data...")
    market_obs_df = None
    predictions_template_df = None

    for (m_df, n_df, p_df) in env.get_prediction_days():
        env.predict(p_df)
        p_df['time'] = m_df.time.min()
        if market_obs_df is None:
            market_obs_df = m_df
            predictions_template_df = p_df
        else:
            market_obs_df = market_obs_df.append(m_df, ignore_index=True)
            predictions_template_df = predictions_template_df.append(p_df, ignore_index=True)

    print("Done, fetching the test data...")
    return market_obs_df, predictions_template_df


if __name__ == '__main__':
    env = twosigmanews.make_env()
    all_market_train_df, all_news_train_df = env.get_training_data()
    warnings.filterwarnings('ignore')

    all_market_train_df = attach_market_level_indicators(all_market_train_df)
    all_market_train_df = attach_asset_level_indicators(all_market_train_df)
    all_market_train_df = add_labels(all_market_train_df)

    features = ['returnsClosePrevMktres10', 'stochk_2_365', 'stochka_14', 'vol_30']
    train_df, test_df = train_test_split(all_market_train_df[features], all_market_train_df['label'],
                                         test_size=0.1,
                                         shuffle=False)
    predictions_template_df = test_df[['time', 'assetCode']]

    estimator = build_prediction_model(all_market_train_df.tail(100000), features=features)

    #test_df, predictions_template_df = fetch_test_data_at_once()
    test_df = attach_market_level_indicators(test_df)
    test_df = attach_asset_level_indicators(test_df)
    test_df['returnsOpenNextMktres10'] = test_df.groupby(['assetCode'])['returnsOpenPrevMktres10'].shift(-11).fillna(0)
    test_df = add_labels(test_df)

    make_random_predictions(test_df, estimator, predictions_template_df)

    # Score estimation
    test_df['returns'] = predictions_template_df['confidenceValue'] * test_df['returnsOpenNextMktres10']
    day_returns = test_df.groupby('time')['returns'].sum()
    print('LB:', day_returns.mean() / day_returns.std())
    # 0.28056 is the result after Kaggle submission

    predictions_template_df['time'] = predictions_template_df['time'].dt.date
    predictions_template_df[['time', 'assetCode', 'confidenceValue']].to_csv('submission.csv', index=False,
                                                                             float_format='%.8f')
