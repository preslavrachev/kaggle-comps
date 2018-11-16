import numpy as np
import os
import time
import warnings
import pandas as pd
from functools import lru_cache
from kaggle.competitions import twosigmanews
# from pandasta import indicators as pdt
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def stochk(df, period):
    min_price = df['m_cap_low'].rolling(period).min()
    max_price = df['m_cap_high'].rolling(period).max()
    return (df['m_cap_close'] - min_price) / (max_price - min_price)


def augment_df(df, set_labels=True):
    df['m_cap_open'] = df['volume'] * df['close']
    df['m_cap_close'] = df['volume'] * df['close']
    df['low'] = df[['open', 'close']].min(axis=1)
    df['high'] = df[['open', 'close']].max(axis=1)
    df['m_cap_low'] = df[['m_cap_open', 'm_cap_close']].min(axis=1)
    df['m_cap_high'] = df[['m_cap_open', 'm_cap_close']].max(axis=1)

    # df = pdt.TaDataFrame(df.reset_index(), indicators=['stochk_14', 'stochk_30', 'atr_30', 'vol_14', 'vol_30'])
    df['stochk_14'] = stochk(df, 14)
    df['stochk_30'] = stochk(df, 30)
    df['vol_14'] = df['volume'] / df['volume'].rolling(14).max()
    df['vol_30'] = df['volume'] / df['volume'].rolling(30).max()
    # df['atr_30'] = df['atr_30'] / df['atr_30'].rolling(30).max()

    if set_labels:
        df.loc[df['returnsOpenNextMktres10'] >= 0, 'label'] = 1
        df.loc[df['returnsOpenNextMktres10'] <= 0, 'label'] = -1

    return df


def build_prediction_model(market_df, features):
    tree = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=2, min_samples_leaf=100)

    # print('Fitting the tree...')
    tree.fit(market_df.dropna()[features], market_df.dropna()[['label']])
    # print('Done fitting the tree...')
    return tree


def make_random_predictions(market_obs_df, estimator, predictions_df=None, calc_score=True):
    x = market_obs_df[features]
    if calc_score:
        print(estimator.score(x, market_obs_df[['label']]))
    calc = estimator.predict_proba(x).max(axis=1) * estimator.predict(x)
    if predictions_df is not None:
        predictions_df['confidenceValue'] = calc
    return calc


if __name__ == '__main__':
    env = twosigmanews.make_env()
    all_market_train_df, all_news_train_df = env.get_training_data()
    warnings.filterwarnings('ignore')

    features = ['returnsClosePrevMktres1', 'returnsClosePrevMktres10', 'stochk_14', 'stochk_30', 'vol_14', 'vol_30']

    df = all_market_train_df.groupby(['time']).mean()
    df = augment_df(df)
    merged_train_df = pd.merge(all_market_train_df,
                               df.reset_index()[['stochk_14', 'stochk_30', 'vol_14', 'vol_30', 'time']], on=['time'])
    merged_train_df.loc[merged_train_df['returnsOpenNextMktres10'] >= 0, 'label'] = 1
    merged_train_df.loc[merged_train_df['returnsOpenNextMktres10'] <= 0, 'label'] = -1

    estimator = build_prediction_model(merged_train_df.tail(10000), features=features)

    days = env.get_prediction_days()
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        sample_time = market_obs_df['time'].tail(1)
        print(sample_time)
        all_market_train_df = all_market_train_df.append(market_obs_df)
        df = all_market_train_df.groupby(['time']).mean()
        df = augment_df(df)
        merged_train_df = pd.merge(all_market_train_df,
                                   df.reset_index()[['stochk_14', 'stochk_30', 'vol_14', 'vol_30', 'time']],
                                   on=['time'])
        merged_train_df.loc[merged_train_df['returnsOpenNextMktres10'] >= 0, 'label'] = 1
        merged_train_df.loc[merged_train_df['returnsOpenNextMktres10'] <= 0, 'label'] = -1

        x = merged_train_df[features].tail(market_obs_df['assetCode'].count()).fillna(0)
        score = ((estimator.predict_proba(x).max(axis=1) - 0.5) * estimator.predict(x))
        # print(score)
        predictions_template_df['confidenceValue'] = score * -1
        env.predict(predictions_template_df)

    env.write_submission_file()
