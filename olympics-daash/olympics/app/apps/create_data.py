import numpy as np
import math
import copy
import re
import pandas as pd

def _create_data():
    oly_df = pd.read_csv('../data/athlete_events.csv', index_col='ID')
    columns=['Name', 'Sex', 'Age','Height', 'Weight', 'Season', 'Sport', 'Event']
    oly_df = oly_df[columns + ['Medal']]
    oly_df = oly_df.dropna(subset=columns)

    oly_df['Weight_met'] = oly_df['Weight']
    oly_df['Height_met'] = oly_df['Height']

    oly_df['Weight_emp'] = oly_df['Weight_met'] * 2.2046226218
    oly_df['Height_emp'] = oly_df['Height_met'] * 0.393701


    all_sport_groupdf = oly_df.groupby('Sport').agg(['count', 'std', 'mean'])
    winner_sport_groupdf = oly_df[oly_df['Medal'].notna()].groupby('Sport').agg(['count', 'mean', 'std'])

    all_sport_groupdf = oly_df.groupby('Sport').agg(['count', 'std', 'mean'])
    all_sport_groupdf.columns = [i[0].lower() + '_all_' +  i[1] for i in all_sport_groupdf.columns]
    all_sport_groupdf = all_sport_groupdf.drop(['height_met_all_count', 'height_emp_all_count', 'weight_emp_all_count', 'weight_met_all_count'], axis=1).rename({'age_all_count' : 'all_count'}, axis=1)


    winner_sport_groupdf = oly_df[oly_df['Medal'].notna()].groupby('Sport').agg(['count', 'mean', 'std'])
    winner_sport_groupdf.columns = [i[0].lower() + '_winner_' +  i[1] for i in winner_sport_groupdf.columns]

    sport_df = pd.merge(all_sport_groupdf, winner_sport_groupdf, left_index=True, right_index=True)
    sport_df = sport_df.drop(['height_met_winner_count', 'height_emp_winner_count', 'weight_emp_winner_count', 'weight_met_winner_count'], axis=1).rename({'age_winner_count' : 'winner_count'}, axis=1)
    sport_df = sport_df[sport_df['all_count'] > 50]
    sport_df = sport_df.replace(np.nan, 0)
    oly_df = oly_df.replace(np.nan, 0)#.reset_index()

    total_medals = np.sum(sport_df['winner_count'])
    sport_df['pct_athletes_winners'] = sport_df['winner_count'] / sport_df['all_count']
    sport_df['pct_total_medals'] = sport_df['winner_count'] / total_medals
    sport_df['weight_winner_z_score'] = 0
    sport_df['age_winner_z_score'] = 0
    sport_df['height_winner_z_score'] = 0

    sport_df.loc['Average Olympian', [f'height_emp_winner_mean', f'weight_emp_winner_mean']] = [oly_df[f'Height_emp'].mean(), oly_df[f'Weight_emp'].mean()]
    sport_df.loc['Average Olympian', [f'height_met_winner_mean', f'weight_met_winner_mean']] = [oly_df[f'Height_met'].mean(), oly_df[f'Weight_met'].mean()]
    sport_df.loc['Average Olympian', 'age_winner_mean'] = oly_df[f'Age'].mean()

    sport_df.loc['Average Olympian', 'pct_athletes_winners'] = 2
    sport_df.loc['Person', :] = 0
    sport_df.loc['Person', 'pct_athletes_winners'] = 2

    return sport_df, oly_df
