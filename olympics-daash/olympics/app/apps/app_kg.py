from apps.app_func import _create_graphs
import pandas as pd

sport_df = pd.read_csv('../data/sport_df_proc.csv', index_col='Sport')
oly_df = pd.read_csv('../data/oly_df_proc.csv')

layout = _create_graphs(lbs_flag = False)
