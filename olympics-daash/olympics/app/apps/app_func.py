from scipy import stats
import plotly.graph_objs as go
import scipy
import math
import copy
import re
import numpy as np
import dash_core_components as dcc
import dash_table
import dash
import dash_html_components as html
from sklearn.decomposition import PCA
import pandas as pd
from app import app
from apps.create_data import _create_data

def _create_graphs(lbs_flag):

    #sport_df, oly_df = _create_data()
    sport_df = pd.read_csv('../data/sport_df_proc.csv', index_col='Sport')
    oly_df = pd.read_csv('../data/oly_df_proc.csv')

    if lbs_flag == True:
        unit_str = 'emp'
    elif lbs_flag == False:
        unit_str = 'met'

    print(unit_str)
    graph1_id = f'graph_{unit_str}'
    graph2_id = f'graph_{unit_str}2'
    graph3_id = f'graph_{unit_str}3'
    table_id = f'table_{unit_str}'

    layout = html.Div([
        html.Div([
            dcc.Graph(
                   id=graph1_id,
                   figure={
                       'data': [
                           {
                           'x': sport_df[f'height_{unit_str}_winner_mean'],
                           'y': sport_df[f'weight_{unit_str}_winner_mean'],
                           'z': sport_df['age_winner_mean'],
                           'type': 'scatter3d',
                           'text': sport_df.index,
                           'mode': 'markers',
                           'colorbar' : dict(title="Colorbar"),
                           'marker': {'size': 15, 'color' : [i if i <= 1 else '#FF0000' for i in sport_df['pct_athletes_winners']], 'showscale' : "True",
                           'colorbar' : {'title' : {'text' : '% Winners'},
                                       'titleside' : 'top',
                                       'tickmode' : 'array',
                                       'tickvals' : [0.02,1, 2],
                                       'ticktext' : ['2%','100%', 'You!'],
                                       'ticks' : 'outside',
                                       'xpad' : '20px'},
                                   }
                           }
                       ],
                       'layout' : go.Layout(clickmode='event+select', showlegend=False,
                           title='Medalist Stats Per Sport', autosize=True,height=600,
                           scene=dict(
                               xaxis={'title' : {'text' : 'Height'}},
                                yaxis={'title': {'text' : 'Weight'}},
                                zaxis={'title': {'text' : 'Age'}}))
                                }
               ),
            dcc.Graph(
                   id=graph2_id,
                   figure={
                       'data': [
                           {
                           'x': sport_df['weight_winner_z_score'],
                           'y': sport_df['pct_total_medals'],
                           'type': 'Scatter',
                           'text': sport_df.index,
                           'mode': 'markers',
                           'showscale' : "False",
                           'marker': {'size': 10, 'color' : sport_df['pct_athletes_winners']}
                           }
                       ],
                       'layout' : go.Layout(clickmode='event+select', showlegend=False,
                           title='Difference per Sport', autosize=True,height=600,
                           scene=dict(
                               xaxis={'title' : {'text' : 'Height'}},
                                yaxis={'title': {'text' : 'Weight'}},
                                zaxis={'title': {'text' : 'Age'}}))
                                }
               ),
            dcc.Graph(
                   id=graph3_id,
                   figure={
                       'data': [
                           {
                           'x': '', #oly_df[f'Height_{unit_str}'],
                           'y': '', #oly_df[f'Weight_{unit_str}'],
                           'z': '', #oly_df['Age'],
                           'type': 'scatter3d',
                           'text': '', #oly_df['Name'],
                           'mode': 'markers',
                           'colorbar' : dict(title="Colorbar"),
                           'marker': {'size': 10}
                           }
                       ],
                       'layout' : go.Layout(clickmode='event+select', showlegend=False,
                           title='', autosize=True,height=600,
                           scene=dict(
                               xaxis={'title' : {'text' : 'Height'}},
                                yaxis={'title': {'text' : 'Weight'}},
                                zaxis={'title': {'text' : 'Age'}}))
                                }
               )

            ], style={"margin-top": "25px", 'margin-left': '420px'}),

    #html.Div(dash_table.DataTable(
       #id=table_id,
       #columns=[{"name": i, "id": i} for i in sport_df.columns],
       #data=sport_df.to_dict('records'),
       #))
       ],
       )

    @app.callback(
        dash.dependencies.Output(graph3_id, "figure"),
        [dash.dependencies.Input("Weight", "value"),
        dash.dependencies.Input("Height", "value"),
        dash.dependencies.Input("Age", "value"),
        dash.dependencies.Input("sport-dropdown", "value")]
    )
    def filter_graph_3(weight_value, height_value, age_value, sport_value):
        oly_df_copy = copy.deepcopy(oly_df)
        oly_df_copy['size'] = 15
        if sport_value != sport_value:
            sport_value = ''
            title = ''
        else:
            title = 'Athletes in Your Sport'
        oly_df_copy['Medal_val'] = [3 if i == 'Gold' else 2 if i == 'Silver' else 1 if i == 'Bronze' else 0 for i in oly_df_copy['Medal']]
        val_list = ['YOU!', sport_value, height_value, weight_value, age_value, 4, 60]
        #val_list = [0 if x is None else x for x in val_list]
        oly_df_copy.loc['Person', [
        'Name', 'Sport', f'Height_{unit_str}', f'Weight_{unit_str}', 'Age', 'Medal_val', 'size']] = val_list

        oly_df_copy = oly_df_copy[oly_df_copy['Sport'] == sport_value]

        return {
            'data': [
                {
                'x': oly_df_copy[f'Height_{unit_str}'],
                'y': oly_df_copy[f'Weight_{unit_str}'],
                'z': oly_df_copy['Age'],
                'hovertemplate' : 'Sport: %{text}<br>Height: %{x}<br>Weight: %{y}<br>Age: %{z}<br>Medal: %{customdata}',
                'customdata' : oly_df_copy['Medal'].replace('0', 'None'),
                'type': 'scatter3d',
                'text': oly_df_copy['Name'],
                'mode': 'markers',
                'colorbar' : dict(title="Colorbar"),
                'marker': {'size': oly_df_copy['size'],
                            'color' : ['#FF0000' if i == 4 else '#FFDF4D' if i == 3 else '#CFCFCF' if i == 2 else '#ac7132' if i == 1 else '#6A6A6A' for i in oly_df_copy['Medal_val']],
                            'opacity' : [1 if i == 4 else .75 if i == 3 else .5 if i == 2 else .25 if i == 1 else .1 for i in oly_df_copy['Medal_val']]

                            }
                }
            ],
            'layout' : go.Layout(clickmode='event+select', showlegend=False,
                title=title, autosize=True,height=600,
                scene=dict(
                    xaxis={'title' : {'text' : 'Height'}},
                     yaxis={'title': {'text' : 'Weight'}},
                     zaxis={'title': {'text' : 'Age'}}))
            }

    @app.callback(
        [dash.dependencies.Output(graph1_id, "figure"),
        dash.dependencies.Output(table_id, "data"),
        dash.dependencies.Output(table_id, "columns"),
        dash.dependencies.Output(graph2_id, "figure")],
        [dash.dependencies.Input("Weight", "value"),
        dash.dependencies.Input("Height", "value"),
        dash.dependencies.Input("Age", "value"),
        dash.dependencies.Input("feature-checklist", "value")]
    )
    def add_person_to_df(weight_value, height_value, age_value, checkbox_list):
        sport_df_copy = copy.deepcopy(sport_df)
        oly_df_copy = copy.deepcopy(oly_df)
        sport_df_copy['size'] = 15
        val_list = [height_value, weight_value, age_value, 2, 30]
        #val_list = [0 if x is None else x for x in val_list]
        sport_df_copy.loc['Person', [
        f'height_{unit_str}_winner_mean', f'weight_{unit_str}_winner_mean', 'age_winner_mean', 'pct_athletes_winners', 'size']] = val_list

        sport_df_copy.loc[:, 'age_winner_z_score'] = -1*(age_value - sport_df_copy['age_winner_mean']) / sport_df_copy['age_winner_std']

        sport_df_copy.loc[:, 'height_winner_z_score'] = -1*(height_value - sport_df_copy[f'height_{unit_str}_winner_mean']) / sport_df_copy[f'height_{unit_str}_winner_std']

        sport_df_copy.loc[:, 'weight_winner_z_score'] = -1*(weight_value - sport_df_copy[f'weight_{unit_str}_winner_mean']) / sport_df_copy[f'weight_{unit_str}_winner_std']

        print(checkbox_list)

        z_score_total = len(sport_df_copy) * [0]
        num_scores = 0
        zip_list = []
        if 'Age' in checkbox_list:
            z_score_total = z_score_total + sport_df_copy['age_winner_z_score']
            num_scores += 1
        if 'Height' in checkbox_list:
            z_score_total = z_score_total + sport_df_copy['height_winner_z_score']
            num_scores += 1
        if 'Weight' in checkbox_list:
            z_score_total = z_score_total + sport_df_copy['weight_winner_z_score']
            num_scores += 1

        meta = {
            "Weight": sport_df_copy[f'weight_{unit_str}_winner_mean'],
            "Height": sport_df_copy[f'height_{unit_str}_winner_mean'],
            "Age": sport_df_copy[f'age_winner_mean']
            }

        sport_df_copy['average_z_score'] = z_score_total / np.sqrt(num_scores)

        sport_df_copy = sport_df_copy.reset_index()
        return [{
        'data': [
            {
            'x': sport_df_copy[f'height_{unit_str}_winner_mean'],
            'y': sport_df_copy[f'weight_{unit_str}_winner_mean'],
            'z': sport_df_copy['age_winner_mean'],
            'type': 'scatter3d',
            'text': sport_df_copy['Sport'],
            'customdata' : [round(i, 2) for i in sport_df_copy['pct_athletes_winners']],
            'hovertemplate' : 'Sport: %{text}<br>Height: %{x}<br>Weight: %{y}<br>Age: %{z}<br>Pct Winners: %{customdata}',
            #'meta': {'colors_pct' : sport_df_copy['pct_athletes_winners']},
            'mode': 'markers',
            'marker': {'size': sport_df_copy['size'], 'color' : [i if i <= 1 else '#FF0000' for i in sport_df_copy['pct_athletes_winners']],
                        'showscale' : "True",
                        'colorbar' : {'title' : {'text' : '% Winners'},
                            'titleside' : 'top',
                            'tickmode' : 'array',
                            'tickvals' : [0.02,1, 2],
                            'ticktext' : ['2%','100%', 'You!'],
                            'ticks' : 'outside',
                            'xpad' : '50px'
                        }, ### Closes colorbar
                    } ### Closes marker
            } ### Closes first element in `data` list
        ], ### data list,
        'layout' : go.Layout(clickmode='event+select',
            title='Medalist Stats Per Sport', autosize=True,height=600, showlegend=False,
            scene=dict(
                xaxis={'title' : {'text' : 'Height'}},
                 yaxis={'title': {'text' : 'Weight'}},
                 zaxis={'title': {'text' : 'Age'}}) ### Scene close
                    ) ### layout
        }, ### Return value

        oly_df_copy.head(100).to_dict('records'), \
        [{"name": i, "id": i} for i in oly_df_copy.head(100).columns], \

        {
            'data': [
                {
                'x': sport_df_copy['average_z_score'],
                'y': round(sport_df_copy['pct_athletes_winners'].replace(2, 0) * 100, 2),
                'customdata' : round(sport_df_copy['pct_total_medals'] * 100, 2),
                'type': 'Scatter',
                'hovertemplate' : 'Sport: %{text}<br>Percent of Total Medals: %{customdata}<br>Percent of Athletes Awarded Medals: %{y}',
                'text': sport_df_copy.Sport,
                'mode': 'markers',
                'marker': {'size': 20, 'color' : sport_df_copy['pct_athletes_winners'].replace(2, 0)}
                }
            ],
            'layout' : go.Layout(clickmode='event+select', showlegend=False,
                title='Difference Plot', autosize=True,height=600,
                    xaxis={'title' : {'text' : 'Difference Value'}},
                     yaxis={'title': {'text' : '% of Athletes Awarded Medals'}}
                 )
        }]


    return layout


"""@app.callback(
    [dash.dependencies.Output("table", "data"),
    dash.dependencies.Output("weight_title", "value")],
    [dash.dependencies.Input("convert_button", "n_clicks")]
)
def convert_weights(n_clicks):
    winner_sport_groupdf_copy = copy.deepcopy(winner_sport_groupdf)
    weight_columns = [i for i in winner_sport_groupdf_copy.columns if re.match('weight_.*', i)]
    ### Convert to pounds
    if n_clicks % 2 == 0:
        ktl_fact = 2.2046226218
        for column in weight_columns:
            winner_sport_groupdf_copy[column] = winner_sport_groupdf_copy[column] * ktl_fact
        return [winner_sport_groupdf_copy.to_dict('records'), 'Weight (lbs.)']
    ### Convert to kilograms
    if n_clicks % 2 == 1:
        ltk_fact = 0.45359237
        for column in weight_columns:
            winner_sport_groupdf_copy[column] = winner_sport_groupdf_copy[column] * ktl_fact
        return [winner_sport_groupdf_copy.to_dict('records'), 'Weight (kgs.)']
"""
