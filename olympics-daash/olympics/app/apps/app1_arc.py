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

oly_df = pd.read_csv('../data/athlete_events.csv', index_col='ID')
columns=['Name', 'Sex', 'Age','Height', 'Weight', 'Season', 'Sport', 'Event']
oly_df = oly_df[columns + ['Medal']]
oly_df = oly_df.dropna(subset=columns)


all_sport_groupdf = oly_df.groupby('Sport').agg(['count', 'std', 'mean'])
all_sport_groupdf.columns = [i[0].lower() + '_' +  i[1] for i in all_sport_groupdf.columns]
all_sport_groupdf = all_sport_groupdf.drop(['height_count', 'weight_count'], axis=1).rename({'age_count' : 'count'}, axis=1)


winner_sport_groupdf = oly_df[oly_df['Medal'].notna()].groupby('Sport').agg(['count', 'mean', 'std'])
winner_sport_groupdf.columns = [i[0].lower() + '_' +  i[1] for i in winner_sport_groupdf.columns]
print(winner_sport_groupdf.columns)
winner_sport_groupdf = winner_sport_groupdf.drop(['height_count', 'weight_count'], axis=1).rename({'age_count' : 'count'}, axis=1)
winner_sport_groupdf = winner_sport_groupdf.replace(np.nan, 0).reset_index()



pca = PCA(n_components=2)
### Do PCA on stats to get X and Y of scatter
pca_drop_list = [i for i in winner_sport_groupdf.columns if re.match('.*_std', i)] + ['count', 'Sport']
pca_df = winner_sport_groupdf.drop(pca_drop_list, axis=1)
print(pca_df.columns)
winner_pca_df = pca.fit_transform(pca_df)
winner_sport_groupdf.loc[:, 'PCA_1'] = winner_pca_df[:, 0]
winner_sport_groupdf.loc[:, 'PCA_2'] = winner_pca_df[:, 1]


layout = html.Div([
        html.Div([
            html.H3(
                "Which Olympic Sport are You Made For?",
                style={"margin-bottom": "0px"}),
            html.Label('Age', style={'margin-top': '10px', 'margin-bottom': '5px'}),
            dcc.Slider(id='Age', min=10, max=70, step=1, value=round(oly_df.Age.mean(), 2),
                        marks={i : str(i) for i in range(10, 71, 10)},
                        ),
            html.Label('Sex', style={"margin-top": "40px"}),
            dcc.Dropdown(
                id='sex-dropdown',
                options=[
                    {'label': 'Male', 'value': 'M'},
                    {'label': 'Female', 'value': 'F'},
                    {'label': 'Other', 'value': 'O'}
                ]),
            html.Label('Weight (kg)', style={"margin-top": "20px"}, id='weight_title'),
            #html.Button('Convert', id='convert_button'),
            dcc.Input(
                id="Weight",
                type='number',
                value=round(oly_df.Weight.mean(), 2)
                ),
            html.Label('Height (cm)', style={"margin-top": "20px"}),
            dcc.Input(
                id="Height",
                type='number',
                value=round(oly_df.Height.mean(), 2)
                )
            ],
            id="header",
            #className="pretty_container four columns",
            style={"margin-bottom": "25px", "width" : "200"}
            ),
    html.Div([
        dcc.Graph(
               id='graph',
               figure={
                   'data': [
                       {
                       'x': winner_sport_groupdf['height_mean'],
                       'y': winner_sport_groupdf['weight_mean'],
                       'z': winner_sport_groupdf['age_mean'],
                       'type': 'scatter3d',
                       'text': winner_sport_groupdf['Sport'],
                       'name': 'Trace 1',
                       'mode': 'markers',
                       'marker': {'size': 10, 'color' : ['DarkSlateGrey'] * len(winner_sport_groupdf)}}
                   ],
                   'layout': {'clickmode': 'event+select', 'autosize' : 'True', 'height' : 600,
                       'xaxis' : {'title' : {'text' : 'Height'}},
                        'yaxis': {'title': {'text' : 'Weight'}},
                        'zaxis': {'title': {'text' : 'Age'}}
                            }
                    }
           ),
        dash_table.DataTable(
           id='table',
           columns=[{"name": i, "id": i} for i in winner_sport_groupdf.columns],
           data=winner_sport_groupdf.to_dict('records'),
           )
        ], style={"margin-top": "25px"})
])

@app.callback(
    dash.dependencies.Output("graph", "figure"),
    [dash.dependencies.Input("Weight", "value"),
    dash.dependencies.Input("Height", "value"),
    dash.dependencies.Input("Age", "value")]
)
def add_pca_to_df(weight_value, height_value, age_value):
    winner_sport_groupdf_copy = copy.deepcopy(winner_sport_groupdf)
    val_list = [height_value, weight_value, age_value]
    val_list = [0 if x is None else x for x in val_list]
    winner_sport_groupdf_copy.loc['Person', ['height_mean', 'weight_mean', 'age_mean']] = val_list
    #person_array = np.asarray(val_list).reshape(1,-1)
    #person_pca = pca.transform(person_array)
    #winner_sport_groupdf_copy.loc['Person', 'PCA_1'] = person_pca[:, 0]
    #winner_sport_groupdf_copy.loc['Person', 'PCA_2'] = person_pca[:, 1]
    return {
    'data': [
        {
        'x': winner_sport_groupdf_copy['height_mean'],
        'y': winner_sport_groupdf_copy['weight_mean'],
        'z': winner_sport_groupdf_copy['age_mean'],
        'type': 'scatter3d',
        'text': winner_sport_groupdf_copy['Sport'],
        'name': 'Trace 1',
        'mode': 'markers',
        'marker': {'size': 10, 'color' : ['DarkSlateGrey'] * (len(winner_sport_groupdf_copy) - 1) + ['red']}
        }
    ],
    'layout': {'clickmode': 'event+select',  'autosize' : 'True','height' : 600,
        'xaxis' : {'title' : {'text' : 'Height'}},
         'yaxis': {'title': {'text' : 'Weight'}},
         'zaxis': {'title': {'text' : 'Age'}}
             }
     }

@app.callback(
    [dash.dependencies.Output("table", "data"),
    dash.dependencies.Output("table", "columns")],
    [dash.dependencies.Input("Age", "value"),
    dash.dependencies.Input("Height", "value"),
    dash.dependencies.Input("Weight", "value")]
)
def add_z_scores_to_df(age_value, height_value, weight_value):
    winner_sport_groupdf_copy = copy.deepcopy(winner_sport_groupdf)
    winner_sport_groupdf_copy.loc[:, 'age_z_score'] = (age_value - winner_sport_groupdf_copy['age_mean']) / winner_sport_groupdf_copy['age_std']
    winner_sport_groupdf_copy.loc[:, 'height_z_score'] = (height_value - winner_sport_groupdf_copy['height_mean']) / winner_sport_groupdf_copy['height_std']
    winner_sport_groupdf_copy.loc[:, 'weight_z_score'] = (age_value - winner_sport_groupdf_copy['weight_mean']) / winner_sport_groupdf_copy['weight_std']
    return winner_sport_groupdf_copy.to_dict('records'), \
            [{"name": i, "id": i} for i in winner_sport_groupdf_copy.columns]


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
