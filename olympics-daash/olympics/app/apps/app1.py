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

sport_df, oly_df = _create_data()

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
            html.Label('Sport', style={"margin-top": "20px"}),
            dcc.Dropdown(
                id='sport-dropdown',
                options=[{"label": i, "value": i} for i in sport_df.Sport]),
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
            style={"margin-bottom": "25px", "width" : "300px", 'height' : '25vh',
            'float' : 'left'}
            ),
    html.Div([
        dcc.Graph(
               id='graph',
               figure={
                   'data': [
                       {
                       'x': sport_df['height_winner_mean'],
                       'y': sport_df['weight_winner_mean'],
                       'z': sport_df['age_winner_mean'],
                       'type': 'scatter3d',
                       'text': sport_df['Sport'],
                       'mode': 'markers',
                       'colorbar' : dict(title="Colorbar"),
                       'marker': {'size': 10, 'color' : sport_df['pct_athletes_winners'], 'colorscale' : "Cividis", 'showscale' : "True",
                       'colorbar' : {'title' : {'text' : '% of Competitors Awarded Medals'},
                                   'titleside' : 'top',
                                   'tickmode' : 'array',
                                   'tickvals' : [0.02,1, 2],
                                   'ticktext' : ['2%','100%', 'You!'],
                                   'ticks' : 'outside',
                                   'xpad' : '50px'},
                               }
                       }
                   ],
                   'layout' : go.Layout(clickmode='event+select', showlegend=False,
                       title='You Compared to the Average Olympic Athlete', autosize=True,height=600,
                       scene=dict(
                           xaxis={'title' : {'text' : 'Height'}},
                            yaxis={'title': {'text' : 'Weight'}},
                            zaxis={'title': {'text' : 'Age'}}))
                            }
           ),
        dcc.Graph(
               id='graph2',
               figure={
                   'data': [
                       {
                       'x': sport_df['weight_winner_z_score'],
                       'y': sport_df['pct_total_medals'],
                       'type': 'Scatter',
                       'text': sport_df['Sport'],
                       'mode': 'markers',
                       'colorbar' : dict(title="Colorbar"),
                       'marker': {'size': 10, 'color' : sport_df['pct_athletes_winners'], 'colorscale' : "Cividis", 'showscale' : "True",
                       'colorbar' : {'title' : {'text' : '% Awarded Medals'},
                                   'titleside' : 'top',
                                   'tickmode' : 'array',
                                   'tickvals' : [0.02,1, 2],
                                   'ticktext' : ['2%','100%', 'You!'],
                                   'ticks' : 'outside',
                                   'xpad' : '150px'},
                               }
                       }
                   ],
                   'layout' : go.Layout(clickmode='event+select', showlegend=False,
                       title='Similar Sports and their pecent medals won', autosize=True,height=600,
                       scene=dict(
                           xaxis={'title' : {'text' : 'Height'}},
                            yaxis={'title': {'text' : 'Weight'}},
                            zaxis={'title': {'text' : 'Age'}}))
                            }
           )
        ], style={"margin-top": "25px", 'margin-left': '320px',  'border-left-style':'dashed', 'border-left-width':'2px'}),

html.Div(dash_table.DataTable(
   id='table',
   columns=[{"name": i, "id": i} for i in sport_df.columns],
   data=sport_df.to_dict('records'),
   )
   )],
   )

@app.callback(
    dash.dependencies.Output("graph", "figure"),
    [dash.dependencies.Input("Weight", "value"),
    dash.dependencies.Input("Height", "value"),
    dash.dependencies.Input("Age", "value")]
)
def add_person_to_df(weight_value, height_value, age_value):
    sport_df_copy = copy.deepcopy(sport_df)
    val_list = ['Person', height_value, weight_value, age_value, 2]
    #val_list = [0 if x is None else x for x in val_list]
    sport_df_copy.loc['Person', [
    'Sport', 'height_winner_mean', 'weight_winner_mean', 'age_winner_mean', 'pct_athletes_winners']] = val_list
    #person_array = np.asarray(val_list).reshape(1,-1)
    #person_pca = pca.transform(person_array)
    #winner_sport_groupdf_copy.loc['Person', 'PCA_1'] = person_pca[:, 0]
    #winner_sport_groupdf_copy.loc['Person', 'PCA_2'] = person_pca[:, 1]
    #color_series = pd.Series([i if i is not None else 'red' for i in sport_df_copy['pct_athletes_winners']])
    return {
    'data': [
        {
        'x': sport_df_copy['height_winner_mean'],
        'y': sport_df_copy['weight_winner_mean'],
        'z': sport_df_copy['age_winner_mean'],
        'type': 'scatter3d',
        'text': sport_df_copy['Sport'],
        'customdata' : [round(i, 2) for i in sport_df_copy['pct_athletes_winners']],
        'hovertemplate' : 'Sport: %{text}<br>Height: %{x}<br>Weight: %{y}<br>Age: %{z}<br>Pct Winners: %{customdata}',
        #'meta': {'colors_pct' : sport_df_copy['pct_athletes_winners']},
        'mode': 'markers',
        'marker': {'size': 10, 'color' : sport_df_copy['pct_athletes_winners'],
                    'colorscale' : "Cividis",
                    'showscale' : "True",
                    'colorbar' : {'title' : {'text' : '% Awarded Medals'},
                        'titleside' : 'top',
                        'tickmode' : 'array',
                        'tickvals' : [0.02,1, 2],
                        'ticktext' : ['2%','100%', 'You!'],
                        'ticks' : 'outside'
                    }, ### Closes colorbar
                } ### Closes marker
        } ### Closes first element in `data` list
    ], ### data list,
    'layout' : go.Layout(clickmode='event+select',
        title='You Compared to the Average Olympic Athlete', autosize=True,height=600, showlegend=False,
        scene=dict(
            xaxis={'title' : {'text' : 'Height'}},
             yaxis={'title': {'text' : 'Weight'}},
             zaxis={'title': {'text' : 'Age'}}) ### Scene close
                ) ### layout
    } ### Return value


@app.callback(
    [dash.dependencies.Output("table", "data"),
    dash.dependencies.Output("table", "columns"),
    dash.dependencies.Output("graph2", "figure")],
    [dash.dependencies.Input("Age", "value"),
    dash.dependencies.Input("Height", "value"),
    dash.dependencies.Input("Weight", "value")]
)
def add_z_scores_to_df(age_value, height_value, weight_value):

    sport_df_copy = copy.deepcopy(sport_df)
    sport_df_copy.loc[:, 'age_winner_z_score'] = (age_value - sport_df_copy['age_winner_mean']) / sport_df_copy['age_winner_std']

    sport_df_copy.loc[:, 'height_winner_z_score'] = (height_value - sport_df_copy['height_winner_mean']) / sport_df_copy['height_winner_std']

    sport_df_copy.loc[:, 'weight_winner_z_score'] = (weight_value - sport_df_copy['weight_winner_mean']) / sport_df_copy['weight_winner_std']

    return sport_df_copy.to_dict('records'), \
            [{"name": i, "id": i} for i in sport_df_copy.columns], \
            {
                'data': [
                    {
                    'x': sport_df_copy['weight_winner_z_score'],
                    'y': sport_df_copy['pct_total_medals'],
                    'type': 'Scatter',
                    'text': sport_df_copy['Sport'],
                    'mode': 'markers',
                    'colorbar' : dict(title="Colorbar"),
                    'marker': {'size': 10, 'color' : sport_df_copy['pct_athletes_winners'], 'colorscale' : "Cividis", 'showscale' : "True",
                    'colorbar' : {'title' : {'text' : '% of Competitors Awarded Medals'},
                                'titleside' : 'top',
                                'tickmode' : 'array',
                                'tickvals' : [0.02,1,2],
                                'ticktext' : ['2%','100%', 'You!'],
                                'ticks' : 'outside',
                                'xpad' : '50px'},
                            }
                    }
                ],
                'layout' : go.Layout(clickmode='event+select', showlegend=False,
                    title='Similar Sports and their pecent medals won', autosize=True,height=600,
                    scene=dict(
                        xaxis={'title' : {'text' : 'Height'}},
                         yaxis={'title': {'text' : 'Weight'}},
                         zaxis={'title': {'text' : 'Age'}}))
            }


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
