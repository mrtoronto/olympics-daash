import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
from apps import app_kg, app_lbs, app_home
from apps.create_data import _create_data
from apps.app_func import _create_graphs

sport_df = pd.read_csv('../data/sport_df_proc.csv', index_col='Sport')
oly_df = pd.read_csv('../data/oly_df_proc.csv')


weight_unit = 'kg'
height_unit = 'cm'
unit_str = 'met'

app.layout = html.Div([
html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div([
            html.H3(
                "Which Olympic Sport are You Made For?",
                ),
            html.Label(
                "Enter your dimensions and see how you compare to the best athletes on the planet!",
                style={"margin-bottom": "0px"}),
            dcc.Link('American Units', href='/apps/app_lbs', style={'margin-top' : '20px'}),
            html.Div(),
            dcc.Link('Metric Units', href='/apps/app_kg', style={'margin-top' : '20px'}),
            html.Label('Age', style={'margin-top': '10px', 'margin-bottom': '5px'}),
            html.Div([
                dcc.Slider(id='Age', min=10, max=70, step=1, value=round(oly_df.Age.mean(), 2),marks={i : str(i) for i in range(10, 71, 10)})
                    ], style={'margin-right' : '15px', 'margin-left': '15px'}),
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
                options=[{"label": i, "value": i} for i in sport_df.index]),
            html.Label(f'Weight ({weight_unit})', style={"margin-top": "20px"}, id='weight_title'),
            #html.Button('Convert', id='convert_button'),
            dcc.Input(
                id="Weight",
                type='number',
                value=round(oly_df[f'Weight_{unit_str}'].mean(), 2)
                ),
            html.Label(f'Height ({height_unit})', style={"margin-top": "20px"}, id='height_title'),
            dcc.Input(
                id="Height",
                type='number',
                value=round(oly_df[f'Height_{unit_str}'].mean(), 2)
                ),
            dcc.Checklist(
                options=[
                    {'label': 'Age', 'value': 'Age'},
                    {'label': 'Weight', 'value': 'Weight'},
                    {'label': 'Height', 'value': 'Height'}
                    ], id='feature-checklist',
                value=['Age', 'Weight', 'Height'],
                labelStyle={'display': 'inline-block'}
                )
            ],
            id="header",
            #className="pretty_container four columns",
            style={"margin-bottom": "25px", "width" : "25%", 'margin-top' : '0px', 'position': 'absolute',
            'float' : 'left', 'margin-right': '25px'}
            ),
            html.Div(id='page-content', style={'float' : 'right'})]
            )]
)])


@app.callback([Output('page-content', 'children'),
            Output('page-content', 'style'),
            Output('weight_title', 'children'),
            Output('height_title', 'children'),
            Output('Weight', 'value'),
            Output('Height', 'value')],
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/app_kg':
        weight_unit = 'kg'
        height_unit = 'cm'
        unit_str = 'met'
        return [app_kg.layout, {'margin-top' : '0px'},
        f'Weight ({weight_unit})', f'Height ({height_unit})',
        round(oly_df[f'Weight_{unit_str}'].mean(), 2),
        round(oly_df[f'Height_{unit_str}'].mean(), 2)]
    elif pathname == '/apps/app_lbs':
        weight_unit = 'lbs'
        height_unit = 'in'
        unit_str = 'emp'
        return [app_lbs.layout, {'margin-top' : '0px'},
        f'Weight ({weight_unit})', f'Height ({height_unit})',
        round(oly_df[f'Weight_{unit_str}'].mean(), 2),
        round(oly_df[f'Height_{unit_str}'].mean(), 2)]
    else:
        return [app_home.layout, {'margin-top' : '0px'}, '', '', 0, 0]

if __name__ == '__main__':
    app.run_server(debug=True)
