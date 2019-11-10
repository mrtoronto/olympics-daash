from app import app
import dash_html_components as html
import dash_core_components as dcc


layout = html.Div([
    html.H3('Choose a unit system on the sidebar to generate some plots!', style={'text-align' : 'center'})
], style={'margin-top' : '50px'})
