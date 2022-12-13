import streamlit as st
import pandas as pd
import pylab as plt
from PIL import Image
import webbrowser
import urllib.request
import pickle  
import base64

st.title('Clase Streamlit')

st.header('METODOS PARA INTRODUCIR TEXTO EN LA PAGINA')

st.subheader('Existen varios metodos para introducir texto')

st.write('introducir texto')


st.write('# Si ponemos "#" es como un h1')
st.write('## Si ponemos "##" es como un h2')
st.write('### Si ponemos "###" es como un h3')


st.info('Esta clase ya casi esta.')
st.error('Esta clase ya casi esta.')
st.success('Esta clase ya casi esta.')
st.warning('Esta clase ya casi esta.')

df = pd.read_csv('/Users/javi/Desktop/ironhack/apuntes_clase/semana_8/Streamlit/src/data/comunio_J6.csv')
st.caption('# Podemos cargar un dataframe y mostrarlo')

filtros, equipos, pos, goles = st.columns(4)

with filtros:
    columnas = df.columns
    selection = st.multiselect('Filtrar Columnas', columnas, default=['Team','Player', 'Position', 'Matchs', 'Goals',
                                                                      'Total_Points'])

with equipos:
    equipo = st.selectbox('Filtrar Equipos', df.Team.unique())

with pos:
    player_pos = st.selectbox('Filtrar Posicion', df.Position.unique())

with goles:
    gol_min, gol_max = st.select_slider('Filtrar por Goles', options=[i for i in range(0, df.Goals.max()+1)],
                                        value=[0, df.Goals.max()])
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
else:
    df1 = df[selection]

    var = df1[(df1.Team == equipo) &
              (df1.Goals >= gol_min) &
              (df1.Goals <= gol_max) &
              (df1.Position == player_pos)]

    st.dataframe(var)

    df_plot = df1[(df1.Team == equipo) &
        (df1.Goals >= gol_min) &
        (df1.Goals <= gol_max) &
        (df1.Position == player_pos)]
    fig, ax = plt.subplots()
    plt.title(f'Puntos Totales del {equipo} por Jugador - Posición {player_pos}')
    ax.barh(y=df_plot.Player, width=df_plot.Total_Points)

    @st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])