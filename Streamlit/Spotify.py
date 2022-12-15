import streamlit as st
import pandas as pd
import pylab as plt
from PIL import Image
import webbrowser
import urllib.request
import pickle
import warnings
warnings.filterwarnings('ignore')
import random
import spotipy
from spotipy.oauth2 import SpotifyOAuth
disableWatchdogWarning = False

st.sidebar.image(Image.open('/Users/javi/Desktop/Proyecto-FInal-Spotify/img/spoti.png'))


app_mode = st.sidebar.selectbox('Select Page',['Home','Global Prediction', 'Spanish Prediction','Playlist'])

if app_mode=='Home':

    st.title('BOP OR FLOP')
    st.subheader("A Spotify analysis to predict hits")
    st.header('Global hits Data')

    st.image(Image.open('/Users/javi/Desktop/Proyecto-FInal-Spotify/img/Global_hits.png'))


    

    df = pd.read_csv('../CSV_primeros/all_songs.csv')

    

    #st.write(df.shape)
    #df = pd.read_csv('../CSV_primeros/datos_spotipy_week_1.csv')


    filtros, artistas = st.columns(2)

    with filtros:
        columnas = df.columns
        selection = st.multiselect('Filtrar Columnas', columnas, default=['artist_name', 'track_name'])
    with artistas:
        artistas = st.selectbox('Filtrar Artistas', df.artist_name.unique())
    #with cancion:
        #cancion = st.selectbox('Filtrar Canciones', df.track_name.unique())
    #with hit :
        #hit_or_not= st.selectbox('Exito o No', df.prediccion.unique())

    df1 = df[selection]

    var = df1[(df1.artist_name == artistas)] #&
                #(df1.track_name == cancion)] #&
                #(df1.prediccion == hit)]
        

    st.dataframe(var)
    

    #uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

elif app_mode == 'Global Prediction':
    st.title("Prediction of Global playlist")
    st.image(Image.open('/Users/javi/Desktop/Proyecto-FInal-Spotify/img/Novedades_global.png'))

    datos_good= pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/streamlit/datos_spotipy_week_1_song.csv')
    
    HGBT = pickle.load(open('/Users/javi/Desktop/Proyecto-FInal-Spotify/CSV_primeros/HGBT.pkl','rb'))
    st.subheader("Enter your CSV")
    datos= pd.read_csv(st.file_uploader("Upload your input CSV file", type=["csv"]))
    #st.write(datos)


    st.subheader("Predicted hit")
    pred=HGBT.predict(datos)
    #st.write(pred)
    pred = pd.Series(pred)
    datos_good['prediccion'] = pred.round(decimals = 0)
    y = datos_good.prediccion.value_counts()
    st.subheader("Predicted No hits and hits")
    datos_good['prediccion'] = datos_good['prediccion'].astype(str)
    def limpiar_prediccion(column):
        if '0' in column:
            return 'No Hit'
        else:
            return 'Hit'
    datos_good['prediccion'] = datos_good['prediccion'].apply(limpiar_prediccion)
    st.write(datos_good[['artist_name','track_name','prediccion']]) 
    #y = y.astype(str)
    #y = y.apply(limpiar_prediccion)
    y.index=['No Hits', 'Hits']
    st.write(y)

elif app_mode == 'Spanish Prediction':
    st.title("Prediction of Spanish playlist")
    st.image(Image.open('/Users/javi/Desktop/Proyecto-FInal-Spotify/img/Novedades_viernes.png'))
    datos_good= pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/CSV_full/data_week_9DEC_SPAIN.csv')
    GBC = pickle.load(open('/Users/javi/Desktop/Proyecto-FInal-Spotify/CSV_primeros/GBC.pkl','rb'))
    st.subheader("Enter your CSV")
    datos= pd.read_csv(st.file_uploader("Upload your input CSV file", type=["csv"]))
    #st.write(datos)


    st.subheader("Predicted hit")
    pred=GBC.predict(datos)
    #st.write(pred)
    pred = pd.Series(pred)
    datos_good['prediccion'] = pred.round(decimals = 0)
    y = datos_good.prediccion.value_counts()
    st.subheader("Predicted No hits and hits")
    datos_good['prediccion'] = datos_good['prediccion'].astype(str)
    def limpiar_prediccion(column):
        if '0' in column:
            return 'No Hit'
        else:
            return 'Hit'
    datos_good['prediccion'] = datos_good['prediccion'].apply(limpiar_prediccion)
    st.write(datos_good[['artist_name','track_name','prediccion']]) 
    y.index=['No Hits', 'Hits']
    st.write(y)

elif app_mode == 'Playlist':
    links= pd.read_csv(st.file_uploader("Upload your input CSV file", type=["csv"]))
    links = links['0'].tolist()
    links_rd = random.sample(links,100)
    token='BQDJ4BKfvhpLd2a0G2z-noYzp5oo0m6PhUPWHGJu2ppNmgxu5LTKCRDzSFdlDp_1uSYNFGHVRRGc7AUckDDSeEXitFrgNz9sRy047MRFELnC1W8YQZvl5KXbGVy_c26uDyfw6rFMzH9f9UaoLRrsYe3rVYjdF8qVMAbdeIFhydJGBwX0S2UMMHFj1mvu9iq6H6Jn7xqQJzqFeGm4nFOrJoHpgq7sofKSUKpApYzEYQ'
    sp = spotipy.Spotify(auth=token)
    sp.user_playlist_add_tracks(user='javi1025', 
                            playlist_id='48KXGa9uNtJXQsax6Pb6LM',
                            tracks=links_rd, 
                            position=None)


