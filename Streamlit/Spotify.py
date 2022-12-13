import streamlit as st
import pandas as pd
import pylab as plt
from PIL import Image
import webbrowser
import urllib.request
import pickle

st.sidebar.image(Image.open('/Users/javi/Desktop/Proyecto-FInal-Spotify/img/spoti.png'))


app_mode = st.sidebar.selectbox('Select Page',['Home','Global Prediction', 'Spanish Prediction'])

if app_mode=='Home':

    st.title('BOP OR FLOP')

    st.header('A way to predict hits based on audio features')

    
    df = pd.read_csv('../CSV_primeros/datos_spotipy_week_1.csv')

    filtros,  = st.columns(1)

    filtros, artistas, cancion, hit = st.columns(4)

    with filtros:
        columnas = df.columns
        selection = st.multiselect('Filtrar Columnas', columnas, default=['artist_name', 'track_name',
        'duration', 'prediccion'])
    with artistas:
        artistas = st.selectbox('Filtrar Artistas', df.artist_name.unique())
    with cancion:
        cancion = st.selectbox('Filtrar Canciones', df.track_name.unique())
    with hit :
        hit_or_not= st.selectbox('Exito o No', df.prediccion.unique())

    df1 = df[selection]

    #var = df1[(df1.artist_name == artistas)] #&
                #(df1.track_name == cancion)] #&
                #(df1.prediccion == hit)]
        

    st.dataframe(df)

    #uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

elif app_mode == 'Global Prediction':
    datos_good= pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/streamlit/datos_spotipy_week_1_song.csv')
    st.title("Prediction of Global playlist")
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
    st.write(datos_good[['artist_name','track_name','prediccion']]) 
    st.write(y)

elif app_mode == 'Spanish Prediction':
    datos_good= pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/CSV_full/data_week_9DEC_SPAIN.csv')
    st.title("Prediction of Spanish playlist")
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
    st.write(datos_good[['artist_name','track_name','prediccion']]) 
    st.write(y)
 


