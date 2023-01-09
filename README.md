# Proyecto-Final-Spotify


[Spotify](https://user-images.githubusercontent.com/114060666/211291220-2bd1308b-dafc-486b-934d-8b0698931744.png)

ÍNDICE!


1. Objetivos 🎯 
2. Pasos seguidos 📋 
3. Visualización 💹 
4. Machine Learning 🤖 

🎯 OBJETIVOS

Crear una base de datos con canciones populares a nivel global, en España y canciones que no son populares.

Visualizar aquellas variables que son mas importantes para el funcionamiento del modelo.

Crear un modelo predictivo para poder determinar si las canciones que nuevas que salen cada semana van a ser hits o no.

Crear una aplicacion de Streamlit para poder realizar la prediccion en tiempo real asi como una playlist en Spotify con éxitos.

📋 PASOS A SEGUIR

1) Extracción

Se han extraido datos de canciones populares y no populares de Kaggle.

Se extraen los datos de las canciones populares de España mediante Spotipy

Las canciones nuevas de cada semana se extraen también mediante Spotipy

2) Transformación

Una vez extraídos los datos y convertidos a dataframes , se  procede a su transformación y limpieza.

a) Dataframe de canciones populares globales y dataframe de canciones populres en España.

Se eliminan las columnas que no son utilies para el análisis.

Se crea una columna para identificar que las canciones son exitos.

Se renombran las columnas necesarias.

Se transforman las columnas para que los datos esten en el formato requerido.


b) Dataframe de canciones no populares.

Se repite el proceso de transformación que se siguio para los dataframes anteriores 

Se añade una columna para indicar que las canciones no son exitos.

c) Dataframe hit or not.

Se concatenan los dataframes de canciones populares con no populares tanto a nivel global como para España

3) Carga de datos en MySQL

Tras guardar todos los dataframes limpios como csv para procedo a crear una nueva base de datos en MySQL llamada Spotify.

💹 VISUALIZACIÓN

Se realiza la visualización de los datos en powerBi, obteniendo un dashboard en el cual se muestran gráficas de las caracteristicas mas importantes.


🤖 MACHINE LEARNING

Utilizando todas las columnas del dataframe se entrenan 27 modelos de machine learning para conocer con cual se obtiene el mejor resultado:

. Predicción nuevos hits Globales:

El modelo que mejor se ajusta para estos datos es HistGradientBoostingClassifier.

Una vez entrenado el modelo se utiliza para predecir cuantas de las canciones nuevas van a ser un hit.

. Predicción nuevos hits España:

El modelo que mejor se ajusta para estos datos es GradientBoostingClassifier.

Se repite el proceso que se realizo para la predicción global



