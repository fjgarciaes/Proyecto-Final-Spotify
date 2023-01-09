# Proyecto-Final-Spotify

![Spotify](https://user-images.githubusercontent.com/114060666/211303123-d4e77169-438f-492e-a23c-500bb2cf0bb5.png)


**Index**


1. Objectives 🎯 
2. Process followed 📋 
3. Visualization 💹 
4. Machine Learning 🤖 
5. Streamlit 🌐
6. Future Steps ➡️

🎯 OBJECTIVES

Create a database with global and Spanish popular and unpopular songs.

Visualize the most important variables for the model to work.

Create a machine learning model to determine whether the songs that come out each week are going to be hits or not.

Create an app to be able to explore the data, make real-time predictions and create a playlist of random hits. 

📋 Process followed

1) Data extraction.

  Most of the data was extracted using Spotipy and the rest was extracted from Kaggle

2) Data transformation.

  Once all the data was extracted and converted into data frames, it was transformed and cleaned as follows:

  a) Global and Spanish popular songs Dataframes.

    The columns that are not useful for the analysis are eliminated.

    The necessary columns are renamed.

    The columns are transformed so that the data is in the required format.
    
    A column is created to identify which songs are hits.

  b) Unpopular songs Dataframe

    The same process used for the popular songs dataframes is repeated for this one.

    A column is added to indicate that the songs are not hits.

  c) Hit or not Dataframe.

    The dataframes of popular songs are concatenated with unpopular songs both globally and for Spain.

  3) Loading data into MySQL

    After saving all the clean dataframes as CSV I proceed to create a new database in MySQL called Spotify.

💹 VISUALIZATION

The data visualization is done using powerBi, obtaining a dashboard in which the most important characteristics are shown in graphs.

🤖 MACHINE LEARNING

Using all the columns of the dataframe, 27 machine learning models are trained in order to know which one gives the best result:

1) Prediction of new Global hits:

  The best fitting model for this data is HistGradientBoostingClassifier.

  Once the model is trained it is used to predict how many of the new songs are going to be a hit.

2) Prediction of new hits Spain:

  The best fitting model for this data is GradientBoostingClassifier.

  The same process as for the global prediction is repeated.

🌐 Streamlit 

A Streamlit application was created in which the data can be observed, predictions can be made and random playlists can be created. 

➡️ Future Steps 

1) Add information about the artists into the dataframes to make the machine learning training process more accurate.

2) Modify the Streamit app to allow users to modify playlists.

3) Add audio playback function in Streanlit 





