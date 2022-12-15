#!/usr/bin/env python
# coding: utf-8

# # Esto es lo que se hace aqui
# 
# Se quitan los duplicados despues de haber hecho el concat, es decir, cuando creo hit_or_not quedando al final 7492
# 
# Se dejan los nombres de los artistas a los cuales se les hace label encoder y despues se normaliza 
# 
# NO se normalizan los hits.
# 
# 242 fallos en total
# 
# En la prediccion salen 73 no hits y 27 hits.
# 
# ### 'artist_name': 0.0693195976045068,
# ### 'energy': 0.07733112450763746,
# ### 'danceability': 0.08524948142697877,
# ### 'loudness': 0.3636710122008491,
# ### 'acousticness': 0.07059834548863766,
# ### 'speechiness': 0.07398172109998695,
# ### 'liveness': 0.05438828306879722,
# ### 'valence': 0.06507654011836563,
# ### 'tempo': 0.05323665446981244,
# ### 'duration': 0.08714724001442802}
# 
# ## 'El mejor modelo es CTR con un mse de 0.34758623915616366'
# 
# 

# In[2]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import pylab as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore') 
#para que salga el grafico
#pd.set_option('display.max_rows', None)

from statistics import mean
import pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR  
from sklearn.tree import ExtraTreeRegressor as ETR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from xgboost import XGBRegressor as XGBR
from catboost import CatBoostRegressor as CTR
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Lasso    
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet 
from sklearn.model_selection import train_test_split as tts   
from sklearn.metrics import mean_squared_error as mse   


# # PROYECTO FINAL EMPIEZA AQUI

# # Voy a probar cosas nuevas que no tienen que ver con la ETL que ya hice a ver que puedo ir metiendo
# 

# In[11]:


canciones = pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/Spoti/songs_normalize.csv')

canciones.head(5)


# In[3]:


canciones.shape


# In[4]:


canciones.info(memory_usage='deep')


# In[3]:


top2020_21 = pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/Spoti/spotify_dataset2020-2021.csv')

top2020_21.head(5)


# In[6]:


top2020_21.head()


# In[7]:


top2020_21.info(memory_usage='deep')


# In[6]:


unpopular = pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/Spoti/unpopular_songs.csv')

unpopular.shape


# In[7]:


unpopular_mal = unpopular[['explicit','mode','popularity','key','track_id','instrumentalness']]


# In[8]:


unpopular.drop(['explicit','mode','popularity','key','track_id','instrumentalness'],axis=1,inplace=True)


# In[9]:


unpopular=unpopular.iloc[:, [10,9,1,0,2,4,3,5,6,7,8]]


# In[10]:


unpopular.head()


# In[13]:


unpopular.rename(columns = {'track_artist':'artist_name', 'track_name':'track_name','duration_ms':'duration'}, inplace = True)


# In[14]:


unpopular['top_hit']=[0 for i in range(len(unpopular))]


# In[15]:


unpopular['duration'] = (unpopular['duration']/1000).round(2)

unpopular.head()


# In[16]:


#unpopular.drop_duplicates().shape==unpopular.shape    


# In[17]:


canciones_mal = canciones[['explicit','year','popularity','key','genre']]


# In[18]:


canciones.drop(['explicit','year','popularity','key','genre','mode','instrumentalness'],axis=1,inplace=True)


# In[19]:


canciones.head()


# In[20]:


canciones=canciones.iloc[:, [0,1,4,3,5,7,6,8,9,10,2]]


# In[21]:


canciones.head()


# In[22]:


top2020_21_mal=top2020_21[['Index','Highest Charting Position','Number of Times Charted',                 'Week of Highest Charting','Streams','Artist Followers',                 'Song ID','Genre','Release Date','Weeks Charted',                'Popularity','Chord']]


# In[23]:


top2020_21.drop(['Index','Highest Charting Position','Number of Times Charted',                 'Week of Highest Charting','Streams','Artist Followers',                 'Song ID','Genre','Release Date','Weeks Charted',                'Popularity','Chord'],axis=1,inplace=True)


# In[24]:


top2020_21=top2020_21.iloc[:, [1,0,3,2,4,6,5,7,10,8,9]]


# In[25]:


top2020_21.head()


# In[4]:


spotify_2022 = pd.read_csv('/Users/javi/Desktop/Proyecto-FInal-Spotify/Spoti/spotify_2022.csv')


# In[5]:


spotify_2022.head()


# In[28]:


spotify_2022.drop('Unnamed: 0',axis=1,inplace=True)


# In[29]:


spotify_2022.rename(columns = {'duration_ms':'duration'}, inplace = True)


# In[30]:


canciones.rename(columns = {'artist':'artist_name', 'song':'track_name','duration_ms':'duration'}, inplace = True)


# In[31]:


canciones.head()


# In[32]:


#top2020_21.columns.str.lower()


# In[33]:


top2020_21.columns = map(str.lower, top2020_21.columns)


# In[34]:


top2020_21.head()


# In[35]:


top2020_21.rename(columns = {'artist':'artist_name', 'song name':'track_name','duration (ms)':'duration'}, inplace = True)


# In[36]:


top2020_21.head()


# In[37]:


len(top2020_21)


# In[38]:


canciones.tail()


# In[39]:


spotify_2022


# In[40]:


all_songs = pd.concat([canciones,top2020_21, spotify_2022]).reset_index(drop=True)


# In[41]:


all_songs['duration'] = pd.to_numeric(all_songs['duration'], errors='coerce')


# In[42]:


all_songs['duration'] = (all_songs['duration']/1000).round(2)

all_songs.head()


# In[43]:


all_songs.shape


# In[44]:


all_songs['top_hit']=[1 for i in range(len(all_songs))]


# In[45]:


#all_songs.drop_duplicates().shape==all_songs.shape    


# In[46]:


unpopular.shape


# In[47]:


unpopular = unpopular.sample(n=3785)


# In[48]:


unpopular.shape


# In[49]:


hit_or_not = pd.concat([all_songs,unpopular]).reset_index(drop=True)


# In[50]:


hit_or_not.shape


# In[51]:


#hit_or_not.drop(['level_0','index'],axis=1,inplace=True)


# In[52]:


hit_or_not.drop_duplicates().shape==hit_or_not.shape    


# In[53]:


hit_or_not=hit_or_not.drop_duplicates()


# In[54]:


hit_or_not.drop_duplicates().shape==hit_or_not.shape   


# In[55]:


hit_or_not.shape


# In[56]:


hit_or_not.energy.value_counts


# In[57]:


hit_or_not['energy'] = pd.to_numeric(hit_or_not['energy'], errors='coerce')


# In[58]:


hit_or_not['danceability'] = pd.to_numeric(hit_or_not['danceability'], errors='coerce')


# In[59]:


hit_or_not['loudness'] = pd.to_numeric(hit_or_not['loudness'], errors='coerce')


# In[60]:


hit_or_not['acousticness'] = pd.to_numeric(hit_or_not['acousticness'], errors='coerce')


# In[61]:


hit_or_not['speechiness'] = pd.to_numeric(hit_or_not['speechiness'], errors='coerce')


# In[62]:


hit_or_not['liveness'] = pd.to_numeric(hit_or_not['liveness'], errors='coerce')


# In[63]:


hit_or_not['valence'] = pd.to_numeric(hit_or_not['valence'], errors='coerce')


# In[64]:


hit_or_not['valence'] = pd.to_numeric(hit_or_not['valence'], errors='coerce')


# In[65]:


hit_or_not['tempo'] = pd.to_numeric(hit_or_not['tempo'], errors='coerce')


# In[66]:


#hit_or_not['duration'] = pd.to_numeric(hit_or_not['duration'], errors='coerce')


# In[67]:


hit_or_not.info(memory_usage='deep')


# In[ ]:





# In[68]:


#hit_or_not['duration'] = (hit_or_not['duration']/1000).round(2)

#hit_or_not.head()


# In[69]:


hit_or_not = hit_or_not.dropna().reset_index(drop=True)


# In[70]:


hit_or_not.info(memory_usage='deep')


# # SQL

# In[ ]:


#with open('../Proyecto-FInal-Spotify/token.txt', 'r') as file:
    #contraseña=file.read()


# In[ ]:


#from sqlalchemy import create_engine


# In[ ]:


#str_conn=f'mysql+pymysql://root:{contraseña}@localhost:3306/proyecto_final'

#cursor=create_engine(str_conn)


# In[ ]:


#hit_or_not.to_sql(name='hit_or_not', con=cursor, if_exists='replace',index=False)


# # COLINEALIDAD

# In[ ]:





# In[ ]:


canciones.shape


# In[ ]:


unpopular.shape


# In[ ]:


hit_or_not.shape


# In[ ]:


#canciones.to_csv('../Proyecto-FInal-Spotify/CSV_primeros/canciones.csv', index=False)


# In[ ]:


#top2020_21.to_csv('../Proyecto-FInal-Spotify/CSV_primeros/top2020_21.csv', index=False)


# In[ ]:


#spotify_2022.to_csv('../Proyecto-FInal-Spotify/CSV_primeros/spotify_2022.csv', index=False)


# In[ ]:


#unpopular.to_csv('../Proyecto-FInal-Spotify/CSV_primeros/unpopular.csv', index=False)


# In[ ]:


#hit_or_not.to_csv('../Proyecto-FInal-Spotify/CSV_full/hit_or_not.csv', index=False)


# # Normalizar antes de Feature importances (Voy a quitar las columnas de los nombre de canciones y de los artistas porque ahora no es algo a lo que le vaya a dar importanci, igual en el futuro si meto variables para los artistas) aun asi creo que el nomre de la cancion no tiene ningun influencia.
# 

# # Voy a hacer label encoder de los nombres de artistas.

# In[71]:


hit_or_not_normal = hit_or_not.copy()


# In[72]:


hit_or_not_normal.drop(['track_name'],axis=1,inplace=True)


# In[73]:


hit_or_not_normal['artist_name'].value_counts()


# In[74]:


hit_or_not_normal.info(memory_usage='deep')


# In[75]:


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

hit_or_not_normal['artist_name']=LabelEncoder().fit_transform(hit_or_not_normal['artist_name'])


# In[76]:


hit_or_not_normal['artist_name'].value_counts()


# In[77]:


hit_or_not_normal.info(memory_usage='deep')


# In[78]:


#hit_or_not_normal = hit_or_not_normal.dropna()


# In[79]:


scaler=StandardScaler()


# In[80]:


hit_or_not_normal[['energy','danceability','loudness','acousticness','speechiness','liveness',                  'valence','tempo','duration','artist_name']] = scaler.fit_transform(hit_or_not_normal[['energy',                'danceability','loudness','acousticness','speechiness','liveness',                  'valence','tempo','duration','artist_name']])


# In[81]:


hit_or_not_normal.head()


# ## Salen valores mas altos que 1 y mas bajos que 0 entonces puede o seguro que esto es porque hay outliers voy a comprobar como salen las cosas sin quitarlos ahora y luego quitandolos ¿Igual deberia normalizar top_hit?????

# In[ ]:


hit_or_not_normal.info(memory_usage='deep')


# In[ ]:





# In[ ]:


hit_or_not_normal.info(memory_usage='deep')


# # Feature importances 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR


X=hit_or_not_normal.drop(columns=['top_hit'])
y=hit_or_not_normal.top_hit

dtr=DTR().fit(X, y)

dict(zip(X.columns, dtr.feature_importances_))

sum(dtr.feature_importances_)

X_norm=StandardScaler().fit_transform(X)
dtr=DTR().fit(X_norm, y)

dict(zip(X.columns, dtr.feature_importances_))

rfr=RFR(n_estimators=2000).fit(X_norm, y)

feat_imp = dict(zip(X.columns, rfr.feature_importances_))   


# In[ ]:


feat_imp


# In[83]:


X=hit_or_not_normal.drop(columns=['top_hit'])
y=hit_or_not_normal.top_hit


# In[ ]:


#hit_or_not_normal.isnull().index


# In[ ]:


#hit_or_not_normal[hit_or_not_normal['artist_name'].isna()]


# In[ ]:


#hit_or_not_normal.loc[2035]


# In[84]:


x_train, x_test, y_train, y_test = tts(X, y, train_size=0.8, test_size=0.2, random_state=42)


# In[ ]:


from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)


# In[ ]:


def predecir(x_train, x_test, y_train, y_test):
    #inicializamos todos los modelos que vamos a probar
    svr=SVR()
    rfr=RFR()
    etr=ETR()
    gbr=GBR()
    xgbr=XGBR()
    ctr=CTR()
    linreg=LinReg()
    lasso=Lasso()
    ridge=Ridge()
    elastic=ElasticNet()
    #los entrenamos
    svr.fit(x_train, y_train)
    rfr.fit(x_train, y_train)
    etr.fit(x_train, y_train)
    gbr.fit(x_train, y_train)
    xgbr.fit(x_train, y_train)
    ctr.fit(x_train, y_train, verbose=0)
    linreg.fit(x_train, y_train)
    lasso.fit(x_train, y_train)
    ridge.fit(x_train, y_train)
    elastic.fit(x_train, y_train)
    #predecimos
    y_pred1 = svr.predict(x_test)
    y_pred2 = rfr.predict(x_test)
    y_pred3 = etr.predict(x_test)
    y_pred4 = gbr.predict(x_test)
    y_pred5 = xgbr.predict(x_test)
    y_pred6 = ctr.predict(x_test)
    y_pred7 = linreg.predict(x_test)
    y_pred8 = lasso.predict(x_test)
    y_pred9 = ridge.predict(x_test)
    y_pred10 = elastic.predict(x_test)
    #calculamos error cuadrático medio (mse)
    mse1 = mse(y_test, y_pred1, squared=False)
    mse2 = mse(y_test, y_pred2, squared=False)
    mse3 = mse(y_test, y_pred3, squared=False)
    mse4 = mse(y_test, y_pred4, squared=False)
    mse5 = mse(y_test, y_pred5, squared=False)
    mse6 = mse(y_test, y_pred6, squared=False)
    mse7 = mse(y_test, y_pred7, squared=False)
    mse8 = mse(y_test, y_pred8, squared=False)
    mse9 = mse(y_test, y_pred9, squared=False)
    mse10 = mse(y_test, y_pred10, squared=False)
    #creamos una lista con todos los mse
    temp = [mse1, mse2, mse3, mse4, mse5, mse6, mse7, mse8, mse9, mse10]
    #pedimos a la función que nos devuelva el valor más bajo de mse
    minimo = min(temp)
    #le ponemos un mensajito para que quede más mono
    if minimo == mse1:
        return f'El mejor modelo es SVR con un mse de {mse1}'
    elif minimo == mse2:
        return f'El mejor modelo es RFR con un mse de {mse2}'
    elif minimo == mse3:
        return f'El mejor modelo es ETR con un mse de {mse3}'
    elif minimo == mse4:
        return f'El mejor modelo es GBR con un mse de {mse4}'
    elif minimo == mse5:
        return f'El mejor modelo es XGBR con un mse de {mse5}'
    elif minimo == mse6:
        return f'El mejor modelo es CTR con un mse de {mse6}'
    elif minimo == mse7:
        return f'El mejor modelo es LINREG con un mse de {mse7}'
    elif minimo == mse8:
        return f'El mejor modelo es LASSO con un mse de {mse8}'
    elif minimo == mse9:
        return f'El mejor modelo es RIDGE con un mse de {mse9}'
    elif minimo == mse10:
        return f'El mejor modelo es ELASTIC con un mse de {mse10}'


# In[ ]:


predecir(x_train, x_test, y_train, y_test)


# In[ ]:





# In[ ]:


ctr=CTR()


# In[ ]:


ctr.fit(x_train, y_train)


# In[ ]:


y_pred = ctr.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


indices = list(x_test.index)


# In[ ]:


probab = hit_or_not_normal.iloc[indices].reset_index(drop=True)


# In[ ]:


ufge


# In[ ]:


y_pred_train=ctr.predict(x_train)

mse_train = mse(y_train, y_pred_train, squared=False)


# In[ ]:


mse_train


# In[ ]:


mse_error = mse(y_test, y_pred, squared=False)


# In[ ]:


mse_error


# In[ ]:


mse_total = mse_error - mse_train

mse_total


# In[ ]:


type(y_pred6)


# In[ ]:


pred = pd.Series(y_pred) 


# In[ ]:


pred_test=pd.DataFrame(y_test).reset_index(drop=True)


# In[ ]:


probab['prediccion'] = pred.round(decimals = 0)


# In[ ]:


pred_test


# In[ ]:


len(y)


# In[ ]:


len(pred)


# In[ ]:


probab['fallo'] = probab['top_hit'] - probab['prediccion']
probab[probab['fallo']!=0]


# # El modelo funciona ahora voy a predecir los posibles exitos de las canciones que salieron la semana pasada de la playlist new music Friday.

# In[ ]:


#datos_spotipy_week_1 = pd.read_csv('../Proyecto-FInal-Spotify/CSV_full/data_week_2DEC.csv')


# In[ ]:


#datos_spotipy_week_1.drop_duplicates().shape==datos_spotipy_week_1.shape    # NO hay duplicados


# In[ ]:


#datos_spotipy_week_1.info(memory_usage='deep')


# In[ ]:


#datos_spotipy_week_1norm = datos_spotipy_week_1.copy()


# In[ ]:


#datos_spotipy_week_1norm['artist_name']=LabelEncoder().fit_transform(datos_spotipy_week_1norm['artist_name'])


# In[ ]:


#datos_spotipy_week_1norm[['energy','danceability','loudness','acousticness','speechiness','liveness',\
                  #'valence','tempo','duration','artist_name']] = scaler.fit_transform(datos_spotipy_week_1norm[['energy',\
                    #'danceability','loudness','acousticness','speechiness','liveness',\
                  #'valence','tempo','duration','artist_name']])


# In[ ]:


#datos_spotipy_week_1norm.head()


# In[ ]:


#datos_spotipy_week_1norm.drop(['track_name'],axis=1,inplace=True)


# In[ ]:


#y_pred = ctr.predict(datos_spotipy_week_1norm)


# In[ ]:


#datos_spotipy_week_1norm.shape


# In[ ]:


#datos_spotipy_week_1norm


# In[ ]:


#x_test.shape


# In[ ]:


#x_test


# In[ ]:





# In[ ]:


#pred1 = pd.Series(y_pred) 


# In[ ]:


#datos_spotipy_week_1['prediccion'] = pred1.round(decimals = 0)


# In[ ]:


#datos_spotipy_week_1.prediccion.value_counts()


# In[ ]:


#hit_or_not['artist_name'].value_counts()


# In[ ]:


#datos_spotipy_week_1


# In[ ]:




