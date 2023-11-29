from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    message = """
        <p>Bienvenido a la API de Juan Cruz Yezzi.</p>

        <p>Puedes probar los endpoints cambiando la url con las siguientes palabras:</p>

        <p>* Devuelve el año con mas horas jugadas para el género dado.</p>
        <p>/playtime_genre/(género, ej: action)</p>

        <p>* Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.</p> 
        <p>/user_for_genre/(género, ej: adventure)</p>

        <p>* Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.</p>  
        <p>/users_recommend/(año, ej: 2014)</p>

        <p>* Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.</P>
        <p>/users_not_recommend/(año, ej: 2015)</p>

        <p>* Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.</p>
        <p>/sentiment_analysis/(año, ej: 2016)</P> 

        <p>* Recomendación de juegos similares en base a un id por género y etiquetas.</p>
        <p>/games_recommend/(id de juego, ej: 643980)</p>
        """
        
    return HTMLResponse(content=message, status_code=200)

#---------------------------------------------------------------------------------------------------------------------------

#Devuelve el año con mas horas jugadas para el género dado.
@app.get("/playtime_genre/{genero}")
def PlayTimeGenre(genero: str):
    
    # Abrir el DataSet
    df = pd.read_parquet("_src/Datasets/merge1.parquet")
    
    # Filtrar el DataFrame para el género especificado
    df_genre = df[df['genres'] == genero]
    
    # Agrupar por el año de lanzamiento y calcular las horas totales jugadas por año
    genre_playtime_by_year = df_genre.groupby('release_date')['playtime_forever'].sum()
    
    # Encontrar el año con más horas jugadas
    year_with_most_playtime = int(genre_playtime_by_year.idxmax())
    
    return {"Año de lanzamiento con más horas jugadas para " + genero: year_with_most_playtime}

#---------------------------------------------------------------------------------------------------------------------------

#Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
@app.get("/user_for_genre/{genero}")
def UserForGenre(genero: str):
    
    # Abrir el DataSet
    df = pd.read_parquet("_src/Datasets/merge2.parquet")
    
    # Filtrar el DataFrame por el género dado
    genre_df = df[df['genres'] == genero]
    
    # Agrupar por usuario y año, sumar las horas jugadas y resetear el índice
    grouped = genre_df.groupby(['user_id', 'release_date'])['playtime_forever'].sum().reset_index()
    
    # Encontrar al usuario con más horas jugadas
    max_user = grouped.groupby('user_id')['playtime_forever'].sum().idxmax()
    
    # Filtrar el DataFrame para obtener solo las filas del usuario con más horas jugadas
    max_user_df = grouped[grouped['user_id'] == max_user]
    
    # Crear un diccionario con el resultado
    result = {
        "Usuario con más horas jugadas para {}".format(genero): max_user,
        "Horas jugadas": max_user_df[['release_date', 'playtime_forever']].rename(columns={'release_date': 'Año', 'playtime_forever': 'Horas'}).to_dict(orient='records')
    }
    
    return result

#---------------------------------------------------------------------------------------------------------------------------

#Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
@app.get("/users_recommend/{anio}")
def UsersRecommend(anio: int):
    
    # Abrir el DataSet
    df = pd.read_parquet("_src/Datasets/merge3.parquet")
    
    # Filtra las revisiones para el año dado y las que cumplen con las condiciones
    reviews_filtradas = df[(df['posted'].dt.year == anio) & (df['recommend'] == True) & (df['sentiment_analysis'] >= 1)]

    # Agrupa las revisiones por el nombre del juego y cuenta las recomendaciones
    top_juegos = reviews_filtradas.groupby('item_name')['sentiment_analysis'].sum().reset_index()

    # Ordena los juegos por la cantidad de recomendaciones en orden descendente
    top_juegos = top_juegos.sort_values(by='sentiment_analysis', ascending=False)

    # Toma los primeros 3 juegos y crea una lista de diccionarios para el retorno
    top_3_juegos = top_juegos.head(3).to_dict(orient='records')

    # Crea la lista con formato deseado
    resultado = [{"Puesto {}: {}".format(i+1, juego['item_name'])} for i, juego in enumerate(top_3_juegos)]

    return resultado

#---------------------------------------------------------------------------------------------------------------------------

#Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
@app.get("/users_not_recommend/{anio}")
def UsersNotRecommend(anio: int):
    
    # Abrir el DataSet
    df = pd.read_parquet("_src/Datasets/merge4.parquet")
    
    # Filtra las revisiones para el año dado y las que cumplen con las condiciones
    reviews_filtradas = df[(df['posted'].dt.year == anio) & (df['recommend'] == False) & (df['sentiment_analysis'] == 0)]

    # Agrupa las revisiones por el nombre del juego y cuenta las recomendaciones
    top_juegos = reviews_filtradas.groupby('item_name').size().reset_index(name='count')

    # Ordena los juegos por la cantidad de recomendaciones en orden ascendente
    top_juegos = top_juegos.sort_values(by='count', ascending=False)

    # Toma los primeros 3 juegos y crea una lista de diccionarios para el retorno
    top_3_juegos = top_juegos.head(3).to_dict(orient='records')

    # Crea la lista con formato deseado
    resultado = [{"Puesto {}: {}".format(i+1, juego['item_name'])} for i, juego in enumerate(top_3_juegos)]

    return resultado

#---------------------------------------------------------------------------------------------------------------------------

#Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
@app.get("/sentiment_analysis/{anio}")
def sentiment_analysis(anio: int):

    # Abrir el DataSet
    df = pd.read_parquet("_src/Datasets/merge5.parquet")

    # Filtra las revisiones para el año dado
    reviews_anio = df[df['posted'].dt.year == anio]

    # Cuenta la cantidad de cada tipo de sentimiento
    sentiment_counts = reviews_anio['sentiment_analysis'].astype(int).value_counts()

    # Crea un diccionario con los resultados
    results = {
        "Negative": int(sentiment_counts.get(0, 0)),  # Si no hay 0, devuelve 0
        "Neutral": int(sentiment_counts.get(1, 0)),   # Si no hay 1, devuelve 0
        "Positive": int(sentiment_counts.get(2, 0))   # Si no hay 2, devuelve 0
    }

    return results

#---------------------------------------------------------------------------------------------------------------------------

#Recomendación de juegos similares en base a un id por género y etiquetas.
@app.get("/games_recommend/{product_id}")
def recomendacion_juego(product_id: int):
    
    # Abrir el DataSet
    data = pd.read_csv("_src/Datasets/steam.csv")

    # Crear una matriz TF-IDF para calcular la similitud del coseno
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

    # Calcular la similitud del coseno entre todos los juegos
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Obtener el índice del juego en función del ID de producto
    index = data[data['item_id'] == product_id].index[0]

    # Obtener puntuaciones de similitud del coseno para todos los juegos
    sim_scores = list(enumerate(cosine_sim[index]))

    # Ordenar los juegos según las puntuaciones de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los 5 juegos más similares (excluyendo el juego de entrada)
    top_games_indices = [i[0] for i in sim_scores[1:6]]
    top_games = data['item_name'].iloc[top_games_indices]

    return top_games.tolist()
