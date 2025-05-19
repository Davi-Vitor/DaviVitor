#importando as bibliotecas
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



# -------------------- 1. Carrega dados --------------------
ratings_url = ("https://raw.githubusercontent.com/Davi-Vitor/DaviVitor/main/ml-100k/u.data")
movies_url = ("https://raw.githubusercontent.com/Davi-Vitor/DaviVitor/main/ml-100k/u.item")

ratings = pd.read_csv(
    ratings_url,
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],)

movies = pd.read_csv(
    movies_url,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "movie_title"],)



# -------------------- 2. Similaridade pelas avaliações --------------------
user_movie = (
    ratings.pivot_table(index="user_id", columns="item_id", values="rating")
    .fillna(0))

sim_rating = cosine_similarity(user_movie.T)
sim_rating = pd.DataFrame(
    sim_rating, index=user_movie.columns, columns=user_movie.columns)



# -------------------- 3. Similaridade pelos títulos --------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(movies["movie_title"])

sim_title = cosine_similarity(tfidf)
sim_title = pd.DataFrame(
    sim_title, index=movies["movie_id"], columns=movies["movie_id"])



# -------------------- 4. Combina as duas --------------------
alpha = 0.2  # peso para o nome do filme (0–1)
sim_combined = alpha * sim_title + (1 - alpha) * sim_rating



# -------------------- 5. Função de recomendação --------------------
def recomendar_filmes(movie_id: int, n: int = 5) -> pd.Series:
    """
    Retorna os n filmes mais similares ao movie_id,
    usando a matriz combinada (nome + ratings).
    """
    return (
        sim_combined[movie_id]
        .drop(movie_id)  # remove o próprio filme
        .nlargest(n))



# -------------------- 6. Exemplo de uso --------------------
if __name__ == "__main__":
    filme_base = int(input("Insira o id do filme de 1 a 1682 (veja a lista dos filmes na planilha excell do github): "))  # ID do filme de referência
    print(f"Top similares a {filme_base}:")
    for idx, score in recomendar_filmes(filme_base, 5).items():
        titulo = movies.loc[movies.movie_id == idx, "movie_title"].iat[0]
        print(f"{titulo} — similaridade {score:.2f}")