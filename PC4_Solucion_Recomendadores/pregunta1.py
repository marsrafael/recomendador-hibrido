# Ejecuta localmente las celdas de la Pregunta 1 (mismo código que en Colab; display -> print)
import sys, shutil
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
from io import BytesIO

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
display = lambda x: print(x.to_string(), "\n")

# Descarga (celda del profesor)
DATA_DIR = Path(sys.argv[1])
DATA_DIR.mkdir(parents=True, exist_ok=True)
URL_MOVIELENS = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ARCHIVOS = ("ratings.dat", "movies.dat")
if not all((DATA_DIR / n).is_file() for n in ARCHIVOS):
    peticion = Request(URL_MOVIELENS, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(peticion, timeout=120) as respuesta:
        contenido = respuesta.read()
    with ZipFile(BytesIO(contenido)) as zip_file:
        for n in ARCHIVOS:
            with zip_file.open(f"ml-1m/{n}") as origen, (DATA_DIR / n).open("wb") as destino:
                shutil.copyfileobj(origen, destino)

# 2. Cargar los datos
ratings = pd.read_csv(DATA_DIR / "ratings.dat", sep="::", engine="python", encoding="latin-1",
                      names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv(DATA_DIR / "movies.dat", sep="::", engine="python", encoding="latin-1",
                     names=["movieId", "title", "genres"])
print("Ratings:", ratings.shape, "| Películas:", movies.shape)
print("Usuarios:", ratings["userId"].nunique(), "| Películas con rating:", ratings["movieId"].nunique())
display(ratings.head())
display(movies.head())

# 3. Dividir en train, val y test
train, temp = train_test_split(ratings, test_size=0.30, random_state=42, stratify=ratings["userId"])
val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["userId"])
print("train:", train.shape, "| val:", val.shape, "| test:", test.shape)

vistas_train = train.groupby("userId")["movieId"].apply(set).to_dict()
gustos_train = {u: g.set_index("movieId")["rating"]
                for u, g in train[train["rating"] >= 4].groupby("userId")}

# 4. Modelo 1: popularidad
popularidad = (train[train["rating"] >= 4].groupby("movieId").size()
               .reindex(movies["movieId"], fill_value=0)
               .sort_values(ascending=False, kind="stable"))

def recomendar_popularidad(userId, k=10):
    vistas = vistas_train.get(userId, set())
    candidatos = popularidad[~popularidad.index.isin(vistas)]
    return candidatos.head(k).index.tolist()

# 5. Modelo 2: TF-IDF sobre títulos
movies["texto"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
vectorizador = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                               ngram_range=(1, 2), stop_words="english")
X_texto = vectorizador.fit_transform(movies["texto"])
sim_texto = pd.DataFrame(cosine_similarity(X_texto),
                         index=movies["movieId"], columns=movies["movieId"])

def recomendar_texto(userId, k=10):
    if userId not in gustos_train:
        return recomendar_popularidad(userId, k)
    gustadas = gustos_train[userId]
    scores = sim_texto[gustadas.index].dot(gustadas.values) / gustadas.sum()
    scores.loc[list(vistas_train[userId])] = -np.inf
    return scores.sort_values(ascending=False).head(k).index.tolist()

# 6. Top-5 y Top-10 para dos usuarios
titulos = movies.set_index("movieId")["title"]
usuarios_demo = [1, 2]
for u in usuarios_demo:
    print(f"\n===== Usuario {u} | vistas en train: {len(vistas_train[u])} =====")
    print("Algunas películas que le gustaron (rating >= 4):")
    display(titulos.loc[gustos_train[u].sort_values(ascending=False).head(8).index].to_frame("title"))
    for k in [5, 10]:
        tabla = pd.DataFrame({
            "Popularidad": titulos.loc[recomendar_popularidad(u, k)].values,
            "TF-IDF títulos": titulos.loc[recomendar_texto(u, k)].values
        }, index=range(1, k + 1))
        print(f"Top-{k}")
        display(tabla)

# 7. Diferencias entre usuarios
u1, u2 = usuarios_demo
for k in [5, 10]:
    comun_pop = len(set(recomendar_popularidad(u1, k)) & set(recomendar_popularidad(u2, k)))
    comun_txt = len(set(recomendar_texto(u1, k)) & set(recomendar_texto(u2, k)))
    print(f"Top-{k} | en común entre usuarios -> Popularidad: {comun_pop}/{k} | TF-IDF: {comun_txt}/{k}")
