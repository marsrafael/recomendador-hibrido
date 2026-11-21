# Ejecuta localmente las celdas de la Pregunta 2 (reutiliza lo definido en la Pregunta 1)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):   # no repetir el output de la P1
    from pregunta1 import *

# 8. Modelo 3: filtrado colaborativo ítem-ítem con similitud coseno
matriz = train.pivot_table(index="userId", columns="movieId", values="rating")
matriz_items = matriz.T.reindex(movies["movieId"]).fillna(0)     # mismas películas que el catálogo
sim_cf = pd.DataFrame(cosine_similarity(matriz_items),
                      index=matriz_items.index, columns=matriz_items.index)
print("Matriz usuario-película:", matriz.shape, "| Similitud CF:", sim_cf.shape)

def puntajes_historial(userId, similitud):
    # promedio ponderado de similitud con las películas que le gustaron
    gustadas = gustos_train[userId]
    scores = similitud[gustadas.index].dot(gustadas.values) / gustadas.sum()
    scores.loc[list(vistas_train[userId])] = -np.inf             # excluir vistas
    return scores

def recomendar_colaborativo(userId, k=10):
    if userId not in gustos_train:
        return recomendar_popularidad(userId, k)
    scores = puntajes_historial(userId, sim_cf)
    return scores.sort_values(ascending=False, kind="stable").head(k).index.tolist()

# 9. Modelo 4: contenido por géneros
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split("|"), token_pattern=None)
matriz_tfidf = tfidf.fit_transform(movies["genres"].fillna(""))
posicion = pd.Series(range(len(movies)), index=movies["movieId"])   # fila de cada película
print("Géneros:", len(tfidf.get_feature_names_out()))

def puntajes_generos(userId):
    # perfil del usuario = suma ponderada de los géneros de las películas que le gustaron
    gustadas = gustos_train[userId]
    perfil = gustadas.values @ matriz_tfidf[posicion[gustadas.index].values].toarray()
    scores = pd.Series(cosine_similarity(perfil.reshape(1, -1), matriz_tfidf).ravel(),
                       index=movies["movieId"])
    scores.loc[list(vistas_train[userId])] = -np.inf             # excluir vistas
    return scores

def recomendar_generos(userId, k=10):
    if userId not in gustos_train:
        return recomendar_popularidad(userId, k)
    scores = puntajes_generos(userId).round(6)
    # muchas películas empatan (mismos géneros): desempate aleatorio con semilla fija por usuario
    desempate = np.random.RandomState(userId).rand(len(scores))
    orden = pd.DataFrame({"score": scores, "desempate": desempate}, index=scores.index)
    return orden.sort_values(["score", "desempate"], ascending=False).head(k).index.tolist()

# 10. Top-5 y Top-10 de ambos modelos para dos usuarios
generos = movies.set_index("movieId")["genres"]

for u in usuarios_demo:
    print(f"\n===== Usuario {u} | vistas en train: {len(vistas_train[u])} =====")
    print("Géneros más frecuentes entre las películas que le gustaron:")
    display(generos.loc[gustos_train[u].index].str.split("|").explode().value_counts().head(5).to_frame("películas"))
    for k in [5, 10]:
        rec_cf, rec_gen = recomendar_colaborativo(u, k), recomendar_generos(u, k)
        tabla = pd.DataFrame({
            "Colaborativo": titulos.loc[rec_cf].values,
            "Géneros CF": generos.loc[rec_cf].values,
            "Contenido (géneros)": titulos.loc[rec_gen].values,
            "Géneros CB": generos.loc[rec_gen].values
        }, index=range(1, k + 1))
        print(f"Top-{k}")
        display(tabla)

# 11. Comparación de los dos enfoques
filas = []
for u in usuarios_demo:
    for k in [5, 10]:
        rec_cf, rec_gen = recomendar_colaborativo(u, k), recomendar_generos(u, k)
        filas.append({"usuario": u, "k": k,
                      "en_comun_CF_vs_generos": len(set(rec_cf) & set(rec_gen)),
                      "popularidad_media_CF": popularidad.loc[rec_cf].mean(),
                      "popularidad_media_generos": popularidad.loc[rec_gen].mean()})
display(pd.DataFrame(filas).round(1))

# Empates del modelo de géneros: películas con el mismo puntaje máximo
for u in usuarios_demo:
    s = puntajes_generos(u).round(6)
    print(f"Usuario {u}: {(s == s.max()).sum()} películas empatan en el primer lugar por géneros "
          f"(géneros: {generos.loc[s.idxmax()]})")

for k in [5, 10]:
    comun_cf = len(set(recomendar_colaborativo(u1, k)) & set(recomendar_colaborativo(u2, k)))
    comun_gen = len(set(recomendar_generos(u1, k)) & set(recomendar_generos(u2, k)))
    print(f"Top-{k} | en común entre usuarios -> Colaborativo: {comun_cf}/{k} | Géneros: {comun_gen}/{k}")
