# Ejecuta localmente las celdas de la Pregunta 3 (reutiliza lo definido en las Preguntas 1 y 2)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from pregunta2 import *
from pregunta2 import posicion, tfidf, matriz_tfidf
from itertools import product
from sklearn.preprocessing import MinMaxScaler

# 12. Puntajes normalizados de los cuatro modelos base para un usuario
MODELOS = ["Popularidad", "CF", "Generos", "Texto"]
pop_base = popularidad.reindex(movies["movieId"]).values.astype(float)
S_cf, S_txt = sim_cf.values, sim_texto.values
ids = movies["movieId"].values

def componentes(userId):
    # matriz (películas x 4) con min-max entre candidatos, y máscara de candidatos (no vistas)
    comp = np.zeros((len(movies), 4))
    comp[:, 0] = pop_base
    if userId in gustos_train:
        g = gustos_train[userId]
        idx, w = posicion[g.index].values, g.values
        comp[:, 1] = w @ S_cf[idx] / w.sum()
        comp[:, 2] = cosine_similarity((w @ matriz_tfidf[idx].toarray()).reshape(1, -1), matriz_tfidf).ravel()
        comp[:, 3] = w @ S_txt[idx] / w.sum()
    candidatos = np.ones(len(movies), dtype=bool)
    candidatos[posicion[list(vistas_train.get(userId, []))].values] = False
    comp[candidatos] = MinMaxScaler().fit_transform(comp[candidatos])
    return comp, candidatos

def ranking(comp, candidatos, pesos, k=10):
    scores = comp @ np.array(pesos)
    scores[~candidatos] = -np.inf                      # excluir vistas
    return ids[np.argsort(-scores, kind="stable")[:k]].tolist()

# 13. Métricas (Precision@K y NDCG@K) para seleccionar parámetros en val
def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def ndcg_at_k(recommended, relevant, k):
    relevance = np.array([1 if item in relevant else 0 for item in recommended[:k]])
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    dcg = np.sum(relevance / discounts)
    idcg = np.sum(np.sort(relevance)[::-1] / discounts)
    return dcg / idcg if idcg > 0 else 0

relevantes_val = val[val["rating"] >= 4].groupby("userId")["movieId"].apply(set).to_dict()

# muestra fija de usuarios de val (para que la búsqueda sea rápida)
usuarios_val = pd.Series(sorted(relevantes_val)).sample(1000, random_state=42).tolist()
cache_val = {u: componentes(u) for u in usuarios_val}

def evaluar_val(pesos, usuarios=usuarios_val, k=10):
    p, n = [], []
    for u in usuarios:
        rec = ranking(*cache_val[u], pesos, k)
        p.append(precision_at_k(rec, relevantes_val[u], k))
        n.append(ndcg_at_k(rec, relevantes_val[u], k))
    return np.mean(p), np.mean(n)

print("Modelos base en val (Precision@10, NDCG@10):")
for i, m in enumerate(MODELOS):
    print(m, np.round(evaluar_val(np.eye(4)[i]), 4))

# 14. Híbrido 1 (ponderado global): mismos pesos para todos, elegidos por NDCG@10 en val
pasos = np.round(np.arange(0.1, 0.71, 0.1), 1)          # cada modelo pesa al menos 0.1
grilla = [c for c in product(pasos, repeat=4) if abs(sum(c) - 1) < 1e-9]
print("\nCombinaciones de pesos probadas:", len(grilla))

ndcg_usuario = {}
for pesos in grilla:
    ndcg_usuario[pesos] = [ndcg_at_k(ranking(*cache_val[u], pesos, 10), relevantes_val[u], 10)
                           for u in usuarios_val]
tabla_ndcg = pd.DataFrame(ndcg_usuario, index=usuarios_val)   # filas: usuarios, columnas: pesos

pesos_h1 = tabla_ndcg.mean().idxmax()
print("Mejores 5 combinaciones en val (Popularidad, CF, Generos, Texto):")
display(tabla_ndcg.mean().sort_values(ascending=False).head(5).round(4).to_frame("NDCG@10 val"))
print("Pesos Híbrido 1:", dict(zip(MODELOS, map(float, pesos_h1))))

# 15. Híbrido 2 (adaptativo): pesos distintos según el tamaño del historial del usuario
n_gustadas = pd.Series({u: len(g) for u, g in gustos_train.items()})
corte_bajo, corte_alto = n_gustadas.quantile([1/3, 2/3]).values   # terciles calculados en train
print("\nCortes de historial (películas con rating >= 4 en train):", corte_bajo, corte_alto)

def segmento_historial(userId):
    n = n_gustadas.get(userId, 0)
    return "bajo" if n <= corte_bajo else "medio" if n <= corte_alto else "alto"

segmentos_val = pd.Series({u: segmento_historial(u) for u in usuarios_val})
pesos_h2 = {}
for s in ["bajo", "medio", "alto"]:
    media = tabla_ndcg[segmentos_val == s].mean()
    pesos_h2[s] = media.idxmax()
    print(f"Segmento {s} ({(segmentos_val == s).sum()} usuarios): pesos {dict(zip(MODELOS, map(float, pesos_h2[s])))}"
          f" | NDCG@10 val = {media.max():.4f} (con pesos del Híbrido 1: {media[pesos_h1]:.4f})")

# 16. Funciones de recomendación de los dos híbridos y comparación en val
def recomendar_hibrido_ponderado(userId, k=10):
    return ranking(*componentes(userId), pesos_h1, k)

def recomendar_hibrido_adaptativo(userId, k=10):
    return ranking(*componentes(userId), pesos_h2[segmento_historial(userId)], k)

filas = {}
for i, m in enumerate(MODELOS):
    filas[m] = evaluar_val(np.eye(4)[i])
filas["Hibrido ponderado"] = evaluar_val(pesos_h1)
p, n = [], []
for u in usuarios_val:
    rec = ranking(*cache_val[u], pesos_h2[segmentos_val[u]], 10)
    p.append(precision_at_k(rec, relevantes_val[u], 10))
    n.append(ndcg_at_k(rec, relevantes_val[u], 10))
filas["Hibrido adaptativo"] = (np.mean(p), np.mean(n))
print("\nComparación en val:")
display(pd.DataFrame(filas, index=["Precision@10", "NDCG@10"]).T.round(4))

# 17. Caso concreto: un usuario donde los dos híbridos dan rankings distintos
print("Segmento de los usuarios demo:", {u: (segmento_historial(u), int(n_gustadas[u])) for u in usuarios_demo})
bajos = [u for u in usuarios_val if segmentos_val[u] == "bajo"]
u_caso = min(bajos, key=lambda u: len(set(recomendar_hibrido_ponderado(u)) & set(recomendar_hibrido_adaptativo(u))))
rec_h1, rec_h2 = recomendar_hibrido_ponderado(u_caso), recomendar_hibrido_adaptativo(u_caso)
print(f"\nUsuario {u_caso} | segmento: {segmento_historial(u_caso)} | gustadas en train: {n_gustadas[u_caso]}")
print("Géneros que más le gustaron:",
      generos.loc[gustos_train[u_caso].index].str.split("|").explode().value_counts().head(3).to_dict())
display(pd.DataFrame({"Hibrido ponderado": titulos.loc[rec_h1].values,
                      "Hibrido adaptativo": titulos.loc[rec_h2].values}, index=range(1, 11)))
print("En común en el Top-10:", len(set(rec_h1) & set(rec_h2)), "de 10")

# puntajes normalizados de las películas que solo aparecen en uno de los dos rankings
comp, cand = componentes(u_caso)
detalle = pd.DataFrame(comp, index=ids, columns=MODELOS)
solo = [m for m in rec_h1 if m not in rec_h2] + [m for m in rec_h2 if m not in rec_h1]
detalle = detalle.loc[solo].round(3)
detalle.insert(0, "title", titulos.loc[solo].values)
detalle["aparece_en"] = ["ponderado" if m in rec_h1 else "adaptativo" for m in solo]
detalle["score_H1"] = (comp[posicion[solo].values] @ np.array(pesos_h1)).round(3)
detalle["score_H2"] = (comp[posicion[solo].values] @ np.array(pesos_h2["bajo"])).round(3)
display(detalle)
