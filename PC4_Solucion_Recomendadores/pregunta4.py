# Ejecuta localmente las celdas de la Pregunta 4 (reutiliza lo definido en las Preguntas 1 a 3)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from pregunta3 import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 18. Métricas que faltan: Recall, diversidad intralista, novedad y cobertura
def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

S_gen = cosine_similarity(matriz_tfidf)                 # similitud de géneros entre películas

def diversidad_intralista(lista, similitud=S_gen, filas=posicion):
    # 1 - similitud promedio entre pares de la lista
    idx = filas[lista].values
    pares = similitud[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]
    return 1 - pares.mean()

# probabilidad de que un usuario haya visto la película en train (suavizada)
prob_pelicula = ((train.groupby("movieId").size().reindex(movies["movieId"], fill_value=0) + 1)
                 / (train["userId"].nunique() + 1))

def novedad(lista, prob=prob_pelicula):
    return float(np.mean(-np.log2(prob.loc[lista].values)))

def cobertura(listas, total_catalogo):
    distintas = set(m for lista in listas for m in lista)
    return len(distintas) / total_catalogo

# 19. Pruebas automáticas de las métricas con ejemplos pequeños
assert precision_at_k([1, 2, 3, 4, 5], {1, 3}, 5) == 0.4            # 2 aciertos de 5
assert recall_at_k([1, 2, 3, 4, 5], {1, 3}, 5) == 1.0               # recupera los 2 relevantes
assert recall_at_k([1, 2, 3, 4, 5], {1, 9}, 5) == 0.5               # recupera 1 de 2
assert recall_at_k([1, 2], set(), 2) == 0                           # sin relevantes
assert ndcg_at_k([1, 3, 7], {1, 3}, 3) == 1.0                       # aciertos arriba: orden perfecto
assert abs(ndcg_at_k([7, 1], {1}, 2) - 1 / np.log2(3)) < 1e-9       # acierto en posición 2
assert ndcg_at_k([7, 8], {1}, 2) == 0                               # sin aciertos
assert cobertura([[1, 2], [2, 3]], 6) == 0.5                        # 3 distintas de 6

sim_prueba = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
filas_prueba = pd.Series([0, 1, 2], index=[10, 20, 30])            # catálogo de prueba
assert diversidad_intralista([10, 20], sim_prueba, filas_prueba) == 0.0   # películas idénticas
assert diversidad_intralista([10, 30], sim_prueba, filas_prueba) == 1.0   # películas sin nada en común

prob_prueba = pd.Series({10: 0.5, 20: 0.25})
assert novedad([10], prob_prueba) == 1.0                            # -log2(0.5)
assert novedad([20], prob_prueba) > novedad([10], prob_prueba)      # menos popular = más novedosa
print("Todas las pruebas automáticas pasaron.")

# 20. Recomendaciones Top-10 de los seis modelos para los usuarios de test
relevantes_test = test[test["rating"] >= 4].groupby("userId")["movieId"].apply(set).to_dict()
print("Usuarios evaluados en test:", len(relevantes_test))

modelos = {
    "Popularidad": recomendar_popularidad,
    "Texto TF-IDF": recomendar_texto,
    "Colaborativo": recomendar_colaborativo,
    "Generos": recomendar_generos,
    "Hibrido ponderado": recomendar_hibrido_ponderado,
    "Hibrido adaptativo": recomendar_hibrido_adaptativo,
}
recomendaciones = {nombre: {u: modelo(u, 10) for u in relevantes_test}
                   for nombre, modelo in modelos.items()}

# 21. Evaluación en test para K = 5 y K = 10
def evaluar_modelo(nombre, k):
    p, r, n, d, nov, listas = [], [], [], [], [], []
    for u, relevantes in relevantes_test.items():
        rec = recomendaciones[nombre][u][:k]            # Top-5 = primeras 5 del Top-10
        p.append(precision_at_k(rec, relevantes, k))
        r.append(recall_at_k(rec, relevantes, k))
        n.append(ndcg_at_k(rec, relevantes, k))
        d.append(diversidad_intralista(rec))
        nov.append(novedad(rec))
        listas.append(rec)
    return {"Precision": np.mean(p), "Recall": np.mean(r), "NDCG": np.mean(n),
            "Coverage": cobertura(listas, len(movies)), "Diversity": np.mean(d), "Novelty": np.mean(nov)}

resultados = {}
for k in [5, 10]:
    resultados[k] = pd.DataFrame({nombre: evaluar_modelo(nombre, k) for nombre in modelos}).T
    print(f"\nResultados en test, K = {k}")
    display(resultados[k].round(4))

# 22. Comparación consolidada (K = 5 y K = 10 en una sola tabla)
results = pd.concat(resultados, axis=1)
results.columns = [f"{m}@{k}" for k, m in results.columns]
display(results.round(4))

resultados[10][["Precision", "Recall", "NDCG"]].plot(kind="bar", figsize=(9, 4), title="Relevancia en test (K = 10)")
plt.tight_layout(); plt.savefig("p4_relevancia.png"); plt.close()
resultados[10][["Coverage", "Diversity"]].plot(kind="bar", figsize=(9, 4), title="Cobertura y diversidad en test (K = 10)")
plt.tight_layout(); plt.savefig("p4_cobertura.png"); plt.close()

# 23. Caso: mayor relevancia con costo en otra dimensión
r10 = resultados[10]
mas_preciso, mas_novedoso = r10["Precision"].idxmax(), r10["Novelty"].idxmax()
print(f"\nMás preciso: {mas_preciso} | Más novedoso: {mas_novedoso}")
display(r10.loc[[mas_preciso, mas_novedoso]].round(4))
print(f"{mas_preciso} tiene {r10.loc[mas_preciso, 'Precision'] / r10.loc[mas_novedoso, 'Precision']:.1f} veces "
      f"la Precision@10 de {mas_novedoso}, pero recomienda solo "
      f"{r10.loc[mas_preciso, 'Coverage'] * len(movies):.0f} películas distintas frente a "
      f"{r10.loc[mas_novedoso, 'Coverage'] * len(movies):.0f}.")
print("Correlación entre Precision@10 y Novelty@10 de los seis modelos:",
      round(r10["Precision"].corr(r10["Novelty"]), 3))

plt.figure(figsize=(6, 4))
plt.scatter(r10["Novelty"], r10["Precision"])
for nombre in r10.index:
    plt.annotate(nombre, (r10.loc[nombre, "Novelty"], r10.loc[nombre, "Precision"]))
plt.xlabel("Novelty@10"); plt.ylabel("Precision@10"); plt.title("Relevancia vs novedad")
plt.tight_layout(); plt.savefig("p4_tradeoff.png"); plt.close()

pd.to_pickle({"resultados": resultados, "recomendaciones": recomendaciones}, sys.argv[1] + "/../p4.pkl")
