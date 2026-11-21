# Ejecuta localmente las celdas de la Pregunta 5 (reutiliza lo definido en las Preguntas 1 a 4)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from pregunta4 import *

# 24. Selección del híbrido con los resultados de validación (NDCG@10 en val)
ndcg_val = {
    "Hibrido ponderado": tabla_ndcg[pesos_h1].mean(),
    "Hibrido adaptativo": np.mean([tabla_ndcg.loc[u, pesos_h2[segmentos_val[u]]] for u in usuarios_val]),
}
hibrido_elegido = max(ndcg_val, key=ndcg_val.get)
print("NDCG@10 en val:", {m: round(float(v), 4) for m, v in ndcg_val.items()})
print("Híbrido elegido:", hibrido_elegido)

def pesos_elegidos(segmento):
    return pesos_h2[segmento] if hibrido_elegido == "Hibrido adaptativo" else pesos_h1

# 25. Funciones para los experimentos de sensibilidad
def componentes_historial(userId, gustadas):
    # igual que componentes(), pero con el historial que se le pase
    comp = np.zeros((len(movies), 4))
    comp[:, 0] = pop_base
    if gustadas is not None and len(gustadas) > 0:
        idx, w = posicion[gustadas.index].values, gustadas.values
        comp[:, 1] = w @ S_cf[idx] / w.sum()
        comp[:, 2] = cosine_similarity((w @ matriz_tfidf[idx].toarray()).reshape(1, -1), matriz_tfidf).ravel()
        comp[:, 3] = w @ S_txt[idx] / w.sum()
    candidatos = np.ones(len(movies), dtype=bool)
    candidatos[posicion[list(vistas_train.get(userId, []))].values] = False   # se excluye todo lo visto en train
    comp[candidatos] = MinMaxScaler().fit_transform(comp[candidatos])
    return comp, candidatos

def segmento_por_cantidad(n):
    return "bajo" if n <= corte_bajo else "medio" if n <= corte_alto else "alto"

def puntajes_experimento(userId, fraccion_historial=1.0):
    gustadas = gustos_train.get(userId)
    if gustadas is not None and fraccion_historial < 1:
        n = max(1, int(round(len(gustadas) * fraccion_historial)))
        gustadas = gustadas.sample(n, random_state=userId)        # el modelo solo "conoce" una parte
    comp, candidatos = componentes_historial(userId, gustadas)
    n_historial = 0 if gustadas is None else len(gustadas)
    scores = comp @ np.array(pesos_elegidos(segmento_por_cantidad(n_historial)))
    scores[~candidatos] = -np.inf
    return scores

# películas "populares": el 5% del catálogo con más ratings >= 4 en train
n_cabeza = int(0.05 * len(movies))
es_popular = np.zeros(len(movies), dtype=bool)
es_popular[posicion[popularidad.index[:n_cabeza]].values] = True
print("Películas consideradas populares:", n_cabeza)

def lista_top(scores, k, tope_populares=None):
    # tope_populares = fracción máxima de la lista que puede venir de las populares
    max_populares = k if tope_populares is None else int(np.ceil(tope_populares * k))
    lista, n_pop = [], 0
    for i in np.argsort(-scores, kind="stable"):
        if es_popular[i]:
            if n_pop >= max_populares:
                continue
            n_pop += 1
        lista.append(ids[i])
        if len(lista) == k:
            break
    return lista

# 26. Listas Top-5 y Top-10 de las tres configuraciones (usuarios de test)
FRACCION_HISTORIAL, TOPE_POPULARES = 0.3, 0.3            # fijados de antemano, no ajustados con test
CONFIGS = ["Original", "Menos historial (30%)", "Tope de populares (30%)"]
listas_exp = {c: {5: {}, 10: {}} for c in CONFIGS}

for u in relevantes_test:
    s_original = puntajes_experimento(u)
    s_reducido = puntajes_experimento(u, FRACCION_HISTORIAL)
    for k in [5, 10]:
        listas_exp["Original"][k][u] = lista_top(s_original, k)
        listas_exp["Menos historial (30%)"][k][u] = lista_top(s_reducido, k)
        listas_exp["Tope de populares (30%)"][k][u] = lista_top(s_original, k, TOPE_POPULARES)

# prueba automática: la configuración original reproduce el híbrido de la Pregunta 4
assert all(listas_exp["Original"][10][u] == recomendaciones[hibrido_elegido][u] for u in relevantes_test)
print("Prueba automática: la configuración original coincide con la Pregunta 4.")

# 27. Comparación con las seis métricas (K = 5 y K = 10)
def evaluar_listas(listas, k, usuarios):
    p, r, n, d, nov, pct, todas = [], [], [], [], [], [], []
    for u in usuarios:
        rec, relevantes = listas[u], relevantes_test[u]
        p.append(precision_at_k(rec, relevantes, k))
        r.append(recall_at_k(rec, relevantes, k))
        n.append(ndcg_at_k(rec, relevantes, k))
        d.append(diversidad_intralista(rec))
        nov.append(novedad(rec))
        pct.append(es_popular[posicion[rec].values].mean())
        todas.append(rec)
    return {"Precision": np.mean(p), "Recall": np.mean(r), "NDCG": np.mean(n),
            "Coverage": cobertura(todas, len(movies)), "Diversity": np.mean(d),
            "Novelty": np.mean(nov), "Pct_populares": np.mean(pct)}

usuarios_test = list(relevantes_test)
resultados_exp = {}
for k in [5, 10]:
    resultados_exp[k] = pd.DataFrame({c: evaluar_listas(listas_exp[c][k], k, usuarios_test) for c in CONFIGS}).T
    print(f"\nExperimentos en test, K = {k}")
    display(resultados_exp[k].round(4))

print("\nCambio frente a la configuración original (K = 10):")
display((resultados_exp[10] - resultados_exp[10].loc["Original"]).drop("Original").round(4))

# 28. Diferencias entre grupos de usuarios (por tamaño de historial en train, K = 10)
grupo_test = pd.Series({u: segmento_historial(u) for u in usuarios_test})
print("Usuarios por grupo:", grupo_test.value_counts().to_dict())

filas = {}
for c in CONFIGS:
    for g in ["bajo", "medio", "alto"]:
        filas[(c, g)] = evaluar_listas(listas_exp[c][10], 10, grupo_test.index[grupo_test == g])
por_grupo = pd.DataFrame(filas).T
display(por_grupo.round(4))

precision_grupo = por_grupo["Precision"].unstack()[["bajo", "medio", "alto"]].loc[CONFIGS]
novedad_grupo = por_grupo["Novelty"].unstack()[["bajo", "medio", "alto"]].loc[CONFIGS]
print("Cambio relativo de Precision@10 frente al original, por grupo (%):")
display(((precision_grupo / precision_grupo.loc["Original"] - 1) * 100).drop("Original").round(1))
print("Cambio de Novelty@10 frente al original, por grupo:")
display((novedad_grupo - novedad_grupo.loc["Original"]).drop("Original").round(3))

# 29. Gráficos de los experimentos
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
precision_grupo.T.plot(kind="bar", ax=ejes[0], title="Precision@10 por grupo de historial", rot=0)
novedad_grupo.T.plot(kind="bar", ax=ejes[1], title="Novelty@10 por grupo de historial", rot=0)
plt.tight_layout(); plt.show()
