# Ejecuta localmente las celdas de la Pregunta 6 (reutiliza lo definido en las Preguntas 1 a 5)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from pregunta5 import *

# 30. Candidatos para el piloto y sus métricas en test (K = 10)
NOMBRE_TOPE = "Hibrido adaptativo + tope"
candidatos_piloto = {nombre: recomendaciones[nombre] for nombre in modelos}
candidatos_piloto[NOMBRE_TOPE] = listas_exp["Tope de populares (30%)"][10]

metricas = pd.DataFrame({c: evaluar_listas(listas, 10, usuarios_test)
                         for c, listas in candidatos_piloto.items()}).T
metricas["Cola_larga"] = 1 - metricas["Pct_populares"]      # % de recomendaciones fuera de las populares
display(metricas.round(4))

# 31. Matriz de decisión ponderada
pesos_decision = {"Precision": 0.30, "NDCG": 0.10,                          # relevancia: 40%
                  "Cola_larga": 0.20, "Novelty": 0.15, "Coverage": 0.10,    # descubrimiento: 45%
                  "Diversity": 0.15}                                        # diversidad: 15%
RETENCION_MINIMA = 0.70      # "sin sacrificar excesivamente": conservar 70% de la mejor Precision@10

def matriz_decision(pesos):
    criterios = list(pesos)
    normalizada = metricas[criterios] / metricas[criterios].max()           # cada criterio de 0 a 1
    matriz = normalizada * pd.Series(pesos)
    matriz["Puntaje"] = matriz.sum(axis=1)
    matriz["Retencion_precision"] = metricas["Precision"] / metricas["Precision"].max()
    matriz["Cumple_relevancia"] = matriz["Retencion_precision"] >= RETENCION_MINIMA
    return matriz.sort_values("Puntaje", ascending=False)

matriz = matriz_decision(pesos_decision)
display(matriz.round(3))
modelo_piloto = matriz[matriz["Cumple_relevancia"]]["Puntaje"].idxmax()
print("Modelo recomendado para el piloto:", modelo_piloto)

# 32. Sensibilidad de la decisión a los pesos
escenarios = {
    "Base (descubrimiento 45%)": pesos_decision,
    "Más relevancia (60%)": {"Precision": 0.45, "NDCG": 0.15, "Cola_larga": 0.10,
                             "Novelty": 0.10, "Coverage": 0.05, "Diversity": 0.15},
    "Más descubrimiento (60%)": {"Precision": 0.20, "NDCG": 0.05, "Cola_larga": 0.30,
                                 "Novelty": 0.20, "Coverage": 0.10, "Diversity": 0.15},
}
filas = {}
for nombre, pesos in escenarios.items():
    m = matriz_decision(pesos)
    validos = m[m["Cumple_relevancia"]]["Puntaje"]
    filas[nombre] = {"Ganador con requisito": validos.idxmax(), "Puntaje": round(validos.max(), 3),
                     "Segundo": validos.index[1], "Puntaje segundo": round(validos.iloc[1], 3),
                     "Ganador sin requisito": m["Puntaje"].idxmax()}
display(pd.DataFrame(filas).T)

# 33. Defensa (1): una recomendación concreta
u = usuarios_demo[0]
scores_u = puntajes_experimento(u)
sin_tope, con_tope = lista_top(scores_u, 10), lista_top(scores_u, 10, TOPE_POPULARES)
print(f"Usuario {u} | géneros que más le gustaron:",
      generos.loc[gustos_train[u].index].str.split("|").explode().value_counts().head(3).to_dict())
display(pd.DataFrame({
    "Hibrido adaptativo": titulos.loc[sin_tope].values,
    "popular": es_popular[posicion[sin_tope].values],
    "Hibrido adaptativo + tope": titulos.loc[con_tope].values,
    "popular ": es_popular[posicion[con_tope].values],
    "ratings >= 4 en train": popularidad.loc[con_tope].values,
}, index=range(1, 11)))

# 34. Defensa (2): prueba automática del modelo elegido
for usuario, lista in candidatos_piloto[NOMBRE_TOPE].items():
    assert len(lista) == 10 and len(set(lista)) == 10                       # 10 películas distintas
    assert not set(lista) & vistas_train[usuario]                           # ninguna vista en train
    assert es_popular[posicion[lista].values].sum() <= 3                    # máximo 3 populares
print(f"Prueba automática superada para {len(candidatos_piloto[NOMBRE_TOPE])} usuarios: "
      "10 películas distintas, ninguna vista en train y como máximo 3 populares por lista.")

# 35. Defensa (3): un resultado experimental inesperado
original, tope = metricas.loc["Hibrido adaptativo"], metricas.loc[NOMBRE_TOPE]
print("Al limitar las películas populares:")
print(f"  Novelty@10   sube de {original['Novelty']:.4f} a {tope['Novelty']:.4f}")
print(f"  Coverage@10  sube de {original['Coverage']:.4f} a {tope['Coverage']:.4f}")
print(f"  Diversity@10 BAJA de {original['Diversity']:.4f} a {tope['Diversity']:.4f}")
cola = ~es_popular
print("Géneros distintos entre las 194 populares:", generos[es_popular].str.split("|").explode().nunique(),
      "| entre el resto:", generos[cola].str.split("|").explode().nunique())
print("Similitud de géneros promedio dentro de las listas (1 - Diversity):",
      f"original {1 - original['Diversity']:.4f} | con tope {1 - tope['Diversity']:.4f}")
