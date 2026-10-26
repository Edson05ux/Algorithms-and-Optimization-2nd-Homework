"""Ejercicio 5: Razón de aproximación con un árbol de expansión mínima (MST).

PREGUNTA: ¿Qué tan cerca estuvo la razón observada del límite teórico 2?
RESPUESTA: En 100 instancias de 9 ciudades, el promedio fue 1.134 y el máximo 1.327.
Todas las razones estuvieron por debajo de 2. Ese valor es una cota superior, no un
resultado que el algoritmo tenga que alcanzar en cada instancia.

PREGUNTA: ¿Qué efecto tuvo aplicar 2-opt al tour MST?
RESPUESTA: El costo promedio bajó de 309.97 con MST puro a 273.99 con MST + 2-opt,
una reducción cercana al 11.6%. Búsqueda local promedió 273.96 y recocido simulado
274.63. En estas pruebas, 2-opt mejoró claramente el MST y quedó casi al nivel de
búsqueda local. Las gráficas muestran las razones y comparan los cuatro métodos.
"""

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


INSTANCE_COUNT = 100
CITY_COUNT = 9
SA_ITERATIONS = 600
SEED = 2026


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[math.dist(a, b) for b in points] for a in points]


def tour_length(distances: list[list[float]], tour: list[int]) -> float:
    return sum(distances[tour[i]][tour[(i + 1) % len(tour)]]
               for i in range(len(tour)))


def mst_prim(distances: list[list[float]]) -> dict[int, list[int]]:
    """Prim desde la ciudad 0; devuelve el árbol como {nodo: hijos}."""
    n = len(distances)
    best = {v: (distances[0][v], 0) for v in range(1, n)}
    children = {v: [] for v in range(n)}
    while best:
        vertex = min(best, key=lambda node: best[node][0])
        parent = best.pop(vertex)[1]
        children[parent].append(vertex)
        for node in best:
            if distances[vertex][node] < best[node][0]:
                best[node] = (distances[vertex][node], vertex)
    return children


def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    return [root] + [node for child in children[root]
                     for node in preorder(children, child)]


def exact_tsp_cost(distances: list[list[float]]) -> float:
    """Óptimo exacto Held-Karp, fijando la ciudad 0 como inicio."""
    n = len(distances)
    dp = {}
    for city in range(1, n):
        dp[(1 << (city - 1), city)] = distances[0][city]
    for mask in range(1, 1 << (n - 1)):
        for city in range(1, n):
            bit = 1 << (city - 1)
            if not mask & bit or mask == bit:
                continue
            previous_mask = mask ^ bit
            dp[(mask, city)] = min(
                dp[(previous_mask, previous)] + distances[previous][city]
                for previous in range(1, n)
                if previous_mask & (1 << (previous - 1))
            )
    full_mask = (1 << (n - 1)) - 1
    return min(dp[(full_mask, city)] + distances[city][0]
               for city in range(1, n))


def two_opt_descent(distances: list[list[float]], route: list[int]) -> list[int]:
    route = route[:]
    while True:
        best_route = route
        best_cost = tour_length(distances, route)
        for i in range(len(route) - 1):
            for k in range(i + 1, len(route)):
                if i == 0 and k == len(route) - 1:
                    continue
                candidate = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
                candidate_cost = tour_length(distances, candidate)
                if candidate_cost < best_cost - 1e-12:
                    best_route, best_cost = candidate, candidate_cost
        if best_route is route:
            return route
        route = best_route


def simulated_annealing(distances: list[list[float]], initial: list[int],
                        rng: random.Random) -> list[int]:
    current = initial[:]
    current_cost = tour_length(distances, current)
    best, best_cost = current[:], current_cost
    pairs = [(i, k) for i in range(len(current) - 1)
             for k in range(i + 1, len(current))
             if not (i == 0 and k == len(current) - 1)]
    for iteration in range(1, SA_ITERATIONS + 1):
        i, k = rng.choice(pairs)
        candidate = current[:i] + current[i:k + 1][::-1] + current[k + 1:]
        candidate_cost = tour_length(distances, candidate)
        temperature = 5.0 * 0.99 ** iteration
        delta = candidate_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current, current_cost = candidate, candidate_cost
        if current_cost < best_cost:
            best, best_cost = current[:], current_cost
    return best


def main() -> None:
    rng = random.Random(SEED)
    ratios = []
    costs = {"MST puro": [], "MST + 2-opt": [],
             "Búsqueda local": [], "Recocido simulado": []}

    for instance in range(INSTANCE_COUNT):
        points = [(rng.uniform(0, 100), rng.uniform(0, 100))
                  for _ in range(CITY_COUNT)]
        distances = distance_matrix(points)
        mst_route = preorder(mst_prim(distances))
        mst_cost = tour_length(distances, mst_route)
        optimum = exact_tsp_cost(distances)
        ratios.append(mst_cost / optimum)
        costs["MST puro"].append(mst_cost)
        costs["MST + 2-opt"].append(tour_length(
            distances, two_opt_descent(distances, mst_route)))

        local_start = list(range(CITY_COUNT))
        rng.shuffle(local_start)
        costs["Búsqueda local"].append(tour_length(
            distances, two_opt_descent(distances, local_start)))
        sa_start = list(range(CITY_COUNT))
        rng.shuffle(sa_start)
        sa_route = simulated_annealing(distances, sa_start, rng)
        costs["Recocido simulado"].append(tour_length(distances, sa_route))

    print(f"Instancias: {INSTANCE_COUNT} con {CITY_COUNT} ciudades cada una")
    print(f"Razón MST/óptimo: promedio={sum(ratios) / len(ratios):.3f}; "
          f"máxima={max(ratios):.3f}")
    for method, method_costs in costs.items():
        print(f"{method}: costo promedio={sum(method_costs) / len(method_costs):.2f}")

    output_dir = Path(__file__).resolve().parent / "graficas"
    output_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=15, color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.axvline(2, color="#d62728", linestyle="--", linewidth=2,
               label="Límite teórico = 2")
    ax.set(title=f"Razón de aproximación MST en {INSTANCE_COUNT} instancias",
           xlabel="Costo MST / costo óptimo", ylabel="Número de instancias")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ratio_file = output_dir / "Ejercicio5_razon_aproximacion.png"
    fig.savefig(ratio_file, dpi=160)
    print("Se abrirá el histograma; ciérralo para ver la comparación de costos.")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot = ax.boxplot(list(costs.values()), tick_labels=list(costs),
                      patch_artist=True, showmeans=True)
    for box, color in zip(plot["boxes"], ["#4c78a8", "#72b7b2", "#f2cf5b", "#e07a5f"]):
        box.set_facecolor(color)
    ax.set(title="Costos de métodos TSP sobre las mismas instancias",
           ylabel="Longitud del tour (menor es mejor)")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    costs_file = output_dir / "Ejercicio5_comparacion_costos.png"
    fig.savefig(costs_file, dpi=160)
    print("Se abrirá la comparación de costos; cierra su ventana para terminar.")
    plt.show()
    plt.close(fig)
    print(f"Gráficas guardadas en: {ratio_file} y {costs_file}")


if __name__ == "__main__":
    main()
