"""Ejercicio 3: Búsqueda local 2-opt para el problema del viajero (TSP).

PREGUNTA: ¿Vecino Más Cercano necesitó menos mejoras que una ruta aleatoria?
RESPUESTA: Sí, en estas 100 pruebas. La ruta aleatoria necesitó 5.54 mejoras en
promedio; Vecino Más Cercano, comenzando desde una ciudad aleatoria, necesitó 1.76.

PREGUNTA: ¿La calidad final también fue distinta?
RESPUESTA: El costo promedio fue 237.44 con inicio aleatorio y 235.28 con Vecino
Más Cercano. La diferencia es pequeña, pero en este experimento Vecino Más Cercano
terminó con una ruta algo más corta. Estos resultados dependen de las ciudades y no
garantizan que ocurra lo mismo en cualquier instancia. Los diagramas muestran las
distribuciones de las 100 pruebas.
"""

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


CITY_COUNT = 10
TRIALS = 100
SEED = 2026


def tour_length(points: list[tuple[float, float]], route: list[int]) -> float:
    return sum(math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
               for i in range(len(route)))


def two_opt_move(route: list[int], i: int, k: int) -> list[int]:
    """Invierte route[i:k+1], reemplazando las dos aristas del movimiento 2-opt."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def nearest_neighbor_tour(points: list[tuple[float, float]], start: int) -> list[int]:
    unvisited = set(range(len(points)))
    unvisited.remove(start)
    route = [start]
    while unvisited:
        current = route[-1]
        next_city = min(unvisited, key=lambda city: math.dist(points[current], points[city]))
        route.append(next_city)
        unvisited.remove(next_city)
    return route


def two_opt_descent(points: list[tuple[float, float]], route: list[int]) -> tuple[list[int], int]:
    """Búsqueda 2-opt de mejor mejora hasta alcanzar un óptimo local."""
    route = route[:]
    improvements = 0
    while True:
        best_route = route
        best_cost = tour_length(points, route)
        for i in range(len(route) - 1):
            for k in range(i + 1, len(route)):
                if i == 0 and k == len(route) - 1:
                    continue  # inversión equivalente a recorrer el mismo ciclo al revés
                candidate = two_opt_move(route, i, k)
                candidate_cost = tour_length(points, candidate)
                if candidate_cost < best_cost - 1e-12:
                    best_route, best_cost = candidate, candidate_cost
        if best_route is route:
            return route, improvements
        route = best_route
        improvements += 1


def main() -> None:
    rng = random.Random(SEED)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(CITY_COUNT)]
    results = {"Aleatoria": {"improvements": [], "costs": []},
               "Vecino más cercano": {"improvements": [], "costs": []}}

    for _ in range(TRIALS):
        random_route = list(range(CITY_COUNT))
        rng.shuffle(random_route)
        nearest_route = nearest_neighbor_tour(points, rng.randrange(CITY_COUNT))
        for label, initial in (("Aleatoria", random_route),
                               ("Vecino más cercano", nearest_route)):
            final_route, improvements = two_opt_descent(points, initial)
            results[label]["improvements"].append(improvements)
            results[label]["costs"].append(tour_length(points, final_route))

    for label, values in results.items():
        print(f"{label}: mejoras promedio={sum(values['improvements']) / TRIALS:.2f}; "
              f"costo local promedio={sum(values['costs']) / TRIALS:.2f}")

    output_dir = Path(__file__).resolve().parent / "graficas"
    output_dir.mkdir(exist_ok=True)
    labels = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].boxplot([results[label]["improvements"] for label in labels], tick_labels=labels,
                    patch_artist=True, boxprops={"facecolor": "#8ecae6"})
    axes[0].set(title="Mejoras 2-opt hasta óptimo local", ylabel="Número de mejoras")
    axes[1].boxplot([results[label]["costs"] for label in labels], tick_labels=labels,
                    patch_artist=True, boxprops={"facecolor": "#90be6d"})
    axes[1].set(title="Calidad del óptimo local", ylabel="Longitud del tour (menor es mejor)")

    # Mostrar cada resultado con una pequeña dispersión horizontal: muchos tours
    # convergen al mismo costo y quedan superpuestos en un boxplot tradicional.
    jitter_rng = random.Random(SEED)
    for position, label in enumerate(labels, start=1):
        for value in results[label]["costs"]:
            jitter = jitter_rng.uniform(-0.12, 0.12)
            axes[1].scatter(position + jitter, value, s=18, alpha=0.45,
                            color="#245c45", edgecolors="none", zorder=3)
        average = sum(results[label]["costs"]) / TRIALS
        axes[1].annotate(f"prom. {average:.2f}", (position, average),
                         xytext=(8, 8), textcoords="offset points", fontsize=8,
                         color="#173f2e")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=10)
    fig.suptitle(f"Comparación de inicios para 2-opt ({TRIALS} pruebas, {CITY_COUNT} ciudades)")
    fig.tight_layout()
    output_file = output_dir / "Ejercicio3_2opt_comparacion.png"
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    print("Se abrirá la gráfica; cierra su ventana para terminar el programa.")
    plt.show()
    plt.close(fig)
    print(f"Gráfica guardada en: {output_file}")


if __name__ == "__main__":
    main()
