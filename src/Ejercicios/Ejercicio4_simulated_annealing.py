"""Ejercicio 4: Ajuste de parámetros con recocido simulado.

PREGUNTA: ¿Cómo afectan los esquemas de enfriamiento el tiempo y la calidad?
RESPUESTA: El esquema exponencial baja la temperatura multiplicándola por un factor
fijo; el lineal la reduce en pasos uniformes. Con estos parámetros, el lineal tarda
más porque realiza alrededor de 3,499 iteraciones, frente a unas 1,377 del exponencial.

En la ejecución de referencia, el costo promedio fue 36.806 para el exponencial y
39.670 para el lineal. Por tanto, en esa prueba el esquema más lento no produjo una
ruta mejor. El resultado depende de los parámetros y las instancias; el tiempo puede
variar según el equipo. Las gráficas muestran la convergencia y la comparación entre
tiempo de ejecución y costo final.
"""

import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt


CITY_COUNT = 20
TRIALS = 20
T0 = 10.0
ALPHA = 0.995
T_MIN = 0.01
LINEAR_ITERATIONS = 3_500


def tour_length(points: list[tuple[float, float]], tour: list[int]) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]])
               for i in range(len(tour)))


def random_tour(n: int, rng: random.Random) -> list[int]:
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def two_opt_successor(tour: list[int], rng: random.Random) -> list[int]:
    pairs = [(i, k) for i in range(len(tour) - 1) for k in range(i + 1, len(tour))
             if not (i == 0 and k == len(tour) - 1)]
    i, k = rng.choice(pairs)
    return tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]


def exponential_schedule(t: int) -> float:
    temperature = T0 * ALPHA ** t
    return temperature if temperature >= T_MIN else 0.0


def linear_schedule(t: int) -> float:
    return max(T0 * (1 - t / LINEAR_ITERATIONS), 0.0)


def simulated_annealing(points: list[tuple[float, float]], initial: list[int],
                        schedule, rng: random.Random) -> tuple[list[int], list[float]]:
    current = initial[:]
    current_cost = tour_length(points, current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    iteration = 1
    while (temperature := schedule(iteration)) > 0:
        candidate = two_opt_successor(current, rng)
        candidate_cost = tour_length(points, candidate)
        delta = candidate_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current, current_cost = candidate, candidate_cost
        if current_cost < best_cost:
            best, best_cost = current[:], current_cost
        history.append(best_cost)
        iteration += 1
    return best, history


def main() -> None:
    point_rng = random.Random(7)
    points = [(point_rng.uniform(0, 10), point_rng.uniform(0, 10))
              for _ in range(CITY_COUNT)]
    schedules = {"Exponencial": exponential_schedule, "Lineal": linear_schedule}
    histories = {name: [] for name in schedules}
    runtimes = {name: [] for name in schedules}
    final_costs = {name: [] for name in schedules}

    for trial in range(TRIALS):
        initial_rng = random.Random(10_000 + trial)
        initial = random_tour(CITY_COUNT, initial_rng)
        for schedule_index, (name, schedule) in enumerate(schedules.items()):
            rng = random.Random(20_000 + trial * 10 + schedule_index)
            start = time.perf_counter()
            best, history = simulated_annealing(points, initial, schedule, rng)
            runtimes[name].append(time.perf_counter() - start)
            final_costs[name].append(tour_length(points, best))
            histories[name].append(history)

    output_dir = Path(__file__).resolve().parent / "graficas"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in histories.items():
        max_length = max(map(len, runs))
        padded = [run + [run[-1]] * (max_length - len(run)) for run in runs]
        mean_curve = [sum(run[i] for run in padded) / len(padded)
                      for i in range(max_length)]
        ax.plot(range(len(mean_curve)), mean_curve, label=name)
    ax.set(title="Recocido simulado: convergencia promedio",
           xlabel="Iteración", ylabel="Mejor costo encontrado (menor es mejor)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    convergence_file = output_dir / "Ejercicio4_convergencia.png"
    fig.savefig(convergence_file, dpi=160)
    print("Se abrirá la gráfica de convergencia; ciérrala para continuar.")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Exponencial": "#3d8b74", "Lineal": "#e07a5f"}
    for name in schedules:
        mean_time = sum(runtimes[name]) / TRIALS
        mean_cost = sum(final_costs[name]) / TRIALS
        time_std = (sum((value - mean_time) ** 2 for value in runtimes[name]) / TRIALS) ** 0.5
        cost_std = (sum((value - mean_cost) ** 2 for value in final_costs[name]) / TRIALS) ** 0.5
        ax.errorbar(mean_time, mean_cost, xerr=time_std, yerr=cost_std, fmt="o",
                    capsize=5, markersize=8, color=colors[name], label=name)
        print(f"{name}: tiempo promedio={mean_time:.4f} s; costo final promedio="
              f"{mean_cost:.3f} (desv. {cost_std:.3f})")
    ax.set(title=f"Tiempo y calidad final ({TRIALS} pruebas)",
           xlabel="Tiempo promedio por ejecución (s)",
           ylabel="Costo final promedio (menor es mejor)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    quality_file = output_dir / "Ejercicio4_tiempo_calidad.png"
    fig.savefig(quality_file, dpi=160)
    print("Se abrirá la gráfica de tiempo y calidad; ciérrala para terminar.")
    plt.show()
    plt.close(fig)
    print(f"Gráficas guardadas en: {convergence_file} y {quality_file}")


if __name__ == "__main__":
    main()
