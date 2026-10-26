"""Ejercicio 1: Aceleración paralela (Parallel Speedup).

PREGUNTA: ¿Por qué la aceleración medida no es perfectamente lineal?
RESPUESTA: La aceleración ideal con p procesos sería p veces. En la práctica,
iniciar procesos, distribuir las tareas y reunir sus resultados consume tiempo.
Además, puede haber partes secuenciales y competencia por el procesador y la memoria.
La ley de Amdahl explica que la parte secuencial limita la aceleración máxima. En
este experimento también se mide el costo de crear el grupo de procesos. La gráfica
compara el speedup medido con el ideal; los tiempos dependen del equipo y su carga.
"""

import multiprocessing as mp
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt


def estimate_pi(seed: int, samples: int = 300_000) -> float:
    """Estima pi con Monte Carlo usando un generador independiente por semilla."""
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


def _estimate_task(task: tuple[int, int]) -> float:
    """Adaptador serializable para enviar semilla y muestras al pool."""
    seed, samples = task
    return estimate_pi(seed, samples)


def main() -> None:
    process_counts = [1, 2, 4, 8]
    seeds = list(range(8))
    samples_per_task = 300_000
    repetitions = 3
    tasks = [(seed, samples_per_task) for seed in seeds]
    measured_times = {count: [] for count in process_counts}
    estimates = {}

    for repetition in range(repetitions):
        for process_count in process_counts:
            start = time.perf_counter()
            with mp.Pool(processes=process_count) as pool:
                estimates[process_count] = pool.map(_estimate_task, tasks)
            measured_times[process_count].append(time.perf_counter() - start)

    elapsed = [sum(measured_times[count]) / repetitions for count in process_counts]
    speedups = [elapsed[0] / current for current in elapsed]
    for count, avg_time, speedup in zip(process_counts, elapsed, speedups):
        pi_estimate = sum(estimates[count]) / len(seeds)
        print(f"Procesos: {count}; tiempo promedio: {avg_time:.3f} s; "
              f"speedup: {speedup:.2f}x; pi promedio: {pi_estimate:.6f}")

    output_dir = Path(__file__).resolve().parent / "graficas"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(process_counts, speedups, marker="o", linewidth=2, label="Speedup medido")
    ax.plot(process_counts, process_counts, linestyle="--", label="Aceleración ideal")
    ax.set(title="Estimación de π: aceleración paralela",
           xlabel="Número de procesos", ylabel="Speedup (T₁ / Tₚ)",
           xticks=process_counts)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_file = output_dir / "Ejercicio1_speedup.png"
    fig.savefig(output_file, dpi=160)
    print("Se abrirá la gráfica; cierra su ventana para terminar el programa.")
    plt.show()
    plt.close(fig)
    print(f"Gráfica guardada en: {output_file}")


if __name__ == "__main__":
    main()
