"""Ejercicio 2: Análisis algorítmico (Greedy Knapsack).

PREGUNTA: ¿Qué obtiene cada criterio en la instancia de capacidad 10?
RESPUESTA: Menor peso elige A y B: usa 7 unidades y obtiene valor 10. Valor/peso
elige B y C: usa las 10 unidades y obtiene valor 18.

PREGUNTA: ¿Por qué menor peso da un resultado inferior en este caso?
RESPUESTA: A pesa poco, pero también aporta muy poco valor. Al elegirlo primero,
el método deja fuera uno de los objetos B o C, que ofrecen más valor. El ejemplo
muestra que el objeto más liviano no siempre forma parte de la mejor combinación.
La gráfica compara objetos seleccionados, capacidad utilizada y valor total.
"""


from pathlib import Path

import matplotlib.pyplot as plt


# Instancia de ejemplo: el objeto ligero de bajo valor/peso desplaza una
# combinación de mayor valor cuando se usa el criterio de menor peso.
ITEMS = ["A", "B", "C"]
VALUES = [1, 9, 9]
WEIGHTS = [2, 5, 5]
CAPACITY = 10


def greedy_knapsack(values: list[int], weights: list[int], capacity: int,
                    criterion: str) -> list[int]:
    """Devuelve índices seleccionados usando peso o valor/peso como prioridad."""
    if criterion == "lightest":
        order = sorted(range(len(values)), key=lambda i: (weights[i], i))
    elif criterion == "value_weight":
        order = sorted(range(len(values)), key=lambda i: (-values[i] / weights[i], i))
    else:
        raise ValueError("criterion debe ser 'lightest' o 'value_weight'")

    chosen = []
    remaining = capacity
    for index in order:
        if weights[index] <= remaining:
            chosen.append(index)
            remaining -= weights[index]
    return chosen


def main() -> None:
    methods = {
        "Menor peso": greedy_knapsack(VALUES, WEIGHTS, CAPACITY, "lightest"),
        "Valor/peso": greedy_knapsack(VALUES, WEIGHTS, CAPACITY, "value_weight"),
    }
    selected_names = {}
    used_capacity = {}
    total_value = {}
    for method, chosen in methods.items():
        selected_names[method] = [ITEMS[i] for i in chosen]
        used_capacity[method] = sum(WEIGHTS[i] for i in chosen)
        total_value[method] = sum(VALUES[i] for i in chosen)
        print(f"{method}: objetos={selected_names[method]}, "
              f"capacidad usada={used_capacity[method]}/{CAPACITY}, "
              f"valor={total_value[method]}")

    output_dir = Path(__file__).resolve().parent / "graficas"
    output_dir.mkdir(exist_ok=True)
    labels = list(methods)
    colors = ["#e07a5f", "#3d8b74"]
    metrics = [
        ("Objetos seleccionados", [len(methods[label]) for label in labels]),
        ("Capacidad utilizada", [used_capacity[label] for label in labels]),
        ("Valor total", [total_value[label] for label in labels]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, values) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel("Cantidad")
        if title == "Objetos seleccionados":
            for bar, label in zip(bars, labels):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                        f"{len(methods[label])}: {', '.join(selected_names[label])}",
                        ha="center", fontsize=9)
            ax.set_ylim(top=max(len(methods[label]) for label in labels) + 0.9)
        else:
            ax.bar_label(bars, padding=3)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle(f"Mochila voraz (capacidad {CAPACITY}) — objetos: A(1,2), B(9,5), C(9,5)")
    fig.tight_layout()
    output_file = output_dir / "Ejercicio2_comparacion_mochila.png"
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    print("Se abrirá la gráfica; cierra su ventana para terminar el programa.")
    plt.show()
    plt.close(fig)
    print(f"Gráfica guardada en: {output_file}")


if __name__ == "__main__":
    main()
