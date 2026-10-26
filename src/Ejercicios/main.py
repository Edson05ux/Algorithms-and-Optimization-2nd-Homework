"""Menú principal para ejecutar los experimentos de src/Ejercicios."""

import subprocess
import sys
from pathlib import Path


EXERCISES = {
    "1": ("Pi paralelo", "Ejercicio1_estimate_pi.py"),
    "2": ("Mochila", "Ejercicio2_knapsack.py"),
    "3": ("TSP 2-opt", "Ejercicio3_2_opt.py"),
    "4": ("Recocido", "Ejercicio4_simulated_annealing.py"),
    "5": ("MST para TSP", "Ejercicio5_MST_prim.py"),
}


def main() -> None:
    exercises_dir = Path(__file__).resolve().parent

    while True:
        print("\n=== EJERCICIOS DE OPTIMIZACIÓN ===")
        for option, (title, _) in EXERCISES.items():
            print(f"{option}. {title}")
        print("0. Salir")

        choice = input("Selecciona un ejercicio: ").strip()
        if choice == "0":
            print("Hasta luego.")
            break
        if choice not in EXERCISES:
            print("Opción no válida. Elige un número del 0 al 5.")
            continue

        title, filename = EXERCISES[choice]
        script_path = exercises_dir / filename
        print(f"\n--- Ejecutando: {title} ---")
        print("Las gráficas se guardan en la misma ruta y se reemplazan al volver a ejecutar.\n")
        result = subprocess.run([sys.executable, str(script_path)], cwd=exercises_dir)
        if result.returncode != 0:
            print(f"El ejercicio terminó con un error (código {result.returncode}).")

        input("\nPresiona Enter para volver al menú...")


if __name__ == "__main__":
    main()
