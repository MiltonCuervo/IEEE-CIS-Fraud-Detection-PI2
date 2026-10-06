"""Sensibilidad exploratoria de Ca; utiliza validación y nunca carga test."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from fraud_cost.scenarios import evaluate_hypothetical_scenarios, HYPOTHETICAL_CA_GRID

ROOT = Path(__file__).resolve().parents[1]


def main():
    scores_path = ROOT / "data/interim/validation_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"Ejecute primero notebook 03: falta {scores_path}")
    scores = pd.read_parquet(scores_path)
    result = evaluate_hypothetical_scenarios(scores, ca_grid=HYPOTHETICAL_CA_GRID)
    table_path = ROOT / "results/tables/escenarios_hipoteticos_validation.csv"
    figure_path = ROOT / "results/figures/ahorro_hipotetico_validation.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(table_path, index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for strategy, block in result.groupby("strategy"):
        ax.plot(block.administrative_cost_ca, block.savings_fraction_vs_reference * 100,
                marker="o", label=strategy)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set(xlabel="Ca hipotético (unidades del dataset)",
           ylabel="Ahorro frente a referencia trivial (%)",
           title="Sensibilidad de Ca en validación posterior")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(result.to_string(index=False))
    print("Ca ilustrativo; intervención cuesta Ca en ambas clases; test no cargado.")


if __name__ == "__main__":
    main()
