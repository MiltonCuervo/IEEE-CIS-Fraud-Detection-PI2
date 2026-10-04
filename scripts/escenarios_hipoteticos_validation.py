"""Evaluate clearly hypothetical binary decision-cost scenarios on validation only.

No costs in this file are approved business estimates. The results are sensitivity
scenarios, use the amount unit as stored in IEEE-CIS, and must not be interpreted
as realized financial savings or used to open the final-test gate.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fraud_cost.scenarios import evaluate_hypothetical_scenarios

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data" / "interim" / "validation_scores.parquet"
OUT_TABLE = ROOT / "results" / "tables" / "escenarios_hipoteticos_validation.csv"
OUT_FIGURE = ROOT / "results" / "figures" / "ahorro_hipotetico_validation.png"
LAMBDA_GRID = (0.25, 0.50, 1.00)  # escenarios hipotéticos de pérdida del monto
B_RATIO_GRID = (0.25, 0.50, 1.00, 2.00)  # fracción/múltiplo de monto mediano típico
FRAUD_BLOCK_RESIDUAL_COST = 0.0  # supuesto idealizado, no costo observado


def main():
    if not SCORES.exists():
        raise FileNotFoundError(f"Faltan scores de validation: {SCORES}; ejecute notebook 03")
    scores = pd.read_parquet(SCORES)
    result = evaluate_hypothetical_scenarios(
        scores,
        lambda_grid=LAMBDA_GRID,
        b_ratio_grid=B_RATIO_GRID,
        fraud_block_residual_cost=FRAUD_BLOCK_RESIDUAL_COST,
    )
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_TABLE, index=False)

    savings = (result[result.strategy.eq("bmr_binary")]
               .pivot(index="loss_fraction_lambda", columns="legitimate_block_cost_ratio_of_validation_median_amount", values="savings_fraction_vs_fixed_proxy")
               .sort_index())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(savings.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(savings.columns)), [f"{x:g}×" for x in savings.columns])
    ax.set_yticks(range(len(savings.index)), [f"λ={x:g}" for x in savings.index])
    ax.set_xlabel("B hipotético / mediana en validation_policy")
    ax.set_ylabel("Fracción de pérdida hipotética λ")
    ax.set_title("Ahorro proxy hipotético de BMR frente a umbral 0,5\nvalidation_policy; C bloqueo fraude = 0")
    for i in range(savings.shape[0]):
        for j in range(savings.shape[1]):
            value = savings.iloc[i, j]
            ax.text(j, i, f"{value:.1%}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Ahorro relativo proxy vs. fijo")
    fig.tight_layout()
    fig.savefig(OUT_FIGURE, dpi=160, bbox_inches="tight")
    reference_amount = float(result["validation_median_amount_reference"].iloc[0])
    fraud_count = int(result["frauds"].iloc[0])
    print(f"validation_policy rows={len(scores):,}; frauds={fraud_count:,}; median amount reference={reference_amount:.4f} dataset units")
    print("Scenarios: lambda=", list(LAMBDA_GRID), "; B/median=", list(B_RATIO_GRID), "; C_block_fraud=0 (hypothetical); review=False; test not loaded")
    print("Saved table:", OUT_TABLE)
    print("Saved figure:", OUT_FIGURE)
    print(result[result.strategy.eq("bmr_binary")][["loss_fraction_lambda", "legitimate_block_cost_ratio_of_validation_median_amount", "cost_total_proxy", "savings_vs_fixed_proxy", "savings_fraction_vs_fixed_proxy", "fraud_recall", "legitimate_block_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
