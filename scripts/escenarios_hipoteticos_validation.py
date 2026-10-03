"""Evaluate clearly hypothetical binary decision-cost scenarios on validation only.

No costs in this file are approved business estimates. The results are sensitivity
scenarios, use the amount unit as stored in IEEE-CIS, and must not be interpreted
as realized financial savings or used to open the final-test gate.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    required = {"TransactionAmt", "isFraud", "probability_calibrated", "action_fixed_0_5"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Faltan columnas en validation_scores: {sorted(missing)}")
    if scores[list(required)].isna().any().any():
        raise ValueError("Hay nulos en columnas requeridas")
    amount = scores["TransactionAmt"].to_numpy(dtype=float)
    y = scores["isFraud"].to_numpy(dtype=int)
    p = np.clip(scores["probability_calibrated"].to_numpy(dtype=float), 0, 1)
    fixed = scores["action_fixed_0_5"].astype(str).to_numpy()
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("TransactionAmt debe ser finito y no negativo")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("isFraud debe ser binario")

    # Escala descriptiva de validation, sin usar etiquetas: solo sirve para definir B hipotético.
    reference_amount = float(np.median(amount))
    fraud = y == 1
    legitimate = ~fraud
    fixed_block = fixed == "block"
    rows = []
    for loss_fraction in LAMBDA_GRID:
        for b_ratio in B_RATIO_GRID:
            legitimate_block_cost = b_ratio * reference_amount
            risk_approve = p * loss_fraction * amount
            risk_block = (1 - p) * legitimate_block_cost + p * FRAUD_BLOCK_RESIDUAL_COST
            bmr = np.where(risk_approve <= risk_block, "approve", "block")
            scenario_rows = []
            for strategy, action in (("fixed_0_5", fixed), ("bmr_binary", bmr)):
                block = action == "block"
                cost = np.zeros(len(y), dtype=float)
                cost[fraud & ~block] = loss_fraction * amount[fraud & ~block]
                cost[fraud & block] = FRAUD_BLOCK_RESIDUAL_COST
                cost[legitimate & block] = legitimate_block_cost
                scenario_rows.append((strategy, action, cost))
            fixed_cost = float(scenario_rows[0][2].sum())
            for strategy, action, cost in scenario_rows:
                block = action == "block"
                row = {
                    "scenario_status": "HIPOTETICO_NO_APROBADO",
                    "strategy": strategy,
                    "rows": len(y),
                    "frauds": int(fraud.sum()),
                    "loss_fraction_lambda": loss_fraction,
                    "legitimate_block_cost_ratio_of_validation_median_amount": b_ratio,
                    "validation_median_amount_reference": reference_amount,
                    "legitimate_block_cost_B_proxy": legitimate_block_cost,
                    "fraud_block_residual_cost_C_proxy": FRAUD_BLOCK_RESIDUAL_COST,
                    "review_enabled": False,
                    "cost_total_proxy": float(cost.sum()),
                    "cost_per_transaction_proxy": float(cost.mean()),
                    "fixed_baseline_cost_proxy": fixed_cost,
                    "savings_vs_fixed_proxy": fixed_cost - float(cost.sum()),
                    "savings_fraction_vs_fixed_proxy": ((fixed_cost - float(cost.sum())) / fixed_cost if fixed_cost > 0 else np.nan),
                    "frauds_blocked": int(np.sum(fraud & block)),
                    "frauds_missed": int(np.sum(fraud & ~block)),
                    "fraud_recall": float(np.sum(fraud & block) / fraud.sum()),
                    "legitimate_block_count": int(np.sum(legitimate & block)),
                    "legitimate_block_rate": float(np.mean(block[legitimate])),
                    "policy_block_count": int(block.sum()),
                }
                rows.append(row)
    result = pd.DataFrame(rows)
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
    ax.set_xlabel("B hipotético / mediana de monto en validation")
    ax.set_ylabel("Fracción de pérdida hipotética λ")
    ax.set_title("Ahorro proxy hipotético de BMR frente a umbral 0,5\nValidation solamente; C bloqueo fraude = 0")
    for i in range(savings.shape[0]):
        for j in range(savings.shape[1]):
            value = savings.iloc[i, j]
            ax.text(j, i, f"{value:.1%}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Ahorro relativo proxy vs. fijo")
    fig.tight_layout()
    fig.savefig(OUT_FIGURE, dpi=160, bbox_inches="tight")
    print(f"Validation rows={len(scores):,}; frauds={int(fraud.sum()):,}; median amount reference={reference_amount:.4f}")
    print("Scenarios: lambda=", list(LAMBDA_GRID), "; B/median=", list(B_RATIO_GRID), "; C_block_fraud=0 (hypothetical); review=False; test not loaded")
    print("Saved table:", OUT_TABLE)
    print("Saved figure:", OUT_FIGURE)
    print(result[result.strategy.eq("bmr_binary")][["loss_fraction_lambda", "legitimate_block_cost_ratio_of_validation_median_amount", "cost_total_proxy", "savings_vs_fixed_proxy", "savings_fraction_vs_fixed_proxy", "fraud_recall", "legitimate_block_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
