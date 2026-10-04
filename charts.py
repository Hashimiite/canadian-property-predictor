"""Matplotlib and Seaborn charts shared by model.py (saved to figures/) and app.py (shown live)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
DOLLARS = mticker.FuncFormatter(lambda x, _: f"${x / 1000:,.0f}k")


def price_trends(df):
    """Estimated house price by year for each province."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    provinces = df[df["PROVINCE"] != "Canada"]
    sns.lineplot(data=provinces, x="YEAR", y="TARGET_VALUE", hue="PROVINCE", linewidth=1.6, ax=ax)
    sns.lineplot(
        data=df[df["PROVINCE"] == "Canada"], x="YEAR", y="TARGET_VALUE",
        color="black", linewidth=2.6, linestyle="--", label="Canada", ax=ax,
    )
    ax.set(title="Estimated new house prices by province", xlabel="Year", ylabel="Estimated price (CAD)")
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def model_comparison(comparison):
    """Test R² and mean absolute error for each candidate model."""
    rows = pd.DataFrame(
        [{"Model": name, "R²": m["r2"], "MAE": m["mae"]} for name, m in comparison.items()]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=rows, x="Model", y="R²", hue="Model", palette="crest", legend=False, ax=axes[0])
    axes[0].set(title="R² on held out data (higher is better)", ylim=(0, 1.1), xlabel="")
    sns.barplot(data=rows, x="Model", y="MAE", hue="Model", palette="flare", legend=False, ax=axes[1])
    axes[1].set(title="Mean absolute error (lower is better)", xlabel="", ylabel="MAE (CAD)")
    axes[1].yaxis.set_major_formatter(DOLLARS)
    axes[1].margins(y=0.12)
    for ax, key, fmt in ((axes[0], "R²", "{:.3f}"), (axes[1], "MAE", "${:,.0f}")):
        for patch, value in zip(ax.patches, rows[key]):
            ax.annotate(fmt.format(value), (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                        ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def actual_vs_predicted(y_test, y_pred, model_name):
    """Scatter of predictions against true prices for the chosen model."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, edgecolor=None, ax=ax)
    low, high = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set(title=f"{model_name}: actual vs predicted", xlabel="Actual price", ylabel="Predicted price")
    ax.xaxis.set_major_formatter(DOLLARS)
    ax.yaxis.set_major_formatter(DOLLARS)
    fig.tight_layout()
    return fig


def feature_importance(features, importances, model_name):
    """Horizontal bar chart of the chosen model's feature importances."""
    data = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(data["Feature"], data["Importance"], color=sns.color_palette("crest", len(data)))
    ax.set(title=f"{model_name}: feature importance", xlabel="Share of importance")
    fig.tight_layout()
    return fig


def forecast_paths(history, forecast):
    """History since 1990 (solid) and the model's forecast (dashed) for each region."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = dict(zip(sorted(history["PROVINCE"].unique()), sns.color_palette("tab20", 11)))
    sns.lineplot(data=history, x="YEAR", y="TARGET_VALUE", hue="PROVINCE", palette=palette, linewidth=1.4, ax=ax)
    sns.lineplot(data=forecast, x="YEAR", y="TARGET_VALUE", hue="PROVINCE", palette=palette,
                 linewidth=1.4, linestyle="--", legend=False, ax=ax)
    ax.axvline(history["YEAR"].max() + 0.5, color="gray", linewidth=1)
    ax.text(history["YEAR"].max() + 1, ax.get_ylim()[1] * 0.97, "forecast", color="gray", fontsize=9, va="top")
    ax.set(title="House prices since 1990 and forecast for the next ten years",
           xlabel="Year", ylabel="Estimated price (CAD)")
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
