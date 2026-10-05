# ==============================================================================
# DeltaCore: deltacore/benchmarks/reporting/report.py
# Markdown report generation for reproducible benchmark runs.
# ==============================================================================

from datetime import datetime, timezone

from deltacore.benchmarks.runners.determinism import get_environment_info
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.schemas.result import AggregateResult


def generate_markdown_report(
    config: BenchmarkConfig,
    aggregate_results: list[AggregateResult],
    observations: list[str] | None = None,
) -> str:
    """Generate a clean, reproducible Markdown summary report of benchmark results.

    Adheres strictly to DeltaCore scientific reporting principles:
    - Never declares an automatic 'winner' or 'best model'.
    - Reports exact numerical metrics with mean +/- std.
    - Explicitly details failure counts and categories.
    - Distinguishes between mathematical properties, empirical observations,
      and benchmark results.

    Args:
        config: Benchmark configuration used.
        aggregate_results: List of AggregateResult objects across evaluated models.
        observations: Optional list of factual empirical observations.

    Returns:
        Formatted Markdown report string.
    """
    env_info = get_environment_info()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# Benchmark Report: {config.experiment_name}",
        "",
        f"**Generated**: {now_str}  ",
        f"**Sequence Length ($T$)**: {config.sequence_length}  ",
        f"**Dimensions**: Key $K={config.key_dim}$, Value $V={config.value_dim}$  ",
        f"**Dtype**: {config.dtype}  ",
        f"**Device**: {config.device}  ",
        "",
        "---",
        "",
        "## 1. Environment & Provenance",
        "",
        "| Parameter | Value |",
        "| :--- | :--- |",
        f"| Python Version | {env_info.get('python_version', 'N/A')} |",
        f"| PyTorch Version | {env_info.get('torch_version', 'N/A')} |",
        f"| CUDA Available | {env_info.get('cuda_available', 'N/A')} |",
        f"| CPU Threads | {env_info.get('num_threads', 'N/A')} |",
        f"| Task Seed | {config.seed} |",
        "",
        "---",
        "",
        "## 2. Evaluated Systems & Baselines",
        "",
        "| Model Identifier | Runs Tested | Successes | Failures |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for agg in aggregate_results:
        lines.append(
            f"| **{agg.model}** | {agg.num_runs} | {agg.num_success} | {agg.num_failures} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Aggregate Performance Summary",
            "",
        ]
    )

    # Collect all available metrics across models
    all_metrics: set[str] = set()
    for agg in aggregate_results:
        all_metrics.update(agg.metric_aggregates.keys())

    sorted_metrics = sorted(all_metrics)
    if sorted_metrics:
        headers = ["Metric"] + [f"{agg.model}" for agg in aggregate_results]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append(
            "| " + " | ".join([":---"] + [":---:"] * len(aggregate_results)) + " |"
        )

        for metric in sorted_metrics:
            row = [f"`{metric}`"]
            for agg in aggregate_results:
                stats = agg.metric_aggregates.get(metric)
                if stats is not None and stats["count"] > 0:
                    mean_val = stats["mean"]
                    std_val = stats["std"]
                    if abs(mean_val) < 1e-4 or abs(mean_val) >= 1e4:
                        cell = f"{mean_val:.2e} ± {std_val:.2e}"
                    else:
                        cell = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    cell = "N/A"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")

    # Document failure reasons if any
    any_failures = any(agg.num_failures > 0 for agg in aggregate_results)
    if any_failures:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Failure Diagnostics",
                "",
                "| Model | Failure Count | Category / Diagnostic Reason |",
                "| :--- | :---: | :--- |",
            ]
        )
        for agg in aggregate_results:
            if agg.num_failures > 0:
                for reason, count in agg.failure_reasons.items():
                    lines.append(f"| {agg.model} | {count} | `{reason}` |")

    # Document empirical observations
    if observations:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 5. Notable Empirical Observations",
                "",
            ]
        )
        for obs in observations:
            lines.append(f"- {obs}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Scientific Honesty & Interpretation Notice",
            "",
            "> **Methodological Guidance**:",
            "> 1. Differences in synthetic benchmark error do not establish real-world task superiority.",
            "> 2. Boundedness across tested sequences demonstrates empirical survival, not a proof of global stability.",
            "> 3. No system is crowned 'best'; choices depend on the operational trade-off between adaptation speed, memory growth, and computational cost.",
            "",
        ]
    )

    return "\n".join(lines)
