"""Sensitivity curves from actual scripted completions; no live endpoint access."""

from __future__ import annotations

import csv
import importlib.metadata
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from servegauge.core.hashing import content_hash
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.stats.multiple import holm
from servegauge.stats.paired_fast import (
    energy_test_fast,
    hamming_test_fast,
    jsd_test_fast,
    logprob_test_fast,
)
from servegauge.validation.common import (
    ROOT,
    named_seed,
    offline_only,
    proportion,
    receipt,
    settings,
    write_json,
)
from servegauge.verdicts.config import load_policy


def observations(
    n: int, *, seed: int, faults: FaultConfig
) -> tuple[list[str], list[dict[str, float]]]:
    backend = ScriptedBackend(seed, faults.model_copy(update={"seed": seed}))
    words, logs = [], []
    body = {
        "model": "mock",
        "messages": [
            {
                "role": "user",
                "content": 'One color. TW_TASK: {"type":"sample",'
                '"choices":["red","blue","green","gold"]}',
            }
        ],
        "temperature": 1.0,
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 1,
    }
    for _ in range(n):
        response = backend.respond(body)
        choice = response["choices"][0]
        words.append(choice["message"]["content"])
        logs.append({row["token"]: row["logprob"] for row in choice["logprobs"]["content"]})
    return words, logs


def power_tests(n: int, *, seed: int, faults: FaultConfig, permutations: int) -> dict[str, Any]:
    first, log_first = observations(n, seed=named_seed(seed, "target"), faults=faults)
    second, log_second = observations(n, seed=named_seed(seed, "reference"), faults=FaultConfig())
    vocabulary = {word: index for index, word in enumerate(("red", "blue", "green", "gold"))}
    x, y = np.eye(4)[[vocabulary[s] for s in first]], np.eye(4)[[vocabulary[s] for s in second]]
    floor = float(load_policy().entries["FP3"]["missing_logprob_floor"])
    methods = {
        "FP1": jsd_test_fast(first, second, seed=seed, resamples=permutations),
        "FP2": hamming_test_fast(first, second, length=16, seed=seed, resamples=permutations),
        "FP3": logprob_test_fast(
            log_first, log_second, missing_floor=floor, seed=seed, resamples=permutations
        ),
        "FP4-statistic": energy_test_fast(x, y, seed=seed, resamples=permutations),
    }
    adjusted = holm([r.p_value for r in methods.values()])
    return {
        name: {
            "statistic": result.statistic,
            "p_value": result.p_value,
            "adjusted_p": p,
            "permutations": result.permutations,
            "exact": result.exact,
        }
        for (name, result), p in zip(methods.items(), adjusted, strict=True)
    }


def power_study(
    *, replicates: int | None = None, out: Path = Path("docs/report")
) -> dict[str, Any]:
    cfg, policy = settings(), load_policy()
    n_replicates = cfg["power"]["replicates"] if replicates is None else replicates
    if n_replicates < 30:
        raise ValueError("power_curves_require_at_least_30_replicates")
    output = receipt("power", {**cfg["power"], "seed": cfg["seed"], "replicates": n_replicates})
    output["packages"] = {
        n: importlib.metadata.version(n) for n in ("numpy", "scipy", "matplotlib")
    }
    output["configuration_notes"] = [
        "One four-category prompt cell; matched n per side, not the full S6 multi-language design",
        "FP4-statistic uses fixed one-hot response vectors, not a neural text embedding model",
        "All methods share scripted completions; Holm corrects their four-test family",
        "F-MIX and F-QSIM use original YAML strengths; "
        "real quantization is not simulated or established",
        "Prices are a fixed scenario, not current quotes, provider invoices or actual spending",
        "Queries include both fresh sides; one-time control calibration "
        "and real reasoning/cache costs are excluded",
    ]
    output["rows"] = []
    output["replicates"] = []
    spend_before = content_hash((ROOT / "budget/dev_spend.json").read_text(encoding="utf-8"))
    scenarios = {
        "clean": FaultConfig(),
        **{
            name: FaultConfig.model_validate(
                yaml.safe_load((ROOT / f"configs/faults/{file}.yaml").read_text(encoding="utf-8"))
            )
            for name, file in (("F-SWAP", "swap"), ("F-MIX", "mix"), ("F-QSIM", "qsim"))
        },
    }
    with offline_only():
        for scenario, faults in scenarios.items():
            for n in cfg["power"]["query_sizes"]:
                trials = [
                    power_tests(
                        n,
                        seed=named_seed(cfg["seed"], "power", scenario, n, index),
                        faults=faults,
                        permutations=cfg["power"]["permutations"],
                    )
                    for index in range(n_replicates)
                ]
                queries = 2 * n
                dollars = (
                    queries
                    * (
                        cfg["power"]["input_tokens_per_query"]
                        * cfg["power"]["scenario_input_usd_per_million"]
                        + cfg["power"]["output_tokens_per_query"]
                        * cfg["power"]["scenario_output_usd_per_million"]
                    )
                    / 1_000_000
                )
                for method in trials[0]:
                    detected = sum(trial[method]["adjusted_p"] <= policy.alpha for trial in trials)
                    power = proportion(detected, n_replicates)
                    output["rows"].append(
                        {
                            "scenario": scenario,
                            "method": method,
                            "n_per_side": n,
                            "queries": queries,
                            "scenario_usd": dollars,
                            "power": power,
                            "scenario_usd_per_detection": dollars / power["estimate"]
                            if detected
                            else None,
                            "cost_per_detection_interval": [
                                dollars / power["ci_high"],
                                dollars / power["ci_low"] if power["ci_low"] > 0 else None,
                            ],
                        }
                    )
                output["replicates"].append(
                    {
                        "scenario": scenario,
                        "n_per_side": n,
                        "seeds": [
                            named_seed(cfg["seed"], "power", scenario, n, i)
                            for i in range(n_replicates)
                        ],
                        "fault_configuration": faults.model_dump(mode="json"),
                        "results": trials,
                    }
                )
                print(f"power {scenario} n={n}: {n_replicates} replicates", flush=True)
    output["spend_hash_before"] = spend_before
    output["spend_hash_after"] = content_hash(
        (ROOT / "budget/dev_spend.json").read_text(encoding="utf-8")
    )
    if output["spend_hash_before"] != output["spend_hash_after"]:
        raise RuntimeError("offline_study_changed_spending")
    output["complete"] = True
    write_json(out / "power.json", output)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (figures / "power.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scenario",
                "method",
                "n_per_side",
                "queries",
                "scenario_usd",
                "detections",
                "replicates",
                "power",
                "ci_low",
                "ci_high",
                "scenario_usd_per_detection",
            ]
        )
        for row in output["rows"]:
            p = row["power"]
            writer.writerow(
                [row[k] for k in ("scenario", "method", "n_per_side", "queries", "scenario_usd")]
                + [p[k] for k in ("successes", "n", "estimate", "ci_low", "ci_high")]
                + [row["scenario_usd_per_detection"]]
            )
    plot_power(output, figures)
    lines = [
        "# Offline fingerprint sensitivity",
        "",
        "Actual model API calls: 0. Actual spending: $0. Reference tier: MOCK.",
        "",
        *["- " + note for note in output["configuration_notes"]],
        "",
        "![Power versus total queries](figures/power_queries.svg)",
        "",
        "![Power versus illustrative token charge](figures/power_usd.svg)",
        "",
        "Each point has a 95% Wilson interval over independent seeds. These curves measure "
        "statistical rejection sensitivity; complete diagnosis power is measured "
        "separately in [faults.md](faults.md).",
        "",
        "[Full results and seeds](power.json) · [Data table](figures/power.csv)",
        "",
        "Low power is retained. No seed, margin, fault severity or p-value threshold "
        "was changed to increase it.",
        "",
    ]
    (out / "power.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def plot_power(output: dict[str, Any], directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "servegauge-validation-v1"
    import matplotlib.pyplot as plt

    for axis_name, xlabel in (
        ("queries", "Total scripted requests (both sides)"),
        ("scenario_usd", "Illustrative token charge (USD; no spending)"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, layout="constrained")
        for ax, method in zip(axes.flat, ("FP1", "FP2", "FP3", "FP4-statistic"), strict=True):
            for scenario in ("F-SWAP", "F-MIX", "F-QSIM", "clean"):
                rows = [
                    r for r in output["rows"] if r["method"] == method and r["scenario"] == scenario
                ]
                xs = [r[axis_name] for r in rows]
                ax.plot(xs, [r["power"]["estimate"] for r in rows], marker="o", label=scenario)
                ax.fill_between(
                    xs,
                    [r["power"]["ci_low"] for r in rows],
                    [r["power"]["ci_high"] for r in rows],
                    alpha=0.12,
                )
            ax.set(title=method, xlabel=xlabel, ylabel="Rejection probability", ylim=(-0.02, 1.02))
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(
            f"ServeGauge: synthetic sensitivity, n={output['configuration']['replicates']} "
            "seeds/point\nFP4 uses one-hot vectors; clean curve is false-positive probability",
            fontsize=11,
        )
        fig.savefig(
            directory / f"power_{'queries' if axis_name == 'queries' else 'usd'}.svg",
            metadata={"Creator": "Hari Krishna Kancharla", "Date": None},
        )
        fig.savefig(
            directory / f"power_{'queries' if axis_name == 'queries' else 'usd'}.png",
            dpi=150,
            metadata={"Author": "Hari Krishna Kancharla"},
        )
        plt.close(fig)
