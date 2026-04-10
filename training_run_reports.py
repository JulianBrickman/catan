from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, full_key))
        elif is_dataclass(value):
            flattened.update(_flatten_dict(asdict(value), full_key))
        else:
            flattened[full_key] = value
    return flattened


def _session_sort_key(session_dir: Path) -> tuple[int, str]:
    parts = session_dir.name.split("_")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2]), session_dir.name
    return 0, session_dir.name


def _safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _gather_update_row(session_dir: Path, update_dir: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    train_metrics = _load_json(update_dir / "train_metrics.json")
    eval_metrics = _load_json(update_dir / "eval_metrics.json")
    update_number = int(update_dir.name.split("_")[1])
    checkpoint_path = session_dir / "checkpoints" / f"checkpoint_{update_number:04d}.pt"

    row: dict[str, Any] = {
        "session_name": session_dir.name,
        "session_path": str(session_dir),
        "update": update_number,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else "",
        "resume_checkpoint": run_config.get("resume_checkpoint"),
        "resumed_update_step": run_config.get("resumed_update_step"),
        "updates_requested": run_config.get("updates"),
    }
    row.update({f"param_{k}": v for k, v in _flatten_dict(run_config.get("ppo_config", {})).items()})
    row.update({f"param_{k}": v for k, v in _flatten_dict(run_config.get("env_config", {})).items()})
    row.update({f"train_{k}": v for k, v in train_metrics.items()})
    row.update({f"eval_{k}": v for k, v in eval_metrics.items()})
    row["param_run_eval_episodes"] = run_config.get("eval_episodes", eval_metrics.get("episodes_evaluated"))
    return row


def _select_best_update(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("eval_average_first_place_rate", 0.0)),
            -float(row.get("eval_truncation_count", 0.0)),
            float(row.get("eval_average_final_vp", 0.0)),
            -float(row.get("eval_average_episode_turns", 0.0)),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_value(row.get(key, "")) for key in fieldnames})


def _format_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def _write_markdown(path: Path, session_rows: list[dict[str, Any]], update_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Training Run Comparison",
        "",
        "This file is auto-generated from `training_runs/` and is meant to make session-to-session comparison easy.",
        "",
        "## Session Summary",
        "",
        "| Session | Best Update | Eval 1st Place | Eval Truncations | Eval Final VP | Eval Turns | Resume From | Max Turns | Rollout Episodes | Eval Episodes | PPO Epochs | Reward Placement | Reward VP Gain | Reward Final VP | Reward Win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in session_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["session_name"],
                    _format_metric(row.get("best_update"), 0),
                    _format_metric(row.get("best_eval_average_first_place_rate")),
                    _format_metric(row.get("best_eval_truncation_count"), 0),
                    _format_metric(row.get("best_eval_average_final_vp")),
                    _format_metric(row.get("best_eval_average_episode_turns"), 2),
                    str(row.get("resume_checkpoint") or ""),
                    _format_metric(row.get("param_max_turns"), 0),
                    _format_metric(row.get("param_rollout_episodes"), 0),
                    _format_metric(row.get("param_run_eval_episodes"), 0),
                    _format_metric(row.get("param_update_epochs"), 0),
                    _format_metric(row.get("param_reward_config.weights.placement_score")),
                    _format_metric(row.get("param_reward_config.weights.vp_gain")),
                    _format_metric(row.get("param_reward_config.weights.final_vp")),
                    _format_metric(row.get("param_reward_config.weights.win")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Update Summary",
            "",
            "| Session | Update | Eval 1st Place | Eval Truncations | Eval Final VP | Eval Turns | Train Final VP | Train Turns | Checkpoint |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in update_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["session_name"],
                    _format_metric(row.get("update"), 0),
                    _format_metric(row.get("eval_average_first_place_rate")),
                    _format_metric(row.get("eval_truncation_count"), 0),
                    _format_metric(row.get("eval_average_final_vp")),
                    _format_metric(row.get("eval_average_episode_turns"), 2),
                    _format_metric(row.get("train_average_final_vp")),
                    _format_metric(row.get("train_average_episode_turns"), 2),
                    str(row.get("checkpoint_path") or ""),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_training_run_reports(base_dir: str | Path = "training_runs") -> tuple[Path, Path]:
    base_path = Path(base_dir)
    session_dirs = sorted(
        [path for path in base_path.iterdir() if path.is_dir() and path.name.startswith("training_session_")],
        key=_session_sort_key,
    )

    update_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for session_dir in session_dirs:
        run_config_path = session_dir / "run_config.json"
        if not run_config_path.exists():
            continue
        run_config = _load_json(run_config_path)
        eval_episodes = run_config.get("env_config", {}).get("reward_config", {})
        _ = eval_episodes
        current_rows: list[dict[str, Any]] = []
        for update_dir in sorted(session_dir.glob("update_*")):
            train_path = update_dir / "train_metrics.json"
            eval_path = update_dir / "eval_metrics.json"
            if not train_path.exists() or not eval_path.exists():
                continue
            row = _gather_update_row(session_dir, update_dir, run_config)
            update_rows.append(row)
            current_rows.append(row)
        best_row = _select_best_update(current_rows)
        if best_row is None:
            continue
        session_summary = {
            "session_name": session_dir.name,
            "session_path": str(session_dir),
            "resume_checkpoint": run_config.get("resume_checkpoint"),
            "best_update": best_row.get("update"),
            "best_checkpoint_path": best_row.get("checkpoint_path"),
            "best_eval_average_first_place_rate": best_row.get("eval_average_first_place_rate"),
            "best_eval_truncation_count": best_row.get("eval_truncation_count"),
            "best_eval_average_final_vp": best_row.get("eval_average_final_vp"),
            "best_eval_average_episode_turns": best_row.get("eval_average_episode_turns"),
            "best_eval_average_episode_reward": best_row.get("eval_average_episode_reward"),
            "best_eval_average_final_placement_score": best_row.get("eval_average_final_placement_score"),
            "latest_update": current_rows[-1].get("update"),
            "latest_checkpoint_path": current_rows[-1].get("checkpoint_path"),
            "latest_eval_average_first_place_rate": current_rows[-1].get("eval_average_first_place_rate"),
            "latest_eval_truncation_count": current_rows[-1].get("eval_truncation_count"),
            "latest_eval_average_final_vp": current_rows[-1].get("eval_average_final_vp"),
            "latest_eval_average_episode_turns": current_rows[-1].get("eval_average_episode_turns"),
        }
        for key, value in current_rows[-1].items():
            if key.startswith("param_"):
                session_summary[key] = value
        session_rows.append(session_summary)

    session_csv = base_path / "training_session_summary.csv"
    update_csv = base_path / "training_update_summary.csv"
    summary_md = base_path / "training_session_report.md"

    _write_csv(session_csv, session_rows)
    _write_csv(update_csv, update_rows)
    _write_markdown(summary_md, session_rows, update_rows)

    return session_csv, summary_md


if __name__ == "__main__":
    update_training_run_reports()
