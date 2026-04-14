from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

from catan_model import CatanPolicyValueNet
from catan_ppo import CatanPPOTrainer, PPOConfig, save_metrics, summarize_evaluation_metrics
from catan_rl_env import CatanEnvConfig, RewardConfig
from training_run_reports import update_training_run_reports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO baseline on the Catan RL environment.")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-episodes", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="training_runs")
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--reward-placement-score", type=float, default=1.0)
    parser.add_argument("--reward-vp-gain", type=float, default=0.0)
    parser.add_argument("--reward-setup-settlement-quality", type=float, default=0.0)
    parser.add_argument("--reward-final-vp", type=float, default=0.0)
    parser.add_argument("--reward-final-vp-margin", type=float, default=0.0)
    parser.add_argument("--reward-win", type=float, default=0.0)
    parser.add_argument("--reward-build-settlement", type=float, default=0.0)
    parser.add_argument("--reward-build-city", type=float, default=0.0)
    parser.add_argument("--reward-turn-penalty", type=float, default=0.0)
    parser.add_argument("--reward-truncation", type=float, default=0.0)
    parser.add_argument("--reward-missed-action-opportunity", type=float, default=0.0)
    return parser


def create_session_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("training_session_"))
    next_index = 1
    if existing:
        indices = []
        for path in existing:
            parts = path.name.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                indices.append(int(parts[2]))
        if indices:
            next_index = max(indices) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"training_session_{next_index:03d}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def build_reward_config(args: argparse.Namespace) -> RewardConfig:
    reward_config = RewardConfig()
    reward_config.weights["placement_score"] = args.reward_placement_score
    reward_config.weights["vp_gain"] = args.reward_vp_gain
    reward_config.weights["setup_settlement_quality"] = args.reward_setup_settlement_quality
    reward_config.weights["final_vp"] = args.reward_final_vp
    reward_config.weights["final_vp_margin"] = args.reward_final_vp_margin
    reward_config.weights["win"] = args.reward_win
    reward_config.weights["settlements_built"] = args.reward_build_settlement
    reward_config.weights["cities_built"] = args.reward_build_city
    reward_config.weights["turn_penalty"] = args.reward_turn_penalty
    reward_config.weights["truncation"] = args.reward_truncation
    reward_config.weights["missed_action_opportunity"] = args.reward_missed_action_opportunity
    return reward_config


def save_episode_artifacts(output_dir: Path, label: str, episode) -> None:
    episode_dir = output_dir / label
    episode_dir.mkdir(parents=True, exist_ok=True)
    if episode.final_state is not None:
        episode.final_state.write_svg(episode_dir / "final_state.svg")
    if episode.final_observation is not None:
        (episode_dir / "final_observation.json").write_text(
            json.dumps(episode.final_observation, indent=2),
            encoding="utf-8",
        )
    if episode.final_event_log is not None:
        (episode_dir / "event_log.txt").write_text("\n".join(episode.final_event_log), encoding="utf-8")
    summary = {
        "winner": episode.winner,
        "placements": episode.placements,
        "turns": episode.turns,
        "truncated": episode.truncated,
    }
    (episode_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if episode.final_state is not None:
        player_stats = episode.final_state.all_player_performance_summaries()
        (episode_dir / "player_stats.json").write_text(json.dumps(player_stats, indent=2), encoding="utf-8")
        lines = ["# Final Player Breakdown", ""]
        for stats in player_stats:
            lines.extend(
                [
                    f"## P{stats['player_id']} ({stats['color_name']})",
                    f"- Total VP: {stats['total_victory_points']}",
                    f"- Visible VP: {stats['visible_victory_points']}",
                    f"- Hidden VP: {stats['hidden_victory_points']}",
                    f"- Settlements Built: {stats['settlements_built']}",
                    f"- Cities Built: {stats['cities_built']}",
                    f"- Roads Built: {stats['roads_built']}",
                    f"- Longest Road Length: {stats['longest_road_length']}",
                    f"- Played Knights: {stats['played_knights']}",
                    f"- Has Longest Road: {stats['has_longest_road']}",
                    f"- Has Largest Army: {stats['has_largest_army']}",
                    f"- Total Resources: {stats['total_resources']}",
                    f"- Resources: {json.dumps(stats['resources'], sort_keys=True)}",
                    f"- Development Cards: {json.dumps(stats['development_cards'], sort_keys=True)}",
                    f"- New Development Cards: {json.dumps(stats['new_development_cards'], sort_keys=True)}",
                    "",
                ]
            )
        (episode_dir / "player_breakdown.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    session_dir = create_session_dir(base_output_dir)

    env_config = CatanEnvConfig(
        seed=args.seed,
        allow_domestic_trade=False,
        max_turns=args.max_turns,
        reward_config=build_reward_config(args),
    )
    from catan_rl_env import CatanRLEnv

    env = CatanRLEnv(env_config)
    observation = env.reset(seed=args.seed)
    model = CatanPolicyValueNet(
        observation["tensor_spec"],
        action_feature_dim=observation["action_tensor_spec"]["action_feature_size"],
    )
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        rollout_episodes=args.rollout_episodes,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
        checkpoint_dir=str(session_dir / "checkpoints"),
    )
    trainer = CatanPPOTrainer(model, env_config=env_config, ppo_config=ppo_config)
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint, load_optimizer=True)

    run_config = {
        "env_config": asdict(env_config),
        "ppo_config": asdict(ppo_config),
        "updates": args.updates,
        "eval_episodes": args.eval_episodes,
        "session_dir": str(session_dir),
        "resume_checkpoint": args.resume_checkpoint,
        "resumed_update_step": trainer.update_step,
    }
    (session_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for _ in range(1, args.updates + 1):
        global_update = trainer.update_step + 1
        episodes = trainer.collect_episodes(
            num_episodes=args.rollout_episodes,
            seed=args.seed + global_update * 1000,
            deterministic=False,
        )
        train_metrics = trainer.train_update(episodes)
        eval_episodes = trainer.collect_episodes(
            num_episodes=args.eval_episodes,
            seed=args.seed + global_update * 2000,
            deterministic=True,
        )
        eval_metrics = summarize_evaluation_metrics(eval_episodes)
        update_dir = session_dir / f"update_{global_update:04d}"
        update_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = update_dir / "train_metrics.json"
        eval_metrics_path = update_dir / "eval_metrics.json"
        save_metrics(metrics_path, train_metrics)
        save_metrics(eval_metrics_path, eval_metrics)
        checkpoint_path = session_dir / "checkpoints" / f"checkpoint_{global_update:04d}.pt"
        trainer.save_checkpoint(
            checkpoint_path,
            extra={
                "train_metrics": asdict(train_metrics),
                "eval_metrics": asdict(eval_metrics),
                "resume_checkpoint": args.resume_checkpoint,
            },
        )
        if episodes:
            save_episode_artifacts(update_dir, "train_final_episode", episodes[-1])
        if eval_episodes:
            save_episode_artifacts(update_dir, "eval_final_episode", eval_episodes[-1])
        update_training_run_reports(base_output_dir)
        print(
            f"update={global_update} train_steps={train_metrics.steps_collected} "
            f"train_avg_reward={train_metrics.average_episode_reward:.4f} "
            f"train_avg_turns={train_metrics.average_episode_turns:.2f} "
            f"train_truncations={train_metrics.truncation_count} train_avg_final_vp={train_metrics.average_final_vp:.4f} "
            f"policy_loss={train_metrics.policy_loss:.4f} value_loss={train_metrics.value_loss:.4f} "
            f"entropy={train_metrics.entropy:.4f} explained_var={train_metrics.explained_variance:.4f} "
            f"eval_avg_reward={eval_metrics.average_episode_reward:.4f} "
            f"eval_truncations={eval_metrics.truncation_count} eval_avg_final_vp={eval_metrics.average_final_vp:.4f} "
            f"eval_first_place={eval_metrics.average_first_place_rate:.4f} "
            f"eval_avg_turns={eval_metrics.average_episode_turns:.2f}"
        )


if __name__ == "__main__":
    main()
