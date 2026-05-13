"""
Training loop for DQN Pong agent.
Runs episodes, logs metrics, saves checkpoints, generates plots,
and auto-exports to TF.js format on completion.
"""

import os
import argparse
import re
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from environment import PongEnv
from agent import DQNAgent

# --- Config ---
NUM_EPISODES = 50_000
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PLOTS_DIR = os.path.dirname(__file__)
MAX_STEPS_PER_EP = 2_000  # safety cap; shorter episodes train much faster
LEARNING_STARTS = 1_000
TRAIN_EVERY = 8
LOSS_LOG_EVERY = 20
TRAIN_WIN_SCORE = 3


def _episode_from_checkpoint(path):
    """Infer the last completed episode from checkpoint metadata or filename."""
    match = re.search(r"ep_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else 0


def train(
    resume_path=None,
    epsilon_override=None,
    episodes=NUM_EPISODES,
    max_steps_per_ep=MAX_STEPS_PER_EP,
    win_score=TRAIN_WIN_SCORE,
    train_every=TRAIN_EVERY,
    learning_starts=LEARNING_STARTS,
    log_interval=LOG_INTERVAL,
    save_interval=SAVE_INTERVAL,
    export=True,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = PongEnv(win_score=win_score)
    agent = DQNAgent()
    start_ep = 1

    if resume_path:
        ckpt = agent.load(resume_path)
        last_ep = ckpt.get("episode") or _episode_from_checkpoint(resume_path)
        start_ep = last_ep + 1
        print(f"→ Resuming from {resume_path} at episode {start_ep}")
        if epsilon_override is not None:
            agent.epsilon = epsilon_override
            print(f"→ Overriding epsilon to {agent.epsilon:.4f}")

    # Metrics accumulators
    ep_rewards = []
    ep_wins = []
    ep_steps = []
    ep_losses_vals = []
    avg_rewards_log = []
    avg_winrate_log = []
    avg_loss_log = []
    avg_q_log = []

    recent_states = []  # for avg Q-value computation

    t_start = time.time()
    total_steps = 0

    print(
        "Training config: "
        f"episodes={episodes}, win_score={win_score}, "
        f"max_steps={max_steps_per_ep}, train_every={train_every}"
    )

    for ep in range(start_ep, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        ep_loss = []
        info = {}
        episode_steps = 0

        for step in range(max_steps_per_ep):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1
            episode_steps += 1

            agent.store(state, action, reward, next_state, done)
            if total_steps >= learning_starts and total_steps % train_every == 0:
                should_log_loss = total_steps % (LOSS_LOG_EVERY * train_every) == 0
                loss = agent.train_step(return_loss=should_log_loss)
                if loss is not None:
                    ep_loss.append(loss)

            total_reward += reward
            state = next_state

            # Collect states for Q-value logging
            if len(recent_states) < 1000:
                recent_states.append(state)

            if done:
                break

        agent.decay_epsilon()
        ep_rewards.append(total_reward)
        ep_wins.append(1.0 if info.get("winner") == "ai" else 0.0)
        ep_steps.append(episode_steps)
        if ep_loss:
            ep_losses_vals.append(np.mean(ep_loss))

        # --- Logging ---
        if ep % log_interval == 0:
            avg_r = np.mean(ep_rewards[-log_interval:])
            avg_w = np.mean(ep_wins[-log_interval:]) * 100
            avg_steps = np.mean(ep_steps[-log_interval:])
            avg_l = np.mean(ep_losses_vals[-log_interval:]) if ep_losses_vals else 0.0
            avg_q = agent.avg_q(np.array(recent_states[-200:], dtype=np.float32)) if recent_states else 0.0

            avg_rewards_log.append(avg_r)
            avg_winrate_log.append(avg_w)
            avg_loss_log.append(avg_l)
            avg_q_log.append(avg_q)

            elapsed = time.time() - t_start
            eps_per_sec = (ep - start_ep + 1) / elapsed
            steps_per_sec = total_steps / elapsed

            print(
                f"Ep {ep:>6d}/{episodes} | "
                f"Reward: {avg_r:>7.2f} | "
                f"Win%: {avg_w:>5.1f}% | "
                f"Loss: {avg_l:.4f} | "
                f"AvgQ: {avg_q:.3f} | "
                f"Eps: {agent.epsilon:.4f} | "
                f"Steps/ep: {avg_steps:.0f} | "
                f"{eps_per_sec:.1f} ep/s | "
                f"{steps_per_sec:.0f} steps/s"
            )

        # --- Checkpoint ---
        if ep % save_interval == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ep_{ep:06d}.pt")
            agent.save(path, episode=ep)
            print(f"  → Checkpoint saved: {path}")

    # --- Final save ---
    final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    agent.save(final_path, episode=episodes)
    print(f"\n✓ Training complete. Final checkpoint: {final_path}")

    # --- Plots ---
    _save_plots(avg_rewards_log, avg_winrate_log, avg_loss_log, log_interval)

    # --- Auto-export ---
    if export:
        print("\n→ Exporting to TensorFlow.js...")
        from export import export_to_tfjs
        export_to_tfjs(final_path)


def _save_plots(rewards, winrates, losses, log_interval=LOG_INTERVAL):
    """Generate and save training curves."""
    x = [i * log_interval for i in range(1, len(rewards) + 1)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, rewards, linewidth=0.8, color="#4fc3f7")
    ax.set_title("Average Reward per 100 Episodes", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "reward_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, winrates, linewidth=0.8, color="#81c784")
    ax.set_title("Win Rate % per 100 Episodes", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win %")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "winrate_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x[:len(losses)], losses, linewidth=0.8, color="#e57373")
    ax.set_title("Average Loss per 100 Episodes", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)

    print("✓ Plots saved: reward_curve.png, winrate_curve.png, loss_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume the DQN Pong agent.")
    parser.add_argument(
        "--resume",
        help="Path to a checkpoint, e.g. training/checkpoints/ep_005000.pt",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Optional epsilon override when resuming, e.g. 0.2",
    )
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EP)
    parser.add_argument("--win-score", type=int, default=TRAIN_WIN_SCORE)
    parser.add_argument("--train-every", type=int, default=TRAIN_EVERY)
    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--no-export", action="store_true", help="Skip TF.js export after training.")
    args = parser.parse_args()
    train(
        resume_path=args.resume,
        epsilon_override=args.epsilon,
        episodes=args.episodes,
        max_steps_per_ep=args.max_steps,
        win_score=args.win_score,
        train_every=args.train_every,
        learning_starts=args.learning_starts,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        export=not args.no_export,
    )
