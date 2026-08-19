"""Day 8: sampling with temperature, top-k, top-p, logit bias.

Builds a fixed seeded logit vector, applies each sampling control, and
reports entropy, nucleus size, bias effect, and draw diversity.
"""

import argparse

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(p)).sum())


def sample(p: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(p.size, size=n, p=p)


def top_k(p: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p)[::-1][:k]
    return order, p[order] / p[order].sum()


def top_p(p: np.ndarray, p_cut: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    idx = min(int(np.searchsorted(cum, p_cut, side="left")) + 1, p.size)
    keep = order[:idx]
    return keep, p[keep] / p[keep].sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 8 lesson: sampling controls")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    n_tokens = 32 if args.smoke else 64
    n_draws = 200 if args.smoke else 400

    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 1.0, size=n_tokens)
    logits[0] += 3.0  # one dominant token

    print(f"mode={mode} n_tokens={n_tokens}")
    for t in (0.5, 1.0, 1.5):
        p = softmax(logits / t)
        print(f"temperature={t} entropy_nats={entropy(p):.3f} top_prob={p.max():.3f}")

    p1 = softmax(logits)
    keep_k, _ = top_k(p1, 5)
    keep_p, _ = top_p(p1, 0.9)
    print(f"top_k=5 kept_tokens={keep_k.size}")
    print(f"top_p=0.9 nucleus_size={keep_p.size}")

    biased = logits.copy()
    biased[7] += 3.0
    delta = softmax(biased)[7] - p1[7]
    print(f"logit_bias_plus3_delta_p={delta:+.3f}")

    draws = sample(p1, rng, n_draws)
    greedy_draws = sample(p1, np.random.default_rng(7), n_draws)
    print(f"diversity_t1={len(np.unique(draws)) / n_draws:.3f}")
    print(f"diversity_greedy={len(np.unique(np.full(n_draws, int(p1.argmax())))) / n_draws:.3f}")


if __name__ == "__main__":
    main()