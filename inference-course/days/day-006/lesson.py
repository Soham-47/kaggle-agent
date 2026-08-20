"""Day 6: tokenization and chat templates.

Trains a miniature character-level BPE tokenizer on a fixed corpus,
tokenizes a prompt, applies a chat template, and reports the token
overhead the template adds.
"""

import argparse

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the fox is quick and the dog is lazy. "
    "quick foxes jump high, lazy dogs sleep low. "
    "the quick brown fox and the lazy dog."
)
PROMPT = "Tell me a short story about a quick fox and a lazy dog."
SYSTEM = "You are a helpful assistant."


class MiniBPE:
    """A character-level byte pair encoder with a fixed merge budget."""

    def __init__(self, corpus: str, num_merges: int):
        self.vocab = sorted(set(corpus))
        self.merges: dict[tuple[str, str], str] = {}
        tokens = list(corpus)
        for _ in range(num_merges):
            counts = {}
            for a, b in zip(tokens, tokens[1:]):
                pair = (a, b)
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break
            merged = best[0] + best[1]
            self.merges[best] = merged
            self.vocab.append(merged)
            next_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    next_tokens.append(merged)
                    i += 2
                else:
                    next_tokens.append(tokens[i])
                    i += 1
            tokens = next_tokens
        self.vocab.sort()

    def encode(self, text: str) -> list[str]:
        """Greedily apply learned merges; return the token list."""
        tokens = list(text)
        changed = True
        while changed:
            changed = False
            out = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in self.merges:
                    out.append(self.merges[(tokens[i], tokens[i + 1])])
                    i += 2
                    changed = True
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens


def apply_chat_template(prompt: str) -> str:
    """Wrap a user prompt in a minimal system/user/assistant template."""
    return (
        f"<|system|>{SYSTEM}<|end|>\n"
        f"<|user|>{prompt}<|end|>\n"
        "<|assistant|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 6 lesson: tokenization and chat templates")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    num_merges = 100 if args.smoke else 400

    bpe = MiniBPE(CORPUS, num_merges)
    prompt_tokens = bpe.encode(PROMPT)
    template = apply_chat_template(PROMPT)
    template_tokens = bpe.encode(template)
    overhead = len(template_tokens) - len(prompt_tokens)

    print(f"mode={mode} vocab_size={len(bpe.vocab)} merges={len(bpe.merges)}")
    print(f"prompt_tokens={len(prompt_tokens)}")
    print(f"templated_tokens={len(template_tokens)}")
    print(f"template_overhead_tokens={overhead}")
    print(f"sample_tokens={prompt_tokens[:8]}...")


if __name__ == "__main__":
    main()