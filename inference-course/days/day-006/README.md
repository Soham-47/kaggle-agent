# Day 6 — Tokenization and chat templates

## Goal

Build a miniature byte-pair-encoding tokenizer from scratch, tokenize a
prompt, apply a chat template, and measure the token overhead the
template adds.

## Prerequisites

Python 3.10 or newer. The course environment. No downloads and no
network access are needed.

## Concept

Tokenization is not a neural net. It is a deterministic string-to-id
map built from subword units, with a vocabulary usually in the tens of
thousands or more. Byte pair encoding (BPE) starts from individual
characters and repeatedly merges the most frequent adjacent pair until
the vocabulary reaches its target size. WordPiece merges by likelihood
instead of raw frequency; SentencePiece trains directly on raw text
without language-specific preprocessing. A common rule of thumb: one
token is about four characters of common English text.

The chat template is the exact string format that merges system, user,
and assistant roles into one flat sequence. It must be implemented
byte for byte. A wrong template quietly tanks quality, because the
model sees a distribution it never trained on. Template tokens count
against the context window, and a prompt that fits on its own may not
fit once the template wraps it.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson trains a miniature BPE tokenizer on a fixed corpus of about
90 characters for 100 merges (400 in full mode), tokenizes a fixed
prompt, applies a chat template with `system`, `user`, and `assistant`
special tokens, and counts the tokens with and without the template.

## Expected observations

- The vocabulary starts at the character set and grows by one unit per
  merge until the corpus runs out of frequent pairs.
- Tokens that the corpus never taught the tokenizer, such as the
  angle-bracket special tokens, split into characters: that is
  out-of-vocabulary handling at work.
- The templated prompt is much longer than the raw prompt: the system
  message and the special tokens all count as context. In a real
  deployment the template and system prompt eat the same context
  budget, which is why chat UIs budget it carefully.

## Metric

`template_overhead_tokens`: the difference between the token count of
the templated prompt and the token count of the raw prompt.

## Sources

- aman.ai, Tokenizer primer (BPE, WordPiece, SentencePiece):
  https://aman.ai/primers/ai/tokenizer/

## Hardware notes

Runs on CPU in both modes; the tokenizer is pure Python, so no GPU path
exists. Full mode only runs more merge iterations.

## Reflection prompts

1. Why does the tokenizer split an unknown word into subwords instead
   of failing?
2. Where does the "four characters per token" rule of thumb break?
3. Why must the chat template match the one used in training exactly?
4. A 4k-token context window with a 500-token system prompt: how many
   tokens are left for the answer?