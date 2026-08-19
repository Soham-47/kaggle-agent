# 100-Day LLM Inference Engineering Course

A GPU-first, hands-on course on LLM inference engineering: serving
runtimes, KV caches, quantization, kernels, and production systems. Ten
phases, one lesson per day. Lessons draw on three references (see
sources/sources.md) but all prose is original.

## Phase summary

| Phase | Days | Focus |
|-------|------|-------|
| 1 | 1-10 | Inference foundations and requirements |
| 2 | 11-20 | Transformer execution and KV cache |
| 3 | 21-30 | GPU hardware and kernels |
| 4 | 31-40 | Serving runtimes and scheduling |
| 5 | 41-50 | Quantization |
| 6 | 51-60 | Speculative decoding and parallelism |
| 7 | 61-70 | Production systems |
| 8 | 71-80 | Embeddings and modalities |
| 9 | 81-90 | Disaggregation and advanced serving |
| 10 | 91-100 | Capstone |

## Day table

## Phase 1 — Inference foundations and requirements (days 1-10)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 001 | What inference is; the runtime/infrastructure/tooling layers | Virk |
| - [ ] | 002 | Latency, throughput, quality tradeoffs | Virk |
| - [ ] | 003 | Metrics: TTFT, TPS, ITL; percentiles vs means | Virk |
| - [ ] | 004 | Requirements: interface, latency budget, unit economics, usage pattern | Virk |
| - [ ] | 005 | Model selection and evals; shared vs dedicated | Virk |
| - [ ] | 006 | Tokenization and chat templates | aman |
| - [ ] | 007 | Prefill vs decode; the autoregressive loop | Virk, Modular |
| - [ ] | 008 | Sampling: temperature, top-k, top-p, logit bias | aman |
| - [ ] | 009 | The roofline model; compute-bound vs memory-bound | Virk |
| - [ ] | 010 | Arithmetic intensity; why decode is memory-bound | Virk |

## Phase 2 — Transformer execution and KV cache (days 11-20)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 011 | Transformer block anatomy: embeddings, norms, FFN, LM head | aman |
| - [ ] | 012 | Scaled dot-product attention with causal masking | aman |
| - [ ] | 013 | Multi-head and cross attention | aman |
| - [ ] | 014 | Reading config.json; architecture strings; MoE | Virk |
| - [ ] | 015 | The KV cache: what is cached, when built vs used | Modular |
| - [ ] | 016 | KV cache memory math and sizing | Modular |
| - [ ] | 017 | PagedAttention: blocks, lookup tables, fragmentation | Modular |
| - [ ] | 018 | FlashAttention: tiling and fusion | Virk, aman |
| - [ ] | 019 | Static vs dynamic vs continuous batching | Modular |
| - [ ] | 020 | Context window budgeting; long-context methods | Virk, aman |

## Phase 3 — GPU hardware and kernels (days 21-30)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 021 | GPU mental model: SMs, CUDA cores, tensor cores, SFU | Virk, aman |
| - [ ] | 022 | Memory hierarchy: registers to HBM; bandwidth | Virk |
| - [ ] | 023 | GPU generations: Hopper, Ada, Blackwell; MIG | Virk |
| - [ ] | 024 | Instances and interconnects: NVLink vs InfiniBand | Virk |
| - [ ] | 025 | CUDA kernels; kernel selection and fusion | Virk |
| - [ ] | 026 | Writing a kernel with Triton (guarded, optional GPU) | Virk |
| - [ ] | 027 | Model formats: safetensors, GGUF, ONNX, TensorRT | Virk |
| - [ ] | 028 | PyTorch profiler workflow | Virk |
| - [ ] | 029 | Nsight Systems and Nsight Compute | Virk |
| - [ ] | 030 | Benchmarking vs profiling; harness design | Virk |

## Phase 4 — Serving runtimes and scheduling (days 31-40)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 031 | vLLM: PagedAttention, block size flag | Modular, Virk |
| - [ ] | 032 | SGLang: RadixAttention, prefix reuse | Virk |
| - [ ] | 033 | TensorRT-LLM: compile workflow | Virk |
| - [ ] | 034 | Engine comparison; when to choose what | Virk |
| - [ ] | 035 | Prefix caching: hashing, system-prompt ordering | Modular, Virk |
| - [ ] | 036 | Cache-aware routing | Virk |
| - [ ] | 037 | Batching math: batch size vs TTFT and throughput | Virk |
| - [ ] | 038 | Queueing: arrivals, batch formation, scheduling | Virk |
| - [ ] | 039 | Structured outputs: JSON mode, grammar, tool calling | Virk |
| - [ ] | 040 | Offline batch inference | Modular |

## Phase 5 — Quantization (days 41-50)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 041 | Number formats: FP16/BF16, FP8, INT8, INT4 | Virk |
| - [ ] | 042 | Quantization basics: scales, per-tensor/per-channel/per-group | Virk |
| - [ ] | 043 | Quantization algorithms: GPTQ, AWQ, SmoothQuant | Virk |
| - [ ] | 044 | KV cache quantization; the attention risk gradient | Virk |
| - [ ] | 045 | Measuring quality loss: perplexity and evals | Virk |
| - [ ] | 046 | FP8 in serving flags; kv-cache dtype | Virk |
| - [ ] | 047 | Weight-only vs weight+activation; calibration | Virk |
| - [ ] | 048 | On-device compression: GGUF and llama.cpp | Virk |
| - [ ] | 049 | Microscaling formats | Virk |
| - [ ] | 050 | Quantization sweep: quality vs compression curve | Virk |

## Phase 6 — Speculative decoding and parallelism (days 51-60)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 051 | Decode bottleneck recap; spare compute | Virk |
| - [ ] | 052 | Speculative decoding: draft-target, acceptance | Virk, aman |
| - [ ] | 053 | Draft-target with a small draft model | Virk |
| - [ ] | 054 | Medusa: multi-head drafting | Virk |
| - [ ] | 055 | EAGLE: hidden-state drafting | Virk |
| - [ ] | 056 | N-gram and lookahead decoding | Virk |
| - [ ] | 057 | When to disable: batch size, temperature | Virk |
| - [ ] | 058 | Tensor parallelism: matmul split, all-reduce | Virk |
| - [ ] | 059 | Pipeline and expert parallelism | Virk |
| - [ ] | 060 | Hybrid parallelism; communication cost math | Virk |

## Phase 7 — Production systems (days 61-70)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 061 | Docker and the NVIDIA container runtime | Virk |
| - [ ] | 062 | A production Dockerfile for a serving engine | Virk |
| - [ ] | 063 | Autoscaling: concurrency, batching, cold starts | Virk |
| - [ ] | 064 | Routing and load balancing | Virk |
| - [ ] | 065 | Multi-cloud capacity management | Virk |
| - [ ] | 066 | Zero-downtime deployment; blue-green | Virk |
| - [ ] | 067 | Metrics: TTFT, TBT, queue depth, GPU utilization | Virk |
| - [ ] | 068 | Dashboards: building one for an inference server | Virk |
| - [ ] | 069 | Tracing: OpenTelemetry end to end | Virk |
| - [ ] | 070 | Load testing: ramping traffic to saturation | Virk |

## Phase 8 — Embeddings and modalities (days 71-80)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 071 | Embedding model inference: batching and normalization | Virk |
| - [ ] | 072 | Similarity search: embed, index, query | aman |
| - [ ] | 073 | VLM inference: image preprocessing and batching | Virk |
| - [ ] | 074 | ASR: Whisper latency optimization | Virk |
| - [ ] | 075 | TTS: streaming real-time speech | Virk |
| - [ ] | 076 | Diffusion image generation: steps, denoiser, VAE | Virk |
| - [ ] | 077 | Video generation: context parallelism | Virk |
| - [ ] | 078 | Multi-modal batching | Virk |
| - [ ] | 079 | Speech-to-speech pipeline | Virk |
| - [ ] | 080 | Long context: chunked prefill and KV offloading | Virk, Modular |

## Phase 9 — Disaggregation and advanced serving (days 81-90)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 081 | Prefill/decode disaggregation | Virk, Modular |
| - [ ] | 082 | Conditional disaggregation and xPyD ratios | Virk |
| - [ ] | 083 | Dynamic disaggregation with NVIDIA Dynamo | Virk |
| - [ ] | 084 | KV transfer over the interconnect | Virk |
| - [ ] | 085 | Long-context KV offloading and paging | Modular |
| - [ ] | 086 | MoE serving: expert parallelism and routing | Virk |
| - [ ] | 087 | Distillation for inference quality and cost | Virk, aman |
| - [ ] | 088 | Evals for a deployed model; LLM-as-judge | aman |
| - [ ] | 089 | Cost modeling: dollars per token | Virk |
| - [ ] | 090 | Capacity planning; when not to self-host | Virk |

## Phase 10 — Capstone (days 91-100)

| Done | Day | Topic | Sources |
|------|-----|-------|---------|
| - [ ] | 091 | Capstone: sketch the full stack for a use case | Virk |
| - [ ] | 092 | Build: a small FastAPI-style serving stack (stdlib, no new deps) | Virk |
| - [ ] | 093 | Load test the capstone server | Virk |
| - [ ] | 094 | Profile the capstone; find and fix the bottleneck | Virk |
| - [ ] | 095 | Quantize the capstone with a quality check | Virk |
| - [ ] | 096 | Speculative decoding acceptance experiment | Virk |
| - [ ] | 097 | Prefix caching hit-rate experiment | Modular |
| - [ ] | 098 | Autoscaling simulation | Virk |
| - [ ] | 099 | Dashboard for the capstone metrics | Virk |
| - [ ] | 100 | Reflect: lessons learned and next steps | Virk |