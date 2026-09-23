---
title: "The GPU Has a 0.8 ms Floor, and My Sensor Model Lives Under It"
description: "Head-to-head serving latency for four model sizes on CPU (ONNX Runtime) versus GPU (PyTorch fp16, copies included): at batch 1 the tiny IMU model runs 11.7x faster on CPU, because a GPU inference can't cost less than its launch-and-copy overhead no matter how small the network is. One batch dimension later, the GPU wins everything by up to 62x. The crossover is sharp, cheap to measure, and worth real money."
slug: "cpu-vs-gpu-serving-crossover"
date: 2026-09-23
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["deployment", "gpu", "cpu inference", "onnx", "latency"]
---

I've measured [CPU serving](/posts/cpu-inference-threading-measured) and [GPU batching](/posts/gpu-inference-batching-measured) in separate posts, which quietly dodged the question that actually decides a deployment: *for this specific model, do I need a GPU at all?* The folk answer is "GPU is faster, CPU is cheaper," which treats speed as a property of the hardware. It isn't — it's a property of the pairing. So, the head-to-head: four models spanning my whole zoo (the 0.1M-parameter HAR CNN, the 1.15M SmallResNet, ResNet18, ResNet50), each served both ways — CPU via ONNX Runtime at default settings, GPU via PyTorch fp16 *including* the host-to-device copy from pinned memory, because your production inputs don't start life on the GPU — at batch sizes 1, 16 and 128.

## The table with a diagonal line through it

| model | batch 1 | batch 16 | batch 128 |
|---|---|---|---|
| HAR CNN (0.1M) | **CPU wins 11.7x** (0.066 vs 0.77 ms) | GPU 1.3x | GPU 8.8x |
| SmallResNet (1.15M) | **CPU wins 1.6x** (0.57 vs 0.91 ms) | GPU 8.9x | GPU 61.8x |
| ResNet18 (11M) | GPU 2.5x | GPU 39.5x | GPU 51.5x |
| ResNet50 (24M) | GPU 2.4x | GPU 39.3x | GPU 38.8x |

Read the batch-1 GPU column of the two small models: 0.77 ms and 0.91 ms — nearly the same number for models that differ 10x in size. That's the tell. At this scale the GPU isn't computing; it's *administrating* — kernel launches, the H2D copy, synchronization — and that overhead has a floor around **0.8 ms on my machine** that exists regardless of how little work follows it. The HAR model's entire forward pass costs 0.066 ms on a CPU, so the GPU loses by an order of magnitude before its first kernel finishes queuing; [quantize the model to int8](/posts/ort-graph-optimization-levels) and CPU's margin stretches to nearly 30x. The rule this table teaches is pleasantly mechanical: **measure your model's CPU batch-1 latency, and if it's under the GPU's overhead floor, the GPU cannot win that workload** — not with more money, not with a better card, because you'd be paying to accelerate the part that was never the bottleneck.

![A tiny package being loaded onto a massive cargo freighter whose loading crane takes longer than a bicycle courier who has already delivered it](./figure-1.jpg)

## One batch dimension changes the winner everywhere

The same HAR model that humiliated the GPU at batch 1 loses at batch 16 and loses 8.8x at batch 128 — and notice *how* the GPU wins: its latency barely moves (0.77 → 0.54 → 0.61 ms) while CPU time grows linearly with work. The GPU's overhead floor doesn't shrink; it *amortizes*. Feed it 128 inputs and the fixed cost divides by 128 while the compute — the thing GPUs are actually good at — finally dominates. SmallResNet makes the same point louder: a 62x GPU win at batch 128, from a model that CPU served faster at batch 1. There is no such thing as "this model is faster on GPU"; there is only "this model *at this batch size* is faster on GPU," and the flip can happen between batch 1 and batch 16.

Which turns the deployment question into a question about your *traffic*, not your model. Online serving of single sensor windows — the natural shape of an IMU product, one prediction per device per moment — is a batch-1 workload, and for my HAR model the correct hardware is the CPU it's already running on: faster, simpler, no CUDA driver in the container, no GPU instance on the bill. Offline scoring, backfills, or anything where requests can pool for even a few milliseconds is a batched workload, and there the GPU is worth 9–62x even for tiny models. The [threading post's](/posts/cpu-inference-threading-measured) concurrency results compose with this: a fleet of single-threaded CPU sessions is the batch-1 answer, dynamic batching onto a GPU is the throughput answer, and the worst configuration in the whole space is the default one — a big model on CPU (ResNet50 at batch 128: 1.7 *seconds* per batch) or a tiny model on GPU, each paying for the other's hardware. Scope: one desktop CPU, one consumer GPU, PyTorch eager fp16 rather than TensorRT — a serving-optimized runtime would lower the GPU floor somewhat, but it can't repeal it, and under it my smallest model comfortably lives.
