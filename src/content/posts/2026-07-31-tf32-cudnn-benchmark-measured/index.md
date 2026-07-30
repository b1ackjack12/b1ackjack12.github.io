---
title: "The 'Free Speedup' Flags, Measured: One Did Almost Nothing on My GPU"
description: "Every PyTorch performance checklist says to enable cudnn.benchmark and TF32. I timed both on an RTX 4080 — the famous flag bought 3%, the flag nobody mentions was already on, and TF32's error is 340x larger."
slug: "tf32-cudnn-benchmark-measured"
date: 2026-07-31
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "performance", "cuda", "gpu", "deep-learning"]
---

There is a pair of lines that appears near the top of every PyTorch performance checklist ever written:

```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")   # allow TF32
```

Free speed, the checklists say. I have pasted these lines into scripts for years without once measuring what they did — which, given how this blog has been going, is exactly the kind of claim I should be pointing a stopwatch at. So I did: PyTorch 2.6.0, RTX 4080 Super, median of 50 timed iterations after warmup, `torch.cuda.synchronize()` around everything. The results rearranged my mental ranking of these flags almost completely.

## TF32 on matmul: real, but count the digits

TF32 is Ampere-and-later NVIDIA hardware running float32 matrix math through tensor cores with a truncated 10-bit mantissa — float32's range, roughly half its precision, in exchange for throughput. PyTorch keeps it **off** for matmul by default; `torch.set_float32_matmul_precision("high")` turns it on.

On an 8192×8192 matmul: **31.9 ms in true fp32, 21.7 ms with TF32 — 1.47x**. Real, repeatable, and a long way from the "up to 8x" tensor-core marketing, because that headline number describes dense tensor-core utilization against a weak baseline, not your actual workload against cuBLAS's already-excellent fp32 kernels.

The cost side, which the checklists skip: I compared both modes against a float64 reference on a 2048×2048 matmul. Max absolute error was **2.1e-4 for fp32 and 7.1e-2 for TF32 — 340x larger**. Before that number alarms you, scale it: the reference values averaged ~36 in magnitude, so TF32's worst case is roughly 0.2% relative error, which two decades of mixed-precision training practice says neural networks shrug off. But "the network shrugs it off" is a statement about training loss, not about your loss-of-significance-prone accumulation, your iterative solver, or your unit tests with `atol=1e-5` that started failing after someone added one innocent-looking line to the top of the script. I have [written before about proxy metrics lying in both directions](/posts/why-perplexity-is-a-vanity-metric-for-llm-evaluation); a 340x error blowup that usually doesn't matter is exactly that shape of fact. Know which case you are.

## cudnn.benchmark: the celebrity flag that bought me 3%

`cudnn.benchmark = True` tells cuDNN to time every available convolution algorithm for your exact shapes on first encounter, then use the winner — paying an autotuning cost up front for faster steady-state convolutions. The checklist promise is significant speedup for any conv net with fixed input sizes.

Measured on a ResNet18 training step (forward + backward + step, batch 32 at 224×224, TF32 fully off to isolate the flag): **21.40 ms without, 20.87 ms with. A 1.03x speedup.** Three percent, on the most conv-heavy mainstream architecture there is. The explanation is undramatic: cuDNN's default heuristics have gotten genuinely good at picking algorithms for common shapes on recent hardware, so exhaustive search has little left to find. The flag's reputation was earned on older cuDNN versions and stranger convolution configurations, and the reputation outlived the gap.

The trade you make for that 3% is not free, and I measured it too. With `benchmark=True`, every *new input shape* triggers re-tuning. Feeding the same network five different image sizes, the first forward pass at each new size cost **103–137 ms — against 5.2 ms steady-state**. A 20-26x penalty per shape, once each. Harmless for a classifier trained at fixed resolution; actively hostile for variable-input workloads — detection pipelines with dynamic resizing, audio models with variable-length windows, [the kind of bucketed sensor batches I deal with](/posts/preparing-imu-sensor-data-for-deep-learning). If your shapes vary a lot, this famous "free speedup" flag can make your pipeline measurably slower.

![A bar chart contrast: a tiny 3 percent gain bar next to a towering per-shape retuning cost bar](./figure-1.jpg)

## The flag nobody mentions was doing the real work all along

Here is the asymmetry that surprised me most. TF32 for *matmul* is off by default and on every checklist. TF32 for *convolutions* — `torch.backends.cudnn.allow_tf32` — is **on by default** and on no checklist I've ever seen.

Same ResNet18 training step: conv TF32 off, 21.40 ms; conv TF32 on, **17.61 ms — 1.22x**, seven times the benefit of the celebrity flag, from a setting most users have never consciously enabled because PyTorch quietly enables it for them. Which means two things. First, if you've ever benchmarked "pure fp32" conv training on Ampere or newer hardware, you probably measured TF32 without knowing it. Second, if you're debugging a numerical discrepancy between GPU and CPU conv outputs, this default is on the suspect list — you can be bitten by reduced precision you never opted into.

Stacked together on this workload: all flags off, 21.40 ms; everything on, 17.61 ms. About 1.2x end to end, of which nearly all came from the default-on flag, a rounding error from the famous one, and the matmul flag contributed nothing here because ResNet18's time lives in convolutions — matmul TF32 pays off in transformer-shaped models, not conv-shaped ones. Which flag matters is a property of *your architecture*, not of the checklist.

## What goes at the top of my scripts now

For my own work: `set_float32_matmul_precision("high")` goes in by default for training (and gets flipped to `"highest"` the moment I'm chasing a numerical discrepancy — it's one line, and [I've already been burned](/posts/quantizing-onnx-models-in-practice) by assuming precision changes are free). `cudnn.benchmark` goes in only for fixed-shape training, without expectations, because on this hardware it's a 3% flag with a 130-millisecond-per-shape temper. And `cudnn.allow_tf32` I now leave alone but *know about*, which is the part that was actually missing.

The meta-lesson is the same one this blog keeps producing, and I'll keep writing it down as long as it keeps being true: performance advice ages, hardware moves, defaults shift underneath the folklore — and the flags everyone recites can be quietly outperformed by the ones nobody mentions. The two-line incantation at the top of this post took me ten years to question and one afternoon to measure. The measuring was cheaper.
