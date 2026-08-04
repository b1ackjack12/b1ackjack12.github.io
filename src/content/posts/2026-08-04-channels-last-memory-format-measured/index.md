---
title: "channels_last: 1.4x Faster in bf16, 4x Slower in fp16, on the Same GPU"
description: "The memory-format one-liner behaved exactly as documented with bfloat16 and catastrophically backfired with float16 on my stack. Timings, the forward/backward split that localizes the problem, and the layout gotchas nobody warns you about."
slug: "channels-last-memory-format-measured"
date: 2026-08-04
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "performance", "gpu", "mixed-precision", "deep-learning"]
---

The performance checklist I've been working through on this blog — [cuDNN and TF32 flags](/posts/tf32-cudnn-benchmark-measured), then [autocast](/posts/mixed-precision-autocast-in-practice) — has one classic line left on it:

```python
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)
```

PyTorch stores image tensors as NCHW by default — all of channel 0's pixels, then channel 1's. Tensor cores prefer the interleaved NHWC layout, where each pixel's channels sit together, and `channels_last` gives you NHWC without changing the tensor's logical shape: same indices, different bytes underneath. The docs promise meaningful speedups for convnets with mixed precision on modern GPUs. Having just set up AMP, this was the obvious next test: same ResNet18 training step, same RTX 4080 Super, PyTorch 2.6.0, median of 50 iterations.

This turned out to be the strangest table this series has produced.

## The table with a hole in it

Training step (forward + backward + optimizer), with and without `channels_last`:

| batch | precision | NCHW | channels_last | verdict |
|---|---|---|---|---|
| 32 | fp32 | 17.2 ms | 50.8 ms | **3.0x slower** |
| 32 | bf16 | 11.1 ms | 9.1 ms | 1.22x faster |
| 32 | fp16 | 10.9 ms | 40.3 ms | **3.7x slower** |
| 128 | fp32 | 77.1 ms | 245.4 ms | **3.2x slower** |
| 128 | bf16 | 47.2 ms | 34.8 ms | 1.36x faster |
| 128 | fp16 | 42.5 ms | 172.5 ms | **4.1x slower** |

The bf16 row is the documentation coming true: 22–36% faster from one line, on top of everything the previous two posts already banked. Chained together — TF32, autocast, channels_last, all in bf16 — the batch-128 training step went from last week's 75.8 ms to 34.8 ms, a 2.2x total that cost four lines of code.

The fp32 rows going backwards is at least a known caveat: without tensor-core-friendly kernels to exploit the layout, you pay transposition costs for nothing. The docs quietly assume mixed precision, and now you've seen the price of ignoring that assumption — you don't just fail to gain, you triple your step time.

The fp16 rows are the hole. fp16 with `channels_last` is *the* canonical combination — it's the pairing NVIDIA's tensor-core guides are written around — and on my stack it trains four times slower than plain NCHW fp16. Not a few percent of noise: 42.5 ms to 172.5 ms.

## Localizing it

My first suspicion was the backward pass, since inference-only numbers looked healthy. So I timed forward and backward separately for fp16, batch 32:

| | forward | backward |
|---|---|---|
| NCHW | 3.9 ms | 7.1 ms |
| channels_last | 14.9 ms | 23.2 ms |

Both halves fall off the same cliff — but only when gradients are in play. Under `torch.inference_mode()`, the identical model and input in fp16 + `channels_last` was *faster* than NCHW (2.46 vs. 3.00 ms at batch 32), exactly as advertised. Same tensors, same layout, same GPU; the only difference is whether autograd is recording. That pattern points at cuDNN algorithm selection taking a bad path for NHWC fp16 convolutions in the training graph on this particular stack — `benchmark=True` was on and fully warmed up, so it's not the autotuner being cold; whatever kernels are reachable in that configuration are simply slow ones.

I want to be careful about scope here: this is one GPU, one PyTorch build (2.6.0+cu126), one OS, one architecture. I am not claiming fp16 + channels_last is broken in general — bf16 on the same machine behaves perfectly, which is strong evidence the mechanism is a narrow kernel-selection gap rather than anything fundamental. But that's precisely the kind of gap that never appears in documentation, release notes, or tutorials, and it inverts the standard advice: on this stack, following the canonical recipe would have quadrupled my training time, and the "second choice" dtype is the one that delivers.

![A fork in a glowing racetrack: one lane smooth and fast, the parallel lane of the same track full of hidden speed bumps](./figure-1.jpg)

## Two layout gotchas for the road

While setting up the benchmark I also confirmed the two behaviors that make `channels_last` a mild footgun even when it's fast.

First, the format is contagious by design: elementwise math, `cat`, and most ops *preserve* channels-last-ness, so one converted input quietly converts everything downstream. That's how it's supposed to work — you convert the model and inputs once at the boundary — but it means a stray converted tensor from a cache or a preprocessing step can flip layouts somewhere deep in a pipeline without any visible change in shapes or values.

Second, `.view()` on a channels-last tensor can simply refuse: `RuntimeError: view size is not compatible with input tensor's size and stride`. Code that flattens feature maps with `x.view(n, -1)` — which describes an enormous amount of model code, including some of mine — breaks the moment its input goes NHWC. `reshape()` or `flatten()` handle it (at the cost of a copy when needed), but if you adopt `channels_last`, that error message will eventually find you.

## Where this leaves the checklist

For my own training runs the outcome is clean: **bf16 + channels_last is now the default** — it stacked another 1.2–1.4x on top of autocast and it dodges both the fp16 scaler machinery and whatever kernel pothole fp16 NHWC hits here. `channels_last` in fp32 stays off, always. And fp16 + channels_last, the combination every tensor-core guide leads with, goes into the same bin as `cudnn.benchmark` from two posts ago: advice that was presumably true on the hardware and software it was written on, transplanted forward by repetition rather than by anyone re-checking it. Four flags into this series, the pattern is consistent enough that I'd state it as a rule: the older and more canonical the performance advice, the more it needs a stopwatch held to it on your actual stack before it touches a real training run.
