---
title: "torch.compile on Windows Isn't Supposed to Work. Here's What I Got Anyway"
description: "PyTorch's compiler officially has no Windows GPU story — no Triton wheels, dead on arrival. With a community wheel, a version pin, and a MAX_PATH workaround, I got it running and timed what it's actually worth."
slug: "torch-compile-on-windows"
date: 2026-08-05
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "torch.compile", "windows", "performance", "deep-learning"]
---

Over the last week I've walked a ResNet18 training step down from 21.4 ms to 9.1 ms — [TF32 and cuDNN flags](/posts/tf32-cudnn-benchmark-measured), [autocast](/posts/mixed-precision-autocast-in-practice), [channels_last](/posts/channels-last-memory-format-measured). The elephant at the end of that checklist is `torch.compile`, PyTorch's whole-graph compiler and the headline feature of the 2.x era. And for me it has always ended the same way, because I develop on Windows:

```text
BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Cannot find a working triton installation.
```

Inductor, the default compile backend, generates GPU kernels through Triton — and Triton ships no official Windows wheels. This isn't a bug; it's the documented state of the world, and it's why my [framework comparison post](/posts/pytorch-vs-jax-2026-framework-guide) waved the feature off entirely. But "unsupported" and "impossible" aren't the same claim, and I'd never actually tested where the line is. This post is the test.

## The backends that work out of the box, and why they don't help

`torch.compile` accepts other backends, and two of them run on a stock Windows install. `backend="aot_eager"` traces the graph and replays it without generating any kernels; `backend="cudagraphs"` adds CUDA graph capture on top. Both compiled and ran my training step without complaint — and `aot_eager` came out at **17.4 ms against eager's 9.5 ms**, nearly twice as slow. No surprise once you know its purpose: it's a debugging backend for isolating tracer issues from codegen issues, not an optimization. It just also happens to be what naive "make it work on Windows" advice sometimes points at. If Inductor is off the table, honest advice is to skip compilation entirely rather than cargo-cult a backend flag.

The real path runs through an unofficial project: [triton-windows](https://github.com/woct0rdho/triton-windows), a community-maintained fork that publishes Windows wheels for Triton. Installing it is one `pip install` — and then two traps, both of which I hit within ten minutes.

## Trap one: pip gives you the wrong version

A bare `pip install triton-windows` handed me Triton 3.7, and Inductor immediately died with `ImportError: cannot import name 'AttrsDescriptor'` — an internal API that newer Tritons removed. Each PyTorch release is welded to a narrow Triton version band, and pip has no way to know that. For PyTorch 2.6 the working pin is:

```bash
pip install "triton-windows<3.3"
```

With 3.2.0 in place, a toy model compiled and ran. Then I pointed it at the actual ResNet18 training step, and it failed somewhere much stranger.

## Trap two: the 260-character wall

Every compile attempt on the real model collapsed with a `FileNotFoundError` on a temp file Triton had just tried to write. The path in the error message is the explanation, once you look at it closely: `C:\Users\VC\AppData\Local\Temp\torchinductor_VC\triton\0\` plus a 43-character hash directory, plus a UUID-stamped temp directory, plus the kernel's filename — and Inductor names fused kernels descriptively. The failing one was called `triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2.ttir`, 110 characters of filename all by itself. Total: past 260 characters, which is Windows' default `MAX_PATH` limit. The file wasn't missing; it was never allowed to exist.

So the fix for one of the more cryptic errors I've seen this year is aggressively mundane — give the caches a short home:

```powershell
$env:TORCHINDUCTOR_CACHE_DIR = "C:\tic"
$env:TRITON_CACHE_DIR = "C:\tic\triton"
```

Kernels the compiler *fuses* get longer names precisely because fusion is working — the feature's success is what pushes the path over the limit. There's a fitting irony in that.

## What it's worth: the numbers

With both traps cleared, everything compiled. Baseline is my current best configuration (bf16 autocast + channels_last), batch 32, same measurement protocol as the rest of this series:

| configuration | first step | steady state | vs. eager |
|---|---|---|---|
| eager | 0.4 s | 9.46 ms | — |
| compile, default | 43.8 s | 8.63 ms | 1.10x |
| compile, reduce-overhead | 9.4 s* | 8.28 ms | 1.14x |
| compile, max-autotune | 225 s* | 8.21 ms | 1.15x |

*The starred compile times ran after the default mode in the same process, so they partially reused cached compilation — cold-start would be worse. The 43.8 s and the 225 s are the honest cold numbers.*

At batch 128 the default mode gave the same shape of result: 35.6 ms to 31.1 ms, 1.14x. So: **10–15% steady-state, for 44 to 225 seconds of upfront compilation.** The arithmetic writes its own recommendation. Saving 0.8 ms per step, the default mode needs roughly fifty thousand steps just to repay its own compile time — a real training run clears that easily, a debugging session never does. And this measures a CNN whose eager path already runs on hand-tuned cuDNN kernels; compilation earns its reputation on models with more fusible pointwise work and Python overhead — transformers above all — where published speedups run considerably higher. On an already-optimized ResNet, 1.15x is the honest ceiling I found.

![A mountain climber reaching a summit marker that adds a small extra height to an already tall mountain, with a long winding trail behind](./figure-1.jpg)

## Verdict, and the tally for the whole series

Would I recommend this setup? With qualifications. It's an unofficial wheel maintained outside the PyTorch project — pin the version, expect each PyTorch upgrade to need a matching Triton bump, and know that `pip uninstall triton-windows` returns you cleanly to eager mode if anything misbehaves. For long training runs on a Windows box, that trade now strikes me as clearly worth it. For everyday iteration, eager mode with the previous three posts' flags remains the sane default; a compiler you wait 44 seconds for is a tool for runs measured in hours.

The week's ledger, batch 32, all measured on the same GPU: 21.4 ms with nothing enabled, 17.6 ms with the flags, 11.1 ms with autocast, 9.1 ms with channels_last, 8.2 ms compiled — **2.6x end to end**, none of it requiring different hardware or a different OS. The most effective single change was autocast; the most labor-intensive per millisecond was today's. That ordering — cheap wins first, compiler last — is the one piece of this series I'd expect to survive contact with anyone else's machine.
