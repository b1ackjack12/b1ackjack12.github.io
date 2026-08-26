---
title: "num_workers=4 Made My Training 47% Slower"
description: "Every PyTorch tutorial says to crank up DataLoader workers. On Windows, that advice backfired: 4 workers made each epoch 47% slower than no workers at all, and 8 workers made it 2.4x slower. The fix is one flag that the tutorials, written on Linux, never needed to mention."
slug: "dataloader-num-workers-windows"
date: 2026-08-26
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "dataloader", "windows", "performance", "training"]
---

The standard advice for a slow PyTorch input pipeline is one line long: set `num_workers` to something bigger than zero and let subprocesses prepare batches while the GPU trains. I've typed that line on autopilot for years. This week I finally measured it on my own machine — Windows, RTX 4080 SUPER, ResNet18 on CIFAR-10 with standard augmentation — and the standard advice, applied the standard way, made training substantially slower. Not "didn't help." Slower, by an amount that scaled with how many workers I added.

This post joins a running series on Windows-specific PyTorch behavior — [torch.compile needed version pinning and a MAX_PATH workaround](/posts/torch-compile-on-windows), [the driver silently spills VRAM to system memory](/posts/training-past-gpu-memory-limits) — and this entry has the best effort-to-payoff ratio of the lot: the fix is literally one keyword argument.

## The measurement

ResNet18 adapted for CIFAR-10, batch 128, RandomCrop + flip + normalize on the CPU side, three epochs per configuration, wall-clock time per epoch. The `first batch` column is how long the loader took to yield its first batch after the epoch started — keep an eye on it, it's the whole story.

| configuration | epoch 1 | epochs 2–3 | first batch, each epoch |
|---|---|---|---|
| workers=0 | 16.9 s | **15.2 s** | 0.03 s |
| workers=2 | 16.8 s | 15.8 s | **7.7 s** |
| workers=4 | 22.9 s | **22.4 s** | **14.6 s** |
| workers=8 | 36.5 s | **36.3 s** | **28.2 s** |
| workers=2 + persistent | 15.3 s | 7.5 s | 0.03 s after ep. 1 |
| workers=4 + persistent | 22.0 s | **7.4 s** | 0.03 s after ep. 1 |
| workers=8 + persistent | 35.3 s | 7.4 s | 0.03 s after ep. 1 |

Top half: more workers, monotonically worse. Four workers is 47% slower than none; eight workers is 2.4x slower. And the first-batch column says exactly where the time goes — with 8 workers, the loader spends the first **28 seconds of every single epoch** yielding nothing.

That 28 seconds is process startup. Windows has no `fork()`, so every DataLoader worker is spawned as a fresh Python interpreter that re-imports torch and torchvision and unpickles its own copy of the dataset — about 3.5 seconds per worker on my machine, and by default the DataLoader **tears all of them down at the end of each epoch and spawns them again**. On Linux, `fork()` makes a worker cost milliseconds, which is why the tutorials never mention any of this: the people writing them have never seen the failure mode. The advice isn't wrong; it's implicitly Linux advice.

`persistent_workers=True` is the entire fix. Workers are spawned once, survive across epochs, and the spawn tax is paid only on epoch 1. Same four workers that were 47% slower become **2x faster** than single-process — 7.4 seconds per epoch. Over a 100-epoch run, the difference between `num_workers=8` with and without that flag is 60 minutes versus 13.

![A relay race where the runners are dismissed and re-hired between every lap, versus a team that stays on the track](./figure-1.jpg)

## Where the plateau is, and the flag that did nothing

Steady-state training time bottoms out at 7.4 s/epoch for 2, 4, and 8 workers alike, so I measured the pipeline alone — same loader, no model — to see through the overlap:

| loader only | steady epoch |
|---|---|
| workers=0 | 11.2 s |
| workers=4 + persistent | 3.3 s |
| workers=8 + persistent | 1.7 s |

Data preparation scales fine: 8 workers really do produce batches 6.6x faster than the main process. But once the loader (3.3 s) is faster than the GPU's compute for the epoch (~7.4 s), it hides completely behind training and extra workers change nothing. That's the plateau — and it means the right worker count is the smallest one that gets the pipeline under the GPU time, not the biggest one your CPU survives. For this workload, two.

The comparison also puts a number on what `num_workers=0` actually costs: the main process spends 11.2 of its 15.2 seconds doing augmentation, serialized with training. The GPU sits idle for most of the epoch — on a machine where the GPU costs ten times what the CPU does. Worth fixing; just not the way I first fixed it.

One honest null result: `pin_memory`, the other flag tutorials hand out for free speed, measured 7.4 s/epoch on versus 7.4 s/epoch off. Its transfer-overlap benefit presumably matters somewhere — larger images, tighter loops — but on this workload it was pure superstition.

## What I actually changed

My loaders now read `num_workers=4, persistent_workers=True`, and nothing else changed. The general lesson is the one this blog keeps re-learning: performance advice ships with an invisible environment attached, and `num_workers` turns out to be a Linux idiom that Windows punishes at three and a half seconds per worker per epoch. The loop that told me so — time per epoch *and* time to first batch, printed side by side — takes ten lines, and the first-batch column identified the problem before I'd formed a theory. Numbers as usual: one machine, torch 2.6.0+cu126, CIFAR-sized samples; your plateau will sit wherever your GPU-to-augmentation ratio puts it.
