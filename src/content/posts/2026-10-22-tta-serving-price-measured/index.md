---
title: "One Good Question Repriced TTA: Ten Views Now Cost 8%"
description: "A good question came up about the test-time augmentation post, covering something my benchmark hadn't: what happens if you batch the flipped copies together with the originals instead of running a second pass? The answer turned into a full repricing. Sequential TTA pays sticker price at every batch size — 2x for a flip, 10x for the full view set, always. Concatenated TTA rides the GPU's fixed-overhead floor: at serving-realistic batch sizes the flip is free to 1.4x, and at batch 1 the entire ten-view ensemble — the biggest accuracy gain this blog has measured — costs eight percent."
slug: "tta-serving-price-measured"
date: 2026-10-22
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["tta", "inference", "gpu", "deployment", "performance"]
---

The [test-time augmentation post](/posts/test-time-augmentation-measured) ended with a price list: +0.56 accuracy for 2.0x inference cost, +0.71 for 10.0x, measured cleanly and scaling exactly with view count. Then a sharp question about it came my way — one my benchmark had quietly dodged: those timings ran the flipped copies as a *second forward pass* — what happens if you concatenate them into the same batch? Does the overhead stay linear, or does something nonlinear happen when the batch doubles?

I measured two rows to answer the question, and the two rows demanded the full table, because they implied the original post's price list is only the *worst-case menu*. Here's the sweep: five batch sizes, the usual CIFAR ResNet18, warmed-up CUDA-event timings, median of up to 100 reps. "Sequential" launches one forward pass per view (how my original benchmark did it, and how most TTA tutorial code does it); "concat" stacks all views into a single batch and launches once.

## The price depends on how you ask

| batch | plain latency | flip, sequential | flip, concat | 10-view, sequential | 10-view, concat |
|---|---|---|---|---|---|
| 1 | 1.98 ms | 2.01x | **1.07x** | 10.26x | **1.08x** |
| 8 | 1.95 ms | 2.01x | **0.97x** | 10.30x | 1.55x |
| 32 | 1.85 ms | 2.02x | 1.45x | 10.20x | 7.88x |
| 128 | 4.95 ms | 2.00x | 2.38x | 9.95x | 12.45x |
| 512 | 24.7 ms | 2.00x | 1.99x | 9.99x | 10.03x |

Start with the sequential columns, because their flatness is the clue to everything else: 2.0x and 10x *at every batch size*, top to bottom. Each extra forward pass pays the full launch-and-sync toll no matter how empty the GPU is. Now look at the plain-latency column: from batch 1 to batch 32, latency barely moves — 1.98, 1.95, 1.85 ms. That's the fixed-overhead floor the [serving-crossover post](/posts/cpu-vs-gpu-serving-crossover) kept running into: at small batches the GPU isn't computing, it's commuting, and a 32-image batch finishes in the same time as a single image.

Concat exploits exactly that slack. At batch 1, stacking ten views makes a batch of ten — still far below the floor's capacity — so the full view set rides along at **1.08x**. The ensemble that won +0.71 in the accuracy post, the single largest reliable gain this blog has measured on any knob, costs eight percent of latency at the batch size most online services actually serve. The flip-only version is measured at 0.97x–1.07x down there, which is to say: free, with the sub-1.0 reading being ordinary timing noise on identical floor-bound launches. Even at batch 32, concat charges 1.45x for the flip against sequential's 2.02x.

![Two elevators making the same trip: one carrying a single rider, the other comfortably carrying ten](./figure-1.jpg)

## Where the discount dies, and a wrinkle I didn't expect

The discount is a loan from the GPU's idle capacity, and it's repaid as the batch fills the machine. By batch 512 the two implementations converge to identical sticker price (1.99x vs 2.00x; 10.03x vs 9.99x) — a saturated GPU has no slack to hide views in, and *how* you deliver the work stops mattering. That's the regime my original benchmark measured, which is why it saw clean multiples; it wasn't wrong, it was just quoting the peak-hour fare.

The wrinkle is batch 128, where concat is *worse* than sequential: 2.38x vs 2.00x for the flip, 12.45x vs 9.95x for ten views. This isn't a scheduling spike — it's the per-image efficiency curve being non-monotonic. On this GPU, batch 128 happens to be an unusually efficient shape (4.95 ms ≈ 38.7 µs/image, better than batch 512's 48.3 µs/image), so concatenating *out of* that sweet spot into batch 256 or 1280 means paying the neighboring, worse per-image rate. Sequential, by contrast, gets to run the sweet-spot shape repeatedly. The lesson isn't "concat is dangerous"; it's that batch-shape performance has local terrain, and a one-line change that doubles your shape can land anywhere on it. If your serving batch sits mid-range, measure both — it's a five-minute benchmark.

So the decision rule, compact enough for a sticky note: compute `batch × views`. If that product stays inside your GPU's flat region (here, up to roughly 64–100 images), TTA via concat is nearly free and you should take the accuracy; if it lands beyond saturation, TTA costs its full multiplier no matter the implementation; in between, measure. And never ship the sequential loop from tutorial code without checking — at small batches it pays double for literally nothing.

Housekeeping: this revises the [TTA post's](/posts/test-time-augmentation-measured) cost column from "2.0x/10.0x" to "up to 2.0x/10.0x, as low as ~1.0x/1.1x at small batches with concatenated views" — the accuracy numbers stand untouched. And a note to self worth keeping: this post exists because someone asked a sharper question about my benchmark than I had asked myself — the entire audit-the-defaults ethos pointed back at me, and exactly the kind of question I hope these posts keep attracting. Scope: one GPU (RTX 4080 SUPER), fp32, one small model whose images are tiny — bigger inputs fill the floor faster, shrinking the free region; the terrain is yours to map, the method travels.

---

**P.S. (2026-10-27).** Two more good questions came in after this went up: does anything change past batch 512, and does the picture hold for other layer configs? I had the harness lying around, so I ran both. Past 512, it turned out to be boring in the best way — per-image cost pins at ~49 µs from 512 all the way through 4096, with sequential and concat both sitting at exactly 2.0x; I half expected something weird at the big shapes and got a flat line. The layer configs were the interesting part. Stock ResNet-18 (7×7 stride-2 stem plus maxpool, which shrinks activations much earlier than my CIFAR stem) keeps its discount much longer: at batch 512 concat was still only 1.65x against sequential's 2.2x, and full price didn't arrive until around 1024. ResNet-50 went the other way — discount mostly gone by 512, though batch 32 still rode at 1.17x. So it was never really about the model: it's about where your card saturates, and the architecture just moves that line. Heavier activations pull it left, lighter ones push it right.
