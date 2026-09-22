---
title: "The One Knob I Never Swept Was the Optimizer. Adam Beat My Tuned SGD by 0.7"
description: "A month of paired t-tests over learning rates, schedules, EMA and augmentation — all of it on top of SGD, chosen by reflex and folklore ('SGD generalizes better for convnets'). A proper face-off with per-optimizer tuning at my fixed 10-epoch budget: Adam wins by +0.71, runs quieter across seeds, and — against the second piece of folklore — plain Adam with coupled L2 beats a tuned AdamW."
slug: "sgd-vs-adam-fixed-budget"
date: 2026-09-22
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "optimizers", "adam", "sgd", "hyperparameters"]
---

When I ranked [training knobs by effect size](/posts/training-knob-effect-sizes), I swept the learning rate, the weight decay, the batch size and the augmentation — and never once questioned the `torch.optim.SGD` at the top of the file. That wasn't a decision; it was a reflex backed by the most durable folklore in deep learning: *SGD generalizes better than Adam for convnets*. Every baseline this blog has produced — the [seed-noise study](/posts/seed-noise-vs-real-improvements), the [EMA sweep](/posts/ema-weights-measured), the [early-stopping autopsy](/posts/early-stopping-checkpoint-selection-measured) — sits on that reflex. So, a fair fight at my standard fixed budget (CIFAR-10 ResNet18, 10 epochs, cosine): each optimizer family gets a one-seed screening grid over its own learning rates and weight decays, the winner of each grid advances to a five-seed final. No optimizer gets my SGD-tuned hyperparameters imposed on it — that's how these comparisons usually get rigged, in either direction.

## The final

| optimizer (tuned) | test accuracy (5 seeds) | vs SGD |
|---|---|---|
| SGD, lr 0.05, wd 5e-4 | 88.95 ±0.31 | — |
| **Adam, lr 1e-3, wd 5e-4 (coupled L2)** | **89.66 ±0.14** | **+0.71** |
| AdamW, lr 1e-3, wd 0.005 | 89.18 ±0.16 | +0.23 |

Adam wins by +0.71 — nearly five standard errors of the difference (t ≈ 4.7), larger than every training-side effect I've measured this month, and it comes with a bonus I didn't expect: **less than half the seed variance** (±0.14 against SGD's ±0.31). The optimizer I never swept was worth nine EMAs. For a blog whose founding sin was [publishing untested defaults](/posts/batch-512-textbook-fixes-measured), there's something fitting about discovering that the deepest default of all — the one below the hyperparameters, the thing the hyperparameters belong *to* — was costing seven-tenths of a point the whole time.

But the folklore isn't wrong so much as *scoped*: "SGD generalizes better" comes from the 100-to-200-epoch regime where SGD's noisier trajectory has time to pay off. At 10 epochs, the race is mostly about how fast you can descend, and adaptive per-parameter step sizes are simply better at descending. My result doesn't overturn the long-schedule claim — it bounds it. Optimizer advice without a schedule attached is as incomplete as [LR-scaling advice without one](/posts/batch-512-textbook-fixes-measured), and short fine-tuning-length budgets are where most practical training now happens.

![Two racers on a short track: a steady long-distance runner built for marathons losing to a sprinter perfectly geared for the distance actually being run](./figure-1.jpg)

## Two footnotes that earn their keep

**Plain Adam beat AdamW.** The modern reflex says always AdamW — decoupled weight decay is *correct*, coupled L2 in Adam is a known theoretical wart. My tuned AdamW finals at 89.18 against plain Adam's 89.66: the theoretically wrong optimizer won by half a point. I screened AdamW at wd 0.05 and 0.005 (89.50 and 89.70 on the screening seed) so it wasn't sabotaged by an untuned decay; the coupled variant just worked better on this short schedule and small dataset. Theoretical cleanliness and empirical performance are, once again, separate axes. Also worth knowing: Adam's learning-rate cliff is real and close — at lr 3e-3 it scored 87.67, worse than everything in the table, so its lead lives inside a narrower LR window than SGD's.

**The screening flattered the winner.** Adam's screening run scored 90.02; the identical config and seed in the finals scored 89.70. Same seed, same code, same GPU — a 0.32 gap, courtesy of [the GPU non-determinism I measured in July](/posts/pytorch-same-seed-different-model). One-seed screening numbers are for *choosing* configs, never for *reporting* them; if I'd quoted the screen, a third of Adam's headline margin would have been atomics and floating-point reassociation. Scope line: one dataset, one small model, one short budget — flip any of those and the ranking may flip with it, which is precisely the point. The optimizer is a hyperparameter. Sweep it like one.
