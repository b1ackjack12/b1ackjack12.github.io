---
title: "EMA's Popular Default Lost Me 3 Points, and the Control Group Dropped a Bombshell"
description: "Weight averaging is sold as free accuracy, so I measured it with the cleanest experiment this blog has run: two EMA copies maintained inside the same training run, compared against the raw weights they shadow — seed noise cancels exactly. The tutorial default decay lost 3 points in 15 out of 15 runs. The tuned one earned +0.08. And the baseline column quietly rewrote a month of this blog's statistics."
slug: "ema-weights-measured"
date: 2026-09-10
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "ema", "training", "hyperparameters", "statistics"]
---

Exponential moving averages of model weights have the best reputation of any training trick: keep a smoothed copy of the parameters, evaluate that instead of the raw endpoint, collect "free accuracy" — every big vision and diffusion codebase ships it. It also has a property that makes it uniquely cheap to audit: the EMA copy lives *inside* the training run. One run yields both the raw weights and their shadow, trained on identical batches from identical initialization — a paired comparison with [the split-luck subtracted exactly](/posts/har-augmentation-fifty-splits), for zero extra GPU time. I ran fifteen seeds of my usual 10-epoch CIFAR ResNet18 (at lr 0.05, [the measured optimum](/posts/training-knob-effect-sizes)), each maintaining two EMA copies: decay 0.99 and the widely copied default 0.999.

## The verdict, both directions

| weights evaluated | accuracy (15 seeds) | paired delta vs raw | wins |
|---|---|---|---|
| raw final weights | 88.86 ±0.17 | — | — |
| EMA, decay 0.99 | 88.94 ±0.14 | **+0.08** (t = +2.7) | 10/15 |
| EMA, decay 0.999 | 85.78 ±0.60 | **−3.08** (t = −23) | **0/15** |

The default decay didn't underperform subtly — it lost three full points, in fifteen runs out of fifteen, with a t-statistic of minus twenty-three. The arithmetic explains it: a decay of 0.999 averages over an effective window of roughly 1/(1−0.999) = 1,000 optimizer steps. My whole training run is 3,900 steps. That shadow is a blend of the final model with weights from a quarter of the schedule ago, when the cosine learning rate was still high and the model was measurably worse — it's not smoothing the endpoint, it's dragging an anchor. The codebases the default comes from train ImageNet for 450,000 steps, where a 1,000-step window is 0.2% of the run and touches only converged weights. Same lesson [the batch-512 scaling rule taught me](/posts/batch-512-textbook-fixes-measured): hyperparameter defaults ship with an invisible schedule attached, and EMA's decay is a *time constant* — copy it without copying the timescale and you copy a different algorithm.

Tuned to my schedule (decay 0.99, a ~100-step window that stays inside the low-LR tail), EMA does deliver its promise, technically: +0.08 of a point, statistically real thanks to pairing (the within-run delta has a standard deviation of just 0.11), practically a rounding error. My hypothesis for why so small: a cosine schedule annealing to zero already *is* a smoothing operator — the final epochs barely move the weights, so the endpoint arrives pre-averaged. EMA earns its keep on constant-LR schedules, mid-training checkpoints, and long runs; at a short cosine-annealed schedule there's almost nothing left for it to smooth. Free accuracy, measured: eight hundredths of a point.

![A runner dragging a long heavy chain of ghostly past versions of themselves across the finish line](./figure-1.jpg)

## The bombshell in the control group

The most consequential number in the table isn't an EMA row. It's the baseline's ±0.17.

[Back in August](/posts/seed-noise-vs-real-improvements) I measured this same pipeline's seed noise at ±1.17 and built a whole statistical sermon on it — single-seed comparisons are coin flips, detecting label smoothing needs 140 seeds, and so on. That measurement was taken at lr 0.1. Today's runs differ in exactly one knob, the [measured-optimal lr 0.05](/posts/training-knob-effect-sizes), and the seed noise didn't shrink a little — it collapsed **sevenfold**. A mistuned learning rate wasn't just costing a point of mean accuracy; it was the primary *source of the variance* I spent a week treating as a law of nature. At ±0.17, the August arithmetic inverts: label smoothing's +0.35 would be a two-sigma effect visible with a handful of seeds, not 140. The noise floor isn't a property of "training neural networks." It's a property of your configuration, and it responds to tuning like everything else.

I want to flag this honestly as the week's real finding, discovered by accident in a control column — a very on-brand way for this blog to learn something. It also retroactively explains August's worst rows: high learning rates don't just lower the mean, they widen the lottery ([batch 512 at lr 0.1: ±1.96](/posts/training-knob-effect-sizes); at lr 0.05: ±0.14). Stability and accuracy peaked at the same knob setting.

## What to write in the config file

For EMA: set the decay so the averaging window 1/(1−d) is a small fraction of your *total* steps and sits entirely inside the annealed tail — for my 3,900-step runs that's 0.99, for a 450k-step ImageNet run 0.999 or slower is genuinely right. If you're on a short cosine schedule, expect the payoff to be cosmetic either way; if you're on constant LR or you evaluate mid-training, EMA is doing real work. And if you copy a decay out of a famous repo, look up how many steps that repo trains for first.

For experiments: the within-run pairing trick generalizes and it's absurdly powerful — detecting today's +0.08 effect unpaired, against even the *good* noise floor, would need roughly seventy seeds per arm; paired, fifteen runs sufficed, and the fifteen runs were the experiment I was already running. Scope as usual: one model, one dataset, 10-epoch schedules, and EMA's reputation was earned on runs a hundred times longer — which is precisely the point.
