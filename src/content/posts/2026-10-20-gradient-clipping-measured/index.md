---
title: "My Gradient Clipping Was Never Clipping. It Was Secretly Multiplying My Weight Decay"
description: "Yesterday warmup turned out to be free insurance against a hot learning rate, so today I asked whether gradient clipping — the other famous stability knob — buys the same policy. It does not. Clipping taxed the tuned recipe, deepened the hot-rate damage, and at an aggressive threshold detonated 14 points. Then the instrumentation explained everything: on this CNN the clip threshold engages on 100% of steps because there are no spikes to clip, and what it's actually doing is rescaling gradients while Adam's coupled weight decay keeps full strength — silently multiplying the regularizer dose into the cliff zone I measured last week. One control experiment with the decay removed recovered 14.8 of the lost points."
slug: "gradient-clipping-measured"
date: 2026-10-20
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "gradient clipping", "weight decay", "hyperparameters", "training"]
---

[Yesterday's experiment](/posts/lr-warmup-measured) ended with a clean verdict: LR warmup is insurance — free at the tuned learning rate, a full payout at an overheated one. Gradient clipping has the exact same folklore job description. It's the other knob people ship "for stability," the one every training-loop template includes with `max_norm=1.0` and no comment. If warmup and clipping are both insurance against early-training violence, they should produce the same 2x2 table. So: same harness, same seeds, Adam at lr 1e-3 and 3e-3, with and without `clip_grad_norm_(1.0)`, plus an aggressive 0.3 arm, five paired seeds each.

## The table that refused to rhyme

| arm | accuracy (5 seeds) | paired Δ | t | wins |
|---|---|---|---|---|
| lr 1e-3 | 89.64 ±0.12 | — | — | — |
| lr 1e-3 + clip 1.0 | 89.01 ±0.21 | **−0.62** | −6.4 | 0/5 |
| lr 3e-3 | 86.98 ±0.34 | — | — | — |
| lr 3e-3 + clip 1.0 | 83.55 ±1.05 | **−3.43** | −10.6 | 0/5 |
| lr 3e-3 + clip 0.3 | **73.13 ±1.04** | **−13.85** | −24.5 | 0/5 |

Warmup's table, inverted on every row. At the tuned rate, where warmup was a harmless no-op, clipping charges a real tax: −0.62, zero wins in five, t of −6.4 — the clearest "this default is hurting you" number the audit series has produced. At the hot rate, where warmup recovered +2.86, clipping *deepens* the hole by another −3.43. And tightening the threshold — which under the insurance theory should mean *more* protection — detonates: −13.85, fifteen accuracy points below where this model lands with no stability aids at all. Whatever this knob is doing on my runs, it isn't the job in its job description.

## There was never anything to clip

So I instrumented it: a full unclipped run at each learning rate, logging the global gradient norm at every one of the 3,900 steps. The result dissolves the premise of the experiment. At lr 1e-3 the median norm is 2.01, the 10th percentile is 1.61, and the all-run maximum — the biggest "spike" ten epochs ever produced — is 6.74. The norms are boringly stable from epoch 0 (median 2.73) to epoch 9 (median 1.97). There are no explosions, no heavy tail, nothing resembling the RNN loss-cliff pathology this technique was invented for. Which means a threshold of 1.0 sits *below the signal*, not above the noise: **clip 1.0 engaged on 100.0% of steps. Clip 0.3, also 100%.** The fuse I installed wasn't a fuse — a fuse that's always blown is just a resistor. On this CNN, `clip_grad_norm_` is a permanent gradient attenuator that rescales every single update, roughly halving them at threshold 1.0 and crushing them ~7x at 0.3.

![A storm rages outside the window while the fuse box absorbs the surge and the lamp burns steady over a sleeping cat](./figure-1.jpg)

That reframes the question but doesn't answer it, because Adam is famously *insensitive* to uniformly rescaled gradients — the second-moment normalization divides the scale right back out. A constant 7x attenuation should cost little. Fifteen points is not little. Something in the update isn't being rescaled along with the gradient.

## The decay term doesn't get clipped

That something is weight decay. [Last week's dose-response](/posts/adam-weight-decay-dose-response) established exactly how Adam's *coupled* decay works: `wd * w` is added to the gradient inside the optimizer — *after* my clipping call has already shrunk the loss gradient. Clip the gradient to a seventh of its natural size and the decay term keeps its full magnitude, so the *ratio* of regularization to learning signal multiplies by ~7. My wd of 5e-4 effectively becomes ~3.5e-3 — and last week's table says 5e-3 is a cliff worth −2 points on its own, with the adaptive machinery amplifying from there. The clipping arms weren't measuring clipping. They were measuring an accidental weight-decay overdose, administered through a side door.

That's a hypothesis with a cheap test: rerun the catastrophe arm (lr 3e-3, clip 0.3) with weight decay removed. Result: **87.91** versus 73.13 — removing the regularizer *recovered 14.8 points*, essentially the entire crater. The clipped-and-undecayed model even edges out the unclipped-but-decayed baseline (86.98), which is its own small curiosity (near-unit-norm gradients into Adam approach sign-SGD, apparently not a bad place at a hot rate) — but I'll leave that thread unpulled at two seeds. The verified claim is the one that matters: clipping's damage on this recipe routes overwhelmingly through its unclipped interaction with coupled weight decay, the third "two knobs sharing one trench coat" the audit series has caught after [EMA-decay-vs-schedule](/posts/ema-decay-dose-response) and [Adam-vs-AdamW decay](/posts/adam-weight-decay-dose-response).

For the config file: before you ship `clip_grad_norm_`, spend one epoch logging `total_norm` — it's returned by the clipping call itself, the diagnostic is literally free. If your threshold sits above a heavy tail that engages on 0.1% of steps, you have a fuse; keep it. If it sits below your median norm, you don't have a stability aid — you have a global gradient rescaler that quietly re-doses any coupled regularizer (AdamW's decoupled decay, note, is immune to this exact trap: it never rides the gradient). And if what you actually wanted was protection against a hot learning rate, yesterday's answer stands: warmup pays that claim in full and charges nothing. Scope: one CNN with BatchNorm, 10-epoch budget, Adam — on architectures whose gradients genuinely spike, clipping has earned its reputation honestly, and nothing here says otherwise. What's dishonest is the default: a threshold copied from a world with spikes into a model that never produces one.
