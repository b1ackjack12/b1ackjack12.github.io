---
title: "Weight Decay 0.05: Catastrophe in Adam, Optimum in AdamW"
description: "The dose-response series reaches the knob I've been inheriting since September: weight_decay=5e-4, copied from SGD folklore into Adam without a second look. Nine arms and forty-five runs later, the two optimizers' curves turn out to be the same shape shifted two orders of magnitude apart — the dose that wrecks Adam for 24 points is AdamW's best setting, my September screening picked AdamW's dose off one seed and got it wrong, and 'the same weight_decay' between the two optimizers is revealed as two different algorithms wearing one config key."
slug: "adam-weight-decay-dose-response"
date: 2026-10-16
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "weight decay", "adamw", "hyperparameters", "training"]
---

Every recipe on this blog since [the optimizer face-off](/posts/sgd-vs-adam-fixed-budget) has carried `weight_decay=5e-4` — a number I originally picked for SGD because every CIFAR repo on the internet uses it, then ported to Adam when Adam won, with exactly zero measurement of whether it belongs there. That's the precise shape of inherited config this [dose-response series](/posts/label-smoothing-dose-response) exists to audit. So: Adam at four doses spanning 2.5 orders of magnitude plus a zero control, AdamW at the matching doses, five seeds each on the usual 10-epoch CIFAR ResNet18 — forty-five runs, with each optimizer's arms compared against its own wd=0 baseline on identical seeds.

The reason to sweep *both* optimizers is that "weight decay" is secretly two different mechanisms sharing a keyword argument. In Adam, the classic implementation adds `wd * w` to the *gradient*, which then flows through the adaptive machinery and gets divided by the running RMS of the gradients. In AdamW, the decay is decoupled — subtracted directly from the weights, scaled only by the learning rate, never touching the adaptive denominator. Same config key, different algorithms. The question a sweep can answer is whether that difference is pedantry or points.

## Two curves, two orders of magnitude apart

| optimizer | wd | accuracy (5 seeds) | Δ vs own wd=0 | t |
|---|---|---|---|---|
| Adam | 0 | 89.32 ±0.29 | — | — |
| Adam | 1e-4 | **89.81 ±0.16** | **+0.49** | +3.8 |
| Adam | 5e-4 | 89.77 ±0.33 | +0.46 | +3.4 |
| Adam | 5e-3 | 87.33 ±0.35 | **−1.99** | −11.6 |
| Adam | 5e-2 | **64.97 ±5.39** | **−24.34** | −10.1 |
| AdamW | 0 | 89.32 ±0.14 | — | — |
| AdamW | 5e-4 | 89.29 ±0.26 | −0.03 | −0.3 |
| AdamW | 5e-3 | 89.32 ±0.21 | +0.00 | +0.0 |
| AdamW | 5e-2 | **89.66 ±0.15** | **+0.34** | +3.0 |

Read Adam's column first: it's the [EMA-decay topology](/posts/ema-decay-dose-response) again — a shelf, a cliff, a crater. The benefit saturates at the smallest dose I tried (1e-4 already buys the full +0.49; 5e-4 adds nothing more), then 5e-3 falls off a cliff, and 5e-2 doesn't converge so much as thrash — 65% mean accuracy with a ±5.4 standard deviation, the largest variance any arm has posted on this blog. My inherited 5e-4, for once, sits safely mid-shelf; the audit acquits it.

Now read AdamW's column, which is the same curve shifted right by roughly two orders of magnitude. At the doses where Adam does all its living and dying — 5e-4, 5e-3 — AdamW registers *nothing*: deltas of −0.03 and +0.00, flat zeros. Its effect only begins at 5e-2, the exact dose that costs Adam twenty-four points. One number, two optimizers: catastrophe in one, optimum in the other. Copying a `weight_decay` between Adam and AdamW isn't tuning, it's [copying a time constant between schedules](/posts/ema-weights-measured) — the same mistake this blog has now measured in its third costume.

The ~100x shift even has a plausible size, not just a direction. Decoupled decay shrinks weights by `lr * wd` per step — at lr 1e-3 and wd 5e-2, that's 5e-5 of each weight. Coupled decay rides the adaptive update, where the gradient (decay term included) gets divided by the running gradient RMS; if that RMS sits around 10⁻², the division inflates the decay's bite by roughly 100x, so Adam at wd 5e-4 lands at about the same effective shrinkage — 5e-5 — as AdamW at 5e-2. Two arms I measured as equally optimal, agreeing through arithmetic I didn't tune to make them agree. I'll flag it as consistency rather than proof (the RMS varies per weight and per epoch; that's the *point* of the coupling being a bug), but the curves' alignment is hard to read any other way.

![Five identical plants watered with increasing doses: parched, healthy, thriving, drooping, and fully submerged](./figure-1.jpg)

## The screening I owe a correction

One row above is a confession. September's face-off chose AdamW's dose by single-seed screening, and that screening said 0.005 beat 0.05. Today's five-seed sweep says the opposite: 0.05 is better by +0.34 (t = 3.0), and 0.005 is indistinguishable from *no decay at all*. The screening seed flipped a coin and I published the coin flip — exactly the failure mode the [seed-noise post](/posts/seed-noise-vs-real-improvements) warned about and [the nondeterminism post](/posts/pytorch-same-seed-different-model) explained, now caught in my own pipeline. For the record, the correction doesn't change the face-off's verdict: AdamW at its true best dose reaches 89.66, which still only ties Adam's 89.77–89.81 at this budget. Adam survives as the house optimizer; its challenger just deserved a fairer fight than one seed gave it.

Config-file summary, in the series' usual shape. For Adam at lr 1e-3 on a run like mine: anything in 1e-4–5e-4 is on the shelf and the choice within it doesn't matter; 10x more is a cliff. For AdamW: the PyTorch default of 0.01 sits comfortably inside the active range (my 5e-2 arm is the shelf's measured edge in the good direction), and anything at Adam-ish magnitudes is a silent no-op — you think you're regularizing, and you're paying for nothing. And across optimizers: never port the number. Not because of superstition, but because the units differ — one is a gradient-space dose filtered through adaptive statistics, the other a direct fraction of the weights per step. The config key being spelled identically is the trap. Scope: one model, one dataset, one 10-epoch cosine budget, lr fixed at 1e-3 — wd and lr interact (decoupled decay scales *with* lr), so these shelf locations travel only as far as the recipe does. The shape, I'd bet on; the x-axis, re-measure.
