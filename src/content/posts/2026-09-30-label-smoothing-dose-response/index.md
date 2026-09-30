---
title: "All of Label Smoothing's Accuracy Arrives at 0.02. Everything After That Is Damage"
description: "Yesterday ended with smoothing 0.1 improving my model while wrecking its probabilities, and an obvious question: is there a dose that keeps the gain without the wreckage? The five-point sweep answers cleanly — the accuracy benefit saturates at the smallest dose I tried, calibration is a strictly monotonic casualty after its sweet spot, and the folklore that smoothing improves calibration turns out to be true at exactly one point on the curve. It just isn't the point everyone copies."
slug: "label-smoothing-dose-response"
date: 2026-09-30
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "label smoothing", "calibration", "hyperparameters", "training"]
---

[Yesterday's experiment](/posts/label-smoothing-calibration-measured) left the strange verdict standing: label smoothing at the standard 0.1 bought +0.41 accuracy and paid for it by tripling calibration error — a 9-point correction administered to a 2.4-point problem. The natural follow-up is the same one that worked for [rotation augmentation](/posts/imu-rotation-angle-sweep): stop arguing about the knob's value and sweep it. Five doses on the Adam recipe — 0 and 0.1 from yesterday's runs, plus fresh five-seed arms at 0.02, 0.05 and 0.2 — with the full instrument panel: accuracy, mean confidence, 15-bin ECE, NLL.

## Two curves that don't care about each other

| dose | accuracy | mean confidence | ECE | NLL |
|---|---|---|---|---|
| 0.0 | 89.72 ±0.25 | 92.5 | 2.82 | 0.311 |
| **0.02** | **90.11 ±0.13** | 90.6 | **1.21** | **0.302** |
| 0.05 | 90.08 ±0.12 | 87.9 | 2.78 | 0.320 |
| 0.1 | 90.13 ±0.34 | 83.9 | 6.38 | 0.358 |
| 0.2 | 90.28 ±0.16 | 75.6 | 14.69 | 0.445 |

The accuracy column is a step function: one jump of ~+0.4 between dose 0 and dose 0.02, then **flat across a 10x range of doses** — 90.08 to 90.28, differences within seed noise. Whatever regularization work label smoothing does for top-1 accuracy, it finishes at the smallest dose I tried; multiplying the dose by ten buys nothing more. The confidence column, meanwhile, never stops moving: every increment strips another few points of confidence off the same 90%-accurate model, mechanically and linearly, all the way down to a model that's right 90% of the time while claiming 76. The two effects everyone bundles as "label smoothing" are separate phenomena on separate curves — a saturating accuracy benefit and a non-saturating confidence deflation — and the standard dose of 0.1 sits at a point where the first has long finished and the second is doing pure damage.

Which redeems the folklore, at exactly one point. At 0.02, the deflation happens to roughly match the baseline's +2.8 overconfidence: ECE *improves* from 2.82 to 1.21, NLL hits its curve-wide best, and the confidence gap shrinks to half a point. "Label smoothing improves calibration" is true — as a statement about the matched dose, not about the technique. Yesterday's harm and the literature's promise are both on this one curve, five millimeters apart.

![A seasoning curve: one pinch transforms the dish, and every spoonful after that only makes it saltier without making it tastier](./figure-1.jpg)

## Where 0.1 came from, and why it didn't travel

The 0.1 default isn't arbitrary — it's inherited, from the Inception-v3 paper that introduced smoothing for *1000-class ImageNet*. And class count is exactly the parameter that doesn't survive the copy-paste: smoothing 0.1 spreads its mass over the wrong classes, so on ImageNet each wrong class receives a homeopathic 0.0001 of target probability, while on my 10 classes each receives 0.01 — a hundred times more aggressive a squeeze on the true-class logit gap for the *same* written-down number. Under that reading, my sweet spot of 0.02 on 10 classes and the canonical 0.1 on 1000 aren't different opinions; they're plausibly the same medicine at class-count-adjusted dosage. I'll hold that as an interpretation rather than a law — one dataset, one model, one budget, and I haven't run the 1000-class version — but the direction of the error is unambiguous: copying `label_smoothing=0.1` into a 10-class problem imports a dose calibrated for a different regime.

Housekeeping notes to close the arc. The stacking-post recipe gets a patch: swapping its 0.1 for 0.02 keeps the accuracy (90.11 vs 90.17 there — same within noise) and turns the probability profile from a liability into the best on the curve, so the [stacked model's](/posts/stacking-the-months-wins) TTA averaging and any future confidence gates now operate on honest numbers. And a small bonus consistent with that post's stabilizer theme: every smoothed arm has roughly *half* the baseline's seed noise (±0.12–0.16 vs ±0.25). The audit-of-defaults series now has a fourth entry, and this one has the cleanest moral so far: the knob was fine, the *number in it* was someone else's, tuned for a problem with a hundred times more classes. Sweeps are cheap. Inheritance isn't.
