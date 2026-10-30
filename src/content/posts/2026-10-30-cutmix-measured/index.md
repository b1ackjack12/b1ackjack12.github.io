---
title: "I Predicted CutMix's Confidence Before Training It. The Accuracy Was the Surprise"
description: "Tuesday's mixup post ended with a claim that was really a prediction: a soft-target technique's mean confidence is computable from its targets before any gradient flows. CutMix is the natural test — same label math, completely different pixels. The prediction landed within three points on the always-on arm and within one point on the standard p=0.5 arm. The part I didn't predict: at a 10-epoch budget CutMix loses to mixup, by a lot — the folklore ranking between the two cousins inverts when the schedule is short."
slug: "cutmix-measured"
date: 2026-10-30
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "cutmix", "calibration", "augmentation", "statistics"]
---

[Tuesday's mixup post](/posts/mixup-measured) ended on a mechanism I claimed wasn't empirical: a soft-target technique's confidence is decided by its targets, not discovered during training. Mixup's targets average 0.750 at α=1.0, and the trained model dutifully reported 72.9. If that's a real mechanism and not a coincidence, it should transfer to a technique with the same label arithmetic but *completely different pixels* — and that technique exists. CutMix draws the same `lam ~ Beta(α, α)`, but instead of alpha-blending two whole images into a ghostly double exposure, it cuts a rectangular patch from one image and pastes it into the other, with the label blended by patch area. Locally, every pixel is clean; only the label is soft. Same target math, different image statistics. If confidence is obedient, CutMix should land on its target average too. That's a prediction, made before training, falsifiable for the price of ten runs.

One wrinkle makes it a sharper test: CutMix's *realized* blend weight isn't quite Beta-distributed, because patches that straddle the image border get clipped smaller and the label weight is recomputed from the surviving area. So instead of trusting the closed-form 0.750, I logged the realized mean of `max(lam, 1−lam)` across every training step — the actual average confidence the loss requested. Two arms on the [usual recipe](/posts/sgd-vs-adam-fixed-budget) (CIFAR-10, adapted ResNet18, Adam lr 1e-3, cosine, 10 epochs), five seeds, paired against the same-seed baseline runs from Tuesday: CutMix applied to every batch, and CutMix applied with probability 0.5 — the duty cycle most real codebases (including timm) actually ship.

## The scoreboard, theory column included

| arm | requested conf (logged) | delivered conf | accuracy | Δ vs baseline (t) | ECE | NLL |
|---|---|---|---|---|---|---|
| baseline | 100 | 92.5 | 89.74 ±0.20 | — | 2.85 | 0.315 |
| mixup α=1.0 (Tue) | 75.0 | 72.9 | 87.98 | −1.76 (t = −12.2) | 15.10 | 0.490 |
| cutmix p=1.0 | **73.1** | **76.2** | 86.65 ±0.26 | **−3.08 (t = −23.9), 0/5** | 10.49 | 0.473 |
| cutmix p=0.5 | **86.4** | **85.6** | 88.81 ±0.22 | −0.92 (t = −7.5), 0/5 | 3.31 | 0.353 |

The prediction held, twice. Always-on CutMix requested an average confidence of 73.1 (the border clipping shaved 1.9 points off the theoretical 75) and the model delivered 76.2 — three points of daylight. The p=0.5 arm is the cleaner hit: half the steps train on clean one-hot targets, half on patched ones, the logged request averages 86.4, and the model reports 85.6. **Within 0.8 points, on a number computed from the augmentation's sampling distribution.** Across mixup and CutMix now, four arms at four different doses (requested 75.0 / 73.1 / 86.4 / 89.9, delivered 72.9 / 76.2 / 85.6 / 85.0), confidence tracks the request everywhere, never off by more than five points while the arms span a seventeen-point range. The confidence column of a soft-target technique is not a training outcome. It's a configuration you set, in expectation, when you pick α and p.

![A tailor sewing a square patch onto a quilt, with finished quilts showing patches of every size](./figure-1.jpg)

A small asymmetry is worth recording for honesty: mixup delivered slightly *under* its request (72.9 vs 75.0) while CutMix delivered slightly *over* (76.2 vs 73.1). A plausible reading is the pixels: a mixup input is globally ambiguous — every pixel is a blend, so the model's own uncertainty discounts further — while a CutMix input is two regions of perfectly clean pixels, each locally unambiguous, letting the network stay a touch surer than the label asks. I'll flag that as interpretation; the three-point residual is near the edge of what these five-seed means resolve.

## The accuracy column is where folklore lost

I expected CutMix to roughly reprise mixup's accuracy story. It did worse — meaningfully. Always-on CutMix costs **−3.08 points (t = −23.9, zero wins in five)**, nearly double mixup's −1.76 at the same α. Even at the shipping-default p=0.5, it's −0.92 with zero wins, where [mixup's conservative dose was a clean null](/posts/mixup-measured). On long schedules the published ranking runs the other way — CutMix is the stronger performer of the pair on ImageNet-scale training. At a tenth of the native schedule, the ranking inverts, and the direction makes mechanical sense: CutMix is the more aggressive intervention. Mixup shows the model a dimmed version of *both* images everywhere; CutMix fully occludes a region, deleting that part of the object outright and demanding the model classify from whatever survived the crop. That's a harder curriculum, which is to say a stronger regularizer, which is to say — [as with mixup](/posts/mixup-measured), [as with EMA](/posts/ema-weights-measured) — a bigger loan against training time than a 10-epoch budget can service. The stronger the medicine, the worse it performs when the course of treatment is cut short.

Calibration-wise, the same overshoot logic from [the label smoothing series](/posts/label-smoothing-dose-response) prices every row: my baseline is only +2.8 overconfident, so a requested confidence of 73 overshoots straight through honest into deeply underconfident (ECE 10.49), while the p=0.5 arm's gentler request of 86 lands close enough that ECE only drifts from 2.85 to 3.31. Note what the duty-cycle knob turned out to be: `p` is a *linear dial on requested confidence* — 0.5 × 100 + 0.5 × 73 ≈ 86.5, logged 86.4 — which makes it the same kind of dose control as label smoothing's ε, just implemented as a coin flip per batch instead of a constant per label.

Scope lines: one dataset, one small model, one budget, and the long-schedule regime — where both techniques earned their reputations — deliberately unmeasured. What this adds to Tuesday's post is the part a single technique couldn't establish: the obedient-confidence arithmetic survived a change of pixel mechanics, a change of realized `lam` distribution, and a duty-cycle knob it predicted linearly. If you ship mixup or CutMix, you can now write down your model's approximate mean confidence before you train — and if you were counting on that number to mean something about calibration, [one fitted scalar afterwards](/posts/temperature-scaling-measured) remains the cheaper, better tool.
