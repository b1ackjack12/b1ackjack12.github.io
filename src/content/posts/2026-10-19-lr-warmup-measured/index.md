---
title: "Warmup Did Nothing for My Learning Rate — Then It Fully Rescued the Wrong One"
description: "Learning-rate warmup is the most cargo-culted schedule fragment in deep learning: transformers need it, so everyone ships it. I measured what it actually does at a 10-epoch CNN budget with a 2x2 experiment — tuned and overheated learning rates, with and without one epoch of linear warmup. At the tuned rate, warmup is statistically nothing. At triple the rate, it claws back all 2.9 lost points and lands exactly on the tuned baseline, to within eight hundredths. Warmup isn't an accelerant. It's insurance, and the premium is zero."
slug: "lr-warmup-measured"
date: 2026-10-19
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "warmup", "learning rate", "hyperparameters", "training"]
---

Learning-rate warmup has the strangest status of any training-recipe fragment: universally shipped, rarely justified. It earned its reputation in transformer training, where skipping it genuinely detonates runs, and from there it diffused into every config file on the internet — including CNN recipes at small scale, where nobody seems to have asked whether it does anything. My own [schedule face-off](/posts/lr-schedule-face-off) never tested it; the [dose-response series](/posts/adam-weight-decay-dose-response) suggests the right question isn't "is warmup good?" but "under what condition does the knob engage?". So, a 2x2: Adam at the tuned lr 1e-3 and at an overheated 3e-3, each with and without one epoch of linear warmup, cosine after, five paired seeds each.

The 3e-3 arm isn't arbitrary. Back in the [optimizer face-off's screening](/posts/sgd-vs-adam-fixed-budget), Adam at 3e-3 was one of the single-seed losers that got eliminated — a learning rate I *know* is wrong for this recipe. That makes it the perfect test subject, because warmup's actual folklore claim isn't "adds accuracy"; it's "prevents early damage from a too-hot start." If that's true, warmup should do nothing where the rate is right and real work where it's wrong — a prediction with a measurable shape.

## Nothing to insure, nothing gained. Something to insure, everything back

| arm | accuracy (5 seeds) | paired comparison | Δ | t | wins |
|---|---|---|---|---|---|
| lr 1e-3 | 89.71 ±0.15 | — | — | — | — |
| lr 1e-3 + warmup | 89.83 ±0.28 | vs lr 1e-3 | +0.12 | +0.8 | 2/5 |
| lr 3e-3 | 86.78 ±0.61 | vs lr 1e-3 | **−2.94** | −12.3 | 0/5 |
| lr 3e-3 + warmup | 89.64 ±0.30 | vs lr 3e-3 | **+2.86** | **+12.0** | **5/5** |
| lr 3e-3 + warmup | — | vs lr 1e-3 | −0.08 | −0.5 | 2/5 |

Row by row, the folklore's testable half comes out exactly true. At the tuned learning rate, warmup is statistical nothing: +0.12 with a t of 0.8 and two wins in five — the same verdict as every knob this blog has audited at a setting that was already right. At the overheated rate, warmup recovers +2.86 of the −2.94 the bad learning rate cost, five wins in five, t of twelve. And the final row is the one that defines what warmup *is*: the rescued 3e-3 run lands within eight hundredths of the tuned baseline — statistically indistinguishable. Warmup didn't push past the ceiling; it returned the run *to* the ceiling. An insurance payout makes you whole. It doesn't make you rich.

The variance column tells the same story in a second voice. The hot rate didn't just lower the mean, it quadrupled the seed noise (±0.61 vs ±0.15) — the [lr-variance coupling](/posts/ema-weights-measured) this blog stumbled into back in September, reproducing right on schedule. Warmup pulled that back too (±0.30). Whatever early-training damage a hot Adam start inflicts, it's partly a lottery, and warmup cancels the lottery along with the mean loss.

![One runner stretching calmly by a bench at sunrise while another sprints straight out their front door mid-stumble, slipper flying](./figure-1.jpg)

## What the mechanism has to look like

One epoch of linear ramp means the model spends ~10% of its budget below full learning rate, and the cosine tail is one epoch shorter — yet the tuned arm paid nothing for it (+0.12, if anything). That asymmetry is the tell: whatever warmup protects against happens in the *first few hundred steps*, when Adam's second-moment estimates are still garbage and a 3e-3 step through randomly-initialized BatchNorm statistics can push weights somewhere the remaining nine epochs can't fully undo. Start slow and those same steps are taken at harmless magnitudes while the optimizer's statistics settle. After that window closes, the two schedules are nearly identical — which is why the insurance costs nothing when it doesn't pay out.

For the config file, the practical reading has two branches. If you tune your learning rate — and [one knob's worth of tuning](/posts/training-knob-effect-sizes) is the single highest-paying activity this blog has measured — warmup is a no-op you can keep or drop; it will neither help nor hurt beyond noise. If you *can't* tune (new dataset, inherited recipe, budget for exactly one run), warmup is the cheapest robustness you can buy: it flattened a 3x learning-rate error into a rounding error, for zero cost at the correct setting. That's also, I suspect, the honest story of why transformers "need" it — their recipes run closer to the stability cliff by design, so the insurable event isn't a mistuned config, it's every run. Scope: one CNN, one dataset, 10-epoch budget, Adam only — and note the asymmetry of the claim: I measured warmup rescuing a *hot* rate, not enabling rates above the tuned optimum. It caught a falling model; I have no evidence it lifts a healthy one.
