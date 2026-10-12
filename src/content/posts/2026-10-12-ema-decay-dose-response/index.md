---
title: "EMA's Decay Isn't a Dial. It's a Plateau, a Cliff, and a 10%-Accuracy Crater"
description: "A month ago EMA's tutorial default lost me three points and the tuned value earned a cosmetic +0.08, so I did what this blog now does to every knob: swept the whole curve. Six decay values tracked as parallel shadows inside the same ten training runs — perfectly paired, zero extra GPU time per arm. The result isn't a curve with an optimum. It's a flat shelf where every reasonable value pays the same +0.1, a cliff edge between 0.995 and 0.999, and at the far end a shadow model that evaluates at coin-flip accuracy."
slug: "ema-decay-dose-response"
date: 2026-10-12
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "ema", "hyperparameters", "training", "statistics"]
---

[September's EMA experiment](/posts/ema-weights-measured) measured exactly two decay values on the SGD recipe and left a dangling thread: 0.999 catastrophically bad, 0.99 cosmetically good, nothing in between and nothing beyond. Then the [stacking post](/posts/stacking-the-months-wins) carried decay 0.99 onto the Adam recipe without re-measuring it — an inherited number of precisely the kind this blog keeps catching in other people's configs. The fix is the same one that worked for [label smoothing](/posts/label-smoothing-dose-response): stop defending a point and sweep the curve. Six decays — 0.9, 0.97, 0.99, 0.995, 0.999, 0.9999 — on the tuned Adam recipe, ten seeds.

The instrument deserves a sentence, because it's the best part. EMA's audit trick from September scales: *all six* shadows ride inside the same training run, updated from the same weight trajectory after every optimizer step. One run yields seven evaluations — raw plus six decays — so every comparison below is paired at the strongest possible level: same seed, same batches, same trajectory, different averaging window. The entire six-arm, ten-seed sweep cost ten training runs, about twenty-five minutes.

## A shelf, then a cliff, then a crater

Each decay's effective averaging window is roughly 1/(1−d) optimizer steps, against a total budget of 3,900 steps. That number turns out to be the only thing on the table that matters.

| decay | window | accuracy (10 seeds) | paired delta vs raw | t | wins |
|---|---|---|---|---|---|
| raw weights | — | 89.75 ±0.18 | — | — | — |
| 0.9 | ~10 steps | 89.80 | +0.050 | +2.7 | 8/10 |
| 0.97 | ~33 | 89.86 | **+0.109** | **+3.6** | 8/10 |
| 0.99 | ~100 | 89.85 | +0.104 | +2.0 | 8/10 |
| 0.995 | ~200 | 89.83 | +0.080 | +1.4 | 6/10 |
| 0.999 | ~1,000 | 85.00 | **−4.75** | −18.4 | 0/10 |
| 0.9999 | ~10,000 | **11.55** | −78.2 | −159 | 0/10 |

Read the top four rows first. Across a 20x range of window sizes — 10 steps to 200 — the paired deltas are +0.05 to +0.11, all within a whisker of each other. There is no optimum to find up here; it's a shelf. The agonizing-over-decimals question ("0.97 or 0.99 or 0.995?") that EMA discussions love is, on this recipe, worth at most six hundredths of a point, and the pairing is tight enough to say so with confidence: the best row (0.97, t = 3.6) is statistically real, but its advantage over its neighbors isn't. Any window that fits comfortably inside the cosine schedule's annealed tail buys the same small, genuine, cosmetic improvement — consistent with September's hypothesis that a schedule annealing to zero already delivers the endpoint pre-smoothed, leaving EMA little to do.

Then the window crosses out of the tail and the curve stops being polite. At 1,000 steps — a quarter of the whole run — the shadow blends in weights from epochs where the learning rate was still high and the model measurably worse: −4.75, ten losses in ten. That's steeper than the −3.08 the same decay cost on [the SGD recipe](/posts/sgd-vs-adam-fixed-budget), so the cliff isn't an SGD quirk; if anything Adam's early trajectory is worse to remember. And at 10,000 steps the window is now 2.5x *longer than the entire training run*, which means the shadow is approximately a uniform average over the whole trajectory — including the random initialization it started from. That model scores 11.55 ±1.55 percent on a 10-class problem. Not degraded: statistically indistinguishable from guessing. The same codebase, the same run, the same trick that earns +0.1 at decay 0.97 hands you a coin-flipper at 0.9999, four keystrokes away.

One more diagnostic hiding in the deltas: their seed-to-seed standard deviation grows monotonically with the window — 0.06 at decay 0.9, 0.16 at 0.99, 0.18 at 0.995, 0.82 at 0.999. The longer the window, the more of the run's noisy history it drags into the final weights, so even *before* the cliff you can see the averaging reaching back into territory that varies run by run. The shelf isn't just flat in the mean; it gets quietly riskier toward its edge.

![A dog walker with six leashes of wildly different lengths: the short-leash dogs trot in step at the heel, while the dog on the longest leash is still far back down the path](./figure-1.jpg)

## The knob is secretly binary

The honest summary of this curve: EMA's decay is not a dial that trades off more-versus-less smoothing benefit. The benefit is a constant — about +0.1 on this recipe, take it or leave it — and the decay only decides whether you get that constant or walk off a cliff. A continuous hyperparameter with a binary outcome: *does the averaging window fit inside the schedule's converged tail, or doesn't it?* Every value that fits pays the same; every value that doesn't is somewhere on the slope between −4.75 and −78.

Which makes the config-file advice almost embarrassingly simple. Compute 1/(1−d), compare it to your total step count, and keep it in the low single-digit percents — for my 3,900-step runs anything from 0.9 to 0.995 is on the shelf, and I'll keep writing 0.99 since the [stacking recipe's](/posts/stacking-the-months-wins) choice turns out to have been safely mid-shelf all along. What you should *not* do is copy a decay out of a repo that trains 100x longer, because the number doesn't carry — a lesson that is now measured twice, on two optimizers, with the second measurement steeper than the first. Scope as always: one model, one dataset, one 10-epoch budget, and the shelf's height (+0.1) is specific to short cosine schedules — on constant LR or longer runs EMA has real work to do and the curve's left half will look different. The cliff's location, though, is arithmetic, and arithmetic travels.
