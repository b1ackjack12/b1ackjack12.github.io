---
title: "Mixup's Confidence Isn't Miscalibrated. It's Obedient"
description: "Mixup arrives with two pieces of folklore: it needs long schedules to pay off, and it improves calibration. At my 10-epoch budget, with the full instrument panel attached — accuracy, mean confidence, 15-bin ECE, NLL — the first one held and the second one inverted. α=0.2 is an accuracy null that doubles calibration error; α=1.0 costs 1.76 points and posts the worst ECE this blog has ever measured. And the confidence columns aren't mysterious at all: you can predict them from the Beta distribution before training starts."
slug: "mixup-measured"
date: 2026-10-28
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "mixup", "calibration", "augmentation", "statistics"]
---

Mixup is the most cited training trick I hadn't put on the bench yet: blend pairs of images with a random weight `lam ~ Beta(α, α)`, blend their labels with the same weight, and train on the result. The original paper reports solid gains — on schedules of 90 to 200 epochs. The follow-up folklore adds a second claim: mixup-trained models are better *calibrated*, with confidence that tracks accuracy instead of exceeding it. Both claims are measurable at my budget, and after [the label smoothing series](/posts/label-smoothing-calibration-measured) I had a suspicion about the second one — mixup is, after all, another technique that trains against soft targets.

Same harness as always: CIFAR-10, the adapted ResNet18, the [tuned Adam recipe](/posts/sgd-vs-adam-fixed-budget) (lr 1e-3, weight decay 5e-4, cosine, 10 epochs), five paired seeds per arm. Three arms: no mixup, the conservative α=0.2 most codebases default to, and the paper's α=1.0 where `lam` is uniform on [0, 1]. The instrument panel is the one from the [label smoothing post](/posts/label-smoothing-calibration-measured): test accuracy, mean confidence of the predicted class, 15-bin expected calibration error, and NLL.

## The table

| arm | accuracy | Δ vs baseline (t) | mean confidence | conf − acc gap | ECE | NLL |
|---|---|---|---|---|---|---|
| baseline | 89.74 ±0.20 | — | 92.5 | **+2.8** | 2.85 | 0.315 |
| mixup α=0.2 | 89.66 ±0.21 | −0.08 (t = −0.65), 2/5 wins | 85.0 | **−4.7** | 4.70 | 0.341 |
| mixup α=1.0 | 87.98 ±0.47 | **−1.76 (t = −12.18), 0/5 wins** | 72.9 | **−15.1** | 15.10 | 0.490 |

Accuracy first, because it's the simpler verdict. At this budget α=0.2 is a clean null — a −0.08 mean delta the pairing can't distinguish from seed noise. α=1.0 is not a null: −1.76 points, zero wins in five seeds, a t-statistic that leaves no room for argument. The folklore that mixup needs long schedules survives contact with the bench, and the mechanism is visible in what mixup *is*: it replaces every clean training image with a blend, so the model never sees an unmodified example. That's a strong regularizer, and regularization is a loan against training time — at 200 epochs the model has capacity and schedule to spare, and taming it pays; at 10 epochs the model is still underfitting, and taming it harder just means arriving at test day having learned less. It's the same budget logic that made [EMA a near-null here](/posts/ema-weights-measured) and keeps showing up whenever I benchmark a technique at a tenth of its home schedule.

## The confidence column was decided before training started

The calibration story is where it gets interesting, because the numbers look dramatic until you realize they were predictable — not qualitatively, *numerically*.

Mixup's training targets are soft: with blend weight `lam`, the dominant class gets target probability `max(lam, 1−lam)`. So before any gradient flows, you can ask what confidence the loss is actually asking for, on average. For α=1.0, `lam` is uniform and the expected dominant-class target is exactly **0.750**. For α=0.2, the Beta(0.2, 0.2) distribution is U-shaped — most blends are nearly pure — and the expectation is **0.899**.

Now reread the confidence column: the α=1.0 model reports mean confidence **72.9** against a predicted 75.0, and the α=0.2 model reports **85.0** against a predicted 89.9 — each a few points below its target average, which is the same discount the [label smoothing post](/posts/label-smoothing-calibration-measured) found, where the model's own residual uncertainty deflates things slightly further. The models aren't confused about their probabilities. They are reporting almost exactly the confidence the training targets instructed them to report, and the Beta distribution wrote the instruction.

![Two paint cans pouring into a mixing tray, the painter inspecting the blended swatches](./figure-1.jpg)

Which is why "mixup improves calibration" inverted here, the same way "label smoothing improves calibration" did. My 10-epoch baseline is only mildly vain: 92.5 confidence on 89.7 accuracy, a +2.8 gap, ECE 2.85. The calibration literature's mixup results come from models with much bigger overconfidence to burn off, where a built-in confidence haircut lands near the truth. Mine didn't need a 7-point haircut (α=0.2), let alone a 17-point one (α=1.0) — so the correction overshot straight through zero into underconfidence, and ECE got worse at both doses: 2.85 → 4.70 → 15.10. That last number is the worst calibration error I've measured on this blog, from a technique with a calibration reputation. NLL agrees at every dose (0.315 → 0.341 → 0.490). The pattern now holds across three label-softening techniques measured on this exact setup: the folklore "X improves calibration" is really "X deflates confidence by a fixed, dose-determined amount" — a correction, not a calibration, and it only helps when your model's overconfidence happens to match the dose. [Label smoothing's matched dose was 0.02](/posts/label-smoothing-dose-response). Mixup's matched dose at this budget would be some homeopathic α I didn't bother to find, because there's a better tool.

## What to actually do

If you're at a short budget and want honest probabilities, the ranking from my own measurements is unambiguous. [Temperature scaling](/posts/temperature-scaling-measured) took the plain baseline to ECE ~1.0 with one scalar fitted on 500 images, accuracy untouched. Native label smoothing at the matched 0.02 dose got ECE 1.21. Mixup at either standard dose made both accuracy and calibration worse — and if you're using mixup anyway for its long-schedule accuracy benefits, know that its deflated confidence is a *systematic, dose-shaped* distortion, which is exactly the kind temperature scaling can mostly re-sharpen (it correctly fitted T = 0.75 on my smoothing-deflated model and recovered most of the damage).

Scope lines, as usual: one dataset, one small model, one optimizer recipe, and above all one budget — 10 epochs against the technique's native 90–200. I haven't measured the long-schedule regime, and nothing here contradicts mixup's accuracy gains there. What I'd defend anywhere is the mechanism, because it isn't empirical: mixup's targets have a computable average confidence, the trained model dutifully converges to it, and whether that lands as "improved calibration" or as the worst ECE on your bench depends entirely on how overconfident you were before the medicine arrived.
