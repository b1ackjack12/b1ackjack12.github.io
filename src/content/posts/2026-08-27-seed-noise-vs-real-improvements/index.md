---
title: "Your +0.3% Improvement Is Probably a Coin Flip"
description: "I trained the identical model 15 times, changing only the random seed, and got accuracies spanning 4.6 percentage points. Then I added a real improvement worth +0.35%p and tried to detect it: a single-seed comparison picked the right winner 59% of the time. Thirty training runs later, some uncomfortable arithmetic about every ablation table I've ever written."
slug: "seed-noise-vs-real-improvements"
date: 2026-08-27
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "reproducibility", "statistics", "deep learning", "evaluation"]
---

A few weeks ago I showed that [the same seed doesn't even give you the same model](/posts/pytorch-same-seed-different-model) on a GPU. Today's question is one level up and more expensive to ignore: when you compare two training configurations — baseline versus your clever new trick — how much of the difference is the trick, and how much is which lottery numbers the seeds drew? I've written ablation comparisons my whole working life on the strength of one run per side. This post is me checking, thirty training runs late, whether any of them meant anything.

## Fifteen identical models

Setup: ResNet18 on CIFAR-10, 10 epochs, SGD with cosine schedule — the kind of short-schedule experiment people actually use to make decisions. Fifteen runs where *nothing* changes except the seed. (Each run takes 98 seconds, thanks to [Tuesday's persistent_workers fix](/posts/dataloader-num-workers-windows) — which is what made a 30-run experiment feel affordable at all.)

The fifteen final test accuracies: mean **87.86%**, standard deviation **1.17%p**, and a full range of **4.58 percentage points** — the luckiest seed landed 89.43%, the unluckiest 84.85%. Same code, same data, same hyperparameters. If those two runs had appeared in a table as "baseline" and "ours", the fake improvement would have been larger than most claimed contributions I've ever read.

To make that concrete, I paired up my fifteen identical runs every possible way and asked how often two *literally identical* configurations, compared with one seed each, differ by the margins papers and internal reports like to claim:

| claimed margin | identical configs differ by at least this |
|---|---|
| ≥ 0.1%p | 96% of pairings |
| ≥ 0.2%p | 90% |
| ≥ 0.3%p | 85% |
| ≥ 0.5%p | 74% |

A +0.3%p "win" is the *expected outcome of comparing a config against itself*. Three out of four pairings clear half a point. You do not need a better method to produce these numbers; you need two seeds.

![A slot machine dispensing different prizes to a line of identical model figurines, each pulling the same lever](./figure-1.jpg)

## Trying to detect a real improvement

Noise alone is only half the experiment. The other half: inject an improvement that is *actually real* and see whether normal practice can detect it. I added label smoothing 0.1 — a legitimately useful trick with literature behind it — and trained fifteen more seeds.

It helped: mean 88.21% versus 87.86%, a true gap of **+0.35%p**, right in the range that gets claimed in ablation tables. Now the uncomfortable part. Comparing one label-smoothing seed against one baseline seed — the standard single-run-each workflow — picks the correct winner in **59% of the 225 possible pairings**. A coin flip scores 50. The workflow I have used for years, applied to a real improvement of typical size, is nine points better than guessing.

It gets worse. Even with all fifteen seeds per arm, the gap isn't statistically significant: the per-seed noise is about 1.05%p, which puts the standard error of the 15-vs-15 comparison at 0.38%p — a t-statistic around 0.9. Fifty minutes of GPU time, thirty trained models, and I *still* can't formally distinguish a real improvement from nothing. The power calculation says detecting a 0.35%p effect through 1.05%p noise at conventional thresholds would take roughly **140 seeds per arm**. Nobody runs that. Which means every short-schedule ablation of this effect size, everywhere, is decided substantially by luck — and the ones that got published are the lucky ones.

## What survives this

Honest scope first: variance shrinks as schedules lengthen — converged 200-epoch CIFAR runs are commonly reported near 0.1–0.2%p std, an order of magnitude tighter than my 10-epoch runs. But that cuts the wrong way. Short schedules are precisely where hyperparameter choices, architecture tweaks, and "quick checks" get decided in practice; the sloppiest statistics live exactly where the decisions are made. And per the [determinism post](/posts/pytorch-same-seed-different-model), all of this sits *on top of* same-seed GPU non-determinism.

What I've actually changed: three seeds minimum for anything I'll act on, five if the gap looks smaller than a point, always reported as mean ± std — and a gap has to beat twice its standard error before I'm allowed to have an opinion about it. Effects too small to detect this way get treated as undetected, not as small wins. That discipline costs 3–5x the compute, which is the real reason nobody applies it — and why I spent Tuesday making each run cost 98 seconds instead of four minutes. Cheap runs aren't just convenient. They're what makes honest statistics affordable.
