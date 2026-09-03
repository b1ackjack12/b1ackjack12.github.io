---
title: "The Most Recommended IMU Augmentation Made My Model Worse in All Five Splits"
description: "Every sensor-data tutorial hands you the same augmentation menu: add noise, scale, rotate. I measured all three on human activity recognition under an honest subject split. Two did nothing detectable. The most popular one — Gaussian jitter — hurt in five out of five splits, and the reason says something about how badly image intuitions transfer to sensors."
slug: "imu-augmentation-measured"
date: 2026-09-03
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["imu", "sensors", "augmentation", "activity recognition", "deep learning"]
---

Back to this blog's home turf. When I [ranked training knobs last week](/posts/training-knob-effect-sizes), image augmentation landed in the heavyweight tier: turning it off cost 2.2 sigma. The natural question for anyone who works with inertial sensors instead of photos — my day job — is whether the sensor equivalent carries the same weight. The standard IMU augmentation menu, repeated across tutorials and papers, is three items: **jitter** (add Gaussian noise), **scaling** (random per-channel amplitude), and **rotation** (random 3D rotation of the sensor frame). I've applied all three on faith at various points. Today: UCI HAR, the honest [subject-split protocol](/posts/imu-har-subject-vs-random-split), five splits per configuration, measured.

One implementation note that matters, since rotation is the domain-specific one: the nine HAR channels are three xyz-triads (body acceleration, gyroscope, total acceleration), and a physical change in sensor mounting rotates *all three by the same matrix* — so that's how I apply it, one random rotation per window, shared across triads. And the angles stay small (±15°, plus a ±30° variant) because total acceleration contains gravity: rotate too far and you're telling the model that lying down looks like standing, which isn't augmentation, it's sabotage.

## Round one: nothing beats anything

Mean accuracy over five random subject splits (nine held-out subjects each), with the split-to-split standard deviation:

| configuration | accuracy | worst held-out subject (mean) |
|---|---|---|
| no augmentation | 94.45 ±1.59 | 78.8% |
| jitter 0.1σ | 93.72 ±1.43 | 77.2% |
| scaling ±10% | 94.49 ±1.77 | 81.0% |
| rotation ±15° | 94.33 ±1.65 | 80.4% |
| rotation ±30° | 94.25 ±1.83 | 80.2% |
| all three combined | 94.54 ±1.62 | 80.4% |

Read naively, this is a wall of nothing: every difference is well inside the ±1.6-point noise, which here is dominated by *which subjects* land in the test set (a lucky draw of easy walkers pushes any config to 97%). By [August's seed-noise arithmetic](/posts/seed-noise-vs-real-improvements), detecting sub-point effects through this would take dozens of splits. Round one verdict: on this dataset, the entire augmentation menu lives in the trick tier — nothing like the 2.2-sigma heavyweight that image augmentation is.

## Round two: pairing changes the verdict on jitter

But this experiment has a structural advantage the CIFAR seed farm didn't: every configuration was run on the *same five splits with the same seeds*. So instead of comparing noisy means, I can compare each config against baseline **within each split** — same test subjects, same initialization, the split-luck variance subtracted away entirely.

Paired like that, scaling and both rotations stay null: deltas scatter around zero, sign flipping split to split. Jitter does not. It loses to baseline in **five splits out of five** — by 0.41, 1.07, 0.85, 0.95, and 0.40 points — a mean of −0.74 with a paired t-statistic of 5.2. Unpaired, that −0.74 hid inside the ±1.6 noise; paired, it's unambiguous. The most commonly recommended augmentation in the sensor literature made this model reliably worse.

The mechanism, I'm fairly confident, is one I've met before: [the LOSO error analysis](/posts/har-loso-error-analysis) showed HAR's hard margin is sit-versus-stand — static postures whose entire signature is a subtle, *constant* difference in how gravity projects onto the sensor axes. Dynamic activities like walking have structure everywhere; the static ones are distinguished by small steady offsets. Gaussian jitter at 0.1σ is precisely a machine for burying small steady offsets. The image-augmentation intuition — "noise teaches robustness" — assumes the signal is big and the details are decoration. In low-amplitude sensor channels, the details *are* the signal.

![A librarian shaking a shelf of books to make readers more robust, while the fine print in the books blurs away](./figure-1.jpg)

## The one hint worth keeping

One column of the first table refuses to be fully null: the **worst held-out subject**. Scaling lifts it from 78.8% to 81.0%, rotation to 80.4% — small gains at exactly the place subject-generalization hurts most, consistent with the theory that these transforms mimic real person-to-person variation (different body mechanics, different device mounting). Honesty requires the caveat: the scaling gain is driven substantially by one split, and five splits can't certify a two-point effect on a per-subject metric. I'd call it a hypothesis worth fifty splits, not a result.

So the measured advice for IMU work, as of today: **don't jitter by default** — it's the one intervention here with solid evidence, and the evidence says harm. Scaling and small rotations are free insurance against mounting and physiology variation that this clean lab dataset may simply not need — UCI HAR's harness is fixed to the waist and its subjects follow a script, so the invariances rotation buys aren't being tested. On messier field data I'd still bet on rotation, but after today I'd measure rather than bet. And methodologically: when your experiment structure allows pairing, pair — it found a five-sigma effect inside noise twice its size, for zero additional GPU minutes. The 30 training runs behind this post took 85 seconds total; the sensor world's models are small enough that there's no excuse for faith-based augmentation.
