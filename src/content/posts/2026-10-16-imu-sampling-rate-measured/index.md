---
title: "I Slowed My IMU's Clock 4x and Accuracy Went Up"
description: "The window-length experiment showed my activity model falls apart when I shorten its 2.56-second view of the world. Today's follow-up asks the orthogonal question: keep the full window, but sample it with a lazier clock — 25, 12.5, 6.25 Hz instead of 50. Fifty paired subject splits later, half the data rate is free, a quarter of it is actually better than the original (t = 4.3, and the hardest subjects gain the most), and the only arm that significantly loses is the one that broke the signal-processing rules. The sensor was oversampled all along."
slug: "imu-sampling-rate-measured"
date: 2026-10-16
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["imu", "har", "sensors", "deployment", "signal processing"]
---

[The window-length experiment](/posts/har-window-length-measured) ended with a clean dose-response: every halving of my HAR model's 2.56-second window cost accuracy, down to −0.87 points and a −1.98 collapse for the worst subject at 32 samples. But cutting the window short is only one of two ways to shrink an input. The other is to keep the full 2.56 seconds and sample them with a slower clock — the same stretch of walking, described by fewer, sparser measurements. For a wearable, that second knob is the interesting one: the sample rate drives the sensor's power draw, the radio bytes, and the model's FLOPs, and 50 Hz is simply what the dataset shipped with. Nobody chose it for the task.

So: [the usual harness](/posts/har-augmentation-fifty-splits), 50 paired subject splits, and the 50 Hz windows downsampled to 25, 12.5 and 6.25 Hz over the same time span. Each rate comes in two flavors, because downsampling has a right way and a lazy way — naive striding (`x[:, :, ::k]`, just drop samples) versus proper decimation (low-pass filter first, then drop; `scipy.signal.decimate`). The gap between those two flavors is itself a measurement, of how much aliasing actually hurts on real signals.

## Half the data is free. A quarter of it is a raise

| rate | samples | flavor | accuracy | Δ vs 50 Hz | t | worst subject Δ |
|---|---|---|---|---|---|---|
| 50 Hz | 128 | — | 92.66 ±2.75 | — | — | — |
| 25 Hz | 64 | naive | 92.75 | +0.09 | +1.6 | +0.32 |
| 25 Hz | 64 | anti-aliased | 92.73 | +0.08 | +1.2 | +0.21 |
| 12.5 Hz | 32 | naive | 92.84 | +0.18 | +2.1 | +0.47 |
| **12.5 Hz** | **32** | **anti-aliased** | **93.12** | **+0.47** | **+4.3** | **+1.93** |
| 6.25 Hz | 16 | naive | 92.37 | **−0.29** | −2.8 | −0.69 |
| 6.25 Hz | 16 | anti-aliased | 92.57 | −0.09 | −0.8 | −0.81 |

Three findings, in ascending order of surprise. First, 25 Hz is simply free: both flavors sit within noise of the baseline, so half the sample rate — half the bytes, half the conv-layer input — costs nothing. Second, at the bottom of the table, 6.25 Hz is where real information finally runs out, and *how* you get there matters: naive striding loses a statistically solid −0.29 while the anti-aliased version turns that into a statistical nothing (−0.09, t = −0.8). That gap is textbook aliasing made visible — above-Nyquist energy folding back into the band as noise when you skip the low-pass filter.

Third, and the reason this post exists: **12.5 Hz with proper decimation beats the full-rate baseline.** Not "matches" — beats, +0.47 with a t of 4.3, 35 wins in 50 splits. And the gain concentrates exactly where this dataset hurts: the worst subject in each split improves by +1.93 on average, the same hardest-subjects margin that [every augmentation experiment](/posts/imu-rotation-angle-sweep) has been fighting for. A quarter of the data, same time span, better model — with the improvement delivered by the step everyone treats as a chore (the anti-aliasing filter), which at this rate doubles the naive flavor's gain (+0.47 vs +0.18).

![Three flipbooks of a runner: a thick one with smooth motion, a thinner one slightly choppy, and a sparse one where the pose jumps between pages](./figure-1.jpg)

## The physics was on the label

None of this is mysterious after the fact; it's Nyquist arithmetic. Human gait has its fundamental around 1–2 Hz with discriminative harmonics mostly below ~6 Hz — and the UCI pipeline even low-pass filtered everything at 20 Hz before windowing. Sampling at 12.5 Hz (Nyquist 6.25 Hz) still covers essentially the whole band that distinguishes walking from climbing stairs; 50 Hz was oversampling the activity by an order of magnitude. Push to 6.25 Hz (Nyquist ≈ 3 Hz) and you finally start clipping real harmonics, which is precisely where the table's first significant loss appears. The *improvement* at 12.5 Hz plausibly stacks two effects I can't fully separate: the decimation filter throws away high-frequency sensor noise the activity never needed (a free denoiser), and the same fixed-size conv kernels now span 4x more of the gait cycle, seeing whole strides instead of fragments. Both point the same direction; I'll leave their shares unassigned.

Put today's table next to the window post's and the contrast is the real lesson. Thirty-two samples as a *cropped* window (0.64 s of context at 50 Hz): −0.87, worst subjects −1.98. The same thirty-two samples as a *decimated* window (full 2.56 s at 12.5 Hz): +0.47, worst subjects +1.93. Identical input tensors, shape-wise — a four-point swing depending on whether those samples cover the whole gait cycle or a fragment of it. The model doesn't need a fast clock; it needs the full sentence, even if every word is shorter.

For the config file: if you're shipping an IMU model, sweep the sample rate downward before you shrink anything else — my deployment numbers from the [CPU serving post](/posts/cpu-vs-gpu-serving-crossover) already showed this model class runs in 66 microseconds at batch 1, and a 4x shorter input only widens that margin while the *sensor* (the actual battery cost on a wearable) does 4x less work. Just don't skip the low-pass filter: it's one `scipy.signal.decimate` call at training time and a trivial IIR on-device, and it's the difference between a free lunch and a slow leak. Scope: one dataset whose signals were already cleanly filtered at capture, one small CNN, and subjects walking in a lab — a noisier consumer IMU may shift the cliff, which is an argument for running this same cheap sweep on your own data, not for trusting 50 Hz.
