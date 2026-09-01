---
title: "The Twenty-Line Test I Kept Promising, and the Tolerance That Makes It Work"
description: "Two posts on this blog end by recommending golden-sample tests and then don't show one. Today I built it and measured the only design decision that matters: the tolerance. Legitimate variation moved my logits by at most 0.006; the weakest bug I could inject moved them by 1.15. That 190x gap is where the test lives."
slug: "golden-sample-test-measured"
date: 2026-09-01
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["testing", "deployment", "pytorch", "debugging", "ci"]
---

I have now ended two separate posts — [the preprocessing damage table](/posts/image-preprocessing-bugs-ranked) and [the model.eval() autopsy](/posts/forgotten-model-eval-measured) — by recommending a golden-sample test and calling it "twenty lines," without writing the twenty lines. That's the kind of unverified claim [last week taught me to distrust](/posts/batch-512-textbook-fixes-measured), so today I built the thing against the same model and the same injected bugs from the damage-table post, and measured the one design question the twenty lines don't answer for you: **how tight should the tolerance be?**

Too loose and bugs slip under it; too tight and every harmless environment difference — new GPU, CPU fallback, a batching change — pages someone at 3 a.m. for nothing. The right answer isn't a philosophy, it's two measurements: the noise floor of *legitimate* variation, and the signal size of *actual* bugs. If there's a gap between them, the test works and the tolerance goes in the gap.

## The harness

Ten raw test images, one per CIFAR-10 class, saved as uint8 exactly as they come off the source — crucially *before* preprocessing, because preprocessing is the main thing under test. Alongside them, the training pipeline's logits for those ten images. The test re-runs raw bytes through whatever the current code does and compares:

```python
import numpy as np
import torch

def test_golden_samples():
    inputs = np.load("golden_inputs.npy")     # (10,32,32,3) uint8, raw RGB
    expected = np.load("golden_logits.npy")   # (10,10) float32
    model = load_model().eval()
    x = preprocess(inputs)                    # the pipeline under test
    with torch.no_grad():
        logits = model(x).cpu().numpy()
    np.testing.assert_allclose(logits, expected, atol=0.1)
```

Twelve lines, as promised, give or take a loader. The interesting number is that `atol=0.1`, which I did not guess.

## Measuring the noise floor

First, everything that is *allowed* to change, and what it did to the logits (max absolute difference across all ten images and classes; logit values span roughly −9 to +21):

| legitimate variation | max &#124;Δlogit&#124; | top-1 |
|---|---|---|
| same GPU, rerun | 0.0 exactly | same |
| checkpoint reload, fresh process | 0.0 exactly | same |
| batch of 1 instead of batch of 10 | 2.2e-03 | same |
| CPU instead of GPU | 1.6e-03 | same |
| channels_last memory format | 2.2e-03 | same |
| autocast fp16 | 6.1e-03 | same |

A side-fact worth pausing on: eval-mode inference on my GPU is *bit-exact* across reruns and checkpoint reloads, even with `cudnn.benchmark` on. The [non-determinism that plagued training](/posts/pytorch-same-seed-different-model) comes overwhelmingly from backward-pass atomics; a lone forward pass, same shapes and same device, replays identically. Drift only appears when the *kernels themselves* legitimately change — different batch size selects different convolution algorithms, CPU uses different math entirely, fp16 rounds — and all of it stays at or under **6e-3**. That's the floor.

## Measuring the bug signals

Then the same bugs from the damage-table post, replayed against the golden set:

| injected bug | max &#124;Δlogit&#124; | top-1 on 10 goldens | accuracy cost (8/20 post) |
|---|---|---|---|
| ImageNet mean/std instead of CIFAR's | **1.15** | **same** | −0.5%p |
| model left in train() mode | 5.0 | **same** | batch-dependent |
| JPEG q=80 re-encode | 5.3 | changed | −6.9%p |
| BGR channel order | 5.7 | changed | −12.7%p |
| resize 32→24→32 round-trip | 6.9 | changed | −16.2%p |
| normalization skipped | 16.5 | changed | −43.2%p |
| forgot /255 | 483 | changed | −72.1%p |

The weakest signal is 1.15 — the ImageNet-stats bug, the one that costs only half a point of accuracy and that I called nearly harmless in August. Even it sits **190x above the noise floor**. So the gap the tolerance lives in spans two orders of magnitude: `atol=0.1` is 16x above the worst legitimate drift and 11x below the weakest bug. Every row above the line passes, every row below fails, with margin to spare on both sides. That's not a clever threshold; it's a wide-open door the measurement walked through.

The column worth the most attention is top-1. Two of the seven bugs — wrong normalization stats and forgotten `eval()` — leave every one of the ten predicted classes unchanged. A spot check of "do the predictions look right" passes them; an accuracy eval needs the full test set to notice half a point. The logit comparison flags both at hundreds of times the noise floor, from ten images, in milliseconds. Comparing raw outputs instead of decisions is the entire value of the test.

![A row of identical tuning forks being struck, with one fork ringing visibly off-pitch and a meter catching the deviation](./figure-1.jpg)

## Notes from actually wiring it up

Store the golden *inputs* as raw uint8, never as preprocessed tensors — a saved tensor smuggles the old pipeline into the test and blinds it to exactly the bugs it exists to catch. Recompute golden logits only when the model is retrained, as part of the training run itself. And run the test at both CI time and serving startup: the damage-table post was about bugs introduced by *rewriting* preprocessing for production, and a C++ serving stack can read the same two `.npy` files.

Scope, honestly: one small model, one dataset, my usual single machine. A deeper network or a heavier input pipeline will have a different noise floor — fp16 drift compounds with depth — so the two tables are not portable constants. The *procedure* is: measure your legitimate variation, measure a couple of injected bugs, and put the tolerance in the gap. Mine took one afternoon, which is less time than I once spent finding a BGR swap by reading serving code line by line.
