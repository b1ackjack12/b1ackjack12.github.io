---
title: "Quantization Cost My Model 0.13 Points. Preprocessing Bugs Cost Up to 72"
description: "I injected the classic image preprocessing mistakes — BGR order, missing normalization, wrong stats, JPEG artifacts, resize errors — into a controlled pipeline one at a time and measured exactly what each one costs. The ranking is not what code reviews optimize for."
slug: "image-preprocessing-bugs-ranked"
date: 2026-08-20
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["image-processing", "deployment", "opencv", "deep-learning", "debugging"]
---

The deployment posts on this blog have been checking the model: [export parity to 4.3e-06](/posts/har-model-to-int8-onnx), [quantization loss measured at 0.13 points](/posts/quantizing-onnx-models-in-practice). But between the verified artifact and the actual camera sits a stage nobody parity-checks: preprocessing, usually rewritten from the training code into another language or library for serving — Python/PIL on one side, C++/OpenCV on the other. Every image engineer I know, myself included, has shipped at least one bug across that rewrite. What I'd never seen is a table of what each classic mistake actually *costs*, so I built one.

The setup: a small CNN trained on CIFAR-10 to **89.51%** test accuracy (15 epochs, 36 seconds — this experiment wants a healthy model, not a record), with a bog-standard pipeline: RGB, divide by 255, normalize with the dataset's mean and std. Then I evaluated the same frozen model through nine broken variants of that pipeline, each reproducing a bug I have either written or debugged in real code. No retraining, no errors thrown anywhere — every variant produces perfectly valid tensors. Only the numbers change.

## The damage table

| pipeline | accuracy | damage |
|---|---|---|
| correct | 89.51% | — |
| ImageNet mean/std instead of CIFAR's | 89.06% | −0.5 |
| JPEG re-encode, quality 80 | 82.58% | −6.9 |
| BGR channel order (OpenCV's default) | 76.81% | −12.7 |
| JPEG quality 50 | 76.07% | −13.4 |
| resize 32→24→32, bilinear | 73.28% | −16.2 |
| JPEG quality 20 | 61.69% | −27.8 |
| resize round-trip, nearest-neighbor | 57.40% | −32.1 |
| normalization skipped (inputs in [0,1]) | 46.36% | −43.2 |
| forgot /255 (0–255 floats into normalize) | 17.40% | −72.1 |

Chance on CIFAR-10 is 10%. The bottom row is a model one step from guessing, produced by one missing line, throwing zero exceptions.

## The bug everyone reviews for is the one that doesn't matter

Start at the top, because it's the funniest row: using ImageNet's mean/std on CIFAR data — the canonical copy-paste error, the one that gets flagged in every code review and tutorial comment section — cost **half a point**. The two normalizations are similar affine maps, and the network's first batch-norm layer re-standardizes activations anyway, absorbing most of the residual shift. I'm not endorsing the bug; it's still wrong, and it will bite harder on models without early normalization layers. But there's a lesson about review culture in the fact that the most-policed preprocessing mistake was the cheapest one on the table by an order of magnitude.

The rows that actually hurt divide cleanly into two families.

## Family one: distribution shifts, dose-dependent

JPEG artifacts and resize errors degrade smoothly with severity — quality 80 costs 7 points, quality 50 costs 13, aggressive nearest-neighbor resizing costs 32. These are the "the camera pipeline isn't what you trained on" family, and their signature is *plausibility*: 76% accuracy doesn't look broken in a demo, it looks like a slightly disappointing model. The interpolation rows deserve a special glance from anyone writing serving code: the same geometric operation done with `INTER_NEAREST` instead of `INTER_LINEAR` — a one-constant difference — **doubled** the damage from 16 to 32 points. Interpolation flags are not a style choice.

BGR belongs to this family too, and it's the one I'd nominate as most dangerous per unit of visibility. OpenCV loads images blue-first; nearly everything else assumes red-first; the swap costs 12.7 points, and — this is the trap — the model still scores 77%. It works well enough to pass a casual smoke test, demo fine on easy inputs, and ship. A bug that halves your error margin while looking like a mediocre Tuesday is far more likely to reach production than one that breaks everything.

![An assembly line where a paint robot has quietly swapped red and blue paint on a row of products, and the inspector further along squints slightly but stamps them approved](./figure-1.jpg)

## Family two: scale bugs, which are fatal

The last two rows aren't degradation, they're destruction. Skip normalization and accuracy falls to 46%; feed 0–255 floats into a normalizer expecting 0–1 and you get 17% — the network receives inputs a hundred standard deviations from anything it saw in training, and the learned features simply don't fire. What makes these bugs common isn't carelessness, it's *interfaces*: `PIL.Image` gives you 0–255 uint8, `ToTensor()` silently rescales to 0–1, OpenCV gives you 0–255 again, and somewhere in a C++ port someone divides by 255 twice or not at all. The dtype is float32 either way. Nothing complains.

## The defense that catches every row

Here's what strikes me about that table as a testing problem: no unit test on the *model* catches any of it, and no type system catches it either — every broken pipeline emits float32 tensors of the right shape. The ONNX parity checks I keep [preaching](/posts/what-actually-breaks-exporting-pytorch-to-onnx) verify the model in isolation; today's table is the complement, the pipeline in isolation, and it needs its own test.

The cheap version has been standard practice in signal processing labs forever: **golden samples**. Freeze a handful of raw input images at training time, record the training pipeline's final logits for them, and make the serving stack — whatever language it's written in — reproduce those logits within tolerance as a startup or CI check. Ten images and an `allclose` catches every single row of the damage table, including the half-point one, because it compares the *pipelines* end to end rather than trusting each stage's plausible-looking output. It's twenty lines, and it would have saved me from every preprocessing bug I've shipped — the count of which I'll keep to myself.

The through-line of the deployment series, then, in one sentence each: the export can be verified to 1e-6, the quantizer costs a tenth of a point when handled with care — and the innocent-looking resize-and-normalize glue code around them can quietly cost you everything from half a point to the whole model. Budget your paranoia accordingly.
