---
title: "I Set Every Seed and Trained Twice. The Models Diverged at Step 1"
description: "torch.manual_seed is not reproducibility. I measured how fast two identically-seeded runs drift apart on GPU, why it happens, and what bit-exact determinism actually costs — 16%, not the 2x folklore claims."
slug: "pytorch-same-seed-different-model"
date: 2026-08-18
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "reproducibility", "gpu", "deep-learning", "debugging"]
---

There's a ritual at the top of every training script, mine included: `torch.manual_seed(0)`, `np.random.seed(0)`, and the comfortable belief that the run is now reproducible. I've typed it for years. Last week, while chasing a one-point accuracy difference I couldn't explain, I finally tested the belief directly: same seed, same data order, same model init, same machine, same everything — train ResNet18 for 200 steps, twice, and compare.

The two loss curves matched at step 0 and **diverged at step 1**. After 200 steps, the largest difference between corresponding weights was 0.237 — for parameters that mostly live between −0.1 and 0.1, that's not drift, that's a different model. Both runs were "seeded." Neither was reproducible.

## Where the randomness sneaks in

The seeds did their job: initialization, data order, and every explicit random draw were identical, which is why step 0 agreed perfectly. The divergence enters in the *backward pass*, and it isn't randomness in the usual sense at all — it's floating-point arithmetic meeting parallel hardware.

Floating-point addition is not associative: `(a + b) + c` and `a + (b + c)` can differ in the last bit. GPU kernels sum gradient contributions across thousands of threads, many using atomic operations whose completion *order depends on the scheduler* — which threads won the race on this particular run, on this particular clock. Same numbers, different addition order, last-bit differences in the gradients. Then SGD amplifies them: a 1e-8 disagreement in a gradient becomes a slightly different weight, which becomes a different activation, and the gap compounds every step. By my run's end, noise that began around the last decimal place a float can represent had grown into 0.237. It's the butterfly effect with a commit hash.

None of this is a PyTorch bug; it's documented behavior, and `cudnn.benchmark` (which I've [previously measured](/posts/tf32-cudnn-benchmark-measured) and left enabled) adds a second layer by autotuning to potentially different — nondeterministic — conv algorithms per run.

![Two identical glowing racing lines leaving one starting gate and slowly fanning apart into visibly different paths across a dark circuit board landscape](./figure-1.jpg)

## Does 0.237 matter? Only when you're asking small questions

For final accuracy, this nondeterminism behaves like one more seed you didn't choose — my [augmentation experiment](/posts/batch-level-gpu-augmentation-accuracy) put ordinary seed-to-seed spread at ±0.3–0.5 points on a small CIFAR setup, and same-seed GPU noise lands in the same statistical bucket. If your effect size dwarfs that, you can ignore this whole post.

The cases where it bites are the small questions, which are most of debugging. "Did my refactor change the numerics?" — you can't diff against the pre-refactor run, because *nothing* diffs clean, ever. "Is this 0.4-point improvement my new loss term or luck?" — under nondeterminism you can't even hold the luck constant while you toggle the loss term. A/B comparisons where the honest answer requires the same noise on both sides are exactly where "seeded but nondeterministic" quietly wastes days. I know because the one-point mystery that started this post turned out to be two runs of the *same code*.

## The fix, and its real price

PyTorch ships the switch; it just isn't the seed:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
# and, for CUDA >= 10.2, before the process starts:
# CUBLAS_WORKSPACE_CONFIG=:4096:8
```

With those set, my two 200-step runs matched **bit-for-bit**: max weight difference exactly 0, every loss value identical to the last digit. Two footnotes from the setup: the `CUBLAS_WORKSPACE_CONFIG` environment variable is mandatory (you get a clear RuntimeError without it), and `use_deterministic_algorithms` will refuse to run ops that simply have no deterministic GPU implementation — ResNet18 sailed through, but if your model contains one of the holdouts, the flag fails loudly and your options are CPU fallback or living without determinism for that op. Loud failure, for once, is the feature.

The folklore price for all this is "2–3x slower," usually delivered as a reason not to bother. Measured, on the ResNet18 training step:

| precision | default | deterministic | cost |
|---|---|---|---|
| fp32 | 17.94 ms | 20.90 ms | +16.5% |
| bf16 autocast | 12.42 ms | 14.37 ms | +15.6% |

Sixteen percent. Some of that isn't even determinism proper — it includes giving up `cudnn.benchmark`, which [I measured at ~3%](/posts/tf32-cudnn-benchmark-measured) on this model. The 2–3x stories presumably come from specific unlucky ops or older stacks, and maybe your model contains one; that's what the two-minute measurement is for. On mine, bit-exact reproducibility costs about as much as a modestly larger batch.

## Where the switch sits now

My default stays fast and nondeterministic — for training throughput, 16% is real money, and ordinary training doesn't need bit-exactness. But the flag has earned a permanent spot in three situations: any debugging session that compares two runs (which is to say, any debugging session), numerics-sensitive refactors where I want `git diff`-grade certainty that outputs didn't change, and small-effect experiments where the noise floor would otherwise eat the result. The deciding question is one sentence: *am I about to compare two runs and interpret the difference?* If yes, 16% is the cheapest insurance in this entire optimization series — and unlike the seeds ritual, it actually buys what it claims to.
