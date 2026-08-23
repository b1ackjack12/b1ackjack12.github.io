---
title: "Forget model.eval() and Your Accuracy Depends on Your Traffic"
description: "The most famous two lines in PyTorch inference, measured in their absence: train-mode batch norm scores exactly chance on single requests and looks nearly fine at batch 256 — which is why load tests can't catch it. Plus what no_grad actually buys (hint: not speed)."
slug: "forgotten-model-eval-measured"
date: 2026-08-24
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["pytorch", "deployment", "debugging", "batchnorm", "deep-learning"]
---

[Last week's post](/posts/image-preprocessing-bugs-ranked) ranked the damage from preprocessing bugs — the glue code *in front of* the model. Today is the other classic inference mistake, the one inside the serving code itself: forgetting `model.eval()`, and its sibling, forgetting `torch.no_grad()`. Every PyTorch tutorial recites these two lines; almost none of them show what actually happens when you omit them. Having a trained CIFAR-10 model (89.51% in correct eval mode) still warm from last week, I measured it.

The punchline is worth stating up front, because it's the mechanism that makes this bug genuinely nasty rather than merely embarrassing: **the damage is a function of your serving batch size.** The same forgotten line costs you everything or nearly nothing depending on how many requests you happen to bundle together — and the direction of that dependence is precisely backwards from how software gets tested.

## What train mode actually changes

`model.train()` vs `model.eval()` toggles two behaviors. Batch norm layers switch between normalizing with *the current batch's* statistics (train) and the running statistics accumulated during training (eval). Dropout switches between randomly zeroing activations (train) and passing everything through (eval). A model left in train mode at inference doesn't crash, warn, or log — every output is a well-formed probability distribution. It's just computed with the wrong math.

## The batch-size table

Evaluating the frozen model in train mode, batching the (shuffled) test set at different sizes:

| serving batch size | accuracy |
|---|---|
| 1 | **10.00%** |
| 2 | 60.00% |
| 4 | 74.51% |
| 16 | 85.73% |
| 64 | 88.15% |
| 256 | 89.09% |
| *(correct eval mode)* | *89.51%* |

At batch 1, the model scores exactly chance — batch norm with a single sample normalizes each feature map by its own statistics, obliterating the scale and shift information the network learned to rely on. At batch 2 it claws back to a coin flip and a half. And by batch 256, train-mode inference sits 0.42 points below correct — a deficit indistinguishable from noise unless you were looking for it.

Now overlay how services get tested. Offline evaluation scripts and load tests process big comfortable batches — 256, 512 — where this bug costs a third of a point and passes every check. Production traffic, especially early, arrives as single requests: batch size one, where the same code is a random number generator. **The better your throughput-oriented testing, the better this bug hides.** It is the only bug I've measured on this blog whose severity is inversely proportional to how hard you test for it.

![A security checkpoint waving through a big orderly group while stopping and searching every single lone traveler at the same gate](./figure-1.jpg)

Two quieter consequences ride along with the headline. First, in train mode your outputs for a given image *depend on the other images in the batch* — the batch statistics are shared — so identical requests return different answers depending on their neighbors, which from a client's perspective is nondeterminism with no visible source. Second, train-mode batch norm keeps *updating its running statistics* while serving: the model rewrites itself with every request. I measured one pass of train-mode "serving" over the test set, then switched back to proper eval mode — accuracy had drifted from 89.51% to 89.12%. Same-distribution traffic cost 0.4 points of permanent damage; production traffic that drifts from the training distribution would steer the statistics correspondingly further. A bug that corrupts state, not just outputs.

For completeness I also trained a variant with dropout (p=0.3) and served it in train mode at large batch: 89.26% became 88.30%, with repeated shuffled evaluations wandering across a ~0.2-point band from the random masks. Dropout-left-on is real but survivable — a point and some jitter. The catastrophic component of forgotten `eval()` is batch norm, and specifically batch norm meeting small batches.

## no_grad: the other line, and what it's actually for

`torch.no_grad()` folklore says "faster inference." Measured on batch-512 forward passes: **7.82 ms with, 7.86 ms without.** No speed difference worth naming — the forward math is identical either way. What changed is memory: **479 MB peak with `no_grad`, 2,224 MB without.** Autograd, preparing for a backward pass that will never come, retains every intermediate activation in the graph; at batch 512 that's 4.6x the footprint, scaling with batch size and model depth. The failure mode isn't slowness — it's an inference server whose memory usage quietly quadruples, OOMs under a traffic spike, or leaks steadily if outputs (each dragging its graph behind it) get accumulated into a list. `torch.inference_mode()` is the stricter modern spelling of the same idea, and what I use in new code.

## The two-line defense for the two-line bug

Everything in this post is preventable with asserts that cost nothing: at the top of the serving entrypoint, `assert not model.training`, and construct outputs under `inference_mode`. And the [golden-sample test from last week](/posts/image-preprocessing-bugs-ranked) catches this whole class too, with one amendment I'd now consider mandatory: **include a batch-size-1 golden sample.** A batch-256 golden check would have signed off on the 89.09% impostor; the single-image check fails it instantly and unmissably, at 10% accuracy.

Ranked against the whole deployment series, forgotten `eval()` at batch 1 is now the single most destructive mistake I've measured — worse than any preprocessing bug, seventy-nine points below baseline, from one absent line that every tutorial warns about and every codebase eventually omits anyway. The warnings clearly aren't the bottleneck. The tests are.
