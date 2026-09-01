---
title: "Batch 1 Uses 8% of Your GPU, and Other Things the Batching Curve Says"
description: "I swept batch size from 1 to 256 across three model sizes in fp32 and fp16 on an RTX 4080 SUPER. The first eight images are literally free, fp16 is slower than fp32 until batch 16, and the biggest batch is never the fastest one. The whole serving policy falls out of three tables."
slug: "gpu-inference-batching-measured"
date: 2026-09-02
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["gpu", "deployment", "performance", "pytorch", "inference"]
---

Last week I measured [how CPU inference should divide its cores](/posts/cpu-inference-threading-measured) and found the default policy tuned for the wrong workload. Today, the GPU-side sibling: when requests hit a GPU model server, how many should you group into a batch? Everyone knows the direction of the answer — batching amortizes overhead, more is faster, up to a point. What I wanted was the actual shape: where the free lunch ends, where the knee is, and what batch 1 — the default of every "wrap the model in FastAPI" tutorial — actually costs. Three model sizes (my CIFAR-scale SmallResNet at 32×32, ResNet18 and ResNet50 at 224×224), batches 1 through 256, fp32 and fp16, H2D copy from pinned memory included because real requests arrive over the wire.

## The flat zone: your first eight images are free

Median wall time *per batch*, fp32:

| batch | SmallResNet | ResNet18 | ResNet50 |
|---|---|---|---|
| 1 | 0.67 ms | 2.13 ms | 4.98 ms |
| 4 | 0.67 ms | 1.89 ms | 5.06 ms |
| 8 | 0.66 ms | 1.91 ms | 4.58 ms |
| 16 | 0.72 ms | 2.95 ms | 8.07 ms |
| 32 | 0.67 ms | 5.53 ms | 18.04 ms |

Read down the columns: for ResNet50, a batch of eight takes *the same time* as a batch of one — the second through eighth images are literally free. SmallResNet's flat zone extends to batch 32; thirty-two CIFAR images cost what one costs. At these sizes the GPU isn't computing so much as *waiting* — kernel launches, memory latency, a few thousand cores with almost nothing to chew on — and extra images ride along in the bubbles. Any server processing these models one request at a time is spending its wall-clock on overhead and calling it inference.

## The knee, and the decline nobody mentions

Push batch size past the flat zone and throughput climbs until the GPU saturates:

| model, precision | batch 1 | peak throughput | batch-1 utilization |
|---|---|---|---|
| SmallResNet fp32 | 1,499 img/s | 82,246 img/s (b128) | **1.8%** |
| SmallResNet fp16 | 1,117 img/s | 139,589 img/s (b256) | 0.8% |
| ResNet18 fp32 | 469 img/s | 5,782 img/s (b32) | **8.1%** |
| ResNet18 fp16 | 388 img/s | 9,895 img/s (b64) | 3.9% |
| ResNet50 fp32 | 201 img/s | 1,982 img/s (b16) | **10.1%** |
| ResNet50 fp16 | 158 img/s | 3,535 img/s (b32) | 4.5% |

The headline column is the last one: batch-1 serving uses between 2% and 10% of what the same silicon delivers at its best batch. A 4080 SUPER serving ResNet18 requests one at a time is, functionally, a twelfth of a GPU billed as a whole one.

But the subtler finding is that peak throughput happens at *moderate* batches — 16 to 64 — and then **declines**. ResNet50 fp32 drops from 1,982 img/s at batch 16 to 1,546 at batch 256, a 22% loss for going bigger; ResNet18 fp32 loses 17% between batch 32 and 256. Once the GPU is saturated, larger batches stop amortizing anything and start thrashing cache. "Crank the batch until OOM" — my default for offline jobs, and I suspect most people's — lands measurably past the optimum. The biggest batch was never the fastest configuration for any model here.

![A cargo ship being loaded far past its efficient waterline while smaller, faster boats depart fully loaded](./figure-1.jpg)

## The fp16 twist

Mixed precision has a strange opening act: **at batch 1, fp16 was slower than fp32 for all three models** — 21% slower for ResNet18 (2.58 vs 2.13 ms), 27% for ResNet50, 34% for SmallResNet. Below batch 16 there's simply not enough arithmetic to feed the tensor cores, and the fp16 kernel plumbing costs more than the math saves — the same kernel-selection pathology that made [fp16 lose to fp32 under channels_last](/posts/channels-last-memory-format-measured) during training. The crossover sits around batch 16, and beyond it fp16 pays properly: 1.7x peak throughput on all three models, on top of halved memory.

So precision and batch size aren't independent choices: fp16 at batch 1 is a small pure loss, fp16 at batch 32+ is the biggest single win in these tables. If you're latency-bound at tiny batches, stay in fp32 and spend your effort getting the batch up instead.

## The serving math

Put the tables together and the policy writes itself. For ResNet18 fp16, batch 32 costs 3.37 ms of compute — so a dynamic batcher that waits a few milliseconds to fill a batch turns 388 img/s into 9,489, a **24x** capacity gain, for single-digit added latency. That's the entire trick behind inference servers' dynamic batching, and the curve says you don't need their scale for it to pay: the gap between "one request, one forward pass" and "wait 5 ms, batch what arrived" is the difference between needing twenty GPUs and needing one.

Numbers as always from one machine — RTX 4080 SUPER, torch 2.6, cudnn.benchmark on, medians of 200 runs — and the knees will sit elsewhere on your silicon. But the shape generalizes, and it takes three minutes to sweep: batch until the per-batch latency stops being flat, check whether you've crossed the fp16 crossover, and stop before the decline. The curve knows your serving config better than the tutorial does.
