---
title: "ONNX Runtime's Optimization Levels Do Completely Different Things to Different Models"
description: "Everyone leaves ONNX Runtime's graph optimization at the default and moves on — reasonable, but it hides where the speed actually comes from. Measured across my three models: the int8 network is a 3.4x win that collapses to a pessimization if optimization is off, ResNet18 gets 35% from layout transformation and almost nothing from operator fusion, and at the 0.05 ms scale I briefly 'discovered' a regression that evaporated on re-measurement."
slug: "ort-graph-optimization-levels"
date: 2026-09-22
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["onnx", "onnxruntime", "deployment", "cpu inference", "optimization"]
---

ONNX Runtime has a `graph_optimization_level` knob with four settings — disable, basic, extended, all — and it defaults to all, which is why nobody ever looks at it. I wanted to look anyway, for the same reason as [the threading post](/posts/cpu-inference-threading-measured): the models I deploy are nothing like the models in vendor benchmarks, and defaults tuned for one can behave strangely on the other. So: my three standing CPU models — the tiny HAR fp32 network (374 KB), [its static-int8 sibling](/posts/har-model-to-int8-onnx) (103 KB), and ResNet18 (43 MB) — at every level, with batch-1 latency over 500 runs, node counts of the optimized graph, and session startup cost.

## Three models, three entirely different stories

| level | HAR int8 (34 nodes) | ResNet18 fp32 (65 nodes) | HAR fp32 (11 nodes) |
|---|---|---|---|
| disable | 0.089 ms (34 nodes) | 9.95 ms (65 nodes) | 0.066 ms (11) |
| basic | 0.089 ms (34) | 9.93 ms (49) | 0.063 ms (11) |
| extended | 0.042 ms (10) | 9.84 ms (40) | 0.047 ms (8) |
| all | **0.026 ms (12)** | **6.37 ms (25)** | 0.058 ms (8) |

The int8 column is the dramatic one, and its top row contains the finding I'd frame: **an int8 model with graph optimization disabled runs slower than the fp32 model it was quantized from** — 0.089 ms against 0.066. Quantized graphs are full of QLinear operators and quantize/dequantize pairs that only pay off once the *extended* pass fuses them into integer kernels (34 nodes collapse to 10, latency halves); before that pass, quantization is a pessimization. If you ever load a quantized model through a runtime path that lowers the optimization level — some wrappers do, "for compatibility" — you silently get the worst of both worlds: int8 accuracy loss at worse-than-fp32 speed.

ResNet18 tells the opposite story about *where* speed lives. The basic and extended passes remove twenty-five nodes' worth of Conv+BN foldings and activation fusions — the optimizations everyone can name — and buy a combined **one percent**. The jump to *all*, which on CPU triggers the NCHWc layout transformation (rewriting the whole graph to a cache-friendly blocked memory format), buys **35%** in one step. Node count drops from 65 to 25 across the levels, but the correlation between nodes removed and milliseconds saved is essentially zero: sixteen nodes for 0.2%, then fifteen nodes for 3.5 ms. Graph size is bookkeeping; memory layout is money.

![Three different machines on a workbench responding to the same tuning dial in completely different ways — one transforms dramatically, one barely reacts until the final click, one just flickers](./figure-1.jpg)

## The regression that wasn't

The HAR fp32 column initially handed me a much better headline: *all* measured 19% slower than *extended* (0.058 vs 0.047), meaning the default would be actively wrong for tiny models — the layout transformation overhead exceeding its benefit at this scale. Plausible mechanism, satisfying contrarian shape, and [I've been burned by exactly that combination before](/posts/imu-rotation-angle-sweep), so I re-ran the comparison three times before writing this paragraph. It evaporated: across trials, extended landed at 0.056–0.064 ms and all at 0.054–0.061, fully overlapping. At the 0.05-millisecond scale, run-to-run drift between whole benchmark invocations is larger than the effect I was about to publish; the 500-run median protects against in-run jitter but not against the machine being in a different mood five minutes later. The honest reading of that column is "extended and all are the same, both modestly better than nothing" — and a reminder that a microbenchmark's precision has to be established before its verdict is quoted.

Two practical footnotes. Optimization runs at session creation and isn't free — ResNet18's startup went from 73 ms to 129 ms at level all, which matters only if you cold-start sessions per request (don't). And you can pay it offline via `optimized_model_filepath`, but ORT itself warns that graphs serialized above *extended* contain hardware-specific transforms — an NCHWc-optimized model saved on my desktop shouldn't ship to a different CPU family. The takeaway is unexciting on purpose: the default is correct, this knob is not a tuning opportunity — it's a footgun that only fires if something lowers it. But knowing *which* pass feeds *your* model is worth the ten minutes: mine told me the int8 deployment lives or dies on operator fusion, while the big fp32 model's entire speed story is memory layout, and no amount of node-counting would have revealed either.
