---
title: "Quantizing an ONNX Model in Practice: The Numbers Nobody Shows You"
description: "I quantized a real trained model with onnxruntime and measured everything — file size, latency, accuracy. Dynamic quantization made my CNN 3x slower, and that's not even the strangest result."
slug: "quantizing-onnx-models-in-practice"
date: 2026-07-22
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["quantization", "onnx", "deployment", "edge computing", "deep-learning"]
---

A while ago I wrote about [quantization from the deployment side](/posts/optimizing-deep-learning-models-for-edge-via-quantization) — PTQ vs. QAT, hardware constraints, calibration data. That post had a weakness I was aware of while writing it: no numbers. Every quantization article says "roughly 4x smaller, significantly faster, minimal accuracy loss," and almost none of them show a measurement you can argue with.

So this time I did the whole thing end to end on my machine and wrote down what actually happened. The setup: a small CNN trained on FashionMNIST to 88.1% test accuracy, exported to ONNX with the checks from [my export post](/posts/what-actually-breaks-exporting-pytorch-to-onnx) (dynamic batch axis, output parity `7.6e-06` against PyTorch). Tools: PyTorch 2.6.0, onnxruntime 1.20, running on a desktop Zen 4 CPU. Every number below is from a real run.

Spoiler: two of the three headline claims held up, and the third failed in an instructive way.

## Attempt 1: dynamic quantization, the one-liner that isn't

The `onnxruntime.quantization` docs start you off with dynamic quantization — weights stored as int8, activation scales computed on the fly at inference time. It really is one call:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("model_fp32.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)
```

This runs without complaint and writes a file 3.9x smaller. Then you try to *load* it:

```text
[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation
for ConvInteger(10) node with name '/conv1/Conv_quant'
```

The quantizer happily emitted a `ConvInteger` node with **signed** int8 inputs, and onnxruntime's CPU execution provider has no kernel for that combination — it only implements the unsigned variant. The tool that produced the model and the runtime that refuses it ship in the same package. Switching to `weight_type=QuantType.QUInt8` produced a model that loads and runs.

This is the same lesson as the export post, one layer up: **a conversion step succeeding tells you nothing until you've loaded and run the artifact.** My quantization scripts now construct an `InferenceSession` immediately after writing the file, before any evaluation.

## Attempt 2: static quantization, which wants two things from you

Static quantization pre-computes activation scales from calibration data, so the graph carries fixed quantize/dequantize parameters instead of measuring ranges at runtime. It needs a preprocessing pass first (shape inference and graph cleanup — skip it and you get warnings, and potentially worse quantization):

```python
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process("model_fp32.onnx", "model_pre.onnx")
```

And it needs calibration data, supplied through a small class you implement yourself:

```python
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType

class MyCalibReader(CalibrationDataReader):
    def __init__(self, dataset, n_samples=200, batch=20):
        idx = np.random.RandomState(0).choice(len(dataset), n_samples, replace=False)
        xs = torch.stack([dataset[i][0] for i in idx])
        self.batches = [xs[i:i+batch].numpy() for i in range(0, n_samples, batch)]
        self.it = iter(self.batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"input": b}

quantize_static("model_pre.onnx", "model_static_int8.onnx",
                MyCalibReader(train_ds), weight_type=QuantType.QInt8)
```

Two hundred samples was enough for this model; the calibration pass took a couple of seconds. Note that the samples go through the *same normalization as inference* — the reader feeds preprocessed tensors, not raw images. Feeding unnormalized data here silently produces garbage scales, which is the file-format version of the calibration mismatch I warned about in the earlier post.

## The numbers

Everything measured with onnxruntime on CPU, median of 200 runs after warmup:

| Model | Size | Test acc | Latency b=1 | Latency b=32 |
|---|---|---|---|---|
| fp32 ONNX | 1527 KB | 88.14% | 0.072 ms | 0.89 ms |
| dynamic int8 | 390 KB | 88.14% | **0.190 ms** | **3.38 ms** |
| static int8 | 392 KB | 88.40% | 0.042 ms | 0.61 ms |

![Bar chart comparing fp32, dynamic int8, and static int8 on size and latency, with dynamic int8 latency towering above the others](./figure-1.jpg)

Three things in that table deserve a closer look.

**Size: exactly as advertised.** 3.9x reduction, both methods. Weights dominate the file and int8 weights are a quarter the bytes. This is the one promise quantization always keeps.

**Dynamic quantization made the model 2.6–3.8x slower.** Not slightly slower — several times slower, and worse at larger batch. The graph explains it: the dynamic model inserts `DynamicQuantizeLinear` nodes that scan activations and compute scales *on every single inference*, then runs `ConvInteger` plus cast/multiply chains to dequantize. For convolutions on a modern desktop CPU, that overhead swamps any integer-math gain. Dynamic quantization earns its keep on models dominated by large matrix multiplies — transformers, RNNs — which is exactly what the onnxruntime docs recommend it for, in a sentence that's easy to skim past. For a CNN it was strictly worse than fp32 on every latency measurement I took.

**Static quantization gained accuracy.** 88.40% vs. 88.14% — the int8 model classified 26 more test images correctly than the fp32 model it was made from. This is noise, not magic: quantization perturbs the decision boundary, and near-boundary samples land on either side by luck. The honest reading is "accuracy unchanged within noise," and if you rerun with a different calibration seed you may land slightly below instead. But it's a useful calibration for expectations: for an ordinary conv net at int8, the accuracy cost can genuinely be ~zero.

## The metric that scared me and the one that mattered

Out of habit from the export post, I also compared raw logits between the fp32 model and the quantized ones. The static int8 model's worst-case logit deviation was **2.34** — not `1e-06`, not `1e-03`. Seeing that number first, before the accuracy, I briefly assumed the quantization was broken.

It wasn't. Argmax agreement with fp32 was 99.49%, and test accuracy went *up*. Logits are unnormalized scores; shifting all of them by a similar amount changes nothing about the prediction. The lesson I'm keeping: **for fp32-to-fp32 conversions, max logit diff is the right check; across a precision change, it's misleading — compare decisions, not logits.** For a classifier that's argmax agreement; for a regression model you'd compare the output against your actual tolerance spec, not against floating-point epsilon.

## Then I tried a real-sized model, and the speedup vanished

A 1.5 MB CNN is a toy, so I ran size and latency on ResNet18 (static int8, same recipe):

| Model | Size | Latency b=1 |
|---|---|---|
| ResNet18 fp32 | 42.6 MB | 6.31 ms |
| ResNet18 static int8 | 10.7 MB | 6.43 ms |

Size: 4x, as always. Speed: **nothing.** Within measurement noise, identical — and this is not a bug. My desktop CPU executes fp32 convolutions through very mature vectorized kernels, and the int8 path (quantize, integer conv, dequantize) merely breaks even against them at this model size. The small CNN got its 1.7x because its fp32 baseline had less optimized headroom, not because int8 is inherently fast.

![Two views of the same quantized model: on a desktop CPU the speedup bar is flat, on an NPU with INT8 MACs it is tall](./figure-2.jpg)

This is the measured version of the argument in the [edge deployment post](/posts/optimizing-deep-learning-models-for-edge-via-quantization): quantization's latency win comes from *hardware with dedicated integer units* — NPU MACs, DSP instructions — not from int8 being magically cheaper everywhere. On my workstation, static quantization of ResNet18 buys memory footprint and bandwidth, which on an embedded target is often the binding constraint anyway. But if you benchmark on your development machine and promise a 3x speedup on the device, you are writing a check the silicon may not cash — in either direction.

## What I'd tell someone doing this for the first time

Construct an `InferenceSession` the moment a quantized file exists, before you celebrate — and if dynamic quantization refuses to load a conv model, it's the `QInt8`/`ConvInteger` trap, not your model. Use static quantization for conv nets and save dynamic for matmul-heavy models. Run `quant_pre_process`, and make the calibration reader feed exactly what inference feeds. Judge the result by task metrics and argmax agreement, not by logit differences that look alarming and mean little. And measure latency on the target, because the desktop numbers answer a question nobody is asking.

Total time from fp32 ONNX to a verified int8 model: under an hour, most of it spent on the `ConvInteger` surprise. The measuring itself is cheap. Not measuring is what gets expensive later.
