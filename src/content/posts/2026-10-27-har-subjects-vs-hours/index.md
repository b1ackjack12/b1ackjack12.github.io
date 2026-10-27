---
title: "I Threw Away Half My Training Data and Lost Nothing. Then I Threw Away Half My People and Lost Three Points"
description: "Every sensor-data project eventually faces the same budgeting question: recruit more subjects, or record longer sessions with the ones you have? The fifty-split harness can answer that directly — a subject-count dose-response from 21 people down to 5, plus the control arm that settles the real question: the same number of windows drawn from all 21 people versus 10 people's complete recordings. Same volume, different diversity, +1.18 points and 40 wins in 50. Hours are cheap. Humans are the dataset."
slug: "har-subjects-vs-hours"
date: 2026-10-27
author: "B1ack"
draft: false
thumbnail: "./thumbnail.jpg"
tags: ["imu", "har", "data collection", "sensors", "statistics"]
---

The HAR lane of this blog keeps circling one fact from different directions: [subject identity is the hard axis](/posts/imu-har-subject-vs-random-split). Split by subject and accuracy drops eight points; the worst subject in any split trails the average by fifteen; [every augmentation win](/posts/har-augmentation-fifty-splits) and [every robustness trick](/posts/har-gyro-failure-insurance) earns its keep disproportionately on the hardest person. All of which raises the question any real project has to answer *before* any of that tuning: when you're planning data collection with a fixed budget, do you spend it recruiting more people or recording more hours per person?

That's measurable with the [usual harness](/posts/har-augmentation-fifty-splits). Fifty paired splits; in each one, 9 random test subjects held out and a pool of 21 to train from. The dose-response arms train on all windows from 21, 15, 10, or 5 randomly chosen pool subjects. And then the arm that actually decides the question: **mix10**, which trains on windows drawn from *all 21* pool subjects but subsampled to exactly match the 10-subject arm's window count, split by split. Same data volume, different human diversity — whatever separates mix10 from s10 is purely the number of distinct people in the recordings.

## The dose-response, and the control that explains it

| training set | windows | accuracy | Δ vs full | t | worst subject Δ |
|---|---|---|---|---|---|
| 21 subjects, all windows | ~7,208 | 92.66 ±2.76 | — | — | — |
| 15 subjects | ~5,143 | 92.05 | −0.61 | −3.5 | −2.30 |
| 10 subjects | ~3,419 | 91.25 | −1.40 | −5.9 | −5.70 |
| 5 subjects | ~1,696 | 89.41 | **−3.24** | −7.0 | **−9.02** |
| **all 21 subjects, half the windows** | ~3,419 | **92.44** | **−0.22** | −1.5 | −0.49 |

Read the dose-response first. Dropping from 21 to 15 subjects costs a statistically solid but modest −0.61. Ten subjects costs −1.40. Five costs −3.24. And as always on this dataset, the mean is the gentle version of the story: the worst-subject column degrades three times faster (−2.30, −5.70, −9.02), because the fewer people the model has met, the worse its odds that *someone* in the test set moves like nobody in the training set. The curve is superlinear in the direction that hurts — each halving of recruitment costs more than the last.

Now the control arm, which is the whole post. mix10 trains on the *same ~3,419 windows* as the 10-subject arm — literally matched per split — but draws them from all 21 people. The paired comparison between those two arms is **+1.18 points (t = 5.3), forty wins in fifty**, and on the worst subject the gap widens to over five points (−0.49 vs −5.70 against baseline). Same storage, same GPU-minutes, same label volume; the only difference is how many distinct humans contributed. Diversity didn't just beat volume — volume barely registered: mix10, running on **47% of the data**, lands within −0.22 of the full dataset, a difference the pairing can't distinguish from zero (t = −1.5). Every hour of recording past a surprisingly early point was, for this model and task, redundant with the hours already on disk. The people weren't.

![Two audition judges: one bored, listening to the same singer five times over; the other delighted, hearing five genuinely different performers](./figure-1.jpg)

## Why the hours run out of things to say

The mechanism is the one the [sampling-rate post](/posts/imu-sampling-rate-measured) found on the time axis, now showing up on the dataset axis: human activity signal is highly redundant. Within one subject, the thousandth window of walking looks overwhelmingly like the hundredth — same gait frequency, same device placement, same personal quirks. Those windows mostly re-teach what the model already knows about *that person*. A new subject, by contrast, delivers the thing the model can't get anywhere else: a new point in the space of bodies, speeds, and wearing styles — precisely the axis the test set will probe when [a stranger puts on the device](/posts/har-loso-error-analysis). In statistical terms, the effective sample size of subject-correlated windows saturates fast; the effective sample size of *subjects* is just the headcount.

The budget translation is blunt. Recording sessions are the cheap part of a study (the subject is already there, instrumented, consented) and recruitment is the expensive part — so the temptation is always to stretch sessions and trim headcount. This table prices that temptation: trading 11 people for longer recordings from 10 costs more accuracy than *deleting half of all your data*. If the choice is 10 people for two hours or 21 people for one, it isn't close — and if your collected dataset is bloated, the order to trim in is now measured: cut hours per person freely, defend every person. Scope lines as usual: one dataset (30 subjects, lab conditions, one device position), one small CNN, and a subject pool capped at 21 — I can't tell you where the subject curve itself saturates, because UCI HAR ran out of people before the model did. Which is, come to think of it, the finding restated.
