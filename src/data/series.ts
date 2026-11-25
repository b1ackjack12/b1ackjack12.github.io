// Series hub metadata. A post joins a series through its `series` frontmatter key;
// order within a series is the post id (folder name, which starts with the date).
export interface SeriesMeta {
  slug: string;
  title: string;
  tagline: string;
  intro: string;
}

export const SERIES: SeriesMeta[] = [
  {
    slug: 'cutmix-tax',
    title: "The CutMix Confidence Tax",
    tagline: 'Mixup, CutMix and Cutout measured for calibration on CIFAR-10 and IMU data: why soft labels deflate confidence, and how much of it reaches clean inputs.',
    intro:
      'This series starts from one prediction, that a soft-label augmentation’s mean confidence can be computed from its labels before training, and follows it through budget sweeps, patched-input probes, temperature scaling, dose-response and a crossover to a 1-D sensor classifier. Every post uses paired seeds or paired subject splits with t-statistics and win counts, and every result links back to the harness that produced it.',
  },
  {
    slug: 'imu-har',
    title: 'IMU Activity Recognition, Measured',
    tagline: 'Human activity recognition on UCI HAR with honest subject splits: preprocessing, augmentation, window length, sampling rate, sensor failure and data budgeting.',
    intro:
      'A small 1-D CNN on accelerometer and gyroscope windows, evaluated on subjects it has never seen. The series moves from preprocessing and subject-wise splitting to a fifty-split paired harness that can resolve sub-point effects, then uses that harness to price augmentation, window length, sampling rate, missing sensors and the recruit-more-subjects question.',
  },
  {
    slug: 'pytorch-performance',
    title: 'PyTorch Training Performance on One GPU',
    tagline: 'DataLoader workers, GPU augmentation, TF32, autocast, channels_last, torch.compile and memory limits, each timed on a Windows RTX 4080 workstation.',
    intro:
      'The performance checklists everyone copies, measured on one real machine. Several of the famous flags did almost nothing, one well-known setting backfired, and the Windows-specific traps turn out to matter more than the tuning advice written on Linux.',
  },
  {
    slug: 'training-knobs',
    title: 'Training Hyperparameters, Dose by Dose',
    tagline: 'Seed noise, optimizers, schedules, EMA, early stopping, label smoothing, weight decay, warmup and clipping measured with paired seeds on a fixed CIFAR-10 budget.',
    intro:
      'It begins by measuring how much accuracy the random seed alone moves, and then holds every training knob to that standard: paired seeds, t-statistics, win counts, and where a knob has a dial, a dose-response curve rather than a single setting. Most of the folklore survives in a weaker form than it is usually stated.',
  },
  {
    slug: 'deployment',
    title: 'From PyTorch to Deployment',
    tagline: 'ONNX export, quantization, preprocessing bugs, inference threading, batching curves and CPU-vs-GPU serving, measured with latency and accuracy numbers.',
    intro:
      'What actually breaks and what actually costs between a trained PyTorch model and a model serving requests: export failures, quantization results that contradict the brochure, preprocessing mistakes ranked by damage, and the serving curves that decide between CPU and GPU.',
  },
];

export const SERIES_BY_SLUG: Record<string, SeriesMeta> = Object.fromEntries(SERIES.map((s) => [s.slug, s]));

// Chronological order for posts. The glob loader uses the frontmatter `slug` as the entry id, so the id
// is not date-prefixed; sort by date and break same-day ties with the file path (folder names start with the date).
export const byDate = (a: any, b: any): number =>
  a.data.date.valueOf() - b.data.date.valueOf() || String(a.filePath ?? a.id).localeCompare(String(b.filePath ?? b.id));
