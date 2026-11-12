# GPU Profiling and the Roofline Model: A Hands-On Guide

> Hands-on guide to diagnosing slow GPU models: the roofline model, PyTorch Profiler, and how to tell **compute-bound**, **memory-bandwidth-bound** and **launch-bound** workloads apart. Runs for free on a Kaggle Tesla T4.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-green)

---

## The question this repo answers

> You profile a model with Nsight Systems, Nsight Compute, and PyTorch Profiler, and it's running well below its expected FLOPS. Walk me through your profiling workflow: how do you use a roofline model to decide if you're compute-bound, memory-bandwidth-bound, or launch-bound, and what do you look at next in each case?

Low FLOPS is a **symptom, not a diagnosis**. This repo shows, with real measurements, that the same GPU can be compute-bound, memory-bound and launch-bound depending on the operation, and how to tell which one you have before changing any code.

---

## What you will learn

- The difference between **FLOPs** (a count) and **FLOPS** (a rate)
- What a **GPU kernel** is (and why it is not the OS kernel)
- **Arithmetic intensity** and how to read a **roofline** chart
- How to separate the three bottlenecks with measurements
- How to read the **PyTorch Profiler** table and the `trace.json` timeline
- What to do next in each case

---

## Quick start (Kaggle, free GPU)

1. Create a new [Kaggle](https://www.kaggle.com) notebook.
2. Under **Session options → Accelerator**, choose a **GPU** (T4 or P100).
3. Upload `notebooks/roofline_experiment.ipynb` or paste the code from `src/benchmark.py`.
4. Run all cells.

Or run locally on any machine with an NVIDIA GPU:

```bash
git clone https://github.com/<your-username>/gpu-profiling-roofline-handson.git
cd gpu-profiling-roofline-handson
pip install torch
python src/benchmark.py
```

> Nsight Systems and Nsight Compute usually need root access and GPU counter permissions, so they may not work in shared notebooks like Kaggle. PyTorch Profiler works everywhere and is what this repo uses.

---

## The experiment: three kernels, three bottlenecks

| Test | What it exercises | Expected class |
|---|---|---|
| 4096×4096 FP16 matmul | Lots of math per byte | Compute-bound |
| 100M-element FP16 vector add | Very little math per byte | Memory-bandwidth-bound |
| 1000 tiny `add_` ops | Launch overhead dominates | Launch-bound |

### Results on a Tesla T4

```
Matmul        : 21.6 TFLOPS
Vector add    : 252 GB/s moved, 0.042 TFLOPS
1000 tiny ops : 7.8 ms total, 7.8 microseconds per op
```

As far as I know, the T4 peaks at about **65 TFLOPS** (FP16 tensor cores) and **320 GB/s** (GDDR6), giving a ridge point near **200 FLOPs/byte**. Check NVIDIA's spec sheet for exact numbers.

| Test | Measured | Peak | Position | Verdict |
|---|---|---|---|---|
| Matmul | 21.6 TFLOPS | ~65 TFLOPS | ~33% of compute roof | Compute-bound, limited headroom |
| Vector add | 252 GB/s | ~320 GB/s | ~79% of memory roof | Memory-bound, near the hardware limit |
| Tiny ops | 7.8 µs/op | n/a | far below both roofs | Launch-bound |

![Roofline chart](images/roofline.png)

The vector add has terrible FLOPS yet is running about as fast as the hardware allows. It is limited by bandwidth, not by bad code.

---

## Key concepts

### Arithmetic intensity

```
AI = FLOPs performed ÷ bytes moved from memory
```

### The roofline

```
Attainable FLOPS = min( Peak FLOPS , AI × Peak Memory Bandwidth )
```

- **Sloped line:** memory roof (low-intensity kernels)
- **Flat line:** compute roof (high-intensity kernels)
- **Ridge point:** where they meet, `Peak FLOPS ÷ Peak bandwidth`

The roofline only describes kernels that are **running**. If the GPU sits idle between kernels, the dots look fine and the run is still slow. That is why the workflow starts with a timeline.

---

## The profiling workflow

0. **Check the expectation.** Use the right peak for the precision and confirm tensor cores can apply.
1. **Whole-program view** (Nsight Systems, PyTorch Profiler trace). Is the GPU busy? Many tiny kernels with gaps means **launch-bound**, so stop here.
2. **Rank the kernels** by GPU time (PyTorch Profiler).
3. **Roofline per kernel** (Nsight Compute). Compute arithmetic intensity and compare achieved FLOPS and bandwidth with both roofs.
4. **Apply the fix** for the class you found (below).
5. **Re-measure end to end.** Fixing one bottleneck often exposes the next.

### What to do next in each case

| Class | How to recognize it | Next moves |
|---|---|---|
| **Memory-bound** | Left of the ridge, near the bandwidth roof | Fuse ops (`torch.compile`), lower precision or quantize, raise batch size, check coalescing and cache hit rate |
| **Compute-bound** | Right of the ridge | Near roof: reduce FLOPs. Below roof: use tensor cores, BF16/FP16/FP8, tile-friendly shapes, check occupancy, stalls and clock throttling |
| **Below both roofs** | Low on both axes | Check occupancy, warp stall reasons, small-problem inefficiency |
| **Launch-bound** | Gaps between tiny kernels in the timeline | CUDA Graphs, `torch.compile(mode="reduce-overhead")`, fuse small ops, batch more work, remove `.item()` / `.cpu()` syncs |

---

## Running the PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    a @ b
    x + y
    many_small()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

If `device_time_total` errors on your version, use `cuda_time_total`.

### Reading the table

- **Self** columns count only that row. **Total** columns include child calls, so parents double-count.
- The real GPU time is **Self CUDA time total** at the bottom.
- Ignore the `Activity Buffer Request` row. It is the profiler's own bookkeeping.
- In the tiny-op rows, the CPU spends roughly 3x longer issuing each op than the GPU spends running it. That is the launch-bound signature.

### Viewing the timeline

1. Download `trace.json` from `/kaggle/working`.
2. Open [ui.perfetto.dev](https://ui.perfetto.dev) and drag the file in.
3. Zoom into the 1000 small kernels and look for **empty gaps**. The matmul and big add appear as solid bars.

---

## Repository structure

```
gpu-profiling-roofline-handson/
├── README.md
├── notebooks/
│   └── roofline_experiment.ipynb
├── src/
│   └── benchmark.py
├── images/
│   └── roofline.png
└── docs/
    └── blog-post.md
```

---

## Glossary

| Term | Meaning |
|---|---|
| **FLOPs / FLOPS** | Operation count / operations per second |
| **Kernel** | A function that runs on the GPU across many threads |
| **Arithmetic intensity** | FLOPs per byte moved |
| **Ridge point** | Intensity where the memory and compute roofs meet |
| **VRAM / global memory** | The GPU's on-board memory (HBM or GDDR) |
| **Tensor cores** | Specialized matrix-math units |
| **Kernel fusion** | Merging ops to avoid round trips to memory |
| **CUDA Graphs** | Record many launches once and replay them cheaply |

---

## Further reading

- [PyTorch Profiler documentation](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Roofline: An Insightful Visual Performance Model (Williams, Waterman, Patterson)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)

---

## Contributing

Issues and pull requests are welcome. If you run the benchmark on another GPU (A100, H100, RTX series), open a PR adding your results to a `results/` folder.

## License

MIT. See `LICENSE`.
