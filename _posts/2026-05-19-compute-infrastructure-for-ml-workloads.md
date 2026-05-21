---
title: Compute Infrastructure for ML Workloads
description: The software stack from kernel to container — drivers, compilers, runtimes, libraries, allocators, frameworks, serving, and orchestration
date: 2026-05-19
categories:
  - Machine Learning
  - Systems
  - Infrastructure
tags:
  - gpu-computing
  - cuda
  - linux
  - distributed-systems
  - performance-optimization
  - kv-cache
pin: true
math: true
mermaid: true
image:
  path: /assets/img/panels/panel17@4x.webp
---

## Overview

Machine learning systems feel deceptively simple from the outside.

You write a few lines of PyTorch, launch a training job, and somewhere beneath the API an enormous amount of machinery moves into place: kernels launch, memory pages move, DMA engines push tensors across PCIe links, schedulers decide who gets compute time, distributed runtimes coordinate machines that may be continents apart.

Most of the time, you never see any of it.

The abstraction is designed that way. Modern ML tooling hides decades of systems engineering behind a remarkably small interface. But when performance collapses, VRAM runs out, GPUs idle unexpectedly, or scaling introduces strange bottlenecks, the abstraction starts to crack. The problem is no longer “inside PyTorch.” It lives somewhere deeper in the stack.

And that stack is older than machine learning itself.

The foundations come from operating systems, computer architecture, networking, and high-performance computing: Unix process models, virtual memory, PCIe, NUMA, compiler toolchains, distributed communication primitives. ML infrastructure did not replace these ideas — it inherited them, layered new abstractions on top, and repurposed them for a new kind of workload.

Understanding modern ML systems therefore means understanding where those layers meet, where they leak, and which one is actually responsible when something goes wrong.

This post walks through the stack from the bottom up: the layers underneath modern ML workloads, what each one owns, and how they interact in practice.

```
┌──────────────────────────────────────────────────┐
│ 9. Orchestration    Docker, Compose, K8s, Ray    │
│ 8. Serving          vLLM, TIS, TGI, Ray Serve    │
│ 7. Framework        PyTorch, JAX, MLX            │
│ 6. Allocator        caching allocator, cgroups   │
│ 5. Math libraries   cuBLAS, cuDNN, NCCL          │
│ 4. Runtime          libcudart, libcuda           │
│ 3. Compiler         nvcc, NVRTC, Triton DSL      │
│ 2. Driver           nvidia.ko, amdgpu.ko         │
│ 1. Kernel           syscalls, /proc, /sys        │
└──────────────────────────────────────────────────┘
              Hardware (CPU, RAM, GPU, NVMe)
```

---

## 1. The kernel layer

A Linux process never talks to hardware directly. It issues system calls into the kernel, and the kernel  through a driver — talks to the device. The reason is protection: every process has its own virtual address space, and a misbehaving program cannot scribble on a PCIe device or another process's memory because it has no way to even name them.

The kernel exposes its state through three surfaces:

-  `/proc`  — per-process and kernel info as virtual files (`/proc/cpuinfo`, `/proc/meminfo`, `/proc/PID/status`).
-  `/sys`  — device tree, driver parameters, cgroup controllers.
-  The ring buffer  — chronological log of kernel events, read by `dmesg`.

These are not real files. Reading them triggers a kernel callback that produces the answer on the fly, which is why `cat /proc/cpuinfo` is essentially free.

The ring buffer is where you go when something at the hardware layer is wrong:

```bash
dmesg -T --level=err,warn               # errors with timestamps
dmesg | grep -i -E 'nvidia|nvrm|amdgpu' # driver-related entries
dmesg | grep -i 'xid'                   # NVIDIA hardware faults
```

An Xid is NVIDIA's hardware error code. Xid 13 is grid sync timeout. Xid 31 is a memory page fault (often a software bug, sometimes failing memory). Xid 79 is a GPU falling off the bus — power or thermal. The codes are stable across driver versions; NVIDIA publishes the table.

The same surface gives you CPU, RAM and disk visibility through dedicated tools that are just structured readers over `/proc` and `/sys`:

```bash
lscpu                            # sockets, cores, threads, NUMA, flags
free -h                          # mem total/used/available/cached
vmstat 1                         # paging activity per second
lsblk -o NAME,SIZE,ROTA,MODEL,TRAN  # block devices
iostat -xz 1                     # per-device IOPS, latency
numactl --hardware               # NUMA distances
lstopo --no-io                   # full topology, ASCII
```

Two `lscpu` outputs matter most for ML. The `flags` line tells you whether AVX-512, AVX10 or AMX (Intel matrix extensions) are available — PyTorch's CPU kernels dispatch on these. And the NUMA section tells you how many memory nodes the box has. On dual-socket machines, every access from socket 0 to a DIMM on socket 1 crosses the UPI / Infinity Fabric link — 30 to 50 percent slower than local. If data-loader threads sit on one socket and the GPU is on the PCIe root of the other, you pay this on every batch:

```bash
numactl --cpunodebind=0 --membind=0 python train.py
```

![NUMA topology](/assets/img/graphics/post_17/gray/numa-topology.png){: .dark }
![NUMA topology](/assets/img/graphics/post_17/gray/numa-topology.webp){: .light }
_Figure 1.0: A misplaced process reaching across UPI to remote memory_


---

## 2. The driver layer

A driver is a kernel module — code loaded into the kernel at runtime — that knows how to talk to a class of device. `nvidia.ko` plus `nvidia_uvm.ko`, `nvidia_drm.ko`, `nvidia_modeset.ko` for NVIDIA. `amdgpu.ko` for AMD. The driver registers with the kernel, binds to devices the kernel enumerates on PCIe, and exposes them as device nodes under `/dev`.

```bash
lsmod | grep nvidia              # loaded modules
modinfo nvidia | head            # version, params, signature
ls /dev/nvidia*                  # nodes userspace opens
cat /proc/driver/nvidia/version  # driver self-report
```

The PCIe bus is a tree, with the CPU root complex at the top. `lspci` walks it:

```bash
lspci -tv                        # tree view
lspci -vvv -s 01:00.0            # detail for one device
nvidia-smi topo -m               # GPU-GPU topology with NVLink
```

In the per-device output, two lines matter: `LnkCap` (what the device can do) and `LnkSta` (what was negotiated). A GPU capable of x16 Gen5 sitting in a slot the BIOS gave x8 Gen4 lanes is an 8× regression that no software can fix.

 Driver vs toolkit, the most common confusion.  A newer driver supports older CUDA, never the reverse  this is forward compatibility. `nvidia-smi` shows the *driver* CUDA version (the maximum it can run). `nvcc --version` shows the *toolkit* CUDA version (what compiles). They are allowed to disagree.

The driver controls one knob people forget: persistence mode. Without it, the driver unloads when the last process exits and reloads on the next `cuda` call — a 1–3 second delay per cold start. For serving, leave it on:

```bash
nvidia-smi -pm 1
```

Same surface gives you clocks, power caps, and MIG slicing (Multi-Instance GPU — partitions one A100/H100 into up to 7 isolated slices, each with its own SMs and memory channels):

```bash
nvidia-smi -ac 1215,1410         # pin memory/SM clocks
nvidia-smi -pl 350               # power limit in watts
nvidia-smi mig -cgi 19,19,19 -C  # create 3 MIG slices
```

For AMD the parallel commands are `rocm-smi`, `rocminfo`, `/sys/class/kfd/`. For Apple Silicon there is no separate driver layer in the same sense — the GPU shares unified memory and is accessed through Metal, which sits closer to a framework than a driver.

A Windows remark: same concepts, different surfaces. Device Manager and `pnputil` replace `lspci`. The Event Viewer replaces `dmesg`. `nvidia-smi` works identically. WSL2 gives a full Linux userspace with GPU passthrough, which is what most people end up using. The rest of this post is Linux because that is where production ML lives.

---

## 3. The compiler layer

Three compilers matter for ML on NVIDIA: `nvcc` (ahead-of-time CUDA C++), NVRTC (runtime CUDA compilation), and Triton DSL (Python-embedded kernel language). They produce the same end artifact — GPU machine code — through different paths.

### 3.1 nvcc and the host compiler

A `.cu` file contains both host code (CPU) and device code (`__global__` / `__device__` functions, GPU). `nvcc` is not really a compiler; it is a driver that splits the input and dispatches each half:

```
foo.cu ──┬─→ host C++   ──→ gcc / clang / msvc ──→ host .o ──┐
         │                                                    ├──→ executable
         └─→ device C++ ──→ cicc ──→ PTX ──→ ptxas ──→ SASS  ┘
```

-  PTX  (Parallel Thread eXecution) is NVIDIA's virtual ISA — architecture-independent intermediate, similar in spirit to LLVM IR or JVM bytecode.
-  SASS  (Streaming Assembler) is real GPU machine code, specific to a compute capability.
-  fatbin  wraps multiple SASS targets plus a PTX fallback in one artifact.

Every NVIDIA GPU has a compute capability number (`sm_XX`):

| GPU            | Capability | `nvcc` flag |
| -------------- | ---------- | ----------- |
| V100           | 7.0        | `sm_70`     |
| T4 / RTX 20xx  | 7.5        | `sm_75`     |
| A100           | 8.0        | `sm_80`     |
| RTX 30xx       | 8.6        | `sm_86`     |
| RTX 4090 / L40 | 8.9        | `sm_89`     |
| H100 / H200    | 9.0        | `sm_90`     |
| B100 / B200    | 10.0       | `sm_100`    |
| RTX 50xx       | 12.0       | `sm_120`    |

A production build embeds SASS for every target you care about plus a PTX fallback:

```bash
nvcc -O3 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \  # PTX fallback
  kernel.cu -o app
```

The trailing `compute_90,code=compute_90` embeds PTX. If the GPU at runtime is newer than every embedded SASS, the driver JIT-compiles the PTX to native (a one-time 1–5 s pause, cached afterward). This is how a PyTorch wheel built against CUDA 11.8 still runs on Blackwell.

 Host compiler coupling.  `nvcc` hands host code to `gcc` / `clang` / `MSVC`, and that compiler's version must be on CUDA's supported list. Mismatches produce errors that look like C++ standard-library bugs but are actually toolchain mismatches:

| CUDA | gcc supported | clang supported |
| ---- | ------------- | --------------- |
| 11.8 | 6.x – 11.x    | up to 14        |
| 12.4 | 6.x – 13.x    | up to 17        |
| 13.2 | 8.x – 14.x    | up to 19        |

```bash
nvcc -ccbin /usr/bin/g++-12 kernel.cu   # pin host compiler explicitly
nvcc -Xcompiler "-fopenmp,-fPIC,-march=native" \
     -Xptxas "-v,-O3" \                 # -v prints reg/shmem usage
     -std=c++17 kernel.cu -o app
```

`-Xptxas -v` is the most useful flag during kernel tuning — it prints per-kernel register count and shared-memory footprint, which directly determines occupancy (how many warps can sit resident on an SM).

 The nvcc family: 

- `ptxas` — assembles PTX to SASS, tuned via `-Xptxas`.
- `cuobjdump`, `nvdisasm` — inspect fatbins, dump SASS.
- `nvprune` — strip unused architectures to shrink wheel size.
- `hipcc` — ROCm's analog, wrapping `clang` with the AMDGPU backend.

![nvcc pipeline](/assets/img/graphics/post_17/gray/nvcc-pipeline.png){: .dark }
![nvcc pipeline](/assets/img/graphics/post_17/gray/nvcc-pipeline.png){: .light }
_Figure 2.0: The host/device split and PTX-to-SASS pipeline_


### 3.2 NVRTC — runtime compilation

NVRTC compiles CUDA C++ strings to PTX in-process, no `nvcc` subprocess. It is what `torch.compile` uses for inductor-generated kernels, what CuPy uses for user kernels, and what plugin systems use to inject custom ops at load time. The cost is one JIT compile per kernel per process; once cached (via PyTorch's inductor cache at `~/.cache/torch/inductor` or via the CUDA JIT cache at `~/.nv/ComputeCache`), warm starts are free.

### 3.3 Triton DSL — kernels in Python

Triton DSL (originally OpenAI Triton, distinct from NVIDIA's Inference Server which we will get to in §8) lets you write GPU kernels in Python with explicit block-level semantics. The compiler — clang-based, MLIR-based since 2023 — lowers Triton IR through PTX to SASS:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

You think in blocks (a tile of work) and Triton handles warp-level scheduling, shared-memory allocation, and tensor-core usage. This is what Flash Attention's reference implementation, Mamba, most of vLLM's custom kernels, and the bulk of `torch.compile`'s generated code are written in. It is the right abstraction when CUDA C++ is too low-level and PyTorch ops are too coarse.

---

## 4. The runtime layer

Two libraries on the userspace side. `libcuda.so` is the  driver API  — low-level, stable across CUDA versions, what the driver itself ships. `libcudart.so` is the  runtime API  — higher-level wrappers, version-tied to the toolkit, what your CUDA C++ code links against by default.

When PyTorch loads, it dlopens `libcuda.so`, queries the driver, allocates a CUDA *context* (per-device, per-process state — page tables, command queues, default streams), and is then ready to launch kernels. Context creation is what costs 1–3 seconds on a cold GPU; persistence mode keeps the driver loaded so subsequent contexts are faster.

The runtime exposes four concepts that ML code touches:

-  Streams  — independent command queues. Kernels in different streams can execute concurrently if the hardware allows it. PyTorch's default stream is what you usually use; `torch.cuda.Stream()` creates more.
-  Events  — synchronization points within or across streams. The basis for `torch.cuda.Event(enable_timing=True)`.
-  Graphs  — captured sequences of kernel launches replayed as a single submission. Eliminates per-launch CPU overhead, critical for decode-heavy inference where each token launches dozens of small kernels.
-  Memory  — `cudaMalloc`, `cudaMallocAsync`, `cudaMallocManaged` (unified), `cudaHostAlloc` (page-locked). The framework's caching allocator (§6) sits on top of these.

```python
import torch

# CUDA Graph capture for a decode step
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(static_input)
g.replay()  # near-zero CPU overhead
```

CUDA Graphs matter more than they sound. In LLM decode, each token launches roughly 50–200 kernels and each launch has ~5 μs of CPU overhead. At 50 tok/s, that is up to 50 ms/sec of pure CPU launch overhead — close to 5% of wall time wasted. Capturing the step into a graph removes it. vLLM and TensorRT-LLM both rely on this.

---

## 5. The math library layer

The runtime gives you the ability to launch kernels. The math libraries give you the kernels worth launching. They are written by the hardware vendor, hand-tuned per architecture, and represent the practical performance ceiling for standard operations.

| Library            | Operations                                            | Used by                            |
| ------------------ | ----------------------------------------------------- | ---------------------------------- |
| cuBLAS / cuBLASLt  | GEMM, batched GEMM, mixed precision                   | every framework's linear/matmul    |
| cuDNN              | Convolutions, RNN, attention primitives               | CNNs, some transformers            |
| CUTLASS            | Templated CUDA C++ GEMM, used to build custom kernels | Flash Attention, custom ops        |
| cuFFT              | FFTs                                                  | signal/audio, diffusion timestep ops |
| NCCL               | All-reduce, all-gather, broadcast across GPUs         | DDP, FSDP, TP, EP                  |
| cuSPARSE           | Sparse matmul                                         | sparse models, some MoE routing    |
| TensorRT           | Full inference graph optimizer, fused kernels         | production inference               |

A few facts that drive optimization decisions:

-  Tensor Cores  are matmul units inside each SM with specific dtype support per generation. A100 supports TF32/BF16/FP16/INT8. H100 adds FP8 (E4M3 and E5M2). B100 adds FP4. cuBLAS will only use Tensor Cores if your matrix dimensions are multiples of 8 (FP16) or 16 (INT8) — non-aligned shapes silently fall back to slower CUDA cores.
-  cuDNN heuristics  pick from many algorithm variants per convolution shape. The choice is cached but the first call to a new shape is slow. This is why dynamic shapes hurt — every new shape re-triggers heuristics.
-  NCCL  picks topology-aware reduction trees based on `nvidia-smi topo -m`. Set `NCCL_DEBUG=INFO` to see what it picked; set `NCCL_P2P_DISABLE=0` to force NVLink P2P; set `NCCL_IB_HCA` to bind to a specific InfiniBand card.

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL torchrun --nproc-per-node=8 train.py
```

A common failure mode: NCCL silently using TCP/IP between two GPUs that are NVLink-connected, because of a kernel module misconfiguration or a missing IPC permission. Throughput drops 10×, training "works", nobody notices for weeks. The fix is reading `NCCL_DEBUG` output.

---

## 6. The allocator layer

The CUDA runtime exposes `cudaMalloc`, but no production framework calls it directly. `cudaMalloc` is slow (synchronous with the device, takes hundreds of microseconds), and naive use of it fragments VRAM badly. PyTorch wraps it in a  caching allocator  that turns allocation into a near-free operation.

### 6.1 The PyTorch caching allocator

The allocator maintains a pool of GPU memory blocks per stream. On `tensor.cuda()` or any op that produces a CUDA tensor:

1. Round the request up to a bucket size (powers of two, roughly).
2. Look for a free block in that bucket.
3. If found, return it. If not, ask `cudaMalloc` for a larger segment and split it.
4. On `del tensor`, the block returns to the pool — it is *not* returned to the driver.

This is why `nvidia-smi` reports more memory used than `torch.cuda.memory_allocated()`. Three numbers describe the state:

```python
torch.cuda.memory_allocated()      # actively used by live tensors
torch.cuda.memory_reserved()       # held by PyTorch's pool
torch.cuda.max_memory_allocated()  # high-water mark
print(torch.cuda.memory_summary()) # full breakdown by bucket
```

### 6.2 Fragmentation — when allocation fails despite free memory

The classic OOM that confuses new users: `nvidia-smi` shows 20 GB free, PyTorch reports `memory_allocated()` of 30 GB on an 80 GB GPU, and a 5 GB allocation fails with `CUDA out of memory. Tried to allocate 5.00 GiB`. The free memory exists but is fragmented across many small blocks — no single contiguous 5 GB region.

This happens whenever allocation sizes vary across iterations: variable sequence lengths, dynamic batches, attention with KV cache. Two mitigations:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`expandable_segments` (added in PyTorch 2.1) lets the allocator grow segments instead of allocating fixed-size ones, dramatically reducing fragmentation for variable workloads. It is what vLLM uses by default.

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

`max_split_size_mb` caps how large a block can be split into smaller pieces, preventing big segments from being chopped up irreversibly.

If you suspect fragmentation, dump the snapshot:

```python
torch.cuda.memory._record_memory_history()
# ... run workload ...
torch.cuda.memory._dump_snapshot("snap.pickle")
# load in https://pytorch.org/memory_viz to visualize
```

### 6.3 Page-locked (pinned) memory

Host-to-device transfers normally go: pageable host memory → driver staging buffer (page-locked) → device. The intermediate copy is wasted bandwidth. If host memory is *pinned* (allocated with `cudaHostAlloc`, kept resident by the kernel, never swapped out), the GPU's DMA engine can copy directly from it to VRAM.

```python
loader = DataLoader(dataset, pin_memory=True, num_workers=8)
# inside the loop:
batch = batch.to(device, non_blocking=True)  # async copy from pinned host
```

The trade-off: pinned memory cannot be paged out, so it counts against physical RAM permanently. Over-pinning starves the rest of the system. Pin the data loader's output, not arbitrary buffers.

### 6.4 cgroups — the kernel's resource governor

Below the framework allocator, the kernel itself enforces resource limits via cgroups v2. Containers, systemd services and `nice`-d shells all run inside one. The controllers relevant to ML:

| Controller | Limits                                          |
| ---------- | ----------------------------------------------- |
| `memory`   | RAM, swap, OOM behavior                         |
| `cpu`      | CPU shares, quota                               |
| `cpuset`   | Which cores and NUMA nodes a process may use    |
| `io`       | Block-device bandwidth and IOPS                 |
| `pids`     | Max processes (matters for DataLoader workers)  |

```bash
# inspect a container's cgroup
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/cpuset.cpus.effective

# pin a training run to socket 0 cores and memory node 0
systemd-run --scope -p AllowedCPUs=0-31 -p AllowedMemoryNodes=0 \
  python train.py
```

GPU memory is *not* cgroup-controlled in mainline kernels. The driver enforces per-process limits internally; containers see GPUs through device passthrough, not cgroup quotas. This is why a process can OOM the GPU without tripping any container memory limit.

![Allocator stack](/assets/img/graphics/post_17/gray/allocator-stack2.png){: .dark }
![Allocator stack](/assets/img/graphics/post_17/gray/allocator-stack.webp){: .light }
_Figure 3.0: The five layers between `tensor.cuda()` and physical VRAM_


---

## 7. The framework layer

PyTorch, JAX and MLX are the three frameworks worth knowing. Each makes different trade-offs in the same design space: how lazy is execution, how is the graph captured, how is autograd implemented, how does it compile.

| Framework   | Execution             | Compilation                                | Distributed                    |
| ----------- | --------------------- | ------------------------------------------ | ------------------------------ |
|  PyTorch  | Eager by default      | `torch.compile` (TorchInductor → Triton DSL) | DDP, FSDP, TP via DTensor      |
|  JAX      | Lazy / traced         | XLA (HLO → LLVM/PTX)                       | pjit, sharding annotations     |
|  MLX      | Lazy, unified memory  | Built-in, no separate compile step         | None for multi-machine         |

The PyTorch story for ML compute optimization is `torch.compile`:

```python
import torch
model = torch.compile(model, mode="reduce-overhead")  # captures CUDA Graphs
out = model(x)  # first call slow, rest fast
```

`mode="reduce-overhead"` captures the model into a CUDA Graph, which is what makes it useful for decode loops. `mode="max-autotune"` triggers more aggressive Inductor autotuning, including searching over Triton kernel configs — slow first run, faster steady state.

JAX's equivalent is `jax.jit`. The compilation target is XLA, which lowers HLO to LLVM IR and then to PTX. The trace-and-compile model is stricter than PyTorch's: shapes must be static, control flow must be expressed via `jax.lax.cond` / `scan` / `while_loop`. The reward is that XLA produces excellent fused kernels for matmul-heavy workloads — most of Google's TPU code lives here.

MLX skips compilation in the explicit sense. Operations build a lazy graph, evaluation (`mx.eval`) executes it. Unified memory removes host-device copies entirely. On an M3 Ultra with 128 GB, a 4-bit-quantized 70B model fits and runs at usable rates, no offloading needed.

### 7.1 Counting work: FLOPs and arithmetic intensity

A FLOP is one floating-point operation. By convention a fused multiply-add (FMA) — one multiply *and* one add in a single Tensor Core instruction — counts as 2 FLOPs, because that is the work that would otherwise be done by two scalar ops. Tensor Cores execute a matrix-multiply-accumulate on a tile (8×4×8 on Ampere WMMA, larger on Hopper WGMMA) per warp per cycle; the FLOP count is the product of the tile's $M \cdot N \cdot K$ times 2.

For a dense transformer with $N$ parameters processing $T$ tokens (Kaplan et al. 2020, *Scaling Laws*):

$$\text{FLOPs}_{\text{forward}} \approx 2NT, \quad \text{FLOPs}_{\text{train step}} \approx 6NT$$

The 2 = one multiply + one add per parameter per token. The 6 = forward + backward (~2×) + optimizer (~1×). This ignores the attention term, which dominates at long context:

$$\text{FLOPs}_{\text{attn}} \approx 4 L H_{kv} d_h S^2 B$$

with $L$ layers, $H_{kv}$ KV heads, $d_h$ head dim, $S$ sequence length, $B$ batch.

 The theoretical ceiling.  Every GPU has a fixed peak FLOPS — SM count × clock × FLOPs-per-cycle for that dtype. It is the wall no kernel can break:

$$\text{FLOPS}_{\text{peak}} = N_{\text{SM}} \cdot f_{\text{clock}} \cdot \text{ops}_{\text{cycle/SM}}$$

| GPU         | FP32  | TF32 (TC) | BF16/FP16 (TC) | FP8 (TC) | FP4 (TC) | HBM BW    |
| ----------- | ----- | --------- | -------------- | -------- | -------- | --------- |
| A100 80 GB  | 19.5  | 156       | 312            | —        | —        | 2.0 TB/s  |
| H100 SXM    | 67    | 495       | 989            | 1979     | —        | 3.35 TB/s |
| H200 SXM    | 67    | 495       | 989            | 1979     | —        | 4.8 TB/s  |
| B100        | ~80   | ~900      | 1800           | 3500     | 7000     | 8 TB/s    |
| B200        | ~90   | ~1125     | 2250           | 4500     | 9000     | 8 TB/s    |

(TFLOPS, dense Tensor Cores; structured-sparsity throughput is ~2×.)

These numbers are aspirational. The H100's 989 BF16 TF/s assumes every SM runs a Tensor Core MMA every cycle with zero memory stalls, zero instruction overhead, and perfect tile alignment. Production training lands at 40–55%; that ratio is MFU.

The closed-form lower bound on a step time is straightforward:

$$T_{\text{step}}^{\text{min}} = \frac{6 N T}{\text{FLOPS}_{\text{peak}} \cdot \text{MFU}}$$

If a 70B model training on 8×H100 with MFU 0.45 takes 4× longer than this formula predicts, the rest is communication, data loading, or stalls — not compute. Walk the stack.

| Tool                            | Type                 | Notes                                             |
| ------------------------------- | -------------------- | ------------------------------------------------- |
| `calflops`                      | Static trace         | pip-installable                                   |
| `fvcore.nn.FlopCountAnalysis`   | Static trace         | Meta, Detectron2                                  |
| `ptflops`                       | Static trace         | older                                             |
| DeepSpeed `flops_profiler`      | Static + dynamic     | training loop                                     |
| Nsight Compute                  | Hardware counters    | `sm__sass_thread_inst_executed_*`                 |
| PyTorch profiler                | Hardware counters    | `with_flops=True`                                 |

Public benchmark sites for comparing models/hardware: `artificialanalysis.ai`, `llm-stats.com`, `mlcommons.org/benchmarks/inference`, the LMSYS leaderboard.

 MFU  (Model FLOPs Utilization) = achieved FLOPS / peak FLOPS. Production training: 40–55% on H100 is excellent. Below 30% means memory- or comm-bound, not compute-bound — fix that before tuning kernels.

The companion metric is  arithmetic intensity  — FLOPs per byte loaded from HBM. The roofline model says the achievable throughput is $\min(\text{peak FLOPS}, \text{intensity} \times \text{HBM bandwidth})$. Below a critical intensity (~290 on H100 for BF16), you are memory-bound; above, compute-bound. Decode is far below; prefill is far above. Knowing which side you are on determines what to optimize.




---

## 8. The serving layer

For training, the framework is enough — `torchrun`, FSDP, that is the toolchain. For serving, there is a separate layer of software whose only job is to wring throughput from a fixed model. Three pieces dominate: **vLLM** (the engine), **NVIDIA Triton Inference Server** (the multi-framework runtime — different thing from Triton DSL), and **Ray Serve** (the routing layer that orchestrates everything).

### 8.1 The inference metric taxonomy

Inference performance is multi-dimensional. A model can have low latency but poor throughput, good P50 but terrible P99, fast decode but expensive prefill. The full picture requires seven categories of metrics, each with different optimization strategies and hardware bottlenecks.

#### 8.1.1 Request-level latency: TTFT, ITL, E2E

Three latencies dominate generative inference. Each has a closed-form lower bound from hardware; the gap to what you measure is your optimization budget.

**TTFT (Time To First Token)** — prefill latency. Compute-bound: all prompt tokens flow through every weight matrix in parallel.

$$\text{TTFT} \approx \frac{2 N \cdot S_{\text{prompt}}}{\text{FLOPS}_{\text{peak}} \cdot \text{MFU}}$$

The factor of 2 comes from forward pass matmuls (QKV projection, attention output, MLP). MFU (Model FLOPS Utilization) captures how efficiently your kernel saturates theoretical peak — typically 0.4–0.6 for inference.

**TPOT / ITL (Inter-Token Latency)** — decode latency. Memory-bound: each token reads the full weight matrix + KV cache from HBM, compute is trivial.

$$\text{ITL} \approx \frac{N \cdot b_w + \text{KV}_{\text{bytes}}}{\text{BW}_{\text{HBM}} \cdot \text{MBU}}$$

MBU (Memory Bandwidth Utilization) is typically 0.6–0.8. Batching helps decode because weights are read once and reused across the batch — ITL drops as $1/B$ until KV reads (which scale with $B$) dominate.

**E2E (End-to-End Latency)** — total time from request arrival to final token.

$$\text{E2E} = T_{\text{queue}} + \text{TTFT} + (N_{\text{gen}} - 1) \cdot \text{ITL} + T_{\text{post}}$$

For interactive applications, E2E < 3–5s is the target for short responses. Longer context or output pushes this higher.

#### 8.1.2 Streaming metrics: TTFB, RTF

For streaming responses, **TTFB (Time To First Byte)** matters more than E2E. TTFB includes network handshake + TTFT + first token serialization. Target: TTFB < 500 ms for responsive feel.

**RTF (Real-Time Factor)** — ratio of processing time to output duration:

$$\text{RTF} = \frac{T_{\text{processing}}}{T_{\text{output duration}}}$$

For text streaming, comfortable reading is ~5 tok/s, so 20 tok/s decode gives RTF ≈ 0.25. For audio generation:

$$\text{RTF}_{\text{audio}} = \frac{T_{\text{text-enc}} + T_{\text{gen}}(N_{\text{aud-tok}}) + T_{\text{vocoder}}}{D_{\text{audio-sec}}}$$

Autoregressive codec models typically achieve RTF 0.2–0.5. Diffusion-based synthesis has RTF 2–10× higher due to iterative denoising.

#### 8.1.3 System throughput: QPS, aggregate tok/s, GPU utilization

**QPS (Queries Per Second)** — how many complete requests finish per second. System-level metric; includes queuing dynamics and batch efficiency.

**Aggregate tok/s** — total tokens generated per second across all concurrent requests:

$$\text{tok/s}_{\text{aggregate}} = \frac{B_{\text{eff}}}{\text{ITL}}$$

where $B_{\text{eff}}$ is effective batch size after KV cache packing. Continuous batching keeps $B_{\text{eff}}$ high by inserting new prefills between decode steps.

**GPU utilization** — percent of time GPU is doing useful work:
- Naive serving: 40–60%
- Optimized continuous batching: 75–85%

Bottlenecks: decode is memory-bound (low SM occupancy), small batches don't saturate, imbalanced request lengths create bubbles.

**MFU vs. MBU** — track MFU (compute utilization) for prefill, MBU (memory bandwidth) for decode. You can't fix decode with more FLOPS; you need higher bandwidth or tensor parallelism to split weight reads.

#### 8.1.4 Tail latency: P50, P95, P99

Averages lie. Track percentiles:

- **P50** — median latency
- **P95** — 95th percentile; 1 in 20 requests is slower
- **P99** — 99th percentile; 1 in 100 requests is slower

P99 is dominated by **stragglers**: long prompts that saturate bandwidth, unlucky scheduling, GC pauses, communication stalls. Mitigation strategies:
- Request hedging (duplicate to replicas)
- Admission control (reject when queue > threshold)
- Prompt length limits

Typical targets for production: P50 < 2s, P95 < 5s, P99 < 10s.

#### 8.1.5 Memory efficiency: KV cache, paging, fragmentation

KV cache is the memory bottleneck for long contexts. Each token stores keys + values across all layers:

$$\text{KV}_{\text{bytes}} = 2 \cdot L \cdot d \cdot S \cdot b_{\text{precision}}$$

Where $L$ = layers, $d$ = hidden dimension, $S$ = sequence length, $b_{\text{precision}}$ = bytes per element (2 for BF16, 1 for INT8).

**PagedAttention** cuts fragmentation by breaking KV cache into fixed-size pages. Traditional contiguous allocation wastes 20–30% on internal fragmentation; paging drops this to <5%.

**Prefix caching** — reuse KV cache for shared prompt prefixes (system instructions, few-shot examples). Hit rate on production systems: 60–80%.

**Key metrics:**
- KV cache memory per request (GB)
- Fragmentation overhead (%)
- Prefix cache hit rate (%)

#### 8.1.6 Advanced technique metrics

**Speculative decoding:**
- **Acceptance rate** — fraction of draft tokens accepted. Good: 60–80%. Below 50%, overhead exceeds gain.
- **Speedup factor**:

$$\text{speedup} = 1 + (r_{\text{accept}} - 1) \cdot \frac{T_{\text{draft}}}{T_{\text{target}}}$$

If draft model is 50× faster and 70% of tokens are accepted: 1.65× overall speedup.

**Flash Attention:**
- **Memory reduction** — drops attention memory from $O(N^2)$ to $O(N)$. Critical for contexts >8K.
- **Speedup** — 2–4× for long contexts (>2K tokens), minimal for short.

**Quantization:**
- **Perplexity degradation** — measure on validation set:
  - INT8 activation + weight: +0.1–0.3 PPL
  - INT4 weight-only: +0.5–1.2 PPL
  - Acceptable threshold: <+2.0 PPL
- **Throughput gain** — INT8: 1.5–2×, INT4: 2.5–3× (memory-bandwidth dominated)

**Chunked prefill:**
- **Chunk size** — split long prompts (512 or 1024 tokens), interleave with decode
- **Bubble overhead** — idle time between chunks. Target: <10%

#### 8.1.7 Multi-modal extensions

**Vision models:**
- **Image encoding time** — ViT forward pass to compress image → vision tokens
- **Vision tokens per image** — typical grid (e.g., 24×24) = 576 tokens per image
- **Prefill cost** — scales with vision token count, not image resolution after encoding

**Audio models:**
- **Audio encoding** — mel-spectrogram + encoder. 1s audio → ~50 tokens typically.
- **Audio decoding** — codec models generate 25–50 tokens/s of audio codes
- **RTF** — autoregressive: 0.3–0.8, diffusion: 2–8



### 8.2 KV cache anatomy

Each attention layer computes:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_h}}\right) V$$

$K$ and $V$ never change once produced for past tokens, so they are cached. Per-token KV size in bytes:

$$\text{kv}_{\text{tok}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_w$$

| Model        | $L$ | $H_{kv}$ | $d_h$ | kv/tok  | @ 4K    | @ 32K |
| ------------ | --- | -------- | ----- | ------- | ------- | ----- |
| Llama-3 8B   | 32  | 8        | 128   | 128 KB  | 512 MB  | 4 GB  |
| Llama-3 70B  | 80  | 8        | 128   | 320 KB  | 1.28 GB | 10 GB |
| Llama-3 405B | 126 | 8        | 128   | 504 KB  | 2 GB    | 16 GB |

At batch 32 with 32K context, Llama-70B needs 320 GB just for KV. The cache, not the weights, is the bottleneck at scale.

### 8.3 PagedAttention — paging applied to KV

OS paging decouples the virtual address space a process sees from physical RAM. Memory is divided into fixed-size pages (typically 4 KB); the MMU keeps a page table mapping virtual pages to physical frames. Benefits: no external fragmentation, sharing across processes, swapping to disk.

```bash
getconf PAGE_SIZE                       # 4096 on x86_64
cat /sys/kernel/mm/transparent_hugepage/enabled
```

PagedAttention (Kwon et al. 2023, vLLM) is the same idea applied to KV cache. Before it, KV was a contiguous tensor `[batch, max_seq_len, ...]` — internal fragmentation (short requests waste their max_seq_len reservation) and external fragmentation (freed slots don't coalesce) meant 60–80% of KV memory was wasted at any moment.

PagedAttention turns the cache into a pool of fixed-size blocks (typically 16 tokens). Each request keeps a  block table  — a small array mapping logical token positions to physical block indices. Exactly an OS page table.

```
Request A logical:   [tok0..15] [tok16..31] [tok32..47]
                          ↓          ↓           ↓
Block table A:           [7]        [2]         [11]

Physical KV pool: [b0][b1][b2*][b3][b4][b5][b6][b7*][b8][b9][b10][b11*]
                          ↑                    ↑                   ↑
                       2nd chunk            1st chunk           3rd chunk
```

Wins: waste drops to under 4%;  prefix sharing  (two requests share a system prompt → share blocks, zero copy); beam search branches share common prefix; cold blocks can swap to host RAM. vLLM reports 2–4× throughput vs naive baseline. The attention kernel is rewritten to gather K/V from this indirected layout — a custom CUDA kernel, since memory is no longer contiguous along the sequence axis.

![PagedAttention](/assets/img/graphics/post_17/gray/paged-attention.png){: .dark }
![PagedAttention](/assets/img/graphics/post_17/gray/paged-attention.webp){: .light }
_Figure 5.0: Contiguous KV with wasted slots vs PagedAttention with a block table_

>  Illustration prompt:  *Side-by-side diagram. Left panel "Contiguous KV cache (naive)": three rows, each a long rectangle pre-allocated to max length but only filled on the left, the empty portion shaded gray and labeled "wasted". Right panel "PagedAttention": same three requests, but their KV is split into small fixed-size blocks (16-token squares) scattered across a shared pool below. Above each request, a "block table" — a small array of arrows pointing to the physical blocks it owns. Show two block tables both pointing to one shared block (prefix sharing), highlighted in a distinct color. Minimal technical illustration.*

### 8.4 Shrinking the cache itself

PagedAttention reclaims waste. Orthogonal architectural choices reduce the cache size to begin with:

| Technique                              | Shrink           | Quality cost | Stage                              |
| -------------------------------------- | ---------------- | ------------ | ---------------------------------- |
|  MHA  (baseline)                     | 1×               | —            | Pretrain                           |
|  MQA  (Shazeer 2019)                 | $H_q$×           | small        | Pretrain — PaLM, Falcon            |
|  GQA  (Ainslie 2023)                 | $H_q / g$×       | ~0           | Pretrain — Llama 2/3, Mistral, Qwen |
|  MLA  (DeepSeek-V2/V3)               | 4–8× vs GQA      | ~0           | Pretrain                           |
|  FP8 KV                              | 2×               | ~0           | Inference (H100+)                  |
|  INT4 KV  (KIVI, KVQuant)            | 4×               | small        | Inference                          |
|  Sliding window  (Mistral 7B)        | $S/W$×           | medium       | Pretrain                           |
|  StreamingLLM  (sinks + window)      | $S/W$×           | small        | Inference                          |
|  PagedAttention                      | reclaims waste   | 0            | Inference                          |

These stack. GQA + FP8 KV = 16× smaller than MHA BF16, no quality loss. Add PagedAttention and the remaining fragmentation is gone.

### 8.5 Throughput techniques

Beyond the cache, three more techniques dominate the throughput story:

 Flash Attention  (Dao et al. 2022) tiles the $QK^T$ computation so $Q$, $K$, $V$ blocks stay in on-chip SRAM and the full $S$ matrix is never materialized in HBM. Online softmax keeps numerics correct. Memory drops $O(N^2) \to O(N)$; wall-clock 2–4× faster. FA2 (2023) improved sequence-dim parallelism. FA3 (2024) added FP8 on Hopper, reaches ~75% MFU on H100. PyTorch's `F.scaled_dot_product_attention` dispatches automatically.

 Continuous (in-flight) batching  (Orca 2022, used by vLLM, TGI, TensorRT-LLM). Static batching makes short requests wait for the longest. Continuous batching changes the batch composition at each decode step — finished requests free their slot, queued requests join mid-batch. 5–10× throughput at typical workloads.

 Chunked prefill  (Sarathi 2023) splits long prefills into ~512-token chunks interleaved with decode, trading a little TTFT for much smoother TPOT under load.

 Speculative decoding.  A small draft model generates $n$ candidate tokens; the large target model scores them in one forward pass; matching tokens are accepted; on first rejection, drafting restarts. Acceptance 60–80% → 2–3× speedup, mathematically equivalent distribution. EAGLE and Medusa do the same without a separate draft model.

![Throughput techniques](/assets/img/graphics/post_17/gray/throughput-techniques.png){: .dark }
![Throughput techniques](/assets/img/graphics/post_17/gray/throughput-techniques.webp){: .light }
_Figure 6.0: Three families of throughput technique_

>  Illustration prompt:  *Mind-map / tree diagram. Center node "GPU inference throughput". Three branches radiating out. Branch 1 "Attention" → leaves: "Flash Attention", "Paged KV", "GQA/MQA/MLA". Branch 2 "Batching" → leaves: "Continuous batching", "Chunked prefill", "Speculative decoding". Branch 3 "Parallelism" → leaves: "Tensor", "Pipeline", "Expert", "Sequence". Minimal flat design, each leaf a small labeled box.*

### 8.6 Multi-GPU parallelism

When the model exceeds one GPU:

| Strategy        | What it splits              | Communication                | Best at                  |
| --------------- | --------------------------- | ---------------------------- | ------------------------ |
|  Tensor (TP)  | Each layer's matrices       | All-reduce after each layer  | Intra-node, NVLink       |
|  Pipeline (PP)  | Layers across GPUs        | Activations between stages   | Cross-node, slower link  |
|  Expert (EP)  | MoE experts                 | All-to-all dispatch          | MoE inference            |
|  Sequence (SP)  | Activations along seq dim | Complements TP               | Long context             |

Rule of thumb for Llama-405B serving: TP within a node (NVLink, 8 GPUs), PP across nodes (InfiniBand). For MoE: TP + EP.

### 8.7 ONNX and graph-level optimization

ONNX is a frozen graph format: ops, tensors, attributes, no Python. Once a model is exported (`torch.onnx.export`, JAX via the StableHLO → ONNX converter, TensorFlow `tf2onnx`), the graph is detached from its source framework and can be rewritten statically.

The optimizer applies whole-graph transforms that eager-mode PyTorch cannot:

-  Constant folding  — fold weights through frozen ops. Batch-norm gets absorbed into the preceding `Conv`; LayerNorm scales merge into the next matmul.
-  Operator fusion  — `Conv + BN + ReLU` → one kernel; `MatMul + Add + GELU` → one fused matmul. The intermediate activations never leave registers.
-  Dead-code elimination  — drop subgraphs whose outputs go unused (common after exporting only the inference path of a training model that had aux losses).
-  Shape specialization  — when shapes are known, swap dynamic-shape kernels for static ones, unlocking pre-tuned code paths.

 The FLOPs angle.  Fusion barely changes the raw FLOP count. What changes is how many bytes those FLOPs cost. `Conv + BN + ReLU` as three kernels writes activation to HBM, reads it back, writes it again — three round-trips. Fused, the activation lives in registers and shared memory; only the final output hits HBM. Same FLOPs, a fraction of the bytes loaded, so  arithmetic intensity rises  and the workload moves up the roofline. A kernel that was memory-bound at intensity 50 may now sit at intensity 200, compute-bound, with realized throughput 2–3× higher even though the FLOP arithmetic is identical.

ONNX Runtime (ORT) executes the optimized graph through  execution providers  — pluggable backends, one per device class:

| EP                  | Hardware                | Notes                                                          |
| ------------------- | ----------------------- | -------------------------------------------------------------- |
| CUDA EP             | NVIDIA                  | cuDNN/cuBLAS, baseline                                         |
| TensorRT EP         | NVIDIA                  | rebuilds the graph as a TensorRT engine, fastest for fixed shapes |
| ROCm EP / MIGraphX  | AMD                     | parallel to TRT                                                |
| OpenVINO EP         | Intel CPU / iGPU / NPU  | best CPU inference, INT8 calibration                           |
| CoreML EP           | Apple                   | Neural Engine + GPU                                            |
| QNN EP              | Qualcomm Hexagon NPU    | mobile/edge                                                    |

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
        "CUDAExecutionProvider",      # fallback
        "CPUExecutionProvider",
    ],
)
out = sess.run(None, {"input": x})
```

Typical wins for vision and small encoder models: ORT + TensorRT EP gives 2–5× over eager PyTorch, mostly from fusion plus INT8/FP16 calibration. For LLMs, vLLM and TensorRT-LLM still beat ORT — PagedAttention, continuous batching, and KV-cache management are not part of the ONNX graph abstraction, so ORT cannot reproduce them.

 Quantization.  The other lever on realized FLOPs is dropping their precision: INT8 instead of FP16, INT4 weights with FP16 activations, FP8 end-to-end on Hopper. The hardware can throughput more ops per cycle at lower precision (an H100 does 2× FP8 vs FP16), and weight reads from HBM shrink by the same ratio, which is the bigger win for decode. The full mechanics — calibration, QAT vs PTQ, GPTQ/AWQ/SmoothQuant, KV quantization — belong in the model-compression post following this one.

### 8.8 Serving runtimes — vLLM, TIS, Ray Serve

 vLLM  is the open-source inference engine that built PagedAttention. Single command, OpenAI-compatible API, continuous batching, prefix caching:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8
```

 NVIDIA Triton Inference Server (TIS)  — the runtime, distinct from Triton DSL. Loads multiple model formats (PyTorch, TensorRT, ONNX, OpenVINO, TF, vLLM-as-backend), batches requests dynamically, supports ensembles (model A → model B in one request), exposes HTTP/gRPC and Prometheus metrics. Where you go when you serve more than just LLMs:

```
name: "llama3_8b"
backend: "vllm"                          # vLLM-as-Triton-backend
max_batch_size: 32
dynamic_batching {
  max_queue_delay_microseconds: 50000    # TTFT vs throughput knob
  preferred_batch_size: [ 8, 16 ]
}
instance_group [ { count: 1, kind: KIND_GPU } ]
```

 Ray Serve  sits above both. It does routing, model composition, autoscaling, and traffic splitting across deployments. Use it when you have many models, A/B tests, or pipelines (embed → rerank → LLM):

```python
from ray import serve

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMReplica:
    def __init__(self):
        from vllm import LLM
        self.engine = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    async def __call__(self, request):
        return self.engine.generate(request.prompt)

serve.run(LLMReplica.bind())
```

For *training* distribution,  Ray Train  wraps PyTorch DDP/FSDP with placement groups:

```python
import ray
from ray.util.placement_group import placement_group

pg = placement_group([{"GPU": 2, "CPU": 16}] * 4, strategy="STRICT_SPREAD")
```

`STRICT_SPREAD` forces one bundle per node — data parallelism. `STRICT_PACK` colocates everything — tensor parallelism over NVLink. Get this wrong and you accidentally tensor-parallel over Ethernet.

For local heterogeneous inference (a Mac, a PC with a GPU, an iPhone all hosting one model),  Exo  splits the model by layer across devices. Throughput is bottlenecked by the slowest link, so it is for batch-1 personal use, not production.

### 8.9 KServe v2 plumbing

KServe is the Kubernetes-native serving CRD. An `InferenceService` resource wraps a model with autoscaling, traffic routing, and a standard request schema. Underneath it sits Knative (scale-to-zero, revisions) and Istio (routing, mTLS); the actual inference is delegated to a backend container — Triton, TorchServe, vLLM, MLServer, or a custom predictor.

The "v2" refers to the  Open Inference Protocol  (originally KFServing v2, now also implemented by Triton Inference Server, MLServer, TorchServe, and OpenVINO Model Server). One REST/gRPC contract, model-agnostic.

```
POST /v2/models/{name}/infer            # inference
GET  /v2/models/{name}                  # metadata
GET  /v2/models/{name}/ready            # per-model readiness
GET  /v2/health/ready                   # server liveness
```

The request body is a typed tensor list — same shape whether the backend is a ResNet on Triton or a Llama on vLLM:

```json
{
  "inputs": [{
    "name": "input_ids",
    "shape": [1, 128],
    "datatype": "INT64",
    "data": [101, 2023, 2003, 1037, 3231, 102]
  }]
}
```

Responses mirror the same envelope under `outputs`. That uniformity is the point: a platform team (Seldon Core, BentoML, ModelMesh, KServe itself) can write one client and one observability pipeline regardless of which framework produced the model.

A minimal `InferenceService` wiring vLLM as the predictor:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-8b
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment   # skip Knative for steady traffic
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 8
    scaleTarget: 80                  # target concurrency per replica
    containers:
      - name: kserve-container
        image: vllm/vllm-openai:latest
        args:
          - --model=meta-llama/Llama-3.1-8B-Instruct
          - --port=8080
        resources:
          limits:
            nvidia.com/gpu: "1"
        readinessProbe:
          httpGet: { path: /v2/health/ready, port: 8080 }
```

Three things this gives you that a raw `vllm serve` container does not:

-  Scale-to-zero via Knative.  No traffic, the pods drain. The first request after idle pays a cold start (model reload, CUDA context, ~10–60 s for an 8B); subsequent requests are warm. Useful for long-tail models you cannot afford to keep resident.
-  Canary and traffic splitting.  Two predictor revisions, 90/10 split, routed by the Istio virtual service KServe generates. Zero-downtime rollouts, automatic rollback on error rate.
-  Transformer and explainer pods.  Pre/post-processing (tokenization, image decode, embedding lookup) runs in a `transformer` sidecar; explainability (SHAP, integrated gradients) runs in an `explainer` sidecar. The predictor stays focused on tensor math, which keeps GPU pods schedulable on GPU nodes and CPU pods on CPU nodes.

The hop count is the cost. A request flows: Istio gateway → Knative activator (only when scaled to zero) → queue-proxy sidecar → predictor container → backend engine. That is up to five hops before the model sees the input tensor. Each hop adds 1–5 ms. For LLM token streaming that is amortized across hundreds of tokens; for short classification calls it can dominate TTFT. Setting `deploymentMode: RawDeployment` skips Knative and drops two hops at the cost of losing scale-to-zero.

When KServe v2 earns its complexity: many models, mixed frameworks, autoscaling matters, you already run Kubernetes. When it does not: one model on one node — a bare vLLM container behind a load balancer is simpler, faster, and easier to reason about.

### 8.10 Estimating VRAM for a model

$$\text{VRAM} \approx \underbrace{N \cdot b_w}_{\text{weights}} + \underbrace{\text{kv}_{\text{tok}} \cdot S \cdot B}_{\text{KV cache}} + \underbrace{\text{activations}}_{\sim \text{small}} + \underbrace{\text{overhead}}_{1\text{-}2 \text{ GB}}$$

For training, multiply weights by 4 (weights + gradients + Adam state). Llama-3 70B BF16 inference, 4K context, batch 1: 140 GB weights + 1.3 GB KV + ~3 GB activations + ~2 GB overhead = ~147 GB. Does not fit on H100 80 GB. Fits on H200 141 GB barely. Fits on 2× H100 with NVLink (160 GB).

---

## 9. The orchestration layer

Compute-intensive workloads are not I/O-intensive ones, and the container settings differ accordingly. Database containers care about volume mounts and network latency; compute containers care about GPU access, shared memory, NUMA pinning, and the fact that the kernel driver lives outside the container.

### 9.1 The NVIDIA container stack

```
GPU hardware
    ↓
NVIDIA kernel driver (host)
    ↓
libnvidia-container
    ↓
nvidia-container-toolkit / CDI
    ↓
Docker / containerd runtime
    ↓
Container (CUDA toolkit, cuDNN, PyTorch)
```

The driver is on the host. The toolkit (libraries linking against the driver) is inside the image. They communicate through `/dev/nvidia*` device nodes that the container runtime injects at startup. CDI (Container Device Interface) is the modern mechanism replacing `--runtime=nvidia`.

```bash
nvidia-smi                          # host
nvidia-container-cli info           # toolkit
docker info | grep -i nvidia
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
docker run --gpus '"device=0,1"' ...                # specific GPUs
docker run --gpus '"capabilities=compute,utility"' ...
```

### 9.2 BuildKit for ML images

ML images are large (5–15 GB) and rebuilt often. BuildKit (default builder since Docker 23.0) makes this bearable: cache mounts (pip/apt caches persist without ending up in the image), multi-stage builds (compile in one stage with full toolchain, copy artifacts to a slim runtime), multi-platform (AMD64 + ARM64 in one command), secret mounts (registry tokens without baking them in).

```dockerfile
# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y python3-pip && \
    pip install --prefix=/install torch==2.4.0 transformers vllm

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /install /usr/local
COPY app/ /app
WORKDIR /app
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "meta-llama/Llama-3.1-8B-Instruct"]
```

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --cache-to type=registry,ref=myregistry/vllm:cache,mode=max \
  --cache-from type=registry,ref=myregistry/vllm:cache \
  -t myregistry/vllm:latest --push .
```

### 9.3 Compose deploy with resources

`docker-compose` `deploy.resources` is the declarative way to pin GPU, CPU and memory. The compute-vs-IO distinction shows up in which fields matter:

```yaml
services:
  inference:
    image: myregistry/vllm:latest
    shm_size: '16gb'              # PyTorch DataLoader, NCCL
    ulimits:
      memlock: -1                 # NCCL/CUDA pinning
      stack: 67108864
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']  # specific GPUs (topology-aware)
              capabilities: [gpu]
          memory: 64G
          cpus: '16'
        limits:
          memory: 128G
    environment:
      - NCCL_DEBUG=INFO
      - CUDA_VISIBLE_DEVICES=0,1
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Three settings that catch people:

-  `shm_size`  — Docker defaults to 64 MB of `/dev/shm`. DataLoader workers and NCCL use shared memory. 64 MB is too small; set 4–16 GB.
-  `ulimits.memlock: -1`  — NCCL and CUDA pin pages. Default ulimit blocks this.
-  `device_ids` vs `count`  — `count: 2` says "any 2 GPUs"; `device_ids: ['0','1']` says "specifically GPU 0 and 1". The latter is what you want when NVLink topology matters.

| Setting                       | Compute-intensive          | I/O-intensive             |
| ----------------------------- | -------------------------- | ------------------------- |
| `deploy.resources.devices`    | GPUs, MIG slices           | Usually nothing           |
| `shm_size`                    | 4–16 GB                    | Default fine              |
| `ulimits.memlock`             | `-1`                       | Default                   |
| `cpus`                        | Pin to NUMA-local cores    | Spread across cores       |
| Volume mounts                 | Models (read-only, large)  | Hot data (write, fsync)   |
| Network                       | Intra-node mostly          | Cross-node primary        |

For a compute job the Docker bridge often matters less than expected — most traffic is intra-node over NVLink. When it does matter (multi-node training, distributed inference), bypass it with `network_mode: host` or a CNI plugin exposing RDMA / InfiniBand.

![NVIDIA container stack](/assets/img/graphics/post_17/gray/nvidia-container-stack.png){: .dark }
![NVIDIA container stack](/assets/img/graphics/post_17/gray/nvidia-container-stack.webp){: .light }
_Figure 7.0: Host driver, runtime, toolkit, container_


---

## 10. Debugging order

When a compute workload is slow or broken, the layers to check, in order, because each layer assumes the ones below it work:

1.  Kernel/driver/device.  `nvidia-smi`, `dmesg | grep -i xid`. If the GPU is missing or throwing Xids, nothing above matters.
2.  PCIe link.  `lspci -vvv | grep -E 'LnkCap|LnkSta'`. Expected x16 Gen5, or did the BIOS downgrade it?
3.  Topology.  `nvidia-smi topo -m`, `lstopo`. Are the right GPUs on NVLink for your TP shards? Are loader threads on the right NUMA node?
4.  Compiler/runtime.  Driver CUDA version ≥ toolkit. Is `torch.compile` actually caching? `~/.cache/torch/inductor`.
5.  Math libraries.  `NCCL_DEBUG=INFO` — are collectives over NVLink or falling back to TCP? cuDNN heuristics caching on stable shapes?
6.  Allocator.  `torch.cuda.memory_summary()`. Fragmenting? Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
7.  Framework.  MFU below 30% on a compute-bound job means kernels are wrong (wrong dtype, wrong layout, no Tensor Core usage).
8.  Serving.  Flash Attention active? Continuous batching on? Prefix caching enabled?
9.  Orchestration.  `shm_size`, `memlock` ulimit, GPU device pinning, network mode for distributed.

Tuning kernels on a GPU that is silently in PCIe x4 mode is wasted effort. Walk bottom-up.

---

## References

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM
- [FlashAttention](https://arxiv.org/abs/2205.14135) · [FA2](https://arxiv.org/abs/2307.08691) · [FA3](https://arxiv.org/abs/2407.08608)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2 — MLA](https://arxiv.org/abs/2405.04434)
- [Orca: A Distributed Serving System](https://www.usenix.org/conference/osdi22/presentation/yu) — continuous batching
- [SARATHI: Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [StreamingLLM — Attention Sinks](https://arxiv.org/abs/2309.17453)
- [Scaling Laws — Kaplan et al.](https://arxiv.org/abs/2001.08361)
- [Triton: an Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) · [NCCL](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch CUDA Caching Allocator](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Docker BuildKit](https://docs.docker.com/build/buildkit/) · [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [Ray docs](https://docs.ray.io/) · [Triton Inference Server](https://github.com/triton-inference-server/server)
