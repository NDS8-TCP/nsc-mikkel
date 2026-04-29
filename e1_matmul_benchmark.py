"""
matmul_benchmark.py -- Matrix multiplication on the GPU: kernel progression.

Three kernels show the optimization ladder:
  matmul1  -- naive: each work-item computes one output element (global memory only)
  matmul2  -- row-based: each work-item computes a full row (enables local memory next)
  matmul3  -- tiled: blocks loaded into local memory; TILE_SIZE x TILE_SIZE reuse

Run this script to time all three and compare against numpy.
"""

from pathlib import Path
import statistics
import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

N         = 1024      # matrix dimension (N x N)
TILE_SIZE = 8
RUNS      = 3         # median of this many kernel launches

HERE = Path(__file__).parent


def load_kernels(ctx):
    sources = [
        (HERE / "matmul_kernel1.cl").read_text(),
        (HERE / "matmul_kernel2.cl").read_text(),
        (HERE / "matmul_kernel3.cl").read_text(),
    ]
    return cl.Program(ctx, "\n".join(sources)).build(
        f"-D DIM={N} -D TILE_SIZE={TILE_SIZE}"
    )


def timed(queue, fn, runs=RUNS):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        queue.finish()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev   = ctx.devices[0]
    print(f"Device: {dev.name}")
    print(f"N={N}  TILE_SIZE={TILE_SIZE}  runs={RUNS}\n")

    rng    = np.random.default_rng(42)
    a_host = rng.random((N, N), dtype=np.float32)
    b_host = rng.random((N, N), dtype=np.float32)
    r_host = np.empty((N, N), dtype=np.float32)

    mf    = cl.mem_flags
    a_dev = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=a_host)
    b_dev = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b_host)
    r_dev = cl.Buffer(ctx, mf.WRITE_ONLY, r_host.nbytes)

    prog = load_kernels(ctx)
    k1   = cl.Kernel(prog, "matmul")
    k2   = cl.Kernel(prog, "matmul2")
    k3   = cl.Kernel(prog, "matmul3")

    expected = a_host @ b_host

    results = {}

    # --- Kernel 1: naive ---
    t = timed(queue, lambda: k1(queue, (N, N), None, a_dev, b_dev, r_dev))
    cl.enqueue_copy(queue, r_host, r_dev); queue.finish()
    assert np.allclose(r_host, expected, atol=1e-3), "matmul1 incorrect"
    results["matmul1\n(naive)"] = t
    print(f"matmul1 (naive):        {t*1000:.1f} ms")

    # --- Kernel 2: row-based ---
    t = timed(queue, lambda: k2(queue, (N,), (min(64, dev.max_work_group_size),), a_dev, b_dev, r_dev))
    cl.enqueue_copy(queue, r_host, r_dev); queue.finish()
    assert np.allclose(r_host, expected, atol=1e-3), "matmul2 incorrect"
    results["matmul2\n(row-based)"] = t
    print(f"matmul2 (row-based):    {t*1000:.1f} ms")

    # --- Kernel 3: tiled local memory ---
    t = timed(queue, lambda: k3(queue, (N, N), (TILE_SIZE, TILE_SIZE), a_dev, b_dev, r_dev))
    cl.enqueue_copy(queue, r_host, r_dev); queue.finish()
    assert np.allclose(r_host, expected, atol=1e-3), "matmul3 incorrect"
    results["matmul3\n(tiled)"] = t
    print(f"matmul3 (tiled):        {t*1000:.1f} ms")

    # --- NumPy reference ---
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = a_host @ b_host
        ts.append(time.perf_counter() - t0)
    t_np = statistics.median(ts)
    results["numpy\nmatmul"] = t_np
    print(f"numpy matmul:           {t_np*1000:.1f} ms")

    # --- Plot ---
    labels = list(results.keys())
    times  = [results[k] * 1000 for k in labels]
    colors = ["#888888", "#888888", "#1f4e8c", "#b03030"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, times, color=colors)
    ax.set_ylabel("Median runtime (ms)")
    ax.set_title(f"Matmul {N}×{N} — kernel progression ({dev.name})")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = HERE / "matmul_benchmark.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
