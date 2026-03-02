def main() -> None:
    import matplotlib.pyplot as plt
    from memory_profiler import memory_usage
    from mandelbrot import compute_mandelbrot as naive_python
    from mandelbrot import compute_mandelbrot_numpy as numpy_python

    # Measured in MiB
    naive_mem_usage: list[float] = memory_usage(
        (naive_python, [-2, 1, -1.5, 1.5, 1024, 1024, 100]))
    numpy_mem_usage: list[float] = memory_usage(
        (numpy_python, [-2, 1, -1.5, 1.5, 1024, 1024, 100]))

    plt.plot(naive_mem_usage, label="Naïve")
    plt.plot(numpy_mem_usage, label="Numpy")
    plt.title("Memory usage for 1024x1024 mandelbrot")
    plt.ylabel("Memory usage [MiB]")
    plt.xlabel("Time [sample index]")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.ticklabel_format(style="plain") # prevent scientific notation
    plt.savefig("l02_memory_profile.png")


if __name__ == "__main__":
    main()
