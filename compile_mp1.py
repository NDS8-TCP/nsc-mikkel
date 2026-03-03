from pathlib import Path
from compile_miniproject import compile_to_ipynb, compile_to_pdf


def main():
    base_dir = Path(".")
    notebooks_mp1: list[Path] = [
        base_dir / "mp1_header.ipynb",
        base_dir / "l01_exercises.ipynb",
        base_dir / "l02_exercises.ipynb",
        base_dir / "l03_exercises.ipynb",
        base_dir / "mp1_footer.ipynb"
    ]
    mp1 = "mp1"

    compile_to_ipynb(base_dir, notebooks_mp1, mp1)
    compile_to_pdf(base_dir, mp1)

if __name__ == "__main__":
    main()