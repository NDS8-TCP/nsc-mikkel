from pathlib import Path
from compile_miniproject import compile_to_ipynb, compile_to_pdf


def main():
    # TODO: include the rest when done
    base_dir = Path(".")
    notebooks_mp1: list[Path] = [
        # base_dir / "mp2_header.ipynb",
        base_dir / "l04_exercises.ipynb",
        base_dir / "l05_exercises.ipynb",
        #base_dir / "l06_exercises.ipynb",
        #base_dir / "mp2_footer.ipynb"
    ]
    mp2 = "mp2"

    compile_to_ipynb(base_dir, notebooks_mp1, mp2)
    compile_to_pdf(base_dir, mp2)

if __name__ == "__main__":
    main()