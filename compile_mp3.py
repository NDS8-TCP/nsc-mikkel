from pathlib import Path
from compile_miniproject import compile_to_ipynb, compile_to_pdf


def main():
    base_dir = Path(".")
    notebooks: list[Path] = [
        base_dir / "mp3_header.ipynb",
        base_dir / "l08_milestones.ipynb",
        #base_dir / "l09_milestones.ipynb",
        #base_dir / "l10_milestones.ipynb",
        base_dir / "mp3_footer.ipynb"
    ]
    mp3 = "mp3"

    compile_to_ipynb(base_dir, notebooks, mp3)
    compile_to_pdf(base_dir, mp3)


if __name__ == "__main__":
    main()