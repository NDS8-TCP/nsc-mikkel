from pathlib import Path
from compile_miniproject import compile_to_ipynb, compile_to_pdf


def main():
    # Main PDF
    base_dir = Path(".")
    notebooks_mp1: list[Path] = [
        base_dir / "mp2_header.ipynb",
        base_dir / "l04_milestones.ipynb",
        base_dir / "l05_milestones.ipynb",
        base_dir / "l06_milestones.ipynb",
        base_dir / "mp2_section2.ipynb",
        base_dir / "l07_milestones.ipynb",
        base_dir / "mp2_footer.ipynb"
    ]
    mp2 = "mp2"

    compile_to_ipynb(base_dir, notebooks_mp1, mp2)
    compile_to_pdf(base_dir, mp2)

    # Lecture 7 strato exercise
    mp2_strato = "mp2_strato"
    l07_strato = [base_dir / "l07_strato_dask_setup.ipynb"]
    
    compile_to_ipynb(base_dir, l07_strato, mp2_strato)
    compile_to_pdf(base_dir, mp2_strato)

if __name__ == "__main__":
    main()