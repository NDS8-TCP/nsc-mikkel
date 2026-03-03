from pathlib import Path
import nbmerge
import nbconvert
import nbformat


def compile_to_ipynb(base_dir: Path, notebooks: list[Path], mp_name: str):
    # compile notebook in the given order
    mp_notebook = nbmerge.merge_notebooks(base_dir, notebooks)

    # write to disk
    mp_file: Path = base_dir / f"{mp_name}.ipynb"
    with mp_file.open("w") as fp:
        nbmerge.write_notebook(mp_notebook, fp)


def compile_to_pdf(base_dir: Path, mp_name: str):
    # read in the already compiled notebook
    mp_file: Path = base_dir / f"{mp_name}.ipynb"
    with mp_file.open("r") as fp:
        mp_notebook = nbformat.read(fp, as_version=4)

    # covert to pdf
    pdf_exporter = nbconvert.PDFExporter()
    pdf_data, _ = pdf_exporter.from_notebook_node(mp_notebook)

    # write out the pdf to disk
    mp_file_pdf: Path = base_dir / f"{mp_name}.pdf"
    with mp_file_pdf.open("wb") as fp:
        fp.write(pdf_data)
