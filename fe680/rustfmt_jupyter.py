import json
import subprocess
import sys
from pathlib import Path

def format_rust_code(code: str) -> str:
    """Runs rustfmt on the given Rust code, wrapping it in a dummy function."""
    wrapped_code = f"fn __dummy__() {{\n{code}\n}}"
    try:
        result = subprocess.run(["rustfmt"], input=wrapped_code, text=True, capture_output=True, check=True)
        formatted_code = result.stdout

        # Remove the dummy function wrapper and adjust indentation
        formatted_lines = formatted_code.splitlines()[1:-1]  # Remove first and last line
        unindented_code = "\n".join(line[4:] if line.startswith("    ") else line for line in formatted_lines)

        # Remove excessive blank lines
        cleaned_code = "\n".join(line for i, line in enumerate(unindented_code.splitlines()) 
                                 if line.strip() or (i > 0 and unindented_code.splitlines()[i - 1].strip()))
        return cleaned_code
    except subprocess.CalledProcessError as e:
        print(f"Error formatting Rust code: {e.stderr}", file=sys.stderr)
        return code  # Return original code on failure

def process_notebook(notebook_path: Path):
    """Processes a Jupyter notebook, formatting Rust code blocks."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    modified = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code" and not (("df!" in "\n".join(cell["source"])) or (":dep" in "\n".join(cell["source"]))):
            formatted_code = format_rust_code("\n".join(cell["source"]))
            if formatted_code.strip() != "\n".join(cell["source"]).strip():
                cell["source"] = formatted_code.splitlines(keepends=True)
                modified = True    
    if modified:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)
        print(f"Formatted Rust code blocks in {notebook_path}")
    else:
        print("No changes made.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rustfmt_jupyter.py <notebook.ipynb>", file=sys.stderr)
        sys.exit(1)
    
    notebook_file = Path(sys.argv[1])
    if not notebook_file.exists():
        print(f"File not found: {notebook_file}", file=sys.stderr)
        sys.exit(1)
    
    process_notebook(notebook_file)
