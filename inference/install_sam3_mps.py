import subprocess
import sys
import tempfile
import shutil
from pathlib import Path


REPO_URL = "https://github.com/jveitchmichaelis/sam3.git"
BRANCH = "device-agnostic"
ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements-sam3-mps.txt"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def patch_pyproject(repo: Path) -> None:
    package_assets = repo / "sam3" / "assets"
    if not package_assets.exists():
        shutil.copytree(repo / "assets", package_assets)

    pyproject = repo / "pyproject.toml"
    text = pyproject.read_text()
    old = '[tool.setuptools]\npackages = ["sam3", "sam3.model"]\n'
    new = (
        '[tool.setuptools.packages.find]\n'
        'include = ["sam3*"]\n'
        'exclude = ["build*", "scripts*", "examples*"]\n\n'
        '[tool.setuptools.package-data]\n'
        'sam3 = ["assets/*.txt.gz"]\n'
    )
    if old not in text:
        raise RuntimeError("Unexpected SAM3 pyproject layout; packaging patch not applied.")
    pyproject.write_text(text.replace(old, new))


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    with tempfile.TemporaryDirectory(prefix="sam3-mps-") as temp_dir:
        repo = Path(temp_dir) / "sam3"
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(repo)])
        patch_pyproject(repo)
        run([sys.executable, "-m", "pip", "install", str(repo)])


if __name__ == "__main__":
    main()
