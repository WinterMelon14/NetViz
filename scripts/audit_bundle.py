import subprocess
import sys
from pathlib import Path


FORBIDDEN_PATH_PREFIXES = {
    "docs",
    "scripts",
    "tests",
    "frontend/src",
    "_internal/docs",
    "_internal/scripts",
    "_internal/tests",
    "_internal/frontend/src",
}
FORBIDDEN_EXACT_PATHS = {
    "contract.py",
    "test.py",
    "transcriber.py",
    "_internal/contract.py",
    "_internal/test.py",
    "_internal/transcriber.py",
    "frontend/vite.config.ts",
    "frontend/eslint.config.js",
    "frontend/package.json",
    "frontend/package-lock.json",
    "_internal/frontend/vite.config.ts",
    "_internal/frontend/eslint.config.js",
    "_internal/frontend/package.json",
    "_internal/frontend/package-lock.json",
}
APPLICATION_SOURCE_PREFIXES = {
    "desktop",
    "util",
    "_internal/desktop",
    "_internal/util",
}
FORBIDDEN_EXACT_MODULES = {
    "contract",
    "desktop.known_model",
    "desktop.selected_files",
}
FORBIDDEN_PACKAGE_MODULES = (
    "torchvision",
    "transformers",
)


def normalized_path(path: Path) -> str:
    return path.as_posix().lower().strip("/")


def is_at_or_below(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def is_forbidden_release_path(relative: Path) -> bool:
    normalized = normalized_path(relative)
    if normalized in FORBIDDEN_EXACT_PATHS:
        return True
    if any(is_at_or_below(normalized, prefix) for prefix in FORBIDDEN_PATH_PREFIXES):
        return True
    return relative.suffix.lower() == ".py" and any(
        is_at_or_below(normalized, prefix) for prefix in APPLICATION_SOURCE_PREFIXES
    )


def forbidden_archive_modules(archive_output: str) -> list[str]:
    modules = {line.strip() for line in archive_output.splitlines() if line.strip()}
    violations = set(modules & FORBIDDEN_EXACT_MODULES)
    for module in modules:
        if any(module == package or module.startswith(package + ".") for package in FORBIDDEN_PACKAGE_MODULES):
            violations.add(module)
    return sorted(violations)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: audit_bundle.py DIST/NetViz")
    bundle = Path(sys.argv[1]).resolve()
    executable = bundle / "NetViz.exe"
    index = bundle / "_internal" / "frontend" / "dist" / "index.html"
    if not executable.is_file() or not index.is_file():
        raise SystemExit("Bundle is missing NetViz.exe or the bundled frontend entrypoint.")

    violations = []
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle)
        if is_forbidden_release_path(relative):
            violations.append(str(relative))
    if violations:
        raise SystemExit("Forbidden release files found:\n" + "\n".join(sorted(set(violations))))

    archive = subprocess.run(
        [sys.executable, "-m", "PyInstaller.utils.cliutils.archive_viewer", "-r", "-b", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    present = forbidden_archive_modules(archive)
    if present:
        raise SystemExit(f"Forbidden modules found in the executable archive: {present}")
    print(f"NetViz bundle audit passed: {bundle}")


if __name__ == "__main__":
    main()
