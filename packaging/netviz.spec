from pathlib import Path

from PyInstaller.utils.hooks import collect_all

project_root = Path(SPEC).resolve().parents[1]
frontend_dist = project_root / "frontend" / "dist"
icon_path = project_root / "packaging" / "assets" / "netviz.ico"
version_path = project_root / "packaging" / "windows_version_info.txt"

def is_runtime_numpy_submodule(module_name):
    parts = module_name.split(".")
    return "tests" not in parts and not module_name.startswith("numpy._pyinstaller")


def is_dependency_test_entry(entry):
    destination = entry[0].replace("\\", "/").replace(".", "/")
    parts = destination.split("/")
    return bool(parts) and parts[0] in {"numpy", "torch"} and "tests" in parts


numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all(
    "numpy",
    include_py_files=False,
    filter_submodules=is_runtime_numpy_submodule,
    exclude_datas=["**/tests/**"],
)

analysis = Analysis(
    [str(project_root / "desktop" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=numpy_binaries,
    datas=[(str(frontend_dist), "frontend/dist"), *numpy_datas],
    hiddenimports=[
        *numpy_hiddenimports,
        "clr",
        "webview.platforms.edgechromium",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cefpython3",
        "gi",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "pytest",
        "tkinter",
        "torchvision",
        "transformers",
    ],
    noarchive=False,
    optimize=1,
)

# Third-party hooks can contribute their own test data or modules after the
# explicit collections above. Remove only dependency-owned test packages; keep
# runtime data and distribution licenses intact.
analysis.datas = TOC(entry for entry in analysis.datas if not is_dependency_test_entry(entry))
analysis.pure = TOC(entry for entry in analysis.pure if not is_dependency_test_entry(entry))

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="NetViz",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
    icon=str(icon_path),
    version=str(version_path),
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NetViz",
)
