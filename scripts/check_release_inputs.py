import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require_file(path: Path) -> bytes:
    if not path.is_file():
        raise SystemExit(f"Required release input is missing: {path}")
    return path.read_bytes()


def check_icon(path: Path) -> None:
    data = require_file(path)
    if len(data) < 6:
        raise SystemExit(f"Windows icon is invalid: {path}")
    reserved, image_type, count = struct.unpack_from("<HHH", data)
    if reserved != 0 or image_type != 1 or len(data) < 6 + count * 16:
        raise SystemExit(f"Windows icon is invalid: {path}")
    sizes = set()
    for index in range(count):
        width, height = struct.unpack_from("BB", data, 6 + index * 16)
        sizes.add(((width or 256), (height or 256)))
    missing = [size for size in (16, 32, 48, 256) if (size, size) not in sizes]
    if missing:
        raise SystemExit(f"Windows icon lacks required sizes: {missing}")


def main() -> None:
    favicon = require_file(ROOT / "frontend" / "public" / "favicon.svg").decode("utf-8")
    if "<svg" not in favicon or "#863bff" in favicon:
        raise SystemExit("frontend/public/favicon.svg is missing or still uses Vite branding.")
    check_icon(ROOT / "packaging" / "assets" / "netviz.ico")
    require_file(ROOT / "frontend" / "dist" / "index.html")
    for path in (ROOT / "LICENSE", ROOT / "packaging" / "windows_version_info.txt"):
        text = require_file(path).decode("utf-8")
        if "[COPYRIGHT HOLDER]" in text or "TBD" in text:
            raise SystemExit(f"Release metadata still has an unresolved copyright holder: {path}")
    print("NetViz release inputs are complete.")


if __name__ == "__main__":
    main()
