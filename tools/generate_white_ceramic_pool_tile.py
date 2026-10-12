from pathlib import Path
from datetime import datetime, timezone
import json
import zipfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSET_ID = "white-ceramic-pool-tile-001"
ASSET_DIR = ROOT / "assets" / ASSET_ID
DOWNLOAD_DIR = ASSET_DIR / "downloads"
INDEX_PATH = ROOT / "index.json"


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def periodic_noise(u, v):
    return (
        0.42 * np.sin(2 * np.pi * (3 * u + 5 * v))
        + 0.24 * np.sin(2 * np.pi * (7 * u - 4 * v) + 1.7)
        + 0.18 * np.cos(2 * np.pi * (11 * u + 9 * v) + 0.6)
        + 0.10 * np.sin(2 * np.pi * (23 * u - 17 * v) + 2.4)
    ).astype(np.float32)


def generate_maps(size: int):
    tiles = 8
    y, x = np.mgrid[0:size, 0:size]
    u = x.astype(np.float32) / size
    v = y.astype(np.float32) / size
    gx, gy = u * tiles, v * tiles
    fx, fy = gx - np.floor(gx), gy - np.floor(gy)
    d = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))

    profile = smoothstep(0.035, 0.075, d).astype(np.float32)
    edge = 1.0 - profile
    n1 = periodic_noise(u, v)
    n2 = periodic_noise((u + 0.173) % 1, (v + 0.319) % 1)

    tx = np.floor(gx).astype(np.int16) % tiles
    ty = np.floor(gy).astype(np.int16) % tiles
    tile_variation = (((tx * 13 + ty * 7 + tx * ty * 3) % 17) / 16.0 - 0.5).astype(np.float32)

    height = np.clip(0.12 + 0.70 * profile + profile * (0.012 * n1 + 0.018 * tile_variation), 0, 1)
    roughness = np.clip(0.23 + 0.43 * edge + 0.06 * (n2 + 1) / 2, 0, 1)
    ao = np.clip(1.0 - 0.48 * edge - 0.05 * (1 - smoothstep(0.06, 0.17, d)), 0.25, 1)

    tile_rgb = np.array([234, 232, 222], dtype=np.float32)
    grout_rgb = np.array([174, 170, 159], dtype=np.float32)
    base = grout_rgb[None, None, :] * (1 - profile[..., None]) + tile_rgb[None, None, :] * profile[..., None]
    base += profile[..., None] * (5 * n1[..., None] + 5 * tile_variation[..., None])
    stain = np.clip((n2 - 0.35) * 0.17, 0, 0.10) * (0.35 + 0.65 * edge)
    stain_rgb = np.array([145, 120, 88], dtype=np.float32)
    base = base * (1 - stain[..., None]) + stain_rgb[None, None, :] * stain[..., None]
    base = np.clip(base, 0, 255).astype(np.uint8)

    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    strength = 20.0
    nx, ny, nz = -dx * strength, dy * strength, np.ones_like(height)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx * inv, ny * inv, nz * inv], axis=-1)
    normal = np.clip((normal * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

    return {
        "BaseColor": (base, "RGB"),
        "Normal": (normal, "RGB"),
        "Roughness": ((roughness * 255).astype(np.uint8), "L"),
        "Height": ((height * 255).astype(np.uint8), "L"),
        "AO": ((ao * 255).astype(np.uint8), "L"),
    }


def save_resolution(size: int, label: str):
    folder = ASSET_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for map_name, (array, mode) in generate_maps(size).items():
        path = folder / f"WhiteCeramicPoolTile_{label}_{map_name}.png"
        Image.fromarray(array, mode=mode).save(path, compress_level=1)
        files.append(path)
    return files


def make_zip(label: str, files):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = DOWNLOAD_DIR / f"White-Ceramic-Pool-Tile-001-{label}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for file in files:
            archive.write(file, arcname=file.name)
        archive.writestr(
            "README.txt",
            "TAfC — White Ceramic Pool Tile 001\n"
            "Maps: Base Color, Normal, Roughness, Height, AO\n"
            f"Resolution: {label}\nSeamless / tileable.\n",
        )
    return path


def update_index(resolution_files, zip_files):
    preview_rel = f"assets/{ASSET_ID}/preview.webp"
    map_base = f"assets/{ASSET_ID}/1K"
    asset = {
        "id": ASSET_ID,
        "slug": "white-ceramic-pool-tile",
        "name": "White Ceramic Pool Tile",
        "category": "Tiles",
        "tags": ["ceramic", "pool", "liminal", "white", "architectural"],
        "description": "A slightly aged seamless white ceramic pool tile with subtle mineral staining and grout variation.",
        "seamless": True,
        "featured": True,
        "license": "TAfC-Free",
        "preview": preview_rel,
        "maps": {
            "baseColor": f"{map_base}/WhiteCeramicPoolTile_1K_BaseColor.png",
            "normal": f"{map_base}/WhiteCeramicPoolTile_1K_Normal.png",
            "roughness": f"{map_base}/WhiteCeramicPoolTile_1K_Roughness.png",
            "height": f"{map_base}/WhiteCeramicPoolTile_1K_Height.png",
            "ao": f"{map_base}/WhiteCeramicPoolTile_1K_AO.png",
        },
        "resolutions": [],
    }
    for label, size in [("1K", 1024), ("2K", 2048), ("4K", 4096)]:
        z = zip_files[label]
        asset["resolutions"].append({
            "label": label,
            "width": size,
            "height": size,
            "download": f"assets/{ASSET_ID}/downloads/{z.name}",
            "sizeBytes": z.stat().st_size,
        })

    catalog = {"schemaVersion": 1, "generatedAt": datetime.now(timezone.utc).isoformat(), "assets": [asset]}
    INDEX_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generated = {}
    zips = {}
    for size, label in [(1024, "1K"), (2048, "2K"), (4096, "4K")]:
        files = save_resolution(size, label)
        generated[label] = files
        zips[label] = make_zip(label, files)

    base_1k = Image.open(ASSET_DIR / "1K" / "WhiteCeramicPoolTile_1K_BaseColor.png").convert("RGB")
    base_1k.resize((512, 512), Image.Resampling.LANCZOS).save(ASSET_DIR / "preview.webp", "WEBP", quality=90, method=4)

    (ASSET_DIR / "README.md").write_text(
        "# White Ceramic Pool Tile 001\n\n"
        "Seamless PBR material generated for TAfC.\n\n"
        "Maps: Base Color, Normal, Roughness, Height, AO.\n"
        "Resolutions: 1K, 2K, 4K.\n",
        encoding="utf-8",
    )
    update_index(generated, zips)
    print("Generated", ASSET_ID)


if __name__ == "__main__":
    main()
