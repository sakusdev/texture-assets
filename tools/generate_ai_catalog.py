from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import colorsys
import json
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources"
ASSETS = ROOT / "assets"
INDEX = ROOT / "index.json"

MATERIALS = [
    {
        "id": "white-ceramic-pool-tile-001", "slug": "white-ceramic-pool-tile", "name": "White Ceramic Pool Tile",
        "category": "Tiles", "tags": ["ceramic", "pool", "white", "glossy", "liminal", "architectural"],
        "description": "Slightly aged white ceramic pool tiles with subtle gloss, grout staining, and realistic surface variation.",
        "featured": True, "kind": "white_tile", "prefix": "WhiteCeramicPoolTile"
    },
    {
        "id": "blue-ceramic-pool-tile-001", "slug": "blue-ceramic-pool-tile", "name": "Blue Ceramic Pool Tile",
        "category": "Tiles", "tags": ["ceramic", "pool", "blue", "glossy", "mosaic", "liminal"],
        "description": "Glossy blue ceramic pool tiles with rich tonal variation and clean grout lines, ideal for pools and spa environments.",
        "featured": True, "kind": "blue_tile", "prefix": "BlueCeramicPoolTile"
    },
    {
        "id": "concrete-wall-001", "slug": "concrete-wall-001", "name": "Concrete Wall 001",
        "category": "Concrete", "tags": ["concrete", "wall", "gray", "brutalist", "architectural", "industrial"],
        "description": "A realistic gray concrete wall texture with pores, stains, and subtle vertical weathering.",
        "featured": True, "kind": "concrete", "prefix": "ConcreteWall001"
    },
    {
        "id": "rusty-metal-panel-001", "slug": "rusty-metal-panel", "name": "Rusty Metal Panel",
        "category": "Metal", "tags": ["metal", "rust", "industrial", "panel", "steel", "aged"],
        "description": "Worn steel panel sections with heavy rust, bolts, and industrial weathering.",
        "featured": True, "kind": "rusty_metal", "prefix": "RustyMetalPanel"
    },
    {
        "id": "wet-concrete-floor-001", "slug": "wet-concrete-floor", "name": "Wet Concrete Floor",
        "category": "Ground", "tags": ["concrete", "wet", "floor", "dark", "reflective", "liminal"],
        "description": "Dark wet concrete with reflective patches, stains, and damp surface variation.",
        "featured": False, "kind": "wet_concrete", "prefix": "WetConcreteFloor"
    },
    {
        "id": "old-plaster-wall-001", "slug": "old-plaster-wall", "name": "Old Plaster Wall",
        "category": "Plaster", "tags": ["plaster", "wall", "aged", "cracked", "beige", "vintage"],
        "description": "Aged plaster wall with peeling areas, fine cracks, and warm off-white discoloration.",
        "featured": False, "kind": "plaster", "prefix": "OldPlasterWall"
    },
]

RESOLUTIONS = [(1024, "1K"), (2048, "2K"), (4096, "4K")]


def make_edges_tileable(arr: np.ndarray, band_ratio: float = 0.035) -> np.ndarray:
    """Gently force opposite borders to match while leaving most AI detail untouched."""
    out = arr.astype(np.float32).copy()
    h, w = out.shape[:2]
    bx = max(4, int(w * band_ratio))
    by = max(4, int(h * band_ratio))
    for i in range(bx):
        t = 1.0 - i / bx
        weight = t * t
        avg = 0.5 * (out[:, i] + out[:, -1 - i])
        out[:, i] = out[:, i] * (1 - weight) + avg * weight
        out[:, -1 - i] = out[:, -1 - i] * (1 - weight) + avg * weight
    for i in range(by):
        t = 1.0 - i / by
        weight = t * t
        avg = 0.5 * (out[i, :] + out[-1 - i, :])
        out[i, :] = out[i, :] * (1 - weight) + avg * weight
        out[-1 - i, :] = out[-1 - i, :] * (1 - weight) + avg * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def rgb_features(rgb: np.ndarray):
    f = rgb.astype(np.float32) / 255.0
    luma = np.clip(f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722, 0, 1)
    mx = f.max(axis=-1)
    mn = f.min(axis=-1)
    sat = np.where(mx > 1e-5, (mx - mn) / np.maximum(mx, 1e-5), 0.0)
    return luma.astype(np.float32), sat.astype(np.float32)


def blur_gray(gray: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.clip(gray * 255, 0, 255).astype(np.uint8), "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0


def derive_maps(rgb: np.ndarray, kind: str):
    luma, sat = rgb_features(rgb)
    low = blur_gray(luma, 10.0)
    mid = blur_gray(luma, 2.0)
    high = np.clip(luma - mid, -0.25, 0.25)

    if kind in ("white_tile", "blue_tile"):
        if kind == "white_tile":
            grout = np.clip((0.78 - luma) / 0.35, 0, 1)
        else:
            grout = np.clip((0.22 - luma) / 0.20, 0, 1) * np.clip((0.45 - sat) / 0.45, 0, 1)
        height = np.clip(0.72 - 0.52 * grout + 0.10 * high, 0, 1)
        if kind == "white_tile":
            rough = np.clip(0.20 + 0.48 * grout + 0.12 * np.abs(high) * 4, 0.15, 0.85)
        else:
            rough = np.clip(0.18 + 0.46 * grout + 0.10 * np.abs(high) * 4, 0.12, 0.82)
        ao = np.clip(1.0 - 0.48 * grout, 0.35, 1)
        metallic = None
    elif kind == "rusty_metal":
        rust = np.clip((sat - 0.18) / 0.45, 0, 1) * np.clip((luma - 0.10) / 0.55, 0, 1)
        seams = np.clip((0.22 - luma) / 0.22, 0, 1)
        height = np.clip(0.56 + 0.16 * high - 0.18 * seams + 0.05 * (low - 0.5), 0, 1)
        rough = np.clip(0.38 + 0.45 * rust + 0.16 * np.abs(high) * 4, 0.28, 0.98)
        ao = np.clip(1.0 - 0.32 * seams - 0.10 * rust, 0.35, 1)
        metallic = np.clip(0.96 - 0.90 * rust - 0.45 * seams, 0, 1)
    elif kind == "wet_concrete":
        detail = np.clip((luma - low) * 1.8, -0.25, 0.25)
        height = np.clip(0.52 + 0.35 * detail + 0.06 * high, 0, 1)
        wet = np.clip((0.52 - low) / 0.40, 0, 1)
        rough = np.clip(0.76 - 0.58 * wet + 0.10 * np.abs(high) * 4, 0.10, 0.92)
        ao = np.clip(0.94 - 0.18 * np.clip(-detail * 3, 0, 1), 0.55, 1)
        metallic = None
    elif kind == "plaster":
        damage = np.clip((0.72 - low) / 0.50, 0, 1)
        height = np.clip(0.58 + 0.28 * (luma - low) + 0.10 * high - 0.10 * damage, 0, 1)
        rough = np.clip(0.76 + 0.16 * damage + 0.10 * np.abs(high) * 3, 0.68, 0.98)
        ao = np.clip(0.97 - 0.24 * damage, 0.55, 1)
        metallic = None
    else:
        pores = np.clip((low - luma) * 3.2, 0, 1)
        height = np.clip(0.54 + 0.30 * (luma - low) + 0.08 * high - 0.14 * pores, 0, 1)
        rough = np.clip(0.72 + 0.18 * pores + 0.08 * np.abs(high) * 4, 0.62, 0.97)
        ao = np.clip(0.98 - 0.28 * pores, 0.55, 1)
        metallic = None

    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    strength = 26.0 if kind in ("white_tile", "blue_tile", "rusty_metal") else 14.0
    nx, ny, nz = -dx * strength, dy * strength, np.ones_like(height)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx * inv, ny * inv, nz * inv], axis=-1)
    normal_u8 = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    result = {
        "Normal": (normal_u8, "RGB"),
        "Roughness": ((rough * 255).astype(np.uint8), "L"),
        "Height": ((height * 255).astype(np.uint8), "L"),
        "AO": ((ao * 255).astype(np.uint8), "L"),
    }
    if metallic is not None:
        result["Metallic"] = ((metallic * 255).astype(np.uint8), "L")
    return result


def save_png(arr: np.ndarray, mode: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path, compress_level=4)


def build_one(material: dict) -> dict:
    asset_id = material["id"]
    candidates = [SOURCES / f"{asset_id}.jpg", SOURCES / f"{asset_id}.webp", SOURCES / f"{asset_id}.png"]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(f"Missing AI-generated source for {asset_id}")

    asset_dir = ASSETS / asset_id
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True)
    downloads = asset_dir / "downloads"
    downloads.mkdir()

    src = Image.open(source).convert("RGB")
    src_arr = make_edges_tileable(np.asarray(src))
    src_img = Image.fromarray(src_arr, "RGB")

    src_img.save(asset_dir / "AI-Source-BaseColor.webp", "WEBP", quality=96, method=6)
    src_img.resize((640, 640), Image.Resampling.LANCZOS).save(asset_dir / "preview.webp", "WEBP", quality=91, method=6)

    zip_files = {}
    map_paths_1k = {}
    for size, label in RESOLUTIONS:
        resized = src_img.resize((size, size), Image.Resampling.LANCZOS)
        rgb = np.asarray(resized, dtype=np.uint8)
        folder = asset_dir / label
        folder.mkdir()

        base_path = folder / f'{material["prefix"]}_{label}_BaseColor.png'
        save_png(rgb, "RGB", base_path)
        generated = derive_maps(rgb, material["kind"])
        files = [base_path]
        if label == "1K": map_paths_1k["baseColor"] = base_path
        for map_name, (arr, mode) in generated.items():
            path = folder / f'{material["prefix"]}_{label}_{map_name}.png'
            save_png(arr, mode, path)
            files.append(path)
            if label == "1K":
                map_paths_1k[map_name[0].lower() + map_name[1:]] = path

        zip_name = f'{material["name"].replace(" ", "-")}-{label}.zip'
        zip_path = downloads / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
            for p in files:
                z.write(p, arcname=p.name)
            z.writestr("README.txt", f'TAfC — {material["name"]}\nAI-generated Base Color with derived PBR maps.\nResolution: {label}\nSeamless edge-corrected.\n')
        zip_files[label] = zip_path

    readme = f'''# {material["name"]}\n\nBase Color source generated with OpenAI image generation for TAfC.\nPBR support maps are derived from that source image, not procedurally substituted artwork.\n\n- Seamless edge-corrected\n- Base Color / Normal / Roughness / Height / AO{' / Metallic' if material['kind'] == 'rusty_metal' else ''}\n- 1K / 2K / 4K\n'''
    (asset_dir / "README.md").write_text(readme, encoding="utf-8")

    maps = {}
    for key, path in map_paths_1k.items():
        maps[key] = path.relative_to(ROOT).as_posix()

    return {
        "id": asset_id,
        "slug": material["slug"],
        "name": material["name"],
        "category": material["category"],
        "tags": material["tags"],
        "description": material["description"],
        "seamless": True,
        "featured": material["featured"],
        "license": "TAfC-Free",
        "sourceType": "ai-generated",
        "source": f"assets/{asset_id}/AI-Source-BaseColor.webp",
        "preview": f"assets/{asset_id}/preview.webp",
        "maps": maps,
        "resolutions": [
            {
                "label": label,
                "width": size,
                "height": size,
                "download": zip_files[label].relative_to(ROOT).as_posix(),
                "sizeBytes": zip_files[label].stat().st_size,
            }
            for size, label in RESOLUTIONS
        ],
    }


def main():
    ASSETS.mkdir(exist_ok=True)
    keep = {m["id"] for m in MATERIALS}
    for child in ASSETS.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child)

    catalog = [build_one(m) for m in MATERIALS]
    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "assets": catalog,
    }
    INDEX.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(catalog)} AI-source TAfC materials")


if __name__ == "__main__":
    main()
