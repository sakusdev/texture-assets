# TAfC Texture Assets

Public data and downloadable assets for **Texture Assets for Creation (TAfC)**.

The frontend lives in `sakusdev/Texture-Assets-for-Creation`.

## Data contract

- `index.json` is the catalog consumed by the TAfC frontend at build time.
- `schema/texture.schema.json` documents and validates individual asset records.
- Map and preview URLs may be absolute URLs or repository-relative paths.
- Large downloadable 1K/2K/4K texture packs should be published through GitHub Releases and referenced by each resolution's `download` field.

Example catalog entry:

```json
{
  "id": "white-ceramic-pool-tile-001",
  "slug": "white-ceramic-pool-tile",
  "name": "White Ceramic Pool Tile",
  "category": "Tiles",
  "description": "A slightly aged seamless ceramic pool tile.",
  "tags": ["ceramic", "pool", "liminal"],
  "seamless": true,
  "featured": true,
  "license": "CC0-1.0",
  "preview": "assets/white-ceramic-pool-tile-001/preview.webp",
  "maps": {
    "baseColor": "assets/white-ceramic-pool-tile-001/maps/basecolor.png",
    "normal": "assets/white-ceramic-pool-tile-001/maps/normal.png",
    "roughness": "assets/white-ceramic-pool-tile-001/maps/roughness.png",
    "height": "assets/white-ceramic-pool-tile-001/maps/height.png",
    "ao": "assets/white-ceramic-pool-tile-001/maps/ao.png"
  },
  "resolutions": [
    {
      "label": "2K",
      "width": 2048,
      "height": 2048,
      "download": "https://github.com/sakusdev/texture-assets/releases/download/white-ceramic-pool-tile-001-v1/white-ceramic-pool-tile-2k.zip"
    }
  ]
}
```

The catalog may be empty while the first production assets are being prepared; the frontend keeps a local demo catalog as a development fallback.
