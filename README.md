# TAfC Texture Assets

Public data and downloadable assets for **Texture Assets for Creation (TAfC)**.

The frontend lives in `sakusdev/Texture-Assets-for-Creation`.

## Data contract

- `index.json` is the catalog consumed by the TAfC frontend at build time.
- `schema/texture.schema.json` documents and validates individual asset records.
- Map and preview URLs may be absolute URLs or repository-relative paths.
- Large downloadable 1K/2K/4K texture packs should be published through GitHub Releases and referenced by each resolution's `download` field.

The catalog may be empty while the first production assets are being prepared; the frontend keeps a local demo catalog as a development fallback.
