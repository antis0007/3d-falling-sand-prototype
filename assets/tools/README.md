# Tool textures

Optional custom tool textures can be dropped here.

Supported file names:
- `brush.ppm`
- `builders_wand.ppm`
- `destructor_wand.ppm`

Texture guidance:
- 16x16 pixel art is the default target.
- Up to 64x64 textures are supported.
- White pixels (`255,255,255`) are treated as transparent for sprite backgrounds.
- Sprites are rendered with nearest-neighbor filtering (no smoothing/blurring).

If a file is missing or fails to load, the game uses built-in fallback textures.
