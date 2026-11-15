# Changelog

## 0.8.0

**Security**
- Images over 400 megapixels are refused before decoding.
- Minimum versions past known CVEs: Pillow 10.3, OpenCV 4.8.1.78, PyTorch 2.6 for maximum quality.
- Web page: frame, sniffing, referrer and caching protections on every response; foreign origins refused.
- Downloads: HTTPS only, size-capped, model checksums re-checked on every load.
- CI with a read-only token and pinned actions; Dependabot; `SECURITY.md`.

**Web page**
- Before/after slider at print size, centered on the largest face (YuNet detector, bundled).
- Clearer verdict after AI sharpening, and the page opens instantly even with PyTorch installed.

**Project**
- New README, `docs/guide.md` and `docs/how-it-works.md`; ruff lint in CI.

## 0.7.0

- `--max-quality`: newer AI models via PyTorch, picked per image. A web-photo model for photos, Real-ESRGAN for artwork, chosen from a test of six models.
- `a4print-setup-ai --max` downloads and verifies the models.

## 0.6.0

- AI results are merged with a plain enlargement layer by layer: tones from the original, edges from the AI, erased texture put back. Tone drift on a portrait fell from 1.5 to 0.2 Lab units and skin texture from 49% to 100% of the original.

## 0.5.0

- Printer calibration: a test page and saved `--shadows` / `--midtones` defaults.
- Lower memory for large posters (3.6 to 2.1 GB peak on a 3x3 poster with AI).
- Web page: download all results as one PDF; keeps the 25 most recent jobs.

## 0.4.0

- Artwork vs photo detection; artwork gets the full AI result.
- Very small text in artwork keeps the plain enlargement, so the AI can't change what it says.
- Preview shows a detail at print size.

## 0.3.0

- `--fill auto` (new default): keeps designed posters whole, padded in their own paper color and grain.
- `a4print-setup-ai`: one-step, checksum-verified Real-ESRGAN install.
- Simpler web page with a plain quality verdict.

## 0.2.0

- Noise detection that ignores texture, gentler denoise that keeps grain, JPEG artifact cleanup.
- `--margin`, `--combine`, `--grid`, percentage regions for `--remove`.
- Poster mode (`--poster`), a local web page (`a4print-web`), PDFs that ask to print at actual size.
- HEIC, AVIF, 16-bit, float and huge images; clear errors for damaged files.
- Tests on Linux, macOS and Windows.

## 0.1.0

- First version: A4 portrait or landscape at 300 DPI, crop / extend / fit, border trimming, inpainting, cleanup, Real-ESRGAN upscaling, PNG, PDF and before/after preview.
