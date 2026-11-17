# Changelog

## 0.15.0

- Screenshots and other images that were enlarged before are detected: a4print finds the size their detail really has (the largest shrink that changes nothing visible) and gives the AI the image at that size. On a 1168 px phone screenshot whose detail was 425 px wide, this removed the scratches and smears the AI drew on walls and the painted look on edges. Photos at their real resolution are left as they are; in a ground-truth test the detector never fired on them.
- The print quality verdict and the DPI on the web page follow the real detail, and the report says when an image was enlarged before.
- Print grain is sized to the real enlargement.

## 0.14.0

- Photos are enlarged with RealESRNet x4plus, with `--ai` and `--max-quality`. The GAN models used before drew mannequin or cartoon faces in crowds and a fish-scale mesh on flat areas. In a ground-truth test on nine photos, RealESRNet was the most faithful on every measure: face SSIM at 3x went from 0.77 to 0.83, and the error in fine detail dropped below that of a plain enlargement. Artwork keeps Real-ESRGAN x4plus.
- `a4print-setup-ai` also sets up the photo model: it downloads the official weights, checks them, and converts them to the ncnn format on your computer, verified against a pinned checksum. Run it once more after updating. Until then, photos are enlarged without AI instead of with a GAN.
- Photos going through the AI skip a4print's own noise and JPEG cleanup, since the model does it and cleaning twice removed real texture.
- Fine print grain on photos enlarged more than 2x, so big enlargements don't look painted. `--no-grain` turns it off.
- `--max-quality` now runs the same models as `--ai`, through PyTorch in full precision; the 4xNomosWebPhoto_esrgan model is no longer used.
- Clearer wording on the web page after AI sharpening.

## 0.13.0

- The crop preview works for posters: the frame has the poster's shape for the chosen paper, grid and orientation, with lines where the sheets meet. `--crop-at` applies to posters too.
- Clearer result wording for posters and chosen crops; README screenshot of the current page.

## 0.12.0

- Web page: "Choose what gets cropped" shows the page-shaped frame on the image; drag it or use the arrow keys.
- `--crop-at X,Y` places the crop anywhere, in percent.
- Automatic crops keep faces in the frame: the crop shifts just enough that faces are not cut.

## 0.11.0

- Web page: erase a date stamp or mark by dragging boxes on the image.
- Smarter erasing (`--remove` and the web tool): only the marks inside a box are filled, so paper grain stays and nearby lines are not smeared in. Solid patches still get the whole box filled.
- Web page: elapsed time while a job runs; other poster sizes, the detail caption and the printer test page follow the chosen paper.
- Fixes from a code review: uploads with non-ASCII or very long names; same-named files from two folders overwriting each other; Letter and A3 outputs being reprocessed on a second run; photo-sheet and `--grid` errors ending in a traceback; one bad image losing a whole photo sheet; negative margins from the web page; very thin images; tiny erase boxes.

## 0.10.0

- Photo sheets: `--per-page 2|4|6|8|9|12` lays several images on each page with dashed cutting guides; photos are turned 90 degrees when that crops less. Each image gets the full pipeline. One PDF for all sheets.
- The command line takes several files or folders at once.
- Web page: a "Photo sheet" print size; all chosen images go into one job.

## 0.9.0

- US Letter and A3 paper: `--paper letter|a3` for pages, posters (sheets of that paper), PDFs and the printer test page. `a4print-calibrate --paper` saves a default.
- Web page: paper choice next to the print sizes, defaulting to Letter in Letter countries (from the browser's region) and A4 elsewhere; size labels in inches for Letter, computed by the poster planner.
- Output files are named after the paper (`photo_letter.pdf`).
- PyPI-ready packaging, publishing workflow, animated demo, social preview, contributing guide and issue forms.

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
