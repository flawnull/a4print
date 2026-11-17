<div align="center">

# a4print

**Print-ready A4 pages and wall posters from any image, with AI upscaling that stays true to the original.**

[![tests](https://github.com/flawnull/a4print/actions/workflows/tests.yml/badge.svg)](https://github.com/flawnull/a4print/actions/workflows/tests.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)
![macOS | Windows | Linux](https://img.shields.io/badge/runs%20on-macOS%20%7C%20Windows%20%7C%20Linux-555)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f5d50)](https://github.com/flawnull/a4print/blob/main/LICENSE)

![A 600 x 400 px photo of an espresso cup becomes a print-ready A4 page; a close-up at print size compares a plain enlargement of the spoon with a4print's sharper result](https://raw.githubusercontent.com/flawnull/a4print/main/docs/images/hero.jpg)

<sub>A 600 x 400 px download printed on A4 with <code>--max-quality</code>. Photo: Rachel Michetti, CC0, from scikit-image.</sub>

</div>

Most images worth printing are too small: a Pinterest download or a phone screenshot comes out soft and blocky on paper. a4print turns any image into a 300 DPI PDF sized for a home printer (A4, Letter or A3), as one page or a poster of several sheets. It fits the image to the page without cutting off what matters, cleans up compression artifacts, and can sharpen with AI models that run on your own computer.

<p align="center">
  <img src="https://raw.githubusercontent.com/flawnull/a4print/main/docs/images/demo.gif" width="640" alt="A divider sweeps across a close-up of the spoon at print size: blocky on the plain enlargement side, crisp on the a4print side">
</p>

## Quick start

```bash
pipx install "a4print[heic] @ git+https://github.com/flawnull/a4print"
```

```bash
a4print-web
```

Drop in an image, pick a size, download the PDF. Or from the terminal:

```bash
a4print photo.jpg --poster 2x2 --ai
```

New to pipx, or want AI upscaling? See the [install guide](https://github.com/flawnull/a4print/blob/main/docs/guide.md#install).

## Features

- **A4, US Letter or A3.** Pick the paper your printer uses; the web page defaults to Letter in the US and Canada, A4 elsewhere.
- **Smart fit.** Designed posters keep their frames and credit lines; the page is extended in the poster's own paper color and grain. Photos are cropped only when little is lost, never through a face, and you can drag the crop yourself on the web page.
- **AI upscaling, kept honest.** Photos get RealESRNet, a model that sharpens what the photo has without drawing new faces or textures; artwork gets Real-ESRGAN's crisp edges. Both were chosen in a test against the original photos. Tones come from the original, tiny text the AI would garble is left as is, and strong enlargements get fine print grain so they don't look painted.
- **Photo sheets.** Put 2 to 12 pictures on one page with cutting guides; photos are turned when that crops less.
- **Posters on a home printer.** Split one image over 2 to 36 A4 sheets, with margins, overlap strips, alignment guides and labeled sheets.
- **Printer calibration.** A one-page test sheet measures how dark your printer prints, and every later print is tuned to it.
- **Cleanup only when needed.** Noise and JPEG artifacts are measured first; clean images are left alone. Black-and-white photos stay neutral gray.
- **Any input.** JPEG, PNG, WebP, AVIF, HEIC, TIFF, CMYK, 16-bit and float scans, transparency, EXIF rotation, color profiles.
- **Local and private.** Nothing is uploaded anywhere. The web page listens only on your own computer.

## Web app

`a4print-web` opens a small page in your browser. Choose images, pick a page, a poster or a photo sheet, and click once. You can drag the crop on a preview, with the poster's sheet lines shown, and erase a date stamp by drawing a box over it. Each result says in plain words how sharp the print will be, and a slider compares 5 x 5 cm of the print at actual size: plain enlargement against a4print, centered on a face when there is one.

![The a4print web page. Left: image, paper and print size, AI option, and the crop preview with the poster's sheet lines on a photo. Right: a result with its verdict and a before/after slider at print size](https://raw.githubusercontent.com/flawnull/a4print/main/docs/images/web.jpg)

AI setup is a button on the same page that downloads the official builds, checks their checksums and tests them. Batches download as one PDF. Uploads stay on your machine and are deleted when you stop the server.

## Posters

<img src="https://raw.githubusercontent.com/flawnull/a4print/main/docs/images/poster.jpg" alt="Poster preview: a graphic divided into four A4 sheets labeled A1, A2, B1 and B2" width="300" align="right">

`--poster 2x2` makes a poster of about A2 from four A4 sheets; `3x3` is about A1. a4print picks portrait or landscape sheets to crop the least, keeps a white margin your printer can handle, and labels each sheet with where it goes and which edges to trim.

A test cuts and reassembles the sheets in software and checks the result against the poster pixel for pixel, so the pieces line up.

<br clear="right"/>

## How it works

Upscaling AIs have two habits that hurt a print: they shift tones, and they wipe out fine texture, which makes skin look waxy. a4print never prints the AI result directly. It splits the image into layers and takes each from the more faithful source: tones and gradients from a plain enlargement of the original, edges from the AI, and lost texture put back where the AI removed it.

| | AI alone | a4print |
|---|---|---|
| Tone drift on a portrait (Lab units; 2 is noticeable) | 1.5 average, 7 in the worst 1% | 0.2 average, 1 in the worst 1% |
| Skin texture kept | 49% | 100% |
| Edge sharpness | 1.41x plain | 1.40x plain |

The models were chosen with a ground-truth test: nine photos, including crowds with small faces, shrunk the way Pinterest shrinks images and enlarged back. GAN models, which draw the crispest detail, scored worst on faces; the model a4print uses for photos scored best on every measure. The full pipeline, the model comparison and the engineering notes are in [how it works](https://github.com/flawnull/a4print/blob/main/docs/how-it-works.md).

## Built with

Python, OpenCV, Pillow and NumPy for image processing; img2pdf and pikepdf for PDFs; Real-ESRGAN (ncnn/Vulkan) and PyTorch with spandrel for AI upscaling; YuNet for face detection. The web page uses only Python's standard library and plain HTML, CSS and JavaScript.

```
a4print/
├── cli.py          command line, report wording
├── web.py          local web page and its server
├── pipeline.py     the processing steps, A4 pages and posters
├── layout.py       page geometry, fill modes, border detection
├── enhance.py      cleanup, tone curve, content and face detection
├── upscale.py      Lanczos, Real-ESRGAN, the layer merge
├── maxq.py         maximum quality models (PyTorch)
├── poster.py       poster planning, sheet slicing, guides
├── calibrate.py    printer test page and saved settings
├── setup_ai.py     verified downloads of AI programs and models
└── files.py        loading, sRGB, PNG/PDF/preview output
```

## Quality and security

About 220 tests run on Linux, macOS and Windows with Python 3.10 and 3.13, using only generated and public-domain images. AI programs and models are downloaded from their authors' releases and checked against pinned SHA-256 checksums; the web page refuses requests from other websites. Details in [SECURITY.md](https://github.com/flawnull/a4print/blob/main/SECURITY.md).

Print only images you have the right to use. `--remove`, which erases date stamps and marks, is meant for your own photos or images you're allowed to edit.

## Docs

- [Guide](https://github.com/flawnull/a4print/blob/main/docs/guide.md): install, every option, posters, calibration, common problems
- [How it works](https://github.com/flawnull/a4print/blob/main/docs/how-it-works.md): the pipeline, the AI merge, model comparison, engineering notes
- [Security](https://github.com/flawnull/a4print/blob/main/SECURITY.md)
- [Changelog](https://github.com/flawnull/a4print/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/flawnull/a4print/blob/main/CONTRIBUTING.md)

## License

MIT, see [LICENSE](https://github.com/flawnull/a4print/blob/main/LICENSE). AI models and bundled components are credited in [THIRD_PARTY_NOTICES.md](https://github.com/flawnull/a4print/blob/main/THIRD_PARTY_NOTICES.md).
