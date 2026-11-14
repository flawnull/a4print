# a4print

[![tests](https://github.com/flawnull/a4print/actions/workflows/tests.yml/badge.svg)](https://github.com/flawnull/a4print/actions/workflows/tests.yml)

Prepare any image (a photo, a poster, a Pinterest download) for borderless printing on A4 paper with a home printer.

For each image, a4print writes:

- `name_a4.png`: 2480 x 3508 px (or 3508 x 2480 landscape), 300 DPI, sRGB profile embedded
- `name_a4.pdf`: one A4 page, image edge to edge, no margins
- `name_preview.jpg`: before and after, side by side

It also reports the image's effective resolution at A4 and warns when it drops below 150 DPI.

## Install

You need Python 3.9 or newer.

```bash
git clone https://github.com/flawnull/a4print.git
```

```bash
cd a4print
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

```bash
pip install -e .
```

This installs the dependencies (Pillow, OpenCV, NumPy, img2pdf) and the `a4print` command. If you'd rather not install, run `pip install -r requirements.txt` and use `python -m a4print` in place of `a4print`.

JPEG, PNG, WebP, AVIF, TIFF, BMP and GIF work out of the box. For iPhone photos (HEIC), add:

```bash
pip install -e ".[heic]"
```

### Optional: AI upscaling

`--ai` uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) through its standalone `realesrgan-ncnn-vulkan` tool. It runs on most GPUs via Vulkan and needs no Python packages.

1. Download `realesrgan-ncnn-vulkan-*-windows/macos/ubuntu.zip` from the [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases).
2. Unzip it. Keep the `models` folder next to the executable.
3. Put that folder on your `PATH`, or point a4print at the executable:

```bash
export A4PRINT_REALESRGAN=/path/to/realesrgan-ncnn-vulkan
```

If `--ai` is used and the tool can't be found, a4print stops and tells you how to install it.

Speed depends on your hardware. With a graphics card it takes a few seconds per image. Without one it still works through a software Vulkan driver (on Linux, `mesa-vulkan-drivers`), but expect about 3 minutes for a 500 px photo.

## Usage

Print a photo, filling the page:

```bash
a4print holiday.jpg
```

Results go to `./a4print_out/`. Use `-o` to choose another folder.

Process a whole folder:

```bash
a4print ~/Downloads/posters -o ~/Desktop/to-print
```

Keep the whole poster visible, filling the gaps with a blurred extension of its edges:

```bash
a4print poster.webp --fill extend
```

Keep the whole image with a plain white border:

```bash
a4print scan.png --fill fit
```

Crop, but keep the top of the image (for portraits where heads would be cut):

```bash
a4print group.jpg --anchor top
```

Strip a white or beige margin a Pinterest pin came with, then fill the page:

```bash
a4print pin.jpg --trim-border
```

Remove a date stamp and a credit line. Each region is `x,y,width,height`, in pixels of the original image or in percent. To find the numbers, save a copy with a labeled grid first:

```bash
a4print old.jpg --grid
```

Then read the coordinates off `old_grid.jpg`:

```bash
a4print old.jpg --remove 1820,1300,260,60 --remove 2%,2%,25%,4%
```

Your printer can't print borderless? Keep a 5 mm white margin so nothing gets cut or rescaled:

```bash
a4print poster.jpg --margin 5
```

Prepare a folder and get one PDF with every page, ready to print in one go:

```bash
a4print ~/Downloads/posters --combine
```

Upscale a small image with Real-ESRGAN:

```bash
a4print small.jpg --ai
```

Illustrations, posters and graphics can take the AI result at full strength:

```bash
a4print illustration.png --ai --ai-strength 1
```

Turn off all tone and sharpness changes:

```bash
a4print artwork.png --no-enhance
```

Sample output:

```
holiday.jpg
  source      4000 x 3000 px
  page        A4 landscape, crop, Lanczos
  resolution  342 DPI at A4, excellent
  cleanup     applied (noise level 3.4)
  saved       a4print_out/holiday_a4.png
  saved       a4print_out/holiday_a4.pdf
  saved       a4print_out/holiday_preview.jpg
```

## Options

| Option | Default | What it does |
|---|---|---|
| `-o, --output-dir DIR` | `a4print_out` | Where results are saved |
| `--orientation auto\|portrait\|landscape` | `auto` | Auto picks landscape for wide images, portrait otherwise |
| `--fill crop\|extend\|fit` | `crop` | See "Fill modes" below |
| `--anchor center\|top\|bottom\|left\|right` | `center` | Which part to keep when cropping |
| `--margin MM` | `0` | White margin around the image, for printers without borderless printing |
| `--trim-border` | off | Remove plain outer margins before fitting |
| `--trim-tolerance N` | `18` | How far (0-255) a margin pixel may differ from the margin color |
| `--remove x,y,w,h` | none | Erase a rectangle by inpainting, in pixels or percent. Repeatable |
| `--grid` | off | Only save `name_grid.jpg`, the image with labeled coordinates |
| `--denoise S` | `1.0` | Noise and JPEG artifact cleanup strength. `0` turns it off. Runs only when needed |
| `--sharpen S` | `0.5` | Unsharp mask amount. `0` turns it off |
| `--sharpen-radius PX` | `2.0` | Unsharp mask radius in output pixels |
| `--shadows S` | `0.03` | Shadow lift, 0 to 0.2 (0.03 lifts black by about 8 levels) |
| `--midtones S` | `0.05` | Midtone lift. `0` turns it off |
| `--no-enhance` | off | Skip denoise, sharpening and tone lift |
| `--ai` | off | Upscale with Real-ESRGAN instead of Lanczos |
| `--ai-model NAME` | `realesrgan-x4plus` | Real-ESRGAN model name (`realesrgan-x4plus-anime` for drawings) |
| `--ai-strength S` | `0.7` | Mix of the AI result with a plain resize. See "AI upscaling" |
| `--min-dpi N` | `150` | Warn below this effective DPI |
| `--no-pdf`, `--no-preview` | off | Skip those outputs |
| `--combine` | off | Also write every page into one PDF, `a4print_all.pdf` |

## Fill modes

- **crop**: scales the image to cover the page and cuts what overflows. Nothing is distorted; the edges are lost.
- **extend**: scales the image to fit inside the page, then fills the empty bands by mirroring the image edges outward and blurring them. The image itself is not changed. Works best when the image's shape is close to A4. With a very wide image on a portrait page, blurred copies of the content show in the bands.
- **fit**: scales the image to fit inside the page and leaves a white border.

## What the processing does

In order:

1. Loads the image, applies EXIF rotation, puts transparency on white, and converts any embedded color profile (Adobe RGB, Display P3, CMYK) to sRGB.
2. Inpaints `--remove` regions with OpenCV (Telea method).
3. Trims plain margins if `--trim-border` is set. The margin color is the median of the outermost pixel ring, so white, beige and any flat color work.
4. Detects black-and-white images: 99% of pixels must be near-neutral (Lab chroma under 6) and almost none clearly colored, so a small colored logo keeps the image in color. B&W images are processed as one gray channel and saved with R = G = B, so a slight tint in the file can't turn into a color cast on paper.
5. Cleans up, only when needed. Small images are cleaned before enlarging, so problems aren't enlarged. Large images are shrunk first, which is faster and already hides most noise.
   - Sensor noise, measured in the flat half of the image (Immerkaer's method with edges excluded, after Tai & Yang 2008), so texture like hair or fabric isn't mistaken for noise. 30% of the original grain is blended back, so photos don't get a plastic look.
   - JPEG artifacts in heavily compressed files (quality under 80, typical for Pinterest and social media). The quality is read from the file's quantization tables, as ImageMagick does.
6. Crops or scales with Lanczos. With `--ai`, images under 240 DPI on paper go through Real-ESRGAN 4x, then Lanczos to the exact size, mixed with a plain resize per `--ai-strength`. Sharper images skip the AI, since it would add minutes for no visible gain.
7. Adds the white border (`fit`), the blurred extension (`extend`) and the `--margin`, if any.
8. Applies a gentle unsharp mask and a shadow and midtone lift to luminance only (Lab L channel). Hue and saturation are left alone. Home inkjets print shadows darker than screens show them, which is what the lift compensates for.

a4print does nothing else. It doesn't retouch faces, change colors, or recompose the image.

The PDF stores the page as a quality-95 JPEG with no chroma subsampling. That keeps it to a few MB; the PNG is the lossless copy.

## AI upscaling

Small images (most Pinterest and social media downloads print at 35-60 DPI on A4) gain the most from `--ai`. Real-ESRGAN draws sharp edges and plausible fine detail such as hair, fur and fabric, where plain resizing gives a blur.

At full strength it also smooths skin into a waxy, airbrushed look. By default a4print mixes 70% of the AI result with 30% of a plain resize. That keeps the sharp edges and brings back natural texture. Use `--ai-strength 1` for illustrations and graphics, and 0.5 if a portrait still looks too smooth.

a4print's JPEG and noise cleanup runs before the AI. Without it, Real-ESRGAN turns JPEG blocks into invented blotches.

## Print quality ratings

The resolution line rates the effective DPI on paper:

| Effective DPI | Rating |
|---|---|
| 240+ | excellent |
| 150-239 | good |
| 100-149 | fine from a distance, soft up close |
| under 100 | low: will look soft or blocky up close |

A poster on a wall is usually seen from a meter or more away, so "fine from a distance" is often good enough. For anything under 100 DPI, find a larger version of the image or use `--ai`.

## Printing tips

- In the print dialog, choose "Actual size" or 100% scale, not "Fit to page".
- Pick borderless printing if your printer supports it. If it doesn't, use `--margin 5` so the printer doesn't crop or shrink the page itself.
- Choose photo paper and the matching paper type in the driver, which matters more than any setting here.

## Folders

a4print processes every image directly inside a folder (not subfolders). Files it wrote itself (`_a4`, `_preview`, `_grid`) are skipped, so running it twice is safe. If two inputs share a name, such as `photo.jpg` and `photo.png`, the outputs keep the extension (`photo_jpg_a4.png`, `photo_png_a4.png`) so nothing is overwritten.

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The tests use generated images only (gradients, shapes, synthetic noise and textures). The Real-ESRGAN test uses a stand-in executable, so it runs without a GPU.

## Credits and license

MIT. See [LICENSE](LICENSE). The fit-on-white layout is adapted from [SECRET-GUEST/PDF_Printer](https://github.com/SECRET-GUEST/PDF_Printer) (MIT); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
