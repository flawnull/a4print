# a4print

Prepare any image (a photo, a poster, a Pinterest download) for borderless printing on A4 paper with a home printer.

For each image, a4print writes:

- `name_a4.png`: 2480 x 3508 px (or 3508 x 2480 landscape), 300 DPI, sRGB profile embedded
- `name_a4.pdf`: one A4 page, image edge to edge, no margins
- `name_preview.jpg`: before and after, side by side

It also reports the image's effective resolution at A4 and warns when it drops below 150 DPI.

## Install

You need Python 3.9 or newer.

```bash
git clone https://github.com/<you>/a4print.git
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

### Optional: AI upscaling

`--ai` uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) through its standalone `realesrgan-ncnn-vulkan` tool. It runs on most GPUs via Vulkan and needs no Python packages.

1. Download `realesrgan-ncnn-vulkan-*-windows/macos/ubuntu.zip` from the [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases).
2. Unzip it. Keep the `models` folder next to the executable.
3. Put that folder on your `PATH`, or point a4print at the executable:

```bash
export A4PRINT_REALESRGAN=/path/to/realesrgan-ncnn-vulkan
```

If `--ai` is used and the tool can't be found, a4print stops and tells you how to install it.

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

Remove a date stamp and a credit line. Coordinates are `x,y,width,height` in pixels of the original image (any image viewer shows the cursor position):

```bash
a4print old.jpg --remove 1820,1300,260,60 --remove 20,20,300,40
```

Upscale a small image with Real-ESRGAN:

```bash
a4print small.jpg --ai
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
  resolution  342 DPI effective at A4
  denoise     applied, noise level 3.4
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
| `--trim-border` | off | Remove plain outer margins before fitting |
| `--trim-tolerance N` | `18` | How far (0-255) a margin pixel may differ from the margin color |
| `--remove x,y,w,h` | none | Erase a rectangle by inpainting. Repeatable |
| `--denoise S` | `1.0` | Denoise strength. `0` turns it off. Skipped on clean images |
| `--sharpen S` | `0.5` | Unsharp mask amount. `0` turns it off |
| `--sharpen-radius PX` | `2.0` | Unsharp mask radius in output pixels |
| `--shadows S` | `0.03` | Shadow lift, 0 to 0.2 (0.03 lifts black by about 8 levels) |
| `--midtones S` | `0.05` | Midtone lift. `0` turns it off |
| `--no-enhance` | off | Skip denoise, sharpening and tone lift |
| `--ai` | off | Upscale with Real-ESRGAN instead of Lanczos |
| `--ai-model NAME` | `realesrgan-x4plus` | Real-ESRGAN model name |
| `--min-dpi N` | `150` | Warn below this effective DPI |
| `--no-pdf`, `--no-preview` | off | Skip those outputs |

## Fill modes

- **crop**: scales the image to cover the page and cuts what overflows. Nothing is distorted; the edges are lost.
- **extend**: scales the image to fit inside the page, then fills the empty bands by mirroring the image edges outward and blurring them. The image itself is not changed. Works best when the image's shape is close to A4. With a very wide image on a portrait page, blurred copies of the content show in the bands.
- **fit**: scales the image to fit inside the page and leaves a white border.

## What the processing does

In order:

1. Loads the image, applies EXIF rotation, puts transparency on white, and converts any embedded color profile (Adobe RGB, Display P3, CMYK) to sRGB.
2. Inpaints `--remove` regions with OpenCV (Telea method).
3. Trims plain margins if `--trim-border` is set. The margin color is the median of the outermost pixel ring, so white, beige and any flat color work.
4. Detects black-and-white images (99% of pixels have Lab chroma under 6). These are processed as one gray channel and saved with R = G = B, so a slight tint in the file can't turn into a color cast on paper.
5. Measures noise (Immerkaer's method) and runs a light non-local-means denoise only if the image is noisy. This happens before scaling so noise isn't enlarged.
6. Crops or scales with Lanczos, or with Real-ESRGAN 4x followed by Lanczos to the exact size when `--ai` is set and the image needs enlarging.
7. Applies a gentle unsharp mask and a shadow and midtone lift to luminance only (Lab L channel). Hue and saturation are left alone. Home inkjets print shadows darker than screens show them, which is what the lift compensates for.

a4print does nothing else. It doesn't retouch faces, change colors, or recompose the image.

The PDF stores the page as a quality-95 JPEG with no chroma subsampling. That keeps it to a few MB; the PNG is the lossless copy.

## Printing tips

- In the print dialog, choose "Actual size" or 100% scale, not "Fit to page".
- Pick borderless printing if your printer supports it. Otherwise the printer adds its own small margin.
- Choose photo paper and the matching paper type in the driver, which matters more than any setting here.

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The tests use generated images only (gradients, shapes, synthetic noise). The Real-ESRGAN test uses a stand-in executable, so it runs without a GPU.

## Credits and license

MIT. See [LICENSE](LICENSE). The fit-on-white layout is adapted from [SECRET-GUEST/PDF_Printer](https://github.com/SECRET-GUEST/PDF_Printer) (MIT); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
