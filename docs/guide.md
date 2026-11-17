# a4print guide

Everything about using a4print. For the short version, see the [README](../README.md). For what happens inside, see [how it works](how-it-works.md).

## Install

You need Python 3.10 or newer. The simplest way is [pipx](https://pipx.pypa.io), which installs command-line tools in their own environment:

```bash
pipx install "a4print[heic] @ git+https://github.com/flawnull/a4print"
```

Don't have pipx? On a Mac with [Homebrew](https://brew.sh), run `brew install pipx` and then `pipx ensurepath`. Elsewhere, run `python3 -m pip install --user pipx` and then `python3 -m pipx ensurepath`. Open a new terminal window afterwards. Check `python3 --version` first: macOS's built-in Python can be 3.9, which is too old; Homebrew or [python.org](https://www.python.org/downloads/) gives you a current one.

This gives you the `a4print` and `a4print-web` commands. The `[heic]` part adds iPhone photo (HEIC) support; leave it out if you don't need it. To update later, run `pipx upgrade a4print`.

Without pipx, plain pip works too, ideally inside a virtual environment:

```bash
pip install "a4print[heic] @ git+https://github.com/flawnull/a4print"
```

JPEG, PNG, WebP, AVIF, TIFF, BMP and GIF work out of the box.

### From a clone, for development

```bash
git clone https://github.com/flawnull/a4print.git
```

```bash
cd a4print
```

```bash
pip install -e ".[dev,heic]"
```

### Optional: AI upscaling

`--ai` uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) through its standalone `realesrgan-ncnn-vulkan` program. Install it with one command:

```bash
a4print-setup-ai
```

This downloads the official build for your system (about 50 MB) from the Real-ESRGAN releases, checks its checksum, puts it in `~/.a4print/realesrgan`, and runs a quick test. It also sets up the photo model, RealESRNet x4plus (a 67 MB download from the same releases). The web page has a **Set up AI sharpening** button that does the same.

a4print picks the model per image. Photos get RealESRNet, which sharpens what the photo has without inventing faces or textures. Artwork and graphics get Real-ESRGAN x4plus, which draws the crispest edges and lettering. If you set up AI before version 0.14, run `a4print-setup-ai` once more to add the photo model; until then, photos are enlarged without AI.

Speed depends on your hardware. With a graphics chip (any recent Mac, or a PC with a graphics card) it takes seconds per image. Without one it falls back to the CPU and can take minutes.

If you already have Real-ESRGAN somewhere else, a4print also finds it on your `PATH` or through `A4PRINT_REALESRGAN=/path/to/realesrgan-ncnn-vulkan`.

### Optional: maximum quality

`--max-quality` runs the same two models with [PyTorch](https://pytorch.org) in full precision. Use it when `--ai` doesn't run on your computer (it needs a Vulkan graphics driver), or on a Mac or a PC with an NVIDIA card where PyTorch runs on the graphics chip.

Set it up once:

```bash
pipx inject a4print torch spandrel
```

```bash
a4print-setup-ai --max
```

The first command adds PyTorch (about 100 MB on a Mac, more on Linux and Windows). The second downloads the two models (about 130 MB), checks their checksums, and runs a quick test. Without pipx, use `pip install torch spandrel` instead of the first command. On a Mac with Apple Silicon, PyTorch runs on the graphics chip.

Why these two models: see [choosing the models](how-it-works.md#choosing-the-models).

## In your browser

Prefer not to use the terminal? Start the web page:

```bash
a4print-web
```

It opens in your browser:

1. Drop in one or more images.
2. Pick the print size: an A4 page, or a poster of 4 or 9 sheets.
3. Tick **Sharpen with AI** for small images, or click **Set up AI sharpening** the first time.
4. Click **Make print files**.

Each result says in plain words how sharp the print will be and what was done, shows a before/after preview, and has a **Download PDF** button. With two or more results, **Download all as one PDF** puts every page in one file to print in one go. The page keeps the 25 most recent results. The other settings (fill, crop position, orientation, margin, border removal) are under **More options**; the defaults suit most images.

Everything runs on your computer: the page only listens on `127.0.0.1` and refuses requests from other websites. Uploaded images and results are deleted when you stop it with Ctrl+C.

## Usage

Print an image on A4 with the automatic settings:

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
a4print group.jpg --fill crop --anchor top
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

Print a poster bigger than A4, split over several sheets (see [Posters](#posters)):

```bash
a4print poster.jpg --poster 2x2
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
| `--fill auto\|crop\|extend\|fit` | `auto` | See [Fill modes](#fill-modes) |
| `--anchor center\|top\|bottom\|left\|right` | `center` | Which part to keep when cropping |
| `--paper a4\|letter\|a3` | `a4` | Paper size. `a4print-calibrate --paper letter` makes Letter the default |
| `--crop-at X,Y` | none | Where to crop, in percent: `0,0` keeps the top left, `50,50` the center, `100,100` the bottom right |
| `--margin MM` | `0`, posters `6` | White margin around the image, for printers without borderless printing |
| `--poster COLSxROWS` | off | Split into a poster of several A4 sheets, e.g. `2x2` or `3x3` |
| `--overlap MM` | `10` | How much neighboring poster sheets overlap, for gluing |
| `--trim-border` | off | Remove plain outer margins before fitting |
| `--trim-tolerance N` | `18` | How far (0-255) a margin pixel may differ from the margin color |
| `--remove x,y,w,h` | none | Erase a rectangle by inpainting, in pixels or percent. Repeatable |
| `--grid` | off | Only save `name_grid.jpg`, the image with labeled coordinates |
| `--denoise S` | `1.0` | Noise and JPEG artifact cleanup strength. `0` turns it off. Runs only when needed |
| `--sharpen S` | `0.5` | Unsharp mask amount. `0` turns it off |
| `--sharpen-radius PX` | `2.0` | Unsharp mask radius in output pixels |
| `--shadows S` | `0.03` | Shadow lift, 0 to 0.2 (0.03 lifts black by about 8 levels) |
| `--midtones S` | `0.05` | Midtone lift. `0` turns it off |
| `--no-grain` | off | No print grain on photos enlarged more than 2x. See [print grain](how-it-works.md#ai-upscaling) |
| `--no-enhance` | off | Skip denoise, sharpening, grain and tone lift |
| `--ai` | off | Upscale with AI: RealESRNet for photos, Real-ESRGAN for artwork |
| `--ai-model NAME` | per image | Use this Real-ESRGAN (ncnn) model for every image |
| `--max-quality` | off | The same AI models through PyTorch, in full precision |
| `--ai-strength S` | `1` | Mix of the AI result with a plain resize. See [AI upscaling](how-it-works.md#ai-upscaling) |
| `--min-dpi N` | `150` | Warn below this effective DPI |
| `--no-pdf`, `--no-preview` | off | Skip those outputs |
| `--combine` | off | Also write every page into one PDF, `a4print_all.pdf` |

## Fill modes

- **auto** (default) picks for you:
  - Artwork with a plain border, like most designed posters, product shots and scans, is kept whole. The page is extended in the image's own border color, with matching paper grain, so frames and credit lines aren't cut off.
  - Photos are cropped to fill the page when that removes 20% or less.
  - Anything else is kept whole on white.
- **crop**: scales the image to cover the page and cuts what overflows. Nothing is distorted; the edges are lost.
- **extend**: scales the image to fit inside the page, then fills the empty bands by mirroring the image edges outward and blurring them. The image itself is not changed. Works best when the image's shape is close to A4. With a very wide image on a portrait page, blurred copies of the content show in the bands.
- **fit**: scales the image to fit inside the page and leaves a white border.

## Photo sheets

`--per-page` puts several images on each page, in a grid of equal slots with dashed cutting guides in the gaps between them:

```bash
a4print beach.jpg dinner.jpg dog.jpg sunset.jpg --per-page 4
```

```bash
a4print ~/Pictures/holiday --per-page 6 --paper letter
```

| `--per-page` | Grid | Each photo on A4 |
|---|---|---|
| 2 | 1 x 2 | about 20 x 14 cm |
| 4 | 2 x 2 | about 10 x 14 cm |
| 6 | 2 x 3 | about 10 x 9 cm |
| 8 | 2 x 4 | about 10 x 7 cm |
| 9 | 3 x 3 | about 6.5 x 9 cm |
| 12 | 3 x 4 | about 6.5 x 7 cm |

Each image goes through the same steps as a full page: the automatic fit, cleanup, AI sharpening if you ask for it, and print tuning. A landscape photo in a portrait slot is turned 90 degrees when that crops less; turn the cut-out print back by hand. All sheets go into one PDF, `a4print_sheets_a4.pdf`, with a preview of the first sheet. The white margin around the page is 5 mm (change it with `--margin`).

On the web page, open **Choose what gets cropped** to see the page-shaped frame on your picture and drag it (or use the arrow keys) to the part you want; the print then fills the page with it. This works for single pages.

When a4print crops a photo by itself, it keeps faces in the frame: the crop moves just far enough that faces, with some room around them, are not cut. `--anchor` or `--crop-at` switch that off.

On the web page, open **Erase a date stamp or mark**, drag a box over each thing to erase on the picture, and make the print as usual. This works with one image at a time.

For photo sheets on the web page, choose **Photo sheet** as the print size, pick how many per page, and drop in all the images at once.

## Posters

`--poster` makes a poster bigger than A4 and splits it over sheets you print at home, then trim and glue together.

| Sheets | Finished size (landscape) |
|---|---|
| `2x2` | 56 x 39 cm, about A2 |
| `3x2` | 84 x 39 cm, a long banner |
| `3x3` | 84 x 57 cm, about A1 |
| `4x4` | 111 x 76 cm |

a4print picks portrait or landscape sheets, whichever gives a poster shape closest to the image's, so the least gets cropped. `--orientation` sets the direction of the whole poster. `--fill`, `--anchor`, the cleanup and `--ai` all work as for single pages.

It writes `name_poster.pdf`, one A4 page per sheet, and `name_poster_preview.jpg`, which shows the sheet grid over the image.

Each sheet has:

- **A white margin** (6 mm by default, change it with `--margin`). Home printers can't print to the edge, so this keeps the image off the part they'd cut or shrink.
- **A label** in the bottom margin, like `B2 (row 2, column 2 of 3x3)`, saying which margins to cut off. Rows run A, B, C from the top; columns 1, 2, 3 from the left.
- **A dashed guide line** inside the overlap strip on its right and bottom, where the next sheet's edge goes.
- **Cut marks** in the margin at the image edges you trim.

To assemble:

1. Print the PDF at 100% scale ("Actual size"), not "Fit to page".
2. Lay the sheets out in order, A1 at top left.
3. On every sheet except A1, cut off the margins its label names (left, top or both), along the cut marks.
4. Working row by row from A1, glue each sheet over its neighbor so its cut edge sits on the dashed line. The overlap hides the line.

The resolution line reports DPI on the finished poster, which is lower than on one A4 page. A poster is usually seen from further away, so 100 DPI or more looks fine on a wall. Very large posters (over 75 megapixels, like 4x4 and up) are rendered at a lower DPI and each sheet is scaled up to 300 DPI when cut out, which keeps memory use reasonable.

## Printer calibration

Home printers differ: many print midtones darker than the screen because ink spreads into the paper, and some lose detail in deep shadows. a4print lifts both a little by default. To tune that for your printer, once:

```bash
a4print-calibrate
```

This writes `a4print_test_page.pdf`. Print it at 100% on the paper you use, and answer two questions on the sheet:

1. Which is the first numbered square you can tell apart from the black band?
2. From arm's length, which gray patch (A-E) matches the fine stripes next to it best?

Then save the answers, for example square 4 and patch C:

```bash
a4print-calibrate --black 4 --gray C
```

To make another paper size your default (US Letter, say), add `--paper letter` to any `a4print-calibrate` command; the test page is then printed on that paper too.

From then on every print uses those values for `--shadows` and `--midtones` (flags still override them; `a4print-calibrate --reset` goes back to the standard values). The web page has the same steps under **More options**.

Judge the sheet on paper, not on screen. The stripe test works because black-and-white stripes reflect half the light whatever the printer does with ink, so a gray patch that matches them prints at the right lightness. Screens and image viewers average the stripes differently, which makes them look darker than they print.

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

- In the print dialog, choose "Actual size" or 100% scale, not "Fit to page". a4print's PDFs ask for this already, and Acrobat and most viewers follow it, but check the dialog once.
- Pick borderless printing if your printer supports it. If it doesn't, use `--margin 5` so the printer doesn't crop or shrink the page itself.
- Choose photo paper and the matching paper type in the driver, which matters more than any setting here.

## Common problems

| Problem | Cause and fix |
|---|---|
| The print is smaller than the paper, with a white border | The print dialog is set to "Fit to page" or "Shrink to fit". Choose "Actual size" or 100%. |
| The edges of the image are cut off by the printer | Your printer can't print borderless. Use `--margin 5`. |
| The print looks soft or blocky | The resolution line says "low". Find a larger version of the image, or use `--ai`. |
| The print is darker than the screen | Normal for home inkjets. Calibrate once for your printer (see [Printer calibration](#printer-calibration)). |
| A black-and-white photo prints with a color tint | Use the printer driver's "grayscale" or "black ink only" setting. a4print's output is neutral gray, but some drivers mix color inks. |
| "incomplete: the download or copy was cut off" | The file is damaged, often from an interrupted download. Download it again. |
| A warning says cropping removes most of the image | The image's shape is far from A4. Use `--fill fit`, `--fill extend`, or a poster shape like `3x2`. |

## Inputs

Anything Pillow can open works: photos with EXIF rotation, PNGs with transparency (placed on white), 16-bit and 32-bit float scans, 1-bit black-and-white scans, CMYK files, and the first frame of animated GIFs. Embedded color profiles are converted to sRGB.

Very large images (over 100 megapixels, like stitched panoramas) are reduced while loading, which is fast and still leaves more detail than any page or poster needs. Files over 400 megapixels are refused, because decoding them would need gigabytes of memory.

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
