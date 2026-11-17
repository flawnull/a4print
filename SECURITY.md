# Security

a4print runs on your own computer. This page says what it does to stay safe, and how to report a problem.

## Reporting a vulnerability

Please report security problems privately, through GitHub's **Report a vulnerability** button on the repository's Security tab, rather than in a public issue. Include the steps to reproduce it and the a4print version (`a4print --version`).

## Security model

**Images are untrusted input.** Any file you open could be crafted.

- Image decoding uses Pillow, OpenCV and pillow-heif, with minimum versions that include fixes for known decoder vulnerabilities (CVE-2024-28219, CVE-2023-4863).
- Files over 400 megapixels are refused before decoding, so a small crafted file can't make a4print allocate gigabytes. Larger-than-needed images are reduced while loading.
- Truncated or malformed files fail with an error; nothing is executed or written outside the output folder.

**The web page (`a4print-web`) is reachable only from this computer.**

- It listens on `127.0.0.1` only, never on the network.
- It checks the `Host` header, so a website can't reach it through DNS rebinding.
- Every request that does something (process, set up AI, save settings) needs a custom header that other websites can't send without a CORS preflight, which the server never approves. Requests with a foreign `Origin` are refused as well.
- Pages and files are sent with `Content-Security-Policy` (`frame-ancestors 'none'`), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `Cache-Control: no-store`.
- Uploaded file names are reduced to safe characters; results are served only from their own random job folder, with no path traversal (covered by tests).
- Uploads and results live in a private temporary folder (mode 700), keep at most the 25 most recent jobs, and are deleted when the server stops.
- The page renders all text with `textContent`, never as HTML.

**Downloads are pinned.** `a4print-setup-ai` downloads Real-ESRGAN and the AI models only from their authors' GitHub releases, over HTTPS, and checks each file against a SHA-256 written into a4print's code. A file that doesn't match is deleted and not used. Model checksums are checked again each time a model is loaded.

**Model files can't run code.** Models are loaded with spandrel, which only unpickles model weights. When `a4print-setup-ai` converts the photo model for `--ai`, it reads the PyTorch file with its own restricted unpickler, which accepts only plain dictionaries and tensors and refuses anything else, and checks that every tensor stays inside its data. It runs only after the download's checksum matched, and the converted file is checked against a second pinned checksum. The `max` extra requires PyTorch 2.6 or newer (CVE-2025-32434).

**Programs are started without a shell.** Real-ESRGAN is run with an argument list, so file names can't inject commands.

**Repository.** GitHub Actions runs with a read-only token, actions are pinned to commit SHAs, and Dependabot proposes dependency updates weekly. The tests use only generated or public-domain images.

## Responsible use

Print only images you have the right to use. The `--remove` option is meant for date stamps and similar marks on your own photos, or on images whose owner allows editing. Removing watermarks or credits from someone else's work can break copyright law.
