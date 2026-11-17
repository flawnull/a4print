# Third-party notices

The fit-on-white page layout in `a4print/layout.py` is adapted from
[SECRET-GUEST/PDF_Printer](https://github.com/SECRET-GUEST/PDF_Printer), used under this license:

```
MIT License

Copyright (c) 2024 nsʇnɾ ʇǝɹɔǝs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## AI models downloaded at setup

a4print doesn't include these files. `a4print-setup-ai` and `a4print-setup-ai --max` download them from their authors' GitHub releases when you ask for them.

- **Real-ESRGAN** (realesrgan-ncnn-vulkan program and the RealESRGAN_x4plus and RealESRNet_x4plus models), by Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan. BSD-3-Clause license. https://github.com/xinntao/Real-ESRGAN. `a4print-setup-ai` converts the RealESRNet_x4plus weights to the ncnn format on your computer; the weights themselves are unchanged apart from being stored at half precision.
- **spandrel**, the library that loads the models, by the chaiNNer Organization. MIT license. https://github.com/chaiNNer-org/spandrel

## YuNet face detector (bundled)

`a4print/data/face_detection_yunet_2023mar.onnx` is the YuNet face detection model from the OpenCV model zoo (https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), included unmodified. a4print uses it only to center the preview's detail view on a face. License:

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
