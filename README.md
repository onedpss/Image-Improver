# Image Improver

Python program for image enhancement via **Real-ESRGAN ncnn-vulkan**

1. Maximum quality without PyTorch
2. Fast mode without AI (interpolation + light sharpness).

## Requirements

- Python 3.11+ (Newer ones will do.)
- [tkinter](https://wiki.python.org/moin/TkInter) (on Linux, the distribution package is usually `tk')

## Installation

```bash
cd "Image Improver"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## AI upscaling (Recommended)

1. Download the **realesrgan-ncnn-vulkan** archive for your OS from [Real-ESRGAN release page](https://github.com/xinntao/Real-ESRGAN/releases) (search for files like `realesrgan-ncnn-vulkan-*-windows.zip` / `*-ubuntu.zip` / `*-macos.zip`).
2. Unzip the contents to the `**vendor/`** folder in the root of the project (next to `Image_Improver.py` ), so that the `models/` folder is located next to the executable file.

Alternatively, in the application, click **"Download NCNN..."** — the archive will be downloaded and unpacked to `vendor/` automatically (Internet is needed). The official zip also contains sample `input`* images, `.mp4` demos, and `README_ubuntu.md` (Linux); the app removes those after auto-extract so only the binary and `models/` stay. If you unpack manually, you can delete those files yourself.

You need **Vulkan** and a GPU driver; errors are possible on some systems without a GPU, then use the **"Fast (without AI)"** mode.

## Launch

```bash
python Image_Improver.py
```

## Limitations

- NCNN can produce small tile seams compared to the PyTorch version (this is a feature of tile inference).
- The repository intentionally does not contain weights and binaries — they are pulled up separately or on the first download.

