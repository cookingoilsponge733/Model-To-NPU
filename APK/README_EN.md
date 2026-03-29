# Model-to-NPU Android Application

**Languages:** [Русский](README.md) | [English](README_EN.md)

**Current target:** SDXL

> [!WARNING]
> The Android app and the surrounding scripts are still being re-tested together with the rest of the repository. The app is already usable, but the documentation remains intentionally conservative until the next full validation pass is complete.

## Status: Working / In active validation

This APK is used to generate images directly on the phone through the Qualcomm NPU.  
The currently implemented target is **SDXL Lightning**.  
After the model files are deployed, the workflow is intended to be **fully standalone** — no PC is needed for normal generation.

## Architecture

```text
┌─────────────────────────────────┐
│         MainActivity            │
│  ┌───────────────────────────┐  │
│  │ Prompt + Negative Prompt  │  │
│  │ Seed / Steps / CFG        │  │
│  ├───────────────────────────┤  │
│  │     Image Preview         │  │
│  ├───────────────────────────┤  │
│  │  Generate / Stop / Save   │  │
│  └───────────────────────────┘  │
│         ⚙️ SettingsActivity     │
│  ┌───────────────────────────┐  │
│  │ Model path               │  │
│  │ Python path / command    │  │
│  │ Verify / Reset           │  │
│  └───────────────────────────┘  │
├─────────────────────────────────┤
│   Shared Downloads path         │
│   → configurable Python3        │
│   → phone_generate.py           │
│   (BPE tokenizer + scheduler)   │
├─────────────────────────────────┤
│       QNN Runtime (HTP/NPU)     │
│  ┌───────┬───────┬───────────┐  │
│  │CLIP-L │CLIP-G │Split UNet │  │
│  │ FP16  │ FP16  │  FP16     │  │
│  ├───────┴───────┤enc + dec  │  │
│  │    VAE FP16   │           │  │
│  └───────────────┴───────────┘  │
└─────────────────────────────────┘
```

## Requirements

- **Phone**: Snapdragon 8 Elite (SM8750) or another device with a compatible Qualcomm NPU
- **Root**: not required for the current default path layout
- **Termux**: Python 3.13+, numpy, Pillow, `termux-setup-storage`
- **Android 11+**: shared Downloads access may require “all files access”
- **QNN**: context binaries must already be deployed (see `scripts/deploy_to_phone.py`)

## Installation

### 1. Prepare Termux

```bash
pkg install python python-numpy python-pillow
termux-setup-storage
```

### 2. Deploy model files from PC

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/contexts \
  --phone-base /sdcard/Download/sdxl_qnn \
  --qnn-lib-dir /path/to/qnn/lib \
  --qnn-bin-dir /path/to/qnn/bin
```

### 3. Build and install the APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Features

- prompt and negative prompt input;
- steps control (1–20);
- CFG control (1.0–7.0);
- reproducible seed handling;
- percentile-based contrast stretching;
- save-to-gallery support;
- stop button for ongoing generation;
- progress bar for CLIP → UNet → VAE stages;
- configurable paths through ⚙️ Settings.

## Model settings

Through ⚙️ in the ActionBar you can:

- change the model directory path (default: `/sdcard/Download/sdxl_qnn`);
- change the Python command/path (default: `python3`);
- verify the required files on the phone;
- reset settings back to detected defaults.

## Performance

| Stage | Time |
| ----- | ---- |
| CLIP-L + CLIP-G | ~2.3 s |
| UNet encoder + decoder × 8 steps (no CFG) | ~113 s |
| UNet encoder + decoder × 8 steps (CFG=3.5) | ~236 s |
| VAE | ~10 s |
| **Total (no CFG)** | **~126 s** |
| **Total (CFG=3.5)** | **~251 s** |

## Files on the phone

```text
/sdcard/Download/sdxl_qnn/      (or your custom Settings path)
├── context/
│   ├── clip_l.serialized.bin.bin
│   ├── clip_g.serialized.bin.bin
│   ├── unet_encoder_fp16.serialized.bin.bin
│   ├── unet_decoder_fp16.serialized.bin.bin
│   └── vae_decoder.serialized.bin.bin
├── phone_gen/
│   ├── generate.py
│   └── tokenizer/
│       ├── vocab.json
│       └── merges.txt
├── lib/    (QNN runtime libs)
├── model/  (optional extra model libs in some flows)
├── bin/    (qnn-net-run)
└── outputs/
```

## Technical notes

- the APK launches `phone_generate.py` without `su`, through a normal shell and a configurable Python command;
- the default layout uses `/sdcard/Download/sdxl_qnn`;
- CFG above `1.0` is noticeably slower because the phone runtime still needs both cond and uncond denoising branches; with a split UNet that translates into substantially more encoder/decoder work per step even after batching optimizations;
- stdout is parsed in real time to display progress;
- the resulting PNG is loaded through `BitmapFactory.decodeFile()`;
- gallery saving uses the Android `MediaStore` API.
