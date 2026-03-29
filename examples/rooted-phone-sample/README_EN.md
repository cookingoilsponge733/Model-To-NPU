# Rooted phone-side sample from `/data/local/tmp/sdxl_qnn`

This folder contains a curated set of **small real files** pulled from a rooted phone deployment at:

`/data/local/tmp/sdxl_qnn`

Its goal is simple: if someone cannot fully rebuild the pipeline yet, they can still inspect a real rooted layout, look at helper scripts, see real QNN config files, open actual log files, and view a couple of generated output images.

## What is included

The sample currently contains:

- `ctxgen_fp16.log`
- `ctxgen_fp16_exit.txt`
- `htp_backend_ext_config_*.json`
- `htp_backend_extensions_*.json`
- `run_ctxgen_fp16.sh`
- `tmp_gen_*.sh`
- `tmp_run_gen.sh`
- `apk_test_v3.png`
- `phone_cfg35_s8.png`

## What is intentionally NOT included

To keep this repository lightweight and legally cleaner, this folder does **not** include:

- model weights;
- LoRA weights;
- QNN context binaries;
- large runtime libraries;
- copied full model `.so` payloads.

That means this folder is for **inspection and education**, not for reproducing the full runtime by itself.

## Source model and attribution

The rooted sample was produced while working with the following model setup:

- **Base checkpoint:** [WAI Illustrious SDXL v1.60](https://civitai.com/models/827184/wai-illustrious-sdxl?modelVersionId=2514310)
- **LoRA:** [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)

### Attribution note

This repository and this sample folder do **not** claim ownership of the base model, LoRA, or their original artistic content.

- All rights to the original model/checkpoint belong to its creator(s) and respective rightsholders.
- All rights to the LoRA belong to its creator(s) and respective rightsholders.
- Use the original distribution pages above to obtain the actual model files and to review their licenses and usage terms.

This folder only redistributes:

- small helper/config files produced during experimentation;
- minimal logs;
- generated example images;
- documentation for educational and research-oriented reference.

## Why this folder exists

This sample is useful for people who want to:

- understand what a rooted phone-side deployment looked like in practice;
- compare rooted layout vs. the newer shared-storage layout;
- inspect the kind of helper scripts and configs that ended up on the device;
- see output examples before they manage to rebuild the full pipeline themselves.

## Historical status

Important: this is a **historical rooted sample**.

The repository now documents `/sdcard/Download/sdxl_qnn` as the default phone-side layout for the newer non-root-oriented flow.

So this folder should be read as:

- a real rooted snapshot;
- a reproducibility aid;
- an educational example of what existed on-device during development.

## Notes about the included files

### `ctxgen_fp16.log` / `ctxgen_fp16_exit.txt`

These files show a real phone-side context generation attempt that ended with exit code `137`, i.e. the process was killed. This is exactly the kind of practical failure that is worth documenting publicly.

### `htp_backend_*.json`

These are real backend configuration variants captured from the device, including gentler and more aggressive profiles used during experimentation.

### `tmp_gen_*.sh` and `tmp_run_gen.sh`

These are not polished public entrypoints. They are tiny helper scripts from real experimentation and are included as examples of the ad-hoc commands used during development.

### `*.png`

The included PNG files are example outputs generated on-device. They are meant to demonstrate that the pipeline produced actual images, not just logs and partially converted graphs.

## Related files in the repository

- historical rooted layout inventory: [`../phone-sdxl-qnn-layout.md`](../phone-sdxl-qnn-layout.md)
- Russian version of this document: [`README_RU.md`](README_RU.md)
- English lessons learned: [`../../SDXL/LESSONS_LEARNED.md`](../../SDXL/LESSONS_LEARNED.md)
- Russian lessons learned: [`../../SDXL/LESSONS_LEARNED_RU.md`](../../SDXL/LESSONS_LEARNED_RU.md)
