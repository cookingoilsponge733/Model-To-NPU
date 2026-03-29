#!/data/data/com.termux/files/usr/bin/python3
"""Quick VAE output diagnostics."""
import numpy as np

RAW = "/data/local/tmp/sdxl_qnn/phone_gen/work/vae/out/Result_0/image_native.raw"

raw = np.fromfile(RAW, np.float16).astype(np.float32)
print(f"size={raw.size}  min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f} std={raw.std():.4f}")

img = raw.reshape(1024, 1024, 3)
img01 = np.clip(img / 2 + 0.5, 0, 1)
print(f"After /2+0.5: min={img01.min():.4f} max={img01.max():.4f} mean={img01.mean():.4f} std={img01.std():.4f}")

lo, hi = np.percentile(img01, [0.5, 99.5])
print(f"Percentile 0.5%={lo:.4f}  99.5%={hi:.4f}  range={hi-lo:.4f}")

# local noise measurement
for y, x in [(400, 400), (200, 600), (512, 512)]:
    patch = img01[y:y+10, x:x+10, :]
    print(f"Patch ({y},{x}) 10x10 std: R={patch[:,:,0].std():.5f} G={patch[:,:,1].std():.5f} B={patch[:,:,2].std():.5f}")

# fp16 step analysis - what's the typical gap between adjacent fp16 values?
ch0 = img01[:,:,0].flatten()
sorted_vals = np.sort(np.unique(ch0))
if len(sorted_vals) > 100:
    gaps = np.diff(sorted_vals[:200])
    print(f"FP16 gaps in ch0 (first 200 unique): min={gaps.min():.6f} max={gaps.max():.6f} mean={gaps.mean():.6f}")
print(f"Unique float values in ch0: {len(sorted_vals)}")

# Save no-stretch version
img_u8_nostretch = (img01 * 255).astype(np.uint8)
from PIL import Image
Image.fromarray(img_u8_nostretch).save("/data/local/tmp/sdxl_qnn/outputs/phone_test1_nostretch.png")
print("Saved no-stretch version")

# Save stretched version for comparison
if hi - lo > 0.05:
    img_st = np.clip((img01 - lo) / (hi - lo), 0, 1)
    img_u8_st = (img_st * 255).astype(np.uint8)
    Image.fromarray(img_u8_st).save("/data/local/tmp/sdxl_qnn/outputs/phone_test1_stretch.png")
    print("Saved stretched version")

# Save median-filtered version
# Simple 3x3 median without scipy
def median_filter_3x3(arr):
    """Simple 3x3 median filter via numpy."""
    h, w, c = arr.shape
    padded = np.pad(arr, ((1,1),(1,1),(0,0)), mode='edge')
    out = np.empty_like(arr)
    for dy in range(3):
        for dx in range(3):
            if dy == 0 and dx == 0:
                stack = padded[dy:dy+h, dx:dx+w, :].reshape(h, w, c, 1)
            else:
                stack = np.concatenate([stack, padded[dy:dy+h, dx:dx+w, :].reshape(h, w, c, 1)], axis=3)
    out = np.median(stack, axis=3)
    return out

img_med = median_filter_3x3(img01)
img_u8_med = (img_med * 255).astype(np.uint8)
Image.fromarray(img_u8_med).save("/data/local/tmp/sdxl_qnn/outputs/phone_test1_median.png")
print("Saved median-filtered version")

print("Done")
