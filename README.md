# 🧠 Model-To-NPU - Run SDXL on Snapdragon phones

[![Download](https://img.shields.io/badge/Download-Model--To--NPU-8A2BE2?style=for-the-badge&logo=github)](https://github.com/cookingoilsponge733/Model-To-NPU/releases)

## 📥 Download

Visit this page to download: [Model-To-NPU Releases](https://github.com/cookingoilsponge733/Model-To-NPU/releases)

## ✅ What this app does

Model-To-NPU lets you run SDXL image generation on Snapdragon phones that support Qualcomm Hexagon NPU features. It uses an ONNX to QNN pipeline so the model can run on the device instead of sending jobs to a cloud service.

Use it to:

- Generate images on your phone
- Keep image work on device
- Use the Qualcomm NPU path for faster local runs
- Work with SDXL models prepared for Android use
- Run the app through a simple Android setup

## 🖥️ Before you start

Use a phone or test device that meets these basic needs:

- Android 12 or newer
- Snapdragon chip with Hexagon NPU support
- At least 8 GB RAM
- 6 GB free storage for the app and model files
- A stable battery charge or charger during setup
- Internet access for the first download

If you plan to prepare files from a Windows PC, use it only for download and file transfer. The app itself runs on Android.

## 📦 What you need

Have these items ready:

- Your Android phone
- A USB cable or file transfer method
- The model files from the release page
- A file manager app on your phone
- Termux if the setup package uses command-based steps

## 🚀 Getting started

1. Open the [Model-To-NPU Releases](https://github.com/cookingoilsponge733/Model-To-NPU/releases) page.
2. Find the latest release.
3. Download the Android package or setup files listed there.
4. If the release includes a model bundle, download that too.
5. Copy the files to your phone if you downloaded them on Windows.

## 📲 Install on Android

1. Open the downloaded APK or install package on your phone.
2. If Android asks for permission to install from this source, allow it.
3. Finish the install.
4. Open the app from your home screen or app list.
5. Give the app storage access if it asks for it.

If the release uses a folder-based setup, keep the app files in one place and do not rename them.

## 🗂️ Set up the model files

1. Open your file manager.
2. Create a folder for the app files if the release does not already include one.
3. Copy the SDXL model files into the folder shown in the release notes.
4. Keep the ONNX, QNN, and config files together.
5. Make sure the file names match the names listed in the release package.

A clean folder layout helps the app find the model without errors.

## ▶️ Run the app

1. Open Model-To-NPU.
2. Select the model folder if the app asks for it.
3. Choose your prompt.
4. Pick an image size if the app offers that option.
5. Start generation and wait for the result.

The first run may take longer while the app loads files and prepares the NPU path.

## 🧩 Recommended settings

Use these settings for a smoother first run:

- Image size: 1024 x 1024
- Steps: 20 to 30
- Batch size: 1
- Prompt length: short to medium
- Storage mode: internal storage

If the app offers quality levels, start with the default choice before changing anything.

## 🛠️ If the app does not start

Try these checks in order:

1. Confirm that your phone uses a supported Snapdragon chip.
2. Make sure all model files are in the correct folder.
3. Check that the app has storage access.
4. Restart the phone and try again.
5. Remove and reinstall the app if the install looks damaged.
6. Re-download the release files if they seem incomplete.

If the app closes during startup, free up storage and close other large apps first.

## 📁 File layout example

Use a simple folder layout like this:

- Model-To-NPU/
  - app files
  - model/
    - sdxl.onnx
    - qnn files
    - config files
  - outputs/
    - generated images

Keep output images in a separate folder so they are easy to find later.

## 🔧 Common terms

- **SDXL**: a large image model used for text-to-image generation
- **ONNX**: a model format used by many AI tools
- **QNN**: Qualcomm Neural Network tools for device inference
- **Hexagon NPU**: the neural engine in supported Snapdragon chips
- **Termux**: a terminal app for Android that can help with setup tasks

## 📌 Good use cases

Model-To-NPU fits these tasks:

- Local image generation on a phone
- Testing SDXL on Snapdragon hardware
- Running an offline image workflow
- Checking NPU-based performance on Android
- Moving model work away from cloud tools

## 🧾 Basic workflow

1. Download the release files.
2. Install the Android app or setup package.
3. Copy the model files to the phone.
4. Open the app.
5. Enter a prompt.
6. Generate the image.
7. Save the output image to your gallery or output folder

## 🔐 Privacy and device use

Since the app runs on your phone, your prompts and image work stay on the device during normal use. That makes it a fit for local AI work when you do not want to send data to another service.

## 📄 Release page

Get the latest files here: [https://github.com/cookingoilsponge733/Model-To-NPU/releases](https://github.com/cookingoilsponge733/Model-To-NPU/releases)

## 🧭 Tips for a smooth first run

- Start with one image at a time
- Use short prompts first
- Keep the phone plugged in
- Close background apps
- Use enough free storage before you begin
- Keep all release files in one folder on your PC before copying them to Android