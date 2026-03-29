# Пример живой SDXL-раскладки на телефоне

Этот файл документирует **реальный снимок каталога**, снятый **2026-03-29** с подключённого телефона через `adb shell su -c ...`.

Он отражает **историческую rooted-раскладку деплоя**. Текущий layout по умолчанию, документированный в репозитории, — это `/sdcard/Download/sdxl_qnn`.

Файл включён как пример того, что пайплайн уже реально создавал на устройстве.  
Крупные бинарные файлы **не** хранятся в репозитории; этот файл — только инвентарь и справка.

## Базовый путь

`/data/local/tmp/sdxl_qnn`

## Какие верхнеуровневые элементы наблюдались

- `bin/`
- `context/`
- `lib/`
- `model/`
- `output/`
- `outputs/`
- `phone_gen/`
- `runtime_work_gen/`
- `python3` (симлинк на Termux Python)
- helper-скрипты и QNN JSON-конфиги

## Какие context binaries наблюдались

```text
clip_l.serialized.bin.bin
clip_g.serialized.bin.bin
unet_encoder_fp16.serialized.bin.bin
unet_decoder_fp16.serialized.bin.bin
unet_lightning8step.serialized.bin.bin
vae_decoder.serialized.bin.bin
```

## Какие model libraries наблюдались

```text
libclip_l.so
libclip_g.so
libunet_lightning_fp16.so
libunet_lightning8step.so
libvae_decoder.so
```

## Какие phone-side generator файлы наблюдались

```text
/data/local/tmp/sdxl_qnn/phone_gen/generate.py
/data/local/tmp/sdxl_qnn/phone_gen/tokenizer/vocab.json
/data/local/tmp/sdxl_qnn/phone_gen/tokenizer/merges.txt
```

## Какие дополнительные live-артефакты были на устройстве

В живом снимке телефона также есть файлы, полезные для разработки и отладки, но **не входящие в минимальную раскладку**:

- `ctxgen_fp16.log`
- `ctxgen_fp16_exit.txt`
- `htp_backend_ext_config_*.json`
- `htp_backend_extensions_*.json`
- `runtime_work_gen/`
- helper shell-скрипты вроде `tmp_gen_cfg35.sh`

## Зачем нужен этот файл

- чтобы показать конкретный результат, который tooling уже действительно создавал;
- чтобы помочь сравнить документированную минимальную раскладку с реальным phone-side deploy;
- чтобы не тащить multi-gigabyte binaries в git.
