#!/system/bin/sh
DR=/data/local/tmp/sdxl_qnn
export LD_LIBRARY_PATH=$DR/lib:$DR/bin:$DR/model
export ADSP_LIBRARY_PATH="$DR/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp"

echo "START $(date)" > $DR/ctxgen_fp16.log
$DR/bin/qnn-context-binary-generator \
  --model $DR/model/libunet_lightning_fp16.so \
  --backend $DR/lib/libQnnHtp.so \
  --binary_file $DR/context/unet_lightning_fp16.serialized.bin \
  --config_file $DR/htp_backend_extensions_fp16_ultragentle.json \
  --log_level warn \
  >> $DR/ctxgen_fp16.log 2>&1
EC=$?
echo "EXIT:$EC" >> $DR/ctxgen_fp16.log
echo "END $(date)" >> $DR/ctxgen_fp16.log
echo $EC > $DR/ctxgen_fp16_exit.txt
