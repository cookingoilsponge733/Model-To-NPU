#!/data/data/com.termux/files/usr/bin/bash
export PATH="/data/data/com.termux/files/usr/bin:$PATH"
export LD_LIBRARY_PATH=/data/local/tmp/sdxl_qnn/lib:/data/local/tmp/sdxl_qnn/bin:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/sdxl_qnn/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp"
cd /data/local/tmp/sdxl_qnn
/data/data/com.termux/files/usr/bin/python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "1girl, anime, colorful, cherry blossoms" --seed 42 --name phone_test1 2>&1
