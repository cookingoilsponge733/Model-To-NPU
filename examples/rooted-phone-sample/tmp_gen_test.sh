#!/data/data/com.termux/files/usr/bin/bash
export LD_LIBRARY_PATH=/data/local/tmp/sdxl_qnn/lib:/data/local/tmp/sdxl_qnn/bin:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH='/data/local/tmp/sdxl_qnn/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp'
cd /data/local/tmp/sdxl_qnn/phone_gen
/data/data/com.termux/files/usr/bin/python generate.py "anime girl with blue eyes, cherry blossoms, detailed, vibrant colors" --seed 777 --name cfg35_default_test
