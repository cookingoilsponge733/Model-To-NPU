#!/data/data/com.termux/files/usr/bin/bash
export LD_LIBRARY_PATH=/data/local/tmp/sdxl_qnn/lib:/data/local/tmp/sdxl_qnn/bin:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH='/data/local/tmp/sdxl_qnn/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp'
cd /data/local/tmp/sdxl_qnn/phone_gen
/data/data/com.termux/files/usr/bin/python generate.py "1girl, anime, colorful, cherry blossoms" --seed 42 --steps 8 --cfg 3.5 --neg "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry" --no-stretch --name phone_cfg35_s8
