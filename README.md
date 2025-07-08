# create lxc

bash -c "$(wget -qLO - https://raw.githubusercontent.com/JorisOpsommer/frigate-yolo/refs/heads/main/lxc/init.sh)"

# logs

dev/shm/logs/frigate

# run frigate in dev mode

## frigate

python3 -m frigate

## web

cd web/
npm run dev

# edgetpu compiler

edgetpu tflite compiler, in wsl:
edgetpu_compiler mobilenet_ssd_int8.tflite
