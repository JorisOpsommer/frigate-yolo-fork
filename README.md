# create lxc

bash -c "$(curl -fsSL https://raw.githubusercontent.com/JorisOpsommer/frigate-yolo-fork/refs/heads/main/lxc/init.sh)"

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

# lxc community scripts original:

https://github.com/community-scripts/ProxmoxVE/blob/main/ct/frigate.sh
https://github.com/community-scripts/ProxmoxVE/blob/b9ac02e74f23863e5d3fbddc8749a51207f826ae/install/frigate-install.sh
https://github.com/community-scripts/ProxmoxVE/blob/b9ac02e74f23863e5d3fbddc8749a51207f826ae/misc/build.func
