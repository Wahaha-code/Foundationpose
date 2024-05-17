# Foundationpose
Implementing 6DOF tracking of equipment based on foundationpose and yolo v8

## Weight
Download all network weights from[here](https://cashkisi-my.sharepoint.com/:f:/g/personal/wenwu_guo_cair-cas_org_hk/EiIIqa0XDKxIrojRXSSAVoEBz6cUKBTdvSllwHx8uFxd2A?e=wQZySx) and put them under the folder `weights/`

## Demo
![AR Applications](assets/ezgif.com-video-to-gif-converter.gif)

## Env setup(GPU 4090): docker
  ```
  cd docker/
  docker pull gww106/foundationpose:tagname
  bash docker/run_container.sh
  ```

If it's the first time you launch the container, you need to build extensions.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```
## Real time tracking
```
python online.py
```
