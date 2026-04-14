### 在云深处lite3进行实物部署

#### 服务器端
```
cd NAVILA-BENCH
python vlm_server.py --model_path ~/navila_ws/na(TAB) --port=54321 --host 0.0.0.0
```

#### lite3 103主机
```
cd navila_ws/scripts
python3 realsense_navila.py --vlm_host 10.15.196.104

```