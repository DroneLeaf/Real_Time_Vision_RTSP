# Real_Time_Vision_RTSP

## Installation Scripts

### Python Packages:

```bash
pip install --upgrade pip
#  CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
```

### System wide tools:
```bash
sudo apt update
sudo apt install ffmpeg
```

## How to run
1. Launch the mediamtx server by running `./mediamtx` inside the folder `RTSP_server_mediamtx_v1_11_3`
2. Run the `Flight_test_everything.py` script by running `python3 Flight_test_everything.py`
3. Open QGC and change the RTSP server address to `rtsp://192.168.144.6:8554/live/processed_stream`
4. Yolo detection boxes should be visible in the video stream. 
5. (Optional) You can chose between YOLO and QR detection by setting the variable in `Flight_test_everything.py` 
```python
detection_mode = "qr" or "yolo"  # @ line 22 in `Flight_test_everything.py` 
```

## Test History

### Achieved 100FPS GPU and 50FPS CPU
#### GPU Specs
`nvidia-smi`

Mon Jun  2 15:44:11 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   49C    P8             18W /   90W |      75MiB /  16384MiB |     29%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1443      G   /usr/lib/xorg/Xorg                       14MiB |
|    0   N/A  N/A            4567      G   /usr/lib/xorg/Xorg                       45MiB |
+-----------------------------------------------------------------------------------------+

#### CPU Specs
`lscpu`

Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Byte Order:                           Little Endian
Address sizes:                        39 bits physical, 48 bits virtual
CPU(s):                               16
On-line CPU(s) list:                  0-15
Thread(s) per core:                   2
Core(s) per socket:                   8
Socket(s):                            1
NUMA node(s):                         1
Vendor ID:                            GenuineIntel
CPU family:                           6
Model:                                141
Model name:                           11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz
Stepping:                             1
CPU MHz:                              2600.000
CPU max MHz:                          5000.0000
CPU min MHz:                          800.0000
