# OpticalFlow-RI (Reference Implementations) for Fluid Mechanics

OpticalFlow-RI is a source code repository of tested and validated optical flow algorithm implementations, being specificaly tuned for fluid mechanics applications with Particle Imaging Velocimetry (PIV) image databases having a maximum displacement of 4.0px between frames.

This repository intends to be a reference library for reuse by anyone interested in evaluating said algorithms for fluid mechanics velocimetry applications.

The library was created with minimal third-party dependencies for the relevant algorithms. This way it is less susceptible to unwanted behavioral changes that may arise by third-party libraries implementation changes/upgrades, ultimately leading to uncalibrated results or API breakage.

The library includes concrete examples on how to run the algorithms along with the calibration parameters.


Current reference implementations include:
- Horn-Schunck (Python, Numba)
- Lucas-Kanade (Python, OpenCL, Numba)
- Farnebäck (Python, OpenCL, Numba)
- Liu-Shen Physics based optimizer (Python, Numba)

The Numba implementations provide CPU acceleration without requiring OpenCL, making them more portable and easier to use.


**Open-Source licensed**

## Sample synthetic PIV images included

| ![Poiseuille flow](https://github.com/CoreRasurae/OpticalFlow-RI/raw/master/gifs/parabolic01.gif) |

## Prerequisites
Python3 is required to run these algorithms.

Implementations tested with Ubuntu 20.04.2LTS and Ubuntu 21.04

Required packages include:
```bash
apt install python3 python3-numpy python3-skimage python3-pyopencl python3-numba
```

Or using pip:
```bash
pip install numpy scikit-image pyopencl numba matplotlib scipy pytest tqdm
```

*Note:* A valid OpenCL device is required to run the OpenCL versions of dense Lucas-Kanade and Farnebäck algorithms. However, the Numba-accelerated versions can run on any CPU without requiring OpenCL.

## Running the examples
There are a number of examples in the examples folder, namely:

| Example file                      | Description                                                                        | Execute                                   |
|-----------------------------------|------------------------------------------------------------------------------------|-------------------------------------------|
| Farneback_Fs0_0.py                | Farnebäck with Null Gaussian pre-filter without pyramidal                          | python3 Farneback_Fs0_0.py                |
| Farneback_Fs0_0_PyrLvls2.py       | Farnebäck with Null Gaussian pre-filter with pyramidal                             | python3 Farneback_Fs0_0_PyrLvls2.py       |
| LiuSE_Farneback_Fs0_0_PyrLvls2.py | Liu-Shen optimized Farnebäck with Null Gaussian pre-filter with pyramidal          | python3 LiuSE_Farneback_Fs0_0_PyrLvls2.py |
| denseLK_Fs2_0.py                  | dense Lucas-Kanade with 2.0px Gaussian pre-filter without pyramidal                | python3 denseLK_Fs2_0.py                  |
| denseLK_Fs2_0_PyrLvls2.py         | dense Lucas-Kanade with 2.0px Gaussian pre-filter with pyramidal                   | python3 denseLK_Fs2_0_PyrLvls2.py         |
| LiuSE_denseLK_Fs2_0_PyrLvls2.py   | Liu-Shen optimized dense Lucas-Kanade with 2.0px Gaussian pre-filter with pyramidal| python3 LiuSE_denseLK_Fs2_0_PyrLvls2.py   |
| PyHSchunck_Fs3_4.py               | Horn-Schunck with 3.4px Gaussian pre-filter without pyramidal                      | python3 PyHSchunck_Fs3_4.py               |
| PyHSchunck_Fs3_4_PyrLvls2.py      | Horn-Schunck with 3.4px Gaussian pre-filter with pyramidal                         | python3 PyHSchunck_Fs3_4_PyrLvls2.py      |
| LiuSE_PyHSchunck_Fs3_4_PyrLvls2.py| Liu-Shen optimized Horn-Schunck with 3.4px Gaussian pre-filter with pyramidal      | python3 LiuSE_PyHSchunck_Fs3_4_PyrLvls2.py|

### Numba-accelerated Examples

| Example file                           | Description                                                                | Execute                                        |
|----------------------------------------|----------------------------------------------------------------------------|------------------------------------------------|
| run_denseLucasKanade_Numba_optimized.py | Optimized Numba implementation of Dense Lucas-Kanade with pyramidal       | python3 run_denseLucasKanade_Numba_optimized.py |
| analyze_horizontal_profile.py          | Analyze horizontal profiles from optical flow results                      | python3 analyze_horizontal_profile.py          |


## Obtaining the code

The official source repository for Optical Flow reference implementations is located in the CoreRasurae Github repository and can be cloned using the following command.

```bash

git clone https://github.com/CoreRasurae/OpticalFlow-RI
```

