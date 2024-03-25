# CLIPtone
[CLIPtone](https://youngchan-k.github.io/nespof) is an physically-valid neural representation that models spectro-polarimetric fields.
The code is based on [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch) and you should follow the [requirements.txt](https://github.com/yenchenlin/nerf-pytorch/blob/master/requirements.txt) provided there.

## Installation
```
git clone https://github.com/youngchan-k/NeSpoF.git
cd nespof
pip install -r requirements.txt
pip install imath
conda install -c conda-forge openexr
(or conda install -c conda-forge openexr-python)
```

## Datasets
Download the data [here](https://drive.google.com/drive/folders/1W7apuXPA3EkyUs8VgZgwdMpnc96aLXXJ). Addtionally, you can generate a synthetic dataset using Mitsuba 3 with reference to this code [here](https://drive.google.com/drive/folders/1IDQsnRMGTXcek4TMs2PjO_0IHs-zxMgz). Place the downloaded dataset according to the following directory structure:

```
data
|-- synthetic
|   |-- ajar
|   |-- cbox_dragon
|   |-- ...
|-- real
    |-- scene_1
    |-- scene_2
    |-- ...
```


## How To Run?
To train NeSpoF on synthetic datasets:
```
python run_nespof.py --config configs/synthetic/{DATASET}.txt
```
replace {DATASET} with ajar | cbox_dragon | cbox_sphere | hotdog.

---

To train NeSpoF on real-world datasets:
```
python run_nespof.py --config configs/real/{DATASET}.txt
```
replace {DATASET} with scene_1 | scene_2 | scene_3 | scene_4.

---

After training, you can also render the video for Stokes vector and polarimetric visualization:
```
python run_nespof.py --config configs/video/{DATASET}.txt
```
replace {DATASET} with ajar | cbox_dragon | cbox_sphere | hotdog | scene_1 | scene_2 | scene_3 | scene_4.
