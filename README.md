# CLIPtone
[CLIPtone](https://hmin970922.github.io/CLIPtone/) is a text-based tone adjustment method with unsupervised manner.
The code is based on [AdaInt](https://github.com/ImCharlesY/AdaInt) and [StyleGAN-nada](https://github.com/rinongal/StyleGAN-nada).


**The current README.md file is not the final version. I will update and upload the revised version soon.**


## Installation
```
git clone https://github.com/hmin970922/CLIPtone.git
cd CLIPtone
pip install -r requirements.txt
```


## Image Datasets
We will release soon...
You can use any image datasets..
논문에서는 MIT-Adobe 5K를 사용..
image 파일과 annotation 파일 필요
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


## Training Target Descriptions
우리는 학습 때 Target description으로 [Color Names Database](https://github.com/meodai/color-names)를 사용..
csv 폴더 내에 [colornames.csv](https://github.com/meodai/color-names/blob/master/src/colornames.csv)

