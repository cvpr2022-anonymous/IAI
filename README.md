# Instance As Identity: A Generic Online Paradigm for Video Instance Segmentation

# prepare env
```
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"

pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
[cu_version} set to cu92 or other version, {torch_version} set to torch1.5.0 or other version.
for example, cuda 9.2 & torch 1.5, use 
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu92/1.5.0/index.html
```

# prepare dataset
1. Download YouTubeVIS from [here](https://youtube-vos.org/dataset/vis/).
2. Symlink the train/validation dataset to `test-speed/data` folder. Put COCO-style annotations under `data/annotations`.
```
IAI
├── mmdet
├── tools
├── configs
├── data
│   ├── train
│   ├── val
│   ├── annotations
│   │   ├── instances_train_sub.json
│   │   ├── instances_val_sub.json
```

# run test
Download pretrained r50 model from [here](https://pan.baidu.com/s/1pXIUp56ehhBe_P052O7s2A)(password: iii0), and put it to `models/iai_condinst_r50.pth`

Run the following command to test speed for R50
```
sh run.sh
```

Download pretrained r101 model from [here](https://pan.baidu.com/s/19dp-VBJ2xTmzGFO6tUMDHA)(password: 512e), and put it to `models/iai_condinst_r101.pth`

Run the following command to test speed for R101
```
sh run_r101.sh
```
