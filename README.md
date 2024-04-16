# TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks [TNNLS 2024]

<p align="center" float="center">
  <img src="https://github.com/ridgerchu/TCJA/blob/main/TCJA.png" width=60%/>
</p>

## How to Run

First clone the repository.

```shell
git clone https://github.com/ridgerchu/TCJA
cd TCJA
pip install -r requirements.txt
```

### Train DVS128 

Detailed usage of the script could be found in the source file.

```shell
python src/dvs128.py -data_dir /path/to/DVSGesture -out_dir runs/dvs128/ -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 1024 -T 20 -epochs 1024 -b 16 -lr 0.001 -amp -j 20
```

and the dataset folder `DVSGesture` should look like:

```
DVSGesture
├── download
│   ├── DvsGesture.tar.gz
│   ├── gesture_mapping.csv
│   ├── LICENSE.txt
│   ├── README.txt
│   ├── trials_to_test.txt
│   └── trials_to_train.txt
├── DVS128_frames_number_20_split_by_number.zip
├── DVSGesture.zip
├── events_np
│   ├── test
│   └── train
├── extract
│   └── DvsGesture
├── frames_number_100_split_by_number
│   ├── test
│   └── train
├── frames_number_10_split_by_number
│   ├── test
│   └── train
...
```

### Train N-Caltech 101

```shell
python src/caltech101.py -data_dir /path/to/NCAL101/ -out_dir runs/caltech101 -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 1024 -T 14 -epochs 1024 -b 16 -lr 0.001 -j 20 -loss mse -amp
```

The `NCAL101` looks like

```
NCAL101
├── events_np
├── extract
├── frames_number_14_split_by_number
└── NCAL101_frames_number_14_split_by_number.zip
```

### Train CIFAR10-DVS

```shell
python src/cifar10dvs.py -data_dir /path/to/CIFAR10DVS/ -out_dir runs/cifar10dvs -opt Adam -device cuda:1 -lr_scheduler CosALR -T_max 1024 -T 20 -epochs 1024 -b 16 -lr 0.001 -j 20
```

The `CIFAR10DVS` looks like

```
CIFAR10DVS/
├── events_np
├── extract
├── extract.zip
├── frames_number_10_split_by_number
├── frames_number_10_split_by_number.zip
├── frames_number_16_split_by_number
├── frames_number_20_split_by_number
└── frames_number_20_split_by_number.zip
```
If you find TCJA module useful in your work, please cite the following source:

```
@article{zhu2022tcja,
  title={TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks},
  author={Zhu, Rui-Jie and Zhao, Qihang and Zhang, Tianjing and Deng, Haoyu and Duan, Yule and Zhang, Malu and Deng, Liang-Jian},
  journal={arXiv preprint arXiv:2206.10177},
  year={2022}
}
```
