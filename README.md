### [Characteristic-preserving Latent Space for Unpaired Cross-domain Translation of 3D-Point Clouds](https://ieeexplore.ieee.org/abstract/document/10158055/)

Jia-Wen Zheng, Jhen-Yung Hsu, Chih-Chia Li, and I-Chen Lin

### Prerequisites
- ubuntu 18.04
- python 3.7
- pytorch 0.4.1
- torchvision 0.2.1
- pyntcloud
- visdom
- mitsuba2 (https://www.mitsuba-renderer.org/)
- flask

### 
### Install
We conducted experiments on NVIDIA GeForce RTX 2080 SUPER and RTX 2070, and installed packages in the following process:
```
conda create -n CharPresTrans python=3.6
- conda install pytorch=0.4.1 cuda92 -c pytorch
- conda install -c pytorch torchvision=0.2.1
- conda install -c conda-forge pyntcloud
- pip install visdom
- pip install pyyaml
- conda install -c conda-forge matplotlib
- conda install -c conda-forge cupy=9.6
- conda install -c anaconda scikit-learn
```

### Dataset
- Shapenet
- LOGAN
    - https://github.com/kangxue/LOGAN
- P2P-Net
    - https://github.com/kangxue/P2P-NET

- Our Paired Arm-and-armless Chairs Dataset
    - https://drive.google.com/drive/folders/1iteBcpUWKzyHEWRt65F4n6qc95C6q2Zk?usp=sharing

### Usage
- training:
    change dataset_path in configs/config_chair_table.yaml
    run visdom
    run python3 train.py --config configs/config_chair_table.yaml
    or python3 train.py --config configs/config_fit_fat.yaml
    or python3 train.py --config configs/config_p2p.yaml
    (loss curves plotted in http://localhost:8097)
- calculate CD and EMD for autoencoder:
    run function calculate_chamfer_emd in test.py
- save images (matplotlib version):
  change the path in configs/config_chair_table.yaml
  - save all reconstructed and transferred results:
run function save_all_images in test.py
  - save shape style mixing results:
run function save_shape_style_mixing_images in test.py
  - save MVS results:
change class_choice in configs/config_chair_table.yaml
run function save_MVS_images in test.py

- save npy file:
change the path in configs/config_chair_table.yaml
  - save reconstructed and transferred npy file:
change points_index in configs/config_chair_table.yaml
run function save_npy_index in test.py
  - save shape style mixing npy file:
change points_index_1 and points_index_2 in
configs/config_chair_table.yaml
run function save_shape_style_mixing_npy in test.py
  - save MVS npy file:
change class_choice in configs/config_chair_table.yaml
change points_index in configs/config_chair_table.yaml
run function save_MVS_npy in test.py
- show tsne results:
run function plot_tsne in test.py
- generate images for paper
download mitsuba2 (https://www.mitsuba-renderer.org/)
save npy file by the functions in test.py
cd plot file
change npy path in ./plot.sh
run source setpath.sh (in mitsuba2 file)
run ./plot.sh
- visualization in web
cd visualization file
run python3 app.py
change the index in function test
  - the results of chair-to-table transfer shown in
http://127.0.0.1:5000/chair2table
  - the results of table-to-chair transfer shown in
http://127.0.0.1:5000/table2chair
change the index in function test_different_z
  - the results of shape style mixing of chair shown in
http://127.0.0.1:5000/shape_mixing_chair
  - the results of shape style mixing of table shown in
http://127.0.0.1:5000/shape_mixing_table
- generate gif
cd visualization file
run python3 app_screenshot.py to generate multiple views
upload images to https://gifmaker.me/ to generate gif files


### Reference
LOGAN, https://github.com/kangxue/LOGAN

PointNet++, https://github.com/charlesq34/pointnet2

P2P-Net, https://github.com/kangxue/P2P-NET

RSCNN, https://github.com/Yochengliu/Relation-Shape-CNN

TreeGCN, https://openaccess.thecvf.com/content_ICCV_2019/papers/Shu_3D_Point_Cloud_Generative_Adversarial_Network_Based_on_Tree_Structured_ICCV_2019_paper.pdf
