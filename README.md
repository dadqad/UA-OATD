# UA-OATD
the implementation of UA-OATD: Deep unified attention-based sequence modeling for online anomalous trajectory detection
## Usage
First, preprocess data, Porto data can be downloaded from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data?select=train.csv.zip.
```
python preprocess_porto.py
```
Then, generate groud truth.
```
python generate_outliers.py --level 3 --point_prob 0.3 --obeserved_ratio 1.0
```
Train GM-VSAE.
```
python train.py --n_cluster 20 --pretrain_epochs 8 --epochs 10 
```
