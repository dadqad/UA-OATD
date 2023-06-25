# UA-OATD
The implementation of UA-OATD: Deep unified attention-based sequence modeling for online anomalous trajectory detection
### Requirements
```
pip install -r requirements.txt
```
### Preprocessing
- Step1: Download Porto data (<tt>train.csv.zip</tt>) from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data.
- Step2: Put the data file in <tt>./data/porto/</tt>, and unzip it as <tt>porto.csv</tt>.
- Step3: Run preprocessing by <tt>python preprocess_porto.py</tt>.
### Generating ground truth
```
python generate_outliers.py --level 3 --point_prob 0.3 --obeserved_ratio 1.0
```
The generation process is referred to https://github.com/liuyiding1993/ICDE2020_GMVSAE.
### Training
Example of training on Porto dataset:
```
python train_uaoatd.py --n_cluster 20 --pretrain_epochs 8 --epochs 10 --dataset porto
```
