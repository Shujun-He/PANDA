from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

LABELS = 'train.csv'
df = pd.read_csv(LABELS)

SEED = 2020
nfolds = 3
splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
splits = list(splits.split(df,df.isup_grade,df.data_provider))
folds_splits = np.zeros(len(df)).astype(np.int)
for i in range(nfolds): folds_splits[splits[i][1]] = i

blank_slides = ["3790f55cad63053e956fb73027179707"]
low_tiss_slides = ['033e39459301e97e457232780a314ab7',
 '0b6e34bf65ee0810c1a4bf702b667c88',
 '3385a0f7f4f3e7e7b380325582b115c9',
 '3790f55cad63053e956fb73027179707',
 '5204134e82ce75b1109cc1913d81abc6',
 'a08e24cff451d628df797efc4343e13c']

for i in range(len(df)):
    if df.image_id[i] in blank_slides or df.image_id[i] in low_tiss_slides:
        folds_splits[i]=-1

df['split'] = folds_splits
df.to_csv('train_{}fold.csv'.format(nfolds),index=False)
