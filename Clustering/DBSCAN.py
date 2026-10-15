import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv")
FEATURES=["CGPA","AptitudeTestScore"]

def run_dbscan(eps=0.5,min_samples=5):
    df=pd.read_csv(DATA); X=df[FEATURES].apply(pd.to_numeric,errors="coerce").dropna()
    Z=StandardScaler().fit_transform(X); labels=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(Z)
    return df.loc[X.index].copy(),labels
