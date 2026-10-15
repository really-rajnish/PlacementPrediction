import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv")
FEATURES=["CGPA","AttendancePercent","AptitudeTestScore","CodingTestScore"]

def run_isolation_forest(contamination=0.01,plot_normal=6000):
    df=pd.read_csv(DATA)
    X=df[FEATURES].apply(pd.to_numeric,errors="coerce").dropna()
    Z=StandardScaler().fit_transform(X)

    labels=IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    ).fit_predict(Z)

    mask=labels==-1
    rng=np.random.RandomState(42)
    normal_idx=np.flatnonzero(~mask)
    normal_idx=rng.choice(
        normal_idx,min(plot_normal,len(normal_idx)),
        replace=False
    )
    anomaly_idx=np.flatnonzero(mask)

    fig,ax=plt.subplots(figsize=(7.6,5.0))
    ax.scatter(
        X.iloc[normal_idx]["CGPA"],
        X.iloc[normal_idx]["AptitudeTestScore"],
        s=9,alpha=.22,label="Normal"
    )
    ax.scatter(
        X.iloc[anomaly_idx]["CGPA"],
        X.iloc[anomaly_idx]["AptitudeTestScore"],
        s=30,marker="x",alpha=.75,label="Anomaly"
    )
    ax.set_title(
        f"Isolation Forest - Anomaly Detection "
        f"(contamination = {contamination:.0%})"
    )
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Aptitude Test Score")
    ax.legend()
    ax.grid(alpha=.15)
    return df.loc[X.index].copy(),labels,fig

if __name__=="__main__":
    df,labels,fig=run_isolation_forest()
    fig.savefig(
        os.path.join(ROOT,"anomaly_detection.png"),
        dpi=150,bbox_inches="tight"
    )
    print("Anomalies:",int((labels==-1).sum()))
