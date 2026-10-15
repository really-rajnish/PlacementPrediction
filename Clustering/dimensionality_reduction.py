import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    import umap.umap_ as umap
except ImportError:
    umap = None

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv")
FEATURES=["CGPA","AttendancePercent","AptitudeTestScore","CodingTestScore"]

def load_features(sample_n=5000):
    df=pd.read_csv(DATA)
    X=df[FEATURES].apply(pd.to_numeric,errors="coerce").dropna()
    Z=StandardScaler().fit_transform(X)
    rng=np.random.RandomState(42)
    n=min(sample_n,len(Z))
    idx=rng.choice(len(Z),n,replace=False)
    return df.iloc[idx].copy(),Z[idx]

def make_plots(output_dir):
    os.makedirs(output_dir,exist_ok=True)
    _,X=load_features()

    labels=KMeans(n_clusters=4,n_init=10,random_state=42).fit_predict(X)

    pca_xy=PCA(n_components=2,random_state=42).fit_transform(X)
    tsne_xy=TSNE(
        n_components=2,perplexity=30,random_state=42,
        init="pca",learning_rate="auto"
    ).fit_transform(X)

    def save(xy,title,xlab,ylab,name):
        fig,ax=plt.subplots(figsize=(7.6,5.0))
        ax.scatter(
            xy[:,0],xy[:,1],c=labels,cmap="tab10",
            s=8,alpha=.45,linewidths=0
        )
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(alpha=.15)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,name),dpi=150,bbox_inches="tight")
        plt.close(fig)

    save(pca_xy,"PCA - 2D Projection","PC1","PC2","pca_projection.png")
    save(tsne_xy,"t-SNE - 2D Projection","t-SNE 1","t-SNE 2","tsne_projection.png")

    if umap is not None:
        umap_xy=umap.UMAP(
            n_components=2,n_neighbors=15,
            min_dist=0.15,random_state=42
        ).fit_transform(X)
        save(
            umap_xy,"UMAP - 2D Projection",
            "UMAP 1","UMAP 2","umap_projection.png"
        )

if __name__=="__main__":
    make_plots(
        os.path.join(ROOT,"static","clustering_images")
    )
