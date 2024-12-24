import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
import os

from pstats import SortKey
from scib_metrics.benchmark import Benchmarker

warnings.simplefilter(action='ignore', category=Warning)
# verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.verbosity = 3             
sc.settings.set_figure_params(dpi=600)

os.chdir('/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/')
# url for source and intermediate data
pancreas_all = sc.read('data/pancreas.h5ad', backup_url='https://figshare.com/ndownloader/files/38814050')

adata2 = pancreas_all.raw.to_adata()
sc.pp.log1p(adata2)
sc.pp.highly_variable_genes(adata2, subset=True)

# split data into batches
batches = adata2.obs['study'].cat.categories.tolist()
alldata = {}
for batch in batches:
    alldata[batch] = adata2[adata2.obs['study'] == batch,]

adatas = list(alldata.values())

# integrate 
import sys
import importlib
sys.path.append('/home/wu/datb1/AutoExtractSingleCell/scanorama_prior')
# reload the module to reflect changes
from scanorama_prior.scanorama import integrate, integrate_scanpy
importlib.reload(sys.modules['scanorama_prior.scanorama'])

from openai import AzureOpenAI
api_key = ""
azure_endpoint = "https://genomic-openai-ca.openai.azure.com/"

client = AzureOpenAI(
    api_key=api_key, api_version="2024-02-01", azure_endpoint=azure_endpoint
)

input_list = adata2.obs.cell_type.unique().tolist() + ['Unknown', 'Alpha cell']
response = client.embeddings.create(
        model="text-embedding-3-large", input=input_list
    )

emb = np.array([x.embedding for x in response.data])
similarity_matrix = pd.DataFrame(emb.dot(emb.T), index=input_list, columns=input_list)

# # run scanorama.integrate
print("Running scanorama.integrate")
integrate_scanpy(adatas,
    type_similarity_matrix=similarity_matrix,
    verbose=True,
    knn=30,
    approx=False,
)

adatas_concat = sc.AnnData.concatenate(*adatas, index_unique=None)
adatas_concat.obsm['X_scanorama_prior'] = adatas_concat.obsm['X_scanorama'].copy()
sc.pp.neighbors(adatas_concat, n_neighbors=30, use_rep='X_scanorama_prior')
sc.tl.umap(adatas_concat)
sc.tl.leiden(adatas_concat)
sc.pl.umap(adatas_concat, color=['cell_type', 'leiden'])

adatas_concat.obsm['X_scanorama_prior_umap'] = adatas_concat.obsm['X_umap'].copy()

emb = np.ones((len(input_list))) # test for constant embeddings
similarity_matrix = pd.DataFrame(emb.dot(emb.T), index=input_list, columns=input_list)

adatas_concat_tmp = sc.read('data/output_raw_scanorama.h5ad')
adatas_concat_tmp.obs.index = adatas_concat_tmp.obs.index
# sc.AnnData.concatenate(*adatas, index_unique=None)
adatas_concat_tmp = adatas_concat_tmp[adatas_concat.obs.index,:].copy()
assert (adatas_concat_tmp.obs_names == adatas_concat.obs_names).all()
adatas_concat.obsm['X_scanorama'] = adatas_concat_tmp.obsm['X_scanorama'].copy()
sc.pp.neighbors(adatas_concat, n_neighbors=30, use_rep='X_scanorama')
sc.tl.umap(adatas_concat)
adatas_concat.obsm['X_scanorama_umap'] = adatas_concat.obsm['X_umap'].copy()
del adatas_concat_tmp

# run PCA and UMAP of un-integrated data
sc.pp.scale(adatas_concat)
sc.pp.pca(adatas_concat)
sc.pp.neighbors(adatas_concat, n_neighbors=30, use_rep='X_pca')
sc.tl.umap(adatas_concat)

print("Running benchmark")
bm = Benchmarker(
    adatas_concat,
    batch_key="study",
    label_key="cell_type",
    embedding_obsm_keys=["X_pca", "X_umap", "X_scanorama", "X_scanorama_umap", "X_scanorama_prior", "X_scanorama_prior_umap"],
    n_jobs=-1,
)

adatas_concat.write('/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/output_scanorama.h5ad')

import pickle
bm.benchmark()
bm.plot_results_table(min_max_scale=False)
with open('/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/bm_scanorama.pkl', 'wb') as f:
    pickle.dump(bm, f)