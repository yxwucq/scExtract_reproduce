import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
import os
import urllib.request

warnings.simplefilter(action='ignore', category=Warning)
# verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.verbosity = 3             
sc.settings.set_figure_params(dpi=80)

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

# modify cell type names
alldata['celseq2'].obs.cell_type = alldata['celseq2'].obs.cell_type.astype(str)
alldata['celseq2'].obs.cell_type[alldata['celseq2'].obs.cell_type == 'beta'] = 'Unknown' # test for missing cell type
alldata['inDrop3'].obs.cell_type = alldata['inDrop3'].obs.cell_type.astype(str)
alldata['inDrop3'].obs.cell_type[alldata['inDrop3'].obs.cell_type == 'ductal'] = 'gamma' # test for wrong cell type
alldata['inDrop1'].obs.cell_type = alldata['inDrop1'].obs.cell_type.astype(str)
alldata['inDrop1'].obs.cell_type[alldata['inDrop1'].obs.cell_type == 'alpha'] = 'Alpha cell' # test for alternative cell type name

adatas = list(alldata.values())

# integrate 
from scanorama_prior.scanorama import integrate, integrate_scanpy
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
similarity_matrix.to_csv('data/similarity_matrix_label_changed.csv', index=True)

import sys
import importlib
sys.path.append('..')
# reload the module to reflect changes
importlib.reload(sys.modules['scanorama_prior.scanorama'])
from scanorama_prior.scanorama import integrate, integrate_scanpy

integrate_scanpy(adatas,
    type_similarity_matrix=similarity_matrix,
    verbose=True,
    knn=30,
    approx=False,
)

adatas_concat = sc.AnnData.concatenate(*adatas)
sc.pp.neighbors(adatas_concat, use_rep='X_scanorama', n_neighbors=30)
sc.tl.umap(adatas_concat)
sc.pl.umap(adatas_concat, color=['study', 'cell_type'], frameon=False)

sc.tl.pca(adatas_concat)
sc.pp.neighbors(adatas_concat, use_rep='X_pca')
sc.tl.umap(adatas_concat)
sc.pl.umap(adatas_concat, color=['cell_type', 'study'])

adatas_concat.write('data/output_scanorama_label_change.h5ad')