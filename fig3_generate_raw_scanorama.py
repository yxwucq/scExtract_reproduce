import scanpy as sc
import scanorama
import importlib
importlib.reload(scanorama)

adata = sc.read('/home/wu/datb1/AutoExtractSingleCell/scanorama/data/pancreas.h5ad')

adata = adata.raw.to_adata()
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, subset=True)

# split data into batches
batches = adata.obs['study'].cat.categories.tolist()
alldata = {}
for batch in batches:
    alldata[batch] = adata[adata.obs['study'] == batch,]

adatas = list(alldata.values())

scanorama.integrate_scanpy(adatas,
    verbose=True,
    knn=30,
    approx=False,
)

adatas_concat = sc.AnnData.concatenate(*adatas, index_unique=None)
sc.pp.neighbors(adatas_concat, n_neighbors=30, use_rep='X_scanorama')
sc.tl.umap(adatas_concat)

adatas_concat.write('/home/wu/datb1/AutoExtractSingleCell/scanorama/data/output_raw_scanorama.h5ad')