import scanpy as sc
# import importlib
# import sys
# sys.path.remove('/home/wu/datb1/AutoExtractSingleCell/cellhint')
import cellhint_prior
print(cellhint_prior.__file__)

adata = sc.read('/home/wu/datb1/AutoExtractSingleCell/scanorama/data/pancreas.h5ad')
adata = adata.raw.to_adata()

sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, subset=True)
sc.pp.scale(adata)
sc.tl.pca(adata)

alignment = cellhint_prior.harmonize(adata, dataset = 'study', cell_type = 'cell_type', use_rep = 'X_pca', \
    use_pct = False)

import pickle
with open('/home/wu/datb1/AutoExtractSingleCell/scanorama/data/output_raw_cellhint.pkl', 'wb') as f:
    pickle.dump(alignment, f)

cellhint_prior.treeplot(alignment)
adata.obs[f"harmonized_low"] = alignment.reannotation.loc[adata.obs_names, ['reannotation']].copy()

cellhint_prior.integrate(adata, batch = 'study', cell_type = f"harmonized_low")

sc.tl.umap(adata)
sc.pl.umap(adata, color = ['study', 'harmonized_low'])

adata.write(f"/home/wu/datb1/AutoExtractSingleCell/scanorama/data/output_raw_cellhint.h5ad")