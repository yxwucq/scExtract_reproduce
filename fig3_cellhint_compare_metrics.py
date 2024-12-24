import sys
sys.path.append('/home/wu/datb1/AutoExtractSingleCell/cellhint_prior')

import cellhint_prior
print(cellhint_prior.__file__)
import scanpy as sc

adata = sc.read('/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/pancreas.h5ad')
adata = adata.raw.to_adata()

sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, subset=True)
sc.pp.scale(adata)
sc.tl.pca(adata)

from openai import AzureOpenAI
api_key = "" # Your API key
azure_endpoint = "https://genomic-openai-ca.openai.azure.com/"

client = AzureOpenAI(
    api_key=api_key, api_version="2024-02-01", azure_endpoint=azure_endpoint
)

input_list = adata.obs.cell_type.unique().tolist() + ['Unknown', 'Alpha cell']
response = client.embeddings.create(
        model="text-embedding-3-large", input=input_list
    )

embedding_dict_path = '/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/embedding_dict.pkl'
embedding_dict = {key: value.embedding for key, value in zip(input_list, response.data)}

# import pickle
# with open(embedding_dict_path, 'wb') as f:
#     pickle.dump(embedding_dict, f)

alignment = cellhint_prior.harmonize(adata, dataset = 'study', cell_type = 'cell_type', use_rep = 'X_pca', \
    use_pct = False, embedding_dict=embedding_dict)

import pickle
with open('/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/output_cellhint.pkl', 'wb') as f:
    pickle.dump(alignment, f)
cellhint_prior.treeplot(alignment)
adata.obs[f"harmonized_low"] = alignment.reannotation.loc[adata.obs_names, ['reannotation']].copy()

cellhint_prior.integrate(adata, batch = 'study', cell_type = f"harmonized_low")

sc.tl.umap(adata)
sc.pl.umap(adata, color = ['study', 'harmonized_low'])

adata.write(f"/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/output_cellhint.h5ad")