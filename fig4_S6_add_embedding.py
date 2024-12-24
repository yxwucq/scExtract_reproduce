import gc
import os
import scanpy as sc
# from scib_metrics.benchmark import Benchmarker
from scextract.integration import extract_celltype_embedding

name_list = ['Immune_ALL_human', 'Lung_atlas_public', 'Blood', 'Spleen', 'TIC', 'HCLA', 'HCLA_sub']
adata_list = ['Immune_ALL_human.h5ad', 'Lung_atlas_public.h5ad',
              'Blood.h5ad', 'Spleen.h5ad', 'TICAtlas.h5ad', 'human_lung_cell_atlas.h5ad',
              'human_lung_cell_atlas_subset.h5ad']

for name, adata_path in zip(name_list, adata_list):
    adata_path = 'results/' + adata_path
    print(f"Processing {name}")

    tmp_path = adata_path.replace('.h5ad', '_integrated_tmp.h5ad')
    embedding_path = adata_path.replace('.h5ad', '_harmonized_embedding.pkl')

    extract_celltype_embedding.extract_celltype_embedding(
        file_list=[tmp_path],
        output_embedding_pkl=embedding_path
    )