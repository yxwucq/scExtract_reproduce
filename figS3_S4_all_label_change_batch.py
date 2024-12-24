import scanpy as sc
import pandas as pd
import os
import numpy as np
import json
from datetime import datetime
import cellhint
import cellhint_prior
from openai import AzureOpenAI
import pickle
import anndata as ad
import scanorama
import scanorama_prior

def remove_none_type(x):
    x = x.replace('NONE = ', '')
    x = x.replace(' = NONE', '')
    x = x.replace('UNRESOLVED = ', '')
    x = x.replace(' = UNRESOLVED', '')
    return x

def run_label_replacement_experiment(
    input_path,
    output_dir,
    api_key,
    azure_endpoint,
    label_changes
):
    """
    Run label replacement experiments on single-cell data
    
    Parameters:
    input_path: Path to input h5ad file
    output_dir: Directory for output files
    api_key: Azure OpenAI API key
    azure_endpoint: Azure OpenAI endpoint
    label_changes: List of tuples (study, old_label, new_label)
    """
    # Load and preprocess data
    adata = sc.read(input_path)
    adata = adata.raw.to_adata()
    adata.raw = adata
    
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, subset=True, batch_key='study')
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    
    # Apply label changes
    adata.obs.cell_type = adata.obs.cell_type.astype(str)
    adata.obs['real_cell_type'] = adata.obs.cell_type.copy()
    for study, old_label, new_label in label_changes:
        mask = (adata.obs.cell_type == old_label) & (adata.obs.study == study)
        adata.obs.loc[mask, 'cell_type'] = new_label
    
    adata.obs['changed_cell_type'] = adata.obs.cell_type.copy()
    # Get embeddings
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_endpoint
    )
    
    input_list = list(set(adata.obs.cell_type.unique().tolist()))
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=input_list
    )
    embedding_dict = {key: value.embedding for key, value in zip(input_list, response.data)}

    # Run CellHint
    print("Running CellHint")
    alignment = cellhint.harmonize(
        adata,
        dataset='study',
        cell_type='cell_type',
        use_rep='X_pca',
        use_pct=False,
    )
    
    # Save results and visualize
    with open(f"{output_dir}/alignment_raw.pkl", 'wb') as f:
        pickle.dump(alignment, f)
    adata.obs["harmonized_low_raw"] = alignment.reannotation.loc[adata.obs_names, ['reannotation']].copy()
    cellhint_prior.integrate(adata, batch='study', cell_type="harmonized_low_raw", n_neighbors=30)

    sc.tl.leiden(adata, resolution=0.7, key_added='leiden_cellhint_raw')
    sc.tl.umap(adata)
    adata.obsm['X_umap_cellhint_raw'] = adata.obsm['X_umap'].copy()

    # Run CellHint prior
    print("Running CellHint prior")
    alignment = cellhint_prior.harmonize(
        adata,
        dataset='study',
        cell_type='cell_type',
        use_rep='X_pca',
        use_pct=False,
        embedding_dict=embedding_dict
    )
    
    # Save results and visualize
    with open(f"{output_dir}/alignment.pkl", 'wb') as f:
        pickle.dump(alignment, f)
        
    adata.obs["harmonized_low_prior"] = alignment.reannotation.loc[adata.obs_names, ['reannotation']].copy()
    cellhint_prior.integrate(adata, batch='study', cell_type="harmonized_low_prior", n_neighbors=30)

    sc.tl.leiden(adata, resolution=0.7, key_added='leiden_cellhint_prior')
    sc.tl.umap(adata)
    adata.obsm['X_umap_cellhint_prior'] = adata.obsm['X_umap'].copy()
        
    # Run Scanorama
    adata = adata.raw.to_adata()
    assert 'X_umap_cellhint_prior' in adata.obsm_keys()
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, subset=True, batch_key='study')
    
    print("Running Scanorama")
    batches = adata.obs['study'].cat.categories.tolist()
    alldata = {}
    for batch in batches:
        alldata[batch] = adata[adata.obs['study'] == batch,]
    adatas = list(alldata.values())
    
    scanorama.scanorama.integrate_scanpy(adatas,
    verbose=True,
    knn=30,
    approx=False,
    )
    
    adata = ad.concat(adatas).copy()
    sc.pp.neighbors(adata, use_rep='X_scanorama', n_neighbors=30)
    
    sc.tl.leiden(adata, resolution=0.7, key_added='leiden_scanorama_raw')
    sc.tl.umap(adata)
    adata.obsm['X_umap_scanorama_raw'] = adata.obsm['X_umap'].copy()
    
    # Run Scanorama prior with uncorrected cell types
    print("Running Scanorama prior")
    batches = adata.obs['study'].cat.categories.tolist()
    alldata = {}
    for batch in batches:
        alldata[batch] = adata[adata.obs['study'] == batch,]
    adatas = list(alldata.values())    
    
    input_list = list(set(adata.obs.cell_type.unique().tolist()))
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=input_list
    )
    emb = np.array([x.embedding for x in response.data])
    similarity_matrix = pd.DataFrame(emb.dot(emb.T), index=input_list, columns=input_list)

    scanorama_prior.scanorama.integrate_scanpy(adatas,
        type_similarity_matrix=similarity_matrix,  # use the generated similarity matrix
        verbose=True,
        knn=30,
        approx=False,
    )
    
    adata = ad.concat(adatas).copy()
    
    sc.pp.neighbors(adata, use_rep='X_scanorama', n_neighbors=30)
    sc.tl.leiden(adata, resolution=0.7, key_added='leiden_scanorama_prior_uncorrected')
    sc.tl.umap(adata)
    adata.obsm['X_umap_scanorama_prior_uncorrected'] = adata.obsm['X_umap'].copy()

    # Run Scanorama prior with harmonized cell types
    adata.obs["cell_type"] = adata.obs[f"harmonized_low_prior"].apply(remove_none_type)
    
    print("Running Scanorama prior with harmonized cell types")
    batches = adata.obs['study'].cat.categories.tolist()
    alldata = {}
    for batch in batches:
        alldata[batch] = adata[adata.obs['study'] == batch,]
    adatas = list(alldata.values())   
    
    input_list = list(set(adata.obs.cell_type.unique().tolist()))
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=input_list
    )
    
    emb = np.array([x.embedding for x in response.data])
    similarity_matrix = pd.DataFrame(emb.dot(emb.T), index=input_list, columns=input_list)

    scanorama_prior.scanorama.integrate_scanpy(adatas,
        type_similarity_matrix=similarity_matrix,  # use the generated similarity matrix
        verbose=True,
        knn=30,
        approx=False,
    )
    
    adata = ad.concat(adatas).copy()
    
    sc.pp.neighbors(adata, use_rep='X_scanorama', n_neighbors=30)
    sc.tl.leiden(adata, resolution=0.7, key_added='leiden_scanorama_prior_harmonized')
    sc.tl.umap(adata)
    adata.obsm['X_umap_scanorama_prior_harmonized'] = adata.obsm['X_umap'].copy()
    
    adata.write(f"{output_dir}/output.h5ad")
    
    return adata, alignment

def create_experiment_table(base_output_dir):
    """
    Creates and manages an experiment tracking system
    
    Parameters:
    base_output_dir: Base directory for all experiment outputs
    """
    def generate_experiment_id():
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    class ExperimentTracker:
        def __init__(self, base_dir):
            self.base_dir = base_dir
            self.table_path = os.path.join(base_dir, 'experiment_table.csv')
            if os.path.exists(self.table_path):
                self.table = pd.read_csv(self.table_path)
            else:
                self.table = pd.DataFrame(columns=[
                    'experiment_id',
                    'date',
                    'label_changes',
                    'alignment_path',
                    'adata_path',
                    'description'
                ])
                
        def add_experiment(self, label_changes, description=""):
            experiment_id = generate_experiment_id()
            experiment_dir = os.path.join(self.base_dir, experiment_id)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Convert label changes to string representation
            label_changes_str = json.dumps(label_changes)
            
            # Create paths
            alignment_path = os.path.join(experiment_dir, 'alignment.pkl')
            adata_path = os.path.join(experiment_dir, 'output.h5ad')
            
            # Add to table
            new_row = pd.DataFrame([{
                'experiment_id': experiment_id,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'label_changes': label_changes_str,
                'alignment_path': alignment_path,
                'adata_path': adata_path,
                'description': description
            }])
            
            self.table = pd.concat([self.table, new_row], ignore_index=True)
            self.table.to_csv(self.table_path, index=False)
            
            return experiment_dir, alignment_path, adata_path
        
        def get_table(self):
            return self.table
        
        def get_experiment_paths(self, experiment_id):
            row = self.table[self.table['experiment_id'] == experiment_id].iloc[0]
            return row['alignment_path'], row['adata_path']
    
    return ExperimentTracker(base_output_dir)

def run_tracked_experiments(tracker, experiments, input_path, api_key, azure_endpoint):
    """
    Run experiments and track them in the table
    """
    for i, label_changes in enumerate(experiments):
        description = f"Experiment {i+1}: " + ", ".join([
            f"{study}:{old}->{new}" 
            for study, old, new in label_changes
        ])
        print(f"=== Running experiment {i+1} ===")
        print(description)
        
        # Get paths from tracker
        experiment_dir, alignment_path, adata_path = tracker.add_experiment(
            label_changes, 
            description
        )
        
        # Run experiment
        adata, alignment = run_label_replacement_experiment(
            input_path=input_path,
            output_dir=experiment_dir,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            label_changes=label_changes
        )
        
    return tracker.get_table()

base_output_dir = '/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/label_change_output'
experiments = [
    # Group 1 - Endocrine cell variations
    [
        ("smartseq2", "alpha", "Alpha cell"),    # Fuzzy match: 1008 cells
        ("celseq2", "beta", "Unknown"),          # Unbiased error: 445 cells
        ("inDrop3", "delta", "gamma")            # Biased error: 161 cells
    ],
    # Group 2 - Mixed exocrine and endocrine
    [
        ("inDrop3", "acinar", "Acinar cell"),    # Fuzzy match: 843 cells
        ("inDrop2", "alpha", "Unknown"),         # Unbiased error: 676 cells
        ("inDrop3", "ductal", "gamma")           # Biased error: 376 cells
    ],
    # Group 3 - Large sample size focus
    [
        ("celseq2", "alpha", "Alpha cell"),      # Fuzzy match: 843 cells
        ("inDrop3", "beta", "Unknown"),          # Unbiased error: 787 cells
        ("smartseq2", "ductal", "delta")         # Biased error: 444 cells
    ],
    # Group 4 - Cross-category testing
    [
        ("inDrop3", "beta", "Endocrine cell"),   # Fuzzy match: 787 cells
        ("inDrop1", "ductal", "Unknown"),        # Unbiased error: 120 cells
        ("celseq2", "alpha", "beta")             # Biased error: 843 cells
    ],
    # Group 5 - Mixed cell types
    [
        ("smartseq2", "ductal", "Ductal cell"),  # Fuzzy match: 444 cells
        ("inDrop3", "alpha", "Unknown"),         # Unbiased error: 1130 cells
        ("celseq2", "beta", "alpha")             # Biased error: 445 cells
    ],
    # Group 6 - Raw data
    []
]

tracker = create_experiment_table(base_output_dir)

results_table = run_tracked_experiments(
    tracker=tracker,
    experiments=experiments,
    input_path='/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/pancreas.h5ad',
    api_key="",
    azure_endpoint="https://genomic-openai-ca.openai.azure.com/"
)

print(results_table)