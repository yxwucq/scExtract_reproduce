import scanpy as sc
import pandas as pd
import os
import gc
import numpy as np
import time
from datetime import datetime
import cellhint
import cellhint_prior
import scanorama
import scanorama_prior
import anndata as ad
from openai import AzureOpenAI

def profile_method(method_name, method_func, adata, **kwargs):
    """
    Profile the running time of a method
    
    Parameters:
    method_name: Name of the method being profiled
    method_func: Function to run
    adata: AnnData object
    kwargs: Additional arguments for the method
    
    Returns:
    float: Running time in seconds
    """
    start_time = time.time()
    method_func(adata, **kwargs)
    end_time = time.time()
    return end_time - start_time

def run_performance_experiment(
    input_path,
    output_dir,
    api_key,
    azure_endpoint,
    n_copies=5  # Number of times to duplicate datasets
):
    """
    Run performance profiling experiments
    
    Parameters:
    tracker: Experiment tracker object
    experiments: Experiment directory from tracker
    input_path: Path to input h5ad file
    api_key: Azure OpenAI API key
    azure_endpoint: Azure OpenAI endpoint
    n_copies: Number of copies to create for scaling test
    """
    results = []
    
    # Initialize OpenAI client for embeddings
    if api_key is not None:
        client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_endpoint
    )

    # Load initial data
    base_adata = sc.read(input_path)
    
    # expand the raw data
    base_adata = ad.concat({name: base_adata.copy() for name in ['a','b','c']}, index_unique="-")
    print(f"Shape of the concatenated data: {base_adata.shape}")
    
    base_adata = base_adata.raw.to_adata()
    
    # Run experiments with increasing dataset sizes
    for i in range(n_copies):
        print(f"=== Running experiment with {i+1} copies of the dataset ===")
        
        # Create copied dataset
        if i == 0:
            adata = base_adata.copy()
        else:
            # Concatenate original dataset i+1 times
            adatas = [base_adata.copy() for _ in range(i+1)]
            # Add suffix to study names to make them unique
            for idx, _adata in enumerate(adatas):
                _adata.obs['study'] = _adata.obs['study'].astype(str) + f'_{idx}'
                _adata.obs.index = _adata.obs.index.astype(str) + f'_{idx}'
            adata = ad.concat(adatas)
        
        # Record dataset stats
        n_cells = adata.n_obs
        n_datasets = len(adata.obs['study'].unique())
        
        # Preprocess for all methods
        adata_prep = adata.copy()
        sc.pp.log1p(adata_prep)
        sc.pp.highly_variable_genes(adata_prep, subset=True, batch_key='study')
        sc.pp.scale(adata_prep)
        sc.tl.pca(adata_prep)
        
        # Profile CellHint
        print("Profiling CellHint...")
        cellhint_time = profile_method(
            "cellhint",
            cellhint.harmonize,
            adata_prep,
            dataset='study',
            cell_type='cell_type',
            use_rep='X_pca',
            use_pct=True,
        )
        
        # Profile CellHint Prior
        print("Profiling CellHint Prior...")
        input_list = list(set(adata.obs.cell_type.unique().tolist()))
        
        # Get embeddings for cell types
        if api_key is not None:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=input_list
            )
            embedding_dict = {key: value.embedding for key, value in zip(input_list, response.data)}
        else:
            # Use dummy embeddings
            embedding_dict = {key: np.ones(100) for key in input_list}
        
        cellhint_prior_time = profile_method(
            "cellhint_prior",
            cellhint_prior.harmonize,
            adata_prep,
            dataset='study',
            cell_type='cell_type',
            use_rep='X_pca',
            use_pct=True,
            embedding_dict=embedding_dict
        )

        del adata_prep
        gc.collect()
        
        # Profile Scanorama
        print("Profiling Scanorama...")
        adata_scan = adata.copy()
        sc.pp.log1p(adata_scan)
        sc.pp.highly_variable_genes(adata_scan, subset=True, batch_key='study')
        
        # Prepare data for Scanorama
        batches = adata_scan.obs['study'].cat.categories.tolist()
        alldata = {batch: adata_scan[adata_scan.obs['study'] == batch,] for batch in batches}
        adatas = list(alldata.values())
        
        del adata_scan
        gc.collect()
        
        scanorama_time = profile_method(
            "scanorama",
            scanorama.integrate_scanpy,
            adatas,
            knn=30,
            approx=False,
            batch_size=1000
        )
        
        # Profile Scanorama Prior
        print("Profiling Scanorama Prior...")
        # Prepare similarity matrix
        input_list = list(set(adata.obs.cell_type.unique().tolist()))
        if api_key is not None:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=input_list
            )
            emb = np.array([x.embedding for x in response.data])
            similarity_matrix = pd.DataFrame(emb.dot(emb.T), index=input_list, columns=input_list)
        else:
            # Use dummy similarity matrix
            similarity_matrix = pd.DataFrame(np.eye(len(input_list)), index=input_list, columns=input_list)
        
        scanorama_prior_time = profile_method(
            "scanorama_prior",
            scanorama_prior.integrate_scanpy,
            adatas,
            type_similarity_matrix=similarity_matrix,
            knn=30,
            approx=False,
            use_gpu=True,
            batch_size=1000
        )
        
        # Record results
        experiment_result = {
            'n_datasets': n_datasets,
            'n_cells': n_cells,
            'cellhint_time': cellhint_time,
            'cellhint_prior_time': cellhint_prior_time,
            'scanorama_time': scanorama_time,
            'scanorama_prior_time': scanorama_prior_time,
            'total_prior_time': cellhint_prior_time + scanorama_prior_time
        }
        results.append(experiment_result)
        
        del adatas
        gc.collect()
        
        # Save results after each iteration
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'performance_results.csv'), index=False)
        
        # Print current results
        print(f"\nResults for iteration {i+1}:")
        print(f"Number of datasets: {n_datasets}")
        print(f"Number of cells: {n_cells}")
        print(f"CellHint time: {cellhint_time:.2f}s")
        print(f"CellHint Prior time: {cellhint_prior_time:.2f}s")
        print(f"Scanorama time: {scanorama_time:.2f}s")
        print(f"Scanorama Prior time: {scanorama_prior_time:.2f}s")
        print(f"Total Prior time: {cellhint_prior_time + scanorama_prior_time:.2f}s")
    
    return results_df

# Example usage
if __name__ == "__main__":
    results = run_performance_experiment(
        input_path='/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/pancreas.h5ad',
        output_dir='/home/wu/datb1/AutoExtractSingleCell/scanorama_prior/data/time_usage',
        api_key="",
        azure_endpoint="https://genomic-openai-ca.openai.azure.com/"
    )
    
    print("\nFinal Performance results:")
    print(results)
    
    print("Performance results:")
    print(results)