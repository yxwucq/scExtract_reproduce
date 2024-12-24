#!/usr/bin/env python3

import gc
import os
import pickle
import scanpy as sc
import argparse
import tempfile
from scib_metrics.benchmark import Benchmarker
from scextract.integration import integrate

def process_and_benchmark(input_path, results_output_path):
    """
    Process a dataset with Scanorama integration and run benchmarking.
    
    Parameters:
    -----------
    input_path : str
        Path to the input h5ad file
    results_output_path : str
        Path where to save the results dictionary
    """
    results_dict = {}
    name = os.path.basename(input_path).replace('.h5ad', '')
    print(f"Processing {name}")

    # Generate paths for final output and embedding
    output_path = input_path.replace('_integrated_tmp.h5ad', '_integrated.h5ad')
    embedding_path = input_path.replace('_integrated_tmp.h5ad', '_harmonized_embedding.pkl')

    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create paths for temporary files
        temp_path1 = os.path.join(temp_dir, 'temp1.h5ad')
        temp_path2 = os.path.join(temp_dir, 'temp2.h5ad')

        # First integration with Scanorama
        print("Running Scanorama integration...")
        integrate.integrate_processed_datasets(
            file_list=[input_path],
            output_path=temp_path1,
            method='scanorama',
            dimred=100,
            batch_size=2000,
        )

        # Process first integration results
        print("Processing initial results...")
        adata = sc.read(temp_path1)
        adata.obsm['X_umap_scanorama'] = adata.obsm['X_umap'].copy()
        adata.write(temp_path2)

        del adata
        gc.collect()

        # Second integration with Scanorama prior
        print("Running Scanorama prior integration...")
        integrate.integrate_processed_datasets(
            file_list=[temp_path2],
            output_path=output_path,  # Write directly to final output
            method='scanorama_prior',
            embedding_dict_path=embedding_path,
            use_gpu=False,
            dimred=100,
            batch_size=2000,
        )

        # Process final results
        print("Processing final results...")
        adata = sc.read(output_path)
        adata.obsm['X_umap_scanorama_prior'] = adata.obsm['X_umap'].copy()
        
        # Calculate PCA-based UMAP
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=30)
        sc.tl.umap(adata)
        adata.obsm['X_umap_pca'] = adata.obsm['X_umap'].copy()

        # Save updated final results
        adata.write(output_path)

        # Run benchmarking
        print("Running benchmarking...")
        bm = Benchmarker(
            adata,
            batch_key="Dataset",
            label_key="cell_type_raw",
            embedding_obsm_keys=['X_umap_pca', 'X_umap_cellhint', 
                               'X_umap_cellhint_prior', 'X_umap_scanorama', 
                               'X_umap_scanorama_prior'],
            n_jobs=-1,
        )

        bm.benchmark()
        results_dict[name] = (adata.shape, bm)
        print(bm.get_results())

        del adata
        gc.collect()

    # Save benchmark results
    print(f"Saving benchmark results to {results_output_path}")
    with open(results_output_path, 'wb') as f:
        pickle.dump(results_dict, f)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process dataset with Scanorama integration and run benchmarking'
    )
    parser.add_argument('input_path', type=str, 
                       help='Path to the input h5ad file')
    parser.add_argument('results_output_path', type=str, 
                       help='Path where to save the results dictionary (should end with .pkl)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.results_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the dataset and run benchmarking
    process_and_benchmark(args.input_path, args.results_output_path)
    print("Processing completed successfully")

if __name__ == '__main__':
    main()