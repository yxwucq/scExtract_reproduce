#!/usr/bin/env python3

import gc
import os
import scanpy as sc
import argparse
import tempfile
from scextract.integration import integrate

def process_dataset(adata_path, output_path):
    """
    Process a single dataset with CellHint integration.
    
    Parameters:
    -----------
    adata_path : str
        Path to the input h5ad file
    output_path : str
        Path to the final output h5ad file
    """
    print(f"Processing {adata_path}")
    
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create path for temporary file
        temp_path = os.path.join(temp_dir, 'temp.h5ad')
        
        # First integration with CellHint
        print("Running CellHint integration...")
        integrate.integrate_processed_datasets(
            file_list=[adata_path],
            output_path=temp_path,
            method='cellhint',
            dimred=50,
        )
        
        # Process first integration results
        print("Processing initial results...")
        adata = sc.read(temp_path)
        sc.tl.umap(adata)
        adata.obs['cell_type'] = adata.obs['cell_type_raw'].copy()
        adata.obsm['X_umap_cellhint'] = adata.obsm['X_umap'].copy()
        
        # Save to a different temporary file
        temp_path2 = os.path.join(temp_dir, 'temp2.h5ad')
        adata.write(temp_path2)
        
        # Clean up memory
        del adata
        gc.collect()
        
        os.remove(temp_path)

        # Second integration with CellHint prior
        print("Running CellHint prior integration...")
        embedding_path = adata_path.replace('.h5ad', '_embedding.pkl')
        integrate.integrate_processed_datasets(
            file_list=[temp_path2],
            output_path=temp_path,
            method='cellhint_prior',
            embedding_dict_path=embedding_path,
            dimred=50,
        )
        
        # Process and save final results
        print("Processing final results...")
        adata = sc.read(temp_path)
        sc.tl.umap(adata)
        adata.obsm['X_umap_cellhint_prior'] = adata.obsm['X_umap'].copy()
        
        # Write final results to the specified output path
        print(f"Saving final results to {output_path}")
        adata.write(output_path)
        
        # Clean up memory
        del adata
        gc.collect()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process dataset with CellHint integration')
    parser.add_argument('adata_path', type=str, help='Path to the input h5ad file')
    parser.add_argument('output_path', type=str, help='Path to the output h5ad file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.adata_path):
        raise FileNotFoundError(f"Input file not found: {args.adata_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Process the dataset
    process_dataset(args.adata_path, args.output_path)
    print("Processing completed successfully")

if __name__ == '__main__':
    main()