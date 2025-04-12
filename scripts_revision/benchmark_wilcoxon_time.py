#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import scanpy as sc
import pandas as pd

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate DEGs from h5ad files and record processing time')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing h5ad files')
    parser.add_argument('--output', type=str, required=True, help='Output metrics file path')
    return parser.parse_args()

def find_h5ad_files(directory):
    """Find all h5ad files in the directory"""
    h5ad_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5ad'):
                h5ad_files.append(os.path.join(root, file))
    
    print(f"Found {len(h5ad_files)} h5ad files in {directory}")
    return h5ad_files

def process_h5ad_file(file_path):
    """Process a single h5ad file and return metrics"""
    
    # Read h5ad file
    adata = sc.read_h5ad(file_path)
    
    # Get file info
    file_name = os.path.basename(file_path)
    cell_count = adata.shape[0]
    
    group_key = 'leiden' if 'leiden' in adata.obs.columns else 'louvain'    
    start_time = time.time()
    sc.tl.rank_genes_groups(adata, group_key, method='wilcoxon', use_raw=False, tie_correct=True)
    processing_time = time.time() - start_time
    
    return {
        'file_name': file_name,
        'cell_count': cell_count,
        'processing_time': processing_time
    }

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Find all h5ad files
    h5ad_files = [x for x in os.listdir(args.input_dir) if x.endswith('.h5ad')]
    h5ad_files = sorted(h5ad_files)[:10]  # Limit to first 10 files for testing
    print(f"Processing {len(h5ad_files)} h5ad files...")
    
    if not h5ad_files:
        print(f"No h5ad files found in {args.input_dir}")
        return
    
    # Process all h5ad files
    results = []
    for file_path in h5ad_files:
        file_path = os.path.join(args.input_dir, file_path)
        print(f"Processing: {file_path}")
        result = process_h5ad_file(file_path)
        results.append(result)
        print(f"Completed {result['file_name']}, cells: {result['cell_count']}, time: {result['processing_time']:.2f}s")
    
    # Create and save results dataframe
    metrics_df = pd.DataFrame(results)
    
    # # Create output directory if needed
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save results
    metrics_df.to_csv(args.output, index=False)
    print(f"Metrics saved to: {args.output}")

if __name__ == "__main__":
    main()