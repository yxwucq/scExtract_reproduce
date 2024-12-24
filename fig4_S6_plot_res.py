import gc
import os
import seaborn as sns
import scanpy as sc
import pickle
import subprocess
from matplotlib import pyplot as plt
from scib_metrics.benchmark import Benchmarker

sc.set_figure_params(vector_friendly=True, dpi_save=600) 
os.makedirs('large_dataset_figures', exist_ok=True)

name_list = ['Immune_ALL_human', 'Lung_atlas_public', 'Blood', 'Spleen', 'HCLA', 'ORAL']
adata_list = ['Immune_ALL_human.h5ad', 'Lung_atlas_public.h5ad',
              'Blood.h5ad', 'Spleen.h5ad', 'human_lung_cell_atlas.h5ad',
              'oral_and_craniofacial_atlas.h5ad']



for name, adata_path in zip(name_list, adata_list):
    pkl_path = os.path.join("results", adata_path.replace('.h5ad', '_benchmark.pkl'))
    if not os.path.exists(pkl_path):
        continue

    with open(pkl_path, 'rb') as f:
        res = pickle.load(f)
    key = list(res.keys())[0]
    print(res)
    adata = res[key][1]._adata
    n_cells = adata.shape[0]
    n_ct = adata.obs['cell_type_raw'].nunique()
    n_ds = adata.obs['Dataset'].nunique()
    fig, axs = plt.subplots(1, 5, figsize=(12.5, 3))
    embedding_list = ['pca', 'cellhint', 'cellhint_prior', 'scanorama', 'scanorama_prior']
    if n_cells > 300000:
        dot_size = 0.1
    elif n_cells > 100000:
        dot_size = 1
    else:
        dot_size = 1
    for i, embedding in enumerate(embedding_list):
        legend_loc = None if i != 4 else 'right margin'
        legend_size = None if i != 4 else 1
        sc.pl.embedding(adata, basis=f"umap_{embedding}", color='Dataset', ax=axs[i], show=False, title=embedding,
                        frameon=False, legend_loc=legend_loc, legend_fontsize=legend_size, size=dot_size, alpha=0.8)
    plt.suptitle(f"{name}: {n_cells} Cellls, {n_ds} Dataset, {n_ct} Cell types", y=1.05)
    plt.tight_layout()
    plt.savefig(f'large_dataset_figures/{name}_umap.pdf', bbox_inches='tight')

    for i, embedding in enumerate(embedding_list):
        legend_loc = None if i != 4 else 'right margin'
        legend_size = None if i != 4 else 1
        sc.pl.embedding(adata, basis=f"umap_{embedding}", color='cell_type_raw', ax=axs[i], show=False, title=embedding,
                        frameon=False, legend_loc=legend_loc, legend_fontsize=legend_size, size=dot_size, alpha=0.8)
    plt.suptitle(f"{name}: {n_cells} Cellls, {n_ds} Dataset, {n_ct} Cell types", y=1.05)
    plt.tight_layout()
    plt.savefig(f'large_dataset_figures/_cell_type_{name}_umap.pdf', bbox_inches='tight')

    df_metrics = res[key][1].get_results(min_max_scale=False)

    sns.set_theme(style='white')
    df_plot = df_metrics.iloc[:-1].copy().reset_index()
    metrics = ['Total', 'KMeans ARI', 'Silhouette label', 'iLISI', 'KBET']

    fig, axes = plt.subplots(1, 6, figsize=(20, 3))

    scatter = sns.scatterplot(df_plot, x='Batch correction', y='Bio conservation', 
                            hue='Embedding', ax=axes[0], palette='husl')
    scatter.get_legend().remove()
    scatter.yaxis.set_ticks_position('left')
    scatter.xaxis.set_ticks_position('bottom')

    bars = []
    for idx, (ax, metric) in enumerate(zip(axes[1:].flat, metrics)):
        bar = sns.barplot(df_plot, x='Embedding', y=metric, hue='Embedding', legend=False, ax=ax, palette='husl')
        
        ax.set_xlabel('')
        ax.set_xticklabels([])
        bar.yaxis.set_ticks_position('left')
        bars.append(bar)

    handles, labels = scatter.get_legend_handles_labels()
    axes[-1].legend(handles, labels, title='Embedding', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join('large_dataset_figures', f'{name}_metrics.pdf'), bbox_inches='tight')
    plt.close()

    res[key][1]._results.columns = res[key][1]._results.columns.str.replace('X_umap_', '')
    res[key][1].plot_results_table(save_dir='large_dataset_figures', min_max_scale=False)
    subprocess.run(['mv', os.path.join('large_dataset_figures', 'scib_results.svg'),
        os.path.join('large_dataset_figures', f'{name}_scib_results.svg')])