import sys
sys.path.append('/home/wu/datb1/AutoExtractSingleCell/scExtract')

import re
import anndata as ad
import logging
import os
import warnings
import time

import pickle
from typing import Optional
from pyfiglet import Figlet
import colorama
from termcolor import colored
import configparser

from scextract.auto_extract.agent import Claude3, Openai, get_cell_type_embedding_by_llm
from scextract.auto_extract.parse_params import Params 
from scextract.auto_extract.preprocess import filter, preprocess, clustering
from scextract.auto_extract.annotation import get_marker_genes, annotate, query_datasets, simple_annotate
from scextract.auto_extract.parse_params import Params 

from scextract.utils.utils import convert_ensembl_to_symbol

from prompt_variation import VariationPrompt

def generate_chat_history_preprocess(
                 pdf_path: str, 
                 output_dir: str,
                 config_path: str = 'config.ini',
                #  output_name: str = 'processed.h5ad',
                 output_response_prefix: str = 'preprocess_resp',
                #  benchmark_no_context_key: Optional[str] = None,
                 ) -> None:
    
    config = configparser.ConfigParser()
    config.read(config_path)

    # if 'openai' in config['API']['TYPE']:
    #     claude_agent = Openai(pdf_path, config_path)
    # elif 'claude' in config['API']['TYPE']:
    #     claude_agent = Claude3(pdf_path, config_path)
    # else:
    #     raise ValueError(f"Model {config['API']['MODEL']} not supported.")
    
    print(f"Using {config['API']['MODEL']} as extraction model")
    params = Params(config_path)
    # claude_agent.initiate_propmt()
    
    with open(os.path.join(output_dir, output_response_prefix, 'original_response.txt'), 'w') as f:
        print("Processing original response...")
        for repeat in range(3):
            claude_agent = Openai(pdf_path, config_path)
            claude_agent.initiate_propmt()
            filter_response = claude_agent.chat(params.get_prompt('FILTER_PROMPT'))
            f.write("========= Repeat {} =========\n".format(repeat+1))
            f.write(filter_response)
            f.write('\n')
            # print(claude_agent.messages)
            # for _ in range(2):
            #     claude_agent.messages.pop()
            # print(claude_agent.messages)
            # sys.exit()
    
    filter_prompt_variants = VariationPrompt.PREPROCESS_PROMPTS
    for key, value in filter_prompt_variants.items():
        print(f"Processing {key}...")
        with open(os.path.join(output_dir, output_response_prefix, f'{key}.txt'), 'w') as f:
            for repeat in range(3):
                claude_agent = Openai(pdf_path, config_path)
                claude_agent.initiate_propmt()
                response = claude_agent.chat(value)
                f.write("========= Repeat {} =========\n".format(repeat+1))
                f.write(response)
                f.write('\n')
                # for _ in range(2):
                #     claude_agent.messages.pop()
                    
    print("Chat history generation completed.")


def generate_chat_history_annotation(adata_path: str,
                 pdf_path: str, 
                 output_dir: str,
                 config_path: str = 'config.ini',
                 output_name: str = 'clustered_unannotated.h5ad',
                 output_log: str = 'auto_extract.log',
                 output_config_pkl: str = 'config.pkl',
                 output_response_prefix: str = 'annotation_resp',
                 benchmark_no_context_key: Optional[str] = None,
                 ) -> None:
    """
    Extracts and processes single-cell data from literature.
    """
    
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please run 'init' to create a config file.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(os.path.join(output_dir, output_log)):
        os.remove(os.path.join(output_dir, output_log))

    config = configparser.ConfigParser()
    config.read(config_path)

    file_handler = logging.FileHandler(os.path.join(output_dir, output_log))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    f = Figlet(font='slant')
    # logging.info('\n'+f.renderText('scExtract')+f" v{__version__}"+'\n')
    colorama.init()
    logging.info(colored(f"Using {config['API']['MODEL']} as extraction model", color='cyan'))
    logging.info(colored(f"Using {config['API']['TOOL_MODEL']} as tool model", color='cyan'))
    
    # Load AnnData object
    logging.info(f'Loading AnnData object from {adata_path}')
    adata = ad.read_h5ad(adata_path)
    
    if adata.shape[0] > 50000:
        logging.warning(colored(f"Large dataset detected. The dataset contains {adata.shape[0]} cells. \
The result extracted by LLM may not be accurate. Please consider manually \
subsetting the data to a smaller size.", color='light_red'))
    
    # Check if Ensembl IDs are present in AnnData object
    for genes in adata.var.index[:10]:
        if genes.startswith('ENSG'):
            logging.info('Ensembl IDs detected in AnnData object.')
            adata = convert_ensembl_to_symbol(adata)
            adata.var_names_make_unique()
            break
    
    logging.info(colored('1. Extracting information from literature', color='cyan', attrs=['bold']))
    
    if 'openai' in config['API']['TYPE']:
        claude_agent = Openai(pdf_path, config_path)
    elif 'claude' in config['API']['TYPE']:
        claude_agent = Claude3(pdf_path, config_path)
    else:
        raise ValueError(f"Model {config['API']['MODEL']} not supported.")
    
    claude_agent.initiate_propmt()

    # Filter and preprocess data
    params = Params(config_path)
    
    logging.info(colored('2. Filtering and preprocessing data', color='cyan', attrs=['bold']))
    filter_response = claude_agent.chat(params.get_prompt('FILTER_PROMPT'))
    logging.info(filter_response)
    params.parse_response(filter_response)
    adata = filter(adata, params)

    logging.info(colored('3. Preprocessing data', color='cyan', attrs=['bold']))
    preprocess_response = claude_agent.chat(params.get_prompt('PREPROCESSING_PROMPT'))
    logging.info(preprocess_response)
    params.parse_response(preprocess_response)
    adata = preprocess(adata, params)
    
    # Clustering
    logging.info(colored('4. Clustering data', color='cyan', attrs=['bold']))
    clustering_response = claude_agent.chat(params.get_prompt('CLUSTERING_PROMPT'))
    logging.info(clustering_response)
    params.parse_response(clustering_response)
    adata = clustering(adata, params)

    # adata.write(os.path.join(output_dir, output_name))
    # import sys; sys.exit()

    if config['OPTIONS'].getboolean('CLEAN_INTERMEDIATE_MESSAGES'):
        logging.info(colored('Cleaning up intermediate messages', color='cyan', attrs=['bold']))
        claude_agent.clear_intermediate_messages()
    
    # Annotate
    logging.info(colored('5. Getting marker genes', color='cyan', attrs=['bold']))
    adata, marker_genes = get_marker_genes(adata, params, fast_mode=config['OPTIONS'].getboolean('FAST_MODE'))
    logging.info(colored('Top 10 marker genes for each cluster:', color='yellow'))
    logging.info(colored(marker_genes, color='yellow'))

    if config['OPTIONS'].getboolean('BENCHMARK_GPTCELLTYPE'):
        logging.info(colored('Benchmarking GPTCellType', color='cyan', attrs=['bold']))
        tissue_name = claude_agent.chat(params.get_prompt('GET_TISSUE_NAME_PROMPT'))
        logging.info(colored(f'Tissue name: {tissue_name}', color='yellow'))
        benchmark_gptcelltype_prompt = params.get_tool_prompt('GPTCELLTYPE_ANNOTATION_PROMPT')
        benchmark_gptcelltype_prompt += "\n".join([f"{k}:{','.join(v)}" for k, v in marker_genes.items()])
        benchmark_gptcelltype_response = claude_agent._tool_retrieve(messages=[{"role": "user", "content": benchmark_gptcelltype_prompt.replace('{tissuename}', tissue_name)}])
        logging.info(colored(benchmark_gptcelltype_response, color='green'))
        benchmark_gptcelltype_response_list = [x for x in benchmark_gptcelltype_response.split('\n') if x]
        benchmark_gptcelltype_response_list = [re.sub(r'^\d+:\s*', '', string) for string in benchmark_gptcelltype_response_list]
        assert len(benchmark_gptcelltype_response_list) == len(marker_genes), 'Number of responses does not match number of clusters.'
        gptcelltype_annotation_dict = {k: v for k, v in enumerate(benchmark_gptcelltype_response_list)}
        adata = simple_annotate(adata, gptcelltype_annotation_dict, params, 'gptcelltype_annotation')

    if benchmark_no_context_key is not None:
        benchmark_no_context_prompt = params.get_tool_prompt('NO_CONTEXT_ANNOTATION_PROMPT').replace('Some can be a mixture of multiple cell types.',
                                                                                        'Some can be a mixture of multiple cell types.' + str(marker_genes))
        benchmark_no_context_summary = claude_agent._tool_retrieve(messages=[{"role": "user", "content": benchmark_no_context_prompt}])
        logging.info(colored(benchmark_no_context_summary, color='green'))
        no_context_annotation_dict = params.parse_annotation_response(benchmark_no_context_summary, simple_annotation=True)
        adata = simple_annotate(adata, no_context_annotation_dict, params, benchmark_no_context_key)
    
    starting_part = 'This is the output of the top 10 marker genes for each cluster:'
    logging.info(colored('6. Annotating clusters', color='cyan', attrs=['bold']))
    
    with open(os.path.join(output_dir, output_response_prefix, 'original_response.txt'), 'w') as f:
        print("Processing original response...")
        for repeat in range(3):
            annotate_prompt = params.get_prompt('ANNOTATION_PROMPT').replace(f"{starting_part}", 
                                                                    f"{starting_part}\n{marker_genes}")
            annotate_response = claude_agent.chat(annotate_prompt)
            f.write("========= Repeat {} =========\n".format(repeat+1))
            f.write(annotate_response)
            f.write('\n')
            for _ in range(2):
                claude_agent.messages.pop()
    
    annotate_prompt_variants = VariationPrompt.ANNOTATION_PROMPTS
    for key, value in annotate_prompt_variants.items():
        print(f"Processing {key}...")
        with open(os.path.join(output_dir, output_response_prefix, f'{key}.txt'), 'w') as f:
            for repeat in range(3):
                value = value.replace(f"{starting_part}", f"{starting_part}\n{marker_genes}")
                response = claude_agent.chat(value)
                f.write("========= Repeat {} =========\n".format(repeat+1))
                f.write(response)
                f.write('\n')
                for _ in range(2):
                    claude_agent.messages.pop()

    adata.write(os.path.join(output_dir, output_name))

    logging.info(colored(f'Processed data saved to {output_dir}/{output_name}', color='cyan'))
    logging.info(colored('Chat history generation completed.', color='cyan', attrs=['bold']))

if __name__ == "__main__":
    adata_path = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/sample11_revision/raw_data/sample11_raw.h5ad'
    pdf_path = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/sample11_revision/raw_data/sample11.pdf'
    output_dir = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/prompts_variants/deepseek_v3/'
    config_path = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/config_deepseek.ini'
    
    # adata_path = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/sample1_revision/raw_data/sample1_raw.h5ad'
    # pdf_path = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/sample1_revision/raw_data/sample1.pdf'
    # output_dir = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/sample1_text_downsample_new'
    # config_path = '/home/wu/datb1/AutoExtractSingleCell/01.benchmark_datasets/sample1_text_downsample_new/config_claude_35.ini'
    
    # generate_chat_history_preprocess(pdf_path, output_dir, config_path)
    generate_chat_history_annotation(adata_path=adata_path,
                                    pdf_path=pdf_path,
                                     output_dir=output_dir,
                                     output_name='clustered_unannotated.h5ad',
                                        config_path=config_path)