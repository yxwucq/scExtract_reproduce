class VariationPrompt:
    ANNOTATION_PROMPTS = {
        # Prompt variations 1
        'Instruction_Reordering': """
Based on gene expression and the detailed discussion from the article, annotate these clusters into cell types using a dictionary format.
This is the output of the top 10 marker genes for each cluster:

Please provide the 'cell type', 'certainty', 'source' and reasoning for each cluster.
You may annotate different groups with the same cell type. You should try to assign a **cell ontology** label to each cluster (e.g. B cell, T cell, etc.),
with modification to make your annotations more concordant with the original paper (e.g. 'CD4+ T cell' or 'T cell 2').
If you cannot tell the cell type, name it as 'Unknown'. Be sure to provide reasoning for the annotation.

OUTPUT_FORMAT (description of the parameters is in the curly braces, do not include the description in the output,
each value in the [] should be quoted so that it is clear that it is a string value):
annotation_dict: {0: [cell_type, 
        certainty, (value chosen from [Low, Medium, High])
        source, (value chosen from [Article-defined, Knowledge-based])
        ...}. "

reasoning: {str, reasoning for the re-annotation}

<example>
<response>
annotation_dict: {0: ['T cell', 'High', 'Article-defined'],
                    1: ['B cell', 'Medium', 'Knowledge-based']}
                    
reasoning: The expression of CD3E and CD3D is high in cluster 0, which is a typical marker of T cells. The expression of CD19 is medium high in cluster 1, which is a typical marker of B cells, but not as high as in cluster 2
</response>
</example>
""",    
        # Prompt variations 2
        'Terminology_Modification': """
This is the output of the top 10 marker genes for each cluster:
        
Analyze the gene expression patterns and information from the research article to assign cell type identities to these clusters.
Please create annotations that include 'cell type', 'confidence level', 'information source' and justification for each cluster.
You can label multiple clusters as the same cell type if appropriate. Assign standard **cell ontology** terminology to each cluster (such as B cell, T cell, etc.),
with specific subtypes when supported by the data or article (like 'CD8+ T cell' or 'Memory B cell').
Label as 'Unidentified' if insufficient evidence exists. Always include your reasoning behind each annotation.
        
RESPONSE_FORMAT (exclude descriptions in curly braces from your output,
ensure all values within [] are properly quoted as strings):
annotation_dict: {0: [cell_type, 
        confidence_level, (select from [Tentative, Moderate, Strong])
        information_source, (select from [Paper-referenced, Background-knowledge])
        ...}. "
                
justification: {str, explanation for cell type assignments}
        
<example>
<response>
annotation_dict: {0: ['T cell', 'Strong', 'Paper-referenced'],
                    1: ['B cell', 'Moderate', 'Background-knowledge']}
                    
justification: Cluster 0 shows strong expression of T cell markers including CD3E and CD3D, consistent with T cell identity. Cluster 1 exhibits moderate expression of CD19, a canonical B cell marker, though not as pronounced as seen in cluster 2
</response>
</example>
""",
        # Prompt variations 3
        'Enhanced_Guidance': """
This is the output of the top 10 marker genes for each cluster:

Your task is to annotate these clusters with appropriate cell types based on:
1. The marker gene expression patterns provided
2. Information from the referenced article
3. Standard cell biology knowledge

For each cluster, provide:
- The most likely cell type (use established cell ontology when possible)
- Your confidence in the annotation (High/Medium/Low)
- Whether your annotation is based primarily on the article or on general knowledge
- A clear explanation of your reasoning

You may assign the same cell type to different clusters if the evidence supports it.
If evidence is insufficient, label the cluster as "Unknown".

OUTPUT_FORMAT:
annotation_dict: {
    cluster_number: ["cell type", "confidence_level", "primary_source"],
    ...
}

reasoning: Detailed explanation of the evidence and logic behind each annotation

<example>
<response>
annotation_dict: {
    0: ["CD4+ T cell", "High", "Article-defined"],
    1: ["Naive B cell", "Medium", "Knowledge-based"],
    2: ["Plasma cell", "High", "Article-defined"]
}

reasoning: Cluster 0 expresses CD3D, CD3E, and CD4, which are definitive markers for CD4+ T cells. The article also specifically identifies this population. Cluster 1 shows CD19, CD20 (MS4A1), and IgD expression but lacks plasma cell markers, suggesting naive B cells, though the article doesn't specifically label this subset. Cluster 2 strongly expresses CD19, CD38, and immunoglobulin genes with downregulated MS4A1, classic for plasma cells, matching the article's description.
</response>
</example>
""",
        # Prompt variations 4
        'Minimalistic_Annotation': """This is the output of the top 10 marker genes for each cluster:

Please annotate these clusters with cell types based on the gene expression data and article information.

For each cluster, include:
- Cell type (use standard cell ontology terms)
- Confidence (High, Medium, or Low)
- Source (Article-defined or Knowledge-based)

Provide your reasoning for each annotation.

REQUIRED FORMAT:
annotation_dict: {cluster_id: ["cell_type", "confidence", "source"], ...}
reasoning: Your explanation for the annotations

<example>
<response>
annotation_dict: {0: ["T cell", "High", "Article-defined"],
                 1: ["B cell", "Medium", "Knowledge-based"]}
                 
reasoning: Cluster 0 shows high expression of T cell markers CD3E and CD3D. Cluster 1 has moderate expression of the B cell marker CD19.
</response>
</example>
""",
        # Prompt variations 5
        'Structural_Reformatting': """This is the output of the top 10 marker genes for each cluster:

Based on these marker genes and the article's discussion, perform cell type annotation for each cluster.

INSTRUCTIONS:
- Use established cell ontology terminology when possible
- Make annotations consistent with the paper's terminology where appropriate
- Provide confidence level (Low/Medium/High) for each annotation
- Indicate if your annotation is primarily based on the article or general knowledge
- Include detailed reasoning for each cluster
- Label clusters as "Unknown" when evidence is insufficient
- Different clusters may be assigned the same cell type if warranted

FORMAT YOUR RESPONSE AS JSON:
{
  "annotations": [
    {
      "cluster": 0,
      "cell_type": "string",
      "confidence": "string", // Must be Low, Medium, or High
      "source": "string", // Must be Article-defined or Knowledge-based
      "reasoning": "string"
    },
    ...
  ],
  "summary": "string" // Brief overview of your annotation approach
}

<example>
<response>
{
  "annotations": [
    {
      "cluster": 0,
      "cell_type": "T cell",
      "confidence": "High",
      "source": "Article-defined",
      "reasoning": "High expression of CD3E and CD3D, which are canonical T cell markers"
    },
    {
      "cluster": 1,
      "cell_type": "B cell",
      "confidence": "Medium",
      "source": "Knowledge-based",
      "reasoning": "Medium expression of CD19, a B cell marker, though not as high as in cluster 2"
    }
  ],
  "summary": "Annotations were primarily based on expression of canonical immune cell markers and corroborated with article descriptions where available"
}
</response>
</example>
"""}
    
    PREPROCESS_PROMPTS = {
      # Prompt variations 1
      'Instruction_Reordering': """Replace the placeholders {} in the following preprocessing parameters with the values used in the article, based on the processing workflow for single-cell datasets described.
If the article doesn't specify a value, use 'default'. 
If the filtering is likely not used in the article, use 'null'. Be sure to provide reasoning for the filtering parameters.

OUTPUT_FORMAT(description of the parameters is after the colon, do not include the description in the output):
filter_cells_min_genes: {int|default|null, minimum number of genes expressed in a cell, usually around 300}
filter_cells_max_genes: {int|default|null, maximum number of genes expressed in a cell, usually around 5000}
filter_cells_min_counts: {int|default|null, minimum allowed total counts per cell usually null}
filter_cells_max_counts: {int|default|null, maximum allowed total counts per cell}
filter_genes_min_cells: {int|default|null, minimum number of cells expressing a gene}
filter_mito_percentage_min: {int|default|null, minimum mitochondrial gene percentage (0,100), usually null}
filter_mito_percentage_max: {int|default|null, maximum mitochondrial gene percentage (0,100)}
filter_ribo_percentage_min: {int|null, minimum ribosomal gene percentage (0,100), usually null}
filter_ribo_percentage_max: {int|null, maximum ribosomal gene percentage (0,100), usually null}

reasoning: {str, reasoning for the filtering parameters}

This is an example of how to extract filter_cells_min_genes, other arguments are extracted in the same way:
<example>
<text>
We filter out cells expressing fewer than 300 genes
</text>
The output should be:
<response>
filter_cells_min_genes: 300

reasoning: The article mentions that 'We filter out cells expressing fewer than 300 genes'
</response>
</example>

Following the processing workflow described in the article for single-cell datasets:
""",
      # Prompt variations 2
      'Terminology_Modification': """Examine the single-cell data preprocessing protocol described in the publication and identify the quality control thresholds.
Complete the preprocessing configuration by replacing the {} placeholders with the exact values mentioned in the article. 
When a specific threshold is not explicitly mentioned, input 'standard'. If a filtering step appears to be omitted entirely, input 'omitted'. 
Provide justification for each parameter selection based on text evidence.

CONFIGURATION_FORMAT(parameter descriptions follow the colons, exclude these descriptions from your response):
cell_gene_count_lower_bound: {integer|standard|omitted, minimum gene count per cell for inclusion, typically ~300}
cell_gene_count_upper_bound: {integer|standard|omitted, maximum gene count per cell for inclusion, typically ~5000}
total_molecule_count_lower_threshold: {integer|standard|omitted, minimum UMI/read count per cell}
total_molecule_count_upper_threshold: {integer|standard|omitted, maximum UMI/read count per cell}
gene_detection_cell_minimum: {integer|standard|omitted, minimum cells where gene must be detected}
mt_gene_proportion_lower_limit: {integer|standard|omitted, minimum mitochondrial gene percentage, typically omitted}
mt_gene_proportion_upper_limit: {integer|standard|omitted, maximum mitochondrial gene percentage allowed}
ribosomal_content_lower_limit: {integer|omitted, minimum ribosomal gene percentage, typically omitted}
ribosomal_content_upper_limit: {integer|omitted, maximum ribosomal gene percentage, typically omitted}

rationale: {text, evidence-based explanation for parameter selections}

Parameter extraction example:
<example>
<text>
We filter out cells expressing fewer than 300 genes
</text>
The output should be:
<response>
cell_gene_count_lower_bound: 300

rationale: The article explicitly states: 'We filter out cells expressing fewer than 300 genes'
</response>
</example>""",
      # Prompt variations 3
      'Enhanced_Guidance': """Following the single-cell RNA-seq preprocessing workflow described in the article, extract the quality control filtering parameters used.

Your task:
1. Carefully read the Methods or Results sections that describe data preprocessing
2. Identify specific numerical thresholds mentioned for cell and gene filtering
3. Replace each {} placeholder with the exact value from the article
4. Use 'default' if a parameter is mentioned but no specific value is given
5. Use 'null' only when you can confidently determine a filter was not applied
6. Provide clear reasoning for each parameter, citing specific text from the article

Important guidance:
- Look for phrases like "we filtered cells with fewer than X genes" or "cells with >X% mitochondrial reads were removed"
- Authors often describe filtering in terms of what was removed, not what was kept
- If a range is given (e.g., "cells with 500-5000 genes"), use the boundaries as min/max values
- Values must be consistent with the article's experimental design and cell types

OUTPUT_FORMAT (provide values only, not the descriptions):
filter_cells_min_genes: {int|default|null, minimum number of genes expressed in a cell, usually around 300}
filter_cells_max_genes: {int|default|null, maximum number of genes expressed in a cell, usually around 5000}
filter_cells_min_counts: {int|default|null, minimum allowed total counts per cell usually null}
filter_cells_max_counts: {int|default|null, maximum allowed total counts per cell}
filter_genes_min_cells: {int|default|null, minimum number of cells expressing a gene}
filter_mito_percentage_min: {int|default|null, minimum mitochondrial gene percentage (0,100), usually null}
filter_mito_percentage_max: {int|default|null, maximum mitochondrial gene percentage (0,100)}
filter_ribo_percentage_min: {int|null, minimum ribosomal gene percentage (0,100), usually null}
filter_ribo_percentage_max: {int|null, maximum ribosomal gene percentage (0,100), usually null}

reasoning: {Detailed explanation with direct quotes from the article supporting each parameter choice}

Example of correct parameter extraction:
<example>
<text>
In the quality control step, we filtered out cells expressing fewer than 300 genes. Additionally, we removed cells with more than 10% mitochondrial reads to exclude low-quality or dying cells.
</text>
The output should be:
<response>
filter_cells_min_genes: 300
filter_cells_max_genes: null
filter_cells_min_counts: null
filter_cells_max_counts: null
filter_genes_min_cells: null
filter_mito_percentage_min: null
filter_mito_percentage_max: 10
filter_ribo_percentage_min: null
filter_ribo_percentage_max: null

reasoning: The article explicitly states "we filtered out cells expressing fewer than 300 genes" which establishes filter_cells_min_genes as 300. It also mentions "we removed cells with more than 10% mitochondrial reads" which sets filter_mito_percentage_max as 10. No other filtering parameters were mentioned in the article, so they are set to null.
</response>
</example>""",
      # Prompt variations 4
      'Minimalistic_Annotation': """Extract the single-cell preprocessing parameters from the article.
Replace {} with values from the article, 'default' if unspecified, or 'null' if unused.

Parameters:
filter_cells_min_genes: {int|default|null}
filter_cells_max_genes: {int|default|null}
filter_cells_min_counts: {int|default|null}
filter_cells_max_counts: {int|default|null}
filter_genes_min_cells: {int|default|null}
filter_mito_percentage_min: {int|default|null}
filter_mito_percentage_max: {int|default|null}
filter_ribo_percentage_min: {int|null}
filter_ribo_percentage_max: {int|null}

reasoning: {brief explanation}

Example:
<example>
<text>We filter out cells expressing fewer than 300 genes</text>
<response>
filter_cells_min_genes: 300

reasoning: Article states "We filter out cells expressing fewer than 300 genes"
</response>
</example>
""",
      # Prompt variations 5
      'Structural_Reformatting': """Following the processing workflow described in the article for single-cell datasets, identify all quality control filtering parameters. 

INSTRUCTIONS:
Analyze the article for specific filtering thresholds applied to:
- Gene counts per cell (minimum and maximum)
- Total RNA counts/UMIs per cell
- Cell counts per gene
- Mitochondrial gene percentage
- Ribosomal gene percentage

FORMAT YOUR RESPONSE AS JSON:
{
  "filtering_parameters": {
    "cells": {
      "min_genes": value or "default" or null,
      "max_genes": value or "default" or null,
      "min_counts": value or "default" or null,
      "max_counts": value or "default" or null
    },
    "genes": {
      "min_cells": value or "default" or null
    },
    "percentages": {
      "mitochondrial": {
        "min": value or "default" or null,
        "max": value or "default" or null
      },
      "ribosomal": {
        "min": value or "default" or null,
        "max": value or "default" or null
      }
    }
  },
  "explanation": "Detailed reasoning for each parameter based on article text"
}

EXAMPLE:
<example>
<text>
We filter out cells expressing fewer than 300 genes
</text>
<response>
{
  "filtering_parameters": {
    "cells": {
      "min_genes": 300,
      "max_genes": null,
      "min_counts": null,
      "max_counts": null
    },
    "genes": {
      "min_cells": null
    },
    "percentages": {
      "mitochondrial": {
        "min": null,
        "max": null
      },
      "ribosomal": {
        "min": null,
        "max": null
      }
    }
  },
  "explanation": "The article explicitly mentions 'We filter out cells expressing fewer than 300 genes', which establishes the minimum gene count threshold at 300. No other filtering parameters are mentioned in the article."
}
</response>
</example>"""
    }
    
    