# Welcome to Biotools docummentaion
Version 1 , January 2022

The main aim of this module is to make the life of molecular biologists easier, thus it adapts well established python functions to biological needs.
These functions were tested on transcriptomics, metabolomics, proteomics and phospho-proteomics datasets, in which the dataframes are usually composed of a matrix + a column corresponding the values of each row to a target of interest i.e. a gene, metabolite etc. Henceforth, the word 'features' will represent genes, metabolites, proteins etc.
This is the first version and I am open for ideas to improve this module : my mail, sezagouri@gmail.com.

# About me:
My name is Saar Ezagouri, a PhD student at the lab of Gad Asher / Weizmann Institute of Science / Israel. 
We study the role of circadian rhythms in metabolism, in particular I study the interaction between physical exercise and the circadian clock.
Outside of the lab I am sports registered dietitian and strength and conditioning trainer.
<a href="linkedin.com/in/saar-ezagouri-959a8b1a8"> My Linkdin </a>

# List of functions included:
**my_pca:** plot principal component analysis resutlts.
**cluster_calc:** calculates the optimal clusters based on K-means clustering.
**k_clustered_hm:** plot a k-means clustered heatmap, based on a desired number of clusters.
**top_bottom:** plot the most changing features in a given data, between conditions.
**plot_feature:** plot a feature across a given list of conditions.
**enrichr_clusters:** compare enrichment terms between group of genes i.e. clusters.

# Documentation:
**biotools.my_pca(df,conditions,pc_x,pc_y)**
Plot principal component analysis resutlts.
**Parameters:** df: a transposed dataframe. first column contains the conditions (experimental groups) and the rest are features.
                conditions: a string, the column name of the conditions in df.

**cluster_calc(matrix,k0,kn)** 
Calculates the optimal clusters based on K-means clustering. The functions scales the data to z-scores before clustering.
Prints out silhouette score for each cluster, and a scatter plot to estimate the goodness of seperation.
**Parameters:** matrix: a dataframe, containing only numerical values.
                k0,kn: an integer, the start and end of an array of clusteres to test.
**Returns:** scores, a df with silhouette scores for each clusters.

**k_clustered_hm(matrix,k)** 
Plot a k-means clustered heatmap, based on a desired number of clusters.
Scale the data based on z-scores by rows, before clustering.
**Parameters:** matrix: a dataframe, containing only numerical values.
                k: an integer, number of clusteres based on which the data will be clustered.
**Returns:** df_mat_clustered, a dataframe, similar to 'matrix' with an additional column corresponding each feature to its relevant cluster.

**top_bottom(df,names,groups,top,bottom)** 
Plot the most changing features in a given data, between conditions.
**Parameters:** df: a dataframe containing the data.
                names: a string, the name of the column containing the names of the features.
                groups: a list of 2 groups to compare.
                top, bottom: an integer, the number of results from the head and tail of a sorted dataframe in a descending order (head is highest).

**plot_feature(names,groups,targets,df,ylimits = 50)** 
Plot list of features across given conditions or experimental groups.
**Parameters:** names: list of features.
                groups: list of strings corresponds to column names to will be grouped together for calculating mean and sem.
                targets: a string, column name in dataframe that contains features names e.g. 'Gene Symbol' 
                df: a dataframe with the data.
                ylimits: y axis limits can be changed for better visualization
                
**enrichr_clusters (gene_list,organism,gene_set,background=20000,cutoff=0.05)** 
Compare enrichment terms between group of genes i.e. clusters.
This function has based on GSEAPY's enrichr, but is adapted to publishing considerations where a user will want to spot enrichment differences between groups.
Unlike GSEApy's enrichr, this function compare clusters per dataset (gene_set) and thus does not support multiple datasets as an input. What I usually do is use this function inside a for loop as follows:
import gseapy as gp
datasets = pd.Dataframe(gp.get_library_name(organism='Mouse')) # can be 'Human','Yeast' etc.
These libaries are highly redundant, and you may prefer to filter this dataframe before looping through 192 databases. You can easily do so based on the names of the datasets and then:
for gene_set in len(range(datasets)):
  enrichr_clusters(gene_list,organism,gene_set = gene_set, background=20000,cutoff=0.05)
**Parameters:** feature_list: a list of lists, each list contains feature names of a cluster e.g. a list of genes.
                organism: ‘Human’, ‘Mouse’, ‘Yeast’, ‘Fly’, ‘Fish’, ‘Worm’ 
                gene_set: a string, has to be an item from this list: gp.get_library_name().
                background=20000 (default) # e.g. 'hsapiens_gene_ensembl'. background to run the analysis against.
                cutoff=0.05 (default). significance threshold .
**Returns:** df_list, enrichment results of all clusteres analyzed for a given dataset.
For more information please refer to GSEApy's docummentation at <a href="https://gseapy.readthedocs.io/en/latest/gseapy_example.html#2.-Enrichr-Example"> GSEApy Docs </a>

# Downloads:
<a href="https://gseapy.readthedocs.io/en/latest/gseapy_example.html#2.-Enrichr-Example"> Biotools Repository </a>
# Dependency:
Numpy,Scipy,Pandas, Matplotlib, Seaborn, GSEAPY, sklearn.
The following are required for all functions of this module and can be copied to your code for your own convinience:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns





