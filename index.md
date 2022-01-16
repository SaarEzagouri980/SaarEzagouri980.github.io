# Welcome to BioPyTools Documentation
Version 1 , January 2022 <br>
© Copyright 2022, Saar Ezagouri <br>
Created with the help and under the supervision of <a href="https://code-maven.com/services"> Mr. Gabor Szabo </a>.

The main aim of this module is to make the life of molecular biologists easier, thus it tailors well established python functions to daily needs of biologists. <br>
These functions were tested on transcriptomics, metabolomics, proteomics and phospho-proteomics databases, in which the dataframes are often composed of a matrix + a column of keys or value identifiers i.e. a gene, metabolite, protein etc. <br>
Henceforth, the word 'features' will represent genes, metabolites, proteins etc. <br>

This is the first version and I am open for ideas of how this module can be improved. <br>
my mail: sezagouri@gmail.com.

# About me: <br>
My name is Saar Ezagouri, a PhD student at the lab of Gad Asher / Weizmann Institute of Science / Israel. <br>
We study the role of circadian rhythms in metabolism, in particular I study the interaction between physical exercise and the circadian clock. <br>
Outside of the lab I am sports registered dietitian (RD) and strength and conditioning trainer. <br>
<a href="https://www.linkedin.com/in/saar-ezagouri-959a8b1a8/"> My LinkedIn </a>

# List of functions included: <br>
- **my_pca:** plot principal component analysis resutlts. <br>
- **cluster_calc:** calculates the optimal clusters based on K-means clustering. <br>
- **k_clustered_hm:** plot a k-means clustered heatmap, based on a desired number of clusters. <br>
- **top_bottom:** plot the most changing features in a given data, between conditions. <br>
- **plot_feature:** plot a feature across a given list of conditions. <br>
- **enrichr_clusters:** compare enrichment terms between group of genes i.e. clusters. <br>

# Documentation: <br>
**my_pca(df,conditions,pc_x,pc_y)** <br>
Plot principal component analysis results. <br>
**Parameters:** <br>
df: a transposed dataframe. First column contains the conditions (experimental groups) and the rest are features. <br>
conditions: a string, the column name of the conditions in df.<br>

**cluster_calc(matrix,k0,kn)** <br>
Calculates the optimal clusters based on K-means clustering. The functions scales the data to z-scores before clustering. <br>
Prints out silhouette score for each cluster, and a scatter plot to estimate the goodness of separation. <br>
**Parameters:** <br>
  matrix: a dataframe, containing only numerical values. <br>
  k0,kn: an integer, the start and end of an array of clusters to test. <br>
**Returns:** scores, a dataframe with silhouette scores for each clusters.

**k_clustered_hm(matrix,k)** <br>
Plot a k-means clustered heatmap, based on a desired number of clusters. <br>
Scale the data based on z-scores by rows, before clustering. <br>
**Parameters:** <br>
matrix: a dataframe, containing only numerical values. <br>
k: an integer, number of clusters based on which the data will be clustered. <br>
**Returns:** df_mat_clustered, a dataframe, similar to 'matrix' with an additional column corresponding each feature to its relevant cluster.

**top_bottom(df,names,groups,top,bottom)** <br>
Plot the most changing features in a given data, between conditions. <br>
**Parameters:** <br>
df: a dataframe containing the data. <br>
names: a string, the name of the column containing the names of the features. <br>
groups: a list of 2 groups to compare. <br>
top, bottom: an integer, the number of results from the head and tail of a sorted dataframe in a descending order (head is highest).

**plot_feature(names,groups,targets,df,colors=sns.color_palette("tab10"),ylab = 'Normalized Expression',ci='sd')** <br>
Plot list of features across given conditions or experimental groups. <br>
**Parameters:** <br>
names: list of features. <br>
groups: list of strings corresponds to column names to will be grouped together for calculating mean and sem. <br>
targets: a string, column name in dataframe that contains features names e.g. 'Gene Symbol'.  <br>
df: a dataframe with the data. <br>
colors: set palette as in seaborn's barplot. See <a href="https://seaborn.pydata.org/tutorial/color_palettes.html
"> Choosing Color Palettes </a> <br>
ylab: a string for y axis label. 'Normalized expression' by default.
ci: confience interval, a string or a float. 'sd', Standard Deviation by default. Set the desired error bar as in seaborn's plots (0.68 = SEM). <br> 
See <a href="https://seaborn.pydata.org/generated/seaborn.barplot.html"> Seaborn Barplot </a> 
                
**enrichr_clusters (gene_list,organism,gene_set,background=20000,cutoff=0.05)** <br>
Compare enrichment terms between group of genes i.e. clusters. <br>
This function has based on GSEAPY's enrichr, but is adapted to publishing considerations where a user will want to spot enrichment differences between groups. <br>
Unlike GSEApy's enrichr, this function compare clusters per dataset (gene_set) and thus does not support multiple datasets as an input. What I usually do is use this function inside a for loop as follows: <br>
import gseapy as gp <br>
datasets = pd.Dataframe(gp.get_library_name(organism='Mouse')) # can be 'Human', 'Yeast' etc. <br>
These libraries are highly redundant, and you may prefer to filter this dataframe before looping through 192 databases. You can easily do so based on the names of the datasets and then: <br>
for gene_set in len(range(datasets)): <br>
  enrichr_clusters(gene_list,organism,gene_set = gene_set, background=20000,cutoff=0.05)
**Parameters:** <br>
feature_list: a list of lists, each list contains feature names of a cluster e.g. a list of genes. <br>
organism: ‘Human’, ‘Mouse’, ‘Yeast’, ‘Fly’, ‘Fish’, ‘Worm’. <br>
gene_set: a string, has to be an item from this list: gp.get_library_name(). <br>
background=20000 (default) # e.g. 'hsapiens_gene_ensembl'. A Background to run the analysis against. <br>
cutoff=0.05 (default). significance threshold . <br>
**Returns:** df_list, enrichment results of all clusters analyzed for a given dataset. <br>
For more information please refer to GSEApy's documentation at <a href="https://gseapy.readthedocs.io/en/latest/gseapy_example.html#2.-Enrichr-Example"> GSEApy Docs </a>

# Downloads: <br>
<a href="https://saarezagouri980.github.io/BioPyTools/"> BioPyTools Repository </a>
# Dependency: <br>
- Numpy,Scipy,Pandas, Matplotlib, Seaborn, GSEAPY, sklearn. <br>

The following are required for all functions of this module and can be copied to your code for your own convenience: <br>

import pandas as pd <br>
import matplotlib.pyplot as plt <br>
import numpy as np <br>
import seaborn as sns
