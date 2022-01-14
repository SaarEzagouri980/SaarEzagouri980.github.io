# BioPyTools Files
<a href="https://github.com/SaarEzagouri980/SaarEzagouri980.github.io/tree/main/BioPyTools"> Git Repository </a> : This repository contains the python files with the functions and an example dataset.

**Using example datasets:** <br>
The famous 'Iris' data can be used to test the functions: 'my_pca', 'k_clustered_hm', 'cluster_calc', and 'top_bottom' that can be easily accessed as follows: <br>
os.chdir(r'Path') # Set current directory <br>
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"# load dataset into Pandas DataFrame <br>
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target']) <br>
df_mat = df.loc[:,df.columns != 'target'] # remove the features column to get a df with only numeric values when needed. <br>
'Deseq_all' dataset can be download from the repository and use to test the functions 'plot_feature' and 'enrichr_clusters'. <br>
**To test 'enrichr_clusters' the following lines should be used:** <br>
df = pd.read_excel(r'') # load  'Deseq_all' dataset <br>
gene_names = list(df['Gene Symbol'])
