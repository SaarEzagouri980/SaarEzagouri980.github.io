# Biotools files:
<a href="https://github.com/SaarEzagouri980/SaarEzagouri980.github.io/edit/main/Biotools/"> Git Repository </a>
This repository contains the python files with the functions and an example dataset.
**Using examples datasets:**
The famous 'Iris' data can be used to test the functions: 'my_pca', 'k_clustered_hm', 'cluster_calc', and 'top_bottom'
And can be easily accessed as follows: 
os.chdir(r'Path') # Set current directory
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
df_mat = df.loc[:,df.columns != 'target'] # remove the features column to get a df with only numeric values when needed.
'Deseq_all' dataset can be download from the repository and use to test the functions 'plot_feature' and 'enrichr_clusters'. 
**To test 'enrichr_clusters' this list should be used:**
df = pd.read_excel(r'') # load  'Deseq_all' dataset
gene_names = df[['Gene Symbol']]



