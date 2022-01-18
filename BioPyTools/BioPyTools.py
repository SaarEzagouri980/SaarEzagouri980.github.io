import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def my_pca (df,conditions,pc_x,pc_y):
    """""
    my_pca, CALCULATES THE PRINICIPAL COMPONENETS AND THE EXPLAINED VARIANCE RATIO OF THE DATASET AND PLOT THEM BY CONDITION.
    df : a transposed dataset. first column contains the conditions.
    conditions : string, column name of conditions column. the rest of the columns in df are the features.
    """""
    from sklearn.preprocessing import StandardScaler
    x = df.select_dtypes(include=np.number)
    x = StandardScaler().fit_transform(x)
    from sklearn.decomposition import PCA
    pca = PCA()
    principalComponents = pca.fit_transform(x)
    PC_number = (np.arange(1,len(principalComponents[0])+1)).tolist()
    PC_number = [str(x) for x in PC_number]
    PC = 'PC'
    principalDf_columns = [PC + n for n in PC_number]
    explained_var_ratio = pd.DataFrame(zip(principalDf_columns,pca.explained_variance_ratio_),columns=['PC','Explained_Var'])
    principalDf = pd.DataFrame(data = principalComponents, columns = principalDf_columns)
    cond = df[conditions]
    finalDf = pd.concat([principalDf, cond], axis = 1)
    print(finalDf)
    fig = plt.figure(figsize = (8,8))
    targets = list(set(df[conditions]))
    print(targets)
    explained_var_ratio_PCx = explained_var_ratio.loc[explained_var_ratio['PC']== pc_x,'Explained_Var'].values*100
    explained_var_ratio_PCy = explained_var_ratio.loc[explained_var_ratio['PC']== pc_y,'Explained_Var'].values*100
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(pc_x+':'+str("%.2f" % explained_var_ratio_PCx)+'%', fontsize = 15)
    ax.set_ylabel(pc_y+':'+str("%.2f" % explained_var_ratio_PCy)+'%', fontsize = 15)
    ax.set_title('PCA', fontsize = 20)
    for t in targets:
        indicesToKeep = finalDf['target'] == t
        ax.scatter(finalDf.loc[indicesToKeep,'PC1']
                   , finalDf.loc[indicesToKeep,'PC2']
                   , s = 50)
    ax.legend(targets)
    return finalDf


def cluster_calc(matrix,k0,kn):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    from scipy.stats import zscore

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    range_n_clusters = np.arange(k0,kn,step=1)
    X = zscore(matrix,axis=1)
    scores = pd.DataFrame()
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        scores.loc[n_clusters,'score'] = silhouette_avg
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X.iloc[:, 0], X.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
    plt.show()
    plt.plot(scores.index,scores.score)
    plt.xticks(range_n_clusters,labels=range_n_clusters)
    plt.xlabel('k')
    plt.ylabel('Sillouette Score')
    plt.title('Sillouette Score')
    return scores


def k_clustered_hm (matrix,k):
    from sklearn.cluster import KMeans
    from matplotlib.patches import Patch
    from scipy.stats import zscore
    matrix = zscore(matrix,axis=1)
    clusterer = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterer.fit_predict(matrix)
    matrix['Cluster'] = cluster_labels
    lut = dict(zip(set(cluster_labels), sns.color_palette()))
    row_colors = matrix['Cluster'].map(lut)
    df_mat_clustered = matrix.sort_values(by=['Cluster'])
    matrix_mat_clustered = df_mat_clustered.loc[:,df_mat_clustered.columns != 'Cluster']
    sns.clustermap(matrix_mat_clustered,row_cluster=False,col_cluster=False, row_colors=row_colors,cmap='vlag',cbar_kws={'label': 'Z-Score'})
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut, title='Clusters',
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.tight_layout()
    return df_mat_clustered


def top_bottom(df,names,groups,top,bottom):

    df2_norm_clean = df.select_dtypes(include=np.number)
    mean_group1 = df2_norm_clean.loc[:,df2_norm_clean.columns.str.contains(groups[0])].mean(axis=1)
    mean_group2 = df2_norm_clean.loc[:,df2_norm_clean.columns.str.contains(groups[1])].mean(axis=1)
    means = pd.DataFrame()
    means[names] = df[names]
    means[groups[0]] = mean_group1
    means[groups[1]] = mean_group2
    means = means[(means != 0).all(1)]
    means['fc'] = means [groups[0]] / means [groups[1]]
    means = means.sort_values(by=['fc'],ascending=False)
    means['fc'] = np.log2(means['fc'])
    top = means.head(top)
    bottom = means.tail(bottom)
    top_n_bottom = pd.melt(pd.concat([top,bottom],axis=0),id_vars=[names,groups[0],groups[1]]).sort_values(['value'],ascending=True)
    col = np.where(top_n_bottom.value<0,'blue','red')
    plt.barh(y=top_n_bottom[names],width=top_n_bottom['value'],color=col)
    plt.xlabel('log2 (Fold Change)')
    plt.title('Maximal Changes')
    plt.tight_layout()


def plot_feature(names,groups,targets,df,colors=sns.color_palette("tab10"),ylab = 'Normalized Expression',ci='sd'):

    for i in range(len(names)):
        name = names[i]
        dff = df.loc[df[targets] == name, :]
        dff_a = pd.melt(dff, id_vars=[targets])
        dff_a = dff_a.loc[:, dff_a.columns != targets]
        for ii in range(len(groups)):
            dff_a.loc[dff_a.variable.str.contains(groups[ii]), 'variable'] = groups[ii]
        val = dff_a['value']
        plt.figure()
        sns.set_theme(style='white')
        sns.barplot(data=dff_a, x='variable', y='value', capsize=0.05, palette=colors,ci=ci)
        sns.swarmplot(data=dff_a, x='variable', y='value', color='black')
        plt.ylim(min(val) - (max(val) * 0.2), max(val) + (max(val) * 0.2))
        plt.ylabel(ylab)
        plt.xlabel('')
        plt.title(name)
        plt.tight_layout()


def enrichr_clusters (feature_list,organism,gene_set,background=20000,cutoff=0.05):

    import gseapy as gp
    cluster_list = list()
    for i in range(len(feature_list)):
        enr = gp.enrichr(gene_list=feature_list[i],
                             organism= organism,
                             gene_sets=gene_set,
                             background= background,
                             outdir='test/enrichr',
                             no_plot= False,
                             cutoff=cutoff,
                             verbose=True)
        cluster = enr.res2d
        cluster['Cluster'] = 'Cluster'+str(i+1)
        cluster_list.append(cluster)
    df_list = pd.concat(cluster_list)
    df_list['Hits'] = df_list["Overlap"].str.split("/", n=1, expand=True)[0]
    sns.set_theme(style='whitegrid', palette=None)
    plt.figure(figsize=(15, 15))
    ax = sns.scatterplot(
        data=df_list, x="Cluster", y="Term", hue="Adjusted P-value", size="Hits", palette='magma_r',
        sizes=(20, 200), legend="full")
    L = len(feature_list)
    plt.xticks([0,L-1], [f'Cluster {i}' for i in range(len(feature_list))])
    plt.xlim(-1, L)
    norm = plt.Normalize(df_list['Adjusted P-value'].min(), df_list['Adjusted P-value'].max())
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
    sm.set_array([])
    h, l = ax.get_legend_handles_labels()
    l_ind = l.index('Hits')
    ax.get_legend().remove()
    plt.legend(h[l_ind:],l[l_ind:],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    cbar = ax.figure.colorbar(sm,shrink=0.4)
    cbar.set_label('Adjusted P-value', rotation=270, labelpad=15)
    plt.title(gene_set)
    plt.tight_layout()
    return df_list

