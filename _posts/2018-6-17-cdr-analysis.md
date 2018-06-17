---
layout: post
title: Analysis of a Call Detail record
---
A call detail record (CDR) is a data collected by telephone operator. It contains a sender, a receiver, a timestamp and the duration in case of a call. It is usually aggregated at for privacy matter. The CDR we use for this analysis is a public dataset collected in the region of Milan in Italy. The dataset is available on Kaggle and called [mobile phone activity](https://www.kaggle.com/marcodena/mobile-phone-activity).
The CDR is pretty rich in information. Our analysis is based on the sms traffic. The data we use is then: the emitting cell identifier, the receiving country and counts of sms. The goal of the analysis is to group cells because their originating sms are similarly distributed. To that end we propose to use four different approaches: the modularity maximization, the information theoretic clustering and co-clustering, a bayesian blockmodeling, and finally, a neural network learning from which we extract semantic free representation of the cells.

1. Table of content
{:toc}

# Modularity based graph clustering

# Information theoretic clustering

This section exploits information theory concept to partition the CDRs. The first analysis performs a clustering of the cell identifier while the seconds performs a coclustering, i.e a simultaneous clustering of cell identifier and countries.   

## Information theoretic clustering

While modularity maximization aims at grouping cells being densely connected, information theoretic clustering groups cells having a similar distribution of sms over other cells. Concretely, modularity tracks clicks and information theoretic clustering captures hubs and peripheral cells. The Figure 1 illustrates the difference in the structures tracked by both approaches.

{% include side_by_side_images.html url1="https://rguigoures.github.io/images/modularity_example.png" width1=350 url2="https://rguigoures.github.io/images/itc_example.png" width2=350 description="Fig.1 - Clustering obtained by modularity maximization (left) and information theoretic clustering (right)" %}

Let's define \\(A\\) the adjacency matrix of size n (number of cells) and \\(C\\) the partition of \\(A\\) into \\(k \times k\\) blocks. The matrix \\(C\\) is a compressed version of the matrix \\(A\\). Compression consists in reducing a large matrix to a smaller matrix, with the minimal information loss. To that end, we can maximize the mutual information of the matrix \\(C\\). Let's denote \\(P\\) the joint probability matrix corresponding to the matrix \\(C\\). The mutual information (MI) is defined as follows:

$$
MI(P) = \displaystyle\sum_i^k \displaystyle\sum_j^k P_{ij} * \log \left( \dfrac{P_{ij}}{P_i P_j} \right)
$$

Let's use a simple example to illustrate the behavior of the mutual information:

$$ A = \begin{pmatrix}
0 & 0 & 2 & 1 \\
0 & 0 & 2 & 1 \\
1 & 2 & 0 & 0 \\
1 & 2 & 0 & 0
\end{pmatrix}
$$

Imagine we want to partition rows and columns into 2 clusters, i.e 4 blocks in the matrix. Grouping cells 1 and 2, as well as cells 3 and 4 produces the best partition. Indeed, both cells 1 and 2 are linked to 3 and 4, and in the same proportions. Conversely, grouping 1 and 3, as well as 2 and 4 produces the worst partition.
The best partition is called \\(C_B\\) and the worst partition \\(C_W\\) (with respective joint probability matrices \\(P_B\\) and \\(P_W\\)).

$$
\begin{align}
C_B = \begin{pmatrix}
0 & 6 \\
6 & 0
\end{pmatrix}
& \mbox{ } &
C_W = \begin{pmatrix}
3 & 3 \\
3 & 3
\end{pmatrix}
\\\\
MI(P_B) = \log(2) & \mbox{ } & MI(P_W) = 0
\end{align}
$$

The mutual information of the worst partition is null. In such a partition, occurences are distributed over the blocks and does not reflect the underlying structure of the initial matrix. This is the lowest bound of the mutual information. Conversely, the best partition maximizes the mutual information. The upper bound of the mutual information is equal to \\(H(P_B)\\), where \\(H\\) represents the Shannon entropy. Note that the modularity maximization algorithm would not be able to produce such a partition but put all nodes in a single cluster.

In order to find the best partition, we apply an algorithm similar to k-means:
1. clusters are initialized randomly and balanced,
2. each cell is allocated to the cluster, so that the mutual information is maximized,
3. iterate until clusters do not change anymore.

K-means algorithms aims at minimizing the intra-cluster variance at each iteration. But according to the Huyghens theorem, minimizing the intra-cluster variance is equivalent to maximizing the inter-class variance since the sum of the intra-cluster and the inter-cluster variance is equal to the data variance. The mutual information, in the case of information theoretic clustering can be seen as an inter-cluster variance maximization.

Let's apply the mutual information maximization algorithm to the call detail record. We fix the number of clusters to five. To evaluate the quality of the clustering, we visualise two matrices. First, the joint probability matrix \\(P\\). Secondly, the mutual information matrix \\(M = \{m_{ij} \forall i,j \in 1..k\}\\), where \\(MI(P) = \sum\sum m_{ij} \\).

First, let's plot the two matrices for randomly initialized clusters.

{% include image.html url="https://rguigoures.github.io/images/density_mi_random.png" width=500 description="Fig.2 - Random partition of the CDR. Joint probability matrix (left) and mutual information matrix (right)" %}

We can see on Figure 2 that the density is similarly distributed in the cell of the matrix. The mutual information also indicates that the density in the cells is close to the expected value in case of random clustering. In other words, \\(P_{ij} \simeq P_i P_j\\). Then, we run the algorithm and plot the same matrices for the obtained partitions.

{% include image.html url="https://rguigoures.github.io/images/density_mi_cluster.png" width=500 description="Fig.3 - Partition of the CDR obtained by maximization of the mutual information. Joint probability matrix (left) and mutual information matrix (right)" %}

After running the algorithm, we can observe the underlying structure of the data emerging. The joint probability shows cells with high density. But this observation does not mean that the partition is meaningful. Indeed if the clusters are unbalanced, bigger clusters are likely to have high density between themselves. In the mutual information matrix, red cells represent excess of cooccurrences. Conversely, blue cells respresent lacks of cooccurences. The clusters can then be interpreted as follow: cells in cluster 4 are grouped together because they excessively interract with themselves and less than expected with clusters 0 and 1. Note that the connections between clusters 0 (or 1) and 4 have quite high density but less than expected.  

## Information theoretic coclustering

# Bayesian blockmodeling

# Latent representation of the cells
