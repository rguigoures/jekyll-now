---
layout: post
title: Analysis of a Call Detail Record (part 1) - from Information Theory to Bayesian Modeling
---
A call detail record (CDR) is a data collected by telephone operator. It contains a sender, a receiver, a timestamp and the duration in case of a call. It is usually aggregated at for privacy matter. The CDR we use for this analysis is a public dataset collected in the region of Milan in Italy. The dataset is available on Kaggle and called [mobile phone activity](https://www.kaggle.com/marcodena/mobile-phone-activity).
The CDR is pretty rich in information. The analysis in this post is based on the sms traffic. The data we use is then: the emitting antenna identifier, the receiving country and call counts. The goal of the analysis is to group antennas because their originating calls are similarly distributed over countries and - simultaneously - group countries because the received calls are distributed over the same antennas. This is called co-clustering. To do so, will first use a method based on information theory and define a set of measures to understand and visualize the results. Then, we will link the information theory to bayesian modeling, showing the benefits and difficulties using such an approach.

In a different post, we propose to train an autoencoder to obtain a latent representation of the antennas, using the same data set.

1. Table of content
{:toc} 

# Data representation

# Information theoretic coclustering

Let's define \\(A\\) the adjacency matrix of size \\(n \times m\\) (number of antennas \\(\times\\) number of countries) and \\(C\\) the partition of \\(A\\) into \\(k \times l\\) blocks (number of clusters of antennas \\(\times\\) number of clusters of countries). The matrix \\(C\\) is a compressed version of the matrix \\(A\\). Compression consists in reducing a large matrix to a smaller matrix, with the minimal information loss. To measure the information loss, we can use the so-called Kullback-Leibler divergence. This concept originates in information theory and measures how many bits we lose to encode a signal A from a signal B. In the present context, we can use it to compare two distributions. Let's introduce \\(P_A\\), the joint probability matrix representing the adjacency matrix \\(A\\), i.e the matrix \\(A\\) that has been normalized by the total number of calls. Similarly, \\(P_C\\) is the joint probability matrix of the cluster adjacency matrix \\(C\\). Finally, \\(\hat{P}_A\\) is a joint probability matrix of size \\(n \times m\\) where cell values are the values of the joint probability between coclusters, normalized by the number of cells in the coclusters. Let's illustrate it:

$$
\begin{align}
A = \begin{pmatrix}
0 & 0 & 2 \\
0 & 0 & 2 \\
1 & 2 & 0 \\
1 & 2 & 0
\end{pmatrix}
& \Rightarrow &
P_A = \begin{pmatrix}
0 & 0 & 0.2 \\
0 & 0 & 0.2 \\
0.1 & 0.2 & 0 \\
0.1 & 0.2 & 0
\end{pmatrix}
\end{align}
$$
<br>
$$
\begin{align}
C = \begin{pmatrix}
0 & 4 \\
6 & 0
\end{pmatrix}
& \Rightarrow &
P_C = \begin{pmatrix}
0 & 0.4 \\
0.6 & 0
\end{pmatrix}
& \Rightarrow &
\hat{P}_A = \begin{pmatrix}
0 & 0 & 0.2 \\
0 & 0 & 0.2 \\
0.15 & 0.15 & 0 \\
0.15 & 0.15 & 0
\end{pmatrix}
\end{align}
$$

The Kullback-Leibler is a non symmetric measure and should be read as follow: \\(KL(P_A \mid \hat{P}_A)\\) denotes the Kullback-Leibler divergence from distribution \\(\hat{P}_A\\) to \\(P_A\\). This is defined as follows.

$$
KL(P_A \mid \hat{P}_A) = P_A \log \left( \dfrac{P_A}{\hat{P}_A} \right)
$$

Let's illustrate the Kullback-Leibler divergence using a simple example. I need 3 apples, 2 oranges and 1 pear to bake a cake. Unfortunately, I can get from the supermarket only bags containing 1 apple, 1 apple and 1 pear. At the end, to make the cake, I need to buy 3 bags and there is 2 pears and 1 orange left. Lets turn is as distributions, the cakes contains 50% apples, 33% oranges and 17% pear. The bags from the supermarket contains 33% of each fruit. The Kullback-Leibler divergence from the bag distribution to the cake distribution is equal to 0.095. Let's now analysis the edge cases: if both the bags and the cake have the same distribution, the Kullback Leibler divergence is null because the there is no fruit left after baking the cake. Conversly, if the bag does not contain oranges, the Kullback Leibler is infinite because, even with an infinite amount of bag, you are not going to bake the cake.

In information theoretic clustering, we try to find the optimal compressed matrix \\(C\\) which minimizes the Kullback-Leibler divergence to the joint probability distribution of the original adjacency matrix \\(A\\), i.e \\(KL(P_A \mid \hat{P}_A)\\). The Kullback Leibler divergence ranges in theory from \\(0\\) to \\(+\infty\\), but in the context of coclustering, the latter case does not happen because there must be interractions between a cluster of antennas and a cluster of countries if the antennas (resp. the countries) it contains have interractions. 

It has been proved\[[^fn1]\] that minimizing the Kullback-Leibler divergence is equivalent the minizing the loss in mutual information between the original data and the compressed data. Mutual information measures how much the partition of countries give information about the partition of antenna, and vice versa. In other words, it measure how confident we are guessing the originating antenna knowing the destination country of the call. The mutual information matrix is defined as follows:

$$
MI_{ij}(P_C) = P_{C,ij} \log \left( \dfrac{P_{C,ij}}{P_{C,i} P_{C,j}} \right)
$$

Let's use a simple example to illustrate the behavior of the mutual information:

$$ A = \begin{pmatrix}
0 & 0 & 2 & 1 \\
0 & 0 & 2 & 1 \\
1 & 2 & 0 & 0 \\
1 & 2 & 0 & 0
\end{pmatrix}
$$

Imagine we want to partition rows and columns into 2 clusters, i.e 4 blocks in the matrix. Grouping antennas 1 and 2, as well as antennas 3 and 4 produces the best partition. Indeed, both antennas 1 and 2 are linked to 3 and 4, and in the same proportions. Conversely, grouping 1 and 3, as well as 2 and 4 produces the worst partition.
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
P_B = \begin{pmatrix}
0 & \frac{1}{2} \\
\frac{1}{2} & 0
\end{pmatrix}
& \mbox{ } &
P_W = \begin{pmatrix}
\frac{1}{4} & \frac{1}{4} \\
\frac{1}{4} & \frac{1}{4}
\end{pmatrix}
\\\\
MI(P_B) = \log(2) & \mbox{ } & MI(P_W) = 0
\end{align}
$$

The mutual information of the worst partition is null. In such a partition, occurences are distributed over the blocks and does not reflect the underlying structure of the initial matrix. This is the lowest bound of the mutual information. Conversely, the best partition maximizes the mutual information. The upper bound of the mutual information is equal to \\(H(P_B)\\), where \\(H\\) represents the Shannon entropy. Note that a clustering algorithm tracking cliques would not be able to produce such a partition but would put all nodes in a single cluster instead.

In order to find the best partition, we apply an agglomerative hierarchical clustering, i.e we start allocating a single antenna to each cluster and we merge them successively so that the mutual information is maximized. Let's apply the algorithm to the call detail record. We fix the number of clusters to five for illustration purpose. To evaluate the quality of the clustering, we visualise two matrices. First, the joint probability matrix \\(P\\). Second, the mutual information matrix \\(M = \{m_{ij} \forall i,j \in 1..k\}\\), where \\(MI(P) = \sum\sum m_{ij} \\).

First, let's plot the two matrices for randomly initialized clusters.

{% include image.html url="https://rguigoures.github.io/images/density_mi_random.png" width=500 description="Fig.2 - Random partition of the CDR. Joint probability matrix (left) and mutual information matrix (right)" %}

We can see on Figure 2 that the density is similarly distributed in the cell of the matrix. The mutual information also indicates that the density in the cells is close to the expected value in case of random clustering. In other words, \\(P_{ij} \simeq P_i P_j\\). Then, we run the algorithm and plot the same matrices for the obtained partitions.

{% include image.html url="https://rguigoures.github.io/images/density_mi_cluster.png" width=500 description="Fig.3 - Partition of the CDR obtained by maximization of the mutual information. Joint probability matrix (left) and mutual information matrix (right)" %}

After running the algorithm, we can observe the underlying structure of the data emerging. The joint probability shows cells with high density. But this observation does not mean that the partition is meaningful. Indeed if the clusters are unbalanced, bigger clusters are likely to have high density between themselves. In the mutual information matrix, red cells represent excess of cooccurrences. Conversely, blue cells respresent lacks of cooccurences. The clusters can then be interpreted as follow: antennas in cluster 4 are grouped together because they excessively interract with themselves and less than expected with clusters 0 and 1. Note that the connections between clusters 0 (or 1) and 4 have quite high density but less than expected.  

# Information theoretic coclustering

One great advantage of information theory based clustering approaches lies in being able to tackle bipartite graphs: it is possible to simultaneously cluster antennas and countries. This is called coclustering.

# Bayesian blockmodeling

Information theoretic clustering directly optimizes the Kullback-Leibler divergence from the partition to the actual data. This approach is valid when the amount of data is large enough to properly estimate the joint probability matrix between antennas and countries. But if it's not the case, we can easily get spurious patterns. One solution to avoid this problem consists in adding a regulariuation term to the optimized criterion. Another solution would be to build a Bayesian model. Actually, the average negative logarithm of the multinomial probability mass function over the cells of the adjacency matrix converges to the Kullback-Leibler divergence from the partition to the actual data.

$$
KL(P_A | \hat{P}_A) \rightarrow -\dfrac{1}{n} \log(f_\mathcal{M}(n, A, \hat{P}_A)) \mbox{ when } n \rightarrow +\infty
$$

where n is the number of observations (sms in th example) and \\(f_\mathcal{M}\\) the probability mass function of the multinomial distribution. This can be easily proved using the Stirling approximation, i.e \\(\log(n!) \rightarrow n\log(n) - n\\) ; when \\(n \rightarrow +\infty\\). 

# Visualisations



# References

[^fn1]: Inderjit S. Dhillon et al., [_Information-theoretic co-clustering_](http://www.cs.utexas.edu/users/inderjit/public_papers/kdd_cocluster.pdf), KDD 2003
