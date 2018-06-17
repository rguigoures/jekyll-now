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

While modularity maximization aims at grouping cells being densely connected, information theoretic clustering groups cells having a similar distribution of sms over other cells. Concretely, modularity tracks clicks and information theoretic clustering captures hubs and peripheral cells. The following image illustrates the difference in the structures tracked by both approaches.

{% include image.html url="https://rguigoures.github.io/images/modularity_example.png" width=100 description="Clustering obtained by modularity maximization" %} {% include image.html url="https://rguigoures.github.io/images/itc_example.png" width=100 description="Clustering obtained by information theoretic clustering" %}


## Information theoretic coclustering

# Bayesian blockmodeling

# Latent representation of the cells
