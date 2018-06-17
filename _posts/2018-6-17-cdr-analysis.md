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

# Bayesian blockmodeling

# Latent representation of the cells
