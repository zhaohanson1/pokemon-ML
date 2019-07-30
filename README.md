# Machine Learning with Pokemon

https://www.kaggle.com/abcsds/pokemon

Applied unsupervised and supervised classification algorithms to a dataset containing the statistics of the first 721 Pokemon, along with the most current Mega-Evolutions.

The data worked with is in pokemon_revised.csv, due to the original csv having flaws. (The mega evolved version of Pokemon would have the same stats and typing as the original.)

## Overview

### Pre-processing
Some categories were one hot encoded, redundant identification was removed, and some information that was missing or incorrect was manually fixed by referencing [Bulbapedia](https://bulbapedia.bulbagarden.net) (In a real world application of ML data, this is not always an option.)

### Unsupervised 
* K-means Clustering: A heuristic algorithm that alternates between finding the mediod of a group and group points to the closest mediod. 
* Spectral Clustering: A technique that is derived from a graph problem of cutting the graph into two similar groups without cutting too much edges weight. The solution involves eigenvectors of the similarity matrix.

### Supervised
* Linear SVM: Maximize the linear margin that separates two (approximate) groups.
* RBF Kernel SVM: Maximize the margin created by a Gaussian function.
* Logistic Regression: Fit the data to a logistic function
* Decision Trees: "Multi-level" regression
* Random Forests: Ensemble version of decision trees