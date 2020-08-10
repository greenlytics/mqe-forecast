# SiteForecast

## Tuning GBDT models
Quick start receipt for training GBDT models:
##### 1) Set number of trees to something as high as possible (e.g. 3000)
##### 2) Run a grid search or random search
##### 3) Finally set number of trees even higher and tune learning rate

The maximum depth of the tree controls the degree of feature interaction that you can model. Usually it is fair to assume that the degree if interactions is fairly low. As a rule of thumb the depth of the tree should be around 4-6 [1]

Shrinkage (or learning rate) is a parameter that exponentially reduced the weight that a tree will have in the final prediction as more and more trees are added. As a general rule of thumb, a model with higher shrinkage (or low learning rate) and more trees will perform better than a model with low shrinkage and few trees. The learning rate should typically be less than 0.1.

Stochastic gradient boosting is doing the same thing as random forest is doing. Either sampling data points or sampling feature set before creating a split point. This typically leads to improved accuracy.
## Model pipeline


## References

### Papers
[1] [Friedman, J. H. "Greedy Function Approximation: A Gradient Boosting Machine"](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
<br>The original paper on gradient boosting.

[2] [Friedman, J. H. "Stochastic Gradient Boosting"](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf)
<br>

[3] [Ke G. et. al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)

[4] [Chen T, Guestrin, C. ,"XGBoost: A Scalable Tree Boosting System"](https://arxiv.org/pdf/1603.02754.pdf)

### Books
[Hastie, T. Tibshirani R. Friedman J. "The Elements of Statistical Learning"](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
<br> Great book for understanding of gradient boosting.

### Videos
[Peter Prettenhofer - Gradient Boosted Regression Trees in scikit-learn](https://www.youtube.com/watch?v=IXZKgIsZRm0)

### Blog posts
[CatBoost vs. Light GBM vs. XGBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)
<br> Comparison of the tree main GBDT implementations.
