# Generating synthetic datasets of a particular size

1. The baselines (GCond, GDEM, and GCSR) take number of nodes as a parameter while generating synthetic dataset.
2. The baselines generate fully connected graphs. Therefore, if the number of nodes is $n$, then the number of edges is $n^2$. Further, the edges have a `float` weight attached to them.
3. The baselines produce dense `float` features regardless of the original space of the features.
4. We assume that `int` representations take half as much space as `float` representations. Throughout our work, we arbitrarily assume the size of one `float` to be `4` and the size of one `int` to be `2`.

Therefore, the size of synthetic dataset with $n$ nodes is $4n^2+4nd$, where $d$ is the number of features. The first term $4n^2$ comes from the fully-connected `float` weighted adjacency matrix, and the second term $4nd$ corresponds to dense `float` features produced by the baseline method.

So, we can compute the number of nodes that should be given to the baseline as input to generate a synthetic dataset of size $t$ by solving $4n^2+4nd-t=0$ for $n$. This gives $n=(\sqrt{d^2+t}-d)/2$.

# Training with baselines

Scripts for training the baseline methods are provided here. They can all be run with `train_all.sh`. Note that since the baselines generate edge weighted graphs, we use the edge weight variants of GAT and GIN to train on the synthetic dataset for these methods. The `train_all.sh` and the script expects the dataset to be in a particular folder format. Examples of some of these are provided as `saved_GCond`, etc.
Note that all the synthetic datasets have not been provided to avoid bloating the repository.
