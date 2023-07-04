# PU-MIL-AD

`PU-MIL-AD` is a GitHub repository containing the **PUMA** [1] algorithm. It refers to the paper titled *Learning from Positive and Unlabeled Multi-Instance Bags in Anomaly Detection*.

Check out the pdf here: *MISSING LINK -- Will be available as soon as the paper is online*.

## Abstract

In the multi-instance learning (MIL) setting instances are grouped together into bags. Labels are provided only for the bags and not on the level of individual instances. A positive bag label means that at least one instance inside the bag is positive, while a negative bag label restricts all the instances in the bag to be negative. MIL data naturally arises in many contexts, such as anomaly detection, where labels are rare and costly, and one often ends up annotating the label for sets of instances. Moreover, in many real-world anomaly detection problems, only positive labels are collected because they usually represent critical events. Such a setting, where only positive labels are provided along with unlabeled data, is called Positive and Unlabeled (PU) learning. Despite being useful for several use cases, there is no work dedicated to learning from positive and unlabeled data in a multi-instance setting for anomaly detection. Therefore, we propose the first method that learns from PU bags in anomaly detection. Our method uses an autoencoder as an underlying anomaly detector. We alter the autoencoder’s objective function and propose a new loss that allows it to learn from positive and unlabeled bags of instances. We theoretically analyze this method. Experimentally, we evaluate our method on 30 datasets and show that it performs better than multiple baselines adapted to work in our setting.

## Contents and usage

The repository contains:
- PUMA.py, a function that allows to use PUMA's algorithm;
- Notebook.ipynb, a notebook showing how to use PUMA on an artificial 2D dataset;
- create_ds.py, a function that generates the artificial 2D dataset for the Notebook;
- build_bags.py, the algorithm that we used to create bags for benchmark datasets, as explained in the paper.

To use PUMA, import the github repository or simply download the files. You can find the benchmark datasets at these links: [[DAMI](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)] and [[ADBench](https://github.com/Minqi824/ADBench/tree/main/datasets/Classical)].


## EXample-wise ConfidEncE of anomaly Detectors (ExCeeD)

Given a dataset with attributes **X** in bag shape (e.g., numpy array with 3 dimensions) and an array with the bag labels (1 for anomalous, 0 for unlabeled), PUMA works as follows. First, you need to specify the network structure as well as the key hyperparameters (# reliable negatives, learning rate, batch_size, epochs, ...). Second, using the fit function you can train PUMA. Finally, the decision function returns the anomaly probabilities for both bags and instances.
Please, check out the Notebook for the details.

## Dependencies

The `gammaGMM` function requires the following python packages to be used:
- [Python 3.9](http://www.python.org)
- [Numpy 1.21.0](http://www.numpy.org)
- [Pandas 1.4.1](https://pandas.pydata.org/)
- [PyOD 1.1.0](https://pyod.readthedocs.io/en/latest/install.html)


## Contact

Contact the author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be).


## References

[1] Perini, L., Vercruyssen, V., Davis, J.: *Learning from Positive and Unlabeled Multi-Instance Bags in Anomaly Detection.* In: the 29TH ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2023.
