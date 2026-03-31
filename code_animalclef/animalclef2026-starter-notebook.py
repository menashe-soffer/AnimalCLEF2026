#!/usr/bin/env python
# coding: utf-8
import pickle
# # 🐾 AnimalCLEF2026 Competition: Official Starter notebook
# 
# The **Goal of the** [AnimalCLEF2026](https://www.kaggle.com/competitions/animal-clef-2026/) competition is to cluster individual animal (loggerhead sea turtles and Texas horned lizards) in photos. This notebook visualize the provided dataset and propose a baseline solution, based on the state-of-the-art re-identification model [MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384). It also suggests some possible improvements for the participants to follow.
# 
# The dataset is split into the training and test sets. For each image from the test set, the goal is to assing in a cluster. Each cluster should correspond to one individual animal. The cluster name must start with `cluster_SeaTurtleID2022_` or `cluster_TexasHornedLizards_`.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12294787%2Ff015de35bcf462f36f74faabb20dd7e6%2FAnimalCLEF2026.png?generation=1767855188505985&alt=media)

# ## Dependencies instalation
# For the competition we provide two Python packages for loading and preprocessing of available datasets ([wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets)) and tools / method for animal re-identification ([wildlife-tools](https://github.com/WildlifeDatasets/wildlife-tools)).

# In[1]:


#!pip install humanize nibabel SimpleITK "numpy<2" "torch<2.3"
#!pip install wildlife-datasets git+https://github.com/WildlifeDatasets/wildlife-tools --quiet --upgrade-strategy only-if-needed
#!pip install "transformers<4.44.0"


# ## Dependencies import
# 
# We load all the required packages.

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
import timm
from transformers import AutoModel
from sklearn.cluster import DBSCAN
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity

import torch
import gc
from monitoring import print_vram_stats


# ## 📊 Visualizing Data
# 
# Since `AnimalCLEF2015` is the child class of `datasets.WildlifeDataset` from [wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/datasets.py), it inherits all its methods and attributes. Its integral part is the `transform` attribute, which performs transforms on the input images (such as converting them to torch tensors). We start with no transform.

# In[3]:


#root = '/kaggle/input/animal-clef-2026'
root = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026/data'
from wildlife_datasets import datasets
#datasets.AnimalCLEF2026.get_data(root)
dataset = datasets.AnimalCLEF2026(root)
dataset_full = AnimalCLEF2026(
    root,
    transform=None,
    load_label=True,
    factorize_label=True,
    check_files=False
)


# The important part of the `dataset` class is the DataFrame `metadata`, which contains information about individual images.

# In[4]:


dataset_full.metadata.head()


# The column dataset shows into which of the two datasets the image belongs (see also the column species) and the split is either the train or the test set. We see that TexasHornedLizards does not have any train set, which corresponds to the real-world situation, where the database of images have been just created and needs to be divided into individuals.

# In[5]:


dataset_full.metadata[['dataset', 'split']].value_counts(sort=False)


# For this purpose of this notebook, we will focus only on the test set and ignore images from the training set. We also create `datasets` corresponding to each species.

# In[6]:


dataset_full = dataset_full.get_subset(dataset_full.df['split'] == 'test')

datasets = {}
for name in dataset_full.metadata['dataset'].unique():
    datasets[name] = dataset_full.get_subset(dataset_full.df['dataset'] == name)
datasets


# 📌 **Plotting a sample grid** of the data. We see that there is a large variability in sea turtles, which were taken underwater in their natural habitat. On the other hands, the images of lizards were taken in captivity.

# In[7]:

debug_dict = dict()

for dataset in datasets.values():
    dataset.plot_grid(n_rows=3, n_cols=4, rotate=False);


# ## Inference with MegaDescriptor and MiewID
# 
# Instead of training a classifier, we can just use out of the shelf pretrained models - [MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384) and [MiewID](https://huggingface.co/conservationxlabs/miewid-msv3). We use MegaDescriptor to extract features from sea turtles and MiewID for lizards and then cosine similarity to compute the similarity for each pair of images. Since the cosine similarity reflects the angle between the feature vectors, high similarity means that the feature vectors are close to each other and should depict the same individual.
# 
# **Note:** It is highly recommended to use the GPU acceleration.

# In[9]:


device = 'cuda'
batch_size = 8#32
print(name)

similarities = {}
for name, dataset in datasets.items():
    # Select the model for feature extraction
    print(name)
    if name in ['SalamanderID2025', 'SeaTurtleID2022']:
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
        size = 384
    elif name in ['LynxID2025', 'TexasHornedLizards']:
        model = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
        size = 512
    else:
        raise ValueError('Name does not exist')
    print(name)
    print_vram_stats()

    # Set the extractor and transform for the images
    matcher = CosineSimilarity()
    extractor = DeepFeatures(model=model, device=device, batch_size=batch_size)
    transform = T.Compose([
        T.Resize(size=(size, size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Set the transform for images
    dataset.set_transform(transform)
    # Extract features
    features = extractor(dataset)
    #
    debug_dict[name] = {'features': features.features}
    #
    print_vram_stats()
    # Compute the similarity matrix
    similarity = matcher(features, features)
    similarities[name] = similarity
    # print(features[0][0])
    # print(similarity[0])
    print_vram_stats()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print_vram_stats()



# The next cell shows that TexasHornedLizards has 274 images. Each of the image has a feature vector of size 2152. The similarity matrix computes the similarity between each pair of images and therefore, it is symmetric and of size (274, 274).

# In[11]:


print(f'Dataset {name} with {len(dataset)} images.')
print(f'Features have size {features.features.shape}.')
print(f'Similarity matrix has shape {similarity.shape}.')


# We intend to use the clustering algorithm DBSCAN on the similarity matrix. Since this algoritm return -1 for unclustered entries, we write a function `relabel_negatives`, which creates new clusters for these clusters (each new cluster with one element). For DBSCAN, we specify `min_samples=2` to allow for small clusters and select the epsilon parameters (determining the radius of the clusters) rather arbitrarily. Also due to DBSCAN requirements, we need to convert the similarity matrix into the distance matrix.

# In[12]:


def relabel_negatives(labels):
    labels = np.array(labels)
    neg_indices = np.where(labels == -1)[0]
    new_labels = np.arange(labels.max()+1, labels.max()+1+len(neg_indices))
    labels[neg_indices] = new_labels
    return labels

def run_DBSCAN(similarity, eps):
    # Convert similarity (high is good) to distance (small is good)
    distance = (np.max(similarity) - np.maximum(similarity, 0)) / np.max(similarity)
    print(similarity.shape)
    # for i in range(274):
    #     print(i, distance[i][:5])
    # Obtain predictions
    clustering = DBSCAN(eps=eps, metric='precomputed', min_samples=2)
    clusters = clustering.fit(distance)
    # Relabel -1 clusters into separate clusters
    return relabel_negatives(clusters.labels_), [distance, clusters.labels_]


# Now we run DBSCAN for all extracted similarity matrices and save the results. We select the parameter eps rather arbitrarily.

# In[16]:


results = None
eps_opt = {
    'LynxID2025': 0.3,
    'SalamanderID2025': 0.2,
    'SeaTurtleID2022': 0.4,
    'TexasHornedLizards': 0.24,
}
for name, similarity in similarities.items():
    # Save the clusters for one dataset
    clusters, dbg = run_DBSCAN(similarity, eps_opt[name])
    #
    debug_dict[name]['distances'] = dbg[0]
    debug_dict[name]['dbscan_result'] = dbg[1]
    debug_dict[name]['labels'] = clusters
    #
    result = pd.DataFrame({
        'image_id': datasets[name].metadata['image_id'],
        'cluster': [f'cluster_{name}_{c}' for c in clusters]
    })
    # Merge the clusters to other datasets
    results = pd.concat((results, result))
results.to_csv('/media/soffer/TOSHIBA EXT/AnimalCLEF2026/debug/submission_.csv', index=False)
#
with open('/media/soffer/TOSHIBA EXT/AnimalCLEF2026/debug/debug_startup_nb', 'wb') as fd:
    pickle.dump(debug_dict, fd)
#


# ## Visualizing parameter eps
# 
# We can run DBSCAN for multiple values or eps and visualize the number of clusters. Small eps correspond to 274 clusters (each cluster is one images) and large eps to 1 cluster (all images are in one cluster)

# In[17]:


eps_all = np.linspace(0.00001, 1, 500)
n_clusters = []
for eps in eps_all:
    clusters, _ = run_DBSCAN(similarity, eps)
    n_clusters.append(len(np.unique(clusters)))

plt.plot(eps_all, n_clusters)
plt.vlines(eps_opt[name], min(n_clusters), max(n_clusters), linestyles='dotted')
plt.title('number of clusters')
plt.xlabel('eps (minimum cluster distance for DBSCAN)');


# We can also inspect the images manually. Plotting the clusters shows that most of the clusters are correct (but ofr example  the last plotted cluster is wrong).

# In[18]:


clusters = run_DBSCAN(similarity, eps_opt[name])
n_plot = 5
i_plot = pd.Series(clusters).value_counts().index
for i in i_plot[:n_plot]:
    dataset.plot_grid(idx = clusters == i, n_cols=3, img_min=500, rotate=False)


# ## Possible improvements
# 
# There are many possible improvements to investigate. We list some of them:
# 
# * Use the training set, for example for validating the performance of hyperparameter selection.
# * Use automatic tools for data preprocessing such as image segmentation.
# * Replace MegaDescriptor or MiewID with another model. It is possible to use their combination, local feature models such as SuperPoint or any other.
# * Replace DBSCAN by a method which is not so sensitive to its eps hyperparameters, for example HDBSCAN.

# In[ ]:




