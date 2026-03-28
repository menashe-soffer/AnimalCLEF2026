import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torchvision.transforms.functional as F
import tqdm
import torch
import torchvision.transforms as T
import timm  # for the mage-descroptor feature extraction net
from transformers import AutoModel
from sklearn.cluster import DBSCAN
#from wildlife_datasets import datasets
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
#from wildlife_tools.similarity import CosineSimilarity

from paths_and_constants import *
from image_tools import UnderwaterEnhance

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Setup Model

models = dict()
models['Mega-384'] = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
# models['Mega-224'] = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-224", pretrained=True, num_classes=0)
models['miewid'] = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
models['DINOv2'] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
models['sigLip'] = timm.create_model('vit_so400m_patch14_siglip_224', pretrained=True, num_classes=0)

# the super dataset
#DATA_ROOT = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026/data'
#datasets.AnimalCLEF2026.get_data(root)
dataset_full = AnimalCLEF2026(
    ROOT_DATA,
    transform=None,
    load_label=True, # return label as 2nd parameter
    factorize_label=True,   # replace string for unique integer
    check_files=False
)

dataset_names = dataset_full.metadata['dataset'].unique()
print(dataset_names)


for model_name in models:

    if model_name != 'miewid':
        continue

    #print(model_name)
    model = models[model_name]
    model.to(device).eval()

    for name in dataset_names:

        # if name != 'SeaTurtleID2022':
        #     continue

        #
        if model_name == 'Mega-384':
            wgt_fname = os.path.join(ROOT_MODELS, 'mega384_refined_{}.pth'.format(name))
            weights = torch.load(wgt_fname)
            # Create a new dictionary without the "model." prefix
            new_weights = {}
            for k, v in weights.items():
                if k.startswith("model."):
                    new_weights[k[6:]] = v  # Remove 'model.' (which is 6 characters)
                else:
                    new_weights[k] = v
            model.load_state_dict(new_weights)
            model.to(device).eval()
        #

        #dataset = dataset_full.get_subset(dataset_full.df['split'] == 'train').get_subset(dataset_full.df['dataset'] == dataset_names[0])
        dataset = dataset_full.get_subset(dataset_full.df['dataset'] == name)
        ds_len = dataset.__len__()
        print('working on {} ({}  examples) model {}'.format(name, ds_len, model_name))

        config = timm.data.resolve_model_data_config(model)
        print(config)
        config['crop_pct'] = 1.0
        transform = timm.data.create_transform(**config, is_training=False)
        if name not in ['LynxID2025', 'TexasHornedLizards']:
            dataset.set_transform(T.Compose([UnderwaterEnhance, transform]))
        dataset.set_transform(transform)
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        #continue

        all_features, all_labels = [], []
        all_features_blr, all_features_mix, all_features_s = [], [], []
        print('extracting features for {} images'.format(ds_len))
        with torch.no_grad():
            for idx in tqdm.tqdm(range(ds_len)):
                img, label = dataset.__getitem__(idx)
                all_labels.append(label)
                # feat = model(img[np.newaxis, :, :, :].to(device)).unsqueeze(0)
                # all_features.append(feat.cpu().numpy())
                feat1 = model(img[np.newaxis, :, :, :].to(device)).unsqueeze(0)
                feat2 = model(torch.flip(img[np.newaxis, :, :, :], dims=[2]).to(device)).unsqueeze(0)
                # img_blr = F.resize(F.resize(img, (24, 24)), img.shape[1:])
                # feat3 = model(img_blr[np.newaxis, :, :, :].to(device)).unsqueeze(0)
                # feat4 = model(torch.flip(img_blr[np.newaxis, :, :, :], dims=[2]).to(device)).unsqueeze(0)


                all_features.append((feat1 + feat2).cpu().numpy() / 2)
                all_features_s.append((feat1 + feat2).cpu().numpy() / 2)
                # all_features_blr.append((feat3 + feat4).cpu().numpy() / 2)
                # all_features_mix.append((feat1 + feat2 + feat3 + feat4).cpu().numpy() / 4)

        fname = os.path.join(ROOT_FEATURES, name + '_' + model_name + '_miewid.npz')
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez(fname, all_features=np.array(all_features), all_labels=np.array(all_labels))
        # np.savez(fname.replace('.npz', '_blr.npz'), all_features=np.array(all_features_blr), all_labels=np.array(all_labels))
        # np.savez(fname.replace('.npz', '_mix.npz'), all_features=np.array(all_features_mix), all_labels=np.array(all_labels))
        np.savez(fname.replace('.npz', '_s.npz'), all_features=np.array(all_features_s), all_labels=np.array(all_labels))

        torch.cuda.empty_cache()

        # extractor = DeepFeatures(model=model, device='cuda', batch_size=8)
        # all_features_ = extractor(dataset=dataset)
        #
        #
        # fname = os.path.join(DATA_ROOT.replace('data', 'features'), name)# + '.npz')
        # os.makedirs(os.path.dirname(fname), exist_ok=True)
        # #np.savez(fname, all_features=np.array(all_features), all_labels=np.array(all_labels))
        # import pickle
        # with open(fname, 'wb') as fd:
        #     pickle.dump(all_features_, fd)
