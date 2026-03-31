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
from my_models import AnimalReIDRefiner

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Setup Model

# models = dict()
# models['Mega-384'] = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
# # models['Mega-224'] = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-224", pretrained=True, num_classes=0)
# models['miewid'] = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
# models['DINOv2'] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# models['sigLip'] = timm.create_model('vit_so400m_patch14_siglip_224', pretrained=True, num_classes=0)

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

model_names = dict({'SalamanderID2025': 'mega384',
                    'SeaTurtleID2022': 'mega384',
                    'LynxID2025': 'miewid',
                    'TexasHornedLizards': 'miewid'})

wgt_files = dict({'SalamanderID2025': None,#os.path.join(ROOT_MODELS, 'mega384_crefined_SSalamanderID2025_mmr.pth'),
                    'SeaTurtleID2022': None,#os.path.join(ROOT_MODELS, 'mega384_crefined_SeaTurtleID2022_mmr.pth'),
                    'LynxID2025': None,
                    'TexasHornedLizards': None})

enhance_sw = dict({'SalamanderID2025': False, #True
                    'SeaTurtleID2022': False,
                    'LynxID2025': False,
                    'TexasHornedLizards': False})

input_size = dict({'SalamanderID2025': (384, 384),
                   'SeaTurtleID2022': (384, 384),
                   'LynxID2025': (512, 512),
                   'TexasHornedLizards': (512, 512)})

output_paths = dict({'SalamanderID2025': os.path.join(ROOT_FEATURES, 'SalamanderID2025_Mega-384'),#'mega384_crefined_SSalamanderID2025_mmr'),
                    'SeaTurtleID2022': os.path.join(ROOT_FEATURES, 'SeaTurtleID2022_Mega-384'),#'mega384_crefined_SeaTurtleID2022_mmr'),
                    'LynxID2025': os.path.join(ROOT_FEATURES, 'LynxID2025_miewid'),
                    'TexasHornedLizards': os.path.join(ROOT_FEATURES, 'TexasHornedLizards_miewid')})


# for model_name, wgt_file in zip(model_names, wgt_files):
#
#     # # #print(model_name)
#     # # model = models[model_name]
#     # model = AnimalReIDRefiner(model_name, wgt_files)
#     # model.to(device).eval()

for db_name in ['SalamanderID2025', 'SeaTurtleID2022', 'LynxID2025', 'TexasHornedLizards']:

    model = AnimalReIDRefiner(model_name=model_names[db_name], weights_file=wgt_files[db_name], use_projector=False)
    model.to(device).eval()


    #dataset = dataset_full.get_subset(dataset_full.df['split'] == 'train').get_subset(dataset_full.df['dataset'] == dataset_names[0])
    dataset = dataset_full.get_subset(dataset_full.df['dataset'] == db_name)
    ds_len = dataset.__len__()
    print('working on {} ({}  examples) model {}'.format(db_name, ds_len, model_names[db_name]))

    # config = timm.data.resolve_model_data_config(model.backbone)
    # print(config)
    # config['crop_pct'] = 1.0
    #
    # transform = timm.data.create_transform(**config, is_training=False)
    # if enhance_sw[db_name]:
    #     dataset.set_transform(T.Compose([UnderwaterEnhance, transform]))

    transform = T.Compose([
        T.Resize(size=input_size[db_name]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    if enhance_sw[db_name]:
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
            feat1 = model(img[np.newaxis, :, :, :].to(device)).unsqueeze(0)
            # feat2 = model(torch.flip(img[np.newaxis, :, :, :], dims=[2]).to(device)).unsqueeze(0)


            all_features.append(feat1.cpu().numpy())
            # all_features.append((feat1 + feat2).cpu().numpy() / 2)
            # all_features_s.append((feat1 + feat2).cpu().numpy() / 2)

    # fname = os.path.join(ROOT_FEATURES, db_name + '_' + model_names[db_name] + '.npz')
    fname = output_paths[db_name] + '.npz'
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    np.savez(fname, all_features=np.array(all_features), all_labels=np.array(all_labels))
    # np.savez(fname.replace('.npz', '_s.npz'), all_features=np.array(all_features_s), all_labels=np.array(all_labels))

    torch.cuda.empty_cache()

    # model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
    # extractor = DeepFeatures(model=model, device='cuda', batch_size=8)
    # all_features_ = extractor(dataset=dataset)
    #
    # for i in range(7):
    #     print('\n', all_features[i].squeeze()[:5])
    #     print('\n', all_features_.features.squeeze()[i, :5])
    #
    #
    # fname = os.path.join(DATA_ROOT.replace('data', 'features'), name)# + '.npz')
    # os.makedirs(os.path.dirname(fname), exist_ok=True)
    # #np.savez(fname, all_features=np.array(all_features), all_labels=np.array(all_labels))
    # import pickle
    # with open(fname, 'wb') as fd:
    #     pickle.dump(all_features_, fd)
