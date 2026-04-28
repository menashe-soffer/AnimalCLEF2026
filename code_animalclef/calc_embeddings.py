import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

import tqdm
import torch
import torchvision.transforms as T
from wildlife_datasets.datasets import AnimalCLEF2026

from paths_and_constants import *
from image_tools import UnderwaterEnhance
from my_models import AnimalReIDRefiner
from model_featue_config import model_feature_config

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



CONFIG = 'rsrch' # 'baseline', 'best, 'rsrch'
config = model_feature_config()
config.select_config_version(CONFIG)


for db_name in ['SalamanderID2025', 'SeaTurtleID2022', 'LynxID2025', 'TexasHornedLizards'][1:2]:

    cfg = config.get_embedding_config(db_name)

    #model = AnimalReIDRefiner(model_name=model_names[db_name], weights_file=wgt_files[db_name], use_projector=False)
    wgt_file = weights_file=os.path.join(ROOT_MODELS, cfg['wgt_file'] + '.pth') if cfg['wgt_file'] is not None else None
    if wgt_file is not None:
        param_file = wgt_file.replace('.pth', '.json').replace('_sil', '')
        if os.path.exists(param_file):
            params = json.load(open(param_file, 'r'))['model_params']
    else:
        params = {'use_projector': False, 'use_marg': False, 'projection_dim': None, 'logit_dim': None, 'K': None}
    #
    params = {'use_projector': True, 'projection_dim': 512, 'use_marg': False, 'logit_dim': 40, 'K': 3}
    cfg['wgt_file'] = wgt_file
    model = AnimalReIDRefiner(model_name=cfg['model_name'], weights_file=wgt_file,
                              use_projector=params['use_projector'], projection_dim=params['projection_dim'],
                              use_marg=params['use_marg'], marg_num_clases=params['logit_dim'], marg_K=params['K'])
    model.to(device).eval()


    #dataset = dataset_full.get_subset(dataset_full.df['split'] == 'train').get_subset(dataset_full.df['dataset'] == dataset_names[0])
    dataset = dataset_full.get_subset(dataset_full.df['dataset'] == db_name)
    ds_len = dataset.__len__()
    print('working on {} ({}  examples) model {}'.format(db_name, ds_len, cfg['model_name']))

    # config = timm.data.resolve_model_data_config(model.backbone)
    # print(config)
    # config['crop_pct'] = 1.0
    #
    # transform = timm.data.create_transform(**config, is_training=False)
    # if enhance_sw[db_name]:
    #     dataset.set_transform(T.Compose([UnderwaterEnhance, transform]))

    transform = T.Compose([
        #T.Resize(size=((384, 384))),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    if cfg['enhance']:
        dataset.set_transform(T.Compose([UnderwaterEnhance, transform]))
    dataset.set_transform(transform)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    #continue

    all_features, all_labels, all_embeddings = [], [], []
    #all_features_blr, all_features_mix, all_features_s = [], [], []
    print('extracting features for {} images'.format(ds_len))
    with torch.no_grad():
        for idx in tqdm.tqdm(range(ds_len)):
            img, label = dataset.__getitem__(idx)
            all_labels.append(label)
            feat1 = model(img[np.newaxis, :, :, :].to(device)).unsqueeze(0)
            emb = model.get_embedding()
            # feat2 = model(torch.flip(img[np.newaxis, :, :, :], dims=[2]).to(device)).unsqueeze(0)


            all_features.append(feat1.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
            # all_features.append((feat1 + feat2).cpu().numpy() / 2)
            # all_features_s.append((feat1 + feat2).cpu().numpy() / 2)

    # fname = os.path.join(ROOT_FEATURES, db_name + '_' + model_names[db_name] + '.npz')
    fname = os.path.join(ROOT_FEATURES, cfg['feat_file'] + '.npz')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    np.savez(fname, all_features=np.array(all_features), all_labels=np.array(all_labels), all_embeddings=np.array(all_embeddings))
    # np.savez(fname.replace('.npz', '_s.npz'), all_features=np.array(all_features_s), all_labels=np.array(all_labels))

    torch.cuda.empty_cache()



