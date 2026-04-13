from paths_and_constants import *

class model_feature_config:

    def __init__(self):

        self.select_config_version('baseline')


    def select_config_version(self, version='baseline'):

        assert version in ['baseline', 'best', 'rsrch']

        if version == 'baseline':
            self.config = self.baseline_config
        if version == 'best':
            self.config = self.best_config
        if version == 'rsrch':
            self.config = self.rsrch_config


    def get_training_config(self, db_name):

        cfg = self.config[db_name]

        return {'model_name': cfg['model_name'],
                'wgt_file': cfg['wgt_file'],
                'enhance': cfg['enhance'],
                'use_projector': cfg['use_projector'],
                'size': cfg['size'],
                'lr': cfg['lr']
                }


    def get_embedding_config(self, db_name):

        cfg = self.config[db_name]

        return {'model_name': cfg['model_name'],
                'wgt_file': cfg['wgt_file'],
                'enhance': cfg['enhance'],
                'use_projector': cfg['use_projector'],
                'size': cfg['size'],
                'feat_file': cfg['feat_file']
                }


    def get_classification_config(self, db_name):

        cfg = self.config[db_name]

        return {'feature_file': cfg['feat_file'],
                'flow': cfg['flow']}



    baseline_config = dict()
    best_config = dict()
    rsrch_config = dict()

    baseline_config['SalamanderID2025'] = dict({'model_name': 'mega384',
                                                'wgt_file': None,
                                                'enhance': False,
                                                'use_projector': False,
                                                'size': (384, 384),
                                                'feat_file': 'SalamanderID2025_Mega-384',
                                                'flow': 0
                                                })

    baseline_config['SeaTurtleID2022'] = dict({'model_name': 'mega384',
                                               'wgt_file': None,
                                               'enhance': False,
                                               'use_projector': False,
                                               'size': (384, 384),
                                               'feat_file': 'SeaTurtleID2022_Mega-384',
                                               'flow': 0
                                               })

    baseline_config['LynxID2025'] = dict({'model_name': 'miewid',
                                          'wgt_file': None,
                                          'enhance': False,
                                          'use_projector': False,
                                          'size': (512, 512),
                                          'feat_file': 'LynxID2025_miewid',
                                          'flow': 0
                                          })

    baseline_config['TexasHornedLizards'] = dict({'model_name': 'miewid',
                                                  'wgt_file': None,
                                                  'enhance': False,
                                                  'use_projector': False,
                                                  'size': (512, 512),
                                                  'feat_file': 'TexasHornedLizards_miewid',
                                                  'flow': 0
                                                  })

    # --------------------------------------------------------------------------------------

    best_config['SalamanderID2025'] = dict({'model_name': 'mega384',
                                                'wgt_file': None,
                                                'enhance': False,
                                                'use_projector': False,
                                                'size': (384, 384),
                                                'feat_file': 'SalamanderID2025_Mega-384',
                                                'flow': 1  # clustering with brute-force grid search
                                                })

    best_config['SeaTurtleID2022'] = dict({'model_name': 'mega384',
                                               'wgt_file': None,
                                               'enhance': False,
                                               'use_projector': False,
                                               'size': (384, 384),
                                               'feat_file': 'SeaTurtleID2022_Mega-384',
                                               'flow': 1  # clustering with brute-force grid search
                                               })

    best_config['LynxID2025'] = dict({'model_name': 'miewid',
                                          'wgt_file': None,
                                          'enhance': False,
                                          'use_projector': False,
                                          'size': (512, 512),
                                          'feat_file': 'LynxID2025_miewid',
                                          'flow': 1  # clustering with brute-force grid search
                                          })

    best_config['TexasHornedLizards'] = dict({'model_name': 'miewid',
                                                  'wgt_file': None,
                                                  'enhance': False,
                                                  'use_projector': False,
                                                  'size': (512, 512),
                                                  'feat_file': 'TexasHornedLizards_miewid',
                                                  'flow': 0
                                                  })

# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------

    rsrch_config['SalamanderID2025'] = dict({'model_name': 'mega384',
                                                'wgt_file': 'SalamanderID2025_Mega-384_rfnd',
                                                'enhance': False,
                                                'use_projector': True,
                                                'size': (384, 384),
                                                'lr': 1e-4,
                                                'feat_file': 'SalamanderID2025_Mega-384',
                                                'flow': 1 # clustering with brute-force grid search
                                                })

    rsrch_config['SeaTurtleID2022'] = dict({'model_name': 'mega384',
                                                'wgt_file': 'SeaTurtleID2022_Mega-384_rfnd',
                                                'enhance': False,
                                                'use_projector': True,
                                                'size': (384, 384),
                                                'lr': 1e-4,
                                                'feat_file': 'SeaTurtleID2022_Mega-384',
                                                'flow': 1 # clustering with brute-force grid search
                                                })

    rsrch_config['LynxID2025'] = dict({'model_name': 'resnet',
                                                'wgt_file': 'LynxID2025_resnet',
                                                'enhance': True,
                                                'use_projector': True,
                                                'size': (512, 512),#(384, 384),#
                                                'lr': 1e-4,
                                                'feat_file': 'LynxID2025_resnet',
                                                'flow': 1 # clustering with brute-force grid search
                                                })

    rsrch_config['TexasHornedLizards'] = dict({'model_name': 'miewid',
                                                'wgt_file': 'TexasHornedLizards_Mega-384_rfnd',
                                                'enhance': False,
                                                'use_projector': False,
                                                'size': (512, 512),
                                                'feat_file': 'TexasHornedLizards_miewid',
                                                'flow': 0
                                                })

#--------------------------------------------------------------------------------------
