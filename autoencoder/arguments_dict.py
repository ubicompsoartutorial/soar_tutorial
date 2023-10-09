import os

import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_args(updated_args=None):
    args = {
        # Data loading parameters
        'window': 50,
        'overlap': 25,
        'input_size': 3,

        # Dataset parameters
        'dataset': 'pamap2',
        'root_dir': '/coc/pcba1/hharesamudram3/ubicomp_soar_tutorial/testing'
                    '/data_preparation/all_data/Sep-05-2023',
        'data_file': 'pamap2.pkl',
        'num_classes': 12,

        # Encoder and decoder parameters
        'kernel_size': 3,
        'padding': 1,

        # Pre-training parameters
        'batch_size': 256,
        'learning_rate': 1e-4,
        'weight_decay': 0.0,
        'num_epochs': 50,
        'patience': 10,

        # Classification parameters
        'classifier_lr': 3e-4,
        'classifier_wd': 1e-4,
        'classifier_batch_size': 256,
        'saved_model_folder': 'Sep-05-2023/',
        'learning_schedule': 'last_layer',  # all_layers
        'classification_model': 'mlp',  # linear

        # Reproducibility
        'random_seed': 42,

        # CUDA
        'device': torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu"),
    }

    # Updating the args if we want to change the location of the prepared
    # data and saved model etc.
    if updated_args is not None:
        args['root_dir'] = updated_args['root_dir']

    return args
