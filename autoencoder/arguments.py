import argparse
import os

import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for the Autoencoder pre-training '
                    'and evaluation')

    # Data loading parameters
    parser.add_argument('--window', type=int, default=100, 
                        help='Window size')
    parser.add_argument('--overlap', type=int, default=50,
                        help='Overlap for sliding window')
    parser.add_argument('--input_size', type=int, default=3,
                        help='Number of channels in input data')
    parser.add_argument('--dataset', type=str, default='pamap2',
                        help='Dataset used for pre-training or classification')
    parser.add_argument('--root_dir', type=str, default='',
                        help='Root directory for the data')
    parser.add_argument('--data_file', type=str, default='pamap2.pkl',
                        help='Name of the processed data file')
    parser.add_argument('--num_classes', type=int, default=12,
                        help='Number of activities to classify')

    # -----------------------------------------------------------
    # Pre-training settings
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for pre-training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for pre-training')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay /l2 regularization for pre-training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Max number of ecpohs for pre-training')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Selecting the GPU to execute it with')
    parser.add_argument('--patience', type=int, default=5,
                        help='number of epochs to wait before early stopping')

    # -----------------------------------------------------------

    # Encoder and decoder params
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='The filter size for the conv layers')
    parser.add_argument('--padding', type=int, default=1,
                        help='The padding size for the conv layers')

    # -----------------------------------------------------------

    # Classification parameters
    parser.add_argument('--classifier_lr', type=float, default=3e-4,
                        help='Learning rate for the classifier')
    parser.add_argument('--classifier_wd', type=float, default=1e-4,
                        help='L2 norm / weight decay for the classifier')
    parser.add_argument('--classifier_batch_size', type=int, default=256,
                        help='Batch size for the classifier')
    parser.add_argument('--saved_model_folder', type=str, default=None,
                        help='The pretrained model folder')
    parser.add_argument('--learning_schedule', type=str, default='last_layer',
                        choices=['last_layer', 'all_layers'],
                        help='whether to train all layers or the last layer')
    parser.add_argument('--classification_model', type=str, default='mlp',
                        choices=['linear', 'mlp'],
                        help='Choosing the classifier: linear is a single FC '
                             'layer whereas MLP is the 3-layer network with '
                             'BN and ReLU and dropout.')

    # ------------------------------------------------------------
    # Setting the fold
    parser.add_argument('--fold', type=int, default=0,
                        help="tracking the fold of the target dataset, "
                             "for 5-fold evaluation")

    # ------------------------------------------------------------
    # Random seed setting
    parser.add_argument('--random_seed', type=int, default=42)

    # -----------------------------------------------------------
    args, _ = parser.parse_known_args()

    args.device = torch.device(
        "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

    args.padding = int(args.kernel_size // 2)

    return args
