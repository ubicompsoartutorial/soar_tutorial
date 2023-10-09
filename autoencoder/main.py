from autoencoder.arguments_dict import load_args
from autoencoder.pretrainer import learn_model
from autoencoder.utils import set_all_seeds

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = load_args()
    set_all_seeds(args['random_seed'])
    print(args)

    # Starting the pre-training
    learn_model(args=args)

    print('------ Pre-training complete! ------')
