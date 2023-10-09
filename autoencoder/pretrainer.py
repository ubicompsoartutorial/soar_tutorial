import copy
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from autoencoder.dataset import load_har_dataset
from autoencoder.meter import RunningMeter, BestMeter
from autoencoder.model import AutoEncoder
from tqdm.auto import tqdm
from autoencoder.utils import compute_best_metrics, update_loss, save_meter, \
    save_model, set_all_seeds


def learn_model(args=None):
    print('Starting the pre-training')
    print(args)

    # Setting seed once again
    set_all_seeds(args['random_seed'])

    # Data loaders
    data_loaders, dataset_sizes = load_har_dataset(args)

    # Tracking meter
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = AutoEncoder(args).to(args['device'])

    optimizer = optim.AdamW(model.parameters(),
                            lr=args['learning_rate'],
                            weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.8)
    criterion = nn.MSELoss()

    trigger_times = 0

    for epoch in tqdm(range(0, args['num_epochs'])):
        since = time()

        # Training
        model, optimizer = train(model,
                                 data_loaders["train"],
                                 criterion,
                                 optimizer,
                                 args,
                                 epoch,
                                 dataset_sizes["train"],
                                 running_meter
                                 )

        scheduler.step()

        # Evaluating on the validation data
        evaluate(model,
                 data_loaders["val"],
                 args,
                 criterion,
                 epoch,
                 phase="val",
                 dataset_size=dataset_sizes["val"],
                 running_meter=running_meter
                 )

        # Saving the logs
        save_meter(args, running_meter)

        # Doing the early stopping check
        if epoch >= 2:
            if running_meter.loss['val'][-1] > best_meter.loss["val"]:
                trigger_times += 1
                # print('Trigger times: {}'.format(trigger_times))

                if trigger_times >= args['patience']:
                    print('Early stopping the model at epoch: {}. The '
                          'validation loss has not improved for {}'.format(
                        epoch, trigger_times))
                    break
            else:
                trigger_times = 0
                # print('Resetting the trigger counter for early stopping')

        # Updating the best weights
        if running_meter.loss["val"][-1] < best_meter.loss["val"]:
            best_meter = compute_best_metrics(running_meter, best_meter)
            running_meter.update_best_meter(best_meter)

            best_model_wts = copy.deepcopy(model.state_dict())

            # Saving the logs
            save_meter(args, running_meter)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Saving the best performing model
    save_model(model, args, epoch=epoch)

    return


def train(model, data_loader, criterion, optimizer, args, epoch, dataset_size,
          running_meter):
    # Setting the model to training mode
    model.train()

    # To track the loss and other metrics
    running_loss = 0.0

    # Iterating over the data
    for inputs, _ in data_loader:
        inputs = inputs.float().to(args['device'])

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)

            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)

    # Statistics
    loss = running_loss / dataset_size
    update_loss(phase="train",
                running_meter=running_meter,
                loss=loss,
                epoch=epoch)

    return model, optimizer


def evaluate(model, data_loader, args, criterion, epoch, phase, dataset_size,
             running_meter):
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0

    # Iterating over the data
    for inputs, _ in data_loader:
        inputs = inputs.float().to(args['device'])

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)

    # Statistics
    loss = running_loss / dataset_size
    update_loss(phase=phase,
                running_meter=running_meter,
                loss=loss,
                epoch=epoch)

    return
