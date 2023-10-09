import numpy as np
import os
import pandas as pd
import random
from datetime import date

# For data preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tqdm.auto import tqdm

# Setting seeds
np.random.seed(42)
random.seed(42)


def load_args():
    print('Parameters for preparing PAMAP2 -- wrist')
    # TODO: add later the location from Drive
    dataset_loc = 'data/PAMAP2_Dataset'

    args = {'dataset_loc': dataset_loc,
            'original_sampling_rate': 100,
            'sampling_rate': 50}
    return args


def map_activity_to_id():
    # chosen activity list
    chosen = {
        24: 'rope jumping', 1: 'lying', 2: 'sitting', 3: 'standing',
        4: 'walking', 5: 'running', 6: 'cycling', 7: 'Nordic walking',
        12: 'ascending stairs', 13: 'descending stairs',
        16: 'vacuum cleaning', 17: 'ironing'
    }

    chosen_activity_list = ['rope jumping', 'lying', 'sitting', 'standing',
                            'walking', 'running', 'cycling', 'Nordic walking',
                            'ascending stairs', 'descending stairs',
                            'vacuum cleaning', 'ironing']

    return chosen, chosen_activity_list


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj,
                                                 test_size=test_size,
                                                 random_state=42)
    print('The train and validation subjects are: {}'.format(train_val_subj))
    print('The test subjects are: {}'.format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj, test_size=val_size,
                                            random_state=42)

    subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}
    print(subjects)

    return subjects


def get_acc_columns(col_start):
    # accelerometer
    sensor_col = np.arange(col_start + 1, col_start + 4)

    return sensor_col


def get_data(args):
    path_pamap2 = args['dataset_loc']
    path_protocol = os.path.join(path_pamap2, 'Protocol')
    path_optional = os.path.join(path_pamap2, 'Optional')

    # The list of files in the protocol folder
    list_protocol = [os.path.join(path_protocol, item) for item in
                     os.listdir(path_protocol)]
    list_protocol.sort()

    # List of files in the optional folder
    list_optional = [os.path.join(path_optional, item) for item in
                     os.listdir(path_optional)]
    list_optional.sort()

    # Concatenating them together
    list_protocol += list_optional

    # Picking the sensor columns for the wrist IMU
    col_start = 3
    sensor_col = get_acc_columns(col_start)

    # Sanity check
    assert len(sensor_col) == 3, \
        ('Check the number of sensor channels obtained! Must be 3, is: {}'
         .format(len(sensor_col)))

    # Placeholders to concatenate later
    sensor_all = np.empty((0, 3))
    target_all = np.empty((0,))
    subject_all = np.empty((0,))
    target_col = 1

    for prot in tqdm(list_protocol):
        data = np.loadtxt(prot)
        
        assert data.shape[1] == 54

        # downsample
        interval = int(args['original_sampling_rate'] / args['sampling_rate'])
        # print('The interval is: {}'.format(interval))
        idx_ds = np.arange(0, data.shape[0], interval)
        data = data[idx_ds]

        # Get activity label
        target = data[:, target_col]

        # handle NaN values
        sensor = np.array(
            [pd.Series(i).interpolate() for i in data[:, sensor_col].T]
        )

        # forward fill and backward fill missing values in the beginning and end
        sensor = np.array(
            [pd.Series(i).fillna(method='ffill') for i in sensor])
        sensor = np.array(
            [pd.Series(i).fillna(method='bfill') for i in sensor]).T
        assert np.all(np.isfinite(sensor)), ('NaN in samples')
        assert sensor.shape[1] == len(sensor_col)

        # get subject
        basename = os.path.splitext(os.path.basename(prot))[0]
        assert basename[:-1] == 'subject10'
        sID = int(basename[-1])
        subject = np.ones((target.shape[0],)) * sID
        print(f"Subject {sID}\t data shape: {data.shape}")

        # Concatenate
        sensor_all = np.concatenate((sensor_all, sensor), axis=0)
        target_all = np.concatenate((target_all, target), axis=0)
        subject_all = np.concatenate((subject_all, subject), axis=0)

    # Putting it back into a dataframe
    df_cols = {'user': subject_all, 'label': target_all}
    locs = ['ankle']
    sensor_names = ['acc']
    axes = ['x', 'y', 'z']

    # Looping over all sensor locations
    count = 0
    sensor_col_names = []
    for loc in locs:
        for name in sensor_names:
            for axis in axes:
                c = loc + '_' + name + '_' + axis
                df_cols[c] = sensor_all[:, count]
                sensor_col_names.append(c)
                count += 1

    df = pd.DataFrame(df_cols)

    # Final size check
    assert df.shape[1] == 5, ('All columns were not copied. '
                              'Expected 5, got {}'.format(df.shape[1]))
    print("==============================")
    print(f'All subjects \t data shape: {df.shape}')
    

    # Removing some classes
    activity_id, activity_list = map_activity_to_id()

    df = df[df.label.isin(activity_id.keys())]
    print('After removal \t data shape: {}'.format(
        df.shape))

    # print('The activities are: {}'.format(np.unique(df['label'])))

    # Need to encode the labels from 0:N-1 than what was available earlier
    le = LabelEncoder()
    encoded = le.fit_transform(df['label'].values)
    df['gt'] = encoded
    
    print("")
    print("Activity mapping:")
    print("Encoded label - activity")
    for k, v in activity_id.items():
        # print(f"{le.transform([k])[0]} - {k} - {v}")
        print(f"{le.transform([k])[0]} - {v}")
    print("")
    
    
    return df, sensor_col_names


def get_data_from_split(df, split, args, sensors):
    # Let us partition by train, val and test splits
    train_data = df[df['user'].isin(split['train'])]
    val_data = df[df['user'].isin(split['val'])]
    test_data = df[df['user'].isin(split['test'])]

    processed = {'train': {'data': train_data[sensors].values,
                           'labels': train_data['gt'].values},
                 'val': {'data': val_data[sensors].values,
                         'labels': val_data['gt'].values},
                 'test': {'data': test_data[sensors].values,
                          'labels': test_data['gt'].values},
                 'fold': split
                 }

    # Sanity check on the sizes
    for phase in ['train', 'val', 'test']:
        assert processed[phase]['data'].shape[0] == \
               len(processed[phase]['labels'])

    for phase in ['train', 'val', 'test']:
        print(f"{phase}\t data shape = {processed[phase]['data'].shape} \t label shape = {processed[phase]['labels'].shape}")

    # Creating logs by the date now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    folder = os.path.join(dir_path, 'all_data', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    os.makedirs(os.path.join(folder, 'unnormalized'), exist_ok=True)
    save_name = 'pamap2.pkl'

    # Saving the joblib file
    name = os.path.join(folder, 'unnormalized', save_name)
    pd.to_pickle(processed, name)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed['train']['data'])
    for phase in ['train', 'val', 'test']:
        processed[phase]['data'] = \
            scaler.transform(processed[phase]['data'])

    # Saving into a joblib file
    name = os.path.join(folder, save_name)

    print("")
    print(f'Saved the processed data to {name}')
    pd.to_pickle(processed, name)

    return processed


def prepare_data(args):
    # Getting all the available data
    df, sensors = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df['user'].values)
    print('The unique subjects are: {}'.format(unique_subj))

    # Performing the train-val-test split=
    split = perform_train_val_test_split(unique_subj)
    processed = get_data_from_split(df=df, split=split, args=args,
                                    sensors=sensors)

    return processed


# ---------------------------------------------------------------------------------------------------------------------
def main():
    args = load_args()
    print(args)

    processed = prepare_data(args)
    print('Data preparation complete!')

    return processed
