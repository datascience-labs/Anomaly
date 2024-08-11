import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn.functional as F

from args import get_parser
parser = get_parser()
args = parser.parse_args()

def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return [0]
    elif dataset == "MSL":
        return [0]
    elif dataset == "SMD":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

def normalize_anomaly_scores(scores):
    """
    Method for normalizing anomaly scores
    :param scores: anomaly_scores
    """
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return normalized_scores

def get_cosine_sim(node_features):
    feature_num = node_features.shape[1]
    cos_ji_mat = []
    node_i_features = node_features.repeat_interleave(node_features.sahpe, dim=1)
    print('node_i_features: ', node_i_features.shape)
    node_j_features = node_features.repeat(1, feature_num, 1)
    print('node_j_features: ', node_j_features.shape)

    for i in range(feature_num):
        cos_ji = F.cosine_similarity(node_i_features[:, i:i+feature_num], node_j_features[:, i:i+feature_num], dim=2)

        if i == 0:
            cos_ji_mat.append(cos_ji)
            cos_ji_mat = torch.cat(cos_ji_mat)
        else:
            cos_ji_mat = torch.hstack((cos_ji_mat, cos_ji))

    cos_ji_mat = cos_ji_mat.view(cos_ji_mat.shape[0], feature_num, feature_num)

    diag = torch.diagonal(cos_ji_mat, dim1=1, dim2=2)
    print(diag)

    cos_ji_mat = cos_ji_mat - torch.diag_embed(diag)
    # print("cos_ji_mat: ", cos_ji_mat.shape)
    return cos_ji_mat

def get_edge_weights(cos_ji_mat):
    # select top k number of nodes with cosine similarity -> each node i have top k number of edges
    edge_weights = torch.topk(cos_ji_mat, args.topk, dim=-1).values  # cosine similarity values between node i and j of the edge
    topk_indices_ji = torch.topk(cos_ji_mat, args.topk, dim=-1).indices

def createLoader(data, node_features, targets, batch_size, train_val_split):
    node_features = node_features[1:-1]
    targets = targets[1:-1]
    node_feat = node_features
    targ = targets

    if train_val_split == True:
        # train, val, test ratio -> 7.2 : 0.8 : 2
        val_split = 0.1

        # shuffle dataset except test dataset
        dataset_size = int(len(targets))
        indices = list(range(dataset_size))

        split = int(np.floor(val_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:dataset_size], indices[:split]
        # print('indices: ', len(train_indices), len(val_indices))

        # separate node_features, edge_indices, edge_features, and targets into train, val, test
        train_n_features = node_features[train_indices]
        val_n_features = node_features[val_indices]
        train_targets = targets[train_indices]
        val_targets = targets[val_indices]

        train_batches = torch.hstack((torch.tensor(np.arange(0, train_n_features.shape[0] // batch_size)).repeat_interleave(batch_size).view(1, -1), torch.tensor(train_n_features.shape[0] // batch_size).repeat(train_n_features.shape[0] % batch_size).view(1, -1))).view(-1)
        val_batches = torch.hstack((torch.tensor(np.arange(0, val_n_features.shape[0] // batch_size)).repeat_interleave(batch_size).view(1, -1), torch.tensor(val_n_features.shape[0] // batch_size).repeat(val_n_features.shape[0] % batch_size).view(1, -1))).view(-1)

        train_loader = torch.utils.data.DataLoader(train_n_features, batch_size=batch_size)# , shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(val_n_features, batch_size=batch_size)#, shuffle=shuffle)

        global train_node_features, val_node_features, train_target
        train_target = train_targets

        train_node_features = train_n_features
        val_node_features = val_n_features
        return train_loader, val_loader, train_targets, val_targets

    else:
        global test_node_features
        test_node_features = node_features
        
        test_batches = torch.hstack((torch.tensor(np.arange(0, node_features.shape[0] // batch_size)).repeat_interleave(batch_size).view(1, -1), torch.tensor(node_features.shape[0] // batch_size).repeat(node_features.shape[0] % batch_size).view(1, -1))).view(-1)
        test_loader = DynamicGraphTemporalSignalBatch(features=node_features, targets=targets, batches=test_batches) #, causes=test_causes)

        return test_loader, targets

def get_data(path, dataset, batch_size, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if dataset == "SMD":

        x_dim = 38
        f = open(os.path.join(path + "_train.pkl"), "rb")
        train_data = pickle.load(f).reshape((-1, x_dim)) #[train_start:train_end, :]
        f.close()
        try:
            f = open(os.path.join(path + "_test.pkl"), "rb")
            test_data = pickle.load(f).reshape((-1, x_dim)) #[test_start:test_end, :]
            f.close()
        except (KeyError, FileNotFoundError):
            test_data = None
        try:
            f = open(os.path.join(path + "_test_label.pkl"), "rb")
            test_label = pickle.load(f).reshape((-1)) #[test_start:test_end]
            f.close()
        except (KeyError, FileNotFoundError):
            test_label = None

    elif dataset in ["SWaT", "WADI", "PSM", "SMAP", "MSL"]:
        train_data = pd.read_csv(path+f"/{dataset}_train.csv", index_col=0)
        test_data = pd.read_csv(path+f"/{dataset}_test.csv", index_col=0)
        # print(train_data, train_data.shape)
        # print(test_data, test_data.shape)
        
        test_label = test_data.iloc[:, -1]
        test_data = test_data.iloc[:, :-1]
        test_data.columns = [''] * len(test_data.columns)

    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    return (train_data, None), (test_data, test_label)

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, lookback):
    # Create a copy of the 'scores' array to store adjusted scores
    adjusted_scores = scores.copy()

    # Get the number of time steps and the number of channels
    num_time_steps, num_channels = adjusted_scores.shape

    # Iterate through each channel
    for channel in range(num_channels):
        # Iterate through time steps starting from 'lookback'
        for t in range(lookback, num_time_steps):
            # Check if any of the past 'lookback' time steps had non-zero score for this channel
            if np.any(adjusted_scores[t - lookback:t, channel]):
                # If any past score was non-zero, set the current score to 0
                adjusted_scores[t, channel] = 0

        # Normalize the scores of the current channel to a range [0, 1]
        min_val = np.min(adjusted_scores[:, channel])
        max_val = np.max(adjusted_scores[:, channel])
        adjusted_scores[:, channel] = (adjusted_scores[:, channel] - min_val) / (max_val - min_val)

    # Return the adjusted scores array
    return adjusted_scores
