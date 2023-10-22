"""
This is not correct. The generated data is not similar to the original data. The standard deviations are much smaller for the generated data compared to the original data, and this problem seems to get worse with decreasing batch size. A similar problem was observed with the wine data set presented in a course video. There is probably something wrong with the AutoEncoder class which was provided for this project.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from TestModel import test_model

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(path)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len = self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len = self.x.shape[0]
        del self.X_train
        del self.X_test

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


class Autoencoder(nn.Module):
    def __init__(self, D_in, H=50, H2=12, latent_dim=3):
        # Encoder
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        # Latent vectors
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def load_and_standardize_data(path):
    df = pd.read_csv(path, sep=",")
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype("float32")
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

def generate_fake(mu, logvar, no_samples, scaler, model):
    """
    With trained model, generate some data
    """
    sigma = torch.exp(logvar / 2)
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([no_samples]))
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
    fake_data = scaler.inverse_transform(pred)
    return fake_data


def main():
    # Get a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data paths. You need paths for the original data, the data with just loan status = 1 and the new augmented dataset.
    path_original = Path("data/loan_continuous.csv")
    path_status1 = Path("data/loan_continuous_status1.csv")
    path_expanded = Path("data/loan_continuous_expanded.csv")

    # Split the data out with loan status = 1. Drop "Loan Status" column and save to a new file.
    data_original = pd.read_csv(path_original, header=0)
    data_status1 = data_original[data_original["Loan Status"] == 1].drop(columns=["Loan Status"])
    data_status1.to_csv(path_status1, index=False)

    # Baseline metrics
    outfile = Path("results/baseline_metrics.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if not outfile.exists():
        test_model(path_original, outfile)

    # Create DataLoaders for training and validation
    train_dataset = DataBuilder(path_status1, train=True)
    test_dataset = DataBuilder(path_status1, train=False)
    ntrain = len(train_dataset)
    ntest = len(test_dataset)

    trainloader = DataLoader(dataset=train_dataset, batch_size=ntrain)
    testloader = DataLoader(dataset=test_dataset, batch_size=ntrain)
    
    # Train and validate the model
    model = Autoencoder(train_dataset.x.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = CustomLoss()
    nepochs = 5000
    train_losses = []
    test_losses = []

    def train(epoch):
        model.train()
        train_loss = 0

        for _, data in enumerate(trainloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_func(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss_avg = train_loss / ntrain
        train_losses.append(train_loss_avg)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Average train loss: {train_loss_avg:.2f}")

    def test(epoch):
        with torch.no_grad():
            test_loss = 0

            for _, data in enumerate(testloader):
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = loss_func(recon_batch, data, mu, logvar)
                test_loss += loss.item()

            test_loss_avg = test_loss / ntest
            test_losses.append(test_loss_avg)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Average test loss: {test_loss_avg:.2f}")
                
    for epoch in range(1, nepochs + 1):
        train(epoch)
        test(epoch)
            
    # Save training progress
    training_progress = pd.DataFrame({"train_loss": train_losses, "test_loss": test_losses})
    outfile = Path("results/training_progress.csv")
    training_progress.to_csv(outfile, index=False)

    # Generate data
    scaler = trainloader.dataset.standardizer

    with torch.no_grad():
        for _, data in enumerate(testloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            try:
                mu_all = torch.cat((mu_all, mu), 0)
                logvar_all = torch.cat((logvar_all, logvar), 0)
            except NameError:
                mu_all = mu
                logvar_all = logvar

    fake_data = generate_fake(mu_all, logvar_all, 50000, scaler, model)
    fake_data = pd.DataFrame(fake_data, columns=data_status1.columns)

    # Check that the reconstructed data is similar to the original data    
    recon_row = scaler.inverse_transform(recon_batch[0].reshape(1, -1).cpu().numpy())
    real_row = scaler.inverse_transform(data[0].reshape(1, -1).cpu().numpy())
    print(recon_row)
    print(real_row)
    
    # Check differences in means and standard deviations of the generated data compared to the original data
    print("RMSD for percent difference in means:")
    original_means = data_status1.mean(axis=0)
    generated_means = fake_data.mean(axis=0)
    percent_diff_sq = (100 * (generated_means - original_means) / original_means)**2
    percent_diff_sq = percent_diff_sq[percent_diff_sq != np.inf]
    rmsd = np.sqrt(np.mean(percent_diff_sq))
    print(rmsd)
    print("RMSD for percent difference in standard deviations:")
    original_std = data_status1.std(axis=0)
    generated_std = fake_data.std(axis=0)
    percent_diff_sq = (100 * (generated_std - original_std) / original_std)**2
    percent_diff_sq = percent_diff_sq[percent_diff_sq != np.inf]
    rmsd = np.sqrt(np.mean(percent_diff_sq))
    print(rmsd)
       
    # Add column of 1s to generated data for loan status
    fake_data["Loan Status"] = 1

    # Combine the generated data with the original dataset. Save to a new file.
    data_expanded = pd.concat([data_original, fake_data], ignore_index=True)
    data_expanded.to_csv(path_expanded, index=False)

    # Classify loan status on the expanded dataset
    outfile = Path("results/expanded_metrics.json")
    test_model(path_expanded, outfile)

if __name__ == "__main__":
    main()
    print("done")
