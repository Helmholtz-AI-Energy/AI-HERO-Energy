import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomLoadDataset(Dataset):
    def __init__(self, data_file, historic_window, forecast_horizon, device=None, normalize=True):
        # Input sequence length and output (forecast) sequence length
        self.historic_window = historic_window
        self.forecast_horizon = forecast_horizon

        # Load Data from csv to Pandas Dataframe
        raw_data = pd.read_csv(data_file, delimiter=',')

        # Group data by city
        groups = raw_data.groupby('City')
        cities = []
        for city, df in groups['Load [MWh]']:
            cities.append(torch.tensor(df.to_numpy(), dtype=torch.float))

        # Generate data tensor and metadata
        self.dataset = torch.stack(cities)
        self.city_nr = self.dataset.shape[0]
        self.samples_per_city = self.dataset.shape[1] - self.historic_window - self.forecast_horizon

        # Normalize Data to [0,1]
        if normalize is True:
            self.data_min = torch.min(self.dataset)
            self.data_max = torch.max(self.dataset)
            self.dataset = (self.dataset - self.data_min) / (self.data_max - self.data_min)

        self.dataset = self.dataset.to(device)

    def __len__(self):
        return self.city_nr * self.samples_per_city

    def __getitem__(self, idx):
        # translate idx (day nr) to array index
        city_idx = idx // self.samples_per_city
        hour_idx = idx % self.samples_per_city
        x = self.dataset[city_idx, hour_idx:hour_idx+self.historic_window].unsqueeze(dim=1)
        y = self.dataset[city_idx, hour_idx+self.historic_window:
                                   hour_idx+self.historic_window + self.forecast_horizon].unsqueeze(dim=1)

        return x, y

    def revert_normalization(self, data):
        return data * (self.data_max - self.data_min) + self.data_min
