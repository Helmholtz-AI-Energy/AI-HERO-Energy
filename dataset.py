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
        groups = raw_data.groupby('Country')
        countries = []
        for country, df in groups['Load [MW]']:
            countries.append(torch.tensor(df.to_numpy(), dtype=torch.float))

        # Generate data tensor and metadata
        self.dataset = torch.stack(countries)
        self.country_nr = self.dataset.shape[0]
        self.samples_per_country = self.dataset.shape[1] - self.historic_window - self.forecast_horizon

        # Normalize each country to [0,1]
        if normalize is True:
            self.data_min = torch.min(self.dataset, dim=-1)[0]
            self.data_max = torch.max(self.dataset, dim=-1)[0]
            normlization = (self.data_max - self.data_min).unsqueeze(-1)
            self.dataset = (self.dataset - self.data_min.unsqueeze(-1)) / normlization

        self.dataset = self.dataset.to(device)

    def __len__(self):
        return self.country_nr * self.samples_per_country

    def __getitem__(self, idx):
        # translate idx (day nr) to array index
        country_idx = idx // self.samples_per_country
        hour_idx = idx % self.samples_per_country
        x = self.dataset[country_idx, hour_idx:hour_idx+self.historic_window].unsqueeze(dim=1)
        y = self.dataset[country_idx, hour_idx+self.historic_window:
                                      hour_idx+self.historic_window + self.forecast_horizon].unsqueeze(dim=1)

        return x, y

