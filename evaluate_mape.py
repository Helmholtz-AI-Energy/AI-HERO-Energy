import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from pandas import DataFrame
from dataset import CustomLoadDataset

# TODO: import your model
from model import LoadForecaster as SubmittedModel


def forecast(forecast_model, forecast_set, device):
    forecast_model.to(device)
    forecast_model.eval()

    batch_size = forecast_set.samples_per_country
    forecast_loader = DataLoader(forecast_set, batch_size=batch_size, shuffle=False)
    forecasts = torch.zeros([len(forecast_set), 7*24], device=device)
    reference = torch.zeros([len(forecast_set), 7*24], device=device)

    for n, (input_seq, output_seq) in enumerate(forecast_loader):
        # TODO: adjust forecast loop according to your model
        with torch.no_grad():
            actual_batch_size = input_seq.shape[0]  # last batch has different size

            # Normalize data
            data_min = torch.min(input_seq)
            data_max = torch.max(input_seq)
            normalized_input = (input_seq - data_min) / (data_max - data_min)

            # Forecast
            hidden = forecast_model.init_hidden(actual_batch_size)
            normalized_prediction, hidden = forecast_model(normalized_input, hidden)

            # Denormalize
            prediction = normalized_prediction.squeeze(dim=-1) * (data_max - data_min) + data_min

            # Store
            forecasts[n * actual_batch_size:n * actual_batch_size + actual_batch_size] = prediction
            reference[n * actual_batch_size:n * actual_batch_size + actual_batch_size] = output_seq.squeeze(dim=-1)
    return forecasts, reference


if __name__ == '__main__':
    print("START")
    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str,
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/AI-HERO_Energy/energy_baseline.pt',
                        help="Model weights path")  # TODO: adapt to your model weights path
    parser.add_argument("--data_dir", type=str, help='Directory containing the data you want to predict',
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/data')
    parser.add_argument("--force_cpu", action='store_true', default=False)
    args = parser.parse_args()

    data_dir = args.data_dir

    weights_path = args.weights_path

    # load model with pretrained weights
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    # TODO: adjust arguments according to your model
    model = SubmittedModel(input_size=1, hidden_size=48, output_size=1, num_layer=1, device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # dataloader
    if os.path.isfile(data_dir):
        data_file = data_dir
    else:
        test_file = os.path.join(data_dir, 'test.csv')
        valid_file = os.path.join(data_dir, 'valid.csv')
        data_file = test_file if os.path.exists(test_file) else valid_file
    testset = CustomLoadDataset(data_file, 7*24, 7*24, normalize=False, device=device)

    # run inference
    forecasts, reference = forecast(model, testset, device)

    normalized = torch.abs(1 - torch.div(forecasts, reference))
    result = 100 * torch.mean(normalized)

    print(f"MAPE: {result}%")
