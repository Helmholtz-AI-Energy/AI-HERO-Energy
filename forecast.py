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

    batch_size = 64
    forecast_loader = DataLoader(forecast_set, batch_size=64, shuffle=False)
    forecasts = torch.zeros([len(forecast_set), 7*24], device=device)
    for n, (input_seq, _) in enumerate(forecast_loader):
        # TODO: adjust forecast loop according to your model
        with torch.no_grad():
            actual_batch_size = len(input_seq)  # last batch has different size
            hidden = forecast_model.init_hidden(actual_batch_size)
            prediction, hidden = forecast_model(input_seq, hidden)
            forecasts[n * batch_size:n * batch_size + actual_batch_size] = prediction.squeeze(dim=-1)
    return forecasts


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str,
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/AI-HERO/energy_baseline.pt',
                        help="Model weights path")  # TODO: adapt to your model weights path
    parser.add_argument("--save_dir", type=str, help='Directory where weights and results are saved', default='.')
    parser.add_argument("--data_dir", type=str, help='Directory containing the data you want to predict',
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/data')
    args = parser.parse_args()

    save_dir = args.save_dir
    data_dir = args.data_dir

    weights_path = args.weights_path

    # load model with pretrained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO: adjust arguments according to your model
    model = SubmittedModel(input_size=1, hidden_size=48, output_size=1, num_layer=1, device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # dataloader
    test_file = os.path.join(data_dir, 'test.csv')
    valid_file = os.path.join(data_dir, 'valid.csv')
    data_file = test_file if os.path.exists(test_file) else valid_file
    testset = CustomLoadDataset(data_file, 7*24, 7*24, device=device)

    # run inference
    normalized_forecasts = forecast(model, testset, device)

    # remove normalization and convert to DataFrame
    forecasts = testset.revert_normalization(normalized_forecasts)
    df = DataFrame(forecasts.to(torch.device('cpu')).numpy())

    # save to csv
    result_path = os.path.join(save_dir, 'forecasts.csv')
    df.to_csv(result_path, header=False, index=False)

    print(f"Done! The result is saved in {result_path}")
