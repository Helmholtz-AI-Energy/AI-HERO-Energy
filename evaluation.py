import os
import json
from argparse import ArgumentParser
import pandas as pd
import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader

from dataset import CustomLoadDataset


baseline_mae = 9.538227766478586



def evaluate(forecasts: torch.Tensor, target: torch.Tensor, reference: float = baseline_mae) -> float:
    assert forecasts.size() == target.size(), f"Forcast shape: {forecasts.size()} not matching target: {target.size()}!"
    criterion = L1Loss()
    mae = criterion(forecasts, target)
    mase = mae.item() / reference
    return mase


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--forecast_path", type=str,
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/AI-HERO/energy_baseline.pt',
                        help="path of the saved forecasts")  # TODO: adapt to your model weights path
    parser.add_argument("--save_dir", type=str, help='Directory where results are saved', default='.')
    parser.add_argument("--data_dir", type=str, help='Directory containing the reference data',
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/data')
    args = parser.parse_args()

    save_dir = args.save_dir
    data_dir = args.data_dir

    forecast_path = args.forecast_path

    # load forecasts
    df = pd.read_csv(forecast_path, header=None)
    forecasts = torch.from_numpy(df.to_numpy())

    # load target
    test_file = os.path.join(data_dir, 'test.csv')
    valid_file = os.path.join(data_dir, 'valid.csv')
    data_file = test_file if os.path.exists(test_file) else valid_file
    testset = CustomLoadDataset(data_file, 7*24, 7*24, normalize=False)
    testloader = DataLoader(testset, len(testset), shuffle=False)

    for _, target in testloader:
        test_acc = evaluate(forecasts, target.squeeze(dim=-1))

    result_path = os.path.join(save_dir, 'score.json')
    with open(result_path, 'w') as f:
        print('Score: ', test_acc)
        json.dump(test_acc, f)

    print('Done! The result is saved in {}'.format(result_path))
