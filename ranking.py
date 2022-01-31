import pandas as pd
import numpy as np
import scipy.stats as stats
from argparse import ArgumentParser


def get_ranking(path_to_csv, task):
    df = pd.read_csv(path_to_csv)

    if task == 'health':
        # invert performance for accuracy so that smaller is better during calculations
        print('Performance metric: higher is better')
        df['Performance'] = 1 - df['Performance']
    else:
        print('Performance metric: smaller is better')

    ranked = np.apply_along_axis(stats.rankdata, 0, df.iloc[:, 1:], 'min')
    print('Ranks for each metric:\n', ranked)
    print('Weighted Scores:\n', weight_ranks(ranked))
    final_ranks = stats.rankdata(weight_ranks(ranked), 'min')

    print('Final Ranks with ties: ', final_ranks)

    # tie breaker: DevelopEnergy
    breaker_rank = ranked[:, 0]
    final_ranks = break_tie(final_ranks, breaker_rank)

    print('Final Ranks: ', final_ranks)

    print('Final Group Ordering:\n', df.set_index(final_ranks)['Group'].sort_index())


def weight_ranks(ranks):
    weighted_ranks = [(0.25 * DevelopEnergy + 0.25 * InferenceEnergy + 0.5 * Accuracy) / 3 for
                      DevelopEnergy, InferenceEnergy, Accuracy in ranks]
    return weighted_ranks


def break_tie(final_ranks, breaker_rank):
    if len(np.unique(final_ranks)) != len(final_ranks):

        # break tie with breaker_rank
        for r in range(1, len(final_ranks)):
            r_mask = (r == final_ranks)
            if sum(r_mask) <= 1:
                continue
            else:

                breakers = breaker_rank[r_mask]
                sorted_idx = np.argsort(breakers)

                final_ranks[r_mask] += sorted_idx

        return final_ranks

    else:
        return final_ranks


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--path_to_csv", type=str)
    parser.add_argument("--task", type=str, help='health / energy')
    args = parser.parse_args()

    get_ranking(args.path_to_csv, args.task)
