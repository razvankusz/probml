import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


def match_outcomes_to_graph(n_samples=3200):
    '''Unfortunately, I was unable to find a community structure in the 
    adjecency matrix. This would have been a good way to reduce the scale of
    the player space, by analysing stronger communities individually.
    '''

    players_df = pd.read_csv('./dota-2-matches/players.csv')
    match_outcomes_df = pd.read_csv('./dota-2-matches/match_outcomes.csv')

    accounts, account_count = np.unique(
        players_df.account_id, return_counts=True)

    n_accounts = len(accounts)
    accounts = np.random.choice(accounts, n_samples)

    matrix = np.eye(n_samples, n_samples)

    for match in match_outcomes_df.itertuples():

        # get the player ids
        players = match[2:7]

        for i, account_1 in enumerate(players):
            for account_2 in players[i + 1:]:
                try:
                    index_1 = np.where(accounts == account_1)[0][0]
                    index_2 = np.where(accounts == account_2)[0][0]

                    matrix[index_1,
                           index_2] = matrix[index_1, index_2] + 1
                    matrix[index_2,
                           index_1] = matrix[index_2, index_1] + 1
                except IndexError as ind_error:
                    pass
    return matrix


def main():
    match_outcomes_to_graph()


if __name__ == '__main__':
    main()
