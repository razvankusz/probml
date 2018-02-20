import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RESULTS_PATH_MU = "./results/mus{}.txt"
RESULTS_PATH_SIGMAS = "./results/ps{}.txt"
TRIMMED_MATCH_PATH = '../dota-2-matches/trimmed_match_history.csv'
TRIMMED_TEST_PATH = '../dota-2-matches/test_match_history.csv'


def team_score_mean_nozero(team, mus, sigmas):
    mu = 0
    sigma = 0

    n_members = 0
    for member in team:
        if member > 0 and member < len(mus):
            n_members += 1
            mu += mus[member]
            sigma += sigmas[member]

    if n_members == 0:
        return (0.0, 1.0)
    mu *= (1.0 / n_members)
    sigma *= (1.0 / n_members) ** .5
    return (mu, sigma)


def guess_winner(score_1, score_2):
    mu_1, sigma_1 = score_1
    mu_2, sigma_2 = score_2

    result = np.random.normal(loc=mu_1 - mu_2, scale=sigma_1 + sigma_2)

    # return 1 if np.random.randint(2) == 1 else 2
    return 1 if mu_1 > mu_2 else 2
    # return 1 if result > 0 else 2


def score(match_history, team_score, ranks):
    mus, sigmas = ranks

    correct_guesses = 0
    for n, row in match_history.iterrows():
        score_1 = team_score(
            row[['player0', 'player1', 'player2', 'player3', 'player4']], mus, sigmas)
        score_2 = team_score(
            row[['player5', 'player6', 'player7', 'player8', 'player9']], mus, sigmas)
        if guess_winner(score_1, score_2) == 1:
            correct_guesses += 1

    return float(correct_guesses) / len(match_history)


def plot_mean_std(mu, sigma):
    t = np.arange(len(mu))
    plt.plot(t, mu, label='mean skill', color='blue')
    plt.fill_between(t, mu - sigma, mu + sigma,
                     facecolor='blue', alpha=0.2)


def plot_player(mus, sigmas, account_id):
    mu = mus[account_id][:-1]
    sigma = sigmas[account_id][:-1]

    plot_mean_std(mu, sigma)
    plt.show()


def get_convergence_values(mus, sigmas):
    # returns flattened mus and sigmas

    try:
        n, m = mus.shape
        return [mu[-2] for mu in mus], [sigma[-2] for sigma in sigmas]
    except:
        return mus, sigmas


def plot_random_players(mus, sigmas, n):
    players = np.random.randint(len(mus), size=n)
    for player in players:
        plt.plot(mus[player])
    plt.show()
    for player in players:
        plt.plot(sigmas[player])
    plt.show()


def main():
    id = '_3_2'
    id = '_3_7'
    id = '_5_7'
    id = '_4hero_8'
    mus = np.loadtxt(RESULTS_PATH_MU.format(id), delimiter=",").T
    sigmas = np.loadtxt(RESULTS_PATH_SIGMAS.format(id), delimiter=",").T

    # plot_player(mus, sigmas, 8)
    match_history = pd.read_csv(TRIMMED_MATCH_PATH)

    test_match_history = pd.read_csv(TRIMMED_TEST_PATH)
    # plot_random_players(mus, sigmas, 3)

    mus, sigmas = get_convergence_values(mus, sigmas)

    print score(test_match_history, team_score_mean_nozero, (mus, sigmas))


if __name__ == '__main__':
    main()
