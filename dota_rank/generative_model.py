import scipy.stats as stats
import numpy as np
import pandas as pd
from ranking import gaussian_ep_model3
N_PLAYERS = 3000
WINNING_SIGMA = 1

SKILL_MEAN = 20
SKILL_SIGMA = 10

SKILL_SIGMA_MEAN = 10
SKILL_SIGMA_SIGMA = 2

N_MATCHES = 2000

N_ITER = 80


def get_outcome(mu1, sigma1, mu2, sigma2):
    # true if first team wins, false otherwise
    return np.random.normal(loc=mu1 - mu2, scale=WINNING_SIGMA) > 0


def generate_player_ranking():
    player_ranking = {
        'account_id': range(N_PLAYERS),
        'skill_mu':  stats.norm.rvs(loc=SKILL_MEAN, scale=SKILL_SIGMA, size=(N_PLAYERS,)),
        'skill_sigma': stats.expon.rvs(loc=SKILL_SIGMA_MEAN, scale=SKILL_SIGMA_SIGMA, size=(N_PLAYERS,)),
    }
    player_ranking = pd.DataFrame(player_ranking)
    player_ranking.skill_mu[0] = 0
    player_ranking.skill_sigma[0] = 1
    return player_ranking


def generate_matches(player_ranking):
    match_history = []
    for n in range(N_MATCHES):
        # pick preliminary team score:
        team_score = stats.norm.rvs(
            loc=SKILL_MEAN, scale=SKILL_SIGMA, size=1)[0]
        team_sigma = stats.norm.rvs(
            loc=SKILL_SIGMA_MEAN, scale=SKILL_SIGMA_SIGMA, size=1)[0]

        # pick number of players per team:
        n_rad = np.random.choice(range(3, 6))
        n_dire = np.random.choice(range(3, 6))

        # select the players whose skills match the game setup
        players = player_ranking.loc[player_ranking.skill_mu < (team_score + team_sigma)] \
            .loc[player_ranking.skill_mu > (team_score - team_sigma)] \
            .sample(n_rad + n_dire, weights=player_ranking.skill_sigma) \
            .account_id.tolist()

        radiance = players[:n_rad]
        dire = players[n_rad:n_rad + n_dire]

        # compute the real score of the teams
        rad_score = player_ranking.iloc[radiance].skill_mu.mean()
        rad_sigma = player_ranking.iloc[radiance]. \
            skill_sigma.sum() / (n_rad ** .5)

        dire_score = player_ranking.iloc[dire].skill_mu.mean()
        dire_sigma = player_ranking.iloc[dire]. \
            skill_sigma.sum() / (n_dire ** .5)

        # compute the game outcome
        outcome = get_outcome(rad_score, rad_sigma, dire_score, dire_sigma)

        # add anonymous players
        radiance = np.pad(radiance, (0, 5 - n_rad), 'constant')
        dire = np.pad(dire, (0, 5 - n_dire), 'constant')

        # prepare match_history row
        if outcome:
            winner = radiance
            loser = dire
            rad_win = True
        else:
            winner = dire
            loser = radiance
            rad_win = False

        row = {
            'match_id': n,
            'player0': winner[0],
            'player1': winner[1],
            'player2': winner[2],
            'player3': winner[3],
            'player4': winner[4],
            'player5': loser[0],
            'player6': loser[1],
            'player7': loser[2],
            'player8': loser[3],
            'player9': loser[4],
            'rad_win': rad_win,
            'rad_score': rad_score,
            'dire_score': dire_score
        }
        match_history.append(row)

    match_history = pd.DataFrame(match_history)
    return match_history


def run_simulation(fname):
    player_ranking = generate_player_ranking()
    match_history = generate_matches(player_ranking)

    player_ranking.to_csv('./gen_results/{}-ranking'.format(fname))
    match_history.to_csv('./gen_results/{}-history'.format(fname))

    gen = gaussian_ep_model3(match_history, player_ranking.account_id)
    mus, sigmas = [], []
    for _ in range(N_ITER):
        mu, sigma = next(gen)
        mus.append(mu)
        sigmas.append(sigma)

    mus, sigmas = np.array(mus), np.array(sigmas)

    np.save('./gen_results/{}-mus'.format(fname), mus)
    np.save('./gen_results/{}-sigmas'.format(fname), sigmas)


def load_simulation(fname):
    player_ranking = pd.read_csv('./gen_results/{}-ranking'.format(fname))
    match_history = pd.read_csv('./gen_results/{}-history'.format(fname))

    mus = np.load('./gen_results/{}-mus.npy'.format(fname))
    sigmas = np.load('./gen_results/{}-sigmas.npy'.format(fname))

    return player_ranking, match_history, mus, sigmas


if __name__ == '__main__':
    run_simulation('test')


# player_ranking = generate_player_ranking()
# generate_matches(player_ranking).describe()
