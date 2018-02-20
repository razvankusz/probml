####

from scipy.stats import truncnorm, norm
from ranking import _get_winners_losers, _get_player_activity, sum_0, _mask_vector_map
from data_reshape import get_trimmed_data
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm

# Using Gibbs sampling


def sample_skills(match_history, P, t, n_jobs):
    pool = ThreadPool(n_jobs)

    match_history_n = [match_history.iloc[0::n_jobs] for i in range(n_jobs)]

    def sample_skill_process(match_history_n):
        s = np.zeros(P)
        for n, match in (match_history_n.iterrows()):

            # sample one player at a time, and repeat multiple times
            winners, losers = _get_winners_losers(match)
            true_winner_count = (winners > 0).sum()
            true_loser_count = (losers > 0).sum()
            pos_winners = winners[winners > 0]
            pos_losers = losers[losers > 0]

            err_sigma = 1
            for j in range(1):
                for winner in pos_winners:

                    mean = true_winner_count * t[n] + \
                        float(true_winner_count) / (true_loser_count) * \
                        np.sum(s[pos_losers]) - \
                        np.sum(s[pos_winners[pos_winners != winner]])

                    sigma = (true_winner_count) * err_sigma

                    s[winner] = norm(
                        loc=mean, scale=sigma).rvs(size=1)[0]

                for loser in pos_losers:

                    mean = true_loser_count * t[n] + \
                        float(true_loser_count) / (true_winner_count) * \
                        np.sum(s[pos_winners]) - \
                        np.sum(s[pos_losers[pos_losers != loser]])

                    sigma = (true_loser_count) * err_sigma

                    s[loser] = norm(
                        loc=mean, scale=sigma).rvs(size=1)[0]
        return s

    results = pool.map(
        sample_skill_process,
        match_history_n
    )

    results = np.array(results)
    pool.close()

    return np.apply_along_axis(lambda x: np.nan_to_num(np.mean(x[x != 0])), arr=results, axis=0)


def gibbs_sampling_0(match_history, players, iterations=100, internal_iter=100):
    M = len(match_history)
    P = len(players)
    err_sigma = 1

    t = np.zeros(M)
    s = np.zeros(P)

    winnings, losses = _get_player_activity(
        match_history, players, ignore_zeros=True)
    true_winner_counts = sum_0(winnings)
    true_loser_counts = sum_0(losses)

    while True:
        # sample game outcome given teams:
        # p(tn | )
        mean = winnings.T.dot(s) / true_winner_counts - \
            losses.T.dot(s) / true_loser_counts
        sigma = np.eye(M) * err_sigma

        t = truncnorm(
            a=np.zeros(M),
            b=np.ones(M) * np.inf,
            loc=mean,
            scale=sigma
        ).rvs(size=(M, M)).diagonal()

        # for n, match in match_history.iterrows():
        #     winners, losers = _get_winners_losers(match)

        #     true_winner_count = (winners > 0).sum()
        #     true_loser_count = (losers > 0).sum()

        #     mean = np.sum(s[winners]) / true_winner_count - \
        #         np.sum(s[losers]) / true_loser_count

        #     t[n] = truncnorm(a=0, b=np.inf, loc=mean,
        #                      scale=err_sigma).rvs(size=1)[0]

        # sample_skills(match_history, P, t, n_jobs=8)
        for n, match in (match_history.iterrows()):

            # sample one player at a time, and repeat multiple times
            winners, losers = _get_winners_losers(match)
            true_winner_count = (winners > 0).sum()
            true_loser_count = (losers > 0).sum()
            pos_winners = winners[winners > 0]
            pos_losers = losers[losers > 0]

            err_sigma = 1
            for j in range(1):
                for winner in pos_winners:

                    mean = true_winner_count * t[n] + \
                        float(true_winner_count) / (true_loser_count) * \
                        np.sum(s[pos_losers]) - \
                        np.sum(s[pos_winners[pos_winners != winner]])

                    sigma = (true_winner_count) * err_sigma

                    s[winner] = norm(
                        loc=mean, scale=sigma).rvs(size=1)[0]

                for loser in pos_losers:

                    mean = true_loser_count * t[n] + \
                        float(true_loser_count) / (true_winner_count) * \
                        np.sum(s[pos_winners]) - \
                        np.sum(s[pos_losers[pos_losers != loser]])

                    sigma = (true_loser_count) * err_sigma

                    s[loser] = norm(
                        loc=mean, scale=sigma).rvs(size=1)[0]
        yield s


def main():
    new_player_history, new_match_history, player_mapping = get_trimmed_data()
    new_players = range(len(player_mapping) + 1)

    s = []
    for i in tqdm(range(100)):
        s.append(next(gibbs_sampling_0(new_match_history, new_players)))
    np.savetxt('./results/gibbs2.txt', np.array(s), delimiter=",")


if __name__ == '__main__':
    main()
