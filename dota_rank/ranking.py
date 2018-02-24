import numpy as np
import pandas as pd
import scipy
import trueskill
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
import scipy.stats
from tqdm import tqdm

from data_reshape import *
#
# Removing warning messages about chained assignment
pd.options.mode.chained_assignment = None


def count_nans(xs):
    return np.sum([1 if np.isnan(x) else 0 for x in xs])


def predict_team_score_mean(ranks, team):
    # ignore null (anonymous) players
    team = team[team != 0]

    # ignore players not in the ranks
    scores = np.array([ranks.get(member, 0) for member in team])
    scores = scores[scores > 0]
    return np.mean(scores)


def predict_rad_wins_random(rad_score, dire_score):
    return np.random.choice([True, False])


def predict_rad_wins_gt(rad_score, dire_score):
    return rad_score > dire_score


def score(ranks, match_history,
          predict_team_score=predict_team_score_mean,
          predict_rad_wins=predict_rad_wins_gt):
    '''
    returns prediction accuracy score.

    ranks df(account_id,rank score)
    match_history : df(player0, ..., player9, radiant_win, match_id)

    predict_team_score : [score] -> score
    predict_rad_wins : [score, score] -> bool
    '''
    correct = 0

    for n, row in match_history.iterrows():
        radiance = row[['player0', 'player1', 'player2', 'player3', 'player4']]
        dire = row[['player5', 'player6', 'player7', 'player8', 'player9']]

        radiance_score = predict_team_score(ranks, radiance)
        dire_score = predict_team_score(ranks, dire)

        if predict_rad_wins(radiance_score, dire_score) == row['radiant_win']:
            correct = correct + 1

    return float(correct) / len(match_history)


def train_test_split(df, frac):
    # randomly sample frac of df and return the partition
    ns = range(len(df))
    ns = np.random.choice(ns, int(len(ns) * frac))

    return (df[~df.index.isin(ns)],
            df[df.index.isin(ns)])


def baseline_rank(players_history):
    matches_played = players_history.groupby('account_id') \
        .agg({'match_id': 'count'}) \
        .reset_index()[['account_id', 'match_id']]
    matches_won = players_history.groupby('account_id') \
        .agg({'win': 'sum'}) \
        .reset_index()['win']

    return pd.DataFrame({
        'account_id': matches_played['account_id'],
        'total_played': matches_played['match_id'],
        'total_won': matches_won,
        'baseline_rank': matches_won / matches_played['match_id'],
    })


def trueskill_rank(match_history, players, mu=5.00, sigma=.5):
    '''
    https://www.kaggle.com/devinanzelmo/dota-2-skill-rating-with-trueskill
    '''

    ts = trueskill.TrueSkill(draw_probability=0, mu=mu, sigma=sigma)

    # need to create a dictionary for all players containting the ratings
    rating_dict = {account_id: ts.create_rating()
                   for account_id
                   in players}

    rating_dict[0] = ts.create_rating(mu=0, sigma=10)

    for n, row in match_history.iterrows():
        radiance = row[['player0', 'player1', 'player2', 'player3', 'player4']]
        dire = row[['player5', 'player6', 'player7', 'player8', 'player9']]

        # ignore the anonymous players
        radiance = radiance[radiance > 0]
        dire = dire[dire > 0]

        if radiance.empty or dire.empty:
            continue

        rad_dict = {
            account_id: rating_dict[account_id]
            for account_id in radiance
        }
        dire_dict = {
            account_id: rating_dict[account_id]
            for account_id in dire
        }

        new_rad, new_dire = ts.rate([rad_dict, dire_dict], ranks=[1, 0])

        for account_id in new_rad.keys():
            rating_dict[account_id] = new_rad[account_id]
        for account_id in new_dire.keys():
            rating_dict[account_id] = new_dire[account_id]

    return [rating.mu for rating in rating_dict.values()], [rating.sigma for rating in rating_dict.values()]


MU = 10
SIGMA = 2


def team_score_trueskill(ranks, team):
    team = team[team != 0]
    scores = [ranks.get(member, None) for member in team]
    # ignore players not in the ranks

    scores = np.array(scores)
    scores_mask = [False if score is None else True for score in scores]
    scores = scores[scores_mask]

    if len(scores) is 0:
        return trueskill.TrueSkill(mu=MU, sigma=SIGMA).create_rating()

    mean_mu = np.mean([score.mu for score in scores])

    mean_var = np.sum([score.sigma ** 2 for sigma in scores]
                      ) / (len(scores) ** 2)
    mean_sigma = np.sqrt(mean_var)

    return trueskill.TrueSkill().create_rating(mean_mu, mean_sigma)


def rad_win_trueskill(radiant, dire):
    diff_mu = radiant.mu - dire.mu
    diff_sigma = np.sqrt(radiant.sigma ** 2 + dire.sigma ** 2)

    result = np.random.normal(loc=diff_mu, scale=diff_sigma)
    return result > 0


def gaussian_ep(G, M):
    def Psi(x): return scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)

    def Lambda(x): return Psi(x) * (Psi(x) + x)
    N = len(G)

    mu_s, p_s = np.empty(M), np.empty(M)
    mu_gs, p_gs = np.zeros((N, 2)), np.zeros((N, 2))
    mu_sg, p_sg = np.empty((N, 2)), np.empty((N, 2))

    while True:
        # 1. Compute marginal skills
        # Let skills be N(miu_s, 1/p_s)
        p_s = np.ones(M) * 1 / 0.5
        mu_s = np.zeros(M)
        for j, (winner, loser) in enumerate(G):
            p_s[winner] += p_gs[j, 0]
            p_s[loser] += p_gs[j, 1]
            mu_s[winner] += mu_gs[j, 0] * p_gs[j, 0]
            mu_s[loser] += mu_gs[j, 1] * p_gs[j, 1]
        mu_s = mu_s / p_s

        # 2. Compute skill -> game messages
        # winner's skill -> game: N(miu_sg[,0], 1/p_sg[,0])
        # loser's skill -> game: N(miu_sg[,1], 1/p_sg[,1])
        p_sg = p_s[G] - p_gs
        mu_sg = (p_s[G] * mu_s[G] - p_gs * mu_gs) / p_sg

        # 3. Compute game -> performance messages
        v_gt = 1 + np.sum(1 / p_sg, 1)
        sigma_gt = np.sqrt(v_gt)
        mu_gt = mu_sg[:, 0] - mu_sg[:, 1]

        # 4. Approximate the marginal on performance differences
        mu_t = mu_gt + sigma_gt * Psi(mu_gt / sigma_gt)
        p_t = 1 / v_gt / (1 - Lambda(mu_gt / sigma_gt))

        # 5. Compute performance -> game messages
        p_tg = p_t - 1 / v_gt
        mu_tg = (mu_t * p_t - mu_gt / v_gt) / p_tg

        # 6. Compute game -> skills messages
        # game -> winner's skill: N(miu_gs[,0], 1/p_gs[,0])
        # game -> loser's skill: N(miu_gs[,1], 1/p_gs[,1])
        p_gs[:, 0] = 1 / (1 + 1 / p_tg + 1 / p_sg[:, 1])  # winners
        p_gs[:, 1] = 1 / (1 + 1 / p_tg + 1 / p_sg[:, 0])  # losers
        mu_gs[:, 0] = mu_sg[:, 1] + mu_tg
        mu_gs[:, 1] = mu_sg[:, 0] - mu_tg

        yield (mu_s, np.sqrt(1 / p_s))


def _get_winners_losers(row):
    winners = row[['player0', 'player1', 'player2', 'player3', 'player4']]
    losers = row[['player5', 'player6', 'player7', 'player8', 'player9']]

    return np.array(winners.tolist()), np.array(losers.tolist())


def _update_priors_1(match_history, mu_w, p_s, m_gs, n_jobs=4):

    pool = ThreadPool(n_jobs)

    match_history_n = [match_history.iloc[0::n_jobs] for i in range(n_jobs)]
    mu_w = [mu_w.copy() for _ in range(n_jobs)]
    p_w = [p_w.copy() for _ in range(n_jobs)]

    results = pool.starmap(
        _message_pass_1_thread,
        [(m_h, (mu_w0, p_w0), m_gw)
         for m_h in match_history_n
         for mu_w0 in mu_w
         for p_w0 in p_w]
    )
    pool.join()
    pool.close()
    print(results)


def _message_pass_1_thread(match_history, q_w, m_gw):
    mu_w, p_w = q_w
    mu_gw, p_gw = m_gw

    for _, row in match_history.iterrows():
        winners, losers = _get_winners_losers(row)
        n = row.match_id

        p_w[winners] += p_gw[n, 0]
        p_w[losers] += p_gw[n, 1]

        mu_w[winners] += mu_gw[n, 0] * p_gw[n, 0]
        mu_w[losers] += mu_gw[n, 1] * p_gw[n, 1]
    return mu_w, p_w


def gaussian_ep_model1(match_history, M):

    def Psi(x):
        return np.nan_to_num(scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x))

    def Lambda(x):
        return Psi(x) * (Psi(x) + x)

    N = len(match_history)
    P = len(M)

    mu_w = pd.Series(data=np.zeros(P), index=M)
    p_w = pd.Series(data=np.ones(P), index=M)

    winnings, losses = _get_player_activity(match_history, M)

    mu_gw, p_gw = np.zeros((N, 2)), np.zeros((N, 2))
    mu_wg, p_wg = np.zeros((10, N)), np.zeros((10, N))

    while True:
        # 1 compute marginal skills N(mu_w, 1/p_w)

        mu_w = np.zeros(P)
        p_w = np.ones(P)

        # p_w += winnings.dot(p_gw[:, 0])
        # p_w += losses.dot(p_gw[:, 1])

        # mu_w += winnings.dot(mu_gw[:, 0] * p_gw[:, 0])
        # mu_w += losses.dot(mu_gw[:, 1] * p_gw[:, 1])

        # mu_w = mu_w / p_w

        for n, row in match_history.iterrows():
            winners, losers = _get_winners_losers(row)

            p_w[winners] += p_gw[n, 0]
            p_w[losers] += p_gw[n, 1]

            mu_w[winners] += mu_gw[n, 0] * p_gw[n, 0]
            mu_w[losers] += mu_gw[n, 1] * p_gw[n, 1]
        # we had actually computed the natural mean
        mu_w = mu_w / p_w

        # 2. Compute skill -> game messages
        for n, row in match_history.iterrows():
            winners, losers = _get_winners_losers(row)

            # subtract precisions, subtract natural means.
            p_wg[0:5, n] = p_w[winners] - p_gw[n, 0]
            mu_wg[0:5, n] = (
                p_w[winners] * mu_w[winners] -
                p_gw[n, 0] * mu_gw[n, 0]
            ) / p_wg[0:5, n]

            p_wg[5:10, n] = p_w[losers] - p_gw[n, 1]
            mu_wg[5:10, n] = (
                p_w[losers] * mu_w[losers] -
                p_gw[n, 1] * mu_gw[n, 1]
            ) / p_wg[5:10, n]

        assert (p_wg < 0).sum() == 0

        # 3. Compute game -> performance messages.
        # Here is the main bit where we can add features / deal with
        # anonymous players.
        # v_gt, mu_gt = np.zeros((N,)), np.zeros((N,))

        v_gt = 1 + np.sum(1 / p_wg[0:5, :], 0) * 0.001 + \
            np.sum(1 / p_wg[5:10, :], 0) * 0.001

        assert (v_gt < 0).sum() == 0
        assert count_nans(v_gt) == 0

        mu_gt = np.sum(mu_wg[0:5, :], 0) - np.sum(mu_wg[5:10, :], 0)
        print(mu_wg[0:5], 602)
        print(mu_wg[5:10], 602)

        mu_gt = mu_gt * 0.1

        assert count_nans(mu_gt) == 0

        sigma_gt = np.sqrt(v_gt)

        # 4. Approximate the marginal on performance differences
        # Nothing changed here
        mu_t = mu_gt + sigma_gt * Psi(mu_gt / sigma_gt)
        p_t = 1 / v_gt / (1 - Lambda(mu_gt / sigma_gt))

        assert count_nans(mu_t) == 0
        assert (p_t < 0).sum() == 0
        assert count_nans(p_t) == 0

        # 5. Compute performance -> game messages
        # Nothing changed here
        p_tg = p_t - 1 / v_gt

        assert count_nans(mu_t) == 0
        assert count_nans(p_t) == 0
        assert count_nans(mu_gt) == 0
        assert count_nans(v_gt) == 0
        assert count_nans(p_tg) == 0

        mu_tg = (mu_t * p_t - mu_gt / v_gt) / p_tg

        assert count_nans(p_tg) == 0
        assert count_nans(mu_tg) == 0

        # 6. Compute game -> skills messages
        p_gw[:, 0] = 1 / (1 + 1 / p_tg + np.sum(1 / p_wg[0:5, :], 0))
        p_gw[:, 1] = 1 / (1 + 1 / p_tg + np.sum(1 / p_wg[5:10, :], 0))

        mu_gw[:, 0] = np.sum(mu_wg[5:10, :], 0) + mu_tg
        mu_gw[:, 1] = np.sum(mu_wg[0:5, :], 0) - mu_tg

        yield (mu_w, np.sqrt(1 / p_w))

###############################################################################
# MODEL 2 -- WHICH TRAINS FASTER BUT ON SMALLER DATASETS
###############################################################################


def _get_player_activity(match_history, players, ignore_zeros=False):
    # returns winnings and losses matrices
    # this only does a good job if we keep match_history small enough
    # in this case, we have around 10000 matches and 130000 players

    # could be made to ignore player number 0, but we'll try this first
    M, P = len(match_history), len(players)

    winnings = lil_matrix((P, M))
    losses = lil_matrix((P, M))

    for n, row in (match_history.iterrows()):
        winners, losers = _get_winners_losers(row)
        for winner in winners:
            if ignore_zeros and winner == 0:
                pass
            else:
                winnings[winner, n] += 1
        for loser in losers:
            if ignore_zeros and loser == 0:
                pass
            else:
                losses[loser, n] += 1

    return csr_matrix(winnings), csr_matrix(losses)


def _mask_vector_map(vector, sparse):
    return sparse.transpose().multiply(vector).transpose()


def gaussian_ep_model2(match_history, players):

    def Psi(x):
        return scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)

    def Lambda(x):
        return Psi(x) * (Psi(x) + x)

    N = len(match_history)
    P = len(players)

    winnings, losses = _get_player_activity(match_history, players)

    mu_w = pd.Series(data=np.zeros(P))
    p_w = pd.Series(data=np.ones(P))

    mu_gw, p_gw = np.zeros((N, 2)), np.zeros((N, 2))
    mu_wg_win, p_wg_win = csr_matrix((P, N)), csr_matrix((P, N))
    mu_wg_loss, p_wg_loss = csr_matrix((P, N)), csr_matrix((P, N))

    v_gt, mu_gt = np.zeros((N,)), np.zeros((N,))

    while True:
        # winnings are the player/match win indication matrix
        p_w += winnings.dot(p_gw[:, 0])
        p_w += losses.dot(p_gw[:, 1])

        mu_w += winnings.dot(mu_gw[:, 0] * p_gw[:, 0])
        mu_w += losses.dot(mu_gw[:, 1] * p_gw[:, 1])

        mu_w = mu_w / p_w

        # 2. Compute skill -> game messages

        # precision of winner message = precisions of winners - game->skill message
        # natural means are subtracted (hence the complication)

        # we write A.multiply(B.power(-1)) for A / B
        # numpy/scipy ignore zeros in this operation.

        p_wg_win = winnings.multiply(
            _mask_vector_map(p_w, winnings) - p_gw[:, 0])

        mu_wg_win = winnings.multiply(_mask_vector_map(p_w * mu_w, winnings) -
                                      p_gw[:, 0] * mu_gw[:, 0])
        mu_wg_win = mu_wg_win.multiply(p_wg_win.power(-1))

        p_wg_loss = losses.multiply(_mask_vector_map(p_w, losses) - p_gw[:, 1])

        mu_wg_loss = losses.multiply(_mask_vector_map(p_w * mu_w, losses) -
                                     p_gw[:, 1] * mu_gw[:, 1])
        mu_wg_loss = mu_wg_loss.multiply(p_wg_loss.power(-1))

        # 3. Compute game -> performance messages.
        # Here is the main bit where we can add features / deal with
        # anonymous players.

        v_gt = 1 + np.array(p_wg_win.power(-1).sum(axis=0)) + \
            np.array(p_wg_loss.power(-1).sum(axis=0))

        mu_gt = np.array(mu_wg_win.sum(axis=0)) - \
            np.array(mu_wg_loss.sum(axis=0))
        sigma_gt = np.sqrt(v_gt)

        # 4. Approximate the marginal on performance differences
        # Nothing changed here
        mu_t = mu_gt + sigma_gt * Psi(mu_gt / sigma_gt)
        p_t = 1 / v_gt / (1 - Lambda(mu_gt / sigma_gt))

        # 5. Compute performance -> game messages
        # Nothing changed here
        p_tg = p_t - 1 / v_gt
        mu_tg = (mu_t * p_t - mu_gt / v_gt) / p_tg

        # 6. Compute game -> skills messages

        p_gw[:, 0] = 1 / \
            (1 + 1 / p_tg + np.array(p_wg_win.power(-1).sum(axis=0)))
        p_gw[:, 1] = 1 / \
            (1 + 1 / p_tg + np.array(p_wg_win.power(-1).sum(axis=0)))

        mu_gw[:, 0] = np.array(mu_wg_loss.sum(axis=0)) + mu_tg
        mu_gw[:, 1] = np.array(mu_wg_win.sum(axis=0)) - mu_tg

        yield (mu_w, np.sqrt(1 / p_w))


def power_ignore_null(xs, power):
    return np.array([0 if x == 0 else x ** power for x in xs])


def c_power_ignore_null(power):
    return (lambda x: power_ignore_null(x, power))
# have to use this because of different behaviour in np


def sum_of_inverse(sparse):
    return np.squeeze(np.asarray(sparse.power(-1).sum(axis=0)))


def sum_0(sparse):
    # return sum on the 0 axis
    return np.squeeze(np.asarray(sparse.sum(axis=0)))


def gaussian_ep_model3(match_history, players, priors=None):
    # In this model, we compute the mean of the non-anonymous players

    def Psi(x):
        result = scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)
        result = np.nan_to_num(result)
        return result

    def Lambda(x):
        return Psi(x) * (Psi(x) + x)

    N = len(match_history)
    P = len(players)

    winnings, losses = _get_player_activity(
        match_history, players, ignore_zeros=True)

    winning_team_counts = sum_0(winnings)
    losing_team_counts = sum_0(losses)

    mu_s = pd.Series(data=np.ones(P) * 4)
    p_s = pd.Series(data=np.ones(P) * .5)

    mu_gs, p_gs = np.zeros((N, 2)), np.zeros((N, 2))
    mu_sg_win, p_sg_win = csr_matrix((P, N)), csr_matrix((P, N))
    mu_sg_loss, p_sg_loss = csr_matrix((P, N)), csr_matrix((P, N))

    v_gt, mu_gt = np.zeros((N,)), np.zeros((N,))

    mu_s00 = scipy.stats.norm(loc=4, scale=10).rvs(size=P)

    while True:
        # winnings are the player/match win indication matrix

        if priors:
            mu_prior, p_prior = priors
            mu_s = mu_prior
            p_s = p_prior
        else:
            mu_s = np.ones(P) * 2
            p_s = np.ones(P) * .5

        p_s += winnings.dot(p_gs[:, 0])
        assert(count_nans(p_s) == 0)
        p_s += losses.dot(p_gs[:, 1])
        assert(count_nans(p_s) == 0)

        assert(count_nans(mu_s) == 0)
        mu_s += winnings.dot(mu_gs[:, 0] * p_gs[:, 0])
        assert(count_nans(mu_s) == 0)
        mu_s += losses.dot(mu_gs[:, 1] * p_gs[:, 1])
        assert(count_nans(mu_s) == 0)

        mu_s = mu_s / p_s
        assert(count_nans(mu_s) == 0)

        # 2. Compute skill -> game messages

        # precision of winner message = precisions of winners - game->skill message
        # natural means are subtracted (hence the complication)

        p_sg_win = winnings.multiply(
            _mask_vector_map(p_s, winnings) - p_gs[:, 0])
        p_sg_win.eliminate_zeros()

        assert(count_nans(sum_0(p_sg_win)) == 0)

        mu_sg_win = winnings.multiply(_mask_vector_map(p_s * mu_s, winnings) -
                                      p_gs[:, 0] * mu_gs[:, 0])
        mu_sg_win = mu_sg_win.multiply(p_sg_win.power(-1))
        assert(count_nans(sum_0(mu_sg_win)) == 0)

        p_sg_loss = losses.multiply(_mask_vector_map(p_s, losses) - p_gs[:, 1])
        p_sg_loss.eliminate_zeros()

        # p_sg_loss = np.absolute(p_sg_loss)
        assert(count_nans(sum_0(p_sg_loss)) == 0)

        mu_sg_loss = losses.multiply(_mask_vector_map(p_s * mu_s, losses) -
                                     p_gs[:, 1] * mu_gs[:, 1])
        assert(count_nans(sum_0(mu_sg_loss)) == 0)
        mu_sg_loss = mu_sg_loss.multiply(p_sg_loss.power(-1))
        assert(count_nans(sum_0(mu_sg_loss)) == 0)

        # 3. Compute game -> performance messages.
        # Here is the main bit where we can add features / deal with
        # anonymous players.

        v_gt = 1 + sum_of_inverse(p_sg_win) * np.power(winning_team_counts, -2) + \
            sum_of_inverse(p_sg_loss) * np.power(losing_team_counts, -2)
        assert((sum_of_inverse(p_sg_loss) < 0).sum() == 0)
        assert((sum_of_inverse(p_sg_win) < 0).sum() == 0)

        v_gt = 1 + sum_of_inverse(p_sg_win) + sum_of_inverse(p_sg_loss)
        # assert((v_gt < 0).sum() == 0)

        assert(count_nans(v_gt) == 0)

        mu_gt = sum_0(mu_sg_win) * np.power(winning_team_counts, -1) - \
            sum_0(mu_sg_loss) * np.power(losing_team_counts, -1)
        # mu_gt = sum_0(mu_sg_win) - sum_0(mu_sg_loss)
        assert(count_nans(mu_gt) == 0)
        sigma_gt = np.sqrt(v_gt)

        assert(count_nans(sigma_gt) == 0)

        # 4. Approximate the marginal on performance differences
        # Nothing changed here

        assert (count_nans(Psi(mu_gt / sigma_gt)) == 0)

        mu_t = mu_gt + sigma_gt * Psi(mu_gt / sigma_gt)
        assert(count_nans(mu_t) == 0)
        p_t = 1 / v_gt / (1 - Lambda(mu_gt / sigma_gt))
        assert(count_nans(p_t) == 0)
        # 5. Compute performance -> game messages
        # Nothing changed here
        p_tg = p_t - 1 / v_gt
        assert(count_nans(p_tg) == 0)

        mu_tg = (mu_t * p_t - mu_gt / v_gt) / p_tg

        # assert(count_nans(mu_tg) == 0)
        # 6. Compute game -> skills messages

        p_gs[:, 0] = 1 / \
            (1 + 1 / p_tg + sum_of_inverse(p_sg_loss))
        #  * np.power(losing_team_counts, -2))
        # assert(count_nans(p_gs[:, 0]) == 0)
        p_gs[:, 1] = 1 / \
            (1 + 1 / p_tg + sum_of_inverse(p_sg_win))
        #  * np.power(winning_team_counts, -2))
        # assert(count_nans(p_gs[:, 1]) == 0)
        mu_gs[:, 0] = sum_0(mu_sg_loss) / losing_team_counts + mu_tg
        # assert(count_nans(mu_gs[:, 0]) == 0)
        mu_gs[:, 1] = sum_0(mu_sg_win) / winning_team_counts - mu_tg
        # assert(count_nans(mu_gs[:, 1]) == 0)
        yield (mu_s, np.sqrt(1 / p_s))


def get_delta_heros(mu_hero, sigma_hero, match_history):

    # first get convergence values:
    mu_hero = mu_hero.T
    sigma_hero = sigma_hero.T
    mu_hero = np.array([mu[-2] for mu in mu_hero])
    sigma_hero = np.array([sigma[-2] for sigma in sigma_hero])

    mu_delta_heros = np.zeros(len(match_history))
    v_delta_heros = np.zeros(len(match_history))
    v_hero = np.power(sigma_hero, 2)
    for n, match in match_history.iterrows():
        winners, losers = _get_winners_losers(match)

        mu_delta_heros[n] = np.sum(
            mu_hero[winners[winners >= 0]]) - np.sum(mu_hero[losers[losers >= 0]])
        v_delta_heros[n] = np.sum(v_hero[winners[winners >= 0]]) + \
            np.sum(v_hero[losers[losers >= 0]])

    return mu_delta_heros, v_delta_heros
# The one where I add information about the hero


def gaussian_ep_model4(match_history, players, mu_delta_heros, v_delta_heros, priors=None):
    # In this model, we compute the mean of the non-anonymous players

    def Psi(x):
        result = scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)
        result = np.nan_to_num(result)
        return result

    def Lambda(x):
        return Psi(x) * (Psi(x) + x)

    N = len(match_history)
    P = len(players)

    winnings, losses = _get_player_activity(
        match_history, players, ignore_zeros=True)

    winning_team_counts = sum_0(winnings)
    losing_team_counts = sum_0(losses)

    mu_s = pd.Series(data=np.ones(P) * 25)
    p_s = pd.Series(data=np.ones(P) * 8)

    mu_gs, p_gs = np.zeros((N, 2)), np.zeros((N, 2))
    mu_sg_win, p_sg_win = csr_matrix((P, N)), csr_matrix((P, N))
    mu_sg_loss, p_sg_loss = csr_matrix((P, N)), csr_matrix((P, N))

    v_gt, mu_gt = np.zeros((N,)), np.zeros((N,))

    mu_s00 = scipy.stats.norm(loc=4, scale=10).rvs(size=P)

    while True:
        # winnings are the player/match win indication matrix

        mu_s = np.ones(P) * 2
        p_s = np.ones(P) * .5

        p_s += winnings.dot(p_gs[:, 0])
        assert(count_nans(p_s) == 0)
        p_s += losses.dot(p_gs[:, 1])
        assert(count_nans(p_s) == 0)

        assert(count_nans(mu_s) == 0)
        mu_s += winnings.dot(mu_gs[:, 0] * p_gs[:, 0])
        assert(count_nans(mu_s) == 0)
        mu_s += losses.dot(mu_gs[:, 1] * p_gs[:, 1])
        assert(count_nans(mu_s) == 0)

        mu_s = mu_s / p_s
        assert(count_nans(mu_s) == 0)

        # 2. Compute skill -> game messages

        # precision of winner message = precisions of winners - game->skill message
        # natural means are subtracted (hence the complication)

        p_sg_win = winnings.multiply(
            _mask_vector_map(p_s, winnings) - p_gs[:, 0])
        p_sg_win.eliminate_zeros()

        assert(count_nans(sum_0(p_sg_win)) == 0)

        mu_sg_win = winnings.multiply(_mask_vector_map(p_s * mu_s, winnings) -
                                      p_gs[:, 0] * mu_gs[:, 0])
        mu_sg_win = mu_sg_win.multiply(p_sg_win.power(-1))
        assert(count_nans(sum_0(mu_sg_win)) == 0)

        p_sg_loss = losses.multiply(_mask_vector_map(p_s, losses) - p_gs[:, 1])
        p_sg_loss.eliminate_zeros()

        # p_sg_loss = np.absolute(p_sg_loss)
        assert(count_nans(sum_0(p_sg_loss)) == 0)

        mu_sg_loss = losses.multiply(_mask_vector_map(p_s * mu_s, losses) -
                                     p_gs[:, 1] * mu_gs[:, 1])
        assert(count_nans(sum_0(mu_sg_loss)) == 0)
        mu_sg_loss = mu_sg_loss.multiply(p_sg_loss.power(-1))
        assert(count_nans(sum_0(mu_sg_loss)) == 0)

        # 3. Compute game -> performance messages.
        # Here is the main bit where we can add features / deal with
        # anonymous players.

        v_gt = 1 + sum_of_inverse(p_sg_win) * np.power(winning_team_counts, -2) + \
            sum_of_inverse(p_sg_loss) * np.power(losing_team_counts, 2) + \
            v_delta_heros
        assert((sum_of_inverse(p_sg_loss) < 0).sum() == 0)
        assert((sum_of_inverse(p_sg_win) < 0).sum() == 0)

        # v_gt = 1 + sum_of_inverse(p_sg_win) + sum_of_inverse(p_sg_loss)
        # assert((v_gt < 0).sum() == 0)

        assert(count_nans(v_gt) == 0)

        mu_gt = sum_0(mu_sg_win) * np.power(winning_team_counts, -1) - \
            sum_0(mu_sg_loss) * np.power(losing_team_counts, -1) + \
            mu_delta_heros
        # mu_gt = sum_0(mu_sg_win) - sum_0(mu_sg_loss)
        assert(count_nans(mu_gt) == 0)
        sigma_gt = np.sqrt(v_gt)

        assert(count_nans(sigma_gt) == 0)

        # 4. Approximate the marginal on performance differences
        # Nothing changed here

        assert (count_nans(Psi(mu_gt / sigma_gt)) == 0)

        mu_t = mu_gt + sigma_gt * Psi(mu_gt / sigma_gt)
        assert(count_nans(mu_t) == 0)
        p_t = 1 / v_gt / (1 - Lambda(mu_gt / sigma_gt))
        assert(count_nans(p_t) == 0)
        # 5. Compute performance -> game messages
        # Nothing changed here
        p_tg = p_t - 1 / v_gt
        assert(count_nans(p_tg) == 0)

        mu_tg = (mu_t * p_t - mu_gt / v_gt) / p_tg

        # assert(count_nans(mu_tg) == 0)
        # 6. Compute game -> skills messages

        p_gs[:, 0] = 1 / \
            ((1 + 1 / p_tg + sum_of_inverse(p_sg_loss)
                * np.power(losing_team_counts, -2)) + v_delta_heros)
        # assert(count_nans(p_gs[:, 0]) == 0)
        p_gs[:, 1] = 1 / \
            ((1 + 1 / p_tg + sum_of_inverse(p_sg_win)
                * np.power(winning_team_counts, -2)) + v_delta_heros)

        # assert(count_nans(p_gs[:, 1]) == 0)
        mu_gs[:, 0] = sum_0(mu_sg_loss) / \
            losing_team_counts + mu_tg + mu_delta_heros
        # assert(count_nans(mu_gs[:, 0]) == 0)
        mu_gs[:, 1] = sum_0(mu_sg_win) / \
            winning_team_counts - mu_tg - mu_delta_heros
        # assert(count_nans(mu_gs[:, 1]) == 0)
        yield (mu_s, np.sqrt(1 / p_s))


def save_model(mus, sigmas, id):
    RESULTS_PATH_MU = "./results/mus{}.txt".format(id)
    RESULTS_PATH_SIGMAS = "./results/ps{}.txt".format(id)

    np.savetxt(RESULTS_PATH_MU, mus, delimiter=",")
    np.savetxt(RESULTS_PATH_SIGMAS, sigmas, delimiter=",")


def run_model(gaussian_ep_model, match_history, players, priors=None, id='0', n_iter=50):
    gen = gaussian_ep_model(match_history, players)
    mus, sigmas = [], []

    for i in tqdm(range(n_iter)):
        mu, sigma = next(gen)
        mus.append(mu)
        sigmas.append(sigma)

    mus = np.array(mus)
    sigmas = np.array(sigmas)

    return mus, sigmas


def run_model_4_hero(index=0):
    new_player_history, new_match_history, player_mapping = hero_id_for_player()
    new_players = range(len(new_player_history.hero_id.unique()) + 2)

    mus, sigmas = run_model(
        gaussian_ep_model3, new_match_history, new_players)

    mu_delta_heros, v_delta_heros = get_delta_heros(
        mus, sigmas, new_match_history)

    new_player_history, new_match_history, player_mapping = get_trimmed_data()
    new_players = range(len(player_mapping) + 1)

    mus, sigmas = run_model(
        lambda x, y: gaussian_ep_model4(
            x, y, mu_delta_heros=mu_delta_heros, v_delta_heros=v_delta_heros),
        new_match_history, new_players)

    save_model(mus, sigmas, '_4hero_{}'.format(index))


def run_model_5(index=0):
    new_player_history, new_match_history, player_mapping = gold_winning_criteria()
    new_players = range(len(player_mapping) + 1)

    mus_1, sigmas_1 = run_model(
        gaussian_ep_model3, new_match_history, new_players)

    new_player_history, new_match_history, player_mapping = get_trimmed_data()
    new_players = range(len(player_mapping) + 1)

    mu_prior = np.nan_to_num(mus_1[-2, :] / np.max(mus_1[-2, :]) * 8)
    sigma_prior = np.nan_to_num(sigmas_1[-2, :] / np.max(sigmas_1[-2, :]))

    p_prior = np.power(sigma_prior, -2)

    mus_2, sigmas_2 = run_model(
        gaussian_ep_model3, new_match_history, new_players, priors=(mu_prior, p_prior))

    save_model(mus_1 + mus_2, np.power(np.power(sigmas_1, 2) +
                                       np.power(sigmas_2, 2), -2), '_5_{}'.format(index))


def run_model_3(index=0):
    new_player_history, new_match_history, player_mapping = get_trimmed_data()
    new_players = range(len(player_mapping) + 1)

    mus, sigmas = run_model(
        gaussian_ep_model3, new_match_history, new_players)

    save_model(mus, sigmas, id='_3_{}'.format(index))


def main():
    print('Preparing Data')
    run_model_3(index=8)


if __name__ == '__main__':
    main()
