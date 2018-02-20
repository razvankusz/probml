import pandas as pd
import numpy as np


def get_players_history(players_df, matches_df):
    # Join the match table in
    players_history = players_df.join(matches_df, on='match_id', rsuffix='m')
    # If the player slot is lower than 10, the player is in the Radiance team
    players_history['win'] = np.logical_or(
        np.logical_and(players_history['player_slot']
                       > 100, ~players_history['radiant_win']),
        np.logical_and(players_history['player_slot']
                       < 100, players_history['radiant_win'])
    )
    return players_history


def get_match_history(players_history):
    '''
    takes the full data and aggregates player information on
    a match to match basis.
    ignore the gore
    '''

    match_history = players_history[['match_id', 'account_id', 'radiant_win']]

    def player0(x):
        return x.iloc[0]

    def rad0(x):
        return x.iloc[0]

    def player1(x):
        return x.iloc[1]

    def player2(x):
        return x.iloc[2]

    def player3(x):
        return x.iloc[3]

    def player4(x):
        return x.iloc[4]

    def player5(x):
        return x.iloc[5]

    def player6(x):
        return x.iloc[6]

    def player7(x):
        return x.iloc[7]

    def player8(x):
        return x.iloc[8]

    def player9(x):
        return x.iloc[9]

    func_list = [player0,
                 player1,
                 player2,
                 player3,
                 player4,
                 player5,
                 player6,
                 player7,
                 player8,
                 player9, ]

    match_history = match_history.groupby('match_id').agg(
        {'account_id': func_list, 'radiant_win': rad0}).reset_index()

    match_history_slice = match_history['account_id']
    match_history_slice['radiant_win'] = match_history['radiant_win']
    match_history_slice['match_id'] = match_history['match_id']
    return match_history_slice


def rotate_representation(match_history):
    # Swapping the columns so the winning team is on the left always
    # The location can be inferred from whether radiant_win is True or False
    need_to_swap = match_history[match_history.radiant_win == False]
    aux = need_to_swap[['player5', 'player6',
                        'player7', 'player8', 'player9']].copy()

    need_to_swap[['player5', 'player6', 'player7', 'player8', 'player9']] = \
        need_to_swap[['player0', 'player1', 'player2', 'player3', 'player4']]

    need_to_swap[['player0', 'player1', 'player2', 'player3', 'player4']] = aux
    match_history[match_history.radiant_win ==
                  False] = need_to_swap
    return match_history


def get_players_df(filename):
    players_df = pd.read_csv(filename)

    players_df_columns = [
        u'match_id', u'account_id', u'hero_id', u'player_slot', u'gold',
        u'gold_spent', u'gold_per_min', u'xp_per_min', u'kills', u'deaths',
        u'assists', u'denies', u'last_hits', u'stuns', u'hero_damage',
        u'hero_healing', u'tower_damage', u'level', u'leaver_status', u'xp_hero',
        u'xp_creep', u'xp_roshan', u'xp_other', u'gold_other', u'gold_death',
        u'gold_buyback', u'gold_abandon', u'gold_sell',
        u'gold_destroying_structure', u'gold_killing_heros',
        u'gold_killing_creeps', u'gold_killing_roshan',
        u'gold_killing_couriers'
    ]

    players_df = players_df[players_df_columns]
    return players_df


def trim_unfrequent_players(players_history, matches):
    new_players_history = players_history.groupby(
        'account_id').filter(lambda x: len(x) >= matches)
    new_players_history = new_players_history.groupby(
        'match_id').filter(lambda x: len(x) == 10)

    return new_players_history


def reindex_players(match_history, player_history):
    # players_history contains information about
    # only a subset of the players references in match history

    # we reindex all the players starting from 1 to len(players)
    # and replace the entries correspondingly.
    # Players not represented in player_history will be set to 0,
    # the id of the anonymous player.

    # returns a modified match_history and a player mapping: old_id->new_id

    new_match_history = match_history[match_history.match_id.isin(
        player_history.match_id)]

    players = np.unique(new_match_history.filter(
        regex='player[0-9]').as_matrix().flatten())

    player_mapping = pd.Series(data=range(len(players)), index=players)

    def _map_player(player, player_mapping):
        if player in player_mapping:
            return player_mapping[player]
        return 0

    def _map_player_mapping(player_mapping):
        return lambda player: _map_player(player, player_mapping)

    def _apply_func(func):
        return lambda x: x.apply(func)

    player_columns = ['player0', 'player1', 'player2', 'player3',
                      'player4', 'player5', 'player6', 'player7', 'player8', 'player9']

    new_match_history[player_columns] = new_match_history[player_columns]. \
        apply(lambda x: x.apply(_map_player_mapping(player_mapping)))
    new_match_history = new_match_history.reset_index()

    new_match_history['n_team1'] = new_match_history.filter(
        regex='player[0-4]').apply(_apply_func(lambda x: 1 if x > 0 else 0)).sum(axis=1)
    new_match_history['n_team2'] = new_match_history.filter(
        regex='player[5-9]').apply(_apply_func(lambda x: 1 if x > 0 else 0)).sum(axis=1)
    new_match_history = new_match_history[np.logical_and(new_match_history['n_team1'] > 0,
                                                         new_match_history['n_team2'] > 0)]
    # new_match_history.drop(['n_team1', 'n_team2'])

    new_match_history = new_match_history.reset_index()
    new_player_history = player_history.copy()
    new_player_history['account_id'] = new_player_history['account_id'].apply(
        _map_player_mapping(player_mapping))
    new_players = range(len(players))

    return new_match_history, new_player_history, new_players, player_mapping


def create_trimmed_data(n_matches=2):
    TRIMMED_MATCH_PATH = '../dota-2-matches/trimmed_match_history.csv'
    TRIMMED_PLAYER_PATH = '../dota-2-matches/trimmed_player_history.csv'
    TRIMMED_PLAYER_MAP = '../dota-2-matches/trimmed_player_mapping.csv'

    print 'creating new data'
    matches_df = pd.read_csv('../dota-2-matches/match.csv')
    players_df = get_players_df('../dota-2-matches/players.csv')

    players_history = get_players_history(players_df, matches_df)
    match_history = get_match_history(players_history)

    M = players_history.account_id.unique()

    new_player_history = trim_unfrequent_players(
        players_history, matches=n_matches)

    new_match_history, new_player_history, new_players, player_mapping = reindex_players(
        match_history, new_player_history)

    new_match_history = rotate_representation(new_match_history)

    new_match_history.to_csv(TRIMMED_MATCH_PATH)
    new_player_history.to_csv(TRIMMED_PLAYER_PATH)
    player_mapping.to_csv(TRIMMED_PLAYER_MAP)

    return new_player_history, new_match_history, player_mapping


def get_mapped_player_hero(player_id, match_index, player_mapping, player_history, match_history):
    if player_id == 0:
        return -1

    match_id = match_history.iloc[match_index].match_id
    hero_id = int(player_history.loc[player_history['match_id'] == match_id]
                  .loc[player_history['account_id'] == player_id]
                  ['hero_id'])
    return hero_id


def vec_get_mapped_player_hero(player_ids, player_mapping, player_history, match_history):
    hero_ids = np.zeros(len(player_ids), dtype=int)

    for match_index, player_id in enumerate(player_ids):
        hero_ids[match_index] = get_mapped_player_hero(
            player_id, match_index, player_mapping, player_history, match_history)

    return hero_ids


def hero_id_for_player():
    new_player_history, new_match_history, player_mapping = get_trimmed_data()
    players = ['player0', 'player1', 'player2', 'player3', 'player4',
               'player5', 'player6', 'player7', 'player8', 'player9']
    for player in players:
        new_match_history[player] = vec_get_mapped_player_hero(
            new_match_history[player], player_mapping, new_player_history, new_match_history)
    return new_player_history, new_match_history, player_mapping


def gold_winning_criteria():
    player_history, match_history, player_mapping = get_trimmed_data()

    def sum_feature(player_history, match_id, feature, team):
        total = 0
        for player in team:
            total += player_history[np.logical_and(
                player_history['account_id'] == player,
                player_history['match_id'] == match_id
            )][feature].sum()
        return total

    gold_result = []
    for n, row in match_history.iterrows():
        team1 = row[['player0', 'player1', 'player2', 'player3', 'player4']]
        team2 = row[['player5', 'player6', 'player7', 'player8', 'player9']]
        match_id = row.match_id

        team1 = team1[team1 > 0]
        team2 = team2[team2 > 0]

        gold_1 = sum_feature(player_history, match_id=match_id,
                             feature='gold_per_min', team=team1)
        gold_2 = sum_feature(player_history, match_id=match_id,
                             feature='gold_per_min', team=team1)

        gold_1 = float(gold_1) / len(team1)
        gold_2 = float(gold_2) / len(team2)

        gold_result.append(True if gold_1 > gold_2 else False)
    match_history['radiant_win'] = gold_result
    match_history = rotate_representation(match_history)
    return player_history, match_history, player_mapping


def create_trimmed_train_test_data(test=.3):
    print 'creating train-test data'
    _, match_history, _ = get_trimmed_data()

    train_match_history = match_history.sample(frac=1 - test)
    test_match_history = match_history[~match_history.index.isin(
        train_match_history.index)]

    train_match_history.reset_index(drop=True)
    test_match_history.reset_index(drop=True)

    TRIMMED_MATCH_PATH = '../dota-2-matches/trimmed_match_history.csv'
    train_match_history.to_csv(TRIMMED_MATCH_PATH)

    TRIMMED_TEST_PATH = '../dota-2-matches/test_match_history.csv'
    test_match_history.to_csv(TRIMMED_TEST_PATH)


def get_trimmed_data():
    TRIMMED_MATCH_PATH = '../dota-2-matches/trimmed_match_history.csv'
    TRIMMED_PLAYER_PATH = '../dota-2-matches/trimmed_player_history.csv'
    TRIMMED_PLAYER_MAP = '../dota-2-matches/trimmed_player_mapping.csv'

    try:
        new_match_history = pd.read_csv(TRIMMED_MATCH_PATH)
        print 'reading matches from file'
        new_player_history = pd.read_csv(TRIMMED_PLAYER_PATH)
        print 'reading players from file'
        player_mapping = pd.read_csv(TRIMMED_PLAYER_MAP)
        print 'reading player mapping from file'
    except:
        print 'creating new data'
        new_player_history, new_match_history, player_mapping = create_trimmed_data(
            n_matches=2)
    return new_player_history, new_match_history, player_mapping


def main():
    create_trimmed_data(n_matches=2)
    create_trimmed_train_test_data()


if __name__ == '__main__':
    main()
