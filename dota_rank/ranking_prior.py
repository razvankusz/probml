'''
In this script, I am using the trueskill library, rather than implementing my own

 We will augment the trueskill algorithm my picking as priors the output of a different 
game, where the team with most gold per minute wins.
'''

import numpy as np
import pandas as pd
import scipy
import trueskill
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
import scipy.stats
from tqdm import tqdm

from data_reshape import *


player_history, match_history, player_mapping = get_trimmed_data()
