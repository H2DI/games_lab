from algorithms import *
from games_def import *

import numpy as np

game = Shapley()
print(np.shape(game.tab))
alg_player1 = BMAlg(3, label="InternalHedge_a 1")
alg_player2 = FTL(3, label="InternalHedge_a 2")

agent_list = [alg_player1, alg_player2]
instance = MultiAgent(game, agent_list)

instance.play_T_times(4600)
