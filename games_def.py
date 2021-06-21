import numpy as np


class NormalFormGame:
    """ Good for small games """

    def __init__(self, M, tab):
        """
            tab[i_1, i_2, ..., i_M] is a vector of payoffs
            tab.shape = N_1, N_2, ..., N_M,  M
        """
        self.M = M
        self.tab = tab

    def compute_exp_payoffs(self, n, plays):
        """
            plays is a tuple of probability distributions over actions (maybe a dict?)
            returns the vector of expected payoffs for actions of player n
        """
        M = self.M
        N_index = np.shape(self.tab)[:-1]
        a = self.tab
        for k in range(M):
            if k < n:
                a = np.sum(
                    [
                        a[(j,) + (slice(None),) * (M - k)] * plays[k][j]
                        for j in range(N_index[k])
                    ],
                    axis=0,
                )
            elif k >= n + 1:
                a = np.sum(
                    [
                        a[(slice(None),) + (j,) + (slice(None),) * (M - k - 1)]
                        * plays[k][j]
                        for j in range(N_index[k])
                    ],
                    axis=0,
                )
        return a[:, n]


class Chicken(NormalFormGame):
    def __init__(self):
        """
            0 is stop, 1 is go
            losses are converted to rewards and normalized in [0, 1]
        """
        M = 2
        tab = np.zeros((2, 2, 2))
        tab[0, 0] = np.array([1, 1])
        tab[0, 1] = np.array([5, 0])
        tab[1, 0] = np.array([0, 5])
        tab[1, 1] = np.array([7, 7])
        tab = tab / np.max(tab)
        tab = 1 - tab
        super().__init__(M, tab)


class RockPaperScissors(NormalFormGame):
    def __init__(self):
        """
            0 is Shi, 1 is Fu, 2 is Mi
        """
        M = 2
        tab = np.zeros((3, 3, 2))
        tab[0, 0] = np.array([1 / 2, 1 / 2])
        tab[0, 1] = np.array([0, 1])
        tab[0, 2] = np.array([1, 0])
        tab[1, 0] = np.array([1, 0])
        tab[1, 1] = np.array([1 / 2, 1 / 2])
        tab[1, 2] = np.array([0, 1])
        tab[2, 0] = np.array([0, 1])
        tab[2, 1] = np.array([1, 0])
        tab[2, 2] = np.array([1 / 2, 1 / 2])
        super().__init__(M, tab)


class Shapley(NormalFormGame):
    def __init__(self):
        """
            From Prediction, Learning, Games Ex 7.2 p.227
            Counterexample for convergence of fictitious play on zero-sum games
            Losses are converted to rewards and normalized in [0, 1]
        """
        M = 2
        tab = np.zeros((3, 3, 2))
        tab[0, 0] = np.array([0, 1])
        tab[0, 1] = np.array([1, 0])
        tab[0, 2] = np.array([1, 1])
        tab[1, 0] = np.array([1, 1])
        tab[1, 1] = np.array([0, 1])
        tab[1, 2] = np.array([1, 0])
        tab[2, 0] = np.array([1, 0])
        tab[2, 1] = np.array([1, 1])
        tab[2, 2] = np.array([0, 1])
        tab = tab / np.max(tab)
        tab = 1 - tab
        super().__init__(M, tab)


class MatchingPennies(NormalFormGame):
    def __init__(self):
        """
            From Prediction, Learning, Games Ex 7.2 p.227
            Counterexample for convergence of fictitious play on zero-sum games
            Losses are converted to rewards and normalized in [0, 1]
        """
        M = 2
        tab = np.zeros((2, 2, 2))
        tab[0, 0] = np.array([1, 0])
        tab[0, 1] = np.array([0, 1])
        tab[1, 0] = np.array([0, 1])
        tab[1, 1] = np.array([1, 0])
        super().__init__(M, tab)


class TestGame1(NormalFormGame):
    def __init__(self):
        """
            3 player game, values were randomly chosen
        """
        M = 3
        tab = np.zeros((3, 3, 3, 3))
        tab[0, 0, 0] = np.array([4, 4, 2])
        tab[0, 0, 1] = np.array([4, 0, 2])
        tab[0, 0, 2] = np.array([1, 4, 2])
        tab[0, 1, 0] = np.array([4, 0, 2])
        tab[0, 1, 1] = np.array([4, 4, 1])
        tab[0, 1, 2] = np.array([1, 5, 2])
        tab[0, 2, 0] = np.array([4, 0, 5])
        tab[0, 2, 1] = np.array([5, 1, 1])
        tab[0, 2, 2] = np.array([1, 5, 2])

        tab[1, 0, 0] = np.array([2, 4, 2])
        tab[1, 0, 1] = np.array([3, 0, 2])
        tab[1, 0, 2] = np.array([4, 4, 2])
        tab[1, 1, 0] = np.array([4, 0, 2])
        tab[1, 1, 1] = np.array([5, 4, 1])
        tab[1, 1, 2] = np.array([1, 5, 1])
        tab[1, 2, 0] = np.array([1, 0, 5])
        tab[1, 2, 1] = np.array([4, 4, 1])
        tab[1, 2, 2] = np.array([1, 5, 2])

        tab[2, 0, 0] = np.array([4, 3, 2])
        tab[2, 0, 1] = np.array([4, 0, 1])
        tab[2, 0, 2] = np.array([2, 4, 2])
        tab[2, 1, 0] = np.array([4, 0, 5])
        tab[2, 1, 1] = np.array([4, 1, 1])
        tab[2, 1, 2] = np.array([1, 5, 2])
        tab[2, 2, 0] = np.array([4, 2, 0])
        tab[2, 2, 1] = np.array([4, 4, 1])
        tab[2, 2, 2] = np.array([1, 5, 2])

        tab = tab / np.max(tab)
        super().__init__(M, tab)
