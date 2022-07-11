import numpy as np


class NormalFormGame:
    """Good for small games"""

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
        plays[k][i]
        """
        M = self.M
        N_index = np.shape(self.tab)[:-1]
        a = self.tab
        for k in range(M):
            assert np.isclose(np.sum(plays[k]), 1)
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

    def check_eps_BR(self, n, plays, play_to_check, eps):
        mixture_payoffs = self.compute_exp_payoffs(n, plays)
        return np.dot(play_to_check, mixture_payoffs) >= np.max(mixture_payoffs) - eps

    def check_eps_NE(self, plays, eps):
        for n in range(self.M):
            if not (self.check_eps_BR(n, plays, plays[n], eps)):
                return False
        return True

    def approx_NE_quality(self, plays):
        eps = 0
        for n in range(self.M):
            mixture_payoffs = self.compute_exp_payoffs(n, plays)
            eps = max(eps, np.max(mixture_payoffs) - np.dot(plays[n], mixture_payoffs))
        return eps


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


class RandomGame(NormalFormGame):
    def __init__(self, shape, zerosum=False, integer=False):
        self.M = shape[-1]
        if integer:
            tab = np.random.randint(0, high=2, size=shape)
        else:
            tab = np.random.rand(*shape)
        if zerosum:
            assert self.M == 2
            tab[:, :, 0] = 1 - tab[:, :, 1]
        self.tab = tab


class HardGame(NormalFormGame):
    def hard_mat(self, n):
        if n == 1:
            r = np.array(
                [
                    [0, 0],
                    [1, -1],
                ]
            )
            return r
        else:
            a = self.hard_mat(n - 1)
            b = np.zeros((2 ** (n - 1), 2 ** (n - 1)))
            c = np.ones((2 ** (n - 1), 2 ** (n - 1)))
            d = -np.ones((2 ** (n - 1), 2 ** (n - 1)))
            left = np.vstack([a, c])
            right = np.vstack([b, d])
            return np.hstack([left, right])

    def __init__(self, n):
        self.M = 2
        K = 2**n
        tab = np.zeros((K, K, 2))
        tab[:, :, 0] = (self.hard_mat(n) + 1) / 2
        tab[:, :, 1] = 1 - tab[:, :, 0]
        self.tab = tab


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


class Stoltz(NormalFormGame):
    def __init__(self):
        """
        From Gilles Stoltz' thesis, as a counterexample for a case when Hedge
        with a constant learning rate can suffer large internal regret.
        If player 1 plays action 0 for the first T/3 rounds, then 1 for the next
        T/3 rounds, then 2 for the last T/3 rounds, and if Player 2 uses Hedge, then
        player 3 suffers large internal regret.
        Losses are converted to rewards and normalized in [0, 1]
        """
        M = 2
        tab = np.zeros((3, 3, 2))
        tab[0, 0] = np.array([0, 0])
        tab[0, 1] = np.array([1, 1])
        tab[0, 2] = np.array([5, 5])
        tab[1, 0] = np.array([1, 1])
        tab[1, 1] = np.array([0, 0])
        tab[1, 2] = np.array([5, 5])
        tab[2, 0] = np.array([2, 2])
        tab[2, 1] = np.array([1, 1])
        tab[2, 2] = np.array([0, 0])
        tab = tab / np.max(tab)
        tab = 1 - tab
        super().__init__(M, tab)


class GuessTheNumber(NormalFormGame):
    def __init__(self, K):
        # Build a random permutation matrix
        P = np.eye(K)
        b = np.arange(K)
        np.random.shuffle(b)
        P = P[b, :]

        C = np.ones((K, K))
        for i in range(K):
            for j in range(K):
                if i > j:
                    C[i, j] = 0
                elif i == j:
                    C[i, j] = 1 / 2
        C = np.dot(P, np.dot(C, np.transpose(P)))
        tab = np.zeros((K, K, 2))
        tab[:, :, 0] = C
        tab[:, :, 1] = 1 - C
        self.M = 2
        self.tab = tab


class Coordination(NormalFormGame):
    def __init__(self):
        """
        No Pure Nash equilibrium
        """
        M = 2
        tab = np.zeros((2, 2, 2))
        tab[0, 0] = np.array([1, 1])
        tab[0, 1] = np.array([0, 0])
        tab[1, 0] = np.array([0, 0])
        tab[1, 1] = np.array([1, 1])
        super().__init__(M, tab)


class MatchingPennies(NormalFormGame):
    def __init__(self):
        """
        No Pure Nash equilibrium
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


class PartyGame:
    """
    The game is M players that can bring up to K guests to a party. The value
    for each player is a function of the total number of guests.
    """

    def __init__(self, M, K, values):
        """
        values is an (M, K*M) table
        values[m, i] is the payoff of player m when the total number of guest
        is i
        """
        self.M = M
        self.K = K

        self.values = values

    def compute_exp_payoffs(self, n, plays):
        """
        plays is a tuple of probability distributions over actions
        returns the vector of expected payoffs for actions of player n
        """
        law_of_sum = np.array([1.0])
        for i, play in enumerate(plays):
            if i == n:
                continue
            else:
                law_of_sum = np.convolve(law_of_sum, play)
        r = np.zeros(self.K)
        for i in range(self.K):
            u = np.zeros(self.K)
            u[i] = 1
            vect = np.convolve(u, law_of_sum)
            r[i] = np.dot(self.values[n, :], vect)
        return r
