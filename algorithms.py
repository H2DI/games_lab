import numpy as np
import scipy.linalg
import warnings


class MultiAgent:
    def __init__(self, game, agent_list, track_profile=False):
        self.game = game
        self.agent_list = agent_list
        self.track_profile = track_profile
        self.M = len(agent_list)

        assert self.M == self.game.M
        self.t = 0
        if self.track_profile:
            self.cumulative_profile = np.zeros(np.shape(self.game.tab)[:-1])

    def initialize(self, plays):
        for n, agent in enumerate(self.agent_list):
            expected_rewards = self.game.compute_exp_payoffs(n, plays)
            agent.update(plays[n], expected_rewards)
        if self.track_profile:
            self.cumulative_profile += self.joint_play(self.t)

    def play(self):
        self.t += 1
        plays = []
        for agent in self.agent_list:
            plays.append(agent.next_play())
        for n, agent in enumerate(self.agent_list):
            expected_rewards = self.game.compute_exp_payoffs(n, plays)
            agent.update(plays[n], expected_rewards)
        if self.track_profile:
            self.cumulative_profile += self.joint_play(self.t)

    def play_T_times(self, T):
        for _ in range(T):
            self.play()

    def joint_play(self, t):
        for n, agent in enumerate(self.agent_list):
            if n == 0:
                r = agent.play_history[t - 1]
            else:
                r = np.tensordot(r, agent.play_history[t - 1], axes=0)
        return r

    def empirical_profile(self, t=-1):
        if t == -1:
            t = self.t
        assert t > 0
        if self.track_profile and t == self.t:
            return self.cumulative_profile / t
        else:
            r = np.zeros(np.shape(self.game.tab)[:-1])
            for time in range(t):
                r += self.joint_play(time)
            return r / t

    def sample_from_eq(self, t=-1):
        if t == -1:
            t = self.t
        t_sample = np.random.randint(t)
        r = []
        for n, agent in enumerate(self.agent_list):
            K = agent.K
            r.append(np.random.choice(K, p=agent.play_history[t_sample]))
        return r

    def check_equilibria(self, dist):
        """
            not finished
            returns r such r[n][i, j] is the constraint comparing
        """
        assert np.isclose(np.sum(dist), 1)
        r = []
        for n, agent in enumerate(self.agent_list):
            r.append(np.zeros((self.K, self.K)))
            for i in range(agent.K):
                for j in range(agent.K):
                    a = 0
                    r[n][i, j] = a
        return


class Agent:
    def __init__(self, K, horizon=-1, label=""):
        self.K = K
        self.t = 0
        self.label = label
        self.horizon = horizon

        self.play_history = []
        self.reward_history = []
        self.received_rewards = []

        self.cumul_received_rewards = 0
        self.all_cumul_rewards = np.zeros(K)
        self.internal_comps = np.zeros((K, K))

        self.ext_regret = []
        self.int_regret = []
        self.swap_regret = []

        self.times_war = []

    def next_play(self):
        pass

    def update(self, play, rewards):
        """
            play is a probability vector (aka a mixed strategy)
            rewards is a vector of floats
        """
        self.t += 1
        self.play_history.append(play)
        self.reward_history.append(rewards)

        self.all_cumul_rewards += rewards
        self.received_rewards.append(np.dot(play, rewards))
        self.cumul_received_rewards += self.received_rewards[-1]

        self.ext_regret.append(
            np.max(self.all_cumul_rewards - self.cumul_received_rewards)
        )
        # internal regret comparison, used to both compute internal regret and
        # for internal regret algorithms
        for i in range(self.K):
            for j in range(self.K):
                self.internal_comps[i, j] += play[i] * (rewards[j] - rewards[i])

        self.int_regret.append(np.max(self.internal_comps))
        self.swap_regret.append(np.max(np.sum(self.internal_comps, axis=0)))

    def fixed_point(self, Q):
        """
            Computes the probability fixed point of a stochastic matrix
        """
        K = len(Q)
        X = np.vstack((np.eye(K) - Q, np.ones(K)))
        b = np.zeros(K + 1)
        b[K] = 1

        v = np.matmul(np.transpose(X), X)
        btilde = np.matmul(np.transpose(X), b)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                p = scipy.linalg.solve(v, btilde)
                # if any(p < 0) or any(p > 1) or not (np.isclose(np.sum(p), 1)):
                #     print("Standard solve: ")
                #     print("Q")
                #     print(Q)
                #     print("p")
                #     print(p)
                return p
            except (Warning, np.linalg.LinAlgError) as e:
                # self.times_war.append(self.t)
                return np.sum(np.power(Q, 10), axis=1) / self.K


class FTL(Agent):
    def next_play(self):
        if self.t == 0:
            return np.ones(self.K) / self.K
        r = np.zeros(self.K)
        r[np.argmax(self.all_cumul_rewards)] = 1
        return r


class Hedge_a(Agent):
    def lr_value(self):
        return np.sqrt(np.log(self.K) / (self.t + 1))

    def next_play(self):
        logweights = self.lr_value() * self.all_cumul_rewards
        max_logweight = np.max(logweights)
        unnormalized_w = np.exp(logweights - max_logweight)
        return unnormalized_w / np.sum(unnormalized_w)


class Hedge(Hedge_a):
    def lr_value(self):
        return np.sqrt(np.log(self.K) / (self.horizon + 1))


class OptimisticHedge_a(Hedge_a):
    def lr_value(self):
        return np.power((self.t + 1), -1 / 4)

    def next_play(self):
        if not (self.reward_history):
            next_pred = np.zeros(self.K)
        else:
            next_pred = self.reward_history[-1]
        logweights = self.lr_value() * (self.all_cumul_rewards + next_pred)
        max_logweight = np.max(logweights)
        unnormalized_w = np.exp(logweights - max_logweight)
        return unnormalized_w / np.sum(unnormalized_w)


class OptimisticHedge(OptimisticHedge_a):
    def lr_value(self):
        return np.power((self.horizon + 1), -1 / 4)


class BMAlg_a(Agent):
    "Swap regret algorithm"

    def __init__(self, K, horizon=-1, label=""):
        super().__init__(K, horizon=horizon, label=label)
        self.sub_algs = [Hedge_a(K, horizon=horizon) for _ in range(K)]
        self.last_Q = np.eye(self.K)

    def next_play(self):
        Q = None
        for alg in self.sub_algs:
            if Q is None:
                Q = np.transpose([alg.next_play()])
            else:
                Q = np.hstack((Q, np.transpose([alg.next_play()])))
        self.last_Q = Q
        return self.fixed_point(Q)

    def update(self, play, rewards):
        super().update(play, rewards)
        for n, alg in enumerate(self.sub_algs):
            alg.update(self.last_Q[:, n], play[n] * rewards)


class BMAlg(BMAlg_a):
    def __init__(self, K, horizon=-1, label=""):
        super().__init__(K, horizon=horizon, label=label)
        self.sub_algs = [Hedge(K, horizon=horizon) for _ in range(K)]
        self.last_Q = np.eye(self.K)


class OptimisticBMAlg_a(BMAlg_a):
    def __init__(self, K, horizon=-1, label=""):
        super().__init__(K, horizon=horizon, label=label)
        self.sub_algs = [OptimisticHedge_a(K, horizon=horizon) for _ in range(K)]
        self.last_Q = np.eye(self.K)


class OptimisticBMAlg(BMAlg_a):
    def __init__(self, K, horizon=-1, label=""):
        super().__init__(K, horizon=horizon, label=label)
        self.sub_algs = [OptimisticHedge(K, horizon=horizon) for _ in range(K)]
        self.last_Q = np.eye(self.K)


class OptimisticAdaHedge(Hedge_a):
    def __init__(self, K, horizon=-1, label=""):
        super().__init__(K, horizon=horizon, label=label)
        self.cum_mix_gap = 0
        self.D = np.log(self.K)
        self.mix_gaps = []
        self.current_lr = np.inf

        self.diffs_linfty = []
        self.diffs_var = []

    def lr_value(self):
        if len(self.reward_history) < 2:
            reward_vec = self.reward_history[-1]
        else:
            reward_vec = self.reward_history[-1] - self.reward_history[-2]
        p = self.play_history[-1]

        self.diffs_linfty.append(max(np.square(reward_vec)))
        self.diffs_var.append(
            np.dot(p, (np.square(reward_vec - np.dot(p, reward_vec))))
        )

        def_mix_gap = max(reward_vec) - np.dot(p, reward_vec)
        if np.isinf(self.current_lr):
            mix_gap = def_mix_gap
        else:
            v_exp = np.exp(self.current_lr * reward_vec)
            if any(np.isinf(v_exp)):
                mix_gap = def_mix_gap
            else:
                mix_gap = -np.dot(p, reward_vec) + 1 / self.current_lr * np.log(
                    np.dot(p, v_exp)
                )
        self.cum_mix_gap += mix_gap
        self.mix_gaps.append(mix_gap)

        if np.isclose(self.cum_mix_gap, 0):
            self.current_lr = np.inf
        else:
            self.current_lr = self.D / self.cum_mix_gap

        return self.current_lr

    def next_play(self):
        if not (self.reward_history):
            return np.ones(self.K) / self.K
        else:
            next_pred = self.reward_history[-1]
        lr = self.lr_value()
        if np.isinf(lr):
            return np.ones(self.K) / self.K
        logweights = lr * (self.all_cumul_rewards + next_pred)
        max_logweight = np.max(logweights)
        unnormalized_w = np.exp(logweights - max_logweight)
        return unnormalized_w / np.sum(unnormalized_w)


class OptimisticBMAdaHedge(BMAlg_a):
    def __init__(self, K, horizon=-1, label=""):
        super().__init__(K, horizon=horizon, label=label)
        self.sub_algs = [OptimisticAdaHedge(K) for _ in range(K)]
        self.last_Q = np.eye(self.K)


class InternalHedge_a(Agent):
    """
    Internal regret algorithm. Broken.

    """

    def lr_value(self):
        return np.sqrt(np.log(self.K * (self.K - 1)) / (self.t + 1))

    def next_play(self):
        logweights = -self.lr_value() * self.internal_comps.copy()
        for i in range(self.K):
            logweights[i, i] = -np.inf
        max_logweight = np.max(logweights)
        unnormalized_Qweights = np.exp(logweights - max_logweight)
        Qweights = unnormalized_Qweights / np.sum(unnormalized_Qweights)
        Q = Qweights  # np.copy(Qweights) Qweights gets modified but thats okay
        for j in range(self.K):
            Q[j, j] = 1 - np.sum(Q[:, j])
        p = self.fixed_point(Q)
        p = np.maximum(np.minimum(p, 1), 0)
        p = p / np.sum(p)
        return p


class InternalHedge(InternalHedge_a):
    def lr_value(self):
        return np.sqrt(np.log(self.K * (self.K - 1)) / (self.horizon + 1))


class RandomPlay(Agent):
    def next_play(self):
        r = np.random.rand(self.K)
        return r / np.sum(r)


class RegretMatching(Agent):
    def next_play(self):
        p = np.zeros(self.K)
        a = self.internal_comps


class UniformPlay(Agent):
    def next_play(self):
        return np.ones(self.K) / self.K
