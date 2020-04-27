from itertools import product

import numpy as np

# No unit, scale is not meaningful.
PLAYER_RATINGS = [
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1350,
    1450,
    1550,
    1600,
    1620,
]
NUM_PLAYERS = len(PLAYER_RATINGS)


def to_elo(gamma, shift=False):
    elos = 400 * np.log10(gamma)
    if shift:
        elos -= np.min(elos)
    return elos


def to_gamma(elo):
    return 10. ** (elo / 400)


def simulate_play(advantage_elo=0, draw_elo=0, num_games=500):

    wij = np.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=np.uint16)
    lij = np.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=np.uint16)
    dij = np.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=np.uint16)

    f = lambda delta: 1. / (1. + 10. ** (-delta / 400))

    for (i, e0), (j, e1) in product(enumerate(PLAYER_RATINGS), repeat=2):
        if i == j:
            continue

        p0 = f(e0 + advantage_elo - e1 - draw_elo)
        p1 = f(e1 - e0 - advantage_elo - draw_elo)
        pd = 1 - p0 - p1
        assert abs(p0 + p1 + pd - 1) < 1e-6
        outcomes = np.random.choice([-1, 0, 1], p=[pd, p0, p1], size=num_games)
        wij[i, j] = np.sum(outcomes ==  0)
        lij[i, j] = np.sum(outcomes == +1)
        dij[i, j] = np.sum(outcomes == -1)

    return wij, lij, dij


def total_variation(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def evaluate_likelihood(gamma, th_a, th_d, wij, lij, dij):
    gj, gi = np.meshgrid(gamma, gamma)

    # player 0 (white) wins
    n_w = np.log(gi) + np.log(th_a)
    d_w = np.log(th_a * gi + th_d * gj)
    ll_w = np.sum(wij * (n_w - d_w))

    # player 0 (white) losses
    n_l = np.log(gj)
    d_l = np.log(gj + th_a * th_d * gi)
    ll_l = np.sum(lij * (n_l - d_l))

    # draws
    ll_d = np.sum(dij * (n_w + n_l + np.log(th_d ** 2 - 1) - d_w - d_l))

    return ll_w + ll_l + ll_d


def mm(wij, lij, dij, fixed_advantage_elo=None, fixed_draw_elo=None, MAX_STEPS=200):
    Wi = np.sum(wij, axis=1)
    Ng = np.sum(wij + lij + dij + dij.T, axis=1)

    gamma = np.ones(NUM_PLAYERS, dtype=np.float32) / NUM_PLAYERS
    theta_a = 1 if fixed_advantage_elo is None else to_gamma(fixed_advantage_elo)
    theta_d = 1 if fixed_draw_elo is None else to_gamma(fixed_draw_elo)

    prev_gamma = gamma
    prev_th_a = theta_a
    prev_th_d = theta_d
    prev_ll = None
    for step in range(MAX_STEPS):

        gj, gi = np.meshgrid(gamma, gamma)

        # MM gamma update
        # NxN
        # will reduce to Nx1 by summing across all rows
        n0 = (wij + dij) * theta_a
        n1 = (lij + dij) * theta_a * theta_d
        n2 = (dij + lij).T
        n3 = (dij + wij).T * theta_d

        d0 = theta_a * gi + theta_d * gj
        d1 = gj + theta_a * theta_d * gi
        d2 = gi + theta_a * theta_d * gj
        d3 = theta_a * gj + theta_d * gi

        Dg = np.sum(n0 / d0 + n1 / d1 + n2 / d2 + n3 / d3, axis=1)
        gamma = Ng / Dg
        gamma = gamma / np.sum(gamma)

        # MM theta_a update
        if fixed_advantage_elo is None:
            Na = np.sum(wij + dij)
            Da0 = np.sum((wij + dij) * gi  / (theta_a * gi + theta_d * gj))
            Da1 = np.sum((lij + dij) * theta_d * gi / (gj + theta_a * theta_d * gi))
            theta_a = Na / (Da0 + Da1)

        # MM theta_d update
        if fixed_draw_elo is None:
            Nd0 = np.sum((wij + dij) * gj / (theta_a * gi + theta_d * gj))
            Nd1 = np.sum((lij + dij) * theta_a * gi / (gj + theta_a * theta_d * gi))
            A = np.sum(dij) / (Nd0 + Nd1)
            theta_d = A + np.sqrt(A ** 2 + 1)

        # Likelihood maximization stats
        ll = evaluate_likelihood(gamma, theta_a, theta_d, wij, lij, dij)
        dll = ll - prev_ll if prev_ll is not None else float('nan')
        tvp = total_variation(prev_gamma, gamma)
        dtha = abs(prev_th_a - theta_a)
        dthd = abs(prev_th_d - theta_d)
        print(f'Step {step}: ll: {ll:0.5f} ({dll:0.5f}), variation from prev: {tvp:0.5f}, dth_a: {dtha}, dth_d: {dthd}')

        #, varation from true: {tvt:0.5f}')

        if tvp < 1e-9 and dtha < 1e-9 and dthd < 1e-9:
            break
        prev_ll = ll
        prev_gamma = gamma
        prev_th_a = theta_a
        prev_th_d = theta_d

    return gamma, theta_a, theta_d


if __name__ == '__main__':
    wij, lij, dij = simulate_play(draw_elo=100, advantage_elo=50)
    gammas, th_a, th_d = mm(wij, lij, dij, fixed_draw_elo=100, fixed_advantage_elo=50)

    print('RESULTS:')
    print('--------')
    print('Gammas:', gammas)
    print('ELO ratings:', to_elo(gammas, shift=True))
    print(f'theta_a: {th_a} ({to_elo(th_a)} ELO)')
    print(f'theta_d: {th_d} ({to_elo(th_d)} ELO)')

    print('Relative strength error:')
    true_gammas = to_gamma(np.array(PLAYER_RATINGS))
    g0, g1 = np.meshgrid(gammas, gammas)
    s0, s1 = np.meshgrid(true_gammas, true_gammas)
    rse = np.abs(g0 / (g0 + g1) - (s0 / (s0 + s1)))
    print(rse)

