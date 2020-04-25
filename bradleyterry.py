from itertools import product

import numpy as np

# No unit, scale is not meaningful.
PLAYER_RATINGS = [
    800,
    1000,
    1200,
    1600,
]
NUM_PLAYERS = len(PLAYER_RATINGS)

NORMALIZED_RATINGS = np.array(PLAYER_RATINGS, dtype=np.float32)
NORMALIZED_RATINGS /= np.sum(NORMALIZED_RATINGS)


def total_variation(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def simulate_play(N=50):
    wij = np.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=np.uint16)

    for (i, e0), (j, e1) in product(enumerate(PLAYER_RATINGS), repeat=2):
        if i == j:
            continue
        p = np.array([e0, e1], dtype=np.float32)
        p /= np.sum(p)
        outcomes = np.random.choice([0, 1], p=p, size=N)
        wij[i, j] += np.sum(outcomes == 0)
        wij[j, i] += np.sum(outcomes == 1)

    return wij


def evaluate_likelihood(wij, ratings):
    rj, ri = np.meshgrid(ratings, ratings)
    ll = np.sum(wij * (np.log(ri) - np.log(ri + rj)))
    return ll


def bradley_terry(wij, MAX_STEPS=50):
    Wi = np.sum(wij, axis=1)
    nij = wij + wij.T

    ratings = np.ones(NUM_PLAYERS, dtype=np.float32) / NUM_PLAYERS
    for step in range(MAX_STEPS):
        prev_ratings = ratings

        # MM update
        rj, ri = np.meshgrid(ratings, ratings)
        ratings = Wi * np.sum(nij / (ri + rj), axis=1) ** -1
        ratings /= np.sum(ratings)

        # Likelihood maximization stats
        ll = evaluate_likelihood(wij, ratings)
        dll = ll - evaluate_likelihood(wij, prev_ratings)
        tvp = total_variation(prev_ratings, ratings)
        tvt = total_variation(NORMALIZED_RATINGS, ratings)
        print(f'Step {step}: ll: {ll:0.5f} ({dll:0.5f}), variation from prev: {tvp:0.5f}, varation from true: {tvt:0.5f}')

        if tvp < 1e-6:
            break

    return ratings


if __name__ == '__main__':
    wij = simulate_play()
    rating_estimates = bradley_terry(wij)
    print('Final ratings:', rating_estimates)
