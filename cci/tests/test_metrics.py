import numpy as np
from cci import metrics

def test_cosine_distance():
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    assert abs(metrics.cosine_distance(u, v) - 1) < 1e-6

def test_cc_turn_zero_novelty():
    D = 0.5
    I = 0.0
    S = 0.1
    cc = metrics.cc_turn(D, I, S)
    assert cc > 0