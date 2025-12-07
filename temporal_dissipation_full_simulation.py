#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Full simulation code for:
# "Temporal dissipation improves robustness of recurrent neural computation"
#
# This script reproduces all numerical results reported in the manuscript:
# - Attractor memory retrieval accuracy (Task 1)
# - Sequence tracking accuracy (Task 2)
# - Nonlinear input-output mapping error (Task 3)
#
# NumPy is required. Matplotlib is optional (for plotting).

import math
import numpy as np

try:
    import matplotlib.pyplot as plt  # noqa
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ======================================================
# 1. Common utilities
# ======================================================

def grad_U(q, a=1.0, b=1.0):
    \"\"\"Gradient of the double-well potential.

    U(q_i) = a * q_i^4 / 4 - b * q_i^2 / 2
    dU/dq_i = a * q_i^3 - b * q_i

    Parameters
    ----------
    q : ndarray, shape (N,)
        State vector.

    Returns
    -------
    ndarray, shape (N,)
        Gradient dU/dq.
    \"\"\"
    return a * q**3 - b * q


def overlaps_all(q, patterns):
    \"\"\"Compute overlap between state q and each stored pattern.

    overlap_mu = mean_i sign(xi_i^mu) * tanh(q_i)

    Parameters
    ----------
    q : ndarray, shape (N,)
    patterns : ndarray, shape (M, N)

    Returns
    -------
    ndarray, shape (M,)
        Overlap values for each pattern.
    \"\"\"
    r = np.tanh(q)
    return np.mean(np.sign(patterns) * r[None, :], axis=1)


def make_hopfield_patterns_and_W(N=50, M=3, seed=0):
    \"\"\"Generate Hopfield-type patterns and weight matrix.

    Parameters
    ----------
    N : int
        Number of units.
    M : int
        Number of patterns.
    seed : int
        Random seed.

    Returns
    -------
    patterns : ndarray, shape (M, N)
    W : ndarray, shape (N, N)
    rng : numpy.random.Generator
    \"\"\"
    rng = np.random.default_rng(seed)
    patterns = rng.choice([-1.0, 1.0], size=(M, N))
    W = np.zeros((N, N))
    for mu in range(M):
        W += np.outer(patterns[mu], patterns[mu])
    W /= N
    np.fill_diagonal(W, 0.0)
    return patterns, W, rng


# ======================================================
# 2. Task 1: Attractor memory retrieval
# ======================================================

def simulate_attractor_task(
    N=50,
    M=3,
    a=1.0,
    b=1.0,
    gamma_q=0.2,
    gamma_p=0.2,
    dt=0.01,
    T=15.0,
    noise_levels=None,
    n_trials=100,
    seed=1,
):
    \"\"\"Attractor memory retrieval task.

    Parameters
    ----------
    N : int
        Number of units.
    M : int
        Number of stored patterns.
    a, b : float
        Potential parameters.
    gamma_q, gamma_p : float
        Dissipation coefficients for q and p.
    dt : float
        Time step.
    T : float
        Total simulation time.
    noise_levels : array-like
        Standard deviations of initial noise.
    n_trials : int
        Number of trials per noise level.
    seed : int
        Random seed.

    Returns
    -------
    noise_levels : ndarray
    acc_diss : ndarray
        Retrieval accuracy in the dissipative regime.
    acc_rev : ndarray
        Retrieval accuracy in the reversible regime.
    \"\"\"
    if noise_levels is None:
        noise_levels = np.array([0.0, 0.4, 1.2])

    patterns, W, rng = make_hopfield_patterns_and_W(N, M, seed=seed)
    n_steps = int(T / dt)

    acc_diss = []
    acc_rev = []

    for noise_std in noise_levels:
        successes_d = 0
        successes_r = 0

        for _ in range(n_trials):
            mu = rng.integers(0, M)
            q0 = patterns[mu] + rng.normal(scale=noise_std, size=N)
            p0 = np.zeros(N)

            # Reversible dynamics (gamma_q = gamma_p = 0)
            q = q0.copy()
            p = p0.copy()
            for _ in range(n_steps):
                dq = p
                dp = -grad_U(q, a, b) - W @ q
                q = q + dt * dq
                p = p + dt * dp
            ov = overlaps_all(q, patterns)
            mu_hat = int(np.argmax(ov))
            if mu_hat == mu:
                successes_r += 1

            # Dissipative dynamics
            q = q0.copy()
            p = p0.copy()
            for _ in range(n_steps):
                dq = p - gamma_q * q
                dp = -grad_U(q, a, b) - W @ q - gamma_p * p
                q = q + dt * dq
                p = p + dt * dp
            ov = overlaps_all(q, patterns)
            mu_hat = int(np.argmax(ov))
            if mu_hat == mu:
                successes_d += 1

        acc_diss.append(successes_d / n_trials)
        acc_rev.append(successes_r / n_trials)

    return np.array(noise_levels), np.array(acc_diss), np.array(acc_rev)


# ======================================================
# 3. Task 2: Sequence tracking
# ======================================================

def simulate_sequence_task(
    N=50,
    M=3,
    a=1.0,
    b=1.0,
    gamma_q=0.2,
    gamma_p=0.2,
    dt=0.01,
    T_total=18.0,
    noise_levels=None,
    n_trials=40,
    seed=2,
):
    \"\"\"Sequence-tracking task (0 -> 1 -> 2).

    External drive changes every T_total/3 seconds to target a different pattern.
    Accuracy is the fraction of time points where the expressed pattern
    matches the target for that segment.
    \"\"\"
    if noise_levels is None:
        noise_levels = np.array([0.0, 0.4, 1.0])

    patterns, W, rng = make_hopfield_patterns_and_W(N, M, seed=seed)
    n_steps = int(T_total / dt)
    segment_steps = n_steps // 3
    alpha = 0.8
    clip_val = 3.0

    acc_diss = []
    acc_rev = []

    for noise_std in noise_levels:
        correct_frac_d = []
        correct_frac_r = []

        for _ in range(n_trials):
            q0 = rng.normal(scale=0.5, size=N)
            p0 = np.zeros(N)

            # Reversible dynamics
            q = q0.copy()
            p = p0.copy()
            correct_count = 0

            for step in range(n_steps):
                seg = step // segment_steps
                if seg >= 3:
                    seg = 2
                mu = seg
                I = alpha * patterns[mu]

                ov = overlaps_all(q, patterns)
                mu_hat = int(np.argmax(ov))
                if mu_hat == mu:
                    correct_count += 1

                dq = p
                dp = -grad_U(q, a, b) - W @ q + I
                q = q + dt * dq
                p = p + dt * dp
                q = np.clip(q, -clip_val, clip_val)
                p = np.clip(p, -clip_val, clip_val)
                if noise_std > 0:
                    q = q + noise_std * math.sqrt(dt) * rng.normal(size=N)

            correct_frac_r.append(correct_count / n_steps)

            # Dissipative dynamics
            q = q0.copy()
            p = p0.copy()
            correct_count = 0

            for step in range(n_steps):
                seg = step // segment_steps
                if seg >= 3:
                    seg = 2
                mu = seg
                I = alpha * patterns[mu]

                ov = overlaps_all(q, patterns)
                mu_hat = int(np.argmax(ov))
                if mu_hat == mu:
                    correct_count += 1

                dq = p - gamma_q * q
                dp = -grad_U(q, a, b) - W @ q - gamma_p * p + I
                q = q + dt * dq
                p = p + dt * dp
                q = np.clip(q, -clip_val, clip_val)
                p = np.clip(p, -clip_val, clip_val)
                if noise_std > 0:
                    q = q + noise_std * math.sqrt(dt) * rng.normal(size=N)

            correct_frac_d.append(correct_count / n_steps)

        acc_diss.append(float(np.mean(correct_frac_d)))
        acc_rev.append(float(np.mean(correct_frac_r)))

    return np.array(noise_levels), np.array(acc_diss), np.array(acc_rev)


# ======================================================
# 4. Task 3: Input-output mapping (reservoir)
# ======================================================

def target_function(x):
    \"\"\"Nonlinear scalar function to approximate.

    f(x1, x2) = sin(pi * x1) * cos(pi * x2)
    \"\"\"
    return math.sin(math.pi * x[0]) * math.cos(math.pi * x[1])


def simulate_io_task(
    N=80,
    a=1.0,
    b=1.0,
    g=0.9,
    gamma_q=0.3,
    gamma_p=0.3,
    dt=0.01,
    T_train=5.0,
    T_test=5.0,
    K_train=80,
    K_test=80,
    noise_levels=None,
    ridge_lambda=1e-3,
    seed=3,
):
    \"\"\"Input-output mapping task using a random reservoir and linear readout.

    Reservoir is run in dissipative, noise-free mode for training.
    At test time, both dissipative and reversible dynamics are evaluated
    under varying noise levels.
    \"\"\"
    if noise_levels is None:
        noise_levels = np.array([0.0, 0.4, 0.8, 1.0])

    rng = np.random.default_rng(seed)

    # Random reservoir connectivity
    W = rng.normal(loc=0.0, scale=g / np.sqrt(N), size=(N, N))
    np.fill_diagonal(W, 0.0)

    # Input embedding matrix
    d_in = 2
    B = rng.normal(loc=0.0, scale=1.0, size=(N, d_in))

    def run_dynamics(x, regime="dissipative", noise_std=0.0, T=5.0):
        steps = int(T / dt)
        q = rng.normal(scale=0.3, size=N)
        p = np.zeros(N)
        I = B @ x

        for _ in range(steps):
            if regime == "reversible":
                dq = p
                dp = -grad_U(q, a, b) - W @ q + I
            else:
                dq = p - gamma_q * q
                dp = -grad_U(q, a, b) - W @ q - gamma_p * p + I
            q = q + dt * dq
            p = p + dt * dp
            if noise_std > 0:
                q = q + noise_std * math.sqrt(dt) * rng.normal(size=N)
        r = np.tanh(q)
        return r

    # Training data (dissipative, noise-free)
    X_train = rng.uniform(low=-1.0, high=1.0, size=(K_train, d_in))
    Y_train = np.array([target_function(x) for x in X_train]).reshape(-1, 1)

    R_train = np.zeros((K_train, N))
    for k in range(K_train):
        R_train[k] = run_dynamics(X_train[k], regime="dissipative", noise_std=0.0, T=T_train)

    RtR = R_train.T @ R_train
    C = (Y_train.T @ R_train) @ np.linalg.inv(RtR + ridge_lambda * np.eye(N))

    # Test data
    X_test = rng.uniform(low=-1.0, high=1.0, size=(K_test, d_in))
    Y_test = np.array([target_function(x) for x in X_test]).reshape(-1, 1)

    mse_diss = []
    mse_rev = []

    for noise_std in noise_levels:
        sq_err_d = []
        sq_err_r = []
        for x, y_true in zip(X_test, Y_test):
            r_d = run_dynamics(x, regime="dissipative", noise_std=noise_std, T=T_test)
            y_hat_d = float(C @ r_d)
            sq_err_d.append((y_hat_d - float(y_true)) ** 2)

            r_r = run_dynamics(x, regime="reversible", noise_std=noise_std, T=T_test)
            y_hat_r = float(C @ r_r)
            sq_err_r.append((y_hat_r - float(y_true)) ** 2)

        mse_diss.append(float(np.mean(sq_err_d)))
        mse_rev.append(float(np.mean(sq_err_r)))

    return np.array(noise_levels), np.array(mse_diss), np.array(mse_rev)


# ======================================================
# 5. Convenience: run all tasks and (optionally) plot
# ======================================================

def run_all():
    # Task 1: attractor
    noise_attr, acc_attr_d, acc_attr_r = simulate_attractor_task()
    print(\"Attractor task:\")
    print(\" noise:\", noise_attr)
    print(\" dissipative accuracy:\", acc_attr_d)
    print(\" reversible accuracy:\", acc_attr_r)
    print()

    # Task 2: sequence tracking
    noise_seq, acc_seq_d, acc_seq_r = simulate_sequence_task()
    print(\"Sequence task:\")
    print(\" noise:\", noise_seq)
    print(\" dissipative accuracy:\", acc_seq_d)
    print(\" reversible accuracy:\", acc_seq_r)
    print()

    # Task 3: IO mapping
    noise_io, mse_io_d, mse_io_r = simulate_io_task()
    print(\"IO mapping task:\")
    print(\" noise:\", noise_io)
    print(\" dissipative MSE:\", mse_io_d)
    print(\" reversible MSE:\", mse_io_r)
    print()

    if HAVE_MPL:
        import matplotlib.pyplot as plt

        # Figure 1: attractor robustness
        plt.figure()
        plt.plot(noise_attr, acc_attr_d, marker=\"o\", label=\"dissipative\")
        plt.plot(noise_attr, acc_attr_r, marker=\"s\", label=\"reversible\")
        plt.xlabel(\"Initial noise σ\")
        plt.ylabel(\"Retrieval accuracy\")
        plt.title(\"Attractor memory robustness\")
        plt.legend()
        plt.tight_layout()

        # Figure 2: sequence tracking
        plt.figure()
        plt.plot(noise_seq, acc_seq_d, marker=\"o\", label=\"dissipative\")
        plt.plot(noise_seq, acc_seq_r, marker=\"s\", label=\"reversible\")
        plt.xlabel(\"Noise σ\")
        plt.ylabel(\"Sequence-tracking accuracy\")
        plt.title(\"Sequence tracking robustness\")
        plt.legend()
        plt.tight_layout()

        # Figure 3: IO mapping robustness
        plt.figure()
        plt.plot(noise_io, mse_io_d, marker=\"o\", label=\"dissipative\")
        plt.plot(noise_io, mse_io_r, marker=\"s\", label=\"reversible\")
        plt.xlabel(\"Noise σ\")
        plt.ylabel(\"MSE\")
        plt.title(\"Input-output mapping robustness\")
        plt.legend()
        plt.tight_layout()

        plt.show()


if __name__ == \"__main__\":
    run_all()
