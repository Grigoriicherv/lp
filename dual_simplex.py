import numpy as np
import argparse
import sys

EPS = 1e-9
U_BOUND = 1e10


def DualSimplex(c, A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    m, n = A.shape
    A_eq = np.hstack([A, np.eye(m)])
    c_eq = np.concatenate([c, np.zeros(m)])
    n_full = n + m

    basis = list(range(n, n_full))
    nbasis = list(range(0, n))

    d = np.ones(n_full)

    x = np.zeros(n_full)

    for j in nbasis:
        if c_eq[j] > EPS:
            d[j] = -1.0
            x[j] = U_BOUND
        else:
            d[j] = 1.0
            x[j] = 0.0

    max_iter = 1000
    for iteration in range(max_iter):
        B = A_eq[:, basis]
        N = A_eq[:, nbasis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return "singular_matrix", None, None

        x_N_vec = x[nbasis]
        rhs = b - N @ x_N_vec
        x_B = B_inv @ rhs
        x[basis] = x_B

        if np.all(x_B >= -EPS):
            if np.any(x[:n] >= U_BOUND - 1.0):
                return "unbounded", None, None
            return "optimal", x[:n], np.dot(c, x[:n])

        violating_rows = [k for k in range(m) if x_B[k] < -EPS]
        row_out = min(violating_rows, key=lambda k: x_B[k])
        idx_leaving = basis[row_out]

        rho = B_inv[row_out, :]
        alpha_row = rho @ N
        sigma = -alpha_row * d[nbasis]

        candidates = [k for k in range(len(nbasis)) if sigma[k] > EPS]

        if not candidates:
            return "infeasible", None, None

        c_B = c_eq[basis]
        y = c_B @ B_inv
        bar_c_N = c_eq[nbasis] - y @ N
        r = d[nbasis] * bar_c_N
        ratios = []
        for k in candidates:
            val = -r[k] / sigma[k]
            ratios.append(val)
        min_ratio = min(ratios)

        best_indices = [k for i, k in enumerate(candidates) if abs(ratios[i] - min_ratio) < EPS]
        col_in_nbasis = min(best_indices, key=lambda k: nbasis[k])
        idx_entering = nbasis[col_in_nbasis]

        d[idx_leaving] = 1.0
        x[idx_leaving] = 0.0
        basis[row_out] = idx_entering
        nbasis[col_in_nbasis] = idx_leaving

    return "iterations_limit", None, None

def proc_cmd():
    parser = argparse.ArgumentParser(description="Dual Simplex")
    parser.add_argument("filename", type=str, help="Path to input file")
    return parser.parse_args()


def main():
    # boilerplate for reading input data
    args = proc_cmd()
    with open(args.filename, 'r', encoding='utf-8') as f:
        n, m = map(int, f.readline().split())
        c = np.array(list(map(float, f.readline().split())))
        A = []
        b = []
        for _ in range(m):
            *row, bi = map(float, f.readline().split())
            A.append(row)
            b.append(bi)
        A = np.array(A)
        b = np.array(b)
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("Solving the linear program using the Dual Simplex method...\n")
    status, solution, objective = DualSimplex(c, A, b)
    print("\nResult:")
    print("Status:", status)
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("No solution found.")
if __name__ == '__main__':
    main()