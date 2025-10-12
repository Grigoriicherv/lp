import numpy as np
import sys
import argparse

# global vars
eps = 1e-9

# Формула Шермана-Моррисона
def _update(B_inv, d, p):
    rp = B_inv[p, :].copy()
    for i in range(B_inv.shape[0]):
        if i != p:
            B_inv[i, :] -= (d[i] / d[p]) * rp
    B_inv[p, :] = rp / d[p]
    return B_inv

def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    m, n = A.shape
    n = n - m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    B_inv = np.linalg.inv(A[:, basis])

    while True:
        B = A[:, basis]
        N = A[:, nbasis]

        x_B = B_inv @ b
        y = B_inv.T @ c[basis]
        reduced_cost = c[nbasis] - N.T @ y

        rc = np.where(reduced_cost > eps)[0]
        if rc.size == 0:
            x = np.zeros(n + m)
            x[basis] = x_B
            z = float(c @ x)
            return "optimal", x, z

        j_in = rc[np.argmin(np.array(nbasis)[rc])]
        entering = nbasis[j_in]
        a_j = A[:, entering]
        d = B_inv @ a_j
        if not np.any(d > eps):
            return "unbounded", None, None

        pos = np.where(d > eps)[0]
        ratios = x_B[pos] / d[pos]
        min_ratio = ratios.min()
        tie = np.where(np.abs(ratios - min_ratio) <= eps)[0]
        pretendents = pos[tie]
        i_out = pretendents[np.argmin(np.array(basis)[pretendents])]
        leaving = basis[i_out]
        basis[i_out] = entering
        nbasis[j_in] = leaving

        B_inv = _update(B_inv, d, i_out)


def Phase1(c, A, b):
    m, n = A.shape
    x0 = n + m

    new_A = np.hstack([A, np.eye(m), -np.ones((m, 1))])
    new_c = np.concatenate([np.zeros(n + m), np.array([-1.0])])

    basis = list(range(n, n + m))
    r = int(np.argmin(b))
    basis[r] = x0
    nbasis = [j for j in range(n + m + 1) if j not in basis]

    status, _, tar = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or tar < -eps:
        return "infeasible", None, None

    if x0 in basis:
        r = basis.index(x0)
        candidates = [j for j in range(n + m)
                      if (j not in basis) and (abs(new_A[r, j]) > eps)]
        enter = min(candidates)
        basis[r] = enter

    A = np.hstack([A, np.eye(m)])
    c = np.concatenate([c, np.zeros(m)])

    basis = [j for j in basis]
    nbasis = [j for j in range(n+m) if j not in basis]

    return PrimalSimplex(c, A, b, basis, nbasis)


def Solve(c, A, b):
    if np.all(b >= 0):
        m = A.shape[0]
        A = np.hstack([A, np.eye(m)])
        c = np.concatenate([c, np.zeros(m)])
        return PrimalSimplex(c, A, b)

    return Phase1(c, A, b)


def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    return parser.parse_args()


def main():
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

    print("Solving the linear program using the Primal Simplex method...\n")
    status, solution, objective = Solve(c, A, b)
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
