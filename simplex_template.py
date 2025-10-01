import numpy as np
import sys
import argparse

# global vars
eps = 0.00001

def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    #  max c^T x
    #  Ax = b
    #  x >= 0
    #  n - число переменных
    #  m - число ограничений
    #  Да, действительно, считаем что n > m (с учетом слаков)
    # Предполагаем что проблема точно feasible но возможно unbounded
    m, n = A.shape
    n -= m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    while True:
        # Подсказка: np.linalg.solve(M, v) решает систему Mx = v

        # TODO: Посчитать reduced cost's 
        reduced_cost = ...

        # TODO: Находим кандидата для входа в базис
        entering_index = ...

        # TODO: Вычисляем направление, не забывая детектировать unbounded
        d = ...

        # TODO: Найти кандидата для выхода из базиса
        leaving_index = ...
        
        # TODO: Обновляем basis и nbasis
        basis = ...
        nbasis = ...

    # TODO: Восстановить исходную систему, восстановить x и вернуть результат
    return "optimal", ..., ...


def Phase1(c, A, b):
    # TODO: Создаем вспомогательную задачу
    new_c = ...
    new_A = ...
    basis = ...
    nbasis = ...

    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None
    
    # TODO: Нужно восстановить исходную задачу
    c = ...
    A = ...
    basis = ...
    nbasis = ...

    return PrimalSimplex(c, A, b, basis, nbasis)


def Solve(c, A, b):
    if np.all(b >= 0):
        # TODO: Добавляем слаки в систему
        return PrimalSimplex(c, A, b)
    
    # Иначе запускаем фазу 1
    return Phase1(c, A, b)

def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
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