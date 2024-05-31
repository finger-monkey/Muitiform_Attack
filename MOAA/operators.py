from MOAA.Solutions import Solution
import numpy as np
import random


def mutation(soln: Solution, pm: float, all_pixels: np.array, zero_prob: float):
    all_pixels = all_pixels.copy()
    pixels = soln.pixels.copy()
    rgbs = soln.values.copy()

    eps_it = max([int(len(soln.pixels) * pm), 1])
    eps = len(soln.pixels)

    # select pixels to keep
    A_ = np.random.choice(eps, size=(eps - eps_it,), replace=False)
    new_pixels = pixels[A_]
    new_rgbs = rgbs[A_]

    # select new pixels to replace
    u_m = np.delete(all_pixels, pixels)
    B = np.random.choice(u_m, size=(eps_it,), replace=False)

    ones_prob = (1 - zero_prob) / 2
    rgbs_ = np.random.choice([-1, 1, 0], size=(eps_it, 3), p=(ones_prob, ones_prob, zero_prob))
    pixels_ = all_pixels[B]

    new_pixels = np.concatenate([new_pixels, pixels_], axis=0)
    new_rgbs = np.concatenate([new_rgbs, rgbs_], axis=0)

    soln.pixels = new_pixels
    soln.values = new_rgbs


def crossover(soln1: Solution, soln2: Solution, pc: float):
    l = max([int(len(soln1.pixels) * pc), 1])
    k = len(soln1.pixels)
    # S1 crossover with S2
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln2.pixels[pi] not in soln1.pixels])

    offspring1 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring1.pixels[switched_pixels] = soln2.pixels[switched_pixels].copy()
        offspring1.values[switched_pixels] = soln2.values[switched_pixels].copy()

    # S2 crossover with S1
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln1.pixels[pi] not in soln2.pixels])
    offspring2 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring2.pixels[switched_pixels] = soln1.pixels[switched_pixels].copy()
        offspring2.values[switched_pixels] = soln1.values[switched_pixels].copy()

    return offspring1, offspring2


def generate_offspring(parents, pc, pm, all_pixels, zero_prob):
    children = []
    for pi in parents:
        offspring1, offspring2 = crossover(pi[0], pi[1], pc)
        mutation(offspring1, pm, all_pixels, zero_prob)
        mutation(offspring2, pm, all_pixels, zero_prob)

        assert len(np.unique(offspring1.pixels)) == len(offspring1.pixels)
        assert len(np.unique(offspring2.pixels)) == len(offspring2.pixels)
        children.extend([offspring1, offspring2])

    return children

def dominates(p, q, objectives):
    isBetter = False
    for i in range(len(objectives)):
        if p.objectives[i] > q.objectives[i]:
            return False
        elif p.objectives[i] < q.objectives[i]:
            isBetter = True
    return isBetter

def fast_nondominated_sort(population, objectives):
    S = [[] for _ in range(len(population))]
    front = [[]]
    n = [0 for _ in range(len(population))]
    rank = [0 for _ in range(len(population))]

    for p in range(len(population)):
        S[p] = []
        n[p] = 0
        for q in range(len(population)):
            if dominates(population[p], population[q], objectives):
                S[p].append(q)
            elif dominates(population[q], population[p], objectives):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if front[0] == []:
                front[0] = [p]
            else:
                front[0].append(p)

    i = 0
    while len(front[i]) != 0:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if Q == []:
                        Q = [q]
                    else:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front)-1]
    return front


def calculate_crowding_distance(front):

    if len(front) == 0:
        return

    num_objectives = len(front[0].objectives)
    for individual in front:
        individual.crowding_distance = 0

    for i in range(num_objectives):
        front.sort(key=lambda x: x.objectives[i])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')

        if front[0].objectives[i] == front[-1].objectives[i]:
            continue

        for j in range(1, len(front) - 1):
            front[j].crowding_distance += (front[j + 1].objectives[i] - front[j - 1].objectives[i]) / \
                                          (front[-1].objectives[i] - front[0].objectives[i])

def tournament_selection(population, tournament_size):

    selected_parents = []
    for _ in range(len(population) // 2):
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
        selected_parents.append((tournament[0], tournament[1]))

    return selected_parents
