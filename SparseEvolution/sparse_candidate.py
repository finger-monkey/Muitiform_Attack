import numpy as np
import torch
from copy import deepcopy
from operator import attrgetter


class SparseCandidate:
    def __init__(self, pixels, values, x, p_size):
        self.pixels = pixels
        self.values = values
        self.x = x
        self.fitnesses = []
        self.is_adversarial = None
        self.w = x.shape[-1]
        self.delta = len(self.pixels)
        self.domination_count = None
        self.dominated_solutions = None
        self.rank = None
        self.crowding_distance = None

        self.loss = None
        self.p_size = p_size

    def copy(self):
        return deepcopy(self)

    def squared_l2_distance(self, img):
        difference = img - self.x
        return float(torch.sum(difference * difference).item())

    def build_adversarial(self):
        x_adv = self.x.clone()
        for i in range(self.delta):
            row, column = divmod(int(self.pixels[i]), self.w)
            value = torch.as_tensor(self.values[i], device=x_adv.device, dtype=x_adv.dtype)
            x_adv[:, row, column] += value * self.p_size

        return x_adv

    def repair(self, epsilon, max_pixels=None):
        
        max_pixels = self.delta if max_pixels is None else int(max_pixels)
        
        _, first = np.unique(self.pixels, return_index=True)
        keep = np.sort(first)[:max_pixels]
        self.pixels = self.pixels[keep]
        self.values = np.asarray(self.values[keep]).clip(-1, 1)
        self.values = np.rint(self.values).astype(np.int64)

        
        for i, pixel in enumerate(self.pixels):
            row, column = divmod(int(pixel), self.w)
            base = self.x[:, row, column].detach().cpu().numpy()
            proposed = base + self.values[i] * self.p_size
            self.values[i, np.abs(proposed) > float(epsilon) + 1e-12] = 0
        self.delta = len(self.pixels)
        return self

    def evaluate(self, loss_function, include_dist, *context):
        fs = loss_function(self, *context)
        self.is_adversarial = bool(fs[0])
        self.fitnesses = list(fs[1:])
        self.fitnesses = np.array(self.fitnesses)
        self.loss = float(self.fitnesses[0]) if len(self.fitnesses) else 0.0

    def dominates(self, soln):
        
        return bool(np.all(self.fitnesses <= soln.fitnesses)
                    and np.any(self.fitnesses < soln.fitnesses))


def pareto_sort(population):
    fronts = [[]]
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in population:
            if individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for individual in fronts[i]:
            for other_individual in individual.dominated_solutions:
                other_individual.domination_count -= 1
                if other_individual.domination_count == 0:
                    other_individual.rank = i + 1
                    temp.append(other_individual)
        i = i + 1
        fronts.append(temp)

    return fronts


def compute_crowding_distance(front):
    if len(front) > 0:
        solutions_num = len(front)
        for individual in front:
            individual.crowding_distance = 0

        for m in range(len(front[0].fitnesses)):
            front.sort(key=lambda individual: individual.fitnesses[m])
            front[0].crowding_distance = 10 ** 9
            front[solutions_num - 1].crowding_distance = 10 ** 9
            m_values = [individual.fitnesses[m] for individual in front]
            scale = max(m_values) - min(m_values)
            if scale == 0: scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[i + 1].fitnesses[m] - front[i - 1].fitnesses[m]) / scale


def compare_crowding(individual, other_individual):
    if (individual.rank < other_individual.rank) or ((individual.rank == other_individual.rank) and (
            individual.crowding_distance > other_individual.crowding_distance)):
        return 1
    else:
        return -1


def __tournament(population, tournament_size):
    participants = np.random.choice(population, size=(tournament_size,), replace=False)
    best = None
    for participant in participants:
        if best is None or (
                compare_crowding(participant, best) == 1):
            best = participant

    return best


def select_parents(population, tournament_size):
    parents = []
    while len(parents) < len(population) // 2:
        parent1 = __tournament(population, tournament_size)
        parent2 = __tournament(population, tournament_size)

        parents.append([parent1, parent2])
    return parents
