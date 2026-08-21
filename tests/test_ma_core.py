import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SparseEvolution.sparse_candidate import SparseCandidate, pareto_sort
from SparseEvolution.evolutionary_attack import EvolutionaryPopulation


def test_candidate_repair_enforces_global_budget():
    epsilon = 8 / 255.0
    base = torch.zeros(3, 2, 2)
    base[0, 0, 0] = epsilon
    candidate = SparseCandidate(
        np.array([0, 1]), np.array([[1, 1, 1], [-1, 0, 1]]),
        base, epsilon)
    candidate.repair(epsilon, max_pixels=2)
    assert candidate.build_adversarial().abs().max().item() <= epsilon + 1e-7
    assert candidate.values[0, 0] == 0


def test_pareto_sort_uses_all_three_objectives():
    class Individual:
        def __init__(self, values):
            self.is_adversarial = False
            self.fitnesses = np.asarray(values)

        def dominates(self, other):
            return bool(np.all(self.fitnesses <= other.fitnesses)
                        and np.any(self.fitnesses < other.fitnesses))

    better = Individual([0.1, 0.2, 0.3])
    worse = Individual([0.2, 0.2, 0.3])
    assert pareto_sort([better, worse])[0] == [better]


def test_population_uses_current_fitness_signature():
    candidate = SparseCandidate(
        np.array([0]), np.array([[0, 0, 0]]), torch.zeros(3, 1, 1), 8 / 255.0)
    received = []

    def fitness(solution):
        received.append(solution)
        return [False, 0.1, 0.2, 0.0]

    population = EvolutionaryPopulation([candidate], fitness, include_dist=True)
    population.evaluate(None, None, None, None)
    assert received == [candidate]
    assert candidate.fitnesses.tolist() == [0.1, 0.2, 0.0]
