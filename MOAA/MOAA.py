import numpy as np
import time
import torch
from torch.nn import functional as F
from MOAA.operators import generate_offspring
from MOAA.Solutions import Solution, fast_nondominated_sort, calculate_crowding_distance, tournament_selection
from operator import attrgetter
import faiss  
from reid.evaluators import extract_features  
from torch.utils.data import DataLoader



class Population:
    def __init__(self, solutions: list, loss_function, include_dist):
        self.population = solutions
        self.fronts = None
        self.loss_function = loss_function
        self.include_dist = include_dist

    def evaluate(self, search_set, search_set2, model1, model2):
        for pi in self.population:
            pi.evaluate(self.loss_function, self.include_dist, search_set, search_set2, model1, model2)

    def find_adv_solns(self, max_dist):
        adv_solns = []
        for pi in self.population:
            if pi.is_adversarial and pi.fitnesses[1] <= max_dist:
                adv_solns.append(pi)

        return adv_solns

class Attack:
    def __init__(self, params,search_set, search_set2, modelTest, modelTest2):
        self.params = params
        self.fitness = []
        self.data = []
        self.search_set = search_set.dataset  
        self.search_set2 = search_set2.dataset  
        self.modelTest = modelTest
        self.modelTest2 = modelTest2


    def completion_procedure(self, population, loss_function, fe, success):
        adversarial_labels = []
        for soln in population.fronts[0]:
            adversarial_labels.append(loss_function.get_label(soln.generate_image()))

        d = {"front0_imgs": [soln.generate_image() for soln in population.fronts[0]],
             "queries": fe,
             "true_label": loss_function.true,
             "adversarial_labels": adversarial_labels,
             "front0_fitness": [soln.fitnesses for soln in population.fronts[0]],
             "fitness_process": self.fitness,
             "success": success
             }

        np.save(self.params["save_directory"], d, allow_pickle=True)
        self.Snoise = population.fronts[0]
    
    def calculate_D(self, f_adv, centroids):
        C = centroids
        D_fadv = torch.matmul((f_adv - C).T, torch.inverse(self.S)) @ (f_adv - C)
        return D_fadv.sum().item()

    def calculate_S(self, f_adv, y_true, model):
        y_pred = model(f_adv).argmax(dim=1)
        S_fadv = (y_pred != y_true).float().mean().item()
        return S_fadv

    def attack(self,noise):
        self.noise = noise
        h, w, c = noise.size()
        pm = self.params["pm"]
        n_pixels = h * w
        all_pixels = np.arange(n_pixels)
        ones_prob = (1 - self.params["zero_probability"]) / 2
        try:
            init_solutions = [Solution(np.random.choice(all_pixels, size=(self.params["eps"]), replace=False),
                                       np.random.choice([-1, 1, 0], size=(self.params["eps"], 3),
                                                        p=(ones_prob, ones_prob, self.params["zero_probability"])),
                                       noise.copy(), self.params["p_size"]) for _ in range(self.params["population_size"])]

            population = Population(init_solutions, self.calculate_fitness, self.params["include_dist"])
            population.evaluate(self.search_set, self.search_set2, self.modelTest, self.modelTest2)
            fe = len(population.population)
            
            for it in range(1, self.params["iterations"]):
                pm = self.params["pm"]
                population.fronts = fast_nondominated_sort(population.population)

                adv_solns = population.find_adv_solns(self.params["max_dist"])
                if len(adv_solns) > 0:
                    self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
                    self.completion_procedure(population, self.calculate_fitness, fe, True)
                    return

                self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)

                for front in population.fronts:
                    calculate_crowding_distance(front)
                parents = tournament_selection(population.population, self.params["tournament_size"])
                children = generate_offspring(parents, self.params["pc"], pm, all_pixels, self.params["zero_probability"])

                offsprings = Population(children, self.calculate_fitness, self.params["include_dist"])
                fe += len(offsprings.population)
                offsprings.evaluate(self.search_set, self.search_set2,self.modelTest, self.modelTest2)
                population.population.extend(offsprings.population)
                population.fronts = fast_nondominated_sort(population.population)
                front_num = 0
                new_solutions = []
                while len(new_solutions) + len(population.fronts[front_num]) <= self.params["population_size"]:
                    calculate_crowding_distance(population.fronts[front_num])
                    new_solutions.extend(population.fronts[front_num])
                    front_num += 1

                calculate_crowding_distance(population.fronts[front_num])
                population.fronts[front_num].sort(key=attrgetter("crowding_distance"), reverse=True)
                new_solutions.extend(population.fronts[front_num][0:self.params["population_size"] - len(new_solutions)])

                population = Population(new_solutions, self.calculate_fitness, self.params["include_dist"])

            population.fronts = fast_nondominated_sort(population.population)
            self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
            self.completion_procedure(population, self.calculate_fitness, fe, False)
        except:

            perturbed_noise = torch.randn_like(noise) * 0.1 
            perturbed_noise = noise + perturbed_noise  
            perturbed_noise = torch.clamp(perturbed_noise, -self.params["epsilon"], self.params["epsilon"]) 
            return perturbed_noise
        return population.fronts[0]

    def calculate_fitness(self, solution):
        f_adv = solution.data
        D_fadv1 = self.calculate_D(f_adv, self.sCentroids)
        D_fadv2 = self.calculate_D(f_adv, self.sCentroids2)
        S_fadv1 = self.calculate_S_on_dataset(f_adv, solution.true_label, self.search_set, self.model)
        S_fadv2 = self.calculate_S_on_dataset(f_adv, solution.true_label, self.search_set2, self.model2)
        fitness = np.exp(-(D_fadv1 + D_fadv2)) + (1 - (S_fadv1 + S_fadv2) / 2)
        solution.fitness = fitness
        return fitness

    def calculate_S_on_dataset(self, f_adv, y_true, dataset, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs + f_adv)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        S_fadv = (total - correct) / total
        return S_fadv