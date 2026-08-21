import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from SparseEvolution.operators import generate_candidates
from SparseEvolution.sparse_candidate import SparseCandidate, pareto_sort, compute_crowding_distance, select_parents
from operator import attrgetter
from torch.utils.data import DataLoader



class EvolutionaryPopulation:
    def __init__(self, solutions: list, loss_function, include_dist):
        self.population = solutions
        self.fronts = None
        self.loss_function = loss_function
        self.include_dist = include_dist

    def evaluate(self, search_set, search_set2, model1, model2):
        for pi in self.population:
            
            
            pi.evaluate(self.loss_function, self.include_dist)

    def find_adv_solns(self, max_dist):
        adv_solns = []
        for pi in self.population:
            if pi.is_adversarial and pi.fitnesses[1] <= max_dist:
                adv_solns.append(pi)

        return adv_solns

class SparseEvolutionAttack:
    def __init__(self, params,search_set, search_set2, modelTest, modelTest2):
        self.params = params
        self.fitness = []
        self.data = []
        self.search_set = getattr(search_set, "dataset", search_set)
        self.search_set2 = getattr(search_set2, "dataset", search_set2)
        self.modelTest = modelTest
        self.modelTest2 = modelTest2
        self.device = next(modelTest.parameters()).device
        self.statistics = {}
        self._retrieval_cache = {}
        self.mean = torch.as_tensor(params.get("normalize_mean", [0.485, 0.456, 0.406]), device=self.device).view(1, 3, 1, 1)
        self.std = torch.as_tensor(params.get("normalize_std", [0.229, 0.224, 0.225]), device=self.device).view(1, 3, 1, 1)

    def _model_outputs(self, model, inputs):
        outputs = model(inputs)
        if isinstance(outputs, (tuple, list)):
            tensors = [value for value in outputs if torch.is_tensor(value)]
            if not tensors:
                raise TypeError("model returned no tensor outputs")
            feature = tensors[0]
            logits = tensors[-1] if len(tensors) > 1 and tensors[-1].ndim == 2 else None
            return feature, logits
        return outputs, None

    def _prepare_retrieval_cache(self, dataset, model):
        key = (id(dataset), id(model))
        if key in self._retrieval_cache:
            return self._retrieval_cache[key]
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(DataLoader(
                    dataset, batch_size=self.params.get("evaluation_batch_size", 64),
                    shuffle=False, num_workers=0)):
                inputs = batch[0].to(self.device)
                feature, _ = self._model_outputs(model, (inputs - self.mean) / self.std)
                features.append(F.normalize(feature.flatten(1), dim=1))
                labels.append(batch[2].to(self.device))
                if batch_index + 1 >= self.params.get("gallery_batches", 4):
                    break
        if not features:
            raise ValueError("empty evolutionary retrieval dataset")
        gallery_features, gallery_labels = torch.cat(features), torch.cat(labels)
        centered = gallery_features - gallery_features.mean(0)
        covariance = centered.T @ centered / max(gallery_features.shape[0] - 1, 1)
        covariance = covariance + self.params.get("covariance_regularization", 1e-4) * torch.eye(
            covariance.shape[0], device=self.device, dtype=gallery_features.dtype)
        inverse = torch.linalg.pinv(covariance)
        centers = {
            int(label.item()): gallery_features[gallery_labels == label].mean(0)
            for label in gallery_labels.unique()
        }
        result = (gallery_features, gallery_labels, inverse, centers)
        self._retrieval_cache[key] = result
        return result

    def _dataset_objectives(self, residual, dataset, model):
        clean_features, adversarial_features = [], []
        correct = total = 0
        gallery_features, gallery_labels, covariance_inverse, class_center_map = \
            self._prepare_retrieval_cache(dataset, model)
        model.eval()
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=self.params.get("evaluation_batch_size", 64), shuffle=False, num_workers=0):
                inputs, labels = batch[0].to(self.device), batch[2].to(self.device)
                clean, _ = self._model_outputs(model, (inputs - self.mean) / self.std)
                adversarial, _ = self._model_outputs(model, (torch.clamp(inputs + residual, 0, 1) - self.mean) / self.std)
                clean_features.append(clean.flatten(1)); adversarial_features.append(adversarial.flatten(1))
                query = F.normalize(adversarial.flatten(1), dim=1)
                distances = torch.cdist(query, gallery_features)
                predicted_labels = gallery_labels[distances.argmin(1)]
                total += labels.numel(); correct += (predicted_labels == labels).sum().item()
                if len(clean_features) >= self.params.get("evaluation_batches", 1): break
        if not clean_features:
            return 0.0, 1.0
        clean = torch.cat(clean_features); adversarial = torch.cat(adversarial_features)
        variance = clean.var(0, unbiased=False).clamp_min(1e-4)
        deviation = -((adversarial - clean).square() / variance).sum(1).mean().item()
        if self.params.get("use_full_covariance", False):
            try:
                class_centers = torch.stack([
                    class_center_map[int(label.item())] for label in labels
                ])
                delta = F.normalize(adversarial, dim=1) - class_centers
                distance = torch.einsum("nd,de,ne->n", delta, covariance_inverse, delta).mean()
                deviation = float(torch.exp(-distance.clamp(max=80)).item())
            except Exception as exc:
                if self.params.get("report_fallback", False):
                    print(f"[SparseEvolution] covariance objective fallback: {type(exc).__name__}: {exc}")
        return deviation, correct / total if total else 1.0

    def filter_by_attention_shift(self, candidates, keep_count):
        if not self.params.get("use_attention_filtering", False) or len(candidates) <= keep_count:
            return candidates
        try:
            handle = None
            batch = next(iter(DataLoader(self.search_set, batch_size=1, shuffle=False, num_workers=0)))
            image = batch[0].to(self.device)
            layer = next((module for module in reversed(list(self.modelTest.modules()))
                          if isinstance(module, nn.Conv2d)), None)
            if layer is None:
                raise RuntimeError("model has no convolutional layer for a spatial attention map")
            activation = {}
            handle = layer.register_forward_hook(lambda _m, _i, output: activation.update(value=output))
            def attention(input_tensor):
                activation.clear()
                self._model_outputs(self.modelTest, input_tensor)
                value = activation["value"]
                value = value.abs().mean(1)
                return value / value.sum((1, 2), keepdim=True).clamp_min(1e-8)
            with torch.no_grad():
                clean_input = (torch.clamp(image + self.noise, 0, 1) - self.mean) / self.std
                clean_attention = attention(clean_input).clone()
            scored = []
            for candidate in candidates:
                adversarial = candidate.build_adversarial()
                with torch.no_grad():
                    adv_input = (torch.clamp(image + adversarial, 0, 1) - self.mean) / self.std
                    adv_attention = attention(adv_input)
                score = float((clean_attention - adv_attention).abs().sum().item())
                scored.append((score, candidate))
            handle.remove()
            handle = None
            scored.sort(key=lambda item: item[0], reverse=True)
            return [candidate for _, candidate in scored[:keep_count]]
        except Exception as exc:
            if handle is not None:
                handle.remove()
            if self.params.get("report_fallback", False):
                print(f"[SparseEvolution] attention filtering fallback: {type(exc).__name__}: {exc}")
            return candidates

    def attention_shift_score(self, clean, adversarial):
        clean = clean.flatten(1)
        adversarial = adversarial.flatten(1)
        clean = clean.abs() / clean.abs().sum(1, keepdim=True).clamp_min(1e-8)
        adversarial = adversarial.abs() / adversarial.abs().sum(1, keepdim=True).clamp_min(1e-8)
        return (clean - adversarial).abs().sum(1)

    def mahalanobis_distance(self, features, centers):
        features = features.reshape(features.shape[0], -1)
        centers = centers.reshape(centers.shape[0], -1)
        covariance = torch.cov(features.T) if features.shape[0] > 1 else torch.eye(features.shape[1], device=features.device)
        covariance = covariance + 1e-4 * torch.eye(covariance.shape[0], device=features.device)
        inverse = torch.linalg.pinv(covariance)
        delta = features[:, None, :] - centers[None, :, :]
        return torch.einsum("ncd,de,nce->nc", delta, inverse, delta).mean()


    def completion_procedure(self, population, loss_function, fe, success):
        adversarial_labels = []
        for soln in population.fronts[0]:
            adversarial_labels.append(bool(soln.is_adversarial))

        d = {"front0_imgs": [soln.build_adversarial() for soln in population.fronts[0]],
             "queries": fe,
             "true_label": None,
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
        if noise.dim() == 4:
            noise = noise[0]
        if noise.dim() != 3:
            raise ValueError("noise must be a 3-D image tensor")
        h, w = noise.shape[-2:]
        pm = self.params["pm"]
        n_pixels = h * w
        all_pixels = np.arange(n_pixels)
        ones_prob = (1 - self.params["zero_probability"]) / 2
        try:
            pixel_count = min(int(self.params.get("eps", 20)), n_pixels)
            init_solutions = [SparseCandidate(np.random.choice(all_pixels, size=pixel_count, replace=False),
                                       np.random.choice([-1, 1, 0], size=(pixel_count, 3),
                                                        p=(ones_prob, ones_prob, self.params["zero_probability"])),
                                       noise.clone(), self.params["p_size"]) for _ in range(self.params["population_size"])]
            for solution in init_solutions:
                solution.repair(self.params.get("epsilon", 8 / 255.0), pixel_count)

            population = EvolutionaryPopulation(init_solutions, self.calculate_fitness, self.params["include_dist"])
            population.evaluate(self.search_set, self.search_set2, self.modelTest, self.modelTest2)
            fe = len(population.population)
            
            for it in range(1, self.params["iterations"]):
                pm = self.params["pm"]
                population.fronts = pareto_sort(population.population)

                adv_solns = population.find_adv_solns(self.params["max_dist"])
                if len(adv_solns) > 0:
                    self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
                    self.completion_procedure(population, self.calculate_fitness, fe, True)
                    return population.fronts[0][0].build_adversarial() - noise

                self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)

                for front in population.fronts:
                    compute_crowding_distance(front)
                parents = select_parents(population.population, self.params["tournament_size"])
                children = generate_candidates(
                    parents, self.params["pc"], pm, all_pixels, self.params["zero_probability"],
                    self.params.get("epsilon", 8 / 255.0), pixel_count)
                children = self.filter_by_attention_shift(
                    children, self.params.get("attention_filter_size", self.params["population_size"]))

                offsprings = EvolutionaryPopulation(children, self.calculate_fitness, self.params["include_dist"])
                fe += len(offsprings.population)
                offsprings.evaluate(self.search_set, self.search_set2,self.modelTest, self.modelTest2)
                population.population.extend(offsprings.population)
                population.fronts = pareto_sort(population.population)
                front_num = 0
                new_solutions = []
                while front_num < len(population.fronts) and len(population.fronts[front_num]) > 0 and len(new_solutions) + len(population.fronts[front_num]) <= self.params["population_size"]:
                    compute_crowding_distance(population.fronts[front_num])
                    new_solutions.extend(population.fronts[front_num])
                    front_num += 1

                if front_num < len(population.fronts) and population.fronts[front_num]:
                    compute_crowding_distance(population.fronts[front_num])
                    population.fronts[front_num].sort(key=attrgetter("crowding_distance"), reverse=True)
                    new_solutions.extend(population.fronts[front_num][0:self.params["population_size"] - len(new_solutions)])

                population = EvolutionaryPopulation(new_solutions, self.calculate_fitness, self.params["include_dist"])

            population.fronts = pareto_sort(population.population)
            self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
            self.completion_procedure(population, self.calculate_fitness, fe, False)
        except Exception as exc:
            self.statistics["fallback"] = {"type": type(exc).__name__, "message": str(exc)}
            if self.params.get("report_fallback", False):
                print(f"[SparseEvolution] fallback to baseline perturbation: {type(exc).__name__}: {exc}")
            if self.params.get("strict_evolution", False):
                raise
            fallback = noise + torch.randn_like(noise) * 0.1
            return torch.clamp(fallback, -self.params.get("epsilon", 8 / 255.0), self.params.get("epsilon", 8 / 255.0)) - noise
        return population.fronts[0][0].build_adversarial() - noise

    def calculate_fitness(self, solution):
        f_adv = solution.build_adversarial()
        solution.repair(self.params.get("epsilon", 8 / 255.0), self.params.get("eps", 20))
        f_adv = solution.build_adversarial()
        residual_norm = float(np.sqrt(solution.squared_l2_distance(f_adv)))
        metric_deviations, correct_rates = [], []
        for dataset, model in ((self.search_set, self.modelTest), (self.search_set2, self.modelTest2)):
            deviation, correct_rate = self._dataset_objectives(f_adv, dataset, model)
            metric_deviations.append(deviation); correct_rates.append(correct_rate)
        retrieval_objective = float(np.mean(correct_rates))
        feature_objective = float(np.mean(metric_deviations))
        solution.fitness = feature_objective
        return [retrieval_objective < 0.5, feature_objective, retrieval_objective, residual_norm]
