import torch
import argparse
import sys
import os
from torch.utils.data import DataLoader

from reid import models
from torch.nn import functional as F
import os.path as osp
from reid import datasets

from reid.utils.data import transforms as T
from torchvision.transforms import Resize
from reid.utils.data.preprocessor import Preprocessor
from reid.evaluators import Evaluator
from torch.optim.optimizer import Optimizer, required
import random
import numpy as np
import math
import time
import json
from datetime import datetime
from reid.evaluators import extract_features
from reid.utils.meters import AverageMeter
import torchvision

from torchvision import transforms

from SparseEvolution.evolutionary_attack import SparseEvolutionAttack
import numpy as np
import argparse
import os

CHECK = 1e-5
SAT_MIN = 0.5
MODE = "bilinear"




def input(sourceName, mteName,mteName2, targetName, split_id, data_dir, height, width,
             batch_size, workers, combine):
    root = osp.join(data_dir, sourceName)
    rootMte = osp.join(data_dir, mteName)
    rootMte2 = osp.join(data_dir, mteName2)
    rootTgt = osp.join(data_dir, targetName)
    sourceSet = datasets.create(sourceName, root, num_val=0.1, split_id=split_id)
    mteSet = datasets.create(mteName, rootMte, num_val=0.1, split_id=split_id)
    mteSet2 = datasets.create(mteName2, rootMte2, num_val=0.1, split_id=split_id)
    tgtSet = datasets.create(targetName, rootTgt, num_val=0.1, split_id=split_id)
    num_classes = sourceSet.num_trainval_ids if combine else sourceSet.num_train_ids

    num_search = mteSet.num_trainval_ids if combine else mteSet.num_train_ids
    num_search2 = mteSet2.num_trainval_ids if combine else mteSet2.num_train_ids

    class_tgt = tgtSet.num_trainval_ids if combine else tgtSet.num_train_ids

    train_transformer = T.Compose([
        Resize((height, width)),
        transforms.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ])

    gradient_based_train = DataLoader(
        Preprocessor(sourceSet.trainval, root=sourceSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    
    search_set = DataLoader(
        Preprocessor(mteSet.trainval, root=mteSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


    search_set2 = DataLoader(
        Preprocessor(mteSet2.trainval, root=mteSet2.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


    
    return sourceSet, tgtSet, mteSet,mteSet2, num_classes,num_search,num_search2, class_tgt,  gradient_based_train, search_set,search_set2


def rescale_check(check, sat, sat_change, sat_min):
    return sat_change < check and sat > sat_min


class MI_SGD(Optimizer):
    def __init__(
            self, params, lr=required, momentum=0, dampening=0, weight_decay=0,
            nesterov=False, max_eps=10 / 255
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign=False,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MI_SGD, self).__init__(params, defaults)
        self.sat = 0
        self.sat_prev = 0
        self.max_eps = max_eps

    def __setstate__(self, state):
        super(MI_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def rescale(self, ):
        for group in self.param_groups:
            if not group["sign"]:
                continue
            for p in group["params"]:
                self.sat_prev = self.sat
                self.sat = (p.data.abs() >= self.max_eps).sum().item() / p.data.numel()
                sat_change = abs(self.sat - self.sat_prev)
                if rescale_check(CHECK, self.sat, sat_change, SAT_MIN):
                    print('rescaled')
                    p.data = p.data / 2

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["sign"]:
                    d_p = d_p / (d_p.norm(1) + 1e-12)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if group["sign"]:
                    p.data.add_(-group["lr"], d_p.sign())
                    p.data = torch.clamp(p.data, -self.max_eps, self.max_eps)
                else:
                    p.data.add_(-group["lr"], d_p)

        return loss


def Update(noiseData, optimizer, gradInfo, max_eps):
    weight_decay = optimizer.param_groups[0]["weight_decay"]
    momentum = optimizer.param_groups[0]["momentum"]
    dampening = optimizer.param_groups[0]["dampening"]
    nesterov = optimizer.param_groups[0]["nesterov"]
    lr = optimizer.param_groups[0]["lr"]

    d_p = gradInfo
    if optimizer.param_groups[0]["sign"]:
        d_p = d_p / (d_p.norm(1) + 1e-12)
    if weight_decay != 0:
        d_p.add_(weight_decay, noiseData)
    if momentum != 0:
        param_state = optimizer.state[noiseData]
        if "momentum_buffer" not in param_state:
            buf = param_state["momentum_buffer"] = torch.zeros_like(noiseData.data)
            buf = buf * momentum + d_p
        else:
            buf = param_state["momentum_buffer"]
            buf = buf * momentum + (1 - dampening) * d_p
        if nesterov:
            d_p = d_p + momentum * buf
        else:
            d_p = buf

        if optimizer.param_groups[0]["sign"]:
            noiseData = noiseData - lr * d_p.sign()
            noiseData = torch.clamp(noiseData, -max_eps, max_eps)
        else:
            noiseData = noiseData - lr * d_p.sign()
    return noiseData


def _mahalanobis_triplet(features, labels, centroids, centroid_labels, covariance_inv, margin=0.3):
    
    delta = features[:, None, :] - centroids[None, :, :]
    distances = torch.einsum("bcd,de,bce->bc", delta, covariance_inv, delta)
    matches = labels[:, None].eq(centroid_labels[None, :])
    if not bool(matches.any(1).all()):
        missing = labels[~matches.any(1)].unique().tolist()
        raise ValueError("missing class centroids for labels {}".format(missing))
    positive = distances.masked_fill(~matches, float("inf")).min(1).values
    negative = distances.masked_fill(matches, float("inf")).min(1).values
    return F.relu(negative - positive + margin).mean()


def Multiform_attack(gradient_based_train_loader, search_set_loader, net, noise, epoch, optimizer,
              centroids, metaCentroids, normalize, covariance_inv=None,
              meta_covariance_inv=None, centroid_labels=None, meta_centroid_labels=None,
              use_paper_objectives=True, run_evolution=False,
              search_set_loader2=None, evolution_models=None, max_steps=None):
    
    
    

    global args
    noise.requires_grad = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mean = torch.Tensor(normalize.mean).view(1, 3, 1, 1).cuda()
    std = torch.Tensor(normalize.std).view(1, 3, 1, 1).cuda()

    net.eval()

    end = time.time()
    optimizer.zero_grad()
    optimizer.rescale()
    for i, ((input, _, pid, _), (metaTest, _, meta_pid, _)) in enumerate(zip(gradient_based_train_loader, search_set_loader)):
        if max_steps is not None and i >= max_steps:
            break
        data_time.update(time.time() - end)
        net.zero_grad()
        input = input.cuda()
        metaTest = metaTest.cuda()


        with torch.no_grad():
            normInput = (input - mean) / std
            feature, _ = net(normInput)
            if not use_paper_objectives:
                scores = centroids.mm(F.normalize(feature.t(), p=2, dim=0))
                _, ranks = torch.sort(scores, dim=0, descending=True)
                pos_i, neg_i = ranks[0, :], ranks[-1, :]
                neg_feature, pos_feature = centroids[neg_i, :], centroids[pos_i, :]

        current_noise = noise
        current_noise = F.interpolate(
            current_noise.unsqueeze(0),
            mode=MODE, size=tuple(input.shape[-2:]), align_corners=True,
        ).squeeze()
        perturted_input = torch.clamp(input + current_noise, 0, 1)
        perturted_input_norm = (perturted_input - mean) / std
        perturbed_feature = net(perturted_input_norm)[0]

        optimizer.zero_grad()

        
        
        if use_paper_objectives:
            pair_loss = _mahalanobis_triplet(
                perturbed_feature, pid.cuda(), centroids, centroid_labels,
                covariance_inv, margin=0.3)
        else:
            pair_loss = 10 * F.triplet_margin_loss(perturbed_feature, neg_feature, pos_feature, 0.3)


        pair_loss = pair_loss.view(1)

        loss = pair_loss


        grad = torch.autograd.grad(loss, noise, create_graph=True)[0]
        noiseOneStep = Update(noise, optimizer, grad, MAX_EPS)

  
        newNoise = F.interpolate(
            noiseOneStep.unsqueeze(0), mode=MODE,
            size=tuple(metaTest.shape[-2:]), align_corners=True,
        ).squeeze()


        if run_evolution:
            if search_set_loader2 is None or evolution_models is None:
                raise ValueError("evolution datasets/models must be passed explicitly")
            search_noise = evolutionary_search(
                search_set_loader, search_set_loader2,
                evolution_models[0], evolution_models[1], noise,
                use_attention_filtering=True)
            newNoise = torch.clamp(newNoise + search_noise, -MAX_EPS, MAX_EPS)

        with torch.no_grad():
            normMte = (metaTest - mean) / std
            mteFeat = net(normMte)[0]
            if not use_paper_objectives:
                scores = metaCentroids.mm(F.normalize(mteFeat.t(), p=2, dim=0))
                metaLab = scores.max(0, keepdim=True)[1]
                _, ranks = torch.sort(scores, dim=0, descending=True)
                pos_i, neg_i = ranks[0, :], ranks[-1, :]
                neg_mte_feat, pos_mte_feat = metaCentroids[neg_i, :], metaCentroids[pos_i, :]

        perMteInput = torch.clamp(metaTest + newNoise, 0, 1)
        normPerMteInput = (perMteInput - mean) / std
        normMteFeat = net(normPerMteInput)[0]

        if use_paper_objectives:
            lossTri = _mahalanobis_triplet(
                normMteFeat, meta_pid.cuda(), metaCentroids, meta_centroid_labels,
                meta_covariance_inv, margin=0.3)
        else:
            lossTri = 10 * F.triplet_margin_loss(normMteFeat, neg_mte_feat, pos_mte_feat, 0.3)


        finalLoss = lossTri  + pair_loss 

        finalLoss.backward()

        losses.update(pair_loss.item())
        optimizer.step()

        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                ">> Train: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Noise l2: {noise:.4f}".format(
                    epoch + 1,
                    i, len(gradient_based_train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses, lossTri=lossTri.item(),
                    noise=noise.norm(),
                )
            )

    noise.requires_grad = False
    print(f"Train {epoch}: Loss: {losses.avg}")
    return losses.avg, noise



def evolutionary_search(search_set,search_set2, modelTest,modelTest2,noise, use_attention_filtering=True,
                        strict_evolution=True):
    
    
    

    
    
    
    params = {
        "population_size": 4,           
        "epsilon": 8 / 255.0,
        "p_size": 8 / 255.0,
        "x": None,  
        "eps": 20,
        "zero_probability": 0.2,
        "pm": 0.2,
        "iterations": 200,
        "pc": 0.3,
        "include_dist": True,
        "save_directory": osp.join(getattr(args, "output_dir", "results"), "evolution.npy"),
        "tournament_size": 3,
        "max_dist": 1.0,
        "use_attention_filtering": use_attention_filtering,  
        "attention_filter_size": 2,
        "evaluation_batches": 1,
        "evaluation_batch_size": 64,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        
        
        "report_fallback": False,
        "strict_evolution": strict_evolution,
        "use_full_covariance": True,
        "covariance_regularization": 1e-4,
        "gallery_batches": 4,
    }

    attack = SparseEvolutionAttack(params,search_set,search_set2,modelTest,modelTest2)

    search_noise = attack.attack(noise)
    return search_noise


def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def compute_class_statistics(features, samples, regularization=1e-4):
    
    labels = torch.as_tensor([pid for _, pid, _ in samples], device=features.device)
    unique_labels = labels.unique(sorted=True)
    centroids = torch.stack([features[labels == label].mean(0) for label in unique_labels])
    centered = features - features.mean(0, keepdim=True)
    denominator = max(features.shape[0] - 1, 1)
    covariance = centered.t().matmul(centered) / denominator
    covariance = covariance + regularization * torch.eye(
        covariance.shape[0], device=features.device, dtype=features.dtype)
    return centroids, unique_labels, torch.linalg.pinv(covariance)


def write_run_manifest(args, extra=None):
    
    directory = getattr(args, "output_dir", "results")
    os.makedirs(directory, exist_ok=True)
    payload = vars(args).copy()
    payload.update(extra or {})
    payload["created_at"] = datetime.now().astimezone().isoformat()
    payload["torch_version"] = torch.__version__
    payload["cuda_version"] = torch.version.cuda
    path = osp.join(directory, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    return path


def test(dataset, net, noise, args, evaluator, epoch):
    print(">> Evaluating network on test datasets...")

    net = net.cuda()
    net.eval()
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def add_noise(img):
        n = noise.cpu()
        img = img.cpu()
        n = F.interpolate(
            n.unsqueeze(0), mode=MODE, size=tuple(img.shape[-2:]), align_corners=True
        ).squeeze()
        return torch.clamp(img + n, 0, 1)

    query_trans = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(), T.Lambda(lambda img: add_noise(img)),
        
        normalize
    ])
    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        transforms.RandomGrayscale(p=1),
        T.ToTensor(), normalize
    ])
    query_loader = DataLoader(
        Preprocessor(dataset.query, root=dataset.images_dir, transform=query_trans),
        batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True
    )
    qFeats, gFeats, testQImage, qnames, gnames = [], [], [], [], []
    with torch.no_grad():
        for (inputs, qname, _, _) in query_loader:
            inputs = inputs.cuda()
            qFeats.append(net(inputs)[0])
            qnames.extend(qname)
        qFeats = torch.cat(qFeats, 0)
        for (inputs, gname, _, _) in gallery_loader:
            inputs = inputs.cuda()
            gFeats.append(net(inputs)[0])
            gnames.extend(gname)
        gFeats = torch.cat(gFeats, 0)
    distMat = calDist(qFeats, gFeats)


    
    evaluator.evaMat(distMat, dataset.query, dataset.gallery)
    return testQImage




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True,
                        help='path to reid dataset')
    parser.add_argument('-s', '--source', type=str, default='sysu_v2',
                        choices=datasets.names())
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('-t', '--target', type=str, default='sysu_v2',
                        choices=datasets.names())
    parser.add_argument('-m', '--mte', type=str, default='sysu_v2',
                        choices=datasets.names())
    parser.add_argument('-m2', '--mte2', type=str, default='sysu_v2',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=50, required=True,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--resumeSearchTgt', type=str, default='', metavar='PATH')
    parser.add_argument('--resumeSearchTgt2', type=str, default='', metavar='PATH')
    parser.add_argument('--resumeTgt', type=str, default='', metavar='PATH')   

    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--combine_trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument("--max-eps", default=8, type=int, help="max eps")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--stage1-steps', default=0, type=int,
                        help='maximum gradient batches per epoch; 0 uses the full loader')
    parser.add_argument('--output-dir', default='results', type=str)
    parser.add_argument('--legacy-objective', action='store_true',
                        help='use the original cosine/Euclidean approximation')
    parser.add_argument('--disable-attention-filter', action='store_true')
    parser.add_argument('--enable-sparse-stage', action='store_true',
                        help='run the optional sparse evolutionary refinement after training')
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    sourceSet, tgtSet, mteSet,mteSet2, num_classes,num_search,num_search2, class_tgt, gradient_based_train, search_set,search_set2 = \
        input(args.source, args.mte, args.mte2, args.target,
                args.split, args.data, args.height,
                args.width, args.batch_size, 8, args.combine_trainval)


    model = models.create(args.arch, pretrained=True, num_classes=num_classes)
    modelTest = models.create(args.arch, pretrained=True, num_classes=num_search)
    modelTest2 = models.create(args.arch, pretrained=True, num_classes=num_search2)
    modelTarget = models.create(args.arch, pretrained=True, num_classes=class_tgt)
    if args.resume:
        checkpoint = torch.load(args.resume)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        try:
            model.load_state_dict(checkpoint)
        except:
            allNames = list(checkpoint.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkpoint[name]
            model.load_state_dict(checkpoint, strict=False)

        checkTest = torch.load(args.resumeSearchTgt)
        if 'state_dict' in checkTest.keys():
            checkTest = checkTest['state_dict']
        try:
            modelTest.load_state_dict(checkTest)
        except:
            allNames = list(checkTest.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkTest[name]
            modelTest.load_state_dict(checkTest, strict=False)

        checkTest2 = torch.load(args.resumeSearchTgt2)
        if 'state_dict' in checkTest2.keys():
            checkTest2 = checkTest2['state_dict']
        try:
            modelTest2.load_state_dict(checkTest2)
        except:
            allNames = list(checkTest2.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkTest2[name]
            modelTest2.load_state_dict(checkTest2, strict=False)

        checkTarget = torch.load(args.resumeTgt)
        if 'state_dict' in checkTarget.keys():
            checkTarget = checkTarget['state_dict']
        try:
            modelTarget.load_state_dict(checkTarget)
        except:
            allNames = list(checkTarget.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkTarget[name]
            modelTarget.load_state_dict(checkTarget, strict=False)


    model.eval()
    modelTest.eval()
    modelTest2.eval()
    modelTarget.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        modelTest = modelTest.cuda()
        modelTest2 = modelTest2.cuda()
        modelTarget = modelTarget.cuda()

    features, _ = extract_features(model,  gradient_based_train, print_freq=10)
    features = torch.stack([features[f] for f, _, _ in sourceSet.trainval]).cuda()
    metaFeats, _ = extract_features(model, search_set, print_freq=10)
    metaFeats = torch.stack([metaFeats[f] for f, _, _ in mteSet.trainval]).cuda()


    centroids, centroid_labels, covariance_inv = compute_class_statistics(
        features, sourceSet.trainval, regularization=1e-4)
    metaCentroids, meta_centroid_labels, meta_covariance_inv = compute_class_statistics(
        metaFeats, mteSet.trainval, regularization=1e-4)
    print("[MAHALANOBIS] identity centroids and regularized covariance statistics prepared")

    
    noise = torch.zeros((3, args.height, args.width)).cuda()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    noise.requires_grad = True
    MAX_EPS = args.max_eps / 255.0
    manifest = write_run_manifest(args, {
        "paper_objectives": not args.legacy_objective,
        "attention_filtering": not args.disable_attention_filter,
        "sparse_stage_enabled": args.enable_sparse_stage,
        "covariance_regularization": 1e-4,
    })
    print("[REPRODUCIBILITY] run manifest:", manifest)

    
    
    optimizer = MI_SGD(
        [{"params": [noise], "lr": MAX_EPS / 10, "momentum": 0.9, "sign": True}],
        max_eps=MAX_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))


    import time

    for epoch in range(args.epoch):
        scheduler.step()
        begin_time = time.time()
        loss, noise = Multiform_attack(
            gradient_based_train, search_set, model, noise, epoch, optimizer,
            centroids, metaCentroids, normalize,
            covariance_inv=covariance_inv,
            meta_covariance_inv=meta_covariance_inv,
            centroid_labels=centroid_labels,
            meta_centroid_labels=meta_centroid_labels,
            use_paper_objectives=not args.legacy_objective,
            run_evolution=False,
            search_set_loader2=search_set2,
            evolution_models=(modelTest, modelTest2),
            max_steps=args.stage1_steps if args.stage1_steps > 0 else None,
        )

        
        
        
        try:
            if args.enable_sparse_stage:
                search_noise = evolutionary_search(
                    search_set, search_set2, modelTest, modelTest2, noise.detach(),
                    use_attention_filtering=not args.disable_attention_filter,
                    strict_evolution=True)
                noise = torch.clamp(noise + search_noise, -MAX_EPS, MAX_EPS).detach()
            else:
                search_noise = None
        except Exception:
            
            
            search_noise = None

        
        if (epoch + 1) % 10 == 0:
            testQImage = test(
                tgtSet, modelTarget, noise, args,
                Evaluator(modelTarget, args.print_freq), epoch)
