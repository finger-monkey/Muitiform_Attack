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
from reid.evaluators import extract_features
from reid.utils.meters import AverageMeter
import torchvision
import faiss

from torchvision import transforms

from MOAA.MOAA import Attack
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


    # tgtSet is transferbility attack model's dataset, 
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


def Multiform_attack(gradient_based_train_loader, search_set_loader, net, noise, epoch, optimizer,
              centroids, metaCentroids, normalize):
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
    for i, ((input, _, pid, _), (metaTest, _, _, _)) in enumerate(zip(gradient_based_train_loader, search_set_loader)):
        data_time.update(time.time() - end)
        model.zero_grad()
        input = input.cuda()
        metaTest = metaTest.cuda()


        with torch.no_grad():
            normInput = (input - mean) / std
            feature, _ = net(normInput)
            scores = centroids.mm(F.normalize(feature.t(), p=2, dim=0))

            _, ranks = torch.sort(scores, dim=0, descending=True)
            pos_i = ranks[0, :]
            neg_i = ranks[-1, :]
        neg_feature = centroids[neg_i, :]  
        pos_feature = centroids[pos_i, :]

        current_noise = noise
        current_noise = F.interpolate(
            current_noise.unsqueeze(0),
            mode=MODE, size=tuple(input.shape[-2:]), align_corners=True,
        ).squeeze()
        perturted_input = torch.clamp(input + current_noise, 0, 1)
        perturted_input_norm = (perturted_input - mean) / std
        perturbed_feature = net(perturted_input_norm)[0]

        optimizer.zero_grad()

        pair_loss = 10 * F.triplet_margin_loss(perturbed_feature, neg_feature, pos_feature, 0.5)


        pair_loss = pair_loss.view(1)

        loss = pair_loss


        grad = torch.autograd.grad(loss, noise, create_graph=True)[0]
        noiseOneStep = Update(noise, optimizer, grad, MAX_EPS)

  
        newNoise = F.interpolate(
            noiseOneStep.unsqueeze(0), mode=MODE,
            size=tuple(metaTest.shape[-2:]), align_corners=True,
        ).squeeze()


        if epoch % 3 == 0:
            search_noise = evolutionary_search(search_set,search_set2, modelTest,modelTest2,noise)
            newNoise = newNoise + search_noise

        with torch.no_grad():
            normMte = (metaTest - mean) / std
            mteFeat = net(normMte)[0]
            scores = metaCentroids.mm(F.normalize(mteFeat.t(), p=2, dim=0))

            metaLab = scores.max(0, keepdim=True)[1]
            _, ranks = torch.sort(scores, dim=0, descending=True)
            pos_i = ranks[0, :]
            neg_i = ranks[-1, :]
        neg_mte_feat = metaCentroids[neg_i, :]  
        pos_mte_feat = metaCentroids[pos_i, :]

        perMteInput = torch.clamp(metaTest + newNoise, 0, 1)
        normPerMteInput = (perMteInput - mean) / std
        normMteFeat = net(normPerMteInput)[0]

        lossTri = 10 * F.triplet_margin_loss(
            normMteFeat, neg_mte_feat, pos_mte_feat, 0.5
        )

        oneHotRealMeta = torch.zeros(scores.t().shape).cuda()
        oneHotRealMeta.scatter_(1, metaLab.view(-1, 1), float(1))


        finalLoss = lossTri  + pair_loss 

        finalLoss.backward()

        losses.update(pair_loss.item())
        optimizer.step()

        # measure elapsed time
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


# Define the evolutionary search method, using the Attack class in MOAA
def evolutionary_search(search_set,search_set2, modelTest,modelTest2,noise):

    params = {
        "population_size": 2,
        "num_generations": 150,
        "mutation_rate": 0.2,
        "crossover_rate": 0.3,
        "epsilon": 8 / 255.0,  
        "p_size": noise.size(),
        "x": None,  # input data
        "eps": 8 / 255.0,
        "zero_probability": 0.2,
        "pm": 0.1,
        "pop_size": 2,
        "iterations": 150,
        "pc": 0.3,
        "include_dist": True,
        "save_directory": "results",
        "tournament_size": 3,
        "max_dist": 1.0,
    }

    attack = Attack(params,search_set,search_set2,modelTest,modelTest2)
  
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
        # transforms.RandomGrayscale(p=0.5),
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


    # evaluate on test datasets
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
    parser.add_argument('--epoch', type=int, default=20)
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
    args = parser.parse_args()


    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # np.random.seed(0)
    # random.seed(0)
    
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
            checkTgt = checkTest['state_dict']
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
            modelTest.load_state_dict(checkTest2, strict=False)

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
    features = torch.stack([features[f] for f, _, _ in sourceSet.trainval])
    metaFeats, _ = extract_features(model, search_set, print_freq=10)
    metaFeats = torch.stack([metaFeats[f] for f, _, _ in mteSet.trainval])


    if args.source == "sysu":
        ncentroids = 395
    else:
        ncentroids = 206

    
    fDim = features.shape[1]
    cluster, metaClu = faiss.Kmeans(fDim, ncentroids, niter=20, gpu=True), \
                       faiss.Kmeans(fDim, ncentroids, niter=20, gpu=True)
    cluster.train(features.cpu().numpy())
    metaClu.train(metaFeats.cpu().numpy())

    centroids = torch.from_numpy(cluster.centroids).cuda().float()
    metaCentroids = torch.from_numpy(metaClu.centroids).cuda().float()
    del metaClu, cluster

    evaluator = Evaluator(modelTest, args.print_freq)
    evaSrc = Evaluator(model, args.print_freq)

    # universal noise
    noise = torch.zeros((3, args.height, args.width)).cuda()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    noise.requires_grad = True
    MAX_EPS = args.max_eps / 255.0

    optimizer = MI_SGD(
        [{"params": [noise], "lr": MAX_EPS / 10, "momentum": 1, "sign": True}],
        max_eps=MAX_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))


    import time

    for epoch in range(args.epoch):
        scheduler.step()
        begin_time = time.time()
        loss, noise = Multiform_attack(
            gradient_based_train, search_set, model, noise, epoch, optimizer,
            centroids, metaCentroids, normalize
        )

        testQImage = test(tgtSet, modelTest, noise, args, evaluator, epoch)

