import torch
from timm.utils import reduce_tensor

from utils.misc import AverageMeter
from utils.model import set_head_training
import torch.distributed as dist

@torch.no_grad()
def validate(C_, args, model, loader):
    if C_.DOWNSTREAM_EVAL == 'linear':
        set_head_training(model, False)
    else:
        model.eval()
    acc = AverageMeter()
    for batch in loader:
        img = batch[0].cuda(non_blocking=True)
        tgt = batch[1].cuda(non_blocking=True)
        preds = model(img).argmax(1)

        torch.cuda.synchronize()
        curr_acc = (preds == tgt).sum()/float(len(tgt))

        if args.distributed:
            reduced_curr_acc = reduce_tensor(curr_acc, args.world_size)
        else:
            reduced_curr_acc = curr_acc

        acc.update(reduced_curr_acc.item(), len(tgt))
    if C_.DOWNSTREAM_EVAL == 'linear':
        set_head_training(model, True)
    else:
        model.train()
    return acc.avg

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda(non_blocking=True)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        # top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    # top5 = top5 * 100.0 / total
    return top1


@torch.no_grad()
def extract_features(args, model, data_loader, use_cuda=True):
    features = None
    labels = None
    for data in data_loader:
        samples = data[0].cuda(non_blocking=True)
        tgts = data[1].cuda(non_blocking=True)
        index = data[2].cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if args.rank == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            labels = torch.zeros(len(data_loader.dataset), dtype=torch.long)
            if use_cuda:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        if args.distributed:
            idx_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            idx_l = list(idx_all.unbind(0))
            idx_all_reduce = torch.distributed.all_gather(idx_l, index, async_op=True)
            idx_all_reduce.wait()
            index_all = torch.cat(idx_l)

            y_all = torch.empty(dist.get_world_size(), tgts.size(0), dtype=tgts.dtype, device=tgts.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, tgts, async_op=True)
            y_all_reduce.wait()
            labels_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()
            output_all = torch.cat(output_l)
        else:
            index_all = index
            output_all = feats
            labels_all = tgts

        # update storage feature matrix
        if args.rank == 0:
            if use_cuda:
                features.index_copy_(0, index_all, output_all)
                labels.index_copy_(0, index_all, labels_all)
            else:
                index_all = index_all.cpu()
                features.index_copy_(0, index_all, output_all.cpu())
                labels.index_copy_(0, index_all, labels_all.cpu())
    return features, labels