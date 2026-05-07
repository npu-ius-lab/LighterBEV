import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler,Subset
import h5py
from sklearn.decomposition import PCA
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import faiss
import kitti_dataset as kitti_dataset
import nclt_dataset 
import csv





def get_args():
    parser = argparse.ArgumentParser(description='LighterBEV')
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    parser.add_argument('--batchSize', type=int, default=4, 
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--cacheBatchSize', type=int, default=64, help='Batch size for caching and testing')
    parser.add_argument('--nEpochs', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    parser.add_argument('--lrStep', type=float, default=10, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
    parser.add_argument('--cachePath', type=str, default='./cache/', help='Path to save cache to.')
    parser.add_argument('--load_from', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--ckpt', type=str, default='best', 
            help='Load_from from latest or best checkpoint.', choices=['latest', 'best'])


    opt = parser.parse_args()
    return opt

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, anchor, positive, negative):
        
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        
        loss = F.relu(pos_dist-neg_dist + self.margin)
        return loss#.mean()

def train_epoch(epoch, model, train_set):
    
    epoch_loss = 0

    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    criterion = TripletLoss().to(device)
    
    
    model.eval()
    

    if epoch >= 1:
        print('====> Building Cache for Hard Mining')
        train_set.mining=False
        train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5: 
            if use_pca:
                pool_size = pca_dim * num_clusters
            else:
                pool_size = model.global_feat_dim
            h5feat = h5.create_dataset("features", 
                    [len(train_set), pool_size], 
                    dtype=np.float32)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=False, 
                collate_fn=kitti_dataset.collate_fn)
            with torch.no_grad():
                for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):
                    
                    query = query.to(device)
                    _, _, global_descs = model(query)
                    h5feat[indices, :] = global_descs.detach().cpu().numpy()
        train_set.mining=True
        train_set.refreshCache()
        
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=True, 
                collate_fn=kitti_dataset.collate_fn)
    
    model.train()

    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader):

        B, C, H, W = query.shape
        input = torch.cat([query, positives, negatives])

        input = input.to(device)
        
        _, _, global_descs = model(input)

        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])
        

        optimizer.zero_grad()

        # no need to train the kps feature
        loss = 0
        num_negs = negatives.shape[0] // B
        for i in range(len(global_descs_Q)):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs*i:num_negs*(i+1)]))
            loss += max_loss
        
        loss /= opt.batchSize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                n_batches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, 
                    ((epoch-1) * n_batches) + iteration)
            

    optimizer.zero_grad()    
    avg_loss = epoch_loss / n_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def infer(eval_set, return_local_feats=False,return_angles = False):
    test_data_loader = DataLoader(
        dataset=eval_set,
        num_workers=opt.threads,
        batch_size=opt.cacheBatchSize,
        shuffle=False
    )

    model.eval()
    model.to('cuda')
    num_samples = len(eval_set)  # 数据集总样本数

    # 初始化输出数组
    if return_local_feats:
        all_local_feats = np.zeros(shape=(num_samples, pca_dim, 200, 200), dtype=np.float32)
        all_global_descs = np.zeros(shape=(num_samples, pca_dim * num_clusters), dtype=np.float32)
        all_rotated_angles = np.zeros(shape=(num_samples,), dtype=np.int32)
        memory_GB = all_local_feats.nbytes / (1024 ** 3)
        print(f"local_feats occupies {memory_GB:.2f} GB")
    else:
        all_global_descs = np.zeros(shape=(num_samples, pca_dim * num_clusters), dtype=np.float32)

    # 初始化当前样本索引
    current_index = 0

    with torch.no_grad():
        for imgs, angle in tqdm(test_data_loader):
            imgs = imgs.to('cuda')
            _, local_feat, global_desc = model(imgs)

            # 获取当前 batch 的大小（最后一个 batch 可能小于 batch_size）
            batch_size_current = imgs.size(0)

            # 填充全局描述子
            all_global_descs[current_index:current_index + batch_size_current, :] = global_desc.detach().cpu().numpy()

            # 填充局部特征（如果需要）
            if return_local_feats:
                all_local_feats[current_index:current_index + batch_size_current, :, :, :] = local_feat.detach().cpu().numpy()
                all_rotated_angles[current_index:current_index + batch_size_current] = angle
            # 更新当前索引
            current_index += batch_size_current

    # 根据是否需要返回局部特征返回结果
    if return_local_feats and return_angles:
        return all_local_feats, all_global_descs, all_rotated_angles
    elif return_local_feats:
        return all_local_feats, all_global_descs
    else:
        return all_global_descs


def testPCA(eval_set, epoch=0, write_tboard=False):
    # TODO global descriptor PCA for faster inference speed
    pass
    # return recalls


def getClusters(cluster_set):
    n_descriptors = 10000
    n_per_image = 25
    n_im = ceil(n_descriptors/n_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                sampler=sampler)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    initcache = join(opt.cachePath, 'desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            all_feats = h5.create_dataset("descriptors", 
                        [n_descriptors, local_dim], 
                        dtype=np.float32)
            
            for iteration, (query, _, _, _) in enumerate(data_loader, 1):
                query = query.to(device)
                local_feat, _, _ = model(query)
                local_feat = local_feat.view(query.size(0), local_dim, -1).permute(0, 2, 1)
                
                batchix = (iteration-1)*opt.cacheBatchSize*n_per_image
                for ix in range(local_feat.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
                    startix = batchix + ix*n_per_image
                    all_feats[startix:startix+n_per_image, :] = local_feat[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(n_im/opt.cacheBatchSize)), flush=True)
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(local_dim, num_clusters, niter=niter, verbose=False)
        kmeans.train(all_feats[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
    
def getClustersPCA(cluster_set,dim = 128,nv_pca = 32):
    n_descriptors = 10000
    n_per_image = 50
    n_im = ceil(n_descriptors/n_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                sampler=sampler)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    initcache = join(opt.cachePath, 'desc_cen.hdf5') 
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad(): 
            model.eval() 
            print('====> Extracting Descriptors') 
            all_feats = h5.create_dataset("descriptors", 
                        [n_descriptors, dim],  
                        dtype=np.float32) 
            
            for iteration, (query, _, _, _) in enumerate(data_loader, 1):
                query = query.to(device)
                # local_feat, _, _ = model(query)
                local_feat, _,  = model.rem(query)
                local_feat = local_feat.view(query.size(0), dim, -1).permute(0, 2, 1)
                
                batchix = (iteration-1)*opt.cacheBatchSize*n_per_image
                for ix in range(local_feat.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
                    startix = batchix + ix*n_per_image
                    all_feats[startix:startix+n_per_image, :] = local_feat[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 20:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(n_im/opt.cacheBatchSize)), flush=True) 

        if nv_pca is not None:
            pca = PCA(nv_pca, random_state=0)
            all_feats_array = all_feats[:]
            pca.fit(all_feats_array)
            all_feats_array = pca.transform(all_feats_array) 
            
            dims = nv_pca
            print('====> Clustering PCA descriptors..')

            niter = 100
            kmeans = faiss.Kmeans(dims, num_clusters, niter=niter, verbose=False)
            kmeans.train(all_feats_array[...])
            
            print('====> Storing centroids', kmeans.centroids.shape)
            h5.create_dataset('centroids', data=kmeans.centroids)

            h5.create_dataset("pca_descriptors", data=all_feats_array)

            h5.create_dataset('pca_mean', data=pca.mean_)
            h5.create_dataset('pca_components', data=pca.components_)
            print('====> Done!')
            # batch = {'centroids':kmeans.centroids,'pca_descriptors':all_feats_array,'pca_mean':pca.mean_,'pca_components':pca.components_}
            return None

        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(dim, num_clusters, niter=niter, verbose=False)
        kmeans.train(all_feats[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    filename = model_out_path+'/'+filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_out_path+'/'+'model_best.pth.tar')

def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    return trainable_params

if __name__ == "__main__":
    opt = get_args()

    device = torch.device("cuda")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Building model')

    from REIN_PCA import REIN


    use_pca = True
    pca_dim = 32
    num_clusters = 64
    model = REIN(rotations=8,pca_dim = pca_dim,use_pca = use_pca,num_clusters = num_clusters)

    
    model = model.cuda()
    local_dim = model.local_feat_dim
    # initialize netvlad with pre-trained or cluster
    if opt.load_from:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.load_from,  'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.load_from, 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.rem.use_pca = True
            model.rem.init_pca = True
            model = model.to(device)
            print(checkpoint['recalls'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
    else:
        initcache = join(opt.cachePath, 'desc_cen.hdf5')
        if not isfile(initcache):
            train_set = kitti_dataset.TrainingDataset()
            print('===> Calculating descriptors and clusters')
            if use_pca:
                getClustersPCA(train_set,dim=local_dim,nv_pca=pca_dim)
            else:
                getClusters(train_set)
        
        if use_pca:
            with h5py.File(initcache, mode='r') as h5: 
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("pca_descriptors")[...]
                pca_mean = h5.get("pca_mean")[...]
                pca_components = h5.get("pca_components")[...]
            model.rem.pca_mean = nn.Parameter(torch.from_numpy(pca_mean))
            model.rem.pca_rot = nn.Parameter(torch.from_numpy(pca_components))
            model.rem.use_pca = True
            model.rem.init_pca = True
            model.pooling.init_params(clsts, traindescs)
            model = model.cuda()
        else:
            with h5py.File(initcache, mode='r') as h5: 
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                model.pooling.init_params(clsts, traindescs) 
                model = model.cuda()


    if opt.mode.lower() == 'train':
        # preparing tensorboard
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))
        
        logdir = writer.file_writer.get_logdir()
        try:
            makedirs(logdir)
        except:
            pass

        with open(join(logdir, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)


        print('===> Loading dataset(s)')

        train_set = kitti_dataset.TrainingDataset() 
        val_set={}
        for seq in ['00', '02', '05', '06']:   
        # for seq in ['2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28','2012-11-16','2013-02-23']:
            val_set[seq] = kitti_dataset.InferDataset(seq=seq)

        # initilize model weights
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
            model.parameters()), lr=opt.lr)    
    
        
        

        best_score = 0

        for epoch in range(opt.nEpochs):
            train_epoch(epoch, model, train_set)
            print('===> Testing')

            eval_seq =  ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
            eval_datasets = []
            eval_global_descs = []
            for seq in eval_seq:   
                test_set = nclt_dataset.InferDataset(seq=seq)   
                global_descs = infer(test_set)
                eval_global_descs.append(global_descs)
                eval_datasets.append(test_set)
            
            recalls_nclt = nclt_dataset.evaluateResults(eval_global_descs, eval_datasets)# (q_descs, db_descs, q_dataset, db_dataset)
            for ii in range(len(recalls_nclt)):
                writer.add_scalars('val', {'NCLT_'+eval_seq[ii+1]: recalls_nclt[ii]}, epoch)
            print(recalls_nclt)
            mean_recall = np.mean(recalls_nclt)

            # print('===> Mean Recall on KITTI: %0.2f'%(np.mean(recalls_kitti)*100))
            print('===> Mean Recall on NCLT : %0.2f'%(np.mean(recalls_nclt)*100))

            is_best = mean_recall > best_score 
            if is_best:   best_score = mean_recall
            
            saveCheckpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': mean_recall,
                    'best_score': best_score,
                    'optimizer' : optimizer.state_dict(),
            }, is_best, logdir)
        
        print('===> Best Recall: %0.2f'%(mean_recall*100))
        writer.close()

    elif opt.mode.lower() == 'test':
        # 文件路径
        print('===> Running evaluation step')
        
        recalls_kitti = []
        print('====> Extracting Features of KITTI and calculating recalls')

        global_save_path = 'global_descripors'
        if not os.path.exists(global_save_path):
            os.makedirs(global_save_path)
        eval_kitti_modes = ['PR_only','global_localization','loop_closure']



        kitti = True
        rot_kitti = True

        nclt = True
        eval_nclt_modes = ['loop_closure','cross_localization']


        if kitti:

            for eval_mode in eval_kitti_modes:
                if eval_mode == 'PR_only':
                    for seq in ['00', '02', '05', '06', '08']:
                        if rot_kitti:
                            print('evaluate place recognition only on rot kitti')
                            test_set = kitti_dataset.InferDataset(seq=seq,sample_inteval=1,rot=True)   
                            global_descs = infer(test_set)
                            recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set)
                            print('===> Recall @ top 1 on Rot KITTI %s: %0.2f'%(seq, recall_top1*100))
                            del global_descs
                        else:
                            print('evaluate place recognition only on kitti')
                            test_set = kitti_dataset.InferDataset(seq=seq,sample_inteval=1,rot=False)   
                            global_descs = infer(test_set)
                            recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set)
                            print('===> Recall @ top 1 on KITTI %s: %0.2f'%(seq, recall_top1*100))
                            del global_descs
                elif eval_mode == 'global_localization':
                    for seq in ['00', '02', '05', '06', '08']:
                        if rot_kitti:
                            # if seq in ['00','08']:
                            print('evaluate global localization on rot-kitti')
                            test_set = kitti_dataset.InferDataset(seq=seq,sample_inteval=1,rot=rot_kitti)  #return a very large local feature mat could be very slow. sample the dataset to reduce ram and time cost
                            local_feats, global_descs, angles= infer(test_set, return_local_feats=True, return_angles=True)  
                            test_set = kitti_dataset.InferDataset(seq=seq,sample_inteval=1,rot=False)
                            recall_top1, success_rate, mean_trans_err, mean_rot_err = kitti_dataset.evaluateResults(seq, global_descs, local_feats, test_set,rot_angles=angles, match_results_save_path="out_imgs/")
                            print("seq:",seq,"success_rate:",success_rate*100,"recall_top1:",recall_top1*100,"mean_trans_err:",mean_trans_err,"mean_rot_err:",mean_rot_err)
                            del local_feats,global_descs
                        else:
                            print('evaluate global localization on kitti')
                            test_set = kitti_dataset.InferDataset(seq=seq,sample_inteval=1,rot=False)  #return a very large local feature mat could be very slow. sample the dataset to reduce ram and time cost
                            local_feats, global_descs = infer(test_set, return_local_feats=True)  
                            recall_top1, success_rate, mean_trans_err, mean_rot_err = kitti_dataset.evaluateResults(seq, global_descs, local_feats, test_set, rot_angles=None,match_results_save_path = "out_imgs/")
                            print("seq:",seq,"success_rate:",success_rate*100,"recall_top1:",recall_top1*100,"mean_trans_err:",mean_trans_err,"mean_rot_err:",mean_rot_err)
                            
                            del local_feats,global_descs
                
                elif eval_mode == 'loop_closure':
                    for seq in ['00', '02', '05', '06','08']:
                        print('evaluate loop closure')
                        test_set = kitti_dataset.InferDataset(seq=seq,sample_inteval=1)   
                        global_descs = infer(test_set)
                        average_precision, f1_max, max_recall = kitti_dataset.evaluateLoopClosureResults(global_descs,test_set)
                        del global_descs,test_set
                print(seq,'Done')

        eval_seq =  ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
        
        # eval_seq = ['2012-11-16']
        if nclt:
            for eval_mode in eval_nclt_modes:
                if eval_mode == 'loop_closure':
                    print('evaluate LoopClosureResults on NCLT')
                    for seq in eval_seq:   
                        test_set = nclt_dataset.InferDataset(seq=seq)   
                        local_feats,global_descs = infer(test_set,return_local_feats=True)
                        path = os.path.join(global_save_path,'LighterBEV_nclt_'+seq + '.npy')
                        average_precision,f1_max,max_recall = nclt_dataset.evaluateLoopClosureResults(global_descs,test_set)

                        recall_top1, success_rate, mean_trans_err, mean_rot_err = nclt_dataset.evaluateGLobalLocResults(global_descs,test_set,local_feats=local_feats,match_results_save_path='out_imgs/')
                        print(f'success_rate:{success_rate}, recall_top1: {recall_top1}, mean_trans_err: {mean_trans_err}, mean_rot_err: {mean_rot_err}')
                        print(seq,'Done')
                        del local_feats,global_descs
                elif eval_mode == 'cross_localization':
                    print('====> evaluate cross_localization')
                    eval_datasets = []
                    eval_global_descs = []
                    recalls_nclt = []
                    splits = 4
                    import math
                    for seq in eval_seq:
                        if seq == '2012-01-15':
                            database_set = nclt_dataset.InferDataset(seq=seq)
                            database_local_feats, database_global_descs = infer(database_set,return_local_feats=True)
                            database_poses = database_set.poses
                        else:
                            test_set = nclt_dataset.InferDataset(seq=seq)
                            dataset_size = len(test_set)
                            split_size = math.ceil(dataset_size / splits)
                            subsets = []
                            tp = 0
                            positives_num = 0
                            all_errs = []

                            for split in range(splits):
                                start_idx = split * split_size
                                end_idx = min(start_idx + split_size, dataset_size) 
                                indices = list(range(start_idx, end_idx))
                                subset = Subset(test_set, indices)

                                query_local_feats, query_global_descs = infer(subset,return_local_feats=True)

                                
                                recalls_dict = nclt_dataset.evaluateResultsV2(database_local_feats, database_global_descs,query_local_feats, query_global_descs, indices, subset,database_set,match_results_save_path='out_imgs/')
                                
                                tp += recalls_dict['tp']
                                positives_num += recalls_dict['all_positives']
                                all_errs += recalls_dict['all_errs']
                                del query_local_feats, query_global_descs

                            all_errs = np.array(all_errs)
                            recall_top1 = tp / positives_num 
                            success_loc = (all_errs[:,0] < 2) & (all_errs[:,1]< 5)
                            success_rate = np.sum(success_loc)/positives_num
                            mean_trans_err = np.mean(all_errs[success_loc,0]) 
                            mean_rot_err = np.mean(all_errs[success_loc,1]) 
                            print(f'seq:{seq} success_rate:{success_rate}, recall_top1: {recall_top1}, mean_trans_err: {mean_trans_err}, mean_rot_err: {mean_rot_err}')

                            recalls_nclt.append(recall_top1)
                    

                    print('\n################# Recall @ top 1 on NCLT ########################\n')
                    mean_recall = np.mean(recalls_nclt)


                    for ii in range(len(eval_seq[1:])):
                        print('%s: %0.2f'%(eval_seq[ii+1], recalls_nclt[ii]*100))
                    
                    print('mean: %0.2f'%(mean_recall*100))

        