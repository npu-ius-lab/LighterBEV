import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data
import h5py
import time
import faiss
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "RANSACCPP"))
import rigid_ransac
import matplotlib.pyplot as plt

def random_sector_mask(image, angle_range=30):

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = max(center)  
    start_angle = random.uniform(0, 360)
    end_angle = start_angle + angle_range
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.ellipse(
        mask, 
        center=center, 
        axes=(radius, radius), 
        angle=0, 
        startAngle=start_angle, 
        endAngle=end_angle, 
        color=255, 
        thickness=-1
    )

    inverted_mask = cv2.bitwise_not(mask)

    masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)
    return masked_image

def collate_fn(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    
    negatives = torch.cat(negatives, 0)
    indices = list(indices)

    return query, positive, negatives, indices




class TrainingDatasetOCL(data.Dataset):
    def __init__(self, dataset_path = '/media/ros/SSData/lbh/BEVPlace_oxford/data/NCLT/',seq='2013-02-23',random_mask=True):
        super().__init__()
        # bev path
        imgs_p = os.listdir(dataset_path+seq+'/bev_imgs/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/bev_imgs/'+i for i in imgs_p]

        
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')

        self.random_mask = random_mask 



    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        query,_ = load_img(self.imgs_path[index],random_mask=self.random_mask, rot=True)
        position = self.poses[index][[4,8,12]]

        frame_dict = {}
        frame_dict['img'] = query
        frame_dict['position'] = position
        frame_dict['index'] = index

        return frame_dict

    def __len__(self):
        return len(self.poses)





class TrainingDataset(data.Dataset):
    def __init__(self, dataset_path = '/media/ros/SSData/lbh/BEVPlace_oxford/data/NCLT/',seq='2013-02-23'):
        super().__init__()
        # bev path
        imgs_p = os.listdir(dataset_path+seq+'/bev_imgs/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/bev_imgs/'+i for i in imgs_p]

        # gt_pose
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')

        # neg, pos threshold
        self.pos_thres = 5
        self.neg_thres = 7 # 
        
        # compute pos and negs for each query
        self.num_neg = 3
        self.positives = [] 
        self.negatives = [] 

        for qi in range(len(self.poses)):
            q_pose = self.poses[qi]
            dises = np.sqrt(np.sum(((q_pose-self.poses)**2)[:,[4,8,12]],axis=1))            
            indexes = np.argsort(dises)
            remap_index = indexes[np.where(dises[indexes]<self.pos_thres)[0]]
            self.positives.append(remap_index)
            self.positives[-1] = self.positives[-1][1:] #exclude query itself
            negs = indexes[np.where(dises[indexes]>self.neg_thres)[0]]
            self.negatives.append(negs)
        
        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images
        self.random_mask = True 

    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            
            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]
            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

        else:
            pos_idx = self.positives[index][0]
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]

        
        query,_ = load_img(self.imgs_path[index],random_mask=self.random_mask, rot=True)
        
        positive,_ = load_img(join(self.imgs_path[pos_idx]),random_mask=self.random_mask, rot=True)

        negatives = []
        
        for neg_i in neg_idx:
            negative,_ = load_img(self.imgs_path[neg_i],random_mask=self.random_mask,rot=True)
            negatives.append(torch.from_numpy(negative))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)





def load_img(path, random_mask=False, rot=False):
    img = cv2.imread(path)
    if  random_mask:
        angle = np.random.randint(5,45)
        img = random_sector_mask(img, angle_range=angle)
    
    if rot:  #np.random.randint(0,360)
        rot_angle = np.random.randint(0,360)
        mat = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2 ), rot_angle, 1)
        img = cv2.warpAffine(img, mat, img.shape[:2]) 
    
    img = cv2.resize(img,(200,200))
    img = img.transpose(2,0,1)
    img = img.astype(np.float32)/256
    if rot:
        return img,rot_angle
    else:
        return img,0


class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path = '/media/ros/SSData/lbh/BEVPlace_oxford/data/NCLT/',sample_inteval=1):
        super().__init__()
        self.sample_inteval = sample_inteval

        # bev path
        imgs_p = os.listdir(dataset_path+seq+'/bev_imgs/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/bev_imgs/'+imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]
        # gt_pose
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')[::sample_inteval]
        self.rot = False

    def __getitem__(self, index):
        
        img,angle = load_img(self.imgs_path[index],rot=self.rot)
        return  img, angle
        
    def __len__(self):
        return len(self.imgs_path)



def evaluateResults(global_descs, datasets, local_feat_total = None, match_results_save_path=None):
    
    # for nclt, we use the seq 2012-02-15 for database, other sequences for query
    gt_thres = 5
    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1]) 
    faiss_index.add(global_descs[0])

    recalls_nclt = []
    for i in range(1, len(datasets)):
        _, predictions = faiss_index.search(global_descs[i], 1)  #top1
        
        all_positives = 0
        tp = 0
        dataset_q = datasets[i]
        dataset_d = datasets[0]
        if match_results_save_path is not None: 
            os.system('mkdir -p ' + match_results_save_path)
            all_errs = []
            local_feats_q = local_feat_total[i].transpose(0,2,3,1)
            local_feats_d = local_feat_total[0].transpose(0,2,3,1)
            
        for q_idx, pred in enumerate(predictions):
            query_idx = q_idx
            gt_dis = (dataset_q.poses[query_idx] - dataset_d.poses)**2
            positives = np.where(np.sum(gt_dis[:,[4,8,12]],axis=1) < gt_thres**2 )[0]
            if len(positives) > 0:
                all_positives += 1
                if pred[0] in positives:
                    tp += 1
                
                if match_results_save_path is not None:

                    index = pred[0]

                    query_im = dataset_q[query_idx][0].transpose(1,2,0)*256
                    db_im = dataset_d[index][0].transpose(1,2,0)*256

                    query_im = query_im.astype(np.uint8)
                    db_im = db_im.astype(np.uint8)

                    fast = cv2.FastFeatureDetector_create()
                    im_side = db_im.shape[0]

                    query_kps = fast.detect(query_im, None)
                    db_kps = fast.detect(db_im, None)

                    
                    query_des = [local_feats_q[query_idx][int(kp.pt[1]),int(kp.pt[0])] for kp in query_kps]
                    db_des = [local_feats_d[index][int(kp.pt[1]),int(kp.pt[0])] for kp in db_kps]
                    
                    query_des = np.array(query_des)
                    db_des = np.array(db_des)
                    
                    matcher = cv2.BFMatcher()
                    matches = matcher.knnMatch(query_des, db_des, k=2)
                    
                    

                    all_match = [m[0] for m in matches]
                    points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match]) 
                    points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

                    H, mask, max_csc_num = rigid_ransac.rigid_ransac((np.array([[im_side//2,im_side//2]]-points1)*0.4),(np.array([[im_side//2,im_side//2]]-points2))*0.4)# cv2.findHomography(points1, points2, cv2.RANSAC, 4.0)
                    
                    q_pose = dataset_q.poses[query_idx]

                    q_pose = np.hstack((q_pose[1:13].reshape(3,4)[:2,:2], q_pose[1:13].reshape(3,4)[:2,3].reshape(-1,1)))
                    q_pose = np.vstack((q_pose,np.array([[0,0,1]])))

                    db_pose = dataset_d.poses[index]
                    db_pose = np.hstack((db_pose[1:13].reshape(3,4)[:2,:2], db_pose[1:13].reshape(3,4)[:2,3].reshape(-1,1)))
                    db_pose = np.vstack((db_pose,np.array([[0,0,1]])))

                    relative_gt = np.linalg.inv(db_pose).dot((q_pose))
                    relative_H = np.vstack((H, np.array([[0,0,1]])))
                    
                    err = np.linalg.inv(relative_H).dot(relative_gt)
                    err_theta = np.abs(np.arctan2(err[0,1], err[0,0])/np.pi*180)
                    err_trans = np.sqrt(err[0,2]**2+err[1,2]**2)

                    if err_theta>5 or err_trans>2:
                        print('bug')
                    all_errs.append([err_trans, err_theta])
                                
                    good_match = [all_match[i] for i in range(len(mask)) if  mask[i]]
                    db_im = db_im*3
                    db_im[:,:,:2]=0


                    im = cv2.drawMatches(query_im.astype(np.uint8), query_kps, db_im.astype(np.uint8), db_kps, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    
                    out_im = np.zeros((im.shape[0]*2, db_im.shape[1]*3,3))
                    out_im[:im.shape[0], :db_im.shape[1]] = query_im
                    out_im[:im.shape[0], db_im.shape[1]:db_im.shape[1]*2] = db_im
                    out_im[:im.shape[0], db_im.shape[1]*2:] = db_im+query_im

                    out_im[-im.shape[0]:, :db_im.shape[1]*2] = im
                    

                    H = relative_H 
                    mat = cv2.getRotationMatrix2D((query_im.shape[0]//2, query_im.shape[0]//2), np.arctan2(-H[0,1], H[0,0])/np.pi*180, 1.0)
                    mat[0,2] -= H[1,2]/0.4
                    mat[1,2] -= H[0,2]/0.4
                    mat = np.vstack((mat,np.array([[0,0,1]])))
                    mat = np.linalg.inv(mat)[:2,:]
                    im_warp = cv2.warpAffine(db_im, mat, query_im.shape[:2])

                    im_warp[:,:,:2]=0
                    out_im[-im.shape[0]:, db_im.shape[1]*2:db_im.shape[1]*3] = im_warp+query_im   
                    # if err_theta>5 or err_trans>2:
                    #     cv2.imwrite(match_results_save_path+str(1000000+query_idx)[1:]+".png", out_im)         
                    cv2.imwrite(match_results_save_path+str(1000000+query_idx)[1:]+".png", out_im)
        
        recall_top1 = tp / all_positives #tp/(tp+fp)
        recalls_nclt.append(recall_top1)
        if match_results_save_path is not None:
            all_errs = np.array(all_errs)
            success_loc = (all_errs[:,0]<2) & (all_errs[:,1]<5)
            success_rate = np.sum(success_loc)/all_positives
            mean_trans_err = np.mean(all_errs[success_loc,1])
            mean_rot_err = np.mean(all_errs[success_loc,0]) 
            print(f'success_rate:{success_rate}, recall_top1: {recall_top1}, mean_trans_err: {mean_trans_err}, mean_rot_err: {mean_rot_err}')
            del local_feats_d, local_feats_q, dataset_q, dataset_d

    return recalls_nclt

def plot_loop_closure_results(loop_pairs, threshold, dataset, save_path=None):
    tp_x = []
    tp_y = []
    fp_x = []
    fp_y = []
    fn_x = []
    fn_y = []
    no_loop_x = []
    no_loop_y = []

    poses = dataset.poses  # 获取地点（poses）数据

    for loop in loop_pairs:
        query_idx = loop[0]  # 查询地点索引
        match_idx = loop[1]  # 匹配地点索引
        distance = loop[2]   # 预测距离
        gt_candidates = loop[3]  # 真实正样本

        if distance < threshold and match_idx in gt_candidates:
            tp_x.append(poses[query_idx, 4])  # 使用 x 坐标
            tp_y.append(poses[query_idx, 8])  # 使用 y 坐标
        elif distance < threshold and match_idx not in gt_candidates:
            fp_x.append(poses[query_idx, 4])
            fp_y.append(poses[query_idx, 8])
        elif distance >= threshold and match_idx in gt_candidates:
            fn_x.append(poses[query_idx, 4])
            fn_y.append(poses[query_idx, 8])
        else:
            no_loop_x.append(poses[query_idx, 4])
            no_loop_y.append(poses[query_idx, 8])

    plt.figure(figsize=(12, 8))

    # 绘制没有回环的地点（细线表示）
    plt.plot(no_loop_x, no_loop_y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, label='No Loop Closure')

    # 绘制 TP、FP、FN
    plt.scatter(tp_x, tp_y, color='green', label='True Positives')
    plt.scatter(fp_x, fp_y, color='red', label='False Positives')
    plt.scatter(fn_x, fn_y, color='blue', label='False Negatives')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Loop Closure Detection Results with Poses')
    plt.legend()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()  # 如果没有提供保存路径，则显示图片



def evaluateLoopClosureResults(global_des, dataset, local_feat_total = None, match_results_save_path=None):

    threshold = 5
    faiss_index = faiss.IndexFlatL2(global_des.shape[1])#qFeat.shape[1])#pool_size)
    faiss_index.add(global_des)
    n_values = [global_des.shape[0]]
    predictions_distances, predictions = faiss_index.search(global_des, max(n_values)) 
    gt = dataset.poses

    loop_pairs = [] 

    for i in (range(110, len(gt)-100)):
        gt_dis = np.sum((gt[i,[4,8]] - gt[:i-100,[4,8]])**2,axis=1)
        gt_candidates = np.where(gt_dis < threshold**2)[0]
        top_candidates = np.where((i-predictions[i])>100)[0]
        top1 = predictions[i][top_candidates[0]]
        top1_dis = predictions_distances[i][top_candidates[0]]
        loop_pairs.append([i, top1, top1_dis, gt_candidates])
    
    
    recalls = []
    precisions = []

    for thres in np.arange(0,4,0.01):
        tp = [loop[0] for loop in loop_pairs if loop[2]<thres and loop[1] in loop[3]]
        fp = [loop[0] for loop in loop_pairs if loop[2]<thres and (loop[1] not in loop[3])]
        fn = [loop[0] for loop in loop_pairs if loop[2]>thres and loop[1] in loop[3]]
        recalls.append((len(tp)+1e-6)/(len(tp)+len(fn)+1e-6))
        precisions.append((len(tp)+1e-6)/(len(tp)+len(fp)+1e-6))

    precisions = np.array(precisions)
    recalls = np.array(recalls)


    average_precision = np.sum(precisions[:-1]*(recalls[1:]- recalls[:-1]))
    f1_scores = 2*recalls*precisions/(recalls+precisions)
    f1_max = np.max(f1_scores)
    f1_max_idx = np.argmax(f1_scores)
    best_thres = f1_max_idx * 0.01 
    max_recall = recalls[np.where(precisions==1)[0][-1]]
    print('AP: %0.8f'%(average_precision))
    print('F1 max: %0.8f'%(f1_max))
    print('max recalls:',max_recall)

    
    return average_precision,f1_max,max_recall





def evaluateGLobalLocResults(global_des, dataset,local_feats = None, match_results_save_path=None):
    
    threshold = 5
    faiss_index = faiss.IndexFlatL2(global_des.shape[1])#qFeat.shape[1])#pool_size)
    faiss_index.add(global_des)
    n_values = [global_des.shape[0]]
    predictions_distances, predictions = faiss_index.search(global_des, max(n_values)) 
    gt = dataset.poses

    loop_pairs = [] 
    tp = 0
    all_positives = 0
    all_errs = []

    if match_results_save_path is not None:
        local_feats = local_feats.transpose(0,2,3,1)
    
    for i in (range(110, len(gt)-100)):
        gt_dis = np.sum((gt[i,[4,8]] - gt[:i-100,[4,8]])**2,axis=1)
        gt_candidates = np.where(gt_dis < threshold**2)[0]
        top_candidates = np.where((i-predictions[i])>100)[0]
        top1 = predictions[i][top_candidates[0]]
        top1_dis = predictions_distances[i][top_candidates[0]]

        if len(gt_candidates)>0:
            all_positives+=1
            if top1 in gt_candidates:
                tp += 1

            if 1:
                if match_results_save_path is not None:
                    index = top1
                    query_idx = i

                    query_im = dataset[query_idx][0].transpose(1,2,0)*256
                    db_im = dataset[index][0].transpose(1,2,0)*256

                    query_im = query_im.astype(np.uint8)
                    db_im = db_im.astype(np.uint8)

                    fast = cv2.FastFeatureDetector_create()
                    im_side = db_im.shape[0]

                    query_kps = fast.detect(query_im, None)
                    db_kps = fast.detect(db_im, None)

                    query_des = [local_feats[query_idx][int(kp.pt[1]),int(kp.pt[0])] for kp in query_kps]
                    db_des = [local_feats[index][int(kp.pt[1]),int(kp.pt[0])] for kp in db_kps]
                    
                    query_des = np.array(query_des)
                    db_des = np.array(db_des)
                    
                    matcher = cv2.BFMatcher()
                    matches = matcher.knnMatch(query_des, db_des, k=2)
                    
                    all_match = [m[0] for m in matches]

                    points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match]) 
                    points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

                    H, mask, max_csc_num = rigid_ransac.rigid_ransac((np.array([[im_side//2,im_side//2]]-points1)*0.4),(np.array([[im_side//2,im_side//2]]-points2))*0.4)# cv2.findHomography(points1, points2, cv2.RANSAC, 4.0)
                    
                    q_pose = dataset.poses[query_idx]

                    q_pose = np.hstack((q_pose[1:13].reshape(3,4)[:2,:2], q_pose[1:13].reshape(3,4)[:2,3].reshape(-1,1)))
                    q_pose = np.vstack((q_pose,np.array([[0,0,1]])))

                    db_pose = dataset.poses[index]
                    db_pose = np.hstack((db_pose[1:13].reshape(3,4)[:2,:2], db_pose[1:13].reshape(3,4)[:2,3].reshape(-1,1)))
                    db_pose = np.vstack((db_pose,np.array([[0,0,1]])))

                    relative_gt = np.linalg.inv(db_pose).dot((q_pose))
                    relative_H = np.vstack((H, np.array([[0,0,1]])))
                    
                    err = np.linalg.inv(relative_H).dot(relative_gt)
                    err_theta = np.abs(np.arctan2(err[0,1], err[0,0])/np.pi*180)
                    err_trans = np.sqrt(err[0,2]**2+err[1,2]**2)

                    # if err_theta > 5 or err_trans > 2:
                    #     print('bug',(query_idx,index))
                    
                    all_errs.append([err_trans, err_theta])
                    
                    good_match = [all_match[i] for i in range(len(mask)) if  mask[i]]
                    db_im = db_im*3
                    db_im[:,:,:2]=0

                    im = cv2.drawMatches(query_im.astype(np.uint8), query_kps, db_im.astype(np.uint8), db_kps, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    
                    out_im = np.zeros((im.shape[0]*2, db_im.shape[1]*3,3))
                    out_im[:im.shape[0], :db_im.shape[1]] = query_im
                    out_im[:im.shape[0], db_im.shape[1]:db_im.shape[1]*2] = db_im
                    out_im[:im.shape[0], db_im.shape[1]*2:] = db_im+query_im

                    out_im[-im.shape[0]:, :db_im.shape[1]*2] = im
                    
                    H = relative_H 
                    mat = cv2.getRotationMatrix2D((query_im.shape[0]//2, query_im.shape[0]//2), np.arctan2(-H[0,1], H[0,0])/np.pi*180, 1.0)
                    mat[0,2] -= H[1,2]/0.4
                    mat[1,2] -= H[0,2]/0.4
                    mat = np.vstack((mat,np.array([[0,0,1]])))
                    mat = np.linalg.inv(mat)[:2,:]
                    im_warp = cv2.warpAffine(db_im, mat, query_im.shape[:2])

                    im_warp[:,:,:2]=0
                    out_im[-im.shape[0]:, db_im.shape[1]*2:db_im.shape[1]*3] = im_warp+query_im                
                    cv2.imwrite(match_results_save_path+str(1000000+query_idx)[1:]+".png", out_im)

    recall_top1 = tp / all_positives #tp/(tp+fp)

    if match_results_save_path is not None:
        all_errs = np.array(all_errs)
        success_loc = (all_errs[:,0]<2) & (all_errs[:,1]<5)
        success_rate = np.sum(success_loc)/all_positives
        mean_trans_err = np.mean(all_errs[success_loc,0])
        mean_rot_err = np.mean(all_errs[success_loc,1]) 

    return recall_top1, success_rate, mean_trans_err, mean_rot_err


   

