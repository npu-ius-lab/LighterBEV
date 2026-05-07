import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data

import h5py

import faiss
import random
from RANSAC import rigidRansac
import torch.nn as nn
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

class Buffer(nn.Module):
    def __init__(self, max_size = 500, similarity_sampling = False, global_dim = 2048,refresh_samples = 200):
        self.max_size = max_size
        print('Memory has %d slots' % max_size)
        self.n_seen_so_far = 0
        self.current_index = 0
        self.positive_threshold = 5
        self.negative_threshold = 10
        self.num_negatives = 3
        self.img_shape = (self.max_size,3,200,200)
        self.positions = np.zeros(shape=(self.max_size,3),order='C').astype('float32')
        self.indices = np.zeros(shape=(self.max_size,)).astype('int32')
        self.sample_pool = torch.zeros(size=self.img_shape)
        self.feature_pool = np.zeros(shape=(self.max_size,global_dim),order='C').astype('float32')
        self.similarity_sampling = similarity_sampling

        self.faiss_index = None
        self.centroids = None

        self.reClusteringThre = max_size
        self.counter = 0
        self.refresh_samples = refresh_samples

    def positionsClustering(self):
        kmeans = faiss.Kmeans(d=3, k=10, niter=100, verbose=False)
        kmeans.train(self.positions)
        self.centroids = kmeans.centroids


    def refresh_featurepool(self, model, batch_size=10):
        iter_num = int(np.ceil(self.current_index / batch_size))  # Total iterations needed

        model.eval()
        for iter_id in range(0, iter_num):
            # Print progress: display the current iteration and total number of iterations
            print(f"refresh_featurepool processing batch {iter_id+1}/{iter_num}...")
            # Get the current batch of samples
            current_sample = self.sample_pool[iter_id * batch_size: (iter_id + 1) * batch_size]
            current_sample = current_sample.to('cuda')
            with torch.no_grad():
                _, _, current_feature = model(current_sample)
                current_feature = current_feature.detach().cpu().numpy()
                
                # Update the feature pool with the current batch's features
                self.feature_pool[iter_id * batch_size: (iter_id + 1) * batch_size] = current_feature
            
            # Release memory after processing the batch
            del current_sample
        model.train()


    def add(self,model,  batch_stream, global_desc):
        samples = batch_stream['img']
        positions = batch_stream['position'].numpy().astype('float32')
        n_samples = batch_stream['img'].shape[0]
        sample_indexes = batch_stream['index']
        place_left = max(0, self.max_size - self.current_index)
        if place_left:
            offset = min(place_left, n_samples)
            self.sample_pool[self.current_index: self.current_index + offset] = (samples[:offset])
            self.positions[self.current_index: self.current_index + offset] = (positions[:offset])
            self.feature_pool[self.current_index: self.current_index + offset] = global_desc
            self.indices[self.current_index: self.current_index + offset] = sample_indexes

            self.current_index += offset
            self.n_seen_so_far += offset
            if self.n_seen_so_far % self.refresh_samples == 0: 
                self.refresh_featurepool(model,batch_size = 50)
            return

        

        if not self.similarity_sampling:
            # add whatever still fits in the buffer
            x, y= samples[place_left:], positions[place_left:]
            indexes = torch.FloatTensor(x.shape[0]).to(x.device).uniform_(0, self.n_seen_so_far).long()
            valid_indexes = (indexes < self.sample_pool.shape[0]).long()
            idx_new_data = valid_indexes.nonzero().squeeze(-1)
            idx_buffer = indexes[idx_new_data]
            self.n_seen_so_far += x.shape[0]

            if self.n_seen_so_far // self.refresh_samples > (self.n_seen_so_far - x.shape[0]) // self.refresh_samples:
                self.refresh_featurepool(model, batch_size=50)
            
            if idx_buffer.numel() == 0:
                return
            assert idx_buffer.max() < self.sample_pool.shape[0]
            assert idx_new_data.max() < x.shape[0]
            assert idx_new_data.max() < y.shape[0]
            # perform overwrite op
            self.sample_pool[idx_buffer] = samples[idx_new_data]
            self.positions[idx_buffer] = positions[idx_new_data]    
            self.feature_pool[idx_buffer] = global_desc[idx_new_data] 
            self.indices[idx_buffer] = sample_indexes[idx_new_data]

        
            

    
    def get(self, batch_size):
        
        positions = self.positions[:self.current_index]
        feature_pool = self.feature_pool[:self.current_index]
        sample_pool = self.sample_pool[:self.current_index]
        
        position_index = faiss.IndexFlatL2(self.positions.shape[1])
        position_index.add(positions)
        distances, matching = position_index.search(positions, len(positions))
        
        query_list = []
        positive_list = []
        negatives_list = []
        neg_num_list = []


        for i in range(batch_size):
            
            over_threshold = True
            max_retry = 3  # maximum number of retries to find a valid positive sample
            retry_count = 0
            while over_threshold and retry_count < max_retry:
                # # Re-sample p_idx if distance is over the threshold
                q_idx = np.random.randint(0, self.current_index) # Re-sample another query index
                if matching.shape[0] <= 1: ##只有一个样本
                    retry_count = max_retry
                    continue
                p_idx = matching[q_idx][1]  ## 最近的一帧 排除自己
                if p_idx == q_idx:
                    q_idx = np.random.randint(0, self.current_index) # Re-sample another query index
                    p_idx = matching[q_idx][1] 
                
                query_position = positions[q_idx].reshape(-1,3)
                positive_position = positions[p_idx]

                over_threshold = np.sum((positive_position - query_position) ** 2) > self.positive_threshold ** 2 # Check again
                retry_count += 1
                

            # If the retry count exceeds max_retry, skip the current sample
            if retry_count >= max_retry:
                continue

            if p_idx == q_idx:
                print('p_idx==q_idx')
                continue

            _, neg_D, neg_I = position_index.range_search(query_position, self.negative_threshold ** 2)
            negatives_in_radius = set(neg_I)
            all_indices = set(range(len(positions)))
            negative_candidates = np.array(list(all_indices - negatives_in_radius))
            
            if negative_candidates.size == 0:
                continue
            else:
                q_feat = feature_pool[q_idx]
                neg_feat = feature_pool[negative_candidates]
                dis_neg = -np.sum((q_feat.reshape(1,-1) - neg_feat)**2,axis=1)

                num_neg = min(len(negative_candidates), self.num_negatives)
                dis_inc_index_tmp = dis_neg.argsort()[:-num_neg-1:-1]
                select_negatives = negative_candidates[dis_inc_index_tmp[:num_neg]]
                neg_num_list.append(num_neg)


            query_list.append(sample_pool[q_idx])
            positive_list.append(sample_pool[p_idx])
            negatives_list.append(sample_pool[select_negatives])


        if not query_list or not positive_list or not negatives_list:
            return None

        query = torch.stack(query_list)
        positive = torch.stack(positive_list)

        negatives = torch.cat(negatives_list, dim=0)

        if negatives.shape[0] == 0:
            return None
        
        return {'query': query, 'positive': positive, 'negatives': negatives, 'neg_num_list':neg_num_list}
        