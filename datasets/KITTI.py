

import itertools
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.BaseDataset import (ImagesFromList, PcFromLists,
                                  input_transform_argumentation_img)
from datasets.DataManagement import (load_pc_file_fix_size, load_pc_files,
                                     loadCloudFromBinary)
from datasets.preprocess.rangeimage_utils import createRangeImage
from utils.logger import print_log

from .build import DATASETS

ImageFile.LOAD_TRUNCATED_IMAGES = True

@DATASETS.register_module()
class KITTITriplet(Dataset):
    def __init__(self, config):
        # config from dataset config file
        self.root_dir = config.data_path
        self.dataset_name = config.NAME
        # config from base config file
        self.mode = config.subset
        assert self.mode in ('train', 'val', 'test')
        self.sequences = config.default_sequences[self.mode]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.bs
        self.pre_save = config.pre_save # middle file for accleration the data loading
        self.img_subdir = config.img_subdir # dataset predefined parent dir
        self.submap_subdir = config.submap_subdir # dataset predefined parent dir
        # image and submap share the same idx, initialized as list, but turn into np.array after initialization
        # all sequences are append together into a same list but with unique idx
        self.qIdx = [] # query index is the total number of query data, whole the query idx are concated together with unique ID
        self.qImages = [] # query images path
        self.qPcs = [] # query pcs path
        self.dbImages = [] # database images path
        self.dbPcs = [] # database pcs path
        # index
        self.pIdx = [] # postive index corresponding to the query index, number is dynamically changed for each query data
        self.nonNegIdx = [] # neg
        self.qEndPosList = []
        self.dbEndPosList = []
        self.all_pos_indices = [] # gt indices, here the positives are treated as gt and the idx is unique in each sequence inside
        self.all_pos_dists = [] # gt distances
        # hyper-parameters
        self.image_size = config.image_size
        self.img_channel = config.img_channel
        self.point_cloud_proxy = config.point_cloud_proxy
        self.points_num = config.points_num
        self.point_channel = config.point_channel
        self.nNeg = config.nNeg # negative number
        self.posDistThr = config.posDistThr # positive distance
        self.negDistThr = config.negDistThr # negative distance
        self.cached_queries = config.cached_queries # cached queries
        self.cached_negatives = config.cached_negatives # cached negatives
        self.mining = config.mining # switcher indicates whether to conduct data mining
        self.get_neighbor = config.get_neighbor # indicator for custom collact_fn()
        # other
        self.transform = input_transform_argumentation_img(size_h=config.image_size[0], 
                                                           size_w=config.image_size[1],
                                                           train=True if self.mode=='train' else False)
        # load data
        for sequence_name in self.sequences:
            # get len of images from sequences so far for global indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)
            # load query and database data from train and val
            if self.mode == 'val' or self.mode == 'test':
                qData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, f"query.csv"), index_col=0)
                dbData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, f"database.csv"), index_col=0)
            else:
                qData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, 'query.csv'), index_col=0)
                # load database data
                dbData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, 'database.csv'), index_col=0)
            # what is the usage of the seq structure? or just some inherit from MSLS, bingo
            # fetch query data, specifically data path
            q_removed_mask, qSeqIdxs, qSeqKeys, qSeqKeys_pc = self.arange_as_seq(qData, 
                                                                os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name), 
                                                                os.path.join(self.root_dir, self.submap_subdir, "sequences", sequence_name),
                                                                point_cloud_proxy=self.point_cloud_proxy)
            # Convert the mask to a Series: removing the missing files
            q_removed_mask = pd.Series(q_removed_mask)
            # Filter the DataFrame using the mask
            qData = qData[q_removed_mask].reset_index(drop=True)
            # load database data
            db_removed_mask, dbSeqIdxs, dbSeqKeys, dbSeqKeys_pc = self.arange_as_seq(dbData, 
                                                                os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name), 
                                                                os.path.join(self.root_dir, self.submap_subdir, "sequences", sequence_name),
                                                                point_cloud_proxy=self.point_cloud_proxy)
            # Convert the mask to a Series
            db_removed_mask = pd.Series(db_removed_mask)
            # Filter the DataFrame using the mask
            dbData = dbData[db_removed_mask].reset_index(drop=True)
            # if there are no query/dabase images,
            # then continue to next sequence
            if len(qSeqIdxs) == 0 or len(dbSeqIdxs) == 0:
                continue
            # here qImages is same as qSeqKeys, this kind of operation is designed for MSLS sequence retrieval task especially
            self.qImages.extend(qSeqKeys)
            self.qPcs.extend(qSeqKeys_pc)
            self.dbImages.extend(dbSeqKeys)
            self.dbPcs.extend(dbSeqKeys_pc)
            self.qEndPosList.append(len(qSeqKeys))
            self.dbEndPosList.append(len(dbSeqKeys))
            # utm coordinates
            utmQ = qData[['East', 'North']].values.reshape(-1, 2)
            utmDb = dbData[['East', 'North']].values.reshape(-1, 2)
            # for pose display export
            self.utmQ = utmQ
            self.utmDb = utmDb
            # find positive images for training and testing
            # for all query images
            neigh = NearestNeighbors(algorithm='brute')
            neigh.fit(utmDb)
            pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)
            # the nearest idxes will be the ground truth when val mode
            self.all_pos_indices.extend(pos_indices) # positves are sorted from far to near which could be infer from distances
            self.all_pos_dists.extend(pos_distances)
            # fetch negative pairs for triplet turple, but the negatives here contains the positives
            if self.mode == 'train':
                nD, negIdx = neigh.radius_neighbors(utmQ, self.negDistThr)
            # get all idx unique in whole dataset
            for q_seq_idx in range(len(qSeqIdxs)):
                p_uniq_frame_idxs = pos_indices[q_seq_idx]
                # the query image has at least one positive
                if len(p_uniq_frame_idxs) > 0:
                    p_seq_idx = np.unique(p_uniq_frame_idxs)
                    # qIdx contains whole sequences, and the index is unique to whole (training or validation) datasets
                    self.qIdx.append(q_seq_idx + _lenQ)
                    self.pIdx.append(p_seq_idx + _lenDb)
                    # in training we have two thresholds, one for finding positives and one for finding data
                    # that we are certain are negatives.
                    if self.mode == 'train':
                        # n_uniq_frame_idxs = [n for nonNeg in nI[q_seq_idx] for n in nonNeg]
                        n_uniq_frame_idxs = negIdx[q_seq_idx]
                        n_seq_idx = np.unique(n_uniq_frame_idxs)
                        self.nonNegIdx.append(n_seq_idx + _lenDb)

        # whole sequence datas are gathered for batch optimization
        # Note that the number of submap is same as the number of images
        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print_log("Exiting...", logger="KITTITriplet")
            print_log("There are no query/database images.", logger="KITTITriplet")
            print_log("Try more sequences", logger="KITTITriplet")
            sys.exit()
        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx, dtype=object) 
        self.pIdx = np.asarray(self.pIdx, dtype=object)
        self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)
        # here only data path is stored
        self.qImages = np.asarray(self.qImages)
        self.qPcs = np.asarray(self.qPcs)
        self.dbImages = np.asarray(self.dbImages)
        self.dbPcs = np.asarray(self.dbPcs)

        if not self.mining and self.mode == "train":
            self.generate_triplets()

    @staticmethod
    def arange_as_seq(data:pd.DataFrame, path_img:str, path_pc:str, point_cloud_proxy:str=None):
        '''
            arrange all query data(images, submaps) into list container, no file io conducted here, thus the data integrity can only be verified by the dataloader
            Return:
                idx in csv file, image path, and pc full path in list container
        '''
        seq_keys, seq_idxs, seq_keys_pc = [], [], []
        # Create a boolean mask to keep only the rows want
        removed_mask = []
        for seq_idx in data.index:
            # iloc is a function of pandas library for get the seq_idx record
            seq = data.iloc[seq_idx]
            seq_key = os.path.join(path_img, 'image_2', seq['ImageFilename'])
            if point_cloud_proxy == "points":
                seq_key_pc = os.path.join(path_pc, 'velodyne', seq['LiDARFilename'])
            elif point_cloud_proxy == "bev_image":
                seq_key_pc = os.path.join(path_pc, 'bev', seq['LiDARFilename'])
            elif point_cloud_proxy == "range_image":
                seq_key_pc = os.path.join(path_pc, 'range_image', seq['LiDARFilename'])
            else:
                raise NotImplementedError(f'Sorry, <{point_cloud_proxy}> is not implemented!')
            # check the validity: remove the invalid file recored ==> avoiding the mismatch between index and actual file
            if os.path.exists(seq_key) and os.path.exists(seq_key_pc):
                removed_mask.append(True)
            else:
                removed_mask.append(False)
                continue
            # append into list
            seq_keys.append(seq_key)
            seq_keys_pc.append(seq_key_pc)
            seq_idxs.append([seq_idx])
        return removed_mask, np.asarray(seq_idxs), seq_keys, seq_keys_pc

    def __len__(self):
        return len(self.triplets)

    def __repr__(self):
        return  (f"{self.__class__.__name__} - #type: {self.mode} - #proxy: {self.point_cloud_proxy} - #database: {len(self.dbImages)} - #queries: {len(self.qIdx)}")

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_info(self):
        return self

    def generate_triplets(self):
        '''
            Purpose: get self.triplets fufilled from whole datasets, 
            Specifically, get the query idxs, the query data is randomly selected each epoch, get its postive and negatives
        '''
        # reset triplets
        self.triplets = []
        # get all query indices
        query_idxs = list(range(len(self.qIdx)))
        qidxs = np.array(query_idxs)
        # build triplets based on randomly selection from data
        for q in qidxs:
            # get query idx
            qidx = self.qIdx[q]
            # get positives
            pidxs = self.pIdx[q]
            # choose a random positive (within positive range default self.posDistThr m, fetch 1 positives)
            pidx = np.random.choice(pidxs, size=1)[0]
            # get negatives, 5 by default
            while True:
                # randomly select negatives from whole sequences in training dataset
                nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)
                # ensure that non of the choice negative images are within the negative range (default 25 m)
                # while-loop check until non nidx existed in self.nonNegIdx[q], due to the nonNegIdx is the samples inside negative range
                if sum(np.in1d(nidxs, self.nonNegIdx[q], assume_unique=True)) == 0:
                    break
            # package the triplet and target, all the indices are the indicex of csv file
            triplet = [qidx, pidx, *nidxs]
            target = [-1, 1] + [0] * len(nidxs)
            self.triplets.append((triplet, target))

    def is_mining(self):
        return self.mining

    def new_epoch(self):
        '''
            Purpose: 
                slice the whole query data into cached subsets based on the cached_queries number
                reset the subcache data from all query indices, shuffle is utilized
                random query subcache samples will be generated, whole query data will be forworded in a subset unit
            Note:
                Whole sequences data will be globed together into self.qIdx
        '''
        if not self.mining and self.triplets == []:
            self.generate_triplets()
            return
        # find how many subsets we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)
        # get all query indices
        arr = list(range(len(self.qIdx)))
        random.shuffle(arr)
        arr = np.array(arr)
        # the subcached_indices will be extracted from shuffled qIdx
        # the whole query data will be divided into subsets using self.cached_queries as interval
        # subcache_indices contains the query data idx in current subset, and covers whole sequences in training datasets
        # to be more aggressive, if cached_queries is smaller than 
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)
        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, model=None, outputdim=None, margin=0.2):
        '''
            Purpose: get self.triplets fufilled from current subset, 
            Specifically, get the query idxs from cached subset, the query data is randomly selected each epoch
                        get its postive and negatives
        '''
        # reset triplets
        self.triplets = []
        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        if self.current_subset >= len(self.subcache_indices):
            print_log('Reset epoch - FIX THIS LATER!', logger="KITTITriplet")
            self.current_subset = 0
        # take n (query,positive,negatives) triplet images from current cached subsets
        qidxs = np.asarray(self.subcache_indices[self.current_subset])
        #print("len(qidxs):\t", qidxs) # should be same as self.cached_queries (or slightly smaller than, for the last subset)
        # build triplets based on randomly selection from data
        if model is None:
            for q in qidxs:
                # get query idx
                qidx = self.qIdx[q]
                # get positives
                pidxs = self.pIdx[q]
                # choose a random positive (within positive range default self.posDistThr m, fetch 1 positives)
                pidx = np.random.choice(pidxs, size=1)[0]
                # get negatives, 5 by default
                while True:
                    # randomly select negatives from whole sequences in training dataset
                    nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)
                    # ensure that non of the choice negative images are within the negative range (default 25 m)
                    # while-loop check until non nidx existed in self.nonNegIdx[q], due to the nonNegIdx is the samples inside negative range
                    if sum(np.in1d(nidxs, self.nonNegIdx[q], assume_unique=True)) == 0:
                        break
                # package the triplet and target, all the indices are the indicex of csv file
                triplet = [qidx, pidx, *nidxs]
                target = [-1, 1] + [0] * len(nidxs)
                self.triplets.append((triplet, target))
            # increment subset counter
            self.current_subset += 1
            return
        print_log("Online hard sample mining...", logger="KITTITriplet")
        # take n=5 positives in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in np.random.choice(idx, size=5, replace=False)])
        nidxs = []
        while len(nidxs) < self.cached_queries // 10:
            # take m = 5*cached_queries is number of negative images
            nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)
            # and make sure that there is no positives among them
            nidxs = nidxs[np.in1d(nidxs, np.unique(
                [i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]
        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.batch_size, 'shuffle': False, 'persistent_workers': False, 
               'num_workers': self.threads, 'pin_memory': True}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.transform), **opt)
        ploader_pc = torch.utils.data.DataLoader(PcFromLists(self.dbPcs[pidxs]), **opt)
        nloader_pc = torch.utils.data.DataLoader(PcFromLists(self.dbPcs[nidxs]), **opt)
        # calculate their descriptors
        model.eval()
        with torch.no_grad():
            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), outputdim).to(self.device) # all query
            pvecs = torch.zeros(len(pidxs), outputdim).to(self.device) # all corresponding positives
            nvecs = torch.zeros(len(nidxs), outputdim).to(self.device) # all corresponding negatives
            batch_size = opt['batch_size']
            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), 
                                 desc='compute query descriptors', 
                                 total=len(qidxs) // batch_size,
                                 position=2, leave=False):
                X, _ = batch
                image_encoding = model.encoder(X.to(self.device))
                vlad_encoding = model.pool(image_encoding)
                qvecs[i * batch_size:(i + 1) * batch_size, :] = vlad_encoding
                del batch, X, image_encoding, vlad_encoding
            # release memory
            del qloader
            for i, batch in tqdm(enumerate(ploader_pc), 
                                 desc='compute positive descriptors', 
                                 total=len(pidxs) // batch_size,
                                 position=2, leave=False):
                X, _ = batch
                X = X.view((-1, 1, self.points_num, self.point_channel))
                vlad_encoding = model(X.to(self.device))
                pvecs[i * batch_size:(i + 1) * batch_size, :] = vlad_encoding
                del batch, X, vlad_encoding
            # release memory
            del ploader_pc
            for i, batch in tqdm(enumerate(nloader_pc), 
                                 desc='compute negative descriptors', 
                                 total=len(nidxs) // batch_size,
                                 position=2, leave=False):
                X, _ = batch
                X = X.view((-1, 1, self.points_num, self.point_channel))
                vlad_encoding = model(X.to(self.device))
                nvecs[i * batch_size:(i + 1) * batch_size, :] = vlad_encoding
                del batch, X, vlad_encoding
            # release memory
            del nloader_pc
        print_log('Searching for hard negatives...', logger="KITTITriplet")
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)
        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        # the first return is the sorted tensor, the second return is the raw index that are sorted
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)
        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()
        # selection of hard triplets
        for q in range(len(qidxs)):
            qidx = qidxs[q]
            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))
            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))
            # take the closest positve
            dPos = pScores[q, pidx][0][0]
            # get distances to all negatives
            dNeg = nScores[q, :]
            # how much are they violating
            loss = dPos - dNeg + margin ** 0.5
            violatingNeg = 0 < loss
            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue
            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]
            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]
            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]
            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]
            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            target = [-1, 1] + [0] * len(hardestNeg)
            self.triplets.append((triplet, target))
        # release memory
        del qvecs, nvecs, pScores, pRanks, nScores, nRanks
        # increment subset counter
        self.current_subset += 1

    @staticmethod
    def collate_fn(batch):
        """
        Create mini-batch tensors from the list of tuples (query, positive, negatives).
        Args:
        batch: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
            - n: negative number
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """
        batch = list(filter(lambda x: x is not None, batch))
        # zip single triplet to batches
        query, query_pc, positive, positive_pc, negatives_imgs, negatives_pcs, indices = zip(*batch)
        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        query_pc = data.dataloader.default_collate(query_pc)
        positive_pc = data.dataloader.default_collate(positive_pc)
        negatives_pcs = data.dataloader.default_collate(negatives_pcs)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives_imgs])
        negatives_imgs = torch.cat(negatives_imgs, 0)
        # the query, positive, negatives indices are merged into list container
        indices = list(itertools.chain(*indices))
        return query, query_pc, positive, positive_pc, negatives_imgs, negatives_pcs, negCounts, indices

    @staticmethod
    def collate_fn_proxy_image(batch):
        """
        Create mini-batch tensors from the list of tuples (query, positive, negatives).
        Args:
        batch: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
            - n: negative number
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """
        batch = list(filter(lambda x: x is not None, batch))
        # zip single triplet to batches
        query, query_pc, positive, positive_pc, negatives_imgs, negatives_pcs, indices = zip(*batch)
        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        query_pc = data.dataloader.default_collate(query_pc)
        positive_pc = data.dataloader.default_collate(positive_pc)
        negatives_pcs = torch.cat(negatives_pcs, 0)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives_imgs])
        negatives_imgs = torch.cat(negatives_imgs, 0)
        # the query, positive, negatives indices are merged into list container
        indices = list(itertools.chain(*indices))
        return query, query_pc, positive, positive_pc, negatives_imgs, negatives_pcs, negCounts, indices

    def _get_item_normal(self, idx):
        '''
            for single triplet
            fetch query image, corresponding postives and negatives, idxes in current sequence data
        '''
        triplet, _ = self.triplets[idx]
        # get query, positive and negative idx both for images and pcs
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]
        # load images and pcs into triplet list
        query = self.transform(Image.open(self.qImages[qidx]).convert("RGB")).unsqueeze(0) # unsqueeze(0) add a new dim before the current dim-0
        positive_img = self.transform(Image.open(self.dbImages[pidx]).convert("RGB")).unsqueeze(0)
        negatives_imgs = [self.transform(Image.open(self.dbImages[idx]).convert("RGB")) for idx in nidx]
        negatives_imgs = torch.stack(negatives_imgs, 0).unsqueeze(0)
        query_pc = load_pc_files([self.qPcs[qidx]], pts_num=self.points_num)#, augment=self.mode=='train')
        positive_pc = load_pc_files([self.dbPcs[pidx]], pts_num=self.points_num)#, augment=self.mode=='train')
        negatives_pcs = load_pc_files([self.dbPcs[idx] for idx in nidx], pts_num=self.points_num)#, augment=self.mode=='train')
        return query, query_pc, positive_img, positive_pc, negatives_imgs, negatives_pcs, [qidx, pidx] + nidx

    def _get_item_proxy_image(self, idx):
        '''
            for single triplet but load image proxy replacing the raw point cloud
            fetch query image, corresponding postives and negatives, idxes in current sequence data
        '''
        triplet, _ = self.triplets[idx]
        # get query, positive and negative idx both for images and pcs
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]
        # load images and pcs into triplet list
        query = self.transform(Image.open(self.qImages[qidx]).convert("RGB")).unsqueeze(0) # unsqueeze(0) add a new dim before the current dim-0
        positive_img = self.transform(Image.open(self.dbImages[pidx]).convert("RGB")).unsqueeze(0)
        negatives_imgs = [self.transform(Image.open(self.dbImages[idx]).convert("RGB")) for idx in nidx]
        negatives_imgs = torch.stack(negatives_imgs, 0).unsqueeze(0)
        query_pc = self.transform(Image.open(self.qPcs[qidx]).convert("RGB")).unsqueeze(0) # unsqueeze(0) add a new dim before the current dim-0
        positive_pc = self.transform(Image.open(self.dbPcs[pidx]).convert("RGB")).unsqueeze(0)
        negatives_pcs = [self.transform(Image.open(self.dbPcs[idx]).convert("RGB")) for idx in nidx]
        negatives_pcs = torch.stack(negatives_pcs, 0).unsqueeze(0)
        return query, query_pc, positive_img, positive_pc, negatives_imgs, negatives_pcs, [qidx, pidx] + nidx

    def __getitem__(self, idx):
        if self.point_cloud_proxy == "points":
            if self.get_neighbor:
                # TODO
                return self._get_item_neighbor(idx)
            else:
                return self._get_item_normal(idx)
        elif self.point_cloud_proxy == "bev_image":
            return self._get_item_proxy_image(idx)
        elif self.point_cloud_proxy == "range_image":
            return self._get_item_proxy_image(idx)
        else:
            raise NotImplementedError(f'Sorry, <{self.point_cloud_proxy}> is not implemented!')


@DATASETS.register_module()
class KITTIPair(Dataset):
    def __init__(self, config):
        # config from dataset config file
        self.root_dir = config.data_path
        self.dataset_name = config.NAME
        self.config = config
        # config from base config file
        self.mode = config.subset
        assert self.mode in ('train', 'val', 'test')
        self.sequences = config.default_sequences[self.mode]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.bs
        self.pre_save = config.pre_save # middle file for accleration the data loading
        self.img_subdir = config.img_subdir # dataset predefined parent dir
        self.submap_subdir = config.submap_subdir # dataset predefined parent dir
        # image and submap share the same idx, initialized as list, but turn into np.array after initialization
        # all sequences are append together into a same list but with unique idx
        self.qIdx = [] # query index is the total number of query data, whole the query idx are concated together with unique ID
        self.qImages = [] # query images path
        self.qPcs = [] # query pcs path
        self.dbImages = [] # database images path
        self.dbPcs = [] # database pcs path
        # index
        self.pIdx = [] # postive index corresponding to the query index, number is dynamically changed for each query data
        self.nonNegIdx = [] # non-neg, should exclude thsese index
        self.qEndPosList = []
        self.dbEndPosList = []
        self.all_pos_indices = [] # gt indices
        self.all_pos_dists = [] # gt distances
        self.query_pidxs = [] # positive idx based on query data
        self.query_nidxs = []
        # hyper-parameters
        self.image_size = config.image_size
        self.img_channel = config.img_channel
        self.point_cloud_proxy = config.point_cloud_proxy
        self.points_num = config.points_num
        self.point_channel = config.point_channel
        self.nPos = config.nPos # positive number
        self.nNeg = config.nNeg # negative number
        self.posDistThr = config.posDistThr # positive distance
        self.negDistThr = config.negDistThr # negative distance
        self.point_limit = config.point_limit # maximum point number
        self.get_neighbor = False # TODO config.get_neighbor # indicator for custom collact_fn()
        self.contrast_mode = config.contrast_mode # if False, do postive and negative searching
        # other
        self.transform = input_transform_argumentation_img(size_h=config.image_size[0], 
                                                           size_w=config.image_size[1],
                                                           train=True if self.mode=='train' else False)
        if self.point_cloud_proxy != "points":
            self.transform_proxy = input_transform_argumentation_img(size_h=config.proxy_image_size[0], 
                                                           size_w=config.proxy_image_size[1],
                                                           train=True if self.mode=='train' else False)
        # load data
        if self.contrast_mode and self.mode == 'train': # gather whole image and point cloud pairs without data mining
            for sequence_name in self.sequences:
                # get len of images from sequences so far for global indexing
                _lenQ = len(self.qImages)
                _lenDb = len(self.dbImages)
                # query and database split is designed for be compatiable with tulpet mode
                qData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, f"query.csv"), index_col=0)
                dbData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, f"database.csv"), index_col=0)
                qData = pd.concat([qData, dbData])
                # both train and validation model requires query data
                q_removed_mask, qSeqIdxs, qSeqKeys, qSeqKeys_pc = self.arange_as_seq(qData, 
                                                            os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name), 
                                                            os.path.join(self.root_dir, self.submap_subdir, "sequences", sequence_name),
                                                            point_cloud_proxy=self.point_cloud_proxy)
                # Convert the mask to a Series
                q_removed_mask = pd.Series(q_removed_mask)
                # Filter the DataFrame using the mask
                qData = qData[q_removed_mask].reset_index(drop=True)
                if len(qSeqIdxs) == 0:
                    continue
                # here qImages is same as qSeqKeys, this kind of operation is designed for MSLS sequence retrieval task especially
                self.qImages.extend(qSeqKeys)
                self.qPcs.extend(qSeqKeys_pc)
                # get all idx unique in whole dataset
                for q_seq_idx in range(len(qSeqIdxs)):
                    self.qIdx.append(q_seq_idx + _lenQ + _lenDb)
            # cast to np.arrays for indexing during training
            self.qIdx = np.asarray(self.qIdx, dtype=object)
            self.pIdx = np.asarray(self.pIdx, dtype=object)
            # here only data path is stored
            self.qImages = np.asarray(self.qImages)
            self.qPcs = np.asarray(self.qPcs)
            return
        # get tuplet data
        for sequence_name in self.sequences:
            # get len of images from sequences so far for global indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)
            # load query and database data from train and val
            if self.mode in ['val','test']:
                qData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, 'query_val.csv'), index_col=0)
                # load database data
                dbData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, 'database_val.csv'), index_col=0)
            else:
                qData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, 'query.csv'), index_col=0)
                # load database data
                dbData = pd.read_csv(os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name, 'database.csv'), index_col=0)
            # what is the usage of the seq structure? or just some inherit from MSLS, bingo
            # fetch query data, specifically data path
            q_removed_mask, qSeqIdxs, qSeqKeys, qSeqKeys_pc = self.arange_as_seq(qData, 
                                                                os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name), 
                                                                os.path.join(self.root_dir, self.submap_subdir, "sequences", sequence_name),
                                                                point_cloud_proxy=self.point_cloud_proxy)
            # Convert the mask to a Series: removing the missing files
            q_removed_mask = pd.Series(q_removed_mask)
            # Filter the DataFrame using the mask
            qData = qData[q_removed_mask].reset_index(drop=True)
            # load database data
            db_removed_mask, dbSeqIdxs, dbSeqKeys, dbSeqKeys_pc = self.arange_as_seq(dbData, 
                                                                os.path.join(self.root_dir, self.img_subdir, "sequences", sequence_name), 
                                                                os.path.join(self.root_dir, self.submap_subdir, "sequences", sequence_name),
                                                                point_cloud_proxy=self.point_cloud_proxy)
            # Convert the mask to a Series
            db_removed_mask = pd.Series(db_removed_mask)
            # Filter the DataFrame using the mask
            dbData = dbData[db_removed_mask].reset_index(drop=True)
            # if there are no query/dabase images,
            # then continue to next sequence
            if len(qSeqIdxs) == 0 or len(dbSeqIdxs) == 0:
                continue
            # here qImages is same as qSeqKeys, this kind of operation is designed for MSLS sequence retrieval task especially
            self.qImages.extend(qSeqKeys)
            self.qPcs.extend(qSeqKeys_pc)
            self.dbImages.extend(dbSeqKeys)
            self.dbPcs.extend(dbSeqKeys_pc)
            self.qEndPosList.append(len(qSeqKeys))
            self.dbEndPosList.append(len(dbSeqKeys))
            # utm coordinates
            utmQ = qData[['East', 'North']].values.reshape(-1, 2)
            utmDb = dbData[['East', 'North']].values.reshape(-1, 2)
            # for pose display export
            self.utmQ = utmQ
            self.utmDb = utmDb
            # find positive images for training and testing
            # for all query images
            neigh = NearestNeighbors(algorithm='brute')
            neigh.fit(utmDb)
            pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)
            # may be the positves should be sorted!
            # the nearest idxs will be the ground truth when val or test mode
            self.all_pos_indices.extend(pos_indices) # positves are sorted from far to near which could be infer from distances
            self.all_pos_dists.extend(pos_distances)
            # fetch negative pairs for triplet turple, but the negatives here contains the positives
            if self.mode == 'train':
                nD, negIdx = neigh.radius_neighbors(utmQ, self.negDistThr)
            # get all idx unique in whole dataset
            for q_seq_idx in range(len(qSeqIdxs)):
                p_uniq_frame_idxs = pos_indices[q_seq_idx]
                # the query image has at least one positive
                if len(p_uniq_frame_idxs) > 0:
                    p_seq_idx = np.unique(p_uniq_frame_idxs)
                    # qIdx contains whole sequences, and the index is unique to whole (training or validation) datasets
                    self.qIdx.append(q_seq_idx + _lenQ)
                    self.pIdx.append(p_seq_idx + _lenDb)
                    # in training we have two thresholds, one for finding positives and one for finding data
                    # that we are certain are negatives.
                    if self.mode == 'train':
                        # n_uniq_frame_idxs = [n for nonNeg in nI[q_uniq_frame_idx] for n in nonNeg]
                        n_uniq_frame_idxs = negIdx[q_seq_idx]
                        n_seq_idx = np.unique(n_uniq_frame_idxs)
                        self.nonNegIdx.append(n_seq_idx + _lenDb)

        # whole sequence datas are gathered for batch optimization
        # Note that the number of submap is same as the number of images
        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print_log("Exiting...", logger="KITTIPair")
            print_log("There are no query/database images.", logger="KITTIPair")
            print_log("Try more sequences", logger="KITTIPair")
            sys.exit()
        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx, dtype=object)
        self.pIdx = np.asarray(self.pIdx, dtype=object)
        if self.mode == 'train':
            self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)
        # here only data path is stored
        self.qImages = np.asarray(self.qImages)
        self.qPcs = np.asarray(self.qPcs)
        self.dbImages = np.asarray(self.dbImages)
        self.dbPcs = np.asarray(self.dbPcs)
        if self.mode == 'train':
            # get the positive and negative based on query data
            # the for-loop could be replaced with array batch operation
            num_query = len(self.dbImages)
            # pseudo positive or negative idx array constructed by the full query idx
            query_idx = np.array(list(range(num_query)))
            # positive and negative mask
            self.query_is_pos_mask = np.zeros([num_query, num_query], dtype=bool)
            self.query_is_neg_mask = np.zeros([num_query, num_query], dtype=bool)
            for q in self.qIdx:
                query_true_positives_idx = np.array(self.pIdx[q], dtype=int)
                self.query_pidxs.append(query_true_positives_idx)
                self.query_is_pos_mask[q, query_true_positives_idx] = True
                # prune the non negative index
                query_true_negatives_idx = query_idx[~np.in1d(query_idx, self.nonNegIdx[q])]
                self.query_nidxs.append(query_true_negatives_idx)
                self.query_is_neg_mask[q, query_true_negatives_idx] = True
            self.query_pidxs = np.asarray(self.query_pidxs, dtype=object)
            self.query_nidxs = np.asarray(self.query_nidxs, dtype=object)

    @staticmethod
    def arange_as_seq(data:pd.DataFrame, path_img:str, path_pc:str, point_cloud_proxy:str=None):
        '''
            arrange all query data(images, submaps) into list container, no file io conducted here, thus the data integrity can only be verified by the dataloader
            Return:
                idx in csv file, image path, and pc full path in list container
        '''
        seq_keys, seq_idxs, seq_keys_pc = [], [], []
        # Create a boolean mask to keep only the rows want
        removed_mask = []
        for seq_idx in data.index:
            # iloc is a function of pandas library for get the seq_idx record
            seq = data.iloc[seq_idx]
            seq_key = os.path.join(path_img, 'image_2', seq['ImageFilename'])
            if point_cloud_proxy == "points":
                seq_key_pc = os.path.join(path_pc, 'velodyne', seq['LiDARFilename'])
            elif point_cloud_proxy == "bev_image":
                seq_key_pc = os.path.join(path_pc, 'bev', seq['LiDARFilename'])
            elif point_cloud_proxy == "range_image":
                seq_key_pc = os.path.join(path_pc, 'velodyne', seq['LiDARFilename'])
                # seq_key_pc = os.path.join(path_pc, 'range_image', seq['LiDARFilename']) # TODO
            else:
                raise NotImplementedError(f'Sorry, <{point_cloud_proxy}> is not implemented!')
            # check the validity: remove the invalid file recored ==> avoiding the mismatch between index and actual file
            if os.path.exists(seq_key) and os.path.exists(seq_key_pc):
                removed_mask.append(True)
            else:
                removed_mask.append(False)
                continue
            # append into list
            seq_keys.append(seq_key)
            seq_keys_pc.append(seq_key_pc)
            seq_idxs.append([seq_idx])
        return removed_mask, np.asarray(seq_idxs), seq_keys, seq_keys_pc

    def __len__(self):
        return len(self.qImages)

    def __repr__(self):
        return  (f"{self.__class__.__name__} pair-wise - #type: {self.mode} - # contrast_mode: {self.contrast_mode} - #database: {len(self.dbImages)} - #queries: {len(self.qIdx)}")

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_info(self):
        return self

    def _get_item_normal(self, idx):
        '''
            fetch query image, corresponding postives and negatives, idxes in current sequence data
        '''
        # load images and pcs into list
        query = self.transform(Image.open(self.qImages[idx]).convert("RGB")).unsqueeze(0) # 1 C H W
        query_pc = load_pc_files([self.qPcs[idx]], pts_num=self.points_num) # 1 N 3
        if self.contrast_mode: # return pair data
            return {'query_img': query, 'query_pc': query_pc}
        pos = np.where(self.query_is_pos_mask[idx, :])[0]
        np.random.shuffle(pos)
        pos_files_img = []
        pos_files_pc = []
        act_num_pos = len(pos)
        pos_idx = []
        # this is for the queries that with insufficient positives!!!
        if act_num_pos == 0:
            return self.__getitem__(idx+1)
        for i in range(self.nPos):
            pidx = i % act_num_pos
            pos_files_img.append(self.dbImages[pos[pidx]])
            pos_files_pc.append(self.dbPcs[pos[pidx]])
            pos_idx.append(pos[pidx])
        # stack postives
        positives_imgs = [self.transform(Image.open(file).convert("RGB")) for file in pos_files_img]
        positives_imgs = torch.stack(positives_imgs, 0) # nPos C H W
        positives_pcs = load_pc_files(pos_files_pc, pts_num=self.points_num) # nPos N 3
        return {
                'query_idx': idx,
                'query_img': query,
                'query_pc': query_pc,
                'pos_idx': np.stack(pos_idx, 0),
                'positives_imgs': positives_imgs,
                'positives_pcs': positives_pcs,
                'is_pos': self.query_is_pos_mask[idx, :],
                'neg_idx': self.query_is_neg_mask[idx, :],
                }

    def _get_item_neighbor(self, idx):
        """fetch image, corresponding point cloud and idx
        Args:
            idx (int): sequence index.
            image (torch.Tensor) image.
            points (torch.Tensor) point cloud.
            points_feats (torch.Tensor) empty place holder.
        """
        data_dict = {}
        data_dict['idx'] = idx
        # image
        image = self.transform(Image.open(self.qImages[idx]).convert("RGB")) # C H W
        data_dict['image'] = image
        # pc
        points = load_pc_file_fix_size(self.qPcs[idx], pts_limit=self.point_limit)#, augment=self.mode=='train') # N 3
        data_dict['query_img'] = points
        # here the input feature is empty infact
        data_dict['query_pc'] = np.ones((points.shape[0], 1), dtype=np.float32)
        return data_dict

    def _get_item_proxy_image(self, idx):
        '''
            fetch query image, corresponding postives and negatives, idxes in current sequence data
            point cloud are proxied as images
        '''
        # load images and pcs into list
        query = self.transform_proxy(Image.open(self.qImages[idx]).convert("RGB")).unsqueeze(0) # 1 C H W
        query_pc = loadCloudFromBinary(self.qPcs[idx]) # 1 N 3
        lidar_image = createRangeImage(query_pc, True)
        query_pc = self.transform_proxy(Image.fromarray(lidar_image)).float().unsqueeze(0) # H W C(3)
        if self.contrast_mode: # return pair data
            return {'query_img': query, 'query_pc': query_pc}
        pos = np.where(self.query_is_pos_mask[idx, :])[0]
        np.random.shuffle(pos)
        pos_files_img = []
        pos_files_pc = []
        act_num_pos = len(pos)
        pos_idx = []
        # this is for the queries that with insufficient positives!!!
        if act_num_pos == 0:
            return self.__getitem__(idx+1)
        for i in range(self.nPos):
            pidx = i % act_num_pos
            pos_files_img.append(self.dbImages[pos[pidx]])
            pos_files_pc.append(self.dbPcs[pos[pidx]])
            pos_idx.append(pos[pidx])
        # stack postives
        positives_imgs = [self.transform(Image.open(file).convert("RGB")) for file in pos_files_img]
        positives_imgs = torch.stack(positives_imgs, 0) # nPos C H W
        positives_pcs = [self.transform_proxy(Image.open(file).convert("RGB")) for file in pos_files_pc]
        positives_pcs = torch.stack(positives_pcs, 0) # nPos C H W
        return {
                'query_idx': idx,
                'query_img': query,
                'query_pc': query_pc,
                'pos_idx': np.stack(pos_idx, 0),
                'positives_imgs': positives_imgs,
                'positives_pcs': positives_pcs,
                'is_pos': self.query_is_pos_mask[idx, :],
                'neg_idx': self.query_is_neg_mask[idx, :],
                }


    def __getitem__(self, idx):
        if self.point_cloud_proxy == "points":
            if self.get_neighbor:
                return self._get_item_neighbor(idx)
            else:
                return self._get_item_normal(idx)
        elif self.point_cloud_proxy == "bev_image": # TODO
            return self._get_item_proxy_image(idx)
        elif self.point_cloud_proxy == "range_image": # TODO
            return self._get_item_proxy_image(idx)
        else:
            raise NotImplementedError(f'Sorry, <{self.point_cloud_proxy}> is not implemented!')

    @staticmethod
    def collate_fn(self, batch):
        pass
        # """
        #     Create mini-batch tensors from the list of tuples (image, pc, idx).
        # Args:
        # batch: list of tuple (image, pc, idx).
        #     - image: torch tensor of shape (3, H, W).
        #     - pc: torch tensor of shape (N, 3).
        #     - idx: negative number
        # Returns:
        #     query: torch tensor of shape (batch_size, 3, h, w).
        #     positive: torch tensor of shape (batch_size, 3, h, w).
        #     negatives: torch tensor of shape (batch_size, n, 3, h, w).
        # """
        # batch = list(filter(lambda x: x is not None, batch))
        # # zip single triplet to batches
        # image, pc, idx = zip(*batch)
        # image = data.dataloader.default_collate(image)
        # idx = data.dataloader.default_collate(idx)
        # query_pc = data.dataloader.default_collate(query_pc)
        # positive_pc = data.dataloader.default_collate(positive_pc)
        # negatives_pcs = data.dataloader.default_collate(negatives_pcs)
        # negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives_imgs])
        # negatives_imgs = torch.cat(negatives_imgs, 0)
        
        # indices = list(itertools.chain(*indices))
        # return image, pc, idx