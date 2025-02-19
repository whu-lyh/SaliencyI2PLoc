
import pickle

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from tqdm import tqdm

from tools import builder
from utils.logger import print_log


@torch.no_grad()
def get_feature(args, base_model, dataloader, aggregator_config):
    """get feature from model which could avoids the memory overflow on large test dataset

    Args:
        args (dict): basic paramsters
        base_model (model): model weights
        dataloader (torch.utils.data.DataLoader): dataloader
        aggregator_config (Easydict): aggregator type is required
    Returns:
        return the corresponding features (torch.Tensor)
    """
    img_feat = []
    pc_feat = []
    # for each batch iteration
    for _, (input_img, input_pc, _) in enumerate(tqdm(dataloader, 
                            leave=False, desc='Feature generation iter'.rjust(15)), start=1):
        # directly fetch global feature
        output_feat_img = base_model.fetch_feat_img(input_img.to(args.local_rank), aggregator_config.image_aggregator)
        output_feat_pc = base_model.fetch_feat_pc(input_pc.to(args.local_rank), aggregator_config.pc_aggregator)
        img_feat.append(output_feat_img.detach())
        pc_feat.append(output_feat_pc.detach())
        # release memory
        del output_feat_img, output_feat_pc
    del input_img, input_pc
    img_feat = torch.cat(img_feat, dim=0).cpu().numpy().astype('float32')
    pc_feat = torch.cat(pc_feat, dim=0).cpu().numpy().astype('float32')
    return img_feat, pc_feat

class MericEvaluator:
    def __init__(self, gt, path:str, model_name:str, dataset_name:str):
        self.file_path = path
        self.model_name = model_name
        self.dataset_name = dataset_name
        # target metric topk
        self.n_values = [1, 5, 10, 15, 20] # the 20+ could be treated as Recall@1%? Top20+ is meaningless
        self.max_num_db = max(self.n_values)
        self.recalls = []
        self.recalls_dict = {} # save in dict
        self.precisions = []
        # gt (list): ground truth which should be obtained from dataset (positives of each query)
        self.gt = gt
        # True, indicates that the similarity is calculated and should be inversed when pr curve called
        self.treat_sim_as_dist = False
    
    def get_topk_torch(self, qFeat, dbFeat):
        """get the top k based on consine similarity for torch tensor
            get the same result as get_topk_numpy()

        Args:
            qFeat (torch.Tensor): image feature embedding
            dbFeat (torch.Tensor): point cloud feature embedding

        Returns:
            similarity and corresponding nearest indices
        """
        self.treat_sim_as_dist = True
        # self.max_num_db = dbFeat.shape[1]
        dot_similarity = qFeat @ dbFeat.T
        sim, indices = torch.topk(dot_similarity, self.max_num_db)
        self.dists, self.preds = np.array(sim), np.array(indices)
        return self.dists, self.preds

    def get_topk_numpy(self, qFeat, dbFeat):
        """get the top k based on consine similarity for numpy array
            get the same result as get_topk_torch()
        Args:
            qFeat (np.array): query(image) feature embedding
            dbFeat (np.array): database(point cloud) feature embedding

        Returns:
            (np.array) distance and corresponding indices
        """
        from sklearn.metrics.pairwise import cosine_similarity
        self.treat_sim_as_dist = True
        # self.max_num_db = dbFeat.shape[1]
        cos_mat = cosine_similarity(qFeat, dbFeat) # same as qFeat @ dbFeat.T
        preds = []
        dists = []
        for index_r in range(cos_mat.shape[0]):
            ind = np.argsort(cos_mat[index_r, :])
            ind_inv = ind[::-1]
            preds.append(ind_inv[:self.max_num_db])
            dists.append(cos_mat[index_r, ind_inv[:self.max_num_db]])
        self.preds, self.dists = np.array(preds), np.array(dists)
        return self.dists, self.preds

    def get_nn_faiss(self, qFeat, dbFeat, faiss_gpu:bool=False):
        """get nearest neighbors based on faiss library

        Args:
            qFeat (np.array): query data feature, size should be B, D (search could be done in a batch)
            dbFeat (np.array): database data feature
            faiss_gpu (bool, optional): if to switch GPU version. Defaults to False.
            # TODO, THE DISTANCE IS NOT RETURNED!!!Do not call this while ploting pr curve
        Returns:
            (np.array) distance and corresponding indices
        """
        self.treat_sim_as_dist = True
        # feature dimension
        feat_dim = qFeat.shape[1]
        # build index
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, feat_dim, flat_config)
        else:
            faiss_index = faiss.IndexFlatL2(feat_dim)
        # add references
        faiss_index.add(dbFeat)
        # len(dbFeat) also for other metrics such as precision-recall curve and average precision
        assert len(dbFeat) > 0
        # search for queries, noted that the results are sorted
        # len(dbFeat) or max(self.max_num_db) get the same results
        dists, preds = faiss_index.search(qFeat, self.max_num_db)
        self.preds, self.dists = np.array(preds), np.array(dists)
        return self.dists, self.preds

    def get_nn_knn(self, qFeat, dbFeat):
        """get nearest neighbors based on sklearn library

        Args:
            qFeat (np.array): query data feature
            dbFeat (np.array): database data feature

        Returns:
            (np.array) distance and corresponding nearest indices
        """
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_jobs=1)
        knn.fit(dbFeat)
        self.dists, self.preds = knn.kneighbors(qFeat, self.max_num_db)  # dist: small to large
        return self.dists, self.preds

    def get_preds(self, verbose=False):
        if verbose:
            print('preds: ', self.preds)#.shape)
            print('dists: ', self.dists)#.shape)
        return self.preds, self.dists

    def get_recall_at_n(self, preds=None, pretty_table_save:bool=False):
        """get the recall based on the query and database data feature

        Args:
            pretty_table_save (bool): if to save the result at table mode

        Returns:
            dict: recall at different topk
        """
        if preds is None:
            preds= self.preds
        correct_at_n = np.zeros(len(self.n_values))
        for qIx, pred in enumerate(preds):
            for i, n in enumerate(self.n_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], self.gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        # len(preds) == num_query, the dataset should make sure that 
        # each query has corresponding postives
        num_query = len(preds)
        recall_at_n = correct_at_n / num_query * 100.0
        # make dict for output
        recalls = {k:v for (k, v) in zip(self.n_values, recall_at_n)}
        self.recalls_dict = recalls
        # for command output
        # for i, n in enumerate(self.n_values):
        #     print('recall@', n, ': {:.2f}\t'.format(recalls[n]), end='')
        return recalls

    def get_pr(self, save_img=False, cal_accuracy=False):
        """get precision and recall value at recall@1
            This figure suits the method whose recall@1 is high

        Args:
            save_img (bool, optional): if to save the pr figure into disk. Defaults to False.
            cal_accuracy (bool, optional): if to calculate the accuracy. Defaults to False.

        Returns:
            narray: precision and recall array
        """
        preds, dists = self.preds, self.dist
        dists_m = np.around(dists[:, 0], 2)
        dists_u = np.array(list(set(dists_m)))
        dists_u = np.sort(dists_u) # small -> large
        # calculate the area of pr
        recalls = []
        precisions = []
        accuracies = []
        for th in tqdm(dists_u, total=len(dists_u), leave=False):
            TPCount = 0
            FPCount = 0
            FNCount = 0
            TNCount = 0
            if self.treat_sim_as_dist: # for similarity as threshold
                for index_q in range(dists.shape[0]):
                    # Positive
                    if dists[index_q, 0] > th:
                        # True
                        if np.any(np.in1d(preds[index_q, 0], self.gt[index_q])):
                            TPCount += 1
                        else:
                            FPCount += 1
                    else:
                        if np.any(np.in1d(preds[index_q, 0], self.gt[index_q])):
                            FNCount += 1
                        else:
                            TNCount += 1
            else: # for distance as threshold
                for index_q in range(dists.shape[0]):
                    # Positive
                    if dists[index_q, 0] < th:
                        # True
                        if np.any(np.in1d(preds[index_q, 0], self.gt[index_q])):
                            TPCount += 1
                        else:
                            FPCount += 1
                    else:
                        if np.any(np.in1d(preds[index_q, 0], self.gt[index_q])):
                            FNCount += 1
                        else:
                            TNCount += 1
            assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
            if TPCount + FNCount == 0 or TPCount + FPCount == 0:
                continue
            recall = TPCount / (TPCount + FNCount)
            precision = TPCount / (TPCount + FPCount)
            if cal_accuracy:
                if TPCount + FPCount + TNCount + FNCount == 0:
                    accuracy = 0
                else:
                    accuracy = (TPCount + TNCount) / (TPCount + FPCount + TNCount + FNCount)
            recalls.append(recall)
            precisions.append(precision)
            if cal_accuracy:
                accuracies.append(accuracy)
        self.recalls = recalls
        self.precisions = precision
        if cal_accuracy:
            return recalls, precisions, accuracies
        else:
            return recalls, precisions

    def get_f1score(self):
        """calculate the F1 score

        Returns:
            float: Max F1 score
        """
        recalls = np.array(self.recalls)
        precisions = np.array(self.precisions)
        ind = np.argsort(recalls)
        recalls = recalls[ind]
        precisions = precisions[ind]
        f1s = []
        for index_j in range(len(recalls)):
            p_m_r = precisions[index_j] * recalls[index_j]
            p_p_r = precisions[index_j] + recalls[index_j]
            if p_p_r > 0:
                f1 = 2 * p_m_r / (p_p_r)
                f1s.append(f1)
        return max(f1s)

    def get_ap(self):
        """calculate the Average Precision
            Detail refer to paper "Evaluation of Object Proposals and ConvNet Features for Landmark-based Visual Place Recognition"

        Returns:
            float: Average Precision
        """
        recalls = np.array(self.recalls)
        precisions = np.array(self.precisions)
        ind = np.argsort(recalls)
        recalls = recalls[ind]
        precisions = precisions[ind]
        ap = 0
        for index_j in range(len(recalls) - 1):
            ap += precisions[index_j] * (recalls[index_j + 1] - recalls[index_j])
        return ap

def save_gt(gt:list, gt_dist:list, eval_dataset:torch.utils.data.Dataset, file_path:str):
    '''
        The image branch is saved for clearification
    '''
    # print_log(type(gt), logger=logger) # <class 'list'>
    gt_index = []
    gt_lists = []
    import json
    import os
    for i in range(len(gt)):
        gt_index.append(os.path.basename(eval_dataset.qImages[i]))
        pos_indices = gt[i]
        pics = [os.path.basename(eval_dataset.dbImages[p]) for p in pos_indices]
        info_dict = {'pics': pics, 'dists': gt_dist[i].tolist()}
        gt_lists.append(info_dict)
    gt_dic = dict(zip(gt_index, gt_lists))
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(gt_dic, indent=4))


@torch.no_grad()
def validate(base_model, base_dataset, args, config, logger, writer=None, 
            wandber=None, epoch_num=0, write_tboard=True):
    """single sequence evaluation is supported currently

    Args:
        base_model (_type_): input model weight
        base_dataset (_type_): dataset to be validated or tested
        args (_type_): macro arguments
        config (_type_): model, dataset configuration info
        logger (_type_): logger to ascii file
        writer (_type_, optional): TF writer. Defaults to None.
        wandber (_type_, optional): wandb writter. Defaults to None.
        epoch_num (int, optional): the current epoch number. Defaults to 0.
        write_tboard (bool, optional): if to print at the TF board. Defaults to True.

    Returns:
        dict: return the cross modality recall at recall@K
    """
    print_log('Extracting features for query and database data ...', logger=logger)
    # generate dataloader for query and database data
    pwdataset, _, dataloader = builder.dataloader_builder(args, base_dataset, mode="pairwise")
    # freeze BN and dropout while validation
    base_model.eval()
    # without gradient and save GPU memory
    with torch.no_grad():
        feat_img, feat_pc = get_feature(args, base_model, dataloader, config.model.aggregator)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_log('Feature extraction finished ...', logger=logger)
    # The gt indices are found by knn from database images
    gt = base_dataset.all_pos_indices
    gt_dist = base_dataset.all_pos_dists
    # build index
    me = MericEvaluator(gt=gt, path="./", dataset_name=base_dataset.get_dataset_name(), model_name="SaliencyI2PLoc")
    n_values = me.n_values
    if args.debug or args.test:
        val_set_name = base_dataset.get_dataset_name() + "_" + str(base_dataset.sequences[0])
        save_gt(gt, gt_dist, base_dataset, file_path='{}/{}_ground_truth.json'.format(args.experiment_path, val_set_name))
        with open('{}/features_{}.pickle'.format(args.experiment_path, val_set_name), 'wb') as f:
            # feature for result plot, data file info for visualization
            feature = {'feat_img': feat_img, 'feat_pc': feat_pc, 'gt': gt, 'num_query': pwdataset.num_query, 
                       'images': pwdataset.images_list, 'pcs': pwdataset.pcs_list}
            pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)
            print_log('Save features to features_{}.pickle.'.format(val_set_name), logger=logger)
    # knn searching
    predictions = {} # each element should be a np.array contains whole query's predictions
    des = ['2D->2D', '2D->3D', '3D->2D', '3D->3D']
    qEndPosTot = 0
    dbEndPosTot = 0
    # qEndPosList storing the query data indices, while dbEndPosList for database data indices
    # all data are loaded from predifined csv files
    for sequence_id, (qEndPos, dbEndPos) in enumerate(zip(base_dataset.qEndPosList, base_dataset.dbEndPosList)):
        # for each sequence, we check the multi metrics
        for i in range(len(des)):
            if i == 0: # 2d->2d
                qTest = feat_img[:pwdataset.num_query]
                dbTest = feat_img[pwdataset.num_query:]
            if i == 1: # 2d->3d
                qTest = feat_img[:pwdataset.num_query]
                dbTest = feat_pc[pwdataset.num_query:]
            if i == 2: # 3d->2d
                qTest = feat_pc[:pwdataset.num_query]
                dbTest = feat_img[pwdataset.num_query:]
            if i == 3: # 3d->3d
                qTest = feat_pc[:pwdataset.num_query]
                dbTest = feat_pc[pwdataset.num_query:]
            # add specific data indices in database for searching, due to all sequences used for test are sotred in one list
            # if there is only one test sequence, this operation is useless, but for multi-sequences!!!
            _, preds = me.get_nn_faiss(qFeat=qTest[qEndPosTot:qEndPosTot + qEndPos, :], dbFeat=dbTest[dbEndPosTot:dbEndPosTot + dbEndPos, :])
            # stack the results for all test sequences
            if sequence_id == 0:
                predictions[i] = preds # size should be (total query number of whole sequences, max(n_values))
            else:
                predictions[i] = np.vstack((predictions[i], preds)) # size should be (total query number of whole sequences, max(n_values))
        # move to next test indices intervals
        qEndPosTot += qEndPos
        dbEndPosTot += dbEndPos
    # result check compared with GT files
    # For each query, there might be too many search results. When calculate the recall, if the fetched result > total number of gt, it is treated as successful
    recall_at_n = {} # double dict container
    for test_index in range(len(des)):
        recall_at_n[test_index] = me.get_recall_at_n(predictions[test_index])
    all_recalls = {}
    if args.local_rank == 0:
        print_log("Performance on dataset {} at epoch {}".format(base_dataset.dataset_name, str(epoch_num)), logger=logger)
        for des_i, name in enumerate(des):
            for i, n in enumerate(n_values):
                if write_tboard and writer is not None and args.local_rank == 0:
                    writer.add_scalar(('{}/Recall@' + str(n)).format(name), recall_at_n[des_i][n], epoch_num)
                if write_tboard and wandber is not None:
                    wandber.log({('{}/Recall@' + str(n)).format(name): recall_at_n[des_i][n]})
                all_recalls[n] = recall_at_n[des_i][n]
            message = '%s Recall: %s' % (name, ['@%i = %.4f' % (k, v/100) for k,v in all_recalls.items()])
            print_log(message, logger=logger)
    # only return 2d->3d metric
    all_recalls_2d_3d = {}
    for i, n in enumerate(n_values):
        all_recalls_2d_3d[n] = recall_at_n[1][n] # 2D->3D task only
    return all_recalls_2d_3d
