import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    sys.path.append(os.path.dirname(p))

from SaliencyI2PLoc.extentions.hyptorch.nn import ToPoincare
from SaliencyI2PLoc.extentions.hyptorch.pmath import dist_matrix


def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class DistilLoss(nn.Module):
    """ Generalized relational knowledge distillation in three manifolds modified from paper DistilVPR
    """
    def __init__(self, config):
        super().__init__()
        self.loss_type = config["loss_type"] # 'MSE'
        self.logit_norm_fn = config["logit_norm_fn"] # 'l2'
        self.logit_mapping_fn = config["logit_mapping_fn"] # 'identity'
        self.curvature = config["curvature"] # 0.1 by default from hpy_metric

        self.crosslogitsimloss_divide = True # True=use cosine similarity

        self.crosslogitdistloss_weight_st2ss = config["crosslogitdistloss_weight_st2ss"] # 0 for discarding st2ss
        self.crosslogitdistloss_weight_tt2st = config["crosslogitdistloss_weight_tt2st"] # 0 for discarding tt2st
        self.crosslogitsimloss_weight_st2ss = config["crosslogitsimloss_weight_st2ss"] # 0 for discarding st2ss
        self.crosslogitsimloss_weight_tt2st = config["crosslogitsimloss_weight_tt2st"] # 0 for discarding tt2st
        self.crosslogitgeodistloss_weight_st2ss = config["crosslogitgeodistloss_weight_st2ss"] # 0 for discarding st2ss
        self.crosslogitgeodistloss_weight_tt2st = config["crosslogitgeodistloss_weight_tt2st"] # 0 for discarding tt2st
        self.rkdgloss_weight = config["rkdgloss_weight"] 
        self.distance_weight = config["distance_weight"] # 0 for discard Euclidean distance
        self.angle_weight = config["angle_weight"] # 0 for discarding cosine distance
        self.geodesic_weight = config["geodesic_weight"] # 0 for discarding hyperbolic distance
        # other setting
        self.init()
    
    def init(self):
        if self.loss_type == 'MSE':
            self.distil_loss_fn = torch.nn.MSELoss()
        elif self.loss_type == 'L1':
            self.distil_loss_fn = torch.nn.L1Loss()
        elif self.loss_type == 'KLDiv':
            self.distil_loss_fn = torch.nn.KLDivLoss()
        else: 
            raise NotImplementedError

        if self.logit_norm_fn == 'softmax':
            self.logit_norm_fn_stu = F.softmax
            self.logit_norm_fn_tea = F.softmax
        elif self.logit_norm_fn == 'log_softmax':
            self.logit_norm_fn_stu = F.log_softmax
            self.logit_norm_fn_tea = F.log_softmax
        elif self.logit_norm_fn == 'identity':
            self.logit_norm_fn_stu = lambda x, dim: x
            self.logit_norm_fn_tea = lambda x, dim: x
        elif self.logit_norm_fn == 'l2':
            self.logit_norm_fn_stu = F.normalize
            self.logit_norm_fn_tea = F.normalize # p=2, dim=1x
        else:
            raise NotImplementedError
        
        if self.logit_mapping_fn == 'identity':
            self.logit_mapping_fns_stu = [lambda x: x]
            self.logit_mapping_fns_tea = [lambda x: x]
        elif self.logit_mapping_fn == 'topoincare':
            self.logit_mapping_fns_stu = [ToPoincare(c=self.curvature, ball_dim=256, riemannian=False, clip_r=None)]
            self.logit_mapping_fns_tea = [ToPoincare(c=self.curvature, ball_dim=256, riemannian=False, clip_r=None)]
        elif self.logit_mapping_fn == 'id_topoin':
            self.logit_mapping_fns_stu = [lambda x: x,
                ToPoincare(c=self.curvature, ball_dim=256, riemannian=False, clip_r=None)
            ]
            self.logit_mapping_fns_tea = [lambda x: x,
                ToPoincare(c=self.curvature, ball_dim=256, riemannian=False, clip_r=None)
            ]
        else:
            raise NotImplementedError

    # ----------------- cross logit dist ----------------- #
    def compute_crosslogitdist_loss(self, logit_stu, logit_tea):
        weights_st2ss = self.crosslogitdistloss_weight_st2ss
        weights_st2ss = weights_st2ss.split('_')
        weights_st2ss = [float(w) for w in weights_st2ss]
        assert len(weights_st2ss) == len(self.logit_mapping_fns_stu)

        weights_tt2st = self.crosslogitdistloss_weight_tt2st
        weights_tt2st = weights_tt2st.split('_')
        weights_tt2st = [float(w) for w in weights_tt2st]
        assert len(weights_tt2st) == len(self.logit_mapping_fns_stu)

        logit_stu_normalized = self.logit_norm_fn_stu(logit_stu, dim=1)
        logit_tea_normalized = self.logit_norm_fn_tea(logit_tea, dim=1)

        distil_loss_st2ss = 0
        distil_loss_tt2st = 0
        i_fn = 0

        for logit_mapping_fn_stu, logit_mapping_fn_tea in zip(self.logit_mapping_fns_stu, self.logit_mapping_fns_tea):

            logit_stu_normalized_mapped = logit_mapping_fn_stu(logit_stu_normalized)
            logit_tea_normalized_mapped = logit_mapping_fn_tea(logit_tea_normalized)
            # for full matrix
            diffmat_ss = logit_stu_normalized_mapped.unsqueeze(0) - logit_stu_normalized_mapped.unsqueeze(1)
            distmat_ss = torch.norm(diffmat_ss, p=2, dim=-1)
            diffmat_st = logit_stu_normalized_mapped.unsqueeze(0) - logit_tea_normalized_mapped.unsqueeze(1)
            distmat_st = torch.norm(diffmat_st, p=2, dim=-1)
            diffmat_tt = logit_tea_normalized_mapped.unsqueeze(0) - logit_tea_normalized_mapped.unsqueeze(1)
            distmat_tt = torch.norm(diffmat_tt, p=2, dim=-1)

            _distil_loss_st2ss = self.distil_loss_fn(distmat_ss, distmat_st) * weights_st2ss[i_fn]
            _distil_loss_tt2st = self.distil_loss_fn(distmat_st, distmat_tt) * weights_tt2st[i_fn]

            distil_loss_st2ss += _distil_loss_st2ss
            distil_loss_tt2st += _distil_loss_tt2st

            i_fn += 1

        return distil_loss_st2ss, distil_loss_tt2st

    # ----------------- cross logit sim ----------------- #
    def compute_crosslogitsim_loss(self, logit_stu, logit_tea):
        weights_st2ss = self.crosslogitsimloss_weight_st2ss
        weights_st2ss = weights_st2ss.split('_')
        weights_st2ss = [float(w) for w in weights_st2ss]
        assert len(weights_st2ss) == len(self.logit_mapping_fns_stu)

        weights_tt2st = self.crosslogitsimloss_weight_tt2st
        weights_tt2st = weights_tt2st.split('_')
        weights_tt2st = [float(w) for w in weights_tt2st]
        assert len(weights_tt2st) == len(self.logit_mapping_fns_stu)

        logit_stu_normalized = self.logit_norm_fn_stu(logit_stu, dim=1)
        logit_tea_normalized = self.logit_norm_fn_tea(logit_tea, dim=1)

        distil_loss_st2ss = 0
        distil_loss_tt2st = 0
        i_fn = 0

        for logit_mapping_fn_stu, logit_mapping_fn_tea in zip(self.logit_mapping_fns_stu, self.logit_mapping_fns_tea):
            logit_stu_normalized_mapped = logit_mapping_fn_stu(logit_stu_normalized)
            logit_tea_normalized_mapped = logit_mapping_fn_tea(logit_tea_normalized)

            norm_s = torch.norm(logit_stu_normalized_mapped, p=2, dim=1, keepdim=True) # [b,1]
            normsqmat_ss = norm_s @ norm_s.T # [b,b]
            norm_t = torch.norm(logit_tea_normalized_mapped, p=2, dim=1, keepdim=True)
            normsqmat_tt = norm_t @ norm_t.T
            normsqmat_st = norm_s @ norm_t.T
            # for full matrix
            simmat_ss = logit_stu_normalized_mapped @ logit_stu_normalized_mapped.T
            simmat_st = logit_stu_normalized_mapped @ logit_tea_normalized_mapped.T
            simmat_tt = logit_tea_normalized_mapped @ logit_tea_normalized_mapped.T

            if self.crosslogitsimloss_divide:
                simmat_ss = simmat_ss / normsqmat_ss
                simmat_st = simmat_st / normsqmat_st
                simmat_tt = simmat_tt / normsqmat_tt

            _distil_loss_st2ss = self.distil_loss_fn(simmat_ss, simmat_st) * weights_st2ss[i_fn]
            _distil_loss_tt2st = self.distil_loss_fn(simmat_st, simmat_tt) * weights_tt2st[i_fn]

            distil_loss_st2ss += _distil_loss_st2ss
            distil_loss_tt2st += _distil_loss_tt2st

            i_fn += 1
        
        return distil_loss_st2ss, distil_loss_tt2st

    # ----------------- cross logit geodesic ----------------- #
    def compute_crosslogitgeodist_loss(self, logit_stu, logit_tea):
        # geodesic distance is for poincare ball
        _, feat_dim = logit_stu.shape
        logit_mapping_fns_stu = [ToPoincare(c=self.curvature, ball_dim=feat_dim, riemannian=False, clip_r=None)]
        logit_mapping_fns_tea = [ToPoincare(c=self.curvature, ball_dim=feat_dim, riemannian=False, clip_r=None)]

        weights_st2ss = self.crosslogitgeodistloss_weight_st2ss
        weights_st2ss = weights_st2ss.split('_')
        weights_st2ss = [float(w) for w in weights_st2ss]
        
        weights_tt2st = self.crosslogitgeodistloss_weight_tt2st
        weights_tt2st = weights_tt2st.split('_')
        weights_tt2st = [float(w) for w in weights_tt2st]

        logit_stu_normalized = self.logit_norm_fn_stu(logit_stu, dim=1)
        logit_tea_normalized = self.logit_norm_fn_tea(logit_tea, dim=1)

        distil_loss_st2ss = 0
        distil_loss_tt2st = 0
        i_fn = 0

        for logit_mapping_fn_stu, logit_mapping_fn_tea in zip(logit_mapping_fns_stu, logit_mapping_fns_tea):
            logit_stu_normalized_mapped = logit_mapping_fn_stu(logit_stu_normalized)
            logit_tea_normalized_mapped = logit_mapping_fn_tea(logit_tea_normalized)
            # for full matrix
            geodistmat_ss = dist_matrix(logit_stu_normalized_mapped, logit_stu_normalized_mapped, c=self.curvature) # [b,b]
            geodistmat_st = dist_matrix(logit_stu_normalized_mapped, logit_tea_normalized_mapped, c=self.curvature)
            geodistmat_tt = dist_matrix(logit_tea_normalized_mapped, logit_tea_normalized_mapped, c=self.curvature)

            _distil_loss_st2ss = self.distil_loss_fn(geodistmat_ss, geodistmat_st) * weights_st2ss[i_fn]
            _distil_loss_tt2st = self.distil_loss_fn(geodistmat_st, geodistmat_tt) * weights_tt2st[i_fn]

            distil_loss_st2ss += _distil_loss_st2ss
            distil_loss_tt2st += _distil_loss_tt2st

            i_fn += 1

        return distil_loss_st2ss, distil_loss_tt2st

    # ----------------- generalized relational KD ----------------- #
    def compute_rkdg_loss(self, logit_stu, logit_tea, squared=False, eps=1e-12):

        assert logit_stu.shape == logit_tea.shape
        assert len(logit_stu.shape) == 2

        B, D = logit_stu.shape
        stu = logit_stu.view(B, -1)
        tea = logit_tea.view(B, -1)

        # RKD distance loss
        # with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

        d = _pdist(stu, squared, eps)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        # with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = stu.unsqueeze(0) - stu.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        # RKD geodesic distance loss
        to_poincare = ToPoincare(c=self.curvature, ball_dim=D, riemannian=False, clip_r=None)
        logit_s_poincare = to_poincare(stu)
        logit_t_poincare = to_poincare(tea)
        geodistmat_ss = dist_matrix(logit_s_poincare, logit_s_poincare, c=self.curvature)
        geodistmat_tt = dist_matrix(logit_t_poincare, logit_t_poincare, c=self.curvature)
        mean_ss = geodistmat_ss[geodistmat_ss > 0].mean()
        mean_tt = geodistmat_tt[geodistmat_tt > 0].mean()
        geodistmat_ss = geodistmat_ss / mean_ss
        geodistmat_tt = geodistmat_tt / mean_tt
        loss_g = F.smooth_l1_loss(geodistmat_ss, geodistmat_tt)
        # Total RKD loss from three-folds
        loss = self.distance_weight * loss_d + self.angle_weight * loss_a + self.geodesic_weight * loss_g
        return loss

    def forward(self, logit_stu, logit_tea, positives_mask=None, negatives_mask=None, adaptor=None):
        assert logit_stu.shape == logit_tea.shape
        assert len(logit_stu.shape) == 2
        distil_loss = 0
        # # crosslogitdist distillation
        # crosslogitdistloss_st2ss, crosslogitdistloss_tt2st = self.compute_crosslogitdist_loss(logit_stu, logit_tea)
        # distil_loss += crosslogitdistloss_st2ss
        # distil_loss += crosslogitdistloss_tt2st
        # # crosslogitsim distillation
        # crosslogitsimloss_st2ss, crosslogitsimloss_tt2st = self.compute_crosslogitsim_loss(logit_stu, logit_tea)
        # distil_loss += crosslogitsimloss_st2ss
        # distil_loss += crosslogitsimloss_tt2st
        # # crosslogitgeodist distillation
        # crosslogitgeodistloss_st2ss, crosslogitgeodistloss_tt2st = self.compute_crosslogitgeodist_loss(logit_stu, logit_tea)
        # distil_loss += crosslogitgeodistloss_st2ss
        # distil_loss += crosslogitgeodistloss_tt2st
        # relational knowledge distillation
        distil_loss += (self.compute_rkdg_loss(logit_stu, logit_tea) * self.rkdgloss_weight)
        return distil_loss