
import torch
import torch.nn as nn
import torch.nn.functional as F

from SaliencyI2PLoc.build import MODELS
from SaliencyI2PLoc.criterion import DistilLoss, InfoNCE_vanilla
from SaliencyI2PLoc.modules import (PointNetVLADEncoder,
                                    PointTransformerEncoder,
                                    SphereNetVLADEncoder,
                                    VisionTransformerEncoder)
from tools import builder


class TokenAttentionHacker(nn.Module):
    def __init__(self, attention, embed_dim, return_attention=False):
        """hack the last attention block without gradient backward

        Args:
            attention (attention block): attention block
            embed_dim (int): the dimension of feature for q k v.
            return_attention (bool, optional): if to return the attention layer
        """
        super().__init__()
        attention = attention.attn
        self.num_heads = attention.num_heads
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = attention.qkv
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.)

        self.return_attention = return_attention

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # batchsize, num_heads, num_patches+1, num_patches+1
        attn = attn.softmax(dim=-1)

        if self.return_attention:
            return attn

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@MODELS.register_module()
class CrossI2P(nn.Module):
    """
        Siamese atchitecture
        Image branch feature extractor
        Pc branch feature extractor
        Feature extractor, Transformer encoder
    """
    def __init__(self, config):
        super().__init__()
        self.proxy = config.proxy
        if config.baseline:
            # *baseline encoder
            self.image_encoder = SphereNetVLADEncoder.from_config(config.image_encoder._base_)
            self.pc_encoder = PointNetVLADEncoder.from_config(config.pc_encoder._base_)
        elif self.proxy:
            # *point proxy mode
            self.image_encoder = VisionTransformerEncoder.from_config(config.image_encoder._base_)
            self.pc_encoder = VisionTransformerEncoder.from_config(config.pc_encoder._base_)
        else:
            # *Transformer encoder
            self.image_encoder = VisionTransformerEncoder.from_config(config.image_encoder._base_)
            self.image_encoder.reset_classifier(0) # only for transformer based encoder
            self.pc_encoder = PointTransformerEncoder.from_config(config.pc_encoder._base_)
            self.pc_encoder.reset_classifier(0) # only for transformer based encoder
        # reproject the feature dimension from Transformer to fix-length as the global feature
        self.embed_dim = config.global_feat_dim
        # project the visual feature to fix dimension
        vision_width = self.image_encoder.embed_dim
        self.image_encoder_feature_forward = False
        if not config.baseline:
            if config.aggregator.image_aggregator.NAME != 'none':
                self.vision_proj = builder.model_builder(config.aggregator.image_aggregator._base_)
                if config.aggregator.image_aggregator.NAME in ['NetVLAD', 'NetVLADLoupe']:
                    self.image_encoder_feature_forward = True
            else:
                self.image_encoder.feat_pool = True
                self.vision_proj = nn.Linear(vision_width, self.embed_dim)
        else:
            self.vision_proj = nn.Identity()
        # project the pc feature to the fix dimension
        pc_width = self.pc_encoder.embed_dim
        self.pc_encoder_feature_forward = False
        if not config.baseline:
            if config.aggregator.pc_aggregator.NAME != 'none':
                self.pc_proj = builder.model_builder(config.aggregator.pc_aggregator._base_)
                if config.aggregator.pc_aggregator.NAME in ['NetVLAD', 'NetVLADLoupe']:
                    self.pc_encoder_feature_forward = True
            else:
                self.pc_encoder.feat_pool = True
                self.pc_proj = nn.Linear(pc_width, self.embed_dim)
        else:
            self.pc_proj = nn.Identity()

        self.do_patch_weight = config.do_patch_weight
        self.discard_cls_token = config.discard_cls_token
        # set the mode with various training pipeline
        self.mode = config.mode
        self.build_loss_func(config)

    def build_loss_func(self, config):
        if self.mode == "vanilla": # mostly for baseline
            self.criterion_query_sim = nn.PairwiseDistance()
            self.criterion_triplet = nn.TripletMarginLoss(margin=0.1 ** 0.5, p=2, reduction='sum')
        elif self.mode == "contrast":
            if config.loss_type.NAME == "InfoNCE":
                self.criterion_ce = InfoNCE_vanilla(config.loss_type._base_)
            else:
                raise NotImplementedError(f'Sorry, <{config.loss_type.NAME}> function is not implemented!')
            self.rkdg = config.loss_type.rkdg
            if self.rkdg:
                self.criterion_distil = DistilLoss(config.loss_type.rkdg_config._base_)
        else:
            raise NotImplementedError(f'Sorry, <{self.mode}> mode is not implemented!')

    def fetch_token_attention_img(self, image_embeds):
        last_block_num = 1
        top_percent = 0.6
        num_token_img = int(top_percent * image_embeds.shape[-2]) # B N C
        # the last not penultimate feature map
        atten_map_img = TokenAttentionHacker(self.image_encoder.blocks[-last_block_num], 
                                             embed_dim=self.image_encoder.embed_dim, return_attention=True).to(image_embeds.device)
        with torch.no_grad():
            attentions_img = atten_map_img(image_embeds)
            attentions_img = attentions_img[:, :, 0, :] # B num_head N N -> B num_head N
            attentions_img = torch.sum(attentions_img, dim=1)
            values_img, indices_img = torch.topk(attentions_img, num_token_img, dim=-1)
        return indices_img, attentions_img

    def fetch_token_attention_pc(self, pc_embeds):
        last_block_num = 1
        top_percent = 0.6
        num_token_pc = int(top_percent * pc_embeds.shape[-2]) # B N C
        # the last not penultimate feature map
        if self.proxy:
            atten_map_pc = TokenAttentionHacker(self.pc_encoder.blocks[-last_block_num], 
                                                embed_dim=self.pc_encoder.embed_dim, return_attention=True).to(pc_embeds.device)
        else:
            atten_map_pc = TokenAttentionHacker(self.pc_encoder.blocks.blocks[-last_block_num], 
                                                embed_dim=self.pc_encoder.embed_dim, return_attention=True).to(pc_embeds.device)
        with torch.no_grad():
            attentions_pc = atten_map_pc(pc_embeds)
            attentions_pc = attentions_pc[:, :, 0, :] # B num_head N N -> B num_head N
            attentions_pc = torch.sum(attentions_pc, dim=1)
            values_pc, indices_pc = torch.topk(attentions_pc, num_token_pc, dim=-1)
        return indices_pc, attentions_pc

    def forward_vanilla(self, samples):
        """Used for vanilla version without momentum distillation only metric learning
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, (1+1+nNeg), 3, H, W). The input images.
                - point_cloud (torch.Tensor): A tensor of shape (batch_size, (1+1+nNeg), N, 3). The input points.
                - nNeg (torch.Tensor): The number of negatives used to split the tensor
        Returns:
        """
        nNeg = samples["nNeg"]
        # query image, positive image and negative images
        image_input = samples["image"]
        batch_size, _, image_channels, H, W, = image_input.shape
        # feed raw 3 channel RGB image into image encoder
        image_input = image_input.view(-1, image_channels, H, W) # B*(1+1+nNeg), C, H, W
        if self.image_encoder_feature_forward:
            image_embeds = self.image_encoder.forward_features(image_input)
        else:
            image_embeds = self.image_encoder(image_input)
        # *No normalize the image global feature
        image_global_feat = self.vision_proj(image_embeds)

        # query pc, positive pc and negative pc submaps
        pc_input = samples["point_cloud"]
        if self.proxy:
            batch_size, _, image_channels, H, W, = pc_input.shape
            pc_input = pc_input.view(-1, image_channels, H, W) # B, 1, C, H, W -> B*1, C, H, W
        else:
            _, _, pts_num, pc_channels = pc_input.shape
            pc_input = pc_input.view((-1, 1, pts_num, pc_channels)) # B, 1, 1, N, C -> B*1, 1, N, C   # feed raw 3 channel xyz pc into pc encoder
        if self.pc_encoder_feature_forward:
            pc_embeds, _, _ = self.pc_encoder.forward_features(pc_input)
        else:
            pc_embeds = self.pc_encoder(pc_input)
        # *No normalize the pc global feature
        pc_global_feat = self.pc_proj(pc_embeds)

        # split feat for each single data
        image_global_feat = image_global_feat.view(batch_size, -1, self.embed_dim)
        global_feat_img_query, global_feat_img_pos, global_feat_img_negs = torch.split(image_global_feat, [1, 1, nNeg], dim=1)
        pc_global_feat = pc_global_feat.view(batch_size, -1, self.embed_dim)
        global_feat_pc_query, global_feat_pc_pos, global_feat_pc_negs = torch.split(pc_global_feat, [1, 1, nNeg], dim=1)

        loss_query = self.criterion_query_sim(global_feat_img_query, global_feat_pc_query)
        loss_2dto2d = self.criterion_triplet(global_feat_img_query, global_feat_img_pos, global_feat_img_negs)
        loss_2dto3d = self.criterion_triplet(global_feat_img_query, global_feat_pc_pos, global_feat_pc_negs)
        loss_3dto2d = self.criterion_triplet(global_feat_pc_query, global_feat_img_pos, global_feat_img_negs)
        loss_3dto3d = self.criterion_triplet(global_feat_pc_query, global_feat_pc_pos, global_feat_pc_negs)
        # return
        loss = loss_query + 0.1 * loss_2dto2d + loss_2dto3d + loss_3dto2d + 0.1 * loss_3dto3d
        return loss

    def forward_contrast(self, samples):
        """Used for contrastive learning mode
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, M, 3, H, W). The input images.
                - point_cloud (torch.Tensor): A tensor of shape (batch_size, M, N, 3). The input points.
        Returns:
        """
        # query image, positive image and negative images
        image_input = samples["image"]
        batch_size, _, image_channels, H, W, = image_input.shape
        # feed raw 3 channel RGB image into image encoder
        image_input = image_input.view(-1, image_channels, H, W)
        if self.image_encoder_feature_forward:
            image_embeds = self.image_encoder.forward_features(image_input) # BNC
        else:
            image_embeds = self.image_encoder(image_input) # BNC
        image_embeds = F.normalize(image_embeds, dim=-1) # BNC # normalize before projection(aggregation) layer

        # query pc, positive pc and negative pc submaps
        pc_input = samples["point_cloud"]
        if self.proxy:
            batch_size, _, image_channels, H, W, = pc_input.shape # B, 1, C, H, W -> B*1, C, H, W
            pc_input = pc_input.view(-1, image_channels, H, W)
        else:
            _, _, pts_num, pc_channels = pc_input.shape
            pc_input = pc_input.view((-1, 1, pts_num, pc_channels)) # B, 1, 1, N, C -> B*1, 1, N, C
        # feed raw 3 channel xyz pc into pc encoder
        if self.pc_encoder_feature_forward:
            if self.proxy:
                pc_embeds = self.pc_encoder.forward_features(pc_input) # BNC
            else:
                pc_embeds, _, _ = self.pc_encoder.forward_features(pc_input) # BNC
        else:
            pc_embeds = self.pc_encoder(pc_input) # BNC
        pc_embeds = F.normalize(pc_embeds, dim=-1) # BNC # normalize before projection(aggregation) layer

        # discard the cls token before aggregation
        if self.discard_cls_token:
            image_embeds = image_embeds[:, 1:, :]
            pc_embeds = pc_embeds[:, 1:, :]
        if self.do_patch_weight:
            # assign weight to tokens based on attention value
            # fetch topk token
            indice_img, attn_img = self.fetch_token_attention_img(image_embeds) # BN, BN
            indice_pc, attn_pc = self.fetch_token_attention_pc(pc_embeds) # BN, BN
            # # normalize the image global feature
            image_global_feat = F.normalize(self.vision_proj(image_embeds, attn_img), dim=-1)
            # # normalize the pc global feature
            pc_global_feat = F.normalize(self.pc_proj(pc_embeds, attn_pc), dim=-1)
        else:
            # normalize the image global feature
            image_global_feat = F.normalize(self.vision_proj(image_embeds), dim=-1)
            # normalize the pc global feature
            pc_global_feat = F.normalize(self.pc_proj(pc_embeds), dim=-1)
        image_global_feat = image_global_feat.view(batch_size, self.embed_dim)
        pc_global_feat = pc_global_feat.view(batch_size, self.embed_dim)
        # return
        loss = self.criterion_ce(image_global_feat, pc_global_feat)
        if self.rkdg:
            loss += self.criterion_distil(image_global_feat, pc_global_feat)
        return loss

    def forward(self, samples):
        if self.mode == "vanilla":
            return self.forward_vanilla(samples)
        elif self.mode == "contrast":
            return self.forward_contrast(samples)
        else:
            raise NotImplementedError(f'Sorry, <{self.mode}> forward function is not implemented!')

    @torch.no_grad()
    def fetch_feat_img(self, image_input:torch.Tensor, agg_type):
        with torch.no_grad():
            if agg_type.NAME in ['NetVLAD', 'NetVLADLoupe']:
                image_embeds = self.image_encoder.forward_features(image_input)
            else:
                image_embeds = self.image_encoder(image_input)
            image_embeds = F.normalize(image_embeds, dim=-1) # BNC # normalize before projection(aggregation) layer
            if self.discard_cls_token:
                image_embeds = image_embeds[:, 1:, :]
            if self.do_patch_weight:
                # fetch topk token
                indice_img, attn = self.fetch_token_attention_img(image_embeds) # BN
                # fetch the corresponding element ats the same position
                # image_embeds = image_embeds.gather(dim=1, index=indice_img.unsqueeze(-1).expand(-1, -1, image_embeds.size(-1)))
                # image_global_feat = F.normalize(self.vision_proj(image_embeds), dim=-1)
                image_global_feat = F.normalize(self.vision_proj(image_embeds, attn), dim=-1)
            else:
                image_global_feat = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_global_feat
    
    @torch.no_grad()
    def fetch_feat_pc(self, pc_input:torch.Tensor, agg_type):
        with torch.no_grad():
            if agg_type.NAME in ['NetVLAD', 'NetVLADLoupe']:
                if self.proxy:
                    pc_embeds = self.pc_encoder.forward_features(pc_input) # BNC
                else:
                    pc_embeds, _, _ = self.pc_encoder.forward_features(pc_input)
            else:
                pc_embeds = self.pc_encoder(pc_input)
            pc_embeds = F.normalize(pc_embeds, dim=-1) # BNC # normalize before projection(aggregation) layer
            if self.discard_cls_token:
                pc_embeds = pc_embeds[:, 1:, :]
            if self.do_patch_weight:
                # fetch topk token
                indice_pc, attn = self.fetch_token_attention_pc(pc_embeds) # BN, BN
                # fetch the corresponding element ats the same position
                # pc_embeds = pc_embeds.gather(dim=1, index=indice_pc.unsqueeze(-1).expand(-1, -1, pc_embeds.size(-1)))
                # pc_global_Feat = F.normalize(self.pc_proj(pc_embeds), dim=-1)
                pc_global_Feat = F.normalize(self.pc_proj(pc_embeds, attn), dim=-1)
            else:
                pc_global_Feat = F.normalize(self.pc_proj(pc_embeds), dim=-1)
        return pc_global_Feat


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
