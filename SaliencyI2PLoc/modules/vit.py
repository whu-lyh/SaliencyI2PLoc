
import math
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_patch_embed
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer

from SaliencyI2PLoc.build import MODELS
from SaliencyI2PLoc.modules.patch_embed import (ConvEmbed, HybridEmbed,
                                             MiniEncoder, PatchEmbed,
                                             SphereConvEmbed_384,
                                             SphereConvEmbed_768)
from utils.misc import count_parameters

# Modified from TIMM=v0.9.9, more fancy designs should refer to the other py files at TIMM repo

def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if not old_size:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        print(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb

def _convert_dinov2(state_dict, model):
    import re
    out_dict = {}
    for k, v in state_dict.items():
        if k == "mask_token":
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict

def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        custom_tokenizer: bool = False,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv
        remove patch embedding part
    """
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    # if 'visual.class_embedding' in state_dict:
    #     return _convert_openai_clip(state_dict, model)
    # elif 'module.visual.class_embedding' in state_dict:
    #     return _convert_openai_clip(state_dict, model, prefix='module.visual.')

    if "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)

    if "encoder" in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'

    if 'visual.trunk.pos_embed' in state_dict:
        # convert an OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if not custom_tokenizer:
                O, I, H, W = model.patch_embed.proj.weight.shape
                if len(v.shape) < 4:
                    # For old models that I trained prior to conv based patchification
                    O, I, H, W = model.patch_embed.proj.weight.shape
                    v = v.reshape(O, -1, H, W)
                if v.shape[-1] != W or v.shape[-2] != H:
                    v = resample_patch_embed(
                        v,
                        (H, W),
                        interpolation=interpolation,
                        antialias=antialias,
                        verbose=True,
                    )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict

def _create_vision_transformer(variant: str, pretrained: bool = False, 
                               custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = partial(checkpoint_filter_fn, custom_tokenizer=custom_tokenizer)

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )

def vit_tiny_patch16_224(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer(
        'vit_tiny_patch16_224', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

def vit_small_patch16_224(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer(
        'vit_small_patch16_224', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

def vit_tiny_patch16_384(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer(
        'vit_tiny_patch16_384', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

def vit_small_patch16_384(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer(
        'vit_small_patch16_384', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

def vit_small_patch14_dinov2(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-S/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_small_patch14_dinov2', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

def vit_base_patch14_dinov2(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-B/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_base_patch14_dinov2', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

def vit_small_patch14_reg4_dinov2(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-S/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_small_patch14_reg4_dinov2', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

def vit_base_patch14_reg4_dinov2(pretrained: bool = False, custom_tokenizer: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-B/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_base_patch14_reg4_dinov2', pretrained=pretrained, custom_tokenizer=custom_tokenizer, **dict(model_args, **kwargs))
    return model

@MODELS.register_module()
class VisionTransformerEncoder(nn.Module):
    @classmethod
    def from_config(self, config):
        # get basic configuration
        global_pool = 'token' if config.class_token else 'avg'
        if config.pretrained:
            model_args = dict(embed_dim=config.embed_dim,
                            num_classes=0,
                            class_token=config.class_token,
                            global_pool=global_pool,
                            pretrained_cfg_overlay=dict(file=config.pretrained_weight_path, custom_load=False),
                        )
        else:
            model_args = dict(embed_dim=config.embed_dim,
                            num_classes=0,
                            class_token=config.class_token,
                            global_pool=global_pool,
                        )
        # get tokenizator stem
        embed_layer_type = config.tokenizer_type
        if embed_layer_type == "patch_embed": # for 224*224 image
            embed_layer = PatchEmbed
            model_args.update({"embed_layer": embed_layer, "custom_tokenizer": False})
        elif embed_layer_type == "patch_embed_any_res": # for any input resolution
            embed_layer = partial(PatchEmbed, img_size=config.img_size)
            model_args.update({"embed_layer": embed_layer, "custom_tokenizer": True, "img_size": config.img_size})
        elif embed_layer_type == "conv_embed":
            embed_layer = ConvEmbed
            model_args.update({"embed_layer": embed_layer, "custom_tokenizer": True})
        elif embed_layer_type == "hybird_embed": # TODO
            embed_layer = partial(HybridEmbed, backbone=MiniEncoder())
        elif embed_layer_type == "sphere_conv_embed": # for 512*1024 image
            embed_layer = SphereConvEmbed_384
            model_args.update({"embed_layer": embed_layer, "custom_tokenizer": True})
        else:
            raise NotImplementedError(f'Sorry, <{embed_layer_type}> embed_layer is not implemented!')
        # get vit stem
        if config.vit_model_name == "vit_tiny_patch16_224": # img_size:224
            visual_encoder = vit_tiny_patch16_224(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 192
        elif config.vit_model_name == "vit_small_patch16_224": # img_size:224
            visual_encoder = vit_small_patch16_224(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 384
        elif config.vit_model_name == "vit_tiny_patch16_384": # img_size:384
            visual_encoder = vit_tiny_patch16_384(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 192
        elif config.vit_model_name == "vit_small_patch16_384": # img_size:384
            visual_encoder = vit_small_patch16_384(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 384
        elif config.vit_model_name == "vit_small_patch14_dinov2": # img_size:518
            visual_encoder = vit_small_patch14_dinov2(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 384
        elif config.vit_model_name == "vit_base_patch14_dinov2": # img_size:518
            visual_encoder = vit_base_patch14_dinov2(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 768
        elif config.vit_model_name == "vit_small_patch14_reg4_dinov2": # img_size:518
            visual_encoder = vit_small_patch14_reg4_dinov2(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 384
        elif config.vit_model_name == "vit_base_patch14_reg4_dinov2": # img_size:518
            visual_encoder = vit_base_patch14_reg4_dinov2(pretrained=config.pretrained, **dict(model_args))
            visual_encoder.embed_dim = 768
        else:
            raise NotImplementedError(f'Sorry, <{config.vit_model_name}> is not implemented!')
        # freeze the transformer blocks
        num_trainable_blocks = int(config.num_trainable_blocks)
        for p in visual_encoder.blocks[:12-num_trainable_blocks].parameters():
            p.requires_grad = False
        return visual_encoder
