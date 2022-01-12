import torch

from models.uncertainty.head_fc2_prob import HeadFc2Prob
from models.uncertainty.head_fc2_sigma import HeadFc2Sigma
from models.uncertainty.head_linear_mcdropout import HeadLinearMCDropout
from models.uncertainty.head_linear_prob import HeadLinearProb
from models.uncertainty.head_linear_sigma import HeadLinearSigma


def build_model_backbone(name, resume=''):
    from models import backbones
    model = eval(name)(False, dropout=0, fp16=True)
    print('build backbone', name)
    if resume != '':
        print('resuming finetune backbone from %s' % resume)
        state_dict = torch.load(resume)
        model.load_state_dict(state_dict)
    return model


def build_model_uncertainty(name, chs, resume=''):
    print('build uncertainty', name)
    uncertainty_dict = {
        'head_fc2_prob': HeadFc2Prob,
        'head_fc2_sigma': HeadFc2Sigma,
        'head_linear_prob': HeadLinearProb,
        'head_linear_sigma': HeadLinearSigma,
    }
    # backbone_name = 'IR_50'
    if len(chs) == 3:
        model = uncertainty_dict[name](in_feat=chs[0], out_size=chs[2], fc1_size=chs[1])
    else:
        model = uncertainty_dict[name](in_feat=chs[0], out_size=chs[2])
    if resume != '':
        print('resuming finetune backbone from %s' % resume)
        state_dict = torch.load(resume)
        model.load_state_dict(state_dict, strict=False)
    return model


def build_model_backbone_mcdropout(backbone_name, chs, resume='', uresume=''):
    from models import backbones
    model = eval(backbone_name)(False, dropout=0, fp16=True)
    print('build backbone', backbone_name)
    u_model = HeadLinearMCDropout(in_channels=chs[0], T=100)
    if resume != '':
        print('resuming finetune backbone from %s' % resume)
        state_dict = torch.load(resume)
        model.load_state_dict(state_dict)
        if uresume == '':
            print(u_model.model)
            checkpoint_u = {
                'fc.weight': state_dict['fc.weight'],
                'fc.bias': state_dict['fc.bias'],
            }
            u_model.model.load_state_dict(checkpoint_u)

    if uresume != '':
        print('resuming finetune backbone from %s' % uresume)
        state_dict = torch.load(resume)
        u_model.load_state_dict(state_dict, strict=False)

    return model, u_model

