from loss.idq_loss import IDQ_Loss
from loss.pc_loss import PC_Loss
from loss.fast_mls_loss import Fast_MLS_Loss
from loss.mls_loss import MLS_Loss


def build_uncertainty_loss(config):
    if 'uncertainty_loss_type' in dir(config) and config.uncertainty_loss_type == 'idq_loss':
        criterion_uncertainty = IDQ_Loss(config.idq_s, config.idq_m)
    elif 'uncertainty_loss_type' in dir(config) and config.uncertainty_loss_type == 'pc_loss':
        criterion_uncertainty = PC_Loss()
    elif 'uncertainty_loss_type' in dir(config) and config.uncertainty_loss_type == 'fast_mls_loss':
        assert config.uncertainty_size == 1
        criterion_uncertainty = Fast_MLS_Loss()
    else:
        criterion_uncertainty = MLS_Loss()
    print('uncertainty_loss_type', config.uncertainty_loss_type)
    if config.uncertainty_loss_type in ['pc_loss', 'loser_loss', 'idq_loss']:
        is_confidence_prob = True
    else:
        is_confidence_prob = False
    return criterion_uncertainty, is_confidence_prob
