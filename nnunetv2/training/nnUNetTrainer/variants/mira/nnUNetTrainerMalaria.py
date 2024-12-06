from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Union, Tuple, List
import torch


class nnUNetTrainerMalaria(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:

        if 'norm_op' not in arch_init_kwargs.keys():
            raise RuntimeError("'norm_op' not found in arch_init_kwargs. This does not look like an architecture "
                               "I can hack BN into. This trainer only works with default nnU-Net architectures.")

        from pydoc import locate
        conv_op = locate(arch_init_kwargs['conv_op'])
        bn_class = get_matching_batchnorm(conv_op)
        arch_init_kwargs['norm_op'] = bn_class.__module__ + '.' + bn_class.__name__
        arch_init_kwargs['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}

        return nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels,
                                                        num_output_channels, enable_deep_supervision)