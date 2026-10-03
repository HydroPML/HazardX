import torch
import torch.nn as nn
import torch.nn.functional as F 
from .segformer_head import SegFormerHead
from . import mix_transformer

class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=4, embedding_dim=256, pretrained_weight=None, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.input_channels = in_channels

        self.encoder = getattr(mix_transformer, backbone)(in_chans=in_channels)
        self.in_channels = self.encoder.embed_dims

        if pretrained_weight:
            self._load_pretrained_weights(pretrained_weight)

        self.decoder = SegFormerHead(
            feature_strides=self.feature_strides,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
        )

        self.classifier = nn.Conv2d(
            in_channels=self.in_channels[-1],
            out_channels=self.num_classes,
            kernel_size=1,
            bias=False,
        )

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = state_dict.copy()

        if any(key.startswith('backbone.layers.') for key in state_dict):
            state_dict = self._convert_mmseg_backbone(state_dict)

        # Handle common wrappers around encoder parameters
        cleaned_state = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            if new_key.startswith('encoder.'):
                new_key = new_key[len('encoder.'):]
            cleaned_state[new_key] = value

        # Remove classification head weights if present
        cleaned_state.pop('head.weight', None)
        cleaned_state.pop('head.bias', None)

        patch_key = 'patch_embed1.proj.weight'
        if patch_key in cleaned_state:
            weight = cleaned_state[patch_key]
            if weight.shape[1] != self.input_channels:
                cleaned_state[patch_key] = self._resize_patch_embed_weight(weight, self.input_channels)

        encoder_state = self.encoder.state_dict()
        filtered_state = {
            k: v for k, v in cleaned_state.items()
            if k in encoder_state and v.shape == encoder_state[k].shape
        }
        if not filtered_state:
            raise RuntimeError(
                f'No compatible encoder tensors found in SegFormer checkpoint: {checkpoint_path}'
            )

        missing, unexpected = self.encoder.load_state_dict(filtered_state, strict=False)
        if missing:
            preview = missing[:5]
            print(f'[SegFormer] Missing keys when loading pretrained weights (showing first 5 of {len(missing)}): {preview}')
        if unexpected:
            preview = unexpected[:5]
            print(f'[SegFormer] Unexpected keys when loading pretrained weights (showing first 5 of {len(unexpected)}): {preview}')
        print(
            f'[SegFormer] Loaded {len(filtered_state)}/{len(encoder_state)} '
            f'encoder tensors from {checkpoint_path}'
        )

    @staticmethod
    def _convert_mmseg_backbone(state_dict):
        """Convert MMSeg MiT backbone keys to this repository's MiT layout."""
        converted = {}
        for key, value in state_dict.items():
            if not key.startswith('backbone.layers.'):
                continue
            parts = key.split('.')
            stage = int(parts[2]) + 1
            layer = parts[3]
            suffix = '.'.join(parts[4:])

            if layer == '0':
                suffix = suffix.replace('projection.', 'proj.', 1)
                converted[f'patch_embed{stage}.{suffix}'] = value
            elif layer == '2':
                converted[f'norm{stage}.{suffix}'] = value
            elif layer == '1':
                block_index, block_suffix = suffix.split('.', 1)
                prefix = f'block{stage}.{block_index}.'
                if block_suffix == 'attn.attn.in_proj_weight':
                    embed_dim = value.shape[0] // 3
                    converted[prefix + 'attn.q.weight'] = value[:embed_dim]
                    converted[prefix + 'attn.kv.weight'] = value[embed_dim:]
                elif block_suffix == 'attn.attn.in_proj_bias':
                    embed_dim = value.shape[0] // 3
                    converted[prefix + 'attn.q.bias'] = value[:embed_dim]
                    converted[prefix + 'attn.kv.bias'] = value[embed_dim:]
                else:
                    replacements = (
                        ('attn.attn.out_proj.', 'attn.proj.'),
                        ('ffn.layers.0.', 'mlp.fc1.'),
                        ('ffn.layers.1.', 'mlp.dwconv.dwconv.'),
                        ('ffn.layers.4.', 'mlp.fc2.'),
                    )
                    for source, target in replacements:
                        block_suffix = block_suffix.replace(source, target, 1)
                    if block_suffix in {'mlp.fc1.weight', 'mlp.fc2.weight'}:
                        value = value.squeeze(-1).squeeze(-1)
                    converted[prefix + block_suffix] = value
        return converted

    @staticmethod
    def _resize_patch_embed_weight(weight: torch.Tensor, new_channels: int) -> torch.Tensor:
        old_channels = weight.shape[1]
        if new_channels == old_channels:
            return weight
        if new_channels < old_channels:
            return weight[:, :new_channels, :, :]
        extra = new_channels - old_channels
        channel_mean = weight.mean(dim=1, keepdim=True)
        repeated = channel_mean.repeat(1, extra, 1, 1)
        return torch.cat([weight, repeated], dim=1)

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, input_data):

        _x = self.encoder(input_data)
        # _x1, _x2, _x3, _x4 = _x
        # cls = self.classifier(_x4)
        x = self.decoder(_x)
        x = F.interpolate(x, size=input_data.size()[2:], mode='bilinear', align_corners=False)
        return x
