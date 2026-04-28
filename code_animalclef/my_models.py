import numpy as np
import torch
import timm
#from pygments.lexer import include
from transformers import AutoModel
import torchvision

from paths_and_constants import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterLinear(nn.Module):

    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        # Weight shape: [num_classes * k, feature_dim]
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 1. Normalize the inputs and the weights (Cosine Similarity)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # 2. Reshape to separate the K centers: [Batch, Classes, K]
        cosine = cosine.view(-1, self.out_features, self.k)

        # 3. The "Policy": Pick the best center for each class
        cosine, _ = torch.max(cosine, dim=2)

        return cosine


import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionPool2d(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        # Each head is a separate "concept seeker"
        self.queries = nn.Parameter(torch.randn(num_heads, in_channels))
        self.v_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # x: [B, 256, 24, 24]
        B, C, H, W = x.shape
        v = self.v_proj(x.view(B, C, -1).permute(0, 2, 1)) # [B, 576, 256]

        # Each head looks at ALL 576 pixels and picks what it likes
        # No spatial constraints = View Invariant
        attn = torch.matmul(self.queries, v.transpose(1, 2)) # [B, num_heads, 576]
        attn = F.softmax(attn / (C**0.5), dim=-1)

        # Result: [B, num_heads, 256] -> e.g., [B, 1024]
        out = torch.matmul(attn, v)
        return out.view(B, -1)



from timm.models.layers import DropPath
class my_wrapped_resnet(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.pooler = MultiHeadAttentionPool2d(256)

        for i_layer, layer in enumerate([self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]):
            for block in layer:
                try:
                    print('before:', block.drop_path)
                except:
                    print('block has no drop path')
                block_drop_rate = min(0.15 * (i_layer + 1), 0.4)
                block.drop_path = DropPath(block_drop_rate) if block_drop_rate > 0 else nn.Identity()
                print('after:', block)



    def forward(self, x):
        # Standard ResNet-18 flow up to Layer 3
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = F.relu(x)#self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # 64 ch, 96x96
        x = self.backbone.layer2(x)  # 128 ch, 48x48
        x = self.backbone.layer3(x)  # 256 ch, 24x24 <-- Stop here

        # # Pooling (Mimicking the head logic)
        # # Using your 224/512 theory:
        # avg_p = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # max_p = F.adaptive_max_pool2d(x, 1).flatten(1)
        # combined = torch.cat([avg_p, max_p], dim=1)  # Results in 512-dim vector
        combined = self.pooler(x)

        return combined




class AnimalReIDRefiner(torch.nn.Module):


    # def load_weight_(self, model_name, weights_file):
    #
    #     weights = torch.load(weights_file)
    #     # Create a new dictionary without the "model." prefix
    #     new_weights = {}
    #     for k, v in weights.items():
    #         if k.startswith("model."):
    #             new_weights[k[6:]] = v  # Remove 'model.' (which is 6 characters)
    #         else:
    #             new_weights[k] = v
    #     self.backbone.load_state_dict(new_weights)


    def __increase_backbone_dropout_prob(self):

        for name, module in self.backbone.named_modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torchvision.ops.StochasticDepth)):
                layer_index = int(name.split('.')[1])
                if (layer_index > 2) and (module.p > 0):
                    target_p = min(0.4, module.p * 4)
                    print(f"Increasing {name} p from {module.p} to {target_p}")
                    module.p = target_p



    def __init__(self, model_name="mega384", use_projector=True, projection_dim=256, weights_file=None,
                 use_marg=False, marg_num_clases=None, marg_K=3):

        super().__init__()

        if model_name == 'mega384':
            self.backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
            if weights_file is not None:
                self.backbone.reset_classifier(0)
            self.direct_feature_access = False
            output_size = 1536
        if model_name == 'mega224':
            self.backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-224", pretrained=True)
            self.backbone.reset_classifier(0)
            self.direct_feature_access = False
        if model_name == 'miewid':
            self.backbone = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
            self.direct_feature_access = False
        if model_name == 'resnet':
            # self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1', )
            # self.backbone.fc = torch.nn.Identity()
            output_size = 256*4#512
            self.backbone = my_wrapped_resnet()
            self.direct_feature_access = False
        if model_name == 'effnet':
            self.backbone = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
            self.__increase_backbone_dropout_prob()
            # ONLY for clusterring!
            self.backbone.features = self.backbone.features[:6]
            self.direct_feature_access = True
            output_size = 112 * 16



        # 4. Add the SimCLR/SupCon standard projection head
        if use_projector:
            #self.backbone.reset_classifier(0)
            self.projector = torch.nn.Sequential(
                torch.nn.BatchNorm1d(output_size),
                torch.nn.Dropout(p=0.4),
                torch.nn.LazyLinear(512),
                torch.nn.BatchNorm1d(512),
                torch.nn.GELU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.LazyLinear(projection_dim)
            )
        else:
            self.projector = torch.nn.Identity()

        if use_marg:
            self.marg = SubCenterLinear(in_features=projection_dim, out_features=marg_num_clases, k=marg_K)
        else:
            self.marg = torch.nn.Identity()



        if weights_file is not None:
            weights = torch.load(os.path.join(ROOT_MODELS, weights_file))
            self.load_state_dict(weights)
            # self.load_weight_(model_name, weights_file)


    def freeze_for_training(self, active_stages=[3]):
        num_stages = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            num_stages += 1
        if num_stages > 5:
            # # meanwhile undo, TDB
            # for param in self.backbone.parameters():
            #     param.requires_grad = True
            return
        try:
            for stage_idx in active_stages:
                for param in self.backbone.layers[stage_idx].parameters():
                    param.requires_grad = True
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True
                    for param in self.backbone.head.parameters():
                        param.requires_grad = True
        except:
            for stage_idx, param in enumerate(self.backbone.parameters()):
                if stage_idx in active_stages:
                    param.requires_grad = True


    def forward(self, x):

        if self.direct_feature_access:
            features = self.backbone.features(x)
            features = F.adaptive_max_pool2d(features, 4).flatten(1)
        else:
            features = self.backbone(x)
        z = self.projector(features)
        self.keep_embed = z
        logits = self.marg(z)

        return logits

    def get_embedding(self):

        return self.keep_embed



if __name__ == '__main__':

    # check save/load
    filename = os.path.join(ROOT_DEBUG, 'check_model_save_load.pth')
    model = AnimalReIDRefiner(use_projector=False)
    torch.save(model.state_dict(), filename)

    model1 = AnimalReIDRefiner(use_projector=False)
    weights = torch.load(filename)
    model1.load_state_dict(weights)
    for p, p1 in zip(model.parameters(), model1.parameters()):
        if ((p != p1).cpu().numpy().sum()) > 0:
            print('conflict')
    print('model without projection passed')


    model = AnimalReIDRefiner(use_projector=True)
    torch.save(model.state_dict(), filename)

    model1 = AnimalReIDRefiner(use_projector=True)
    weights = torch.load(filename)
    model1.load_state_dict(weights)
    for p, p1 in zip(model.parameters(), model1.parameters()):
        if ((p != p1).cpu().numpy().sum()) > 0:
            print('conflict')
    print('model with projection passed')

    model2 = AnimalReIDRefiner(use_projector=True, weights_file=filename)
    for p, p1 in zip(model.parameters(), model2.parameters()):
        if ((p != p1).cpu().numpy().sum()) > 0:
            print('conflict')
    print('model with projection passed')

    os.remove(filename)




