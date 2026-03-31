import torch
import timm
#from pygments.lexer import include
from transformers import AutoModel



class AnimalReIDRefiner(torch.nn.Module):


    def load_weight_(self, model_name, weights_file):

        weights = torch.load(weights_file)
        # Create a new dictionary without the "model." prefix
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                new_weights[k[6:]] = v  # Remove 'model.' (which is 6 characters)
            else:
                new_weights[k] = v
        self.model.load_state_dict(new_weights)



    def __init__(self, model_name="mega384", use_projector=True, projection_dim=256, weights_file=None):

        super().__init__()

        if model_name == 'mega384':
            self.backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        if model_name == 'mega224':
            self.backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-224", pretrained=True)
        if model_name == 'miewid':
            self.backbone = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)

        #self.backbone.reset_classifier(0)

        # 4. Add the SimCLR/SupCon standard projection head
        if use_projector:
            self.backbone.reset_classifier(0)
            input_dim = self.backbone.num_features
            self.projector = torch.nn.Sequential(
                # torch.nn.Linear(feature_dim, feature_dim),
                # torch.nn.GELU(),
                # torch.nn.Linear(feature_dim, projection_dim)  # Projects to a 128-D unit sphere
                # First Layer: Expand/Project to a higher space for mixing
                torch.nn.Linear(input_dim, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                # Second Layer: Compress to your 256-D SupCon space
                torch.nn.Linear(512, projection_dim)
            )
        else:
            self.projector = torch.nn.Identity()

        if weights_file is not None:
            self.load_weight_(model_name, weights_file)


    def freeze_for_training(self, active_stages=[3]):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for stage_idx in active_stages:
            for param in self.backbone.layers[stage_idx].parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        for param in self.backbone.head.parameters():
            param.requires_grad = True

    def forward(self, x):

        features = self.backbone(x)
        z = self.projector(features)

        return z#torch.nn.functional.normalize(z, p=2, dim=1)


