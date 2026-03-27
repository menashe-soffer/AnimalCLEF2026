import torch
import torch.nn.functional as F
import timm





class AnimalReIDRefiner(torch.nn.Module):
    def __init__(self, model_name="hf-hub:BVRA/MegaDescriptor-L-384", use_projector=True, projection_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)

        # 4. Add the SimCLR/SupCon standard projection head
        if use_projector:
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
                torch.nn.Linear(512, 256)
            )
        else:
            self.projector = torch.nn.Identity()


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

        return torch.nn.functional.normalize(z, p=2, dim=1)


