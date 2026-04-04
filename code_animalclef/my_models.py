import numpy as np
import torch
import timm
#from pygments.lexer import include
from transformers import AutoModel

from paths_and_constants import *



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
            #self.backbone.reset_classifier(0)
            self.projector = torch.nn.Sequential(
                torch.nn.LazyLinear(512),
                torch.nn.BatchNorm1d(512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                # Second Layer: Compress to your 256-D SupCon space
                torch.nn.Linear(512, projection_dim)
            )
        else:
            self.projector = torch.nn.Identity()

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
            active_stages = np.arange(start=int(0.95 * num_stages), stop=num_stages)
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

        features = self.backbone(x)
        z = self.projector(features)

        return torch.nn.functional.normalize(z, p=2, dim=1)#z#



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




