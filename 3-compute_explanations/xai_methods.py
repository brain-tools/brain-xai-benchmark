from zennit.composites import EpsilonPlusFlat, ExcitationBackprop, EpsilonPlus, EpsilonAlpha2Beta1Flat, EpsilonAlpha2Beta1, EpsilonGamma
from zennit.attribution import Gradient, SmoothGrad
from zennit.torchvision import ResNet3DCanonizer
from zennit.densenet_canonizers import DefaultDenseNetCanonizer3D
from captum.attr import GuidedBackprop, GuidedGradCam, InputXGradient, DeepLift
from pytorch_grad_cam import GradCAM, HiResCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import numpy as np
import sys
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/1-idp_correction')
from atlas_methods import warp_atlas
sys.path.append('/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/brain-xai-benchmark/2-model_training/models/ResNet')
from resnet import BasicBlock, Bottleneck

class ExplComputer:
    def __init__(self, model_type, train_mean, train_std, deep_lift_mean_img, full_atlas):
        self.model_type = model_type
        self.train_mean = train_mean
        self.train_std = train_std
        self.mean_img = deep_lift_mean_img
        self.full_atlas = full_atlas

    @torch.enable_grad()
    @torch.inference_mode(False)
    def get_explanation(self, model, x, xai_method, coefs=None):
        """Get explanation for each input in batch x using the specified XAI method."""
        print(xai_method)

        grad_x = x.clone().requires_grad_()
        grad_x = (grad_x - self.train_mean) / self.train_std
        
        # zennit
        if xai_method in ["LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop", "LRP_EpsilonGamma_0", "LRP_EpsilonGamma_0.001", "LRP_EpsilonGamma_0.01", "LRP_EpsilonGamma_0.1", "LRP_EpsilonGamma_1",]:
            targets = torch.ones((x.shape[0], 1)).to(x.device)
        
        # captum
        elif xai_method in ["DeepLift", "GuidedBackprop", "GuidedGradCam", "InputXGradient", "DeepLift_mean_img"]:
            targets = torch.zeros((x.shape[0]), dtype=torch.int64).to(x.device)

        if xai_method == "GradCAM":

            if self.model_type == "ResNet":
                target_layers = [model.layer4[-1]]     
            elif self.model_type == "DenseNet":
                target_layers = [model.features[-1]]

            targets = []
            for i in range(0, x.shape[0]):
                targets.append(ClassifierOutputTarget(0))
            
            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_x, targets=targets)

            return grayscale_cam

        if xai_method == "GradCAM_l1":

            target_layers = [model.layer1[-1]]

            targets = []
            for i in range(0, x.shape[0]):
                targets.append(ClassifierOutputTarget(0))

            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_x, targets=targets)

            return grayscale_cam
        
        if xai_method == "GradCAM_l2":

            target_layers = [model.layer2[-1]]

            targets = []
            for i in range(0, x.shape[0]):
                targets.append(ClassifierOutputTarget(0))

            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_x, targets=targets)

            return grayscale_cam
        
        if xai_method == "GradCAM_l3":

            if self.model_type == "ResNet":
                target_layers = [model.layer3[-1]]     
            
            elif self.model_type == "DenseNet":
                target_layers = [model.features[-2]]

            targets = []
            for i in range(0, x.shape[0]):
                targets.append(ClassifierOutputTarget(0))

            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_x, targets=targets)

            return grayscale_cam
    
        if xai_method == "HiResCAM":

            if self.model_type == "ResNet":
                target_layers = [model.layer3[-1]]     
            
            elif self.model_type == "DenseNet":
                target_layers = [model.features[-1]]

            targets = []
            for i in range(0, x.shape[0]):
                targets.append(ClassifierOutputTarget(0))

            with HiResCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_x, targets=targets)

            return grayscale_cam
        
        if xai_method == "EigenCAM":

            if self.model_type == "ResNet":
                target_layers = [model.layer4[-1]]     
            
            elif self.model_type == "DenseNet":
                target_layers = [model.features[-1]]

            targets = []
            for i in range(0, x.shape[0]):
                targets.append(ClassifierOutputTarget(0))
            
            with EigenCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_x, targets=targets)
                
            return grayscale_cam
        
        if xai_method == "LayerCAM":

            if self.model_type == "ResNet":
                target_layers = [model.layer4[-1]]     
            
            elif self.model_type == "DenseNet":
                target_layers = [model.features[-1]]

            targets = []
            for i in range(0, x.shape[0]):
                targets.append(ClassifierOutputTarget(0))
            
            with LayerCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_x, targets=targets)
                
            return grayscale_cam

        if xai_method == "LRP_EpsilonPlusFlat":

            if self.model_type == "ResNet":
                canonizer = ResNet3DCanonizer(BasicBlock, Bottleneck)

            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer3D()

            composite = EpsilonPlusFlat(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution
        
        if xai_method == "LRP_EpsilonPlus":

            if self.model_type == "ResNet":
                canonizer = ResNet3DCanonizer(BasicBlock, Bottleneck)

            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer3D()

            composite = EpsilonPlus(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution
        
        if xai_method == "LRP_EpsilonAlpha2Beta1":

            if self.model_type == "ResNet":
                canonizer = ResNet3DCanonizer(BasicBlock, Bottleneck)

            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer3D()

            composite = EpsilonAlpha2Beta1(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution

        if xai_method == "LRP_EpsilonAlpha2Beta1Flat":

            if self.model_type == "ResNet":
                canonizer = ResNet3DCanonizer(BasicBlock, Bottleneck)

            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer3D()

            composite = EpsilonAlpha2Beta1Flat(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution
        
        if xai_method == "ExcitationBackprop":
            
            if self.model_type == "ResNet":
                canonizer = ResNet3DCanonizer(BasicBlock, Bottleneck)

            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer3D()

            composite = ExcitationBackprop(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]

            return attribution
        
        if xai_method in ["LRP_EpsilonGamma_0", "LRP_EpsilonGamma_0.001", "LRP_EpsilonGamma_0.01", "LRP_EpsilonGamma_0.1", "LRP_EpsilonGamma_1"]:
            
            gamma = float(xai_method.split("_")[-1])
            
            if self.model_type == "ResNet":
                canonizer = ResNet3DCanonizer(BasicBlock, Bottleneck)

            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer3D()

            composite = EpsilonGamma(canonizers=[canonizer], gamma=gamma)

            with Gradient(model=model, composite=composite) as attributor:
                # compute the model output and attribution
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution

        if xai_method == "SmoothGrad":

            with SmoothGrad(noise_level=0.1, n_iter=20, model=model) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]

            return attribution
        
        if xai_method == "DeepLift":

            dl = DeepLift(model)
            attribution = dl.attribute(grad_x, target=targets, baselines=0)
            return attribution[:, 0, :, :, :].detach().cpu().numpy()
        
        if xai_method == "GuidedBackprop":
            
            gbp = GuidedBackprop(model)
            attribution = gbp.attribute(grad_x, target=targets)
            return attribution[:, 0, :, :, :].detach().cpu().numpy()
        
        if xai_method == "GuidedGradCam":

            if self.model_type == "ResNet":
                guided_gc = GuidedGradCam(model, model.layer4[-1])
            elif self.model_type == "DenseNet":
                guided_gc = GuidedGradCam(model, model.features[-1])
            
            attribution = guided_gc.attribute(grad_x, target=targets, interpolate_mode="nearest")
            return attribution[:, 0, :, :, :].detach().cpu().numpy()

        if xai_method == "InputXGradient":
            
            input_x_gradient = InputXGradient(model)
            attribution = input_x_gradient.attribute(grad_x, target=targets)
            return attribution[:, 0, :, :, :].detach().cpu().numpy()
        
        if xai_method == "DeepLift_mean_img":
            
            dl = DeepLift(model)
            attribution = dl.attribute(grad_x, target=targets, baselines=self.mean_img.to(x.device))
            return attribution[:, 0, :, :, :].detach().cpu().numpy()

        if xai_method == "AtlasOcclusion":
            # infer batch size
            B = x.shape[0]

            # 1. Load atlas
            atlas_mni = self.full_atlas.get_fdata().astype(int)

            # 2. Get original output
            output_clean = model(grad_x)

            # 3. Inverse-warp atlas
            warped_atlas = np.zeros(x.shape)

            # coefs is coming from dataloader batch
            for i in range(B):
                warped_atlas_subj, indices = warp_atlas(
                    coefs,
                    device="cuda",
                    atlas=atlas_mni,
                    i=i)

                warped_atlas[i, 0, :, :, :] = warped_atlas_subj

            # Region IDs from the original atlas
            region_ids = np.unique(atlas_mni)
            region_ids = region_ids[region_ids > 0]  # skip background


            # 4. Prepare attribution tensor
            attributions = np.zeros((B, x.shape[2], x.shape[3], x.shape[4]), dtype=np.float32)

            # 5. Loop over atlas regions
            for region_id in region_ids:
                # for each batch subject
                occluded_x = grad_x.clone()
                region_sizes = []
                for i in range(B):

                    # mask for this region in subject space
                    region_mask = (warped_atlas[i, 0] == region_id)

                    if not np.any(region_mask):
                        # region not present after warping
                        continue

                    # ========================
                    # Compute replacement value
                    # ========================
                    # Take mean intensity of that region from the mean image
                    replacement_vals = self.mean_img[0, 0, atlas_mni == region_id]
                    region_replacement = float(replacement_vals.mean())

                    # ========================
                    # Apply occlusion
                    # ========================
                    # occluded_x: shape [B, 1, X, Y, Z]
                    occluded_x[i, 0, region_mask] = region_replacement
                    region_sizes.append(region_mask.sum())
                # 6. Forward pass for this region
                with torch.no_grad():
                    output_occ = model(occluded_x)

                # 7. Compute (scaled) output difference for this region
                deltas = np.abs((output_clean - output_occ).detach().cpu().numpy())
                norm_deltas = deltas[:,0] / np.array(region_sizes)

                # 8. Map back to 3D space for each subject in batch
                for i in range(B):
                    region_mask = (warped_atlas[i, 0] == region_id)
                    attributions[i][region_mask] = norm_deltas[i]

            return attributions

        raise ValueError(f"Unknown xai_method: {xai_method}")