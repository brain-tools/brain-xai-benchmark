
import numpy as np
import torch
from zennit.composites import EpsilonPlusFlat, GuidedBackprop, ExcitationBackprop, EpsilonPlus, EpsilonGammaBox, EpsilonAlpha2Beta1Flat, EpsilonAlpha2Beta1
from zennit.torchvision import ResNetCanonizer
from zennit.attribution import Gradient, SmoothGrad
from captum.attr import IntegratedGradients, GuidedBackprop, LRP, Saliency, GuidedGradCam, Occlusion, InputXGradient, DeepLiftShap, KernelShap, DeepLift
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

@torch.enable_grad()
@torch.inference_mode(False)
def get_explanation(model, x, mode, target, shap_baseline):
    
    # zennit
    if mode in ["LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop"]:
        targets = torch.eye(1000)[[target]].to(x.device)

    # captum
    elif mode in ["DeepLift", "GuidedBackprop", "GuidedGradCam", "InputXGradient", "DeepLiftShap", "DeepLift_mean_img"]:
        targets = torch.zeros((x.shape[0]), dtype=torch.int64).to(x.device)
        targets = target

    if mode == "GradCAM":

        target_layers = [model.layer4[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)

        return grayscale_cam
    

    if mode == "HiResCAM":

        target_layers = [model.layer4[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with HiResCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)

        return grayscale_cam
    
    if mode == "GradCAMPlusPlus":

        target_layers = [model.layer4[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)

        return grayscale_cam
    
    if mode == "GradCAM_smooth":

        target_layers = [model.layer4[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets, aug_smooth=True, eigen_smooth=True)

        return grayscale_cam
    

    if mode == "HiResCAM_smooth":

        target_layers = [model.layer4[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with HiResCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets, aug_smooth=True, eigen_smooth=True)

        return grayscale_cam
    
    if mode == "GradCAMPlusPlus_smooth":

        target_layers = [model.layer4[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets, aug_smooth=True, eigen_smooth=True)

        return grayscale_cam

    if mode == "GradCAM_l1":

        target_layers = [model.layer1[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)

        return grayscale_cam
    
    if mode == "GradCAM_l2":

        target_layers = [model.layer2[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)

        return grayscale_cam
    
    if mode == "GradCAM_l3":

        target_layers = [model.layer3[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)

        return grayscale_cam
    
    if mode == "GradCAM_l4":

        target_layers = [model.layer4[-1]]

        targets = []
        for i in range(0, x.shape[0]):
            targets.append(ClassifierOutputTarget(target))
            
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)

        return grayscale_cam

    if mode == "LRP_EpsilonPlusFlat":

        canonizer = ResNetCanonizer()
        composite = EpsilonPlusFlat(canonizers=[canonizer])

        with Gradient(model=model, composite=composite) as attributor:
            output, attribution = attributor(x, targets)

        attribution = attribution.detach().cpu().numpy()
        
        return attribution
    
    if mode == "LRP_EpsilonPlus":

        canonizer = ResNetCanonizer()
        composite = EpsilonPlus(canonizers=[canonizer])

        with Gradient(model=model, composite=composite) as attributor:
            output, attribution = attributor(x, targets)

        attribution = attribution.detach().cpu().numpy()
        
        return attribution
    
    if mode == "LRP_EpsilonAlpha2Beta1":

        canonizer = ResNetCanonizer()
        composite = EpsilonAlpha2Beta1(canonizers=[canonizer])

        with Gradient(model=model, composite=composite) as attributor:
            output, attribution = attributor(x, targets)

        attribution = attribution.detach().cpu().numpy()
        
        return attribution

    if mode == "LRP_EpsilonAlpha2Beta1Flat":

        canonizer = ResNetCanonizer()
        composite = EpsilonAlpha2Beta1Flat(canonizers=[canonizer])

        with Gradient(model=model, composite=composite) as attributor:
            output, attribution = attributor(x, targets)

        attribution = attribution.detach().cpu().numpy()
        
        return attribution
    
    if mode == "SmoothGrad":

        with SmoothGrad(noise_level=0.1, n_iter=20, model=model) as attributor:
            output, attribution = attributor(x, targets)

        attribution = attribution.detach().cpu().numpy()

        return attribution

    if mode == "ExcitationBackprop":
        
        canonizer = ResNetCanonizer()
        composite = ExcitationBackprop(canonizers=[canonizer])

        with Gradient(model=model, composite=composite) as attributor:
            output, attribution = attributor(x, targets)

        attribution = attribution.detach().cpu().numpy()

        return attribution

    if mode == "DeepLift":

        dl = DeepLift(model)
        attribution = dl.attribute(x, target=targets, baselines=0)
        return attribution.detach().cpu().numpy()
    
    if mode == "DeepLiftShap":
        
        dl = DeepLiftShap(model)
        attribution = dl.attribute(x, target=targets, baselines=shap_baseline)
        return attribution.detach().cpu().numpy()

    if mode == "GuidedBackprop":
        
        gbp = GuidedBackprop(model)
        attribution = gbp.attribute(x, target=targets)
        return attribution.detach().cpu().numpy()
    
    if mode == "GuidedGradCam":

        guided_gc = GuidedGradCam(model, model.layer4[-1])
        attribution = guided_gc.attribute(x, target=targets, interpolate_mode="nearest")
        return attribution.detach().cpu().numpy()

    if mode == "InputXGradient":
        
        input_x_gradient = InputXGradient(model)
        attribution = input_x_gradient.attribute(x, target=targets)
        return attribution.detach().cpu().numpy()
    
    if mode == "DeepLift_mean_img":
        
        dl = DeepLift(model)
        attribution = dl.attribute(x, target=targets, baselines=0)
        return attribution.detach().cpu().numpy()