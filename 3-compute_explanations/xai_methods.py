from zennit.composites import EpsilonPlusFlat, GuidedBackprop, ExcitationBackprop, EpsilonPlus, EpsilonGammaBox, EpsilonAlpha2Beta1Flat, EpsilonAlpha2Beta1
from zennit.densenet_canonizers import ThreshSequentialCanonizer, DefaultDenseNetCanonizer
from zennit.attribution import Gradient, SmoothGrad
from zennit.torchvision import ResNetCanonizer
from captum.attr import IntegratedGradients, GuidedBackprop, LRP, Saliency, GuidedGradCam, Occlusion, InputXGradient, DeepLiftShap, KernelShap, DeepLift
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch

class ExplComputer:
    def __init__(self, model_type, train_mean, train_std, deep_lift_mean_img):
        self.model_type = model_type
        self.train_mean = train_mean
        self.train_std = train_std
        self.mean_img = deep_lift_mean_img

    @torch.enable_grad()
    @torch.inference_mode(False)
    def get_explanation(self, model, x, xai_method):
        """Get explanation for each input in batch x using the specified XAI method."""
        print(xai_method)

        grad_x = x.clone().requires_grad_()
        grad_x = (grad_x - self.train_mean) / self.train_std
        
        # zennit
        if xai_method in ["LRP_EpsilonPlusFlat", "LRP_EpsilonPlus", "LRP_EpsilonAlpha2Beta1", "LRP_EpsilonAlpha2Beta1Flat", "SmoothGrad", "ExcitationBackprop"]:
            targets = torch.ones((x.shape[0], 1)).to(x.device)
        
        # captum
        elif xai_method in ["DeepLift", "GuidedBackprop", "GuidedGradCam", "InputXGradient", "DeepLiftShap", "DeepLift_mean_img"]:
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
    

        if xai_method == "LRP_EpsilonPlusFlat":

            if self.model_type == "ResNet":
                canonizer = ResNetCanonizer()
            
            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer()

            composite = EpsilonPlusFlat(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution
        
        if xai_method == "LRP_EpsilonPlus":

            if self.model_type == "ResNet":
                canonizer = ResNetCanonizer()
            
            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer()

            composite = EpsilonPlus(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution
        
        if xai_method == "LRP_EpsilonAlpha2Beta1":

            if self.model_type == "ResNet":
                canonizer = ResNetCanonizer()
            
            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer()

            composite = EpsilonAlpha2Beta1(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution

        if xai_method == "LRP_EpsilonAlpha2Beta1Flat":

            if self.model_type == "ResNet":
                canonizer = ResNetCanonizer()
            
            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer()

            composite = EpsilonAlpha2Beta1Flat(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]
            
            return attribution
        
        if xai_method == "SmoothGrad":

            with SmoothGrad(noise_level=0.1, n_iter=20, model=model) as attributor:
                output, attribution = attributor(grad_x, targets)

            attribution = attribution.detach().cpu().numpy()[:, 0, :, :, :]

            return attribution
        
        if xai_method == "ExcitationBackprop":
            
            if self.model_type == "ResNet":
                canonizer = ResNetCanonizer()
            
            elif self.model_type == "DenseNet":
                canonizer = DefaultDenseNetCanonizer()

            composite = ExcitationBackprop(canonizers=[canonizer])

            with Gradient(model=model, composite=composite) as attributor:
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
