import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetection:
    def __init__(self,model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features,14*2)
        self.model.load_state_dict(torch.load(model_path))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229,0.225,0.224])
        ])

    def predict(self,img): #only predict first frame as camera is not moving

        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0) # img to [img]

        with torch.no_grad():
            out = self.model(image_tensor)
        
        keypoints = out.squeeze().cpu().numpy()

        h,w = img.shape[:2]

        keypoints[::2] *= w/224.0
        keypoints[1::2] *= h/224.0

        return keypoints

