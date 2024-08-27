import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetection:
    def __init__(self,model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features,14*2)  
        self.model.load_state_dict(torch.load(model_path,map_location='cpu'),strict=False)
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

        h,w = img_rgb.shape[:2]

        keypoints[::2] *= (w)/244.0
        keypoints[1::2] *= (h)/244.0

        return keypoints

    def draw_keypoints(self,img,keypo):
        # print("len",keypo[1])
        for i in range(0,len(keypo),2):
            # print(i)
            x = int(keypo[i])
            y = int(keypo[i+1])
            cv2.putText(img,str(i//2),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(120,180,34))
            cv2.circle(img,(x,y),5,(120,180,23),-1)
        return img
    
    def draw_kp_video(self,video,keypo):
        out = []
        for frame in video:
            frame = self.draw_keypoints(frame,keypo)
            out.append(frame)
        return out