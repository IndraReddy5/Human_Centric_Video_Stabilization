"""Detects the human and returns the frame with only human"""

import torch
import torchvision.transforms as T
import cv2


class BackgroundRemover:
    """Removes the background from the video frames."""

    def __init__(self, device="cpu"):
        """Initializes the background remover model."""
        self.device = torch.device(device)
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True
        )
        self.model.to(self.device).eval()
        self.person_class_id = 15
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def remove_background(self, frame):
        """Removes the background from the frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]

        output_predictions = output.argmax(0)
        mask = (output_predictions == self.person_class_id).byte().cpu().numpy()

        mask_resized = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        foreground = cv2.bitwise_and(frame, frame, mask=mask_resized)
        return foreground
