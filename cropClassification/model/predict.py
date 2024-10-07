import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from chipping import generate_stacked_image
from PIL import Image
import cv2


# Prediction function
def predict_image(image_path, model, chip_size=224, overlap=32, class_num=3, device='cpu'):
    image = cv2.imread(image_path)  # Load image as BGR
    stacked_image = generate_stacked_image(image)
    stacked_image = stacked_image.astype(np.float32) / 255.0  # Normalize image
    img_h, img_w, _ = stacked_image.shape

    canvas_soft = np.zeros((class_num, img_h, img_w), dtype=np.float32)
    canvas_hard = np.zeros((img_h, img_w), dtype=np.int)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for row in range(0, img_h, chip_size - overlap):
            for col in range(0, img_w, chip_size - overlap):
                chip = stacked_image[row:row + chip_size, col:col + chip_size, :]
                pad_h = max(0, chip_size - chip.shape[0])
                pad_w = max(0, chip_size - chip.shape[1])
                if pad_h > 0 or pad_w > 0:
                    chip = np.pad(chip, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                chip = torch.from_numpy(np.transpose(chip, (2, 0, 1))).unsqueeze(0).to(device)
                out = model(chip)
                soft_preds = F.softmax(out, dim=1).squeeze(0).cpu().numpy()
                canvas_soft[:, row:row + chip_size, col:col + chip_size] += soft_preds[:, :chip_size, :chip_size]

        canvas_hard = np.argmax(canvas_soft, axis=0)
    return canvas_hard

# Save prediction function
def save_prediction(pred_mask, output_path):
    pred_img = Image.fromarray(pred_mask.astype(np.uint8))
    pred_img.save(output_path)

# Plot prediction function
def plot_prediction(image_path, label_path, pred_mask, title="Prediction"):
    image = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(label, cmap='gray')
    axs[1].set_title('Label')
    axs[1].axis('off')

    axs[2].imshow(pred_mask, cmap='gray')
    axs[2].set_title('Prediction')
    axs[2].axis('off')

    plt.suptitle(title)
    plt.show()