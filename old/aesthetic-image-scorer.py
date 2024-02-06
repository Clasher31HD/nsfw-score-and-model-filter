import torch
import torch.nn as nn
import clip

# Load the CLIP model and preprocess function
clip_model, preprocess = clip.load("ViT-L/14", device="cuda" if torch.cuda.is_available() else "cpu")

# Load the trained aesthetic score predictor model
state_name = "sac+logos+ava1-l14-linearMSE.pth"
predictor = nn.Linear(768, 1)  # Assuming the input size of the MLP is 768 (CLIP embedding size)
predictor.load_state_dict(torch.load(state_name, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
predictor.eval()


# Function to calculate the aesthetic score for an image based on a prompt
def calculate_aesthetic_score(image, prompt):
    # Preprocess the image
    image = preprocess(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    # Encode the image using CLIP
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Pass the image features through the aesthetic score predictor model
    score = predictor(image_features).item()

    return score

# Example usage
image_path = "path/to/your/image.jpg"
image = images.open_image(image_path)

prompt = "test i call"

aesthetic_score = calculate_aesthetic_score(image, prompt)
print("Aesthetic Score:", aesthetic_score)
