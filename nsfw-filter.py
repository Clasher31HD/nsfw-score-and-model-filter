import opennsfw2 as n2
from PIL import Image

print("Starting")
image_path = "C:/Users/I539356/Downloads/blume.JPG"

try:
    img = Image.open(image_path)
    img.thumbnail((512, 512))

    nsfw_probability = n2.predict_image(image_path)

    print("Probability: " + str(nsfw_probability))
except OSError as e:
    print(f"Skipping image '{image_path.name}' due to an error: {str(e)}")

print("End")