import sys
import os
import torch
from database import weather_cache
from google import genai
from google.genai import types
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

genai.api_key = os.getenv('apikey')

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

def generate_prompt(weather_data, gender, style):
    temp = weather_data["temp_f"]
    condition = weather_data["condition"]["text"].lower()
    if "rain" in condition or "storm" in condition:
        weather_prompt = "raincoat and waterproof shoes"
    elif temp < 50:
        weather_prompt = "warm winter outfit with a coat and boots"
    elif temp < 70:
        weather_prompt = "light jacket and jeans for mild weather"
    else:
        weather_prompt = "light summer outfit with short sleeves and sunglasses"
    prompt = f"Suggest a {style} outfit for a {gender} in {weather_prompt}."
    return prompt

def call_google_genai(prompt):
    client = genai.Client(api_key=genai.api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a stylist who gives outfit suggestions in one sentence and mentions the gender of the wearer, and you only give simple descriptions of the clothes -- no fabric, just what type of clothes and what they look like."
        ),
        contents=prompt,
    )
    return response.text.strip()

def encode_text(prompt):
    inputs = clip_processor(text=prompt, return_tensors="pt")
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

def find_best_image(prompt, folder_path):
    best_score = -1
    best_image_path = None
    prompt_vec = encode_text(prompt)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(folder_path, filename)
        image_vec = encode_image(image_path)
        similarity = torch.nn.functional.cosine_similarity(prompt_vec, image_vec).item()
        if similarity > best_score:
            best_score = similarity
            best_image_path = image_path
    return best_image_path

def get_outfit_images(description, base_folder):
    categories = {
        "top": os.path.join(base_folder, "tops"),
        "bottom": os.path.join(base_folder, "bottoms"),
        "shoes": os.path.join(base_folder, "shoes"),
        "outerwear": os.path.join(base_folder, "outerwear")
    }
    selected_images = {}
    for category, path in categories.items():
        if not os.path.isdir(path):
            continue
        category_prompt = f"{description}, focusing on {category}"
        best_image = find_best_image(category_prompt, path)
        if best_image:
            selected_images[category] = best_image
    return selected_images

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python weather.py <city> <gender> <style> <wardrobe_path>")
        sys.exit(1)

    city = sys.argv[1]
    gender = sys.argv[2]
    style = sys.argv[3]
    wardrobe = sys.argv[4]

    weather_data = weather_cache.get_weather(city)
    prompt = generate_prompt(weather_data, gender, style)
    print(f"Generated prompt: {prompt}")

    outfit_description = call_google_genai(prompt)
    print(f"Outfit suggestion: {outfit_description}")

    outfit_images = get_outfit_images(outfit_description, wardrobe)
    print("Selected clothing items:")
    for category, image_path in outfit_images.items():
        readable_path = os.path.normpath(image_path)
        print(f"{category}: {readable_path}")