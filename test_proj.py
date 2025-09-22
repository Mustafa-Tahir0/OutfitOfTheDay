import os
import unittest
from unittest import mock
import torch
from PIL import Image
from proj import generate_prompt, encode_text, encode_image, find_best_image, get_outfit_images
from database import weather_cache

weather_sample = {
    "temp_f": 45,
    "condition": {"text": "Rain"}
}

class TestProjFunctions(unittest.TestCase):

    def test_generate_prompt_cold_rain(self):
        prompt = generate_prompt(weather_sample, "female", "formal")
        self.assertIn("formal outfit for a female in raincoat and waterproof shoes", prompt.lower())

    def test_encode_text_output_shape(self):
        prompt = "a cozy winter outfit"
        vec = encode_text(prompt)
        self.assertIsInstance(vec, torch.Tensor)
        self.assertGreater(vec.shape[1], 0)

    def test_encode_image_output_shape(self):
        dummy_path = "test_temp_img.jpg"
        Image.new("RGB", (224, 224), color="red").save(dummy_path)
        vec = encode_image(dummy_path)
        self.assertIsInstance(vec, torch.Tensor)
        self.assertGreater(vec.shape[1], 0)
        os.remove(dummy_path)

    def test_find_best_image(self):
        tmp_dir = "temp_imgs"
        os.makedirs(tmp_dir, exist_ok=True)

        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i*40, i*40, i*40))
            img.save(os.path.join(tmp_dir, f"img{i}.jpg"))

        prompt = "a bright summer outfit"
        result = find_best_image(prompt, tmp_dir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith(".jpg"))
        for file in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, file))
        os.rmdir(tmp_dir)

    @mock.patch("database.weather_cache.requests.get")
    def test_get_weather_api_fallback(self, mock_get):
        mock_get.return_value.json.return_value = {
            "current": {"temp_f": 72.0, "condition": {"text": "Sunny"}}
        }
        weather = weather_cache.get_weather("TestCity")
        self.assertIn("temp_f", weather)
        self.assertIn("condition", weather)

    @mock.patch("proj.find_best_image")
    def test_get_outfit_images(self, mock_find):
        mock_find.return_value = "/fake/path.jpg"
        result = get_outfit_images("summer outfit", 'wardrobe')
        self.assertLessEqual(set(result.keys()), {"top", "bottom", "shoes", "outerwear"})
        self.assertTrue(all(path.endswith(".jpg") for path in result.values()))