import base64
import json
import os
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

p = os.path.dirname(os.path.realpath(__file__))


def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except:
        return {}


def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


class ComfyUI_NanoBanana_V2:
    def __init__(self, api_key=None):
        env_key = os.environ.get("GEMINI_API_KEY")

        # Common placeholder values to ignore
        placeholders = {"token_here", "place_token_here", "your_api_key",
                        "api_key_here", "enter_your_key", "<api_key>"}

        if env_key and env_key.lower().strip() not in placeholders:
            self.api_key = env_key
        else:
            self.api_key = api_key
            if self.api_key is None:
                config = get_config()
                self.api_key = config.get("GEMINI_API_KEY")

    @classmethod
    def IS_CHANGED(self, **kwargs) -> str:
        # 返回任意的随机数即可
        prompt = kwargs.get("prompt")
        md5_hash = hash(prompt)
        return str(md5_hash) + ":" + str(datetime.now().time())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Generate a high-quality, photorealistic image",
                    "multiline": True,
                    "tooltip": "Describe what you want to generate or edit"
                }),
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {
                    "default": "generate",
                    "tooltip": "Choose the type of image operation"
                }),
            },
            "optional": {
                "reference_image_1": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Primary reference image for editing/style transfer"
                }),
                "reference_image_2": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Second reference image (optional)"
                }),
                "reference_image_3": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Third reference image (optional)"
                }),
                "reference_image_4": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fourth reference image (optional)"
                }),
                "reference_image_5": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fifth reference image (optional)"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Gemini API key (paid tier required)"
                }),
                "api_endpoint": ("STRING", {
                    "default": "litellm-internal.123u.com/vertex_ai",
                    "tooltip": "API endpoint URL"
                }),
                "model_id": ("STRING", {
                    "default": "gemini-2.5-flash-image-preview",
                    "tooltip": "Model ID to use"
                }),
                "batch_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "Number of images to generate (costs multiply)"
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Creativity level (0.0 = deterministic, 2.0 = very creative)"
                }),
                "max_output_tokens": ("INT", {
                    "default": 32768,
                    "min": 1024,
                    "max": 65536,
                    "step": 1024,
                    "tooltip": "Maximum output tokens"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Top-p sampling parameter"
                }),
                "enable_safety": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable content safety filters"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "operation_log")
    FUNCTION = "nano_banana_generate"
    CATEGORY = "Nano Banana V2 (Gemini 2.5 Flash Image)"
    DESCRIPTION = "Generate and edit images using Google's Gemini 2.5 Flash Image model with streaming API. Throws errors instead of placeholder images."

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]

        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        return Image.fromarray(image_np, mode='RGB')

    def prepare_images_for_api(self, img1=None, img2=None, img3=None, img4=None, img5=None):
        """Convert up to 5 tensor images to base64 format for API"""
        image_parts = []

        # Process all provided images (up to 5)
        for i, img in enumerate([img1, img2, img3, img4, img5], 1):
            if img is not None:
                # Handle both single images and batched images
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 4:  # Batch of images
                        # Take only the first image from batch to avoid confusion
                        pil_image = self.tensor_to_image(img[0])
                    else:  # Single image
                        pil_image = self.tensor_to_image(img)

                    image_parts.append(self._image_to_base64_part(pil_image))

        return image_parts

    def _image_to_base64_part(self, pil_image):
        """Convert PIL image to base64 format for API"""
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        return {
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64.b64encode(img_bytes).decode('utf-8')
            }
        }

    def build_prompt_for_operation(self, prompt, operation, has_references=False):
        """Build optimized prompt based on operation type"""

        base_quality = "Generate a high-quality, photorealistic image"

        if operation == "generate":
            if has_references:
                final_prompt = f"{base_quality} inspired by the style and elements of the reference images. {prompt}."
            else:
                final_prompt = f"{base_quality} of: {prompt}."

        elif operation == "edit":
            if not has_references:
                raise ValueError("Edit operation requires reference images")
            final_prompt = f"Edit the provided reference image(s). {prompt}. Maintain the original composition and quality while making the requested changes."

        elif operation == "style_transfer":
            if not has_references:
                raise ValueError("Style transfer requires reference images")
            final_prompt = f"Apply the style from the reference images to create: {prompt}. Blend the stylistic elements naturally."

        elif operation == "object_insertion":
            if not has_references:
                raise ValueError("Object insertion requires reference images")
            final_prompt = f"Insert or blend the following into the reference image(s): {prompt}. Ensure natural lighting, shadows, and perspective."

        return final_prompt

    def parse_sse_response(self, response_text):
        """Parse Server-Sent Events response format"""
        images = []
        text_content = ""

        # Split by lines and process data: lines
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('data: '):
                try:
                    json_data = json.loads(line[6:])  # Remove 'data: ' prefix

                    if 'candidates' in json_data:
                        for candidate in json_data['candidates']:
                            if 'content' in candidate and 'parts' in candidate['content']:
                                for part in candidate['content']['parts']:
                                    # Extract text
                                    if 'text' in part:
                                        text_content += part['text']

                                    # Extract images
                                    if 'inlineData' in part and 'data' in part['inlineData']:
                                        image_data = base64.b64decode(part['inlineData']['data'])
                                        images.append(image_data)

                except json.JSONDecodeError:
                    continue

        return images, text_content

    def call_nano_banana_api(self, prompt, image_parts, temperature, max_output_tokens, top_p, enable_safety,
                             api_endpoint, model_id):
        """Make API call using streaming endpoint like the curl example"""

        if not self.api_key:
            raise ValueError("No API key provided. Gemini 2.5 Flash Image requires a PAID API key.")

        # Build request payload similar to the curl example
        parts = [{"text": prompt}]
        parts.extend(image_parts)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"],
                "topP": top_p
            }
        }

        # Add safety settings if disabled (like in curl example)
        if not enable_safety:
            payload["safetySettings"] = [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "OFF"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "OFF"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "OFF"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "OFF"
                }
            ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        url = f"https://{api_endpoint}/v1/projects/huanle-gemini/locations/global/publishers/google/models/{model_id}:streamGenerateContent"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()

            # Parse the SSE response
            images, text_content = self.parse_sse_response(response.text)

            if not images:
                raise RuntimeError(f"No images generated. Response text: {text_content[:200]}...")

            return images, text_content

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing API response: {str(e)}")

    def nano_banana_generate(self, prompt, operation, reference_image_1=None, reference_image_2=None,
                             reference_image_3=None, reference_image_4=None, reference_image_5=None, api_key="",
                             api_endpoint="litellm-internal.123u.com/vertex_ai",
                             model_id="gemini-2.5-flash-image-preview",
                             batch_count=1, temperature=1.0, max_output_tokens=32768, top_p=0.95, enable_safety=False):

        # Validate and set API key
        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key})

        if not self.api_key:
            raise ValueError(
                "NANO BANANA ERROR: No API key provided!\n\nGemini 2.5 Flash Image requires a PAID API key.\nGet yours at: https://aistudio.google.com/app/apikey\nNote: Free tier users cannot access image generation models.")

        try:
            # Process reference images (up to 5)
            image_parts = self.prepare_images_for_api(
                reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5
            )
            has_references = len(image_parts) > 0

            # Build optimized prompt (this will raise ValueError if operation requires references but none provided)
            final_prompt = self.build_prompt_for_operation(prompt, operation, has_references)

            # Log operation start
            operation_log = f"NANO BANANA V2 OPERATION LOG\n"
            operation_log += f"Operation: {operation.upper()}\n"
            operation_log += f"Reference Images: {len(image_parts)}\n"
            operation_log += f"Batch Count: {batch_count}\n"
            operation_log += f"Temperature: {temperature}\n"
            operation_log += f"Max Output Tokens: {max_output_tokens}\n"
            operation_log += f"Top P: {top_p}\n"
            operation_log += f"Safety Filters: {enable_safety}\n"
            operation_log += f"API Endpoint: {api_endpoint}\n"
            operation_log += f"Model: {model_id}\n"
            operation_log += f"Prompt: {final_prompt[:150]}...\n\n"

            # Track all generated images
            all_generated_images = []

            # Generate images for each batch
            for i in range(batch_count):
                try:
                    operation_log += f"Batch {i + 1}: Making API call...\n"

                    # Make API call
                    batch_images, response_text = self.call_nano_banana_api(
                        final_prompt, image_parts, temperature, max_output_tokens, top_p, enable_safety, api_endpoint,
                        model_id
                    )

                    all_generated_images.extend(batch_images)
                    operation_log += f"Batch {i + 1}: Generated {len(batch_images)} images\n"

                    if response_text:
                        operation_log += f"Response text: {response_text[:100]}...\n"

                except Exception as batch_error:
                    operation_log += f"Batch {i + 1} failed: {str(batch_error)}\n"
                    if i == 0:  # If first batch fails, raise error
                        raise batch_error

            # Process generated images into tensors
            if not all_generated_images:
                raise RuntimeError("No images were generated across all batches")

            generated_tensors = []
            for img_binary in all_generated_images:
                try:
                    # Convert binary to PIL image
                    image = Image.open(BytesIO(img_binary))

                    # Ensure it's RGB
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Convert to numpy array and normalize
                    img_np = np.array(image).astype(np.float32) / 255.0

                    # Create tensor with correct dimensions
                    img_tensor = torch.from_numpy(img_np)[None,]
                    generated_tensors.append(img_tensor)
                except Exception as e:
                    operation_log += f"Error processing image: {e}\n"

            if not generated_tensors:
                raise RuntimeError("Failed to process any generated images")

            # Combine all generated images into a batch tensor
            combined_tensor = torch.cat(generated_tensors, dim=0)

            # Calculate approximate cost
            approx_cost = len(generated_tensors) * 0.039  # ~$0.039 per image
            operation_log += f"\nEstimated cost: ~${approx_cost:.3f}\n"
            operation_log += f"Successfully generated {len(generated_tensors)} image(s)!"

            return (combined_tensor, operation_log)

        except Exception as e:
            # Re-raise the error instead of returning placeholder
            raise RuntimeError(f"NANO BANANA V2 ERROR: {str(e)}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "ComfyUI_NanoBanana": ComfyUI_NanoBanana_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_NanoBanana": "Nano Banana V2 (Gemini 2.5 Flash Image)",
}
