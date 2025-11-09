import os
import numpy as np
import cv2
import re
import PIL.Image
import base64
from io import BytesIO
import openai


def predict_number_plate(img, api_key):
    try:
        print(f"Input image shape: {img.shape if hasattr(img, 'shape') else 'No shape attribute'}")
        print(f"Input image type: {type(img)}")

        # Convert numpy array to PIL Image for base64 encoding
        if isinstance(img, np.ndarray):
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print("Converted BGR to RGB")
            else:
                img_rgb = img
                print("Image already in correct format")

            pil_img = PIL.Image.fromarray(img_rgb)
        else:
            pil_img = img

        # Convert PIL Image to base64
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print("Converted image to base64")

        # Use OpenAI GPT-4o Vision API
        print("Starting OpenAI Vision API processing...")
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the vehicle number plate text from this image. Return only the number plate text, nothing else. If no number plate is visible, return 'NO_PLATE_FOUND'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        result_text = response.choices[0].message.content.strip()
        print(f"OpenAI Vision result: '{result_text}'")

        if result_text and result_text != "NO_PLATE_FOUND":
            # Clean the text but preserve spaces and common plate characters
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s\-]', '', result_text).strip()
            print(f"Final cleaned text: '{cleaned_text}'")
            return cleaned_text, 0.95  # High confidence for GPT-4o
        else:
            print("No number plate found")
            return None, None

    except Exception as e:
        print(f"OCR processing error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
