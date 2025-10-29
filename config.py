
import os

MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "media")
UPLOAD_DIR = os.path.join(MEDIA_ROOT, "uploads")
RESULT_DIR = os.path.join(MEDIA_ROOT, "results")

# You can customize default model repo ids here if needed.
SD15_REPO = "runwayml/stable-diffusion-v1-5"
CONTROLNETS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "lineart": "lllyasviel/sd-controlnet-lineart",
    # "openpose": "lllyasviel/sd-controlnet-openpose",  # optional
}
