import cv2
import numpy as np
import argparse
import os
from PIL import Image # Pillow for image conversion
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as LamaConfig, HDStrategy, LDMSampler
import torch

# --- lama_cleaner imports ---
# Make sure lama_cleaner is installed: pip install lama-cleaner
# try:
#     from lama_cleaner.model_manager import ModelManager
#     from lama_cleaner.schema import Config as LamaConfig, HDStrategy, LDMSampler
#     import torch # To check for CUDA
# except ImportError:
#     print("Error: lama-cleaner or its dependencies are not installed.")
#     print("Please install it first: pip install lama-cleaner torch torchvision torchaudio opencv-python Pillow")
#     print("For GPU support, ensure PyTorch is installed with CUDA. Visit https://pytorch.org/")
#     exit(1)

# --- Global variables for mouse callback ---
drawing = False
ix, iy = -1, -1
mask_cv = None # OpenCV format mask (single channel, uint8)
img_display = None
original_img_cv = None # OpenCV format original image
BRUSH_SIZE = 20
WINDOW_NAME = "Draw Mask - ENTER: Process | C: Clear | +/-: Brush Size | ESC: Quit"
lama_model = None # To store the loaded LaMa model

def draw_mask_callback(event, x, y, flags, param):
    global ix, iy, drawing, img_display, mask_cv, BRUSH_SIZE

    current_img_for_display = img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cv2.circle(mask_cv, (x, y), BRUSH_SIZE, (255), -1) # Mask is white (255) on black (0)
        cv2.circle(current_img_for_display, (x, y), BRUSH_SIZE, (0, 0, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(mask_cv, (ix, iy), (x, y), (255), BRUSH_SIZE * 2)
            cv2.line(current_img_for_display, (ix, iy), (x, y), (0, 0, 255), BRUSH_SIZE * 2)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask_cv, (x, y), BRUSH_SIZE, (255), -1)
        cv2.circle(current_img_for_display, (x, y), BRUSH_SIZE, (0, 0, 255), -1)

    cv2.imshow(WINDOW_NAME, current_img_for_display)

def initialize_lama_model():
    global lama_model
    if lama_model is not None:
        return lama_model

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available. Using CPU for processing, which will be significantly slower.")
    else:
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


    print("Initializing LaMa model... (This may take a while on first run as models are downloaded)")
    # Available models: 'lama', 'ldm', 'zits', 'mat', 'fcf', 'manga'
    # 'lama' is generally a good default for inpainting.
    # For more control, you can explore other models and their specific settings.
    try:
        model_manager = ModelManager(
            name="lama", # Or 'ldm' for latent diffusion, etc.
            device=device,
            # You can specify other model-specific parameters if needed,
            # e.g., for LDM: ldm_steps=50, ldm_sampler=LDMSampler.ddim
        )
        lama_model = model_manager
        print("LaMa model initialized successfully.")
    except Exception as e:
        print(f"Error initializing LaMa model: {e}")
        print("Make sure you have a working internet connection for model download on first run.")
        print("Ensure PyTorch and lama-cleaner are correctly installed.")
        exit(1)
    return lama_model

def process_with_lama(image_pil, mask_pil, lama_config_override=None):
    """
    Processes the image with the loaded LaMa model.
    Args:
        image_pil (PIL.Image.Image): Original image in PIL format.
        mask_pil (PIL.Image.Image): Mask image in PIL format (grayscale 'L').
                                   White areas (255) are inpainted.
        lama_config_override (LamaConfig, optional): Override default config.

    Returns:
        PIL.Image.Image: Inpainted image in PIL format.
    """
    global lama_model
    if lama_model is None:
        lama_model = initialize_lama_model()

    # lama-cleaner's __call__ method directly takes PIL images
    # The mask should be a single channel image where 255 indicates the area to inpaint
    try:
        # The model object itself is callable
        inpainted_image_pil = lama_model(image=image_pil, mask=mask_pil, config=lama_config_override)
        return inpainted_image_pil
    except Exception as e:
        print(f"Error during LaMa processing: {e}")
        return None


def main():
    global mask_cv, img_display, original_img_cv, BRUSH_SIZE, WINDOW_NAME

    parser = argparse.ArgumentParser(description="Interactive watermark removal using LaMa (via lama-cleaner).")
    parser.add_argument("image_path", help="Path to the input image with watermark.")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    original_img_cv = cv2.imread(args.image_path)
    if original_img_cv is None:
        print(f"Error: Could not read image from {args.image_path}.")
        return

    print("Image loaded. A window will pop up for mask drawing.")
    print("Instructions:")
    print(" - Draw over the watermark (left-click and drag).")
    print(f" - Press 'ENTER' or 'S' to apply LaMa inpainting and save.")
    print(f" - Press 'C' to clear the mask.")
    print(f" - Press '+' or '=' to increase brush size.")
    print(f" - Press '-' or '_' to decrease brush size.")
    print(f" - Press 'ESC' to quit.")

    # Initialize mask (OpenCV format: single channel, black)
    mask_cv = np.zeros(original_img_cv.shape[:2], dtype=np.uint8)
    img_display = original_img_cv.copy()

    # Pre-initialize model to show download progress early if it's the first run
    initialize_lama_model()


    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, draw_mask_callback)

    while True:
        cv2.imshow(WINDOW_NAME, img_display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Exiting without processing.")
            break
        elif key == ord('c') or key == ord('C'):
            print("Mask cleared.")
            mask_cv = np.zeros(original_img_cv.shape[:2], dtype=np.uint8)
            img_display = original_img_cv.copy()
        elif key == ord('+') or key == ord('='):
            BRUSH_SIZE = min(100, BRUSH_SIZE + 2)
            print(f"Brush size: {BRUSH_SIZE}")
        elif key == ord('-') or key == ord('_'):
            BRUSH_SIZE = max(1, BRUSH_SIZE - 2)
            print(f"Brush size: {BRUSH_SIZE}")
        elif key == 13 or key == ord('s') or key == ord('S'):  # ENTER or S
            if not np.any(mask_cv):
                print("No mask drawn. Please draw a mask or press ESC to quit.")
                continue

            print("Processing image with LaMa... This might take some time.")
            cv2.destroyAllWindows() # Close drawing window before processing

            # Convert OpenCV image (BGR) to PIL Image (RGB)
            original_img_pil = Image.fromarray(cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2RGB))
            # Convert OpenCV mask (single channel) to PIL Image (grayscale 'L')
            mask_pil = Image.fromarray(mask_cv, mode='L')


            # --- Optional: Advanced LaMa Configuration ---
            # You can create a LamaConfig object to fine-tune behavior
            # For example, if you want to try High Definition strategies:
            # lama_config = LamaConfig(
            #     hd_strategy=HDStrategy.RESIZE, # or CROP or ORIGINAL
            #     hd_strategy_crop_margin=128,
            #     hd_strategy_crop_trigger_size=800,
            #     hd_strategy_resize_limit=720
            # )
            # result_pil = process_with_lama(original_img_pil, mask_pil, lama_config_override=lama_config)
            # --- Default Configuration ---
            result_pil = process_with_lama(original_img_pil, mask_pil)


            if result_pil:
                # Convert PIL Image (RGB) back to OpenCV image (BGR) for saving
                result_cv = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

                path_parts = os.path.splitext(args.image_path)
                output_path = f"{path_parts[0]}.new{path_parts[1]}"

                try:
                    cv2.imwrite(output_path, result_cv)
                    print(f"Inpainted image saved to: {output_path}")
                    # cv2.imshow("Result (Press any key)", result_cv) # Optionally show result
                    # cv2.waitKey(0)
                except Exception as e:
                    print(f"Error saving image: {e}")
            else:
                print("Processing failed.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()