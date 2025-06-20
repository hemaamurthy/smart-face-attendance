import face_recognition
import os
import cv2
import numpy as np
from PIL import Image # Import Pillow for robust image handling
import io # To handle image bytes with Pillow

# --- Helper function to ensure image is 8-bit RGB for face_recognition ---
# Moved from app.py to this utility file for reusability and consistency.
def process_image_for_face_recognition(img_data_bytes):
    """
    Decodes image bytes (from file or base64) into a PIL Image,
    converts it to 8-bit RGB, then returns it as a NumPy array (RGB).
    This function makes image loading robust against various formats.
    """
    try:
        # Use Pillow to open image from bytes (more robust than cv2.imdecode for varied formats)
        pil_img = Image.open(io.BytesIO(img_data_bytes))

        # Convert to 8-bit RGB mode. This handles grayscale, CMYK, indexed, RGBA (flattens alpha).
        # face_recognition expects 8-bit RGB for best results.
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Convert PIL Image to NumPy array (which face_recognition expects as input)
        # The array will be in RGB order, which is what face_recognition expects.
        np_img = np.array(pil_img)

        # Ensure the numpy array is of type uint8 (8-bit)
        if np_img.dtype != np.uint8:
            np_img = np_img.astype(np.uint8)

        # Final check for shape (height, width, 3)
        if len(np_img.shape) != 3 or np_img.shape[2] != 3:
            print(f"Warning: Processed image has shape {np_img.shape}, expected (H, W, 3)")
            return None # Return None if the shape is unexpected, let caller handle
            
        return np_img

    except Exception as e:
        print(f"Error in process_image_for_face_recognition: {e}")
        return None

def load_known_faces(known_faces_dir):
    """
    Loads face images from a directory, processes them to ensure 8-bit RGB format,
    and then computes and returns their face encodings and names.
    """
    known_encodings, known_names, known_ids = [], [], [] # Added known_ids for consistency
    
    for filename in os.listdir(known_faces_dir):
        # Case-insensitive check for common image extensions
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(known_faces_dir, filename)
            
            try:
                # Read image bytes
                with open(full_path, 'rb') as f:
                    img_bytes = f.read()

                # Process the image using the robust helper
                # This ensures the image is always 8-bit RGB before face_recognition
                rgb_image = process_image_for_face_recognition(img_bytes)
                
                if rgb_image is None:
                    print(f"⚠️ Failed to process image file: {filename} in load_known_faces. Skipping.")
                    continue

                encodings = face_recognition.face_encodings(rgb_image)

                if encodings:
                    known_encodings.append(encodings[0])
                    # Extract name and user_id from filename (e.g., "John_123.jpg")
                    # Assuming format "name_userId.jpg"
                    base_name = os.path.splitext(filename)[0]
                    parts = base_name.split("_")
                    if len(parts) >= 2:
                        name = parts[0]
                        user_id = "_".join(parts[1:]) # Rejoin if user_id itself contains underscores
                    else:
                        print(f"Warning: Filename format unexpected for {filename}. Cannot extract name/ID. Skipping.")
                        continue
                    
                    known_names.append(name)
                    known_ids.append(user_id) # Store user_id
                else:
                    print(f"⚠️ No face detected in registered image: {filename}. Skipping.")

            except Exception as e:
                print(f"⚠️ Error loading or processing registered face {filename}: {e}. Skipping.")
                continue

    return known_encodings, known_names, known_ids # Return known_ids as well

def recognize_face(frame_rgb, known_encodings, known_names, known_ids):
    """
    Recognizes a face in a given RGB frame.
    
    Args:
        frame_rgb (numpy.ndarray): The input image frame, which MUST be an 8-bit RGB NumPy array.
                                   It is expected to have already been processed by
                                   process_image_for_face_recognition or similar.
        known_encodings (list): List of known face encodings.
        known_names (list): List of corresponding known names.
        known_ids (list): List of corresponding known user IDs.

    Returns:
        tuple: (name, user_id) of the recognized face, or (None, None) if no face is recognized.
    """
    # Verify input frame is indeed RGB and 8-bit
    if not (isinstance(frame_rgb, np.ndarray) and frame_rgb.dtype == np.uint8 and 
            len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 3):
        print("Error: Input frame to recognize_face is not a valid 8-bit RGB NumPy array.")
        return None, None

    # No cvtColor here, as the input 'frame_rgb' is already expected to be RGB
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    if not face_encodings:
        return None, None # No face detected in the current frame

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            return known_names[best_match_index], known_ids[best_match_index] # Return name and ID

    return None, None # No match found
