import os
import requests
import cv2
import torch
import open_clip
from sentence_transformers import util
from PIL import Image
import numpy as np
import time
import json
from flair.data import Sentence
from flair.models import SequenceTagger
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
import shutil

# Load the NER tagger (Flair model for Named Entity Recognition)
tagger = SequenceTagger.load("ner")

# Specify the path to your custom service.env file
load_dotenv(dotenv_path='service1.env')

prefix = os.getenv("PREFIX")

# Function to extract names from text content using Flair NER
def extract_names_from_text(text_content):
    sentence = Sentence(text_content)
    tagger.predict(sentence)
    names = [entity.text for entity in sentence.get_spans('ner') if entity.tag == 'PER']
    return names

# Function to check if a personality exists in the database and return the category (per_type)
def check_personality_category(per_name: str, check_personality_api: str):
    try:
        response = requests.get(check_personality_api, params={"per_name": per_name}, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if data.get("result") is True:
            # Personality found, return the per_type
            return data.get("per_type")
        else:
            # Personality not found, return default category "Personality"
            return "Personality"
    except requests.exceptions.RequestException as e:
        print(f"Error checking personality category for '{per_name}': {e}")
        return "Personality"  # Default category if there is an error
        
# Function to save extracted names into the database
def save_category(id, name, category):
    save_api_endpoint = os.getenv("SAVE_CATEGORY_API")
    params = {
        "id": id,
        "name": name,
        "category": category
    }
    try:
        # Send a POST request to save the name in the database
        response = requests.post(save_api_endpoint, params=params, verify=False)
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"Error saving category: {e}")
        return None
        
def update_personality_code(id):
    update_api = os.getenv("UPDATE_PERSONALITY_CODE_API")
    response = requests.patch(update_api, json={"id": id})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to update personality code for ID {id}: {response.text}")
        return None
        
def update_database(api_endpoint, video_url, frame, personality_name, frame_time):
    params = {
        "video_url": video_url,
        "personality_name": personality_name,
        "frame": frame,
        "frame_time": frame_time
    }

    try:
        # Send a POST request to the specified API endpoint
        response = requests.post(api_endpoint, params=params)
        
        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
        
        return response

    except Exception as e:
        print(f"An error occurred while updating the database: {e}")
        return None
        
# Function to get video data from the API
def get_video_data(get_api):
    try:
        # Make a GET request to fetch video data
        response = requests.get(get_api, verify=False)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching video data: {e}")
        return None

# Step 1: Fetch frame URLs (URLs of frames for the video)
def get_frame_urls(api_url, id):
    try:
        full_url = f"{api_url}?id={id}"
        response = requests.post(full_url, headers={"accept": "application/json"}, verify=False)
        
        if response.status_code == 200:
            data = response.json()
            selected_frames = [{"image_url": item["image_url"], "frame_datetime": item["frame_datetime"]} 
                               for item in data if "image_url" in item and "frame_datetime" in item]
            return selected_frames
        else:
            print(f"Failed to get data. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Step 2: Detect faces using OpenCV's Haar Cascade
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Step 3: Process and crop faces from images
def process_and_crop_face(image, save_dir, image_filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    faces = detect_face(image)
    for idx, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        cropped_face_filename = os.path.join(save_dir, f'{os.path.splitext(image_filename)[0]}_face_{idx+1}.jpg')
        cv2.imwrite(cropped_face_filename, face)
        print(f"Saved face {idx+1} to {cropped_face_filename}")
    return len(faces)  # Return number of faces detected

# Step 4: Filter blurry and non-eye images
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def has_visible_eyes(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_region)
        if len(eyes) > 0:
            return True 
    return False 

def filter_image(image, image_filename, save_dir, blur_threshold=100):
    if is_blurry(image, threshold=blur_threshold):
        print(f"Image {image_filename} is blurry. Skipping...")
        return False
    
    if not has_visible_eyes(image):
        print(f"Image {image_filename} has no visible eyes. Skipping...")
        return False

    output_path = os.path.join(save_dir, image_filename)
    cv2.imwrite(output_path, image)
    print(f"Saved filtered image {image_filename} to {save_dir}")
    return True

# Step 5: Image matching (Face Recognition)
# Initialize device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the CLIP model and transformations
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

# Function to encode an image using the CLIP model
def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

# Function to generate a cosine similarity score between two images
def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    return round(float(cos_scores[0][0]) * 100, 2)

# Function to find the matching person in GCS data based on face recognition
def find_matching_persons(test_image_path, organization_name, credentials_path, bucket_name, threshold=70.0):
    best_score = 0.0
    matched_name = "Unknown"
    storage_client = storage.Client.from_service_account_json(credentials_path)
    # List all channels and persons in the GCS folder structure
    blobs = storage_client.list_blobs(bucket_name, prefix=f"{prefix}/{organization_name}", delimiter='/')

    # Iterate over each channel (subfolder in the organization folder)
    for blob in blobs:
        channel_name = blob.name.split('/')[1]  # Extract the channel name
        print(f"Looking in channel: {channel_name}")

        # Now iterate over the persons (subfolders inside each channel)
        for person_blob in storage_client.list_blobs(bucket_name, prefix=f"{prefix}/{organization_name}/{channel_name}", delimiter='/'):
            person_name = person_blob.name.split('/')[2]  # Person name
            print(f"Looking at person: {person_name}")

            # Now loop through images inside the person folder
            for train_image_blob in storage_client.list_blobs(bucket_name, prefix=f"{prefix}/{organization_name}/{channel_name}/{person_name}", delimiter='/'):
                if train_image_blob.name.endswith(('.jpg', '.png')):  # Check for image files
                    # Download the image data
                    image_data = train_image_blob.download_as_bytes()
                    train_img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)

                    # Save the temporary file to disk (required by the generateScore function)
                    temp_train_image_path = f"/tmp/{train_image_blob.name.split('/')[-1]}"
                    cv2.imwrite(temp_train_image_path, train_img)

                    # Compute the score between the test image and the training image
                    score = generateScore(test_image_path, temp_train_image_path)

                    # Update best score and matched name if score exceeds threshold
                    if score > best_score:
                        best_score = score
                        if best_score > threshold:
                            matched_name = person_name  # Update matched person name

    return matched_name, best_score
    
def cleanup_folder(folder_path):
    """Delete all files in the specified folder."""
    if os.path.exists(folder_path):
        # Remove all files and subdirectories
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)  # Recreate the empty folder
        print(f"Cleaned up folder: {folder_path}")

def process_video(video_url, text_content, id, organization_name, bucket_name, credentials_path):
    # Initialize GCS client
    storage_client = storage.Client.from_service_account_json(credentials_path)

    # Ensure the directory for downloaded images exists
    temp_folder = f"{id}_downloaded_images"
    filtered_images_folder = f"{id}_filtered_images"
    
    # Clean up the folders if they already exist (remove previous images)
    cleanup_folder(temp_folder)
    cleanup_folder(filtered_images_folder)
    
    # Create the necessary directories if they don't exist
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(filtered_images_folder, exist_ok=True)

    # Extract names from the text content using NER
    names_from_text = extract_names_from_text(text_content)
    
    # Iterate over all extracted names and update the database
    for name_to_check in names_from_text:
        category = check_personality_category(name_to_check, os.getenv("CHECK_PERSONALITY_API"))
        
        # Save the name and category to the database
        response = save_category(id, name_to_check, category)
        
        if response and response.status_code == 200:
            print(f"Name from text '{name_to_check}' updated successfully with category '{category}'.")
            # Call update_personality_code after saving the category for names from text
            update_response = update_personality_code(id)
            if update_response:
                print(f"Personality code for ID {id} updated successfully.")
            else:
                print(f"Failed to update personality code for ID {id}.")
        else:
            print(f"Failed to update name from text '{name_to_check}' with category '{category}'. Error: {response.text}")

    # Process frames from the video
    api_url = os.getenv("FRAMES_API_URL")
    frames = get_frame_urls(api_url, id)
    if not frames:
        print(f"Failed to fetch frames for video ID {id}.")
        return

    # Process each frame for face detection, filtering, and matching
    for idx, frame in enumerate(frames):
        image_url = frame["image_url"]
        image_filename = f"frame_{idx+1}.jpg"
        image_path = os.path.join(temp_folder, image_filename)
        
        try:
            # Download the image for the current frame
            img_data = requests.get(image_url).content
            image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            print(f"Downloaded {image_url} and processed for face matching.")

            # Process and crop faces
            if image is None:
                print(f"Error reading image {image_filename}. Skipping...")
                continue

            num_faces = process_and_crop_face(image, filtered_images_folder, image_filename)

            if num_faces == 0:
                print(f"No faces detected in {image_filename}. Skipping...")
                continue

            # Filter the image
            if filter_image(image, image_filename, filtered_images_folder):
                print(f"Image {image_filename} passed filtering.")
                
                # Use GCS to match the test image with images stored in the "Training_Data" folder
                matched_name, best_score = find_matching_persons(image_path, organization_name, credentials_path, bucket_name)
                print(f"Image {image_filename} - Matched Person: {matched_name}, Similarity Score: {best_score}")
                
                # After finding the matched person, check the category of the matched name
                matched_category = check_personality_category(matched_name, os.getenv("CHECK_PERSONALITY_API"))
                
                # Extract frame time from the frame's datetime
                frame_datetime = frame['frame_datetime']
                datetime_obj = datetime.fromisoformat(frame_datetime)
                frame_time = datetime_obj.time()  # Extract time only (e.g., 14:30:00)

                # Call the update_database function to update the database with frame time and matched name
                update_response = update_database(os.getenv("UPDATE_API_ENDPOINT"), video_url, frame, matched_name, frame_time)
                if update_response and update_response.status_code == 200:
                    print(f"Database updated with matched name '{matched_name}' and frame time '{frame_time}' successfully.")
                else:
                    print(f"Failed to update database with matched name '{matched_name}' and frame time '{frame_time}'. Error: {update_response.text}")
                
                # Save the matched name and its category in the database
                response = save_category(id, matched_name, matched_category)
                if response and response.status_code == 200:
                    print(f"Matched name '{matched_name}' with category '{matched_category}' updated successfully.")
                    # Call update_personality_code after saving the matched name and category
                    update_response = update_personality_code(id)
                    if update_response:
                        print(f"Personality code for ID {id} updated successfully.")
                    else:
                        print(f"Failed to update personality code for ID {id}.")
                else:
                    print(f"Failed to update matched name '{matched_name}' with category '{matched_category}'. Error: {response.text}")
            else:
                print(f"Image {image_filename} did not pass filtering.")

        except Exception as e:
            print(f"Error processing {image_url}: {e}")
            
# Main loop for fetching video data
def main():
    get_api = os.getenv("GET_API")
    set_api = os.getenv("SET_API")

    while True:
        try:
            # Fetch video data from API
            data = get_video_data(get_api)

            if not data:
                print("No valid data received. Sleeping for 5 seconds.")
                time.sleep(5)
                continue  # Skip to the next iteration if no valid data

            video_url = data.get('clip_url')
            id = data.get('id')
            text_content = data.get('substory')
            channel_code = data.get('channel_code')
            
            if not video_url:
                print("No valid video URL. Sleeping for 5 seconds.")
                time.sleep(5)
                continue  # Skip to the next iteration if no valid video URL
            
            print(f"Processing video ID: {id} from channel code: {data['channel_code']}")

            # Process video and update database with extracted names
            process_video(video_url, text_content, id, os.getenv("ORGANIZATION_NAME"), os.getenv("BUCKET_NAME"), os.getenv("CREDENTIALS_PATH"))
            
            # Update status after processing the video
            response = requests.patch(set_api, params={"id": id}, verify=False)
            if response.status_code == 200:
                print(f"Status set successfully for video ID: {id}")
            else:
                print(f"Failed to set status for video ID: {id}. Error: {response.text}")

            # Sleep to prevent continuous polling (adjust the interval as needed)
            time.sleep(5)
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)  # Allow some time before retrying

if __name__ == "__main__":
    main()
