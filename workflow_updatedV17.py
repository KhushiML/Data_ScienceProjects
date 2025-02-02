import cv2        
import numpy as np                                                                                                                                  
import tensorflow as tf                                                                                                                             
import requests                                                                                                                                     
from tensorflow.keras.models import load_model                                                                                                     
from tensorflow.keras.preprocessing.image import ImageDataGenerator                                                                                 
import os                                                                                                                                           
from datetime import datetime                                                                                                                       
from PIL import Image                                                                                                                               
from io import BytesIO                                                                                                                             
import base64                                                                                                                                       
import json                                                                                                                                        
import re                                                                                                                                           
import uuid                                                                                                                                         
from google.cloud import storage                                                                                                                  
import time    
import torch
import sys, json
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def categorize(frame_url, text_content):
    GPT4V_KEY = os.getenv('GPT4V_KEY')
    GPT4V_ENDPOINT = os.getenv('GPT4V_ENDPOINT')                                                                                                     
    headers ={                                                                                                                                             
                "Content-Type": "application/json",                                                                                                                 
                 "api-key": GPT4V_KEY,                                                                                                                           
               }                                                                                                                                                   
    prompt_template = """
    You are an AI assistant tasked with analyzing a frame from a news broadcast to identify and classify the individual shown in the image. You will also be provided with text content that could be relevant for gathering information.

    Categories:

    - Personality: A well-known public figure such as a politician, actor, or other celebrity. Consider their facial features, attire, and the likelihood of them appearing in this context. Generally, politicians are in white attire.

    - Anchor: A news anchor who presents the news from the studio. Look for professional attire, the presence of a news desk, and any background elements like studio graphics. **If there are multiple anchor images shown in the bottom part of the image frame, ignore them for classification.**

    - Reporter: A journalist who actively reports the news, often live from the event location. Reporters typically hold a microphone or other visible journalistic tools (e.g., notepad, camera, earpiece) and are often labeled on-screen with titles such as "LIVE REPORT" or similar. They are often in field gear or professionally dressed for the outdoors and may have a visible news logo. Avoid classifying politicians or public figures giving interviews or speeches as reporters. Individuals in advertisements or crowds should not be classified as reporters unless identified as part of a news report.

    - Guest: An individual invited to speak on a specific topic, often for their expertise or experience. Look for individuals in interview settings, either in-studio or remotely, and context clues like lower-third titles or related graphics.

    - Crowd: A group of individuals or bystanders in the background of the image who are not the focus of the news report and are not holding a microphone or journalistic tools.

    Analysis Instructions:

    1. Analyze the persons you are getting from the text content. If you are able to identify the person from a frame, categorize them, and if there is a crowd or if you are unable to process the frame, mark it as UNKNOWN.

    2. Contextual Analysis: Assess the environment, such as studio settings, outdoor locations, or interview formats, which may provide clues about the individual's role.

    3. Identifying Details: Look for facial recognition hints, attire (e.g., suits, casual wear, field gear), and accessories like microphones or earpieces.

    4. Text Analysis: Analyze any associated text that might provide context, such as subtitles, banners, or on-screen information.

    5. Name Recognition: Consider the provided name of the person and cross-reference it with the visual information.

    6. Visual Cues: Consider background elements like text overlays, graphics, or any identifying markers (e.g., studio logos, live report banners).

    7. Known Figures: If the person is a well-known public figure, utilize your knowledge to assist in classification.

    8. Purpose of Appearance: Evaluate the likely reason the individual is on screen—are they delivering news, providing commentary, or being interviewed?

    Text Support:

    - You may be provided with additional text relevant to the image clip. Use it to identify the person’s name and profession, if possible. If the name is present in the text and matches the image, return that name along with the category.
 
    Image Support:

    - Identify the name of the person from the context of the image (e.g., their appearance, attire, background).

    Edge Cases:

    - If there is more than one person in the image, analyze and classify each individual.

    Response Format:

    For each identified person, return the information in this format:

    - Name: [identified person name]
    - Category: [Personality/Anchor/Reporter/Guest/Crowd]

    If the personality name is not identified, return the response in the following format:

    - Name: Unknown
    - Category: [Personality/Anchor/Reporter/Guest/Crowd]
    """
    
    try:
        # Fetch the image from the frame URL
        response = requests.get(frame_url)
        img = Image.open(BytesIO(response.content))

        # Convert image to base64 string
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
        
        
            # Generate the prompt using the provided name
        full_prompt = f"{prompt_template}\n\nText Content: {text_content}"

            # Prepare the payload for the GPT-4V API
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": full_prompt},
                {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]}
            ]
        }

            # Send the request to the GPT-4V API
        api_response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload).json()
        
        #print(frame_url)

            # Extract name and category from the API response
        classification = api_response.get('choices', [{}])[0].get('message', {}).get('content', 'No classification')
        print(f"Classification: {classification}")

            # Parse the name and category from the classification response
        name_match = re.search(r"Name:\s*([A-Za-z\s]+)(?=\n|$)", classification)
        #name_match = re.search(r"Name: ([\w\s]+)", classification)
        category_match = re.search(r"Category: (\w+)", classification)
            
        if name_match and category_match:
            name = name_match.group(1)
            category = category_match.group(1)
            clean_name = name.strip().upper()
            return clean_name, category
        else:
            print(f"Failed to extract name or category for {frame_url}.")
            return "Unknown", "Unknown"


    except requests.RequestException as e:
        print(f"Error fetching or processing the frame: {e}")
        return "Unknown", "Unknown"


def process_image(image_url):
    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        # Open the image with Pillow
        image = Image.open(BytesIO(response.content))

        # Convert the image to RGB (OpenCV uses BGR by default)
        image = image.convert('RGB')

        # Convert Pillow image to NumPy array
        image_np = np.array(image)

        # Convert RGB to BGR (OpenCV uses BGR color format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        return image_bgr
    else:
        print(f"Failed to retrieve image from URL: {image_url}")
        return None

def save_to_bucket(image, file_name):
    bucket_name = os.getenv('BUCKET_NAME')
    credentials_path = os.getenv('CREDENTIALS_PATH')
    
    try:
        # Initialize the GCS client and access the bucket
        client = storage.Client.from_service_account_json(credentials_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Convert the image to JPEG format
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image to JPEG format.")

        # Upload the image to the bucket
        blob.upload_from_string(encoded_image.tobytes(), content_type="image/jpeg")
        return blob.public_url

    except GoogleAPIError as e:
        print(f"Google Cloud Storage error: {e}")
    except ValueError as e:
        print(f"Image processing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None

def save_unknown_to_db(id, cropped_image_url, video_url, frame_url):
    unknown_save_api = os.getenv("SAVE_UNKNOWN_API")
    params = {
        "substory_id": id,
        "per_name": "UNKNOWN",
        "cropped_image_url": cropped_image_url,
        "clip_url": video_url,
        "frame": frame_url
    }

    try:
        #print(params)
        response = requests.post(unknown_save_api, json=params)
        #print(response)
        response.raise_for_status()  # Raises an exception for HTTP errors
        #print(response.json())
        return response

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None

def update_database(api_endpoint, video_url, frame, personality_name, frame_time):
    params = {
        "video_url": video_url,
        "personality_name": personality_name,
        "frame": frame,
        "frame_time": frame_time
    }

    try:
        #print(params)
        # Make an HTTP POST request to the specified API endpoint
        response = requests.post(api_endpoint, params=params)
        
        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
        
        return response

    except Exception as e:
        print(f"An error occurred while updating the database: {e}")  # Handle all exceptions

    return None  # Return None if any error occurs

def detect_face(image):
    #image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_eye_data = []
    
    # Loop over detected faces to detect eyes within each face
    for (x, y, w, h) in faces:
        roi_gray = image[y:y+h, x:x+w]  # Region of Interest (face area)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Only consider the face if at least two eyes are detected
        if len(eyes) >= 1:
            face_eye_data.append(((x, y, w, h), eyes))

    return face_eye_data 


def predict_person(api_endpoint, model, video_url, frame_url, frame_time, train_generator, id):
    try:
        image = process_image(frame_url)
        face_eye_data = detect_face(image)

        if len(face_eye_data) == 0:
            print("No face with two eyes found")
            return []

        print("Face eye data", face_eye_data)

        predictions = []
        person_names = list(train_generator.class_indices.keys())

        for i, ((x, y, w, h), eyes) in enumerate(face_eye_data):
            face_image = cv2.resize(image[y:y+h, x:x+w], (224, 224))
            face_image = face_image.astype("float32") / 255.0
            face_image = np.expand_dims(face_image, axis=0)

            pred = model.predict(face_image)
            person_index = np.argmax(pred)
            accuracy = pred[0][person_index]

            # Removed the threshold check
            padding_w = int(1 * w)  # Adjust these multipliers based on the padding you want
            padding_h = int(1 * h)
            
            x_new = max(0, x - padding_w)
            y_new = max(0, y - padding_h)
            w_new = min(image.shape[1] - x_new, w + 2 * padding_w)
            h_new = min(image.shape[0] - y_new, h + 2 * padding_h)

            cropped_face_image = image[y_new:y_new+h_new, x_new:x_new+w_new]
            unknown_face_filename = f"unknown_faces/{uuid.uuid4()}.jpg"

            # Here, we assume that if the accuracy is below a certain point, we still process as "UNKNOWN"
            if accuracy < 0.85:  # Optional, keep as a check for processing unknowns
                print("UNKNOWN found accuracy < threshold")
                
                # Save cropped image to GCS
                cropped_image_url = save_to_bucket(cropped_face_image, unknown_face_filename)
                print(cropped_image_url)

                # Update database with "UNKNOWN" entry
                response = update_database(api_endpoint, video_url, cropped_image_url, "UNKNOWN", frame_time)
                if response and response.status_code == 200:
                    print("Unknown frame processed and saved successfully.")
                else:
                    print(f"Failed to process frame. Error: {response.text if response else 'No response from API'}")
                
                # Save the unknown face to the master table
                response1 = save_unknown_to_db(id, cropped_image_url, video_url, frame_url)
                if response1 and response1.status_code == 201:
                    print("Unknown saved successfully to master table.")
                else:
                    print(f"Failed to save to master table. Error: {response1.text if response1 else 'No response from API'}")
                
                predictions.append("UNKNOWN")
            else:
                predicted_name = person_names[person_index]
                predictions.append(predicted_name)

        print("Predictions", predictions)
        return predictions

    except Exception as e:
        print(f"Error predicting personality: {e}")
        return []


def get_video_url(get_api):
    try:
        # Make an HTTP GET request to the specified API endpoint
        response = requests.get(get_api)
        
        # Check if the request was successful (status code 200)
        response.raise_for_status()  # Raises an HTTPError if the status code indicates an error
        
        # Attempt to parse the response as JSON
        data = response.json()
        return data

    except Exception as e:
        print(f"An error occurred: {e}")  # Handle all exceptions

    return None

def get_frames(api_url, id):
    try:
        # Prepare the request URL with the query parameter 'id'
        full_url = f"{api_url}?id={id}"

        # Send a POST request to the API
        response = requests.post(full_url, headers={"accept": "application/json"})

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            # Select frames based on the number of image URLs
            selected_frames = [ {"image_url": item["image_url"], "frame_datetime": item["frame_datetime"]} for item in data
                if "image_url" in item and "frame_datetime" in item]
            if len(selected_frames) > 10:
                interval = 5  # 10 seconds = 5 frames at 1 frame per 2 seconds
                selected_frames = selected_frames[::interval]
            else:
                selected_frames = selected_frames

            return selected_frames
        else:
            print(f"Failed to get data. Status code: {response.status_code}")
            return []

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def save_category(id, name, category):
    save_api_endpoint = os.getenv("SAVE_CATEGORY_API")
    params = {
        "id": id,
        "name": name,
        "category": category
    }

    try:
        # Make an HTTP POST request to the save API endpoint
        response = requests.post(save_api_endpoint, params=params)
        
        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError if the status code indicates an error
        
        return response

    except Exception as e:
        print(f"An error occurred while saving the category: {e}")  # Handle all exceptions

    return None  # Return None if any error occurs


def set_status(set_api, id):
    params = {
        "id": id
    }

    try:
        # Make an HTTP PATCH request to the set status API endpoint
        response = requests.patch(set_api, params=params)
        
        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError if the status code indicates an error
        
        # Print and return the JSON response
        #print(response.json())
        return response

    except Exception as e:
        print(f"An error occurred while setting the status: {e}")  # Handle all exceptions

    return None  # Return None if any error occurs


def process_video(model, train_generator, video_url, selected_frames, api_endpoint, id, text_content, channel_code):
    # Ensure directory exists
    check_personality_api = os.getenv("CHECK_PERSONALITY_API")

    for frame in selected_frames:
        frame_url = frame['image_url']
        frame_datetime = frame['frame_datetime']
        datetime_obj = datetime.fromisoformat(frame_datetime)
        frame_time = datetime_obj.time()

        personality_names = predict_person(api_endpoint, model, video_url, frame_url, frame_time, train_generator, id)

        # Analyze the frame using the OpenAI API
        openai_name, openai_category = categorize(frame_url, text_content)
        print(openai_name)
        print(openai_category)

        name_to_check = None

        # Determine the name to check
        if openai_name:
            name_to_check = openai_name
        elif personality_names:
            name_to_check = personality_names[0]  # Use the first predicted name if OpenAI fails

        # Check if OpenAI category requires validation
        if openai_category in ["Anchor", "Reporter"] and name_to_check != "UNKNOWN":
            # Validate against both categories
            categories_to_check = ["Anchor", "Reporter"]

            # Check the personality against the master table for both categories
            is_valid = False
            for category in categories_to_check:
                check_response = requests.get(check_personality_api, params={
                    "channel_code": channel_code,
                    "per_name": name_to_check,
                    "per_type": category
                })
                check_data = check_response.json()

                if check_data.get("result"):
                    is_valid = True
                    break  # No need to check further if found valid

            if is_valid:
                # Update the database with the validated name
                response = update_database(api_endpoint, video_url, frame_url, name_to_check, frame_time)
                result = save_category(id, name_to_check, openai_category)

                if response.status_code == 200:
                    print(f"Frame processed and saved successfully with name: {name_to_check}.")
                else:
                    print(f"Failed to process frame with name. Error: {response.text}")
            else:
                print(f"Personality '{name_to_check}' not found in the master for channel '{channel_code}'. Discarding this frame.")
                continue  # Discard this frame

        elif openai_name:
            # If category is not Anchor or Reporter, update with OpenAI name
            response = update_database(api_endpoint, video_url, frame_url, openai_name, frame_time)
            result = save_category(id, openai_name, openai_category)

            if response.status_code == 200:
                print(f"Frame processed and saved successfully with OpenAI name and category.")
            else:
                print(f"Failed to process frame with OpenAI name and category. Error: {response.text}")

        elif personality_names:
            # No OpenAI name; use the model's first predicted name
            model_name = personality_names[0]
            print(f"No OpenAI name. Using model-predicted name: {model_name}")

            # Update the database without checking personality if category is not Anchor or Reporter
            response = update_database(api_endpoint, video_url, frame_url, model_name, frame_time)
            result = save_category(id, model_name, openai_category)

            if response.status_code == 200:
                print(f"Frame processed and saved successfully with model-predicted name.")
            else:
                print(f"Failed to process frame with model-predicted name. Error: {response.text}")

    return None

def update_personality_code(id):
    update_api = os.getenv["UPDATE_PERSONALITY_CODE_API"]
    response = requests.patch(update_api, json={"id": id})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to update personality code for ID {id}: {response.text}")
        return None


def main():

    last_mp4_url = None
    get_api = os.getenv("GET_API")
    frames_api_url = os.getenv("FRAMES_API_URL")
    detect_faces_api = os.getenv("DETECT_FACES_API")
    set_api = os.getenv("SET_API")
    train_dir = os.getenv("TRAIN_DIR")
    model = os.getenv("MODEL_PATH")
    
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical'
    )
    
    while True:
        try:
            data = get_video_url(get_api)
            print(data)

            if data is None:
                print("No valid data received. Sleeping for 5 seconds.")
                time.sleep(5)
                continue
            
            video_url = data.get('clip_url')
            id = data.get('id')
            text_content = data.get('text_content')
            start_time = data.get('start_time')
            end_time = data.get('end_time')
            channel_code = data.get('channel_code')

            if not video_url:
                print("No valid data received. Sleeping for 5 seconds.")
                set_status(set_api, id)
                print(f"Set status for video ID: {id}")
                time.sleep(5)
                continue  # Avoid proceeding with invalid URL

            # Check if start and end dates are not equal
            if start_time and end_time:
                start_dt = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
                end_dt = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
                
                if start_dt.date() != end_dt.date():
                    print(f"Skipping video ID: {id} because start and end dates are not equal.")
                    set_status(set_api, id)
                    continue  # Skip processing this ID

            if video_url and video_url != last_mp4_url:
                print(f"Processing video ID: {id} with channel_code: {channel_code}")

                selected_frames = get_frames(frames_api_url, id)
                if not selected_frames:
                    print(f"Failed to get frames for video ID: {id}. Skipping this video.")
                    set_status(set_api, id)
                    continue

                print(f"Selected frames: {selected_frames}")
                
                try:
                    process_video(model, train_generator, video_url, selected_frames, detect_faces_api, id, text_content, channel_code)
                    print(f"Processed video ID: {id}")
                except Exception as e:
                    print(f"Error processing video ID: {id}. Error: {e}")
                    continue  # Skip to the next video
                
                # Update personality codes after processing the video
                update_result = update_personality_code(id)
                if update_result is not None:
                    print(f"Updated personality codes for video ID: {id}")
                else:
                    print(f"Failed to update personality codes for video ID: {id}")

                set_status(set_api, id)
                print(f"Set status for video ID: {id}")
                
                last_mp4_url = video_url

                # Kill the process and restart it
                print("Killing the process and restarting...")
                os.execv(sys.executable, ['python'] + sys.argv)

        except Exception as main_error:
            print(f"Error in main loop: {main_error}")
            time.sleep(5)  # Allow some time before retrying

if __name__ == "__main__":
    main()
