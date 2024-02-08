#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import speech_recognition as sr
import pyttsx3


# In[ ]:


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# In[ ]:


def listen():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration = 1)
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        return query.lower()
    
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Can you please repeat?")
        return None


# In[ ]:


def main():
    speak("Hello! I'm your voice assistant. How can I help you today?")

    while True:
        query = listen()

        if query:
            if "hello" in query:
                speak("Hi there! How can I assist you?")
            elif "goodbye" in query or "exit" in query:
                speak("Goodbye! Have a great day.")
                break
            else:
                speak("I'm sorry. I don't understand that command. Please try again.")
                


# In[ ]:


if __name__ == "__main__":
    main()

