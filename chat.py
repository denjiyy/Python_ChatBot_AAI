import os
import json
import random
import datetime
import csv
import ssl
import streamlit as st
from transformers import pipeline
import re
from fuzzywuzzy import fuzz

ssl._create_default_https_context = ssl._create_unverified_context

file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

model_path = './fine_tuned_model'
nlp = pipeline("text-classification", model=model_path, tokenizer=model_path)

def chatbot(input_text):
    prediction = nlp(input_text)
    intent = prediction[0]['label']

    matched_intent = None
    for intent_data in intents['intents']:
        if intent_data['tag'].lower() == intent.lower():
            matched_intent = intent_data
            break

        for pattern in intent_data['patterns']:
            if fuzz.ratio(pattern.lower(), input_text.lower()) > 80:
                matched_intent = intent_data
                break

        if matched_intent:
            break

    if matched_intent:
        response = random.choice(matched_intent['responses'])

    else:
        response = fallback_response()
        learn_response = learn_from_user(input_text)
        if learn_response:
            response = learn_response

    return response



def fallback_response():
    return "I'm not sure I understand. Could you please clarify or try asking something else?"


def learn_from_user(input_text):
    response = st.text_input(
        f"I don't understand '{input_text}'. Would you like to add this to my knowledge? If yes, type the corresponding response.")

    if response:
        new_intent = {
            "tag": "user_added_intent",
            "patterns": [input_text],
            "responses": [response]
        }

        # Append new intent to the JSON file
        with open(file_path, "r") as file:
            intents = json.load(file)

        intents['intents'].append(new_intent)

        with open(file_path, "w") as file:
            json.dump(intents, file, indent=4)

        return f"Thanks for helping me learn! I've added this to my knowledge: '{input_text}' with response: '{response}'"
    return None


def main():
    st.title("Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # Track conversation history

    # Hide Streamlit Menu for better UX
    st.sidebar.markdown("# Menu")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Go to", menu)

    if choice == "Home":
        st.subheader("Start Chatting")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:", "")

        if user_input:
            st.session_state.conversation.append(f"You: {user_input}")
            response = chatbot(user_input)
            st.session_state.conversation.append(f"Chatbot: {response}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            for message in st.session_state.conversation:
                st.write(message)

            # End conversation if user says goodbye
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.session_state.conversation.clear()
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.header("About This Project")
        st.write("""
            This project aims to create a simple chatbot that can respond to user queries based on defined intents.
            The chatbot uses a fine-tuned model to predict the intent of user inputs and provide corresponding responses.
        """)

        st.subheader("Technology Used:")
        st.write("""
            - **Streamlit**: Used for creating the interactive web interface.
            - **Transformers (Hugging Face)**: Used for the pre-trained language model to handle intent classification.
            - **CSV**: Used for logging user inputs and chatbot responses for conversation history.
            - **Fuzzywuzzy**: Used to improve pattern matching and handle input variations.
        """)

        st.subheader("How It Works:")
        st.write("""
            1. The user interacts with the chatbot by typing in the input box.
            2. The input is processed and classified based on pre-defined patterns in the intents JSON file.
            3. The chatbot selects a response based on the intent prediction and displays it.
            4. All conversations are logged into a CSV file for future reference.
        """)

if __name__ == '__main__':
    main()