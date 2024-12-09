import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

# Load user preferences from file (or create if doesn't exist)
try:
    with open('user_preferences.json', 'r') as file:
        user_preferences = json.load(file)
except FileNotFoundError:
    user_preferences = {"liked_genres": [], "liked_songs": [], "disliked_songs": []}


def save_user_preferences():
    with open('user_preferences.json', 'w') as file:
        json.dump(user_preferences, file)


def update_user_preferences(feedback, song):
    if feedback == "liked":
        user_preferences["liked_songs"].append(song)
    elif feedback == "disliked":
        user_preferences["disliked_songs"].append(song)
    save_user_preferences()


while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")

                # If the intent is for music suggestions, handle feedback
                if tag == "personalized_music":
                    feedback = input("Did you like this song? (liked/disliked/skip): ").lower()
                    if feedback in ["liked", "disliked"]:
                        update_user_preferences(feedback, response.split("'")[1])  # Extract song name from response

                break
    else:
        print(f"{bot_name}: I do not understand...")
