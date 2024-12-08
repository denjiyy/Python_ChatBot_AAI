import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random


class ComprehensiveMusicalAI:
    def __init__(self):
        self.load_data()
        self.conversation_mode = "general"
        self.user_profile = {}
        self.initialize_neural_network()

    def initialize_neural_network(self):
        # Load a pre-trained language model
        model_name = "gpt2"  # Replace with a larger model if needed (e.g., "EleutherAI/gpt-neo-125M")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def load_data(self):
        self.genres = ["Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip-Hop", "R&B", "Country", "Folk", "Metal",
                       "Reggae", "Blues"]
        self.moods = ["Happy", "Sad", "Energetic", "Relaxed", "Angry", "Romantic", "Melancholic", "Excited", "Calm",
                      "Nostalgic"]
        self.instruments = ["Guitar", "Piano", "Drums", "Violin", "Saxophone", "Trumpet", "Bass", "Flute", "Cello",
                            "Synthesizer"]

    def get_next_question(self):
        prompts = [
            "What's your favorite genre of music?",
            "Do you play any musical instruments?",
            "How does music make you feel?",
            "What's your opinion on modern music trends?",
            "Do you enjoy live concerts or prefer listening to recorded music?",
        ]
        return random.choice(prompts)

    def process_response(self, question, response):
        # Use GPT-2 to generate a neural network response
        prompt = f"User's question: {question}\nUser's answer: {response}\nAI response:"
        return self.generate_response(prompt)

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the initial prompt from the generated text
        return response[len(prompt):].strip()


class MusicalAIChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Comprehensive Musical AI Chatbot")
        self.root.geometry("800x900")
        self.root.configure(bg="#f0f0f0")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", background="#4CAF50", foreground="white", font=("Arial", 12), padding=10)
        self.style.map("TButton", background=[("active", "#45a049")])
        self.style.configure("TEntry", font=("Arial", 12))

        self.ai = ComprehensiveMusicalAI()
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Comprehensive Musical AI Chatbot", font=("Arial", 24, "bold"))
        title_label.pack(pady=(0, 20))

        self.chat_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Arial", 12), height=30)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=10)

        self.message_entry = ttk.Entry(input_frame, font=("Arial", 12))
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.message_entry.bind("<Return>", self.send_message)

        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=(10, 0))

        self.display_message(
            "Welcome to the Comprehensive Musical AI Chatbot! Let's discuss your musical preferences.",
            "green"
        )

    def send_message(self, event=None):
        user_input = self.message_entry.get()
        if user_input.strip():
            self.display_message("You: " + user_input, "blue")
            ai_response = self.ai.process_response(self.ai.get_next_question(), user_input)
            self.display_message("AI: " + ai_response, "green")
            self.message_entry.delete(0, tk.END)

    def display_message(self, message, color):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n\n", color)
        self.chat_display.tag_config("blue", foreground="blue")
        self.chat_display.tag_config("green", foreground="green")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicalAIChatbot(root)
    root.mainloop()
