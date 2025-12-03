import torch
from dotenv import load_dotenv
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
import PIL.Image as Image
import os  

# ---------------------------
# CONFIGURATION (REPLACE YOUR KEY)
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

SYSTEM_PROMPT = """
Your name is Advait.
understand the user command and respond accordingly.
You are a friendly, wise Yoga & Ayurveda specialist.
You focus only on topics related to:
- Yoga and its benefits
- Spirituality & breathing practices
- Meditation techniques
- Ayurvedic herbs and medicines
- Natural healing

Keep answers short, clean, and helpful like a knowledgeable wellness coach.
You avoid topics unrelated to health, yoga, and Ayurveda.
you always respond in a positive and encouraging tone.
Your developer name is Sanjiv.
"""

chat = model.start_chat(history=[
    {"role": "user", "parts": SYSTEM_PROMPT}

])

print("Advait===> Namaste! How can I help you with Yoga & Ayurveda today?\n")


# ---------------------------
# STABLE DIFFUSION SETUP
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
).to(device)


# ---------------------------
# IMAGE GENERATION FUNCTION
# ---------------------------

def generate_image(prompt):
    """Generate AI image from text prompt and save with auto numbering."""

    # Create folder if not exists
    folder_path = "images"
    os.makedirs(folder_path, exist_ok=True)

    # Count and auto-increment image name
    existing_images = [f for f in os.listdir(folder_path) if f.startswith("advait_")]
    image_number = len(existing_images) + 1

    file_name = f"advait_{image_number}.png"
    file_path = os.path.join(folder_path, file_name)

    # Generate image
    with torch.no_grad():
        img = pipe(prompt).images[0]

    img.save(file_path)
    img.show()

    print(f"Advait===> Healing artwork created: {file_name} 🌄🧘‍♂️✨")
    print(f"Saved in folder: {folder_path}/\n")


# ---------------------------
# MAIN CHAT LOOP
# ---------------------------

while True:
    user_input = input("YOU===> ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\nAdvait===> Stay healthy, Sanjiv 🌿✨ Namaste 🙏")
        break

    # Generate image when prompt starts with create
    if user_input.lower().startswith("create"):
        prompt = user_input.replace("create", "").strip()
        if prompt == "":
            print("Advait===> Please describe the image after 'create', like:")
            print("create yoga guru meditating on mountains")
        else:
            print("Advait===> Creating a peaceful visual… please relax 🌸✨")
            generate_image(prompt)
        continue

    # Normal chat
    response = chat.send_message(user_input)
    print("Advait===>", response.text, "\n")
