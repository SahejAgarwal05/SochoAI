from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import Ollama
from langchain.tools import tool
import base64
import os
from diffusers import StableDiffusionPipeline


from PIL import Image
import torch
youtube_channel_id ='sdfdsfsdfgsg'
def get_available_file_name(image_name: str) -> str:
    base_name, ext = os.path.splitext(image_name)
    counter = 1
    while os.path.exists(image_name):
        image_name = f"{base_name}_{counter}{ext}"
        counter += 1
    return image_name

def load_stable_diffusion():
    model_id = "CompVis/stable-diffusion-v1-4"  # Replace with your local Stable Diffusion 1.5 model path if needed
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline.to("cuda")  # Ensure GPU acceleration
    return pipeline

pipeline = load_stable_diffusion()
# Helper function to encode an image to Base64
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file at {image_path} not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define a custom tool with parameter descriptions
@tool
def custom_tool(prompt: str, width: int = 512, height: int = 512) -> str:
    """
    Generates an image using Stable Diffusion based on the provided prompt.
    The image is saved with an automatically generated unique name.

    Parameters:
        prompt (str): The textual description of the desired image.
        width (int): The width of the generated image in pixels. Defaults to 512.
        height (int): The height of the generated image in pixels. Defaults to 512.

    Returns:
        str: A message indicating where the generated image has been saved.
    """
    # Load Stable Diffusion pipeline
    pipeline = load_stable_diffusion()
    
    # Generate image
    image = pipeline(prompt, width=width, height=height).images[0]

    # Generate unique file name
    image_name = get_available_file_name()

    # Save image
    image.save(image_name)
    image.show()
    return f"Generated image saved at {image_name}."

# Initialize the Ollama LLM
llm = Ollama(model="llama3.2-vision")

# Define the tools
tools = [
    Tool(
        name="Custom Tool",
        func=custom_tool,
        description='generates image with Stable Diffusion based on the provided prompt',
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Start the interactive session
print("Multimodal Assistant: Type '/end' to exit.")

chat_history = [
    {
        "role": "system",
        "content": (
            "You are a highly intelligent multimodal assistant capable of analyzing both text and image inputs. "
            "Provide detailed and accurate responses based on user inputs, including visual context."
            '''Helpful guide to prompt stabel dffusion:

            '''
        )
    }
]

while True:
    # Get user input
    user_input = input("You: ").strip()

    # Exit if the user types /end
    if user_input.lower() == "/end":
        print("Ending the session. Goodbye!")
        break

    # Collect multiple image paths if the user wants to add images
    new_message = {"role": "user", "content": user_input}
    images = []
    add_images = input("Do you want to add images? (yes/no): ").strip().lower()
    if add_images == "yes":
        while True:
            image_path = input("Enter the image path (or type 'done' to finish): ").strip()
            if image_path.lower() == "done":
                break
            encoded_image = encode_image_to_base64(image_path)
            if encoded_image:
                images.append(encoded_image)
                print(f"Image {image_path} added successfully.")
        if images:
            new_message["images"] = images
    elif add_images != "no":
        print("Invalid response. Skipping image attachment.")

    # Append the message to chat history
    chat_history.append(new_message)

    # Use the agent to process the user's input
    try:
        response = agent.run({"input": user_input})
        print("Assistant:", response)
    except Exception as e:
        print(f"An error occurred while processing your input: {e}")

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

def load_stable_diffusion(model_name="CompVis/stable-diffusion-v1-4"):
    # Load the Stable Diffusion Image-to-Image pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(device)
    return pipeline

def get_available_file_name(base_name="output", extension="png"):
    import os
    i = 1
    while os.path.exists(f"{base_name}_{i}.{extension}"):
        i += 1
    return f"{base_name}_{i}.{extension}"

def edit_image(input_image_path, prompt, strength=0.75, guidance_scale=7.5, width=512, height=512):
    # Load the pipeline
    pipeline = load_stable_diffusion()

    # Load the input image
    input_image = Image.open(input_image_path).convert("RGB")
    input_image = input_image.resize((width, height))  # Resize to match the model's requirements

    # Generate the edited image
    edited_image = pipeline(prompt=prompt, image=input_image, strength=strength, guidance_scale=guidance_scale).images[0]

    # Generate a unique file name for the edited image
    image_name = get_available_file_name()

    # Save and display the edited image
    edited_image.save(image_name)
    edited_image.show()
    return f"Edited image saved at {image_name}."


