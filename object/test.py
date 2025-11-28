import base64
from io import BytesIO
from PIL import Image
from langchain_ollama import OllamaLLM

def convert_to_base64(image_path):
    """Convert image to base64 string"""
    with Image.open(image_path) as img:
        # --- THE FIX IS HERE ---
        # Convert RGBA (transparent) to RGB (standard colors)
        img = img.convert("RGB")
        # -----------------------

        # Resize to max 1024x1024 to save speed
        img.thumbnail((1024, 1024))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_image(image_path, prompt):
    print("Loading model... (this may take a moment)")
    llm = OllamaLLM(model="llava")
    
    try:
        image_b64 = convert_to_base64(image_path)
        # Bind the image to the prompt
        llm_with_image = llm.bind(images=[image_b64])
        return llm_with_image.invoke(prompt)
    except FileNotFoundError:
        return "Error: Could not find the image file. Check the path!"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Ensure this matches your actual file name
    image_file = r"C:\Users\karan\OneDrive\Desktop\gitdemo\AR-glasses\object\image.png" 
    
    question = "Describe this image in less than 20 words and read any text inside it."
    
    print(f"Analyzing {image_file}...")
    result = analyze_image(image_file, question)
    print("\n--- RESULT ---\n")
    print(result)