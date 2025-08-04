from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch

class AIRecipeGenerator:
    def __init__(self):
        # Use DistilGPT-2 for faster, lighter model
        model_name = "distilgpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_recipe(self, ingredients="chicken, rice, vegetables"):
        prompt = f"Recipe for {ingredients}:\n\nIngredients:\n- {ingredients.replace(', ', '\n- ')}\n\nInstructions:\n1."
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=100, truncation=True)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_length=200,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean output
        recipe = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return recipe

# Alternative: Using pipeline (easier but less control)
class SimpleRecipeAI:
    def __init__(self):
        self.generator = pipeline("text-generation", model="distilgpt2", max_length=150)
    
    def generate_recipe(self, ingredients="chicken and rice"):
        prompt = f"Here's a simple recipe for {ingredients}:\n\nIngredients: {ingredients}\nInstructions:\n1."
        result = self.generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7)
        return result[0]['generated_text']

# Usage examples
if __name__ == "__main__":
    # Method 1: More control
    print("=== Method 1: Custom Implementation ===")
    ai1 = AIRecipeGenerator()
    recipe1 = ai1.generate_recipe("salmon, asparagus, lemon")
    print(recipe1)
    
    print("\n=== Method 2: Pipeline (Simpler) ===")
    ai2 = SimpleRecipeAI()
    recipe2 = ai2.generate_recipe("pasta and tomatoes")
    print(recipe2)

# Install requirements:
# pip install transformers torch