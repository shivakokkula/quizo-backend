import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
import re
from typing import List, Dict, Tuple, Optional
import warnings
import random
from collections import Counter
warnings.filterwarnings('ignore')

class RecipeDataset(Dataset):
    """Custom dataset for recipe generation"""
    
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features[0])
    
    def __getitem__(self, idx):
        feature_dict = {
            'text': torch.FloatTensor(self.features[0][idx]),
            'cuisine': torch.LongTensor([self.features[1][idx]]),
            'difficulty': torch.LongTensor([self.features[2][idx]]),
            'time': torch.FloatTensor([self.features[3][idx]]),
            'servings': torch.FloatTensor([self.features[4][idx]])
        }
        
        target_dict = {
            'ingredients': torch.FloatTensor(self.targets['ingredients'][idx]),
            'instructions': torch.FloatTensor(self.targets['instructions'][idx]),
            'nutrition': torch.FloatTensor(self.targets['nutrition'][idx])
        }
        
        return feature_dict, target_dict

class RecipeGeneratorNN(nn.Module):
    """Simplified Neural Network for Recipe Generation"""
    
    def __init__(self, 
                 text_input_dim,  # Dynamic based on actual TF-IDF features
                 cuisine_vocab_size=50,
                 difficulty_vocab_size=10,
                 max_ingredients=15,
                 max_instructions=10):
        
        super(RecipeGeneratorNN, self).__init__()
        
        self.max_ingredients = max_ingredients
        self.max_instructions = max_instructions
        
        # Text processing - simplified
        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Embedding layers
        self.cuisine_embedding = nn.Embedding(cuisine_vocab_size, 16)
        self.difficulty_embedding = nn.Embedding(difficulty_vocab_size, 8)
        
        # Combined feature processing
        combined_dim = 128 + 16 + 8 + 2  # text + cuisine + difficulty + time + servings
        
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Output heads - simplified to ingredient categories and basic instructions
        self.ingredient_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_ingredients),  # Binary: ingredient present or not
            nn.Sigmoid()
        )
        
        self.instruction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_instructions),  # Binary: instruction type present or not
            nn.Sigmoid()
        )
        
        self.nutrition_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # calories, protein, carbs, fat
            nn.ReLU()  # Ensure positive values
        )
    
    def forward(self, text, cuisine, difficulty, time, servings):
        # Process text
        text_features = self.text_encoder(text)
        
        # Process embeddings
        cuisine_emb = self.cuisine_embedding(cuisine).squeeze(1)
        difficulty_emb = self.difficulty_embedding(difficulty).squeeze(1)
        
        # Combine features
        combined = torch.cat([
            text_features, cuisine_emb, difficulty_emb, time, servings
        ], dim=1)
        
        features = self.feature_combiner(combined)
        
        # Generate outputs
        ingredients = self.ingredient_head(features)
        instructions = self.instruction_head(features)
        nutrition = self.nutrition_head(features)
        
        return {
            'ingredients': ingredients,
            'instructions': instructions,
            'nutrition': nutrition
        }

class CommercialRecipeGenerator:
    """
    Production-Ready Recipe Generation System
    - Fixed dimension issues
    - Simplified architecture for reliability
    - Windows-compatible
    """
    
    def __init__(self, max_features=1000):  # Reduced from 5000
        
        self.max_features = max_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Predefined ingredient and instruction categories
        self.ingredient_categories = [
            'protein', 'vegetables', 'grains', 'dairy', 'spices', 'oils', 
            'herbs', 'fruits', 'nuts', 'legumes', 'sauces', 'sweeteners',
            'baking', 'condiments', 'beverages'
        ]
        
        self.instruction_categories = [
            'prep', 'heat', 'cook', 'mix', 'season', 'serve', 
            'bake', 'boil', 'fry', 'simmer'
        ]
        
        # Encoders
        self.cuisine_encoder = LabelEncoder() 
        self.difficulty_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        
        # Model
        self.model = None
        self.is_trained = False
        
        # Recipe templates for realistic generation
        self.recipe_templates = {
            'pasta': {
                'ingredients': ['pasta', 'olive oil', 'garlic', 'salt', 'pepper'],
                'instructions': ['boil water', 'cook pasta', 'heat oil', 'add garlic', 'combine', 'serve']
            },
            'stir_fry': {
                'ingredients': ['oil', 'onion', 'garlic', 'vegetables', 'soy sauce'],
                'instructions': ['heat oil', 'add onion', 'add vegetables', 'stir fry', 'season', 'serve']
            },
            'soup': {
                'ingredients': ['broth', 'vegetables', 'onion', 'salt', 'pepper'],
                'instructions': ['heat broth', 'add vegetables', 'simmer', 'season', 'serve hot']
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def extract_ingredient_features(self, ingredients: List[str]) -> np.ndarray:
        """Convert ingredients to category features"""
        features = np.zeros(len(self.ingredient_categories))
        
        ingredient_text = ' '.join([self.preprocess_text(str(ing)) for ing in ingredients])
        
        # Map ingredients to categories
        category_keywords = {
            'protein': ['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu', 'meat'],
            'vegetables': ['onion', 'tomato', 'pepper', 'carrot', 'potato', 'mushroom', 'spinach'],
            'grains': ['rice', 'pasta', 'bread', 'flour', 'wheat', 'oats'],
            'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt'],
            'spices': ['salt', 'pepper', 'cumin', 'paprika', 'cinnamon', 'ginger'],
            'oils': ['oil', 'olive', 'coconut', 'vegetable'],
            'herbs': ['basil', 'oregano', 'thyme', 'parsley', 'cilantro'],
            'fruits': ['lemon', 'lime', 'apple', 'tomato', 'orange'],
            'nuts': ['almond', 'walnut', 'peanut', 'cashew'],
            'legumes': ['beans', 'lentil', 'chickpea', 'pea'],
            'sauces': ['sauce', 'vinegar', 'soy', 'ketchup'],
            'sweeteners': ['sugar', 'honey', 'maple', 'syrup'],
            'baking': ['flour', 'baking', 'yeast', 'vanilla'],
            'condiments': ['mustard', 'mayo', 'ketchup'],
            'beverages': ['water', 'broth', 'stock', 'wine']
        }
        
        for i, category in enumerate(self.ingredient_categories):
            keywords = category_keywords.get(category, [])
            for keyword in keywords:
                if keyword in ingredient_text:
                    features[i] = 1.0
                    break
        
        return features
    
    def extract_instruction_features(self, instructions: List[str]) -> np.ndarray:
        """Convert instructions to category features"""
        features = np.zeros(len(self.instruction_categories))
        
        instruction_text = ' '.join([self.preprocess_text(str(inst)) for inst in instructions])
        
        # Map instructions to categories
        category_keywords = {
            'prep': ['chop', 'dice', 'slice', 'cut', 'peel', 'wash'],
            'heat': ['heat', 'warm', 'preheat'],
            'cook': ['cook', 'fry', 'saute'],
            'mix': ['mix', 'stir', 'combine', 'whisk'],
            'season': ['season', 'salt', 'pepper', 'taste'],
            'serve': ['serve', 'plate', 'garnish'],
            'bake': ['bake', 'oven', 'roast'],
            'boil': ['boil', 'water'],
            'fry': ['fry', 'pan'],
            'simmer': ['simmer', 'reduce']
        }
        
        for i, category in enumerate(self.instruction_categories):
            keywords = category_keywords.get(category, [])
            for keyword in keywords:
                if keyword in instruction_text:
                    features[i] = 1.0
                    break
        
        return features
    
    def prepare_training_data(self, recipes_data: List[Dict]) -> Tuple:
        """Prepare data for training"""
        
        texts = []
        cuisines = []
        difficulties = []
        cooking_times = []
        servings_list = []
        
        ingredients_targets = []
        instructions_targets = []
        nutrition_targets = []
        
        for recipe in recipes_data:
            # Extract features
            name = recipe.get('name', '')
            description = recipe.get('description', '')
            text = f"{name} {description}"
            texts.append(text)
            
            cuisines.append(recipe.get('cuisine', 'unknown'))
            difficulties.append(recipe.get('difficulty', 'medium'))
            cooking_times.append(float(recipe.get('cooking_time', 30)))
            servings_list.append(float(recipe.get('servings', 4)))
            
            # Process targets
            ingredients = recipe.get('ingredients', [])
            instructions = recipe.get('instructions', [])
            nutrition = recipe.get('nutrition', {})
            
            ingredients_targets.append(self.extract_ingredient_features(ingredients))
            instructions_targets.append(self.extract_instruction_features(instructions))
            
            nutrition_targets.append([
                float(nutrition.get('calories', 300)),
                float(nutrition.get('protein', 15)),
                float(nutrition.get('carbs', 30)),
                float(nutrition.get('fat', 10))
            ])
        
        # Process text features
        text_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        print(f"Text features shape: {text_features.shape}")
        
        # Encode categorical features
        cuisine_encoded = self.cuisine_encoder.fit_transform(cuisines)
        difficulty_encoded = self.difficulty_encoder.fit_transform(difficulties)
        
        # Scale numerical features
        time_servings = np.column_stack([cooking_times, servings_list])
        time_servings_scaled = self.scaler.fit_transform(time_servings)
        
        features = [
            text_features,
            cuisine_encoded,
            difficulty_encoded,
            time_servings_scaled[:, 0],  # cooking times
            time_servings_scaled[:, 1]   # servings
        ]
        
        targets = {
            'ingredients': np.array(ingredients_targets),
            'instructions': np.array(instructions_targets),
            'nutrition': np.array(nutrition_targets)
        }
        
        return features, targets
    
    def train(self, recipes_data: List[Dict], epochs=50, batch_size=8, learning_rate=0.001):
        """Train the model"""
        
        print("Preparing training data...")
        features, targets = self.prepare_training_data(recipes_data)
        
        # Get actual text feature dimension
        text_input_dim = features[0].shape[1]
        print(f"Text input dimension: {text_input_dim}")
        
        # Split data
        indices = list(range(len(features[0])))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_features = [[feat[i] for i in train_idx] for feat in features]
        val_features = [[feat[i] for i in val_idx] for feat in features]
        
        train_targets = {k: v[train_idx] for k, v in targets.items()}
        val_targets = {k: v[val_idx] for k, v in targets.items()}
        
        # Create datasets
        train_dataset = RecipeDataset(train_features, train_targets)
        val_dataset = RecipeDataset(val_features, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model with correct dimensions
        self.model = RecipeGeneratorNN(
            text_input_dim=text_input_dim,
            cuisine_vocab_size=len(self.cuisine_encoder.classes_),
            difficulty_vocab_size=len(self.difficulty_encoder.classes_),
            max_ingredients=len(self.ingredient_categories),
            max_instructions=len(self.instruction_categories)
        ).to(self.device)
        
        # Loss functions and optimizer
        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        print(f"Training on {self.device}...")
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                # Move to device
                text = batch_features['text'].to(self.device)
                cuisine = batch_features['cuisine'].to(self.device)
                difficulty = batch_features['difficulty'].to(self.device)
                time = batch_features['time'].to(self.device)
                servings = batch_features['servings'].to(self.device)
                
                target_ingredients = batch_targets['ingredients'].to(self.device)
                target_instructions = batch_targets['instructions'].to(self.device)
                target_nutrition = batch_targets['nutrition'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(text, cuisine, difficulty, time, servings)
                
                # Calculate losses
                ingredient_loss = criterion_bce(outputs['ingredients'], target_ingredients)
                instruction_loss = criterion_bce(outputs['instructions'], target_instructions)
                nutrition_loss = criterion_mse(outputs['nutrition'], target_nutrition)
                
                total_loss = ingredient_loss + instruction_loss + 0.1 * nutrition_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
            
            scheduler.step()
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        text = batch_features['text'].to(self.device)
                        cuisine = batch_features['cuisine'].to(self.device)
                        difficulty = batch_features['difficulty'].to(self.device)
                        time = batch_features['time'].to(self.device)
                        servings = batch_features['servings'].to(self.device)
                        
                        target_nutrition = batch_targets['nutrition'].to(self.device)
                        
                        outputs = self.model(text, cuisine, difficulty, time, servings)
                        val_loss += criterion_mse(outputs['nutrition'], target_nutrition).item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
        
        self.is_trained = True
        print("Training completed!")
    
    def generate_recipe(self, 
                       description: str,
                       cuisine: str = 'italian',
                       difficulty: str = 'medium',
                       cooking_time: int = 30,
                       servings: int = 4) -> Dict:
        """Generate a recipe from description"""
        
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Prepare inputs
            text_features = self.tfidf_vectorizer.transform([description]).toarray()
            
            # Handle unknown cuisines/difficulties
            if cuisine in self.cuisine_encoder.classes_:
                cuisine_encoded = self.cuisine_encoder.transform([cuisine])[0]
            else:
                cuisine_encoded = 0
                
            if difficulty in self.difficulty_encoder.classes_:
                difficulty_encoded = self.difficulty_encoder.transform([difficulty])[0]
            else:
                difficulty_encoded = 0
            
            # Scale numerical features
            time_servings_scaled = self.scaler.transform([[cooking_time, servings]])[0]
            
            # Convert to tensors
            text_tensor = torch.FloatTensor(text_features).to(self.device)
            cuisine_tensor = torch.LongTensor([[cuisine_encoded]]).to(self.device)
            difficulty_tensor = torch.LongTensor([[difficulty_encoded]]).to(self.device)
            time_tensor = torch.FloatTensor([[time_servings_scaled[0]]]).to(self.device)
            servings_tensor = torch.FloatTensor([[time_servings_scaled[1]]]).to(self.device)
            
            # Generate
            outputs = self.model(text_tensor, cuisine_tensor, difficulty_tensor, time_tensor, servings_tensor)
            
            # Decode outputs
            ingredients = self.decode_ingredients(outputs['ingredients'][0], description, cuisine)
            instructions = self.decode_instructions(outputs['instructions'][0], description, cuisine)
            nutrition = {
                'calories': max(100, int(outputs['nutrition'][0][0].item())),
                'protein': max(5, int(outputs['nutrition'][0][1].item())),
                'carbs': max(10, int(outputs['nutrition'][0][2].item())),
                'fat': max(3, int(outputs['nutrition'][0][3].item()))
            }
            
            return {
                'name': ' '.join(description.split()[:3]).title(),
                'description': description,
                'cuisine': cuisine,
                'difficulty': difficulty,
                'cooking_time': cooking_time,
                'servings': servings,
                'ingredients': ingredients,
                'instructions': instructions,
                'nutrition': nutrition
            }
    
    def decode_ingredients(self, ingredient_probs: torch.Tensor, description: str, cuisine: str) -> List[str]:
        """Generate realistic ingredients based on probabilities"""
        
        # Get active ingredient categories
        threshold = 0.3
        active_categories = []
        
        for i, prob in enumerate(ingredient_probs):
            if prob > threshold:
                active_categories.append(self.ingredient_categories[i])
        
        # Generate specific ingredients based on categories and context
        ingredients = []
        
        # Base ingredients mapping
        ingredient_map = {
            'protein': ['chicken breast', 'ground beef', 'salmon fillet', 'tofu', 'eggs'],
            'vegetables': ['onion', 'bell pepper', 'tomatoes', 'garlic', 'mushrooms', 'spinach'],
            'grains': ['rice', 'pasta', 'bread', 'quinoa', 'couscous'],
            'dairy': ['cheese', 'milk', 'butter', 'cream', 'yogurt'],
            'spices': ['salt', 'black pepper', 'paprika', 'cumin', 'oregano'],
            'oils': ['olive oil', 'vegetable oil', 'coconut oil'],
            'herbs': ['fresh basil', 'parsley', 'thyme', 'cilantro'],
            'fruits': ['lemon', 'lime', 'tomatoes'],
            'nuts': ['almonds', 'walnuts', 'pine nuts'],
            'legumes': ['black beans', 'chickpeas', 'lentils'],
            'sauces': ['soy sauce', 'tomato sauce', 'vinegar'],
            'sweeteners': ['sugar', 'honey'],
            'baking': ['flour', 'baking powder'],
            'condiments': ['mustard', 'mayonnaise'],
            'beverages': ['chicken broth', 'water', 'wine']
        }
        
        # Cuisine-specific modifications
        cuisine_mods = {
            'italian': {'spices': ['oregano', 'basil'], 'cheese': 'parmesan', 'oil': 'olive oil'},
            'asian': {'sauce': 'soy sauce', 'oil': 'sesame oil', 'spices': ['ginger', 'garlic']},
            'indian': {'spices': ['turmeric', 'cumin', 'coriander'], 'oil': 'ghee'},
            'mexican': {'spices': ['cumin', 'chili powder'], 'vegetables': ['bell peppers', 'onions']}
        }
        
        # Generate ingredients
        for category in active_categories[:8]:  # Limit to 8 categories
            if category in ingredient_map:
                options = ingredient_map[category]
                
                # Apply cuisine modifications
                if cuisine in cuisine_mods and category in cuisine_mods[cuisine]:
                    ingredient = cuisine_mods[cuisine][category]
                    if isinstance(ingredient, list):
                        ingredient = random.choice(ingredient)
                else:
                    ingredient = random.choice(options)
                
                # Add quantity
                quantity = self.estimate_quantity(ingredient, servings)
                ingredients.append(f"{quantity} {ingredient}")
        
        # Ensure minimum ingredients
        if len(ingredients) < 4:
            basic_ingredients = ['2 tbsp olive oil', '1 medium onion', '2 cloves garlic', 'salt and pepper to taste']
            for basic in basic_ingredients:
                if len(ingredients) < 6:
                    ingredients.append(basic)
        
        return ingredients
    
    def decode_instructions(self, instruction_probs: torch.Tensor, description: str, cuisine: str) -> List[str]:
        """Generate realistic cooking instructions"""
        
        threshold = 0.3
        active_instructions = []
        
        for i, prob in enumerate(instruction_probs):
            if prob > threshold:
                active_instructions.append(self.instruction_categories[i])
        
        # Generate step-by-step instructions
        instructions = []
        
        instruction_templates = {
            'prep': "Wash and chop all vegetables. Prepare ingredients.",
            'heat': "Heat oil in a large pan over medium heat.",
            'cook': "Cook until tender and golden brown, about 5-7 minutes.",
            'mix': "Mix all ingredients together until well combined.",
            'season': "Season with salt and pepper to taste.",
            'serve': "Serve hot and enjoy!",
            'bake': "Bake in preheated oven according to recipe time.",
            'boil': "Bring water to a boil in a large pot.",
            'fry': "Fry until golden brown and crispy.",
            'simmer': "Reduce heat and simmer for 15-20 minutes."
        }
        
        # Create logical instruction sequence
        instruction_order = ['prep', 'heat', 'cook', 'mix', 'season', 'bake', 'boil', 'fry', 'simmer', 'serve']
        
        for step in instruction_order:
            if step in active_instructions:
                instructions.append(instruction_templates[step])
        
        # Ensure minimum instructions
        if len(instructions) < 3:
            instructions = [
                "Prepare all ingredients as needed.",
                "Cook according to your preferred method.",
                "Season to taste and serve hot."
            ]
        
        return instructions[:6]  # Limit to 6 steps
    
    def estimate_quantity(self, ingredient: str, servings: int = 4) -> str:
        """Estimate realistic quantities"""
        
        base_quantities = {
            'chicken': '500g', 'beef': '400g', 'fish': '400g', 'tofu': '300g',
            'rice': '1 cup', 'pasta': '300g', 'onion': '1 large', 'garlic': '3 cloves',
            'oil': '2 tbsp', 'butter': '2 tbsp', 'cheese': '100g',
            'tomatoes': '2 large', 'bell pepper': '1 large', 'mushrooms': '200g',
            'milk': '1 cup', 'cream': '1/2 cup', 'flour': '2 cups',
            'salt': '1 tsp', 'pepper': '1/2 tsp', 'herbs': '2 tbsp'
        }
        
        # Scale for servings
        for key in base_quantities:
            if key in ingredient.lower():
                return base_quantities[key]
        
        return '1 cup'  # Default quantity
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ingredient_categories': self.ingredient_categories,
            'instruction_categories': self.instruction_categories,
            'max_features': self.max_features
        }, f"{filepath}_model.pth")
        
        # Save preprocessors
        with open(f"{filepath}_preprocessors.pkl", 'wb') as f:
            pickle.dump({
                'cuisine_encoder': self.cuisine_encoder,
                'difficulty_encoder': self.difficulty_encoder,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler
            }, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        # Load model data  
        checkpoint = torch.load(f"{filepath}_model.pth", map_location=self.device)
        
        self.ingredient_categories = checkpoint['ingredient_categories']
        self.instruction_categories = checkpoint['instruction_categories']
        self.max_features = checkpoint['max_features']
        
        # Load preprocessors
        with open(f"{filepath}_preprocessors.pkl", 'rb') as f:
            preprocessors = pickle.load(f)
            self.cuisine_encoder = preprocessors['cuisine_encoder']
            self.difficulty_encoder = preprocessors['difficulty_encoder']
            self.tfidf_vectorizer = preprocessors['tfidf_vectorizer']
            self.scaler = preprocessors['scaler']
        
        # Get text input dimension from loaded vectorizer
        text_input_dim = len(self.tfidf_vectorizer.get_feature_names_out())
        
        # Recreate and load model
        self.model = RecipeGeneratorNN(
            text_input_dim=text_input_dim,
            cuisine_vocab_size=len(self.cuisine_encoder.classes_),
            difficulty_vocab_size=len(self.difficulty_encoder.classes_),
            max_ingredients=len(self.ingredient_categories),
            max_instructions=len(self.instruction_categories)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


def create_sample_recipes():
    """Create diverse sample training data"""
    return [
        {
            'name': 'Spaghetti Carbonara',
            'description': 'creamy pasta with eggs and bacon',
            'cuisine': 'italian',
            'difficulty': 'medium',
            'cooking_time': 25,
            'servings': 4,
            'ingredients': [
                '400g spaghetti pasta', '200g bacon diced', '4 large eggs',
                '100g parmesan cheese grated', '2 cloves garlic minced',
                'black pepper fresh ground', 'salt to taste'
            ],
            'instructions': [
                'boil large pot salted water cook spaghetti',
                'cook bacon pan until crispy',
                'whisk eggs parmesan bowl',
                'drain pasta reserve pasta water',
                'mix pasta bacon pan',
                'add egg mixture toss quickly',
                'season salt pepper serve hot'
            ],
            'nutrition': {'calories': 520, 'protein': 25, 'carbs': 55, 'fat': 22}
        },
        
    ]
