import os
import json
import random
from collections import Counter, defaultdict
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---- DATA LOADING ----
DATA_PATH = "./train.json"  # <-- Replace with your large Kaggle/open recipes file path

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, encoding='utf8') as f:
        DATASET = json.load(f)
else:
    # Fallback: tiny built-in sample (for quick testing)
    DATASET = [
        {"id": 10259, "cuisine": "greek", "ingredients": ["romaine lettuce", "black olives", "grape tomatoes", "garlic", "pepper", "purple onion", "seasoning", "garbanzo beans", "feta cheese crumbles"]},
        {"id": 25693, "cuisine": "southern_us", "ingredients": ["plain flour", "ground pepper", "salt", "tomatoes", "ground black pepper", "thyme", "eggs", "green tomatoes", "yellow corn meal", "milk", "vegetable oil"]},
        {"id": 22213, "cuisine": "indian", "ingredients": ["water", "vegetable oil", "wheat", "salt"]},
        {"id": 20130, "cuisine": "filipino", "ingredients": ["eggs", "pepper", "salt", "mayonaise", "cooking oil", "green chilies", "grilled chicken breasts", "garlic powder", "yellow onion", "soy sauce", "butter", "chicken livers"]},
    ]

# ---- DATA ANALYSIS ----

def build_cuisine_index(dataset):
    cuisine_ings = defaultdict(list)
    for item in dataset:
        cuisine_ings[item['cuisine']].extend([ing.lower().strip() for ing in item['ingredients']])
    cuisine_stats = {}
    for cuisine, ings in cuisine_ings.items():
        ctr = Counter(ings)
        cuisine_stats[cuisine] = {
            'most_common': [ing for ing,_ in ctr.most_common(20)],
            'all_ingredients': list(set(ings)),
        }
    return cuisine_stats

CUISINE_STATS = build_cuisine_index(DATASET)
CUISINE_LIST = sorted(CUISINE_STATS.keys())

# ---- LOGIC ----

def ai_generate_recipe_name(description, cuisine):
    prefix_map = {
        'greek': ['Classic', 'Island-style', 'Aegean', 'Mediterranean'],
        'southern_us': ['Southern', 'Homestyle', 'Cajun', 'Creole'],
        'indian': ['Spiced', 'Royal', 'Traditional', 'Masala'],
        'filipino': ['Pinoy', 'Island', 'Traditional'],
        'chinese': ['Sichuan', 'Cantonese', 'Wok-Fried', 'Traditional'],
        'american': ['Homestyle', 'Classic', 'Country'],
        'mexican': ['Authentic', 'Fiesta', 'Spicy', 'Street'],
        'japanese': ['Modern', 'Traditional', 'Tokyo-Style'],
        # ... add more mappings if you wish!
    }
    main = description.title()
    pre = random.choice(prefix_map.get(cuisine, ['Global', 'Fusion', 'Modern']))
    return f"{pre} {main}"

def ai_generate_ingredients(description, cuisine, servings):
    # Take sample + extras
    pool = CUISINE_STATS.get(cuisine, {}).get('most_common', [])
    if not pool:
        pool = [ing for item in DATASET for ing in item['ingredients']][:20]
    random.shuffle(pool)
    base_ings = pool[:7]
    # Add core item from user's description if word matches
    desc_words = [w.lower() for w in description.split()]
    added = False
    for w in desc_words:
        matches = [ing for ing in pool if w in ing]
        if matches:
            base_ings = [matches[0]] + [ing for ing in base_ings if ing != matches[0]]
            added = True
            break
    # Basic amount scaling (randomized for variety)
    result = []
    for ing in base_ings:
        if any(x in ing for x in ['oil', 'sauce', 'vinegar']):
            amt = f"{random.randint(1,3)*servings} tbsp"
        elif any(x in ing for x in ['flour', 'rice', 'sugar', 'meal']):
            amt = f"{random.randint(40,80)*servings}g"
        elif any(x in ing for x in ['egg', 'eggs']):
            amt = f"{max(1, servings//2+1)}"
        elif ing in ['water', 'milk']:
            amt = f"{random.randint(100,180)*servings}ml"
        else:
            amt = f"{random.randint(30,110)*servings}g"
        result.append(f"{amt} {ing}")
    return result

def ai_generate_instructions(description, cuisine):
    desc = description.lower()
    steps = []
    # Simple heuristics based on cuisine and description
    steps.append("1. Prepare all ingredients: clean, peel, and chop as needed.")
    if cuisine == 'indian':
        steps.append("2. Heat oil in a pan. Add whole spices, then onions, cook until golden.")
        steps.append("3. Add remaining spices and sauté, then add vegetables/proteins.")
        steps.append("4. Add water, simmer until cooked. Serve hot with rice or bread.")
    elif cuisine == 'greek':
        steps.append("2. Mix fresh vegetables and cheese in a large bowl.")
        steps.append("3. Add seasonings and olive oil, toss well. Chill and serve.")
    elif cuisine == 'southern_us':
        steps.append("2. Mix dry ingredients in one bowl, liquids in another.")
        steps.append("3. Combine gently, cook/fry/bake as recipe calls for.")
        steps.append("4. Serve warm with classic accompaniments.")
    elif cuisine == 'filipino':
        steps.append("2. Heat oil, add proteins and sauté until browned.")
        steps.append("3. Add aromatics, sauces, and vegetables.")
        steps.append("4. Simmer until tender and flavorful.")
    else:
        steps.append("2. Cook main protein/veg with oil. Add spices/seasonings.")
        steps.append("3. Simmer/bake/stir-fry until done. Rest briefly, serve.")
    return steps

def ai_generate_tags(description, cuisine):
    tags = [cuisine.replace('_', '-')]
    desc = description.lower()
    for word in ['vegetarian','vegan','quick','healthy','spicy','salad','soup']:
        if word in desc: tags.append(word)
    return list(set(tags))

# ---- ROUTES ----

@app.route('/', methods=['GET'])
def index():
    select_html = "\n".join(
        [f'<option value="{c}">{c.replace("_"," ").title()}</option>' for c in CUISINE_LIST]
    )
    # Double curly braces to escape for JavaScript templates in f-strings!
    return f"""<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Recipe Generator</title>
  <style>
    body {{ background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); font-family:sans-serif; min-height:100vh; margin:0; }}
    .container {{ max-width:630px; background:#fff; border-radius:18px; margin:48px auto;box-shadow:0 8px 32px #5554; padding:36px 28px; }}
    h1 {{ color:#495; font-size:2em; margin-bottom:8px; }} label{{margin-top:18px;font-weight:700;display:block;}}
    input,select{{width:100%;padding:9px 12px;margin-top:9px;font-size:1.05em;border-radius:7px;border:1.5px solid #ddd}}
    button{{background:#667eea;color:#fff;font-size:1.13em;border:none;padding:12px 30px;border-radius:19px;margin:20px 0;cursor:pointer;}}
    .recipe h2{{color:#363;text-shadow:1px 2px 2px #ccd4;}}
    ul,ol{{margin-left:24px;}}
    .tags span{{background:#667eea;color:#fff;border-radius:13px;padding: 3px 13px;margin:8px 7px 0 0;display:inline-block;}}
  </style>
</head>
<body>
<div class="container">
  <h1>🌎 Recipe Generator</h1>
  <form id="recipeForm">
    <label for="description">Recipe Description:</label>
    <input id="description" name="description" required placeholder="e.g. spicy paneer tikka, southern fried chicken">
    <label for="cuisine">Cuisine:</label>
    <select id="cuisine" name="cuisine" required>
      {select_html}
    </select>
    <label for="servings">Servings:</label>
    <input id="servings" name="servings" type="number" min="1" max="12" value="4" required>
    <button type="submit">Generate Recipe</button>
  </form>
  <div id="result"></div>
</div>
<script>
document.getElementById('recipeForm').addEventListener('submit',async function(e){{
  e.preventDefault();
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '<p>Generating recipe, one moment...</p>';
  const post = {{
    description: document.getElementById('description').value,
    cuisine: document.getElementById('cuisine').value,
    servings: document.getElementById('servings').value
  }};
  try {{
    const resp = await fetch('/generate',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(post)}});
    const data = await resp.json();
    let html = `<div class="recipe">
      <h2>${{data.name}}</h2>
      <div><i>Cuisine:</i> ${{data.cuisine}} &nbsp; | &nbsp; <i>Servings:</i> ${{data.servings}}</div>
      <div class="section"><h3>Ingredients</h3><ul>${{data.ingredients.map(i=>`<li>${{i}}</li>`).join('')}}</ul></div>
      <div class="section"><h3>Instructions</h3><ol>${{data.instructions.map(s=>`<li>${{s}}</li>`).join('')}}</ol></div>
      <div class="tags">${{data.tags.map(t=>`<span>${{t}}</span>`).join('')}}</div>
    </div>`;
    resultDiv.innerHTML = html;
  }} catch(e) {{
    resultDiv.innerHTML='<p>Error generating recipe. Please try again.</p>';
  }}
}});
</script>
</body></html>
"""

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    desc = data.get('description','delicious meal')
    cuisine = data.get('cuisine', random.choice(CUISINE_LIST))
    servings = int(data.get('servings', 4))
    recipe = {
        'name': ai_generate_recipe_name(desc, cuisine),
        'cuisine': cuisine.replace("_", " ").title(),
        'servings': servings,
        'ingredients': ai_generate_ingredients(desc, cuisine, servings),
        'instructions': ai_generate_instructions(desc, cuisine),
        'tags': ai_generate_tags(desc, cuisine)
    }
    return jsonify(recipe)

if __name__ == '__main__':
    app.run(debug=True)