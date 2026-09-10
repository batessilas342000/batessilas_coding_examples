import streamlit as st
import pandas as pd
import spacy
import os
import gdown
from annoy import AnnoyIndex
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model 'en_core_web_lg'...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"

@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):  
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    return pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()

ingredient_list = load_ingredient_data()

# Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []

    for ing in ingredient_list:
        vec = nlp(ing.lower()).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), filtered_ingredients

ingredient_vectors, filtered_ingredient_list = compute_embeddings()

# Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = AnnoyIndex(dim, metric="angular")  #  Uses angular distance (1 - cosine similarity)
    
    for i, vec in enumerate(ingredient_vectors):
        index.add_item(i, vec)
    
    index.build(50)  #  More trees = better accuracy
    return index
annoy_index = build_annoy_index()

#  Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.any(vec1) and np.any(vec2) else 0

def direct_search_alternatives(ingredient):
    input_vector = nlp(ingredient.lower().strip()).vector
    if np.any(input_vector):
        cosine_scores = []
        for i in range(len(filtered_ingredient_list)):
            target_name = filtered_ingredient_list[i]
            if target_name.lower() == ingredient.lower().strip():
                continue
            score = cosine_similarity(input_vector, ingredient_vectors[i])
            cosine_scores.append((target_name,score))

        cosine_scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = [item[0] for item in cosine_scores[:3]]
        return top_3
    return ["Ingredient not found"]



#  Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient):
    input_vector = nlp(ingredient.lower().strip()).vector
    if np.any(input_vector):
        indicies = annoy_index.get_nns_by_vector(input_vector, 4)
        results = []
        for i in indicies:
            similar_ingredient = filtered_ingredient_list[i]
            if similar_ingredient.lower() != ingredient.lower().strip():
                results.append(similar_ingredient)
        return results 
    return ["Ingredient not found"]

#  Generate Recipe
def generate_recipe(ingredients, cuisine):
    input_text = (
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
        f"Let's create a dish inspired by {cuisine} cuisine with these ingredients. Here are the preparation and cooking instructions:"
    )    
    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], 
                             max_length=250, num_return_sequences=1,
                             repetition_penalty=1.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

def get_helper_response(user_question, current_recipe):
    prompt = (f"You are a kitchen helper that is helping a user with this recipe:\n{current_recipe}"
              f"User Question:{user_question}\n"
              "Provide a helpful, concise answer based on the recipe above.")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(inputs["input_ids"], man_length = 250, num_return_sequences=1, repitition_penalty=1.2)

    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt,"").strip()

#  Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])
if st.button("Generate Recipe", use_container_width=True) and ingredients:
    st.session_state["recipe"] = generate_recipe(ingredients, cuisine)

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    st.download_button(label="📂 Save Recipe", 
                       data=st.session_state["recipe"], 
                       file_name="recipe.txt", 
                       mime="text/plain")
    
    # Add in the kitchen helper feature
    st.markdown("---")
    st.header("Your Kitchen Helper")

    for message in st.session.state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if follow_up := st.chat_input("Ask a question (e.g., 'How do I dice an oninon')"):
        st.session_state.messages.append({"role":"user", "content": follow_up})
        with st.chat_message("user"):
            st.markdown(follow_up)

        with st.chat_message("assistant"):
            if "recipe" in st.session_state:
                response = get_helper_response(follow_up, st.session_state["recipe"])
                st.markdown(response)
                st.session_state.messages.append({"role":"assistant", "content":response})
            else:
                st.warning("Please generate a recipe first!")

    #  Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio("Select Search Method:", ["Annoy (Fastest)", "Direct Search (Best Accuracy)"], index=0)

    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives
        }
        alternatives = search_methods[search_method](ingredient_to_replace)
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
