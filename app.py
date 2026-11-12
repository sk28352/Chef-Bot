"""
ChefBot: AI recipe generator with semantic ingredient search and an Ingredient Guard.

- Generates recipes with a small chat LLM (TinyLlama-1.1B-Chat) using configurable
  decoding (greedy, sampling, beam search) and selectable system-prompt styles.
- Finds ingredient alternatives with spaCy word vectors, using exact cosine search
  or fast approximate nearest neighbors (Annoy).
- Ingredient Guard: verifies the recipe actually uses the requested ingredients,
  retries with a stronger prompt when it doesn't, and links to real recipes for comparison.

Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import spacy
import os
import gdown
from annoy import AnnoyIndex
import numpy as np
import time
import re
from urllib.parse import quote_plus  # safely encode text for search URLs
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model 'en_core_web_lg'...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


import torch
@st.cache_resource
def load_llm():
   
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use the GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Half precision on GPU saves memory
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device
model, tokenizer, device = load_llm()
print(f"Model loaded on: {model.device}")


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

def direct_search_alternatives(ingredient, top_n=3):
    # Step 1: embed the input the same way the dataset was embedded (lowercase)
    query = ingredient.lower().strip()
    query_vec = nlp(query).vector
    if not np.any(query_vec):
        return ["No embedding found for this ingredient (unknown word)"]

    # Step 2: compare against EVERY ingredient, one by one
    scores = []
    for name, vec in zip(filtered_ingredient_list, ingredient_vectors):
        if name.lower().strip() == query:
            continue  # skip the ingredient itself (it would score 1.0)
        scores.append((name, cosine_similarity(query_vec, vec)))

    # Step 3: sort by similarity, highest first
    scores.sort(key=lambda pair: pair[1], reverse=True)

    # Step 4: return the top matches, with scores for analysis
    return [f"{name} ({score:.2f})" for name, score in scores[:top_n]]

#  Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient, top_n=3):
    # Step 1: get the prebuilt index (cached, so this returns instantly)
    index = build_annoy_index()

    # Step 2: embed the query exactly like the dataset was embedded
    query = ingredient.lower().strip()
    query_vec = nlp(query).vector
    if not np.any(query_vec):
        return ["No embedding found for this ingredient (unknown word)"]

    # Step 3: ask Annoy for nearest neighbors (one extra, in case the query itself is returned)
    ids, distances = index.get_nns_by_vector(query_vec, top_n + 1, include_distances=True)

    results = []
    for i, dist in zip(ids, distances):
        name = filtered_ingredient_list[i]
        if name.lower().strip() == query:
            continue  # skip the ingredient itself
        cos_sim = 1 - (dist ** 2) / 2  # convert Annoy's angular distance back to cosine similarity
        results.append(f"{name} ({cos_sim:.2f})")

    # Step 4: return the top matches
    return results[:top_n]

def generate_recipe(ingredients, cuisine, gen_settings, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Ingredients: {ingredients}\n"
            f"Cuisine: {cuisine}\n"
            f"Create a recipe using these ingredients."
        )},
    ]
    # Format the conversation the way TinyLlama-Chat was trained to see it
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(input_ids)  # no padding, so every token is real

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=400,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        **gen_settings,
    )
    # Keep only the newly generated tokens (everything after the prompt)
    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =====================================================================
#  Ingredient Guard
#  (1) verify the recipe actually uses the user's ingredients,
#  (2) auto-retry with a stronger prompt if any are missing,
#  (3) link to real human-made recipes so users can sanity-check the output.
# =====================================================================

def check_ingredient_coverage(ingredients, recipe_text, threshold=0.7):
    """For each user ingredient, return (ingredient, status, detail).
    status is 'exact', 'similar', or 'missing'."""
    # Prefer checking only the ingredient list, so mentions in the title or a closing line
    # ("Enjoy your Spinach Latte!") don't count as actually using the ingredient
    section = re.search(r"ingredients?\s*:(.*?)(instructions?|directions?|steps?|method)\s*:",
                        recipe_text, re.IGNORECASE | re.DOTALL)
    if section:
        body = section.group(1)
    else:
        # Fallback: no clear sections, so just skip a short first line (likely the title)
        lines = recipe_text.strip().split("\n")
        body = "\n".join(lines[1:]) if len(lines) > 1 and len(lines[0].split()) <= 10 else recipe_text

    recipe_lower = body.lower()
    recipe_doc = nlp(recipe_lower)
    recipe_lemmas = {tok.lemma_ for tok in recipe_doc}  # set = instant membership checks

    results = []
    for ing in [i.strip() for i in ingredients.split(",") if i.strip()]:
        ing_lower = ing.lower()
        ing_doc = nlp(ing_lower)
        ing_lemmas = [tok.lemma_ for tok in ing_doc if tok.is_alpha and not tok.is_stop]

        # Layer 1 + 2: whole-word match ("tea" must not match "teaspoon"),
        # or every word's dictionary form (lemma) appears in the recipe
        whole_word = re.search(rf"\b{re.escape(ing_lower)}\b", recipe_lower)
        if whole_word or (ing_lemmas and all(l in recipe_lemmas for l in ing_lemmas)):
            results.append((ing, "exact", None))
            continue

        # Layer 3: find the recipe word whose meaning is closest (reuses the cosine similarity from ingredient search)
        best_word, best_score = None, 0.0
        if np.any(ing_doc.vector):
            for tok in recipe_doc:
                if tok.has_vector and tok.is_alpha and not tok.is_stop:
                    score = cosine_similarity(ing_doc.vector, tok.vector)
                    if score > best_score:
                        best_word, best_score = tok.text, score

        if best_score >= threshold:
            results.append((ing, "similar", f"{best_word} ({best_score:.2f})"))
        else:
            results.append((ing, "missing", None))
    return results


def generate_with_guard(ingredients, cuisine, gen_settings, system_prompt, max_retries=2):
    """Generate, verify ingredient coverage, and retry with a stronger prompt if needed.
    Returns (recipe, coverage, chosen_attempt, total_attempts)."""
    prompt = system_prompt
    best = None
    for attempt in range(1, max_retries + 2):
        recipe = generate_recipe(ingredients, cuisine, gen_settings, prompt)
        coverage = check_ingredient_coverage(ingredients, recipe)
        missing = [ing for ing, status, _ in coverage if status == "missing"]

        # Keep the attempt with the fewest missing ingredients (a later retry could be worse)
        if best is None or len(missing) < best[3]:
            best = (recipe, coverage, attempt, len(missing))
        if not missing:
            break
        # Retry with an explicit instruction naming exactly what was missed
        prompt = system_prompt + f" IMPORTANT: The recipe MUST use these ingredients: {', '.join(missing)}."

    return best[0], best[1], best[2], attempt  # also return how many attempts were actually made


def build_verify_links(recipe_text, ingredients, cuisine):
    """Build YouTube/Google search links so users can compare with real recipes.
    Uses the recipe's title if it looks like one, otherwise falls back to ingredients + cuisine."""
    first_line = next((line.strip() for line in recipe_text.split("\n") if line.strip()), "")
    title = re.sub(r"^(title|recipe)\s*:\s*", "", first_line, flags=re.IGNORECASE)  # drop "Title:" prefix
    title = re.sub(r"\brecipe\b", "", title, flags=re.IGNORECASE).strip(" :-*#")

    if title and len(title.split()) <= 10:
        query = title
    else:  # first line was a paragraph, not a title
        query = f"{ingredients} {'' if cuisine == 'Any' else cuisine}".strip()

    encoded = quote_plus(f"{query} recipe")  # spaces -> '+', special characters escaped
    links = {
        "youtube": f"https://www.youtube.com/results?search_query={encoded}",
        "google": f"https://www.google.com/search?q={encoded}",
    }
    return query, links


#  Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])

# ---- Sidebar: generation settings ----
st.sidebar.header("⚙️ Generation Settings")
strategy = st.sidebar.radio("Decoding strategy", ["Greedy", "Sampling", "Beam search"])

if strategy == "Greedy":
    gen_settings = {"do_sample": False, "num_beams": 1}
elif strategy == "Sampling":
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, step=0.1)
    top_k = st.sidebar.slider("top_k (0 = off)", 0, 100, 50)
    top_p = st.sidebar.slider("top_p (1.0 = off)", 0.1, 1.0, 1.0, step=0.05)
    gen_settings = {"do_sample": True, "num_beams": 1,
                    "temperature": temperature, "top_k": top_k, "top_p": top_p}
else:  # Beam search
    num_beams = st.sidebar.slider("Number of beams", 2, 10, 5)
    gen_settings = {"do_sample": False, "num_beams": num_beams, "early_stopping": True}

# ---- Sidebar: system prompt selection ----
st.sidebar.header("📝 System Prompt")
SYSTEM_PROMPTS = {
    "Structured format": "You are a professional chef. Always respond in this exact order: "
                         "1) a recipe title, 2) an ingredient list with quantities, "
                         "3) numbered step-by-step instructions. Use only the given ingredients "
                         "plus common pantry staples.",
    "Concise": "You are a chef who writes very short recipes. Give a title, at most 6 ingredients, "
               "and at most 4 short steps. No extra commentary.",
    "Detailed": "You are a cooking instructor. Write a highly detailed recipe: title, ingredients "
                "with exact quantities, numbered steps with timings and temperatures, and tips "
                "for beginners.",
    "Creative": "You are an experimental chef known for surprising flavor combinations. "
                "Create an unconventional recipe that pairs the given ingredients in an "
                "unexpected way, and briefly explain why the pairing works.",
}
prompt_style = st.sidebar.selectbox("Prompt style", list(SYSTEM_PROMPTS.keys()))
system_prompt = SYSTEM_PROMPTS[prompt_style]

# ---- Sidebar: Ingredient Guard toggle ----
st.sidebar.header("🛡️ Ingredient Guard")
use_guard = st.sidebar.checkbox("Verify ingredients & auto-retry", value=True)


# ---- Generate button ----
if st.button("Generate Recipe", use_container_width=True) and ingredients:
    with st.spinner("Cooking up a recipe..."):
        start = time.perf_counter()
        if use_guard:
            recipe, coverage, chosen, attempts = generate_with_guard(ingredients, cuisine, gen_settings, system_prompt)
        else:
            recipe = generate_recipe(ingredients, cuisine, gen_settings, system_prompt)
            coverage, chosen, attempts = None, 1, 1
        elapsed = time.perf_counter() - start
    st.session_state["recipe"] = recipe
    st.session_state["coverage"] = coverage
    st.session_state["chosen"] = chosen
    st.session_state["attempts"] = attempts
    # Save inputs WITH the recipe so the verify links match it even if the inputs are edited later
    st.session_state["recipe_ingredients"] = ingredients
    st.session_state["recipe_cuisine"] = cuisine
    guard_label = "Guard ON" if use_guard else "Guard OFF"
    st.session_state["gen_info"] = f"{prompt_style} | {strategy} | {guard_label} | {gen_settings} | {elapsed:.1f}s"

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.caption(st.session_state.get("gen_info", ""))
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    # ---- Ingredient check results ----
    coverage = st.session_state.get("coverage")
    if coverage:
        st.markdown("#### 🛡️ Ingredient Check")
        st.caption(f"Made {st.session_state['attempts']} attempt(s); "
                   f"showing attempt {st.session_state['chosen']} (fewest missing ingredients)")
        for ing, status, detail in coverage:
            if status == "exact":
                st.markdown(f"✅ **{ing}**: used in the recipe")
            elif status == "similar":
                st.markdown(f"🟡 **{ing}**: not found exactly; closest match: {detail}")
            else:
                st.markdown(f"❌ **{ing}**: still missing after retries")

    # ---- Verify with real recipes ----
    query, links = build_verify_links(st.session_state["recipe"],
                                      st.session_state.get("recipe_ingredients", ""),
                                      st.session_state.get("recipe_cuisine", "Any"))
    st.markdown("#### 🔗 Verify with real recipes")
    st.caption(f"AI recipes can contain mistakes. Compare with human-made recipes for: \"{query}\"")
    col1, col2 = st.columns(2)
    col1.link_button("🎥 Search YouTube", links["youtube"], use_container_width=True)
    col2.link_button("🔎 Search Google", links["google"], use_container_width=True)

    st.download_button(label="📂 Save Recipe", 
                       data=st.session_state["recipe"], 
                       file_name="recipe.txt", 
                       mime="text/plain")

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
        start = time.perf_counter()
        alternatives = search_methods[search_method](ingredient_to_replace)
        search_time = time.perf_counter() - start
        st.caption(f"{search_method} took {search_time * 1000:.1f} ms")
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")