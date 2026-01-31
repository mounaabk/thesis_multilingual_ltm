from typing_extensions import final
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import os

# Languages and models to evaluate
languages = ["English", "Dutch", "German", "Arabic"]
model_names = [
    "google/gemma-3-12b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

# Creating a dictionary for item translations
translations = {
    "bread": ["bread", "brood", "brot", "خبز", "الخبز", "أرغفة", "رغيف"],
    "milk": ["milk", "melk", "milch", "حليب", "الحليب", "ألبان"],
    "cheese": ["cheese", "kaas", "käse", "جبنة", "الجبنة", "أجبان", "جبن"],
    "carrot": ["carrot", "wortel", "karotte", "جزر", "الجزر", "جزرة", "جزرتين"],
    "chicken": ["chicken", "kip", "hähnchen", "دجاج", "الدجاج", "دجاجة", "دجاجتين"],
    "toilet paper": ["toilet paper", "toiletpapier", "toilettenpapier", "ورق", "ورق تواليت", "لفافة", "مناديل مرحاض"],
    "toothpaste": ["toothpaste", "tandpasta", "zahnpasta", "معجون", "معجون أسنان", "معجونتي"],
    "soap": ["soap", "zeep", "seife", "صابون", "الصابون", "صابونين"],
    "leek": ["leek", "prei", "lauch", "كراث", "الكراث", "كراثتين"],
    "celery": ["celery", "selderij", "sellerie", "كرفس", "الكرفس", "سيلري"],
    "steak": ["steak", "biefstuk", "steak", "ستيك", "الستيك", "شريحة لحم", "شريحتين"],
    "potato": ["potato", "aardappel", "kartoffel", "بطاطس", "البطاطس", "بطاطا", "بطاطتين"]
}

# Index mapping for languages
lang_to_idx = {"English": 0, "Dutch": 1, "German": 2, "Arabic": 3}

# Storage for final evaluation report
final_report = {}

# Safely extract and parse JSON from model response
def parse_flexible(text):
    # Search for JSON-style list structure in the text
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            content = json_match.group(0).replace("'", '"') 
            data = json.loads(content)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except:
            pass 

    # Fallback: manually extract items and quantities if JSON parsing fails
    extracted_items = []
    lines = text.split('\n')
    for line in lines:
        numbers = re.findall(r'\d+', line)
        if numbers:
            qty = int(numbers[0])
            item_name = re.sub(r'[\d\-\.\(\)\*•:]', '', line).strip().lower()
            if item_name:
                extracted_items.append({"item": item_name, "quantity": qty})
    
    return extracted_items

# Main evaluation loop
for model_id in model_names:
    print(f"Loading model: {model_id}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    model_results = {}

    # Iterate through each evaluation language
    for lang in languages:
        print(f"Processing: {lang}...", flush=True)
        if lang == "English":
            input_file = "shopping_sentences.txt"
        else:
            input_file = f"shopping_{lang.lower()}.txt"

        with open(input_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Load reference answers for scoring
        with open("shopping_answers.txt", "r", encoding="utf-8") as f:
            ground_truth_lines = [l.strip() for l in f if l.strip()]

        # Initialize score tracking for multi-turn metrics
        scores = {
            "Turn 1": {"any": [], "target": []},
            "Turn 2": {"any": [], "target": []},
            "Turn 3": {"any": [], "target": []}
        }
        
        history = []
        turn_pos = 1
        gt_idx = 0
        current_user_messages = [] # Buffer for turns without generation

        for line in lines:
            # Handle conversational reset at line 13
            if turn_pos == 13:
                history = []
                current_user_messages = []
                turn_pos = 1
                continue
            
            # Buffer user input
            current_user_messages.append(line)
            
            # Trigger model generation at turn boundaries (lines 4, 8, 12)
            if turn_pos in [4, 8, 12]:
                # Combine buffered messages into a single prompt for chat consistency
                combined_prompt = "\n".join(current_user_messages)
                history.append({"role": "user", "content": combined_prompt})
                current_user_messages = [] # Reset turn buffer

                # Generate model output
                input_ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids, 
                        max_new_tokens=400, 
                        do_sample=False, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()
                
                # Maintain chat history
                history.append({"role": "assistant", "content": response})
                
                # Calculate turn index and parse response
                actual_turn_num = turn_pos // 4
                json_data = parse_flexible(response)

                # Score metrics against ground truth
                if gt_idx < len(ground_truth_lines):
                    gt_dict = {n.strip().lower(): int(q) for pair in ground_truth_lines[gt_idx].split(",") for n, q in [pair.split(":")]}
                    
                    for mode in ["any", "target"]:
                        found_real, correct_qty, hallu_count = set(), 0, 0
                        
                        # Map target language variants for strict language scoring
                        target_variants_map = {} 
                        for eng_key, variants in translations.items():
                            if lang == "Arabic": 
                                target_variants_map[eng_key] = [v.lower() for v in variants[3:]]
                            else: 
                                target_variants_map[eng_key] = [variants[lang_to_idx[lang]].lower()]

                        for entry in json_data:
                            if not isinstance(entry, dict): continue
                            item_name = str(entry.get("item", "")).lower()
                            qty = entry.get("quantity", 0)
                            
                            matched_key, is_target = None, False
                            for eng_key, variants in translations.items():
                                if any(v.lower() in item_name for v in variants):
                                    matched_key = eng_key
                                    if any(tv in item_name for tv in target_variants_map[eng_key]):
                                        is_target = True
                                    break
                            
                            if matched_key in gt_dict:
                                if mode == "any" or (mode == "target" and is_target):
                                    if matched_key not in found_real:
                                        found_real.add(matched_key)
                                        if qty == gt_dict[matched_key]: 
                                            correct_qty += 1
                            else:
                                hallu_count += 1

                        # Calculate final turn score (Mean of Recall, Quantity Accuracy, and Hallucination Penalty)
                        s1 = len(found_real) / len(gt_dict) if gt_dict else 0
                        s2 = correct_qty / len(gt_dict) if gt_dict else 0
                        s3 = 1.0 if hallu_count == 0 else 0.0

                        final_score = (s1 + s2 + s3) / 3.0
                        scores[f"Turn {actual_turn_num}"][mode].append(final_score)
                    
                    gt_idx += 1
                
                # Log interaction for transparency
                with open("shopping_model_responses.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(f"ID: {model_id} | Lang: {lang} | Turn: {actual_turn_num} | JSON: {json.dumps(json_data)}\n")

            turn_pos += 1

        # Summarize results for the current language
        print(f"Summary for {lang}:", flush=True)
        lang_res = {}
        for t in ["Turn 1", "Turn 2", "Turn 3"]:
            any_score = sum(scores[t]['any'])/len(scores[t]['any']) if scores[t]['any'] else 0
            tar_score = sum(scores[t]['target'])/len(scores[t]['target']) if scores[t]['target'] else 0
            lang_res[t] = {"any": f"{any_score:.2f}", "target": f"{tar_score:.2f}"}
            print(f"  {t} -> Cross-Lingual: {any_score:.2f}, Target-Language: {tar_score:.2f}", flush=True)
        
        model_results[lang] = lang_res

    # Cleanup memory and store model results
    final_report[model_id] = model_results
    del model; torch.cuda.empty_cache()

# Generate comprehensive output report
with open("shopping_final_report.txt", "w", encoding="utf-8") as f:
    for m_id, results in final_report.items():
        f.write(f"Model: {m_id}\n")
        for lang, t_scores in results.items():
            f.write(f"{lang}:\n")
            for turn, s in t_scores.items():
                f.write(f"  {turn}: Any language: {s['any']}, Target language: {s['target']}\n")
        f.write("\n")

print("Shopping Task Evaluation Completed.", flush=True)
