import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

# Languages and models to evaluate
languages = ["English", "Dutch", "German", "Arabic"]
model_names = [
    "meta-llama/Llama-3.1-8B-Instruct", 
    "Qwen/Qwen2.5-7B-Instruct", 
    "google/gemma-3-12b-it", 
    "mistralai/Mistral-7B-Instruct-v0.3"
]

# Creating a dictionary for color translations
translations = {
    "red": ["red", "rood", "rot", "أحمر", "الأحمر"],
    "white": ["white", "wit", "weiß", "أبيض", "الأبيض"],
    "blue": ["blue", "blauw", "blau", "أزرق", "الأزرق"],
    "green": ["green", "groen", "grün", "أخضر", "الأخضر"],
    "yellow": ["yellow", "geel", "gelb", "أصفر", "الأصفر"],
    "pink": ["pink", "roze", "rosa", "وردي", "الوردي", "زهر", "الزهر"],
    "magenta": ["magenta", "magenta", "magenta", "أرجواني", "الماجنتا", "ماجنتا"],
    "cyan": ["cyan", "cyaan", "cyan", "سماوي", "سماوى", "سيان"],
    "purple": ["purple", "paars", "lila", "أرجواني", "بنفسجي", "بنفسجى"],
    "tan": ["tan", "tan", "tan", "تان", "اسمر", "أسمر"],
    "khaki": ["khaki", "kaki", "khaki", "خاكي", "الكاكي", "خاكى"],
    "beige": ["beige", "beige", "beige", "بيج", "البيج"],
    "cream": ["cream", "crème", "creme", "كريم", "الكريمة", "كريمي", "كريمى"],
    "brown": ["brown", "bruin", "braun", "بني", "البني", "بنى", "البنى"],
    "olive": ["olive", "olijf", "oliv", "زيتوني", "الزيتوني", "زيتونى", "الزيتونى"]
}

# Index mapping for languages
lang_to_idx = {"English": 0, "Dutch": 1, "German": 2, "Arabic": 3,}

# Where to store final results
final_report = {}

# Evaluation loop
for model_id in model_names:
    print(f"Loading model: {model_id}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    model_results = {}

    # Load and process each language
    for lang in languages:
        print(f"\nWorking on: {lang}...", flush=True)
        if lang == "English":
            input_file = "colours_sentences.txt"
        else:
            input_file = f"colours_{lang.lower()}.txt"

        with open(input_file, "r", encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip()]

        # Load  and process expected answers
        with open("colours_answers.txt", "r", encoding="utf-8") as f:
            expected_answers = [l.strip() for l in f if l.strip()]
        
        responses = []

        # Single turn prompting
        for i in range(0, len(sentences), 4):
            conv_block = sentences[i : i+4]
            
            # Combine into single turn
            single_turn_prompt = " ".join(conv_block)
            
            # Prepare prompt
            history = [{"role": "user", "content": single_turn_prompt}]
            prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
            
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            responses.append(resp.replace("\n", " ").strip())

        # Save raw responses to a central file
        model_name = model_id.split("/")[-1]
        with open("colours_model_responses.txt", "a", encoding="utf-8") as output_responses:
            output_responses.write(f"Model: {model_id}, Language: {lang}\n")
            for idx, r in enumerate(responses):
                output_responses.write(f"{r}\n")
            output_responses.write("\n")
        
        # Evaluation
        total_correct = 0
        target_correct = 0  # Correct in the target language
        other_lang_correct = 0 # Correct but in a different language (often English)

        target_idx = lang_to_idx[lang]

        for idx, r in enumerate(responses):
            eng_key = expected_answers[idx].lower()
            all_variants = translations.get(eng_key, [eng_key])
            
            # Check if any variant is in the response
            if any(word in r.lower() for word in all_variants):
                total_correct += 1

                # Check if the correct answer is in the target language
                if lang == "Arabic":
                    # Arabic has different spellings for some colours
                    target_variants = all_variants[3:]
                else:
                    target_variants = [all_variants[target_idx]]

               # Check if any of the target language variants are in the response
                if any(tv in r.lower() for tv in target_variants):
                    target_correct += 1
                else:
                    # Colours is correct but in different language
                    other_lang_correct += 1

        # Calculate accuracies
        acc = total_correct / len(responses)
        target_acc = target_correct / len(responses)
        other_acc = other_lang_correct / len(responses)

        # Results per model and language
        model_results[lang] = {
            "Total Accuracy": f"{acc:.2f}",
            "Target Language Accuracy": f"{target_acc:.2f}",
            "Cross Language Accuracy": f"{other_acc:.2f}"
        }
        print(f"{lang} -> Total: {acc:.2f} (Where {target_acc:.2f} in own language, {other_acc:.2f} in other languages)")

    # Store results for the model
    final_report[model_id] = model_results
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()

# Save  final results to a txt file
with open("colours_final_report.txt", "w", encoding="utf-8") as f:
    for m_id, results in final_report.items():
        f.write(f"Model: {m_id}\n")  
        for lang, scores in results.items():
            line = f"{lang}: Total accuracy: {scores['Total Accuracy']}, Target language accuracy: {scores['Target Language Accuracy']}, Cross language accuracy: {scores['Cross Language Accuracy']}\n"
            f.write(line)
        f.write("\n") 
