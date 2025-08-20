from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

# Load model + tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

app = Flask(__name__)

# --- Helper functions ---
def softmax(logits):
    exp_logits = torch.exp(logits)
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

def entropy(prob_dist):
    return -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))

def get_token_confidence(entropy_value, vocab_size):
    max_entropy = math.log(vocab_size)
    confidence = max(0, 100 * (1 - entropy_value / max_entropy))
    return confidence

# --- Analyze functions ---
def analyze_prompt(prompt, content_tokens_only=True, content_fraction=0.5):
    """
    Computes model confidence.
    content_tokens_only: if True, compute only on last fraction of tokens
    content_fraction: fraction of tokens considered content (0-1)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0]
    vocab_size = logits.shape[-1]
    token_entropies = [entropy(torch.softmax(step_logits, dim=-1)).item() for step_logits in logits]

    # Use only the last fraction of tokens as "content"
    if content_tokens_only:
        n_tokens = len(token_entropies)
        start_idx = int(n_tokens * (1 - content_fraction))
        token_entropies = token_entropies[start_idx:]

    token_confidences = [get_token_confidence(e, vocab_size) for e in token_entropies]
    overall_confidence = sum(token_confidences) / len(token_confidences)
    return overall_confidence
def consensus_score(prompt, num_samples=5, content_fraction=0.5):
    """
    Generate multiple completions and measure agreement.
    Returns a score from 0 to 1 indicating consensus.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    vocab_size = model.config.vocab_size
    completions = []

    # Generate num_samples completions
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + 20,  # generate 20 tokens
                do_sample=True,
                top_k=50,   # randomness for sampling
                top_p=0.95,
            )
        token_ids = outputs[0][inputs['input_ids'].shape[1]:]  # only new tokens
        completions.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())

    # Compute pairwise agreement
    agreement_counts = []
    for i in range(len(completions)):
        for j in range(i+1, len(completions)):
            # simple overlap: fraction of matching words
            words_i = completions[i].split()
            words_j = completions[j].split()
            if len(words_i) == 0 or len(words_j) == 0:
                agreement_counts.append(0)
            else:
                matches = sum(1 for w in words_i if w in words_j)
                agreement_counts.append(matches / max(len(words_i), len(words_j)))

    # Consensus score = average pairwise agreement
    if len(agreement_counts) == 0:
        return 0.0
    return sum(agreement_counts) / len(agreement_counts)

def prompt_strength_score(prompt):
    # 1. Length / Specificity heuristic
    length_score = min(len(prompt.split()) / 20, 1.0)

    # 2. Constraint heuristic
    constraints_keywords = ['summarize', 'list', 'explain', 'format', 'audience', 'example', 'step', 'only']
    constraint_score = min(sum([1 for kw in constraints_keywords if kw in prompt.lower()]) / 3, 1.0)

    # 3. Model confidence heuristic (compute over content tokens)
    model_confidence = analyze_prompt(prompt, content_tokens_only=True) / 100  # normalize 0-1

    # Weighted average
    total_score = (0.3 * length_score) + (0.3 * constraint_score) + (0.4 * model_confidence)
    return round(total_score * 100, 1)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    prompt = data.get("prompt", "")
    confidence = (round(analyze_prompt(prompt, content_tokens_only=True), 1))
    strength = (prompt_strength_score(prompt))
    consensus = consensus_score(prompt)
    final_score = 1.6*(0.4 * confidence + 0.4 * strength + 0.2 * consensus)
    return jsonify({
        "confidence": final_score,
        "strength": 1.4*strength,
        "consensus": 1.6*consensus
    })

if __name__ == "__main__":
    # DEVELOPMENT: use Flask’s debug server
    # app.run(debug=True)

    # PRODUCTION on Windows using Waitress
    from waitress import serve
    print("Serving on http://0.0.0.0:8000")
    serve(app, host="0.0.0.0", port=8000)
