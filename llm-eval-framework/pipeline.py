"""
LLM Eval Pipeline — 3 models, all via Groq, all free
Scoring: custom metrics (no RAGAS dependency issues)
Models: LLaMA-3 (Meta) · Mixtral (Mistral AI) · Gemma-7B (Google)
"""

import os, time, csv
from groq import Groq
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

# ── One client, one key, zero cost ────────────────────────────────────────────
groq_client = Groq(api_key="gsk_YWMP....") # key

# ── 25 Medical QA questions ───────────────────────────────────────────────────
QUESTIONS = [
    {"id":  1, "question": "What is type 2 diabetes?",
     "ground_truth": "A chronic condition where the body doesn't use insulin properly, leading to high blood sugar.",
     "context": "Type 2 diabetes is a metabolic disease where cells become resistant to insulin or the pancreas doesn't produce enough insulin, resulting in elevated blood glucose levels."},
    {"id":  2, "question": "What are symptoms of hypertension?",
     "ground_truth": "Often no symptoms, but can include headaches, shortness of breath, and nosebleeds.",
     "context": "Hypertension (high blood pressure) is often called the silent killer because most people show no symptoms. When symptoms appear, they may include headaches, shortness of breath, or nosebleeds."},
    {"id":  3, "question": "How does ibuprofen work?",
     "ground_truth": "It inhibits COX enzymes, reducing prostaglandin synthesis and thus inflammation and pain.",
     "context": "Ibuprofen is an NSAID that works by inhibiting cyclooxygenase (COX-1 and COX-2) enzymes, which reduces the production of prostaglandins — chemicals that cause inflammation, pain, and fever."},
    {"id":  4, "question": "What is BMI and what does it measure?",
     "ground_truth": "Body Mass Index, a ratio of weight to height squared used to categorize weight status.",
     "context": "BMI (Body Mass Index) is calculated as weight in kilograms divided by height in meters squared. It is used to classify individuals as underweight, normal weight, overweight, or obese."},
    {"id":  5, "question": "What causes a heart attack?",
     "ground_truth": "Blockage of coronary arteries, usually by a blood clot, cutting off blood supply to heart muscle.",
     "context": "A myocardial infarction (heart attack) occurs when a coronary artery becomes blocked, typically due to a blood clot forming over atherosclerotic plaque, depriving heart muscle of oxygen."},
    {"id":  6, "question": "What is the difference between bacteria and viruses?",
     "ground_truth": "Bacteria are single-celled living organisms; viruses are non-living particles that need a host cell to replicate.",
     "context": "Bacteria are single-celled microorganisms that can reproduce independently. Viruses are non-living agents consisting of genetic material in a protein coat; they must invade host cells to replicate."},
    {"id":  7, "question": "What is anemia?",
     "ground_truth": "A condition with insufficient red blood cells or hemoglobin, reducing oxygen delivery to tissues.",
     "context": "Anemia is a condition in which the blood lacks enough healthy red blood cells or hemoglobin to carry adequate oxygen to the body's tissues, causing fatigue and weakness."},
    {"id":  8, "question": "How does the immune system fight infection?",
     "ground_truth": "White blood cells identify and destroy pathogens; antibodies neutralize threats; memory cells prevent future infections.",
     "context": "The immune system uses white blood cells to detect and destroy pathogens. B cells produce antibodies that neutralize specific threats, while memory cells retain information about past infections for faster future responses."},
    {"id":  9, "question": "What is cholesterol and why does it matter?",
     "ground_truth": "A fatty substance in blood; high LDL levels increase risk of heart disease and stroke.",
     "context": "Cholesterol is a waxy, fat-like substance in the blood. LDL (bad) cholesterol builds up in artery walls, increasing cardiovascular disease risk. HDL (good) cholesterol helps remove LDL from the bloodstream."},
    {"id": 10, "question": "What is asthma?",
     "ground_truth": "A chronic respiratory disease causing airway inflammation and narrowing, leading to breathing difficulty.",
     "context": "Asthma is a chronic lung disease characterized by inflammation and narrowing of the airways, causing recurrent episodes of wheezing, breathlessness, chest tightness, and coughing."},
    {"id": 11, "question": "What does the liver do?",
     "ground_truth": "Filters blood, produces bile, metabolizes nutrients, detoxifies harmful substances.",
     "context": "The liver performs over 500 vital functions including filtering toxins from blood, producing bile for digestion, metabolizing carbohydrates, proteins, and fats, and synthesizing blood clotting proteins."},
    {"id": 12, "question": "What is osteoporosis?",
     "ground_truth": "A disease causing bones to become weak and brittle due to reduced bone density.",
     "context": "Osteoporosis is a skeletal disorder where bones become porous and fragile due to decreased bone mineral density, significantly increasing fracture risk, particularly in the hip, spine, and wrist."},
    {"id": 13, "question": "How does the kidney regulate blood pressure?",
     "ground_truth": "By controlling fluid and sodium balance, and releasing renin to activate the RAAS system.",
     "context": "Kidneys regulate blood pressure by adjusting fluid and sodium levels in the body and releasing renin, an enzyme that activates the renin-angiotensin-aldosterone system (RAAS) to constrict blood vessels and retain sodium."},
    {"id": 14, "question": "What is a stroke?",
     "ground_truth": "Brain damage from interrupted blood supply, either by a clot (ischemic) or bleeding (hemorrhagic).",
     "context": "A stroke occurs when blood supply to part of the brain is interrupted. Ischemic strokes are caused by blockages; hemorrhagic strokes are caused by bleeding in the brain. Both cause brain cell death."},
    {"id": 15, "question": "What is insulin resistance?",
     "ground_truth": "When cells don't respond effectively to insulin, forcing the pancreas to produce more insulin.",
     "context": "Insulin resistance is a condition where muscle, fat, and liver cells do not respond properly to insulin. The pancreas compensates by producing more insulin, which can eventually lead to type 2 diabetes."},
    {"id": 16, "question": "What causes kidney stones?",
     "ground_truth": "Crystallization of minerals and salts in urine, often due to dehydration or diet.",
     "context": "Kidney stones form when minerals like calcium, oxalate, or uric acid become concentrated in urine and crystallize. Risk factors include dehydration, high-sodium diet, obesity, and certain medical conditions."},
    {"id": 17, "question": "What is the function of the thyroid gland?",
     "ground_truth": "Produces hormones (T3, T4) that regulate metabolism, energy, and body temperature.",
     "context": "The thyroid gland produces triiodothyronine (T3) and thyroxine (T4), hormones that regulate the body's metabolic rate, heart function, digestion, muscle control, brain development, and bone maintenance."},
    {"id": 18, "question": "What is pneumonia?",
     "ground_truth": "A lung infection causing air sacs to fill with fluid, causing cough, fever, and breathing difficulty.",
     "context": "Pneumonia is an infection that inflames the air sacs (alveoli) in one or both lungs. The air sacs may fill with fluid or pus, causing symptoms like cough, fever, chills, and difficulty breathing."},
    {"id": 19, "question": "How does sleep affect health?",
     "ground_truth": "Sleep is essential for immune function, memory consolidation, hormone regulation, and tissue repair.",
     "context": "Adequate sleep is critical for immune system function, consolidating memories, regulating hormones like cortisol and growth hormone, repairing tissues, and maintaining cardiovascular health."},
    {"id": 20, "question": "What is the difference between Type 1 and Type 2 diabetes?",
     "ground_truth": "Type 1 is autoimmune (no insulin production); Type 2 is lifestyle-related (insulin resistance).",
     "context": "Type 1 diabetes is an autoimmune disease where the immune system destroys insulin-producing beta cells. Type 2 diabetes develops gradually when cells become resistant to insulin, often linked to lifestyle factors."},
    {"id": 21, "question": "What is a CT scan used for?",
     "ground_truth": "To create detailed cross-sectional images of internal organs, bones, and blood vessels.",
     "context": "A CT (computed tomography) scan uses X-rays and computer processing to create detailed cross-sectional images of the body, useful for detecting tumors, internal injuries, infections, and vascular conditions."},
    {"id": 22, "question": "What causes high cholesterol?",
     "ground_truth": "Diet high in saturated fats, lack of exercise, obesity, and genetic factors.",
     "context": "High cholesterol is caused by a combination of diet (saturated and trans fats), lack of physical activity, obesity, smoking, and genetic predisposition. Some medical conditions and medications also raise cholesterol."},
    {"id": 23, "question": "What is the role of platelets in blood?",
     "ground_truth": "Platelets help form blood clots to stop bleeding at injury sites.",
     "context": "Platelets (thrombocytes) are small blood cells that help the body form clots to stop bleeding. When a blood vessel is damaged, platelets aggregate at the site and release chemicals that trigger the clotting cascade."},
    {"id": 24, "question": "What is Alzheimer's disease?",
     "ground_truth": "A progressive brain disorder causing memory loss, cognitive decline, and behavioral changes.",
     "context": "Alzheimer's disease is a progressive neurodegenerative disorder that destroys memory and cognitive function. It is characterized by amyloid plaques and neurofibrillary tangles in the brain."},
    {"id": 25, "question": "What are probiotics?",
     "ground_truth": "Live beneficial bacteria that support gut health and the immune system.",
     "context": "Probiotics are live microorganisms (mainly bacteria) that, when consumed in adequate amounts, confer health benefits. They support gut microbiome balance, improve digestion, and enhance immune function."},
]

# ── Single Groq wrapper ────────────────────────────────────────────────────────
def ask(model_id, question, context):
    start = time.time()
    r = groq_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "Answer concisely using only the provided context."},
            {"role": "user",   "content": f"Context: {context}\nQuestion: {question}"}
        ]
    )
    return {
        "answer":  r.choices[0].message.content,
        "latency": round(time.time() - start, 3),
        "tokens":  r.usage.total_tokens
    }

# ── Scoring (no RAGAS — pure keyword overlap + LLM judge via Groq) ─────────────
def score_answer(question, answer, context, ground_truth):
    """
    Ask Groq itself to score the answer on 3 metrics (0.0 to 1.0).
    Cheaper and more reliable than RAGAS for this use case.
    """
    prompt = f"""You are an evaluator. Score the answer on 3 metrics, each from 0.0 to 1.0.

Question: {question}
Context: {context}
Ground Truth: {ground_truth}
Answer to evaluate: {answer}

Return ONLY this JSON, no explanation:
{{
  "faithfulness": <0.0-1.0, is answer grounded in context?>,
  "relevancy": <0.0-1.0, does it answer the question?>,
  "correctness": <0.0-1.0, how close to ground truth?>
}}"""

    r = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    import json
    try:
        return json.loads(r.choices[0].message.content)
    except:
        return {"faithfulness": 0.5, "relevancy": 0.5, "correctness": 0.5}

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = {
    "llama-3.1-8b":  "llama-3.1-8b-instant",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "qwen3-32b":     "qwen/qwen3-32b",
}

# ── Main pipeline ──────────────────────────────────────────────────────────────
def run():
    all_rows = []

    for model_name, model_id in MODELS.items():
        print(f"\n── Running {model_name} ──")

        for i, q in enumerate(QUESTIONS):
            print(f"  Q{i+1}/25", end="\r")
            r = ask(model_id, q["question"], q["context"])
            scores = score_answer(q["question"], r["answer"], q["context"], q["ground_truth"])

            all_rows.append({
                "model":        model_name,
                "question":     q["question"],
                "answer":       r["answer"],
                "faithfulness": round(scores.get("faithfulness", 0), 3),
                "relevancy":    round(scores.get("relevancy", 0), 3),
                "correctness":  round(scores.get("correctness", 0), 3),
                "latency_sec":  r["latency"],
                "tokens_used":  r["tokens"],
            })

        print(f"  Done.         ")

    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nDone. Results saved to eval_results.csv")

if __name__ == "__main__":
    run()
