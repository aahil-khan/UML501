# 🚀 JEE Model Benchmark - Project Roadmap

## 📋 Project Overview
This document outlines the step-by-step process to complete the JEE Question Paper Generator Model Evaluation phase, from dataset preparation to final model selection.

---

## ✅ Phase 1: Dataset Preparation

### Step 1.1: Source Question Dataset
**Goal:** Collect 50-100 high-quality JEE questions across subjects

**Options:**
- [ ] **Manual Curation**: Collect from JEE previous year papers, reference books
- [ ] **Online Sources**: Scrape from educational websites (with proper permissions)
- [ ] **Existing Datasets**: Look for publicly available JEE question datasets on Kaggle, GitHub
- [ ] **Combination Approach**: Mix of all above for diversity

**Requirements per question:**
- Subject (Physics/Chemistry/Math)
- Topic (e.g., Kinematics, Thermodynamics, Calculus)
- Difficulty level (Easy/Medium/Hard)
- Question text
- 4 options (for MCQs)
- Correct answer
- Detailed solution/explanation

### Step 1.2: Structure the Dataset
**Format:** CSV or JSON

**Recommended structure:**
```
id, subject, topic, difficulty, question, option_a, option_b, option_c, option_d, correct_answer, solution
```

**Distribution guidelines:**
- Physics: ~35% (15-35 questions)
- Chemistry: ~35% (15-35 questions)
- Mathematics: ~30% (15-30 questions)
- Easy: 30%, Medium: 50%, Hard: 20%

**Action Items:**
- [ ] Create `jee_questions_dataset.csv` file
- [ ] Ensure balanced distribution across subjects
- [ ] Validate all questions have correct answers
- [ ] Review solutions for clarity

---

## ⚙️ Phase 2: Environment Setup

### Step 2.1: Install Required Dependencies
- [ ] Update the notebook cell with actual installation:
```python
!pip install openai transformers pandas matplotlib sympy python-dotenv torch
```

### Step 2.2: Configure API Keys
**For OpenAI (GPT models):**
- [ ] Sign up for OpenAI API at https://platform.openai.com
- [ ] Generate API key
- [ ] Create `.env` file in project root:
```
OPENAI_API_KEY=your_api_key_here
```
- [ ] Update notebook to load from `.env`:
```python
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
```

**For Hugging Face models:**
- [ ] Sign up at https://huggingface.co
- [ ] Generate access token (for gated models)
- [ ] Add to `.env`:
```
HF_TOKEN=your_token_here
```

### Step 2.3: Hardware Considerations
**For running local models (Mistral, LLaMA):**
- [ ] Check GPU availability (NVIDIA with CUDA support recommended)
- [ ] Alternative: Use Hugging Face Inference API (cloud-based)
- [x] Alternative: Use Ollama for local LLM serving

---

## 🤖 Phase 3: Model Integration

### Step 3.1: Expand Model List
**Current:** Only GPT-4o-mini

**Suggested additions:**
- [x] **GPT-4o** (full version for comparison)
- [ ] **GPT-3.5-turbo** (cost-effective baseline)
- [x] **Mistral-7B-Instruct** (open-source alternative)
- [x] **LLaMA-2-7B-chat** (Meta's model)
- [ ] **Gemini-Pro** (Google's API) - *if accessible*
- [ ] **Claude-3-Haiku** (Anthropic) - *if accessible*

### Step 3.2: Implement Additional Model Functions
Update the notebook with:

```python
def generate_with_gemini(prompt):
    # Implementation for Google Gemini
    pass

def generate_with_ollama(model_name, prompt):
    # For local LLM serving
    import requests
    response = requests.post('http://localhost:11434/api/generate', 
                            json={'model': model_name, 'prompt': prompt})
    return response.json()['response']
```

### Step 3.3: Handle Rate Limits & Costs
- [ ] Implement retry logic with exponential backoff
- [ ] Add cost tracking for API calls
- [ ] Consider batching requests
- [ ] Set up error handling for timeouts

---

## 🧪 Phase 4: Prompt Engineering

### Step 4.1: Refine Prompt Template
**Current prompt is good, but consider:**
- [ ] Add few-shot examples (1-2 sample Q&A pairs)
- [ ] Specify exact JSON format with schema
- [ ] Add constraints (e.g., "do not change the fundamental concept")
- [ ] Include difficulty guidelines

**Enhanced prompt example:**
```python
def build_prompt(question, solution, difficulty):
    return f'''
You are an expert JEE question generator.

Task: Generate ONE new JEE-style question testing the SAME concept as the example below.
- Change numbers, context, or scenario
- Maintain {difficulty} difficulty level
- Provide 4 distinct options with only ONE correct answer
- Include a clear step-by-step solution

Example Question: {question}
Example Solution: {solution}

Output ONLY valid JSON:
{{
  "question": "Your generated question here",
  "options": ["A", "B", "C", "D"],
  "correct_answer": "B",
  "explanation": "Step-by-step solution"
}}
'''
```

### Step 4.2: Test Prompt Variations
- [ ] Create 2-3 prompt variants
- [ ] Test on 5 sample questions
- [ ] Select best-performing prompt template

---

## 🔍 Phase 5: Generation & Collection

### Step 5.1: Load Full Dataset
- [ ] Replace hardcoded `data` list with CSV loading:
```python
df = pd.read_csv('jee_questions_dataset.csv')
```

### Step 5.2: Run Generation Pipeline
- [ ] Execute generation for all models on all questions
- [ ] Implement progress tracking (tqdm progress bar)
- [ ] Save intermediate results (in case of failures)
- [ ] Log any errors or failed generations

**Recommended addition:**
```python
from tqdm import tqdm
import json

results = []
failed_generations = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row['question'], row['solution'])
    for model_name, func in models.items():
        try:
            output = func(prompt)
            results.append({
                'id': idx,
                'model': model_name,
                'subject': row['subject'],
                'topic': row['topic'],
                'difficulty': row['difficulty'],
                'base_question': row['question'],
                'generated_output': output
            })
        except Exception as e:
            failed_generations.append({
                'id': idx, 'model': model_name, 'error': str(e)
            })
            print(f"❌ Failed: {model_name} on question {idx}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('generation_results.csv', index=False)

if failed_generations:
    pd.DataFrame(failed_generations).to_csv('failed_generations.csv', index=False)
```

### Step 5.3: Parse Generated Outputs
- [ ] Extract JSON from model responses
- [ ] Handle malformed outputs
- [ ] Create structured columns for question, options, answer, explanation

---

## ✅ Phase 6: Validation

### Step 6.1: Automated Validation
**Implement checks for:**
- [ ] **JSON validity**: Can the output be parsed?
- [ ] **Completeness**: All required fields present?
- [ ] **Option count**: Exactly 4 options?
- [ ] **Answer validity**: Correct answer is one of the options?
- [ ] **Numeric accuracy**: Use SymPy for mathematical expressions
- [ ] **Format consistency**: Similar structure to base question?

**Enhanced validation function:**
```python
def validate_generation(generated_output, base_question):
    scores = {
        'json_valid': 0,
        'complete': 0,
        'options_count': 0,
        'answer_valid': 0,
        'numeric_correct': 0
    }
    
    try:
        data = json.loads(generated_output)
        scores['json_valid'] = 1
        
        if all(k in data for k in ['question', 'options', 'correct_answer', 'explanation']):
            scores['complete'] = 1
        
        if len(data.get('options', [])) == 4:
            scores['options_count'] = 1
        
        if data.get('correct_answer') in data.get('options', []):
            scores['answer_valid'] = 1
        
        # Add numeric validation logic here
        
    except:
        pass
    
    return scores
```

### Step 6.2: Apply Validation
- [ ] Run validation on all generated outputs
- [ ] Add validation scores to results dataframe
- [ ] Filter out completely invalid outputs

---

## 👥 Phase 7: Human Evaluation

### Step 7.1: Prepare Evaluation Template
- [ ] Generate `evaluation_template.csv` from the notebook
- [ ] **Recommendation**: Sample ~20-30 questions per model (not all) for manual review
- [ ] Create evaluation guidelines document

**Evaluation criteria (1-5 scale):**
1. **Conceptual Accuracy**: Does it test the same concept?
2. **Clarity**: Is the question clear and unambiguous?
3. **Creativity**: Is it different enough from the original?
4. **Answer Validity**: Is the answer correct and well-explained?
5. **Formatting**: Is it well-structured and exam-ready?

### Step 7.2: Conduct Manual Evaluation
- [ ] Recruit 2-3 evaluators (subject experts preferred)
- [ ] Distribute evaluation template
- [ ] Collect scored responses
- [ ] Calculate inter-rater reliability (optional but recommended)

### Step 7.3: Aggregate Scores
- [ ] Average scores across evaluators
- [ ] Identify outliers or disputed questions
- [ ] Re-evaluate questionable items

---

## 📊 Phase 8: Analysis & Visualization

### Step 8.1: Statistical Analysis
- [ ] Calculate mean scores per model
- [ ] Calculate standard deviation
- [ ] Perform statistical significance tests (t-test, ANOVA)
- [ ] Create correlation matrix between criteria

**Enhanced analysis:**
```python
# Overall performance
avg_scores = scored.groupby('model').agg({
    'conceptual_accuracy': ['mean', 'std'],
    'clarity': ['mean', 'std'],
    'creativity': ['mean', 'std'],
    'answer_validity': ['mean', 'std'],
    'formatting': ['mean', 'std']
})

# By subject
subject_performance = scored.groupby(['model', 'subject'])[
    ['conceptual_accuracy', 'clarity', 'creativity', 'answer_validity', 'formatting']
].mean()

# By difficulty
difficulty_performance = scored.groupby(['model', 'difficulty'])[
    ['conceptual_accuracy', 'clarity', 'creativity', 'answer_validity', 'formatting']
].mean()
```

### Step 8.2: Create Visualizations
- [ ] Bar chart: Overall model comparison (already in notebook)
- [ ] Heatmap: Model performance by subject
- [ ] Box plot: Score distribution per model
- [ ] Radar chart: Multi-dimensional comparison
- [ ] Line plot: Performance by difficulty level

**Additional visualization code:**
```python
import seaborn as sns

# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(avg_scores, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Model Performance Heatmap')
plt.show()

# Radar chart
from math import pi

categories = ['Conceptual', 'Clarity', 'Creativity', 'Answer', 'Format']
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

for model in scored['model'].unique():
    values = scored[scored['model']==model][
        ['conceptual_accuracy', 'clarity', 'creativity', 'answer_validity', 'formatting']
    ].mean().values
    values = np.concatenate((values, [values[0]]))
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    ax.plot(angles, values, label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.legend()
plt.show()
```

### Step 8.3: Cost-Performance Analysis
- [ ] Track API costs per model
- [ ] Calculate cost per question
- [ ] Create cost-benefit chart (quality vs. price)

---

## 🏆 Phase 9: Decision & Documentation

### Step 9.1: Model Selection
- [ ] Rank models by overall score
- [ ] Consider secondary factors:
  - Cost per question
  - Generation speed
  - Consistency across subjects
  - Ease of deployment
- [ ] Make final decision with justification

### Step 9.2: Create Final Report
**Document should include:**
- [ ] Executive summary
- [ ] Dataset description
- [ ] Models evaluated
- [ ] Evaluation methodology
- [ ] Results and visualizations
- [ ] Final model selection with reasoning
- [ ] Limitations and future work

### Step 9.3: Prepare for Integration
- [ ] Document API configuration for chosen model
- [ ] Create reusable prompt template
- [ ] Write model wrapper function
- [ ] Plan RAG pipeline integration

---

## 🔄 Phase 10: Optional Enhancements

### Suggested Improvements
- [ ] **Fine-tuning**: Fine-tune open-source model on JEE dataset
- [ ] **Ensemble approach**: Combine outputs from multiple models
- [ ] **Feedback loop**: Incorporate human feedback to improve prompts
- [ ] **Automated scoring**: Train classifier to predict human scores
- [ ] **Topic modeling**: Ensure generated questions cover diverse sub-topics
- [ ] **Difficulty calibration**: Validate difficulty levels with IRT models

### Code Quality
- [ ] Refactor notebook into modular functions
- [ ] Add error handling throughout
- [ ] Create configuration file for settings
- [ ] Add unit tests for validation functions
- [ ] Document all functions with docstrings

---

## 📝 Suggested Changes to Current Notebook

### 1. **Add Configuration Cell** (at the top)
```python
# Configuration
CONFIG = {
    'dataset_path': 'jee_questions_dataset.csv',
    'results_path': 'generation_results.csv',
    'eval_path': 'evaluation_template.csv',
    'sample_size': 50,  # Number of questions to evaluate
    'models': ['gpt-4o-mini', 'gpt-3.5-turbo', 'mistral-7b'],
    'random_seed': 42
}
```

### 2. **Improve Data Loading**
```python
# Load dataset with validation
df = pd.read_csv(CONFIG['dataset_path'])
print(f"📊 Loaded {len(df)} questions")
print(f"Subjects: {df['subject'].value_counts().to_dict()}")
print(f"Difficulty: {df['difficulty'].value_counts().to_dict()}")

# Validate dataset
assert 'question' in df.columns, "Missing 'question' column"
assert 'solution' in df.columns, "Missing 'solution' column"
```

### 3. **Add Progress Tracking**
```python
from tqdm.notebook import tqdm
# Use tqdm in generation loops
```

### 4. **Implement Retry Logic**
```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

@retry_on_failure()
def generate_with_gpt(prompt):
    # existing code
```

### 5. **Add Cost Tracking**
```python
# Track tokens and costs
cost_tracker = {'gpt-4o-mini': {'tokens': 0, 'cost': 0}}

def track_cost(model, tokens_used):
    rates = {
        'gpt-4o-mini': 0.00015 / 1000,  # per token
        'gpt-4o': 0.03 / 1000,
    }
    cost_tracker[model]['tokens'] += tokens_used
    cost_tracker[model]['cost'] += tokens_used * rates.get(model, 0)
```

### 6. **Better Output Parsing**
```python
import json
import re

def extract_json(text):
    """Extract JSON from model output that may contain extra text"""
    # Try direct parsing
    try:
        return json.loads(text)
    except:
        pass
    
    # Try finding JSON block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    return None
```

### 7. **Add Checkpointing**
```python
import pickle

def save_checkpoint(data, filename='checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename='checkpoint.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Use in generation loop
checkpoint = load_checkpoint()
if checkpoint:
    results = checkpoint
    print("📂 Resumed from checkpoint")
```

---

## 📅 Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Dataset Preparation | 3-5 days | None |
| Environment Setup | 1 day | None |
| Model Integration | 2-3 days | Environment |
| Prompt Engineering | 1-2 days | Dataset |
| Generation & Collection | 1-2 days | All above |
| Validation | 1 day | Generation |
| Human Evaluation | 3-5 days | Validation |
| Analysis | 2 days | Evaluation |
| Documentation | 1-2 days | Analysis |
| **Total** | **15-23 days** | |

---

## 🎯 Success Criteria

- [ ] Dataset of 50-100 validated JEE questions created
- [ ] At least 3 different LLMs successfully tested
- [ ] Automated validation pipeline working
- [ ] Manual evaluation completed with clear scoring
- [ ] Comprehensive visualizations created
- [ ] Final model selected with data-driven justification
- [ ] Complete documentation and reproducible notebook
- [ ] Cost-performance analysis documented

---

## 📚 Resources & References

### Datasets
- JEE Previous Year Papers: https://jeemain.nta.nic.in/
- Kaggle JEE Datasets: Search "JEE questions"

### APIs & Documentation
- OpenAI: https://platform.openai.com/docs
- Hugging Face: https://huggingface.co/docs
- Ollama (local LLMs): https://ollama.ai/

### Tools
- SymPy Documentation: https://docs.sympy.org/
- Pandas User Guide: https://pandas.pydata.org/docs/
- Matplotlib Gallery: https://matplotlib.org/stable/gallery/

### Papers
- "Automatic Question Generation using LLMs" - Search on arXiv
- "Educational Question Generation: A Survey" - Search on Google Scholar

---

## 🤝 Next Steps After This Phase

1. **RAG Pipeline Development**: Integrate chosen model with vector database
2. **Question Bank Expansion**: Scale up dataset to 1000+ questions
3. **Paper Generation Logic**: Implement balancing algorithms
4. **Web Interface**: Build Flask/FastAPI backend and frontend
5. **Deployment**: Deploy to cloud platform (AWS/Azure/GCP)

---

**Good luck with your JEE Question Paper Generator project! 🚀**
