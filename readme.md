# CLIP Safety Alignment System

## 🛡️ Overview
This application provides a robust multi-modal safety filtering system that leverages advanced AI models to ensure safe and responsible content interaction across text and image inputs.

## ✨ Key Features
- Multi-modal safety scoring (text and image)
- CLIP text-image similarity analysis
- GPT-4 Vision chat interface
- Adaptive safety constraint mechanism
- Comprehensive toxicity detection

### Main Menu
- Displays options for CLIP Mode and GPT-4 Vision Chat Mode

### Image Selection
- Interactive image browsing
- Support for direct file path or directory browsing

### Safety Scoring
- Detailed toxicity scores for text and image
- Visual representation of safety constraints

### CLIP Analysis
- Text-image similarity scores
- Safety constraint evaluation

## 🧮 Mathematical Foundation: Toxicity Barrier Function

### Barrier Function Equation
The core of our safety mechanism is the Toxicity Barrier Function:

B(Ti, Tt) = Ti + Tt - (C - C * e^(-λ))

Where:
- B: Barrier Function
- Ti: Image Toxicity Score ∈ [0, 1]
- Tt: Text Toxicity Score ∈ [0, 1]
- C: Threshold Ceiling
- λ (Lambda): Adaptive Sensitivity Parameter

### Mapping to Control Barrier Functions (CBFs)

#### Traditional CBF Formulation
In continuous-time control systems, a CBF is typically expressed as:
- B(x) ≤ 0 defines the safe set
- dB/dt ≤ 0 ensures the system remains safe

#### Our Toxicity Barrier Function Adaptation

| Feature | Traditional CBF | Toxicity Barrier Function |
|---------|-----------------|---------------------------|
| **State Variable** | System state x | Toxicity scores (Ti, Tt) |
| **Safe Set Condition** | B(x) ≤ 0 | B(Ti, Tt) ≤ 0 |
| **Safety Constraint** | Maintains system within safe boundary | Filters toxic multimodal content |
| **Time Dependency** | dB/dt ≤ 0 (continuous) | Instantaneous evaluation |
| **Adaptability** | Control inputs u | Parameters C and λ |

### Safety Evaluation Criteria
- If B(Ti, Tt) ≤ 0: Content is SAFE
- If B(Ti, Tt) > 0: Content is UNSAFE (filtered)

## 🛠️ Prerequisites
- Python 3.9+
- Conda (recommended)
- API Credentials:
  * Sightengine API credentials
  * OpenAI API key

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://your-repository-url.git
cd project-directory
```

### 2. Create Conda Environment
```bash
conda env create -f env.yml
conda activate bar_safe
```

### 3. Configure Credentials
Edit the script and replace the following placeholders:
- `SIGHTENGINE_USER`: Your Sightengine API user ID
- `SIGHTENGINE_SECRET`: Your Sightengine API secret
- `OPENAI_API_KEY`: Your OpenAI API key

## 🔧 Dependencies
- NumPy
- PyTorch
- Transformers
- Detoxify
- Requests
- CLIP Model
- Sightengine API
- OpenAI GPT-4 Vision API

## 🎮 Usage

### Running the Application
```bash
python Bar_safe.py
```

### Modes

#### 1. CLIP Mode
- Analyze text-image similarity
- Perform safety scoring
- Get text-image matching score

#### 2. GPT-4 Vision Chat Mode
- Interactive chat with images
- Real-time safety filtering
- Conversation history management

### Navigation
- Use numeric keys to select modes
- Type 'browse' to select images
- Use 'clear' to reset conversation
- 'exit' or 'back' to return to main menu

## 🔒 Safety Mechanism Parameters

### Threshold Ceiling (C)
- Adjusts overall safety sensitivity
- Range: 0.0 to 1.0
- Higher C: More permissive filtering
- Lower C: More restrictive filtering

### Adaptive Sensitivity (λ)
- Controls threshold reduction dynamics
- Range: 0.0 to 2.0
- Higher λ: Rapid threshold convergence
- Lower λ: Gradual threshold adaptation

### Computational Example
```python
# Safety constraint calculation
safety_threshold = C - C * math.exp(-lambda_val)
B = ti + tt - safety_threshold
is_safe = B <= 0
```

## Theoretical Insights
1. Exponential decay of safety threshold
2. Dynamic, adaptive filtering mechanism
3. Multimodal toxicity integration
4. Probabilistic safety assessment

### Mathematical Properties
- Continuous function
- Bounded between 0 and 1
- Non-linear threshold adaptation
- Symmetric toxicity consideration


## ⚠️ Limitations
- Requires active API credentials
- Performance depends on API response times
- Safety scoring is probabilistic
