  <img src="hajrahahah.jpg" alt="Hafiza Hajrah Rehman" width="150" style="border-radius: 50%; border: 3px solid #4A90E2; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  <h1> Hafiza Hajrah Rehman</h1>
  <p><em>Data Analyst |Data Scientist | AI Security & Interpretability Specialist | GenAI Enthusiast</em></p>
  <p>Master’s Candidate in Data Science & AI @ Saarland University, Germany 🇩🇪</p>
  <p> B.Sc. Computer Science @ University of Central Punjab, Pakistan 🇵🇰 </p>
  
  <br>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/hajrahrehman/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/hajraRehman)
[![Email](https://img.shields.io/badge/Email-Contact%20Me-red?logo=gmail)](mailto:hafizahajra6@gmail.com)

---

## 🎓 Education

### **M.Sc. Data Science and Artificial Intelligence**  
*Saarland University, Germany*   
- **Key Courses**:
  - Machine Learning 
  - Neural Networks: Theory and Implementation  
  - Generative AI 
  - Statistical NLP     
  - Advances in AI for Autonomous Driving 
  - German as a Foreign Language A1 

### **B.Sc. Computer Science (Honors)**  
*University of Central Punjab, Pakistan* | *Graduated 2023*   
- **Relevant Coursework**:  
  - Artificial Intelligence 
  - Data Analysis Techniques 
  - Mathematics for Machine Learning 
  - Introduction to Data Science   
  - Final Year Project: Deep Learning for Pedestrian Danger Estimation  

---

## 🧠 Featured Projects

### 🔐 Membership Inference Attack (MIA) on ResNet18  
*Evaluating Privacy Leakage in Pretrained Models*  
- Implemented shadow model training and attack classifier in **PyTorch** to determine if a sample was in the training set.  
- Achieved **>80% attack success rate** on CIFAR-10 subset — exposing critical model privacy risks.  
- *Skills: Adversarial ML, Model Security, PyTorch, Privacy Evaluation*  
🔗 [View Code](https://github.com/hajraRehman/Membership-Inference-Attack-on-Resnet18)

---

### 🕵️ Model Stealing via Mock API  
*Reverse-Engineering a Protected Encoder*  
- Designed black-box attack to extract model behavior via query synthesis and fine-tuning.  
- Successfully replicated model with **<5% accuracy drop** — demonstrating API vulnerability in production systems.  
- *Skills: Model Extraction, Transfer Learning, API Security, PyTorch*  
🔗 [View Code](https://github.com/hajraRehman/Model-Stealing-via-Mock-API)

---
### 🛡️ Robust Adversarial Training for CIFAR-10
*Building Models That Survive Attacks*
- Implemented adversarial training using Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) to harden ResNet18 against evasion attacks.
- Trained model on adversarially perturbed CIFAR-10 images — improved robust accuracy from 0% → 48% under PGD attack, while maintaining 78% clean accuracy.
- Evaluated trade-off between robustness and generalization — key insight for deploying models in safety-critical environments.
- *Skills: Adversarial ML, Robust Training, PyTorch, FGSM, PGD, ResNet, Model Defense*
🔗 [View Code](https://github.com/hajraRehman/Robust-Adversarial-Training-for-CIFAR-10)
---

### 🧾 Interpretability of ResNet using Grad-CAM & Network Dissection  
*What Does Your Model Actually “See”?*  
- Mapped neuron activations in **ResNet18/50** (trained on ImageNet/Places365) to human concepts (e.g., “flamingo”, “terrier”) using Broden dataset.  
- Applied **Grad-CAM** and **LIME** to visualize decision-making on 10 ImageNet classes — enhancing model transparency for safety-critical domains.  
- *Skills: Explainable AI (XAI), Computer Vision, Model Interpretability, Visualization*  
🔗 [View Visuals](#) | [Code](https://github.com/hajraRehman/Explainability-Analysis-of-ResNet-Models-using-Network-Dissection-Grad-CAM-LIME)

---

### 🧪 Fine-tuning Chemical LLMs with LoRA & Influence Sampling  
*Parameter-Efficient Learning for Scientific Domains*  
- Compared **LoRA, BitFit, iA3** for fine-tuning chemical language models using Hugging Face.  
- Evaluated data selection via **influence scores + diversity sampling** — found LoRA most stable; influence scores require caution.  
- *Skills: LLM Fine-tuning, LoRA, Hugging Face, Chemical AI, Efficient Learning*  
🔗 [GitHub Repo](https://github.com/hajraRehman/Fine-Tuning-Chemical-Language-Models-with-LoRA-Influence-Sampling)

---

### 🏥 Medical Expense Prediction with XGBoost & SMOTE  
*Healthcare Analytics with Real-World Impact*  
- Built end-to-end ML pipeline: preprocessing, feature selection, modeling (XGBoost, SVM), evaluation.  
- **XGBoost (R²=0.89)** for regression, **SVM + SMOTE (F1=0.92)** for classification — handled class imbalance effectively.  
- *Skills: Data Analysis, Feature Engineering, Scikit-learn, XGBoost, SMOTE, Healthcare Analytics*  
🔗 [Code + Dataset](https://github.com/hajraRehman/Predicting-Healthcare-Utilization-and-Expenditure-Using-Machine-Learning)

---
### 🩻 Label-Efficient Tumor Detection in Chest X-rays using SimCLR + Grad-CAM  
*Self-Supervised Learning for Medical Anomaly Detection*  
- Trained **SimCLR** on 5,606 unlabeled chest X-rays to learn robust visual features without annotations.  
- Fine-tuned on only **9.9% labeled tumor cases** using **focal loss (γ=2) + weighted sampling** to handle 10:1 class imbalance.  
- Achieved **AUC 0.65** (vs 0.58 baseline) — proved SSL reduces annotation cost by ~70%.  
- Used **Grad-CAM** for interpretability — attention maps aligned with radiologist-marked tumor regions.  
- Architecture: **ResNet18 + SSL projection head + progressive unfreezing** — preserved SSL features while adapting to tumors.  
🔗 [View Code & Report](https://github.com/hajraRehman/Self-Supervised-Anomaly-Detection-in-Medical-Imaging) 

---

## 🛠️ Technical Skills

### 💻 Languages & Tools
- **Programming**: Python, SQL, MATLAB
- **ML/DL**: PyTorch, Scikit-learn, XGBoost, TensorFlow, Hugging Face, OpenCV
- **Data Analysis**: Pandas, NumPy, EDA, Statistical Analysis, Feature Engineering, Matplotlib, Seaborn, Tableau, RapidMiner
- **Explainability & Security**: Grad-CAM, LIME, Network Dissection, Membership Inference, Model Stealing
- **Cloud & DevOps**: AWS (SageMaker, S3, EC2), Git, Docker, Jupyter, Google Colab
- **Databases**: MySQL, SQL Queries (Joins, Aggregations)

### 🧩 Methods & Frameworks
- **ML**: Regression, Classification, Clustering, SMOTE, Hyperparameter Tuning, Cross-Validation
- **DL**: CNN (ResNet), Transformers, Fine-tuning, LoRA, BitFit, iA3
- **NLP**: Masked LM, Tokenization, Embeddings, Prompt Engineering
- **AI Security**: Adversarial Training, Privacy Attacks, Robustness Evaluation

### 🌐 Domains
Computer Vision | NLP | Generative AI | Healthcare Analytics | Autonomous Driving | AI Security | Trustworthy AI | Chemical Informatics

---

## 📊 Data Analysis & Science Strengths

✅ **End-to-End Pipelines**: From data cleaning → EDA → modeling → evaluation → visualization  
✅ **Real-World Datasets**: Healthcare, ImageNet, CIFAR-10, Chemical compounds, Code repositories  
✅ **Tools Mastery**: RapidMiner (for workflow design), Tableau (dashboards), Pandas (data wrangling)  
✅ **Statistical Rigor**: Hypothesis testing, correlation analysis, outlier detection, distribution analysis  
✅ **Visualization**: Heatmaps (Grad-CAM), activation maps, ROC curves, bar plots, scatter plots — all used to communicate insights

---

## 📄 Resume & Certifications

📄  📄 [Download My CV (PDF)](https://github.com/hajraRehman/hajraRehman.github.io/raw/main/Rehman_HafizaHajrah_CV.pdf)

### Certifications
- AWS Machine Learning Foundations  
- AWS Data Engineering  
- AWS Natural Language Processing

---

## 🌍 About Me


I’m Hafiza Hajrah Rehman, a passionate and technically rigorous Data Scientist & AI Researcher currently pursuing my M.Sc. in Data Science and Artificial Intelligence at Saarland University, Germany, with a solid academic foundation from my B.Sc. in Computer Science from the University of Central Punjab, Pakistan. I specialize in building trustworthy, interpretable, and secure AI systems, with hands-on expertise spanning Adversarial Machine Learning, Model Interpretability (XAI), Parameter-Efficient LLM Fine-tuning, and End-to-End Data Science Pipelines.

My strength lies in transforming raw, complex data into actionable insights. As a skilled Data Analyst, I excel in Exploratory Data Analysis (EDA), Statistical Analysis, Feature Engineering, and Visualization using Python (Pandas, NumPy, Matplotlib, Seaborn), RapidMiner, and Tableau. I’ve successfully applied these skills in real-world scenarios — from predicting healthcare utilization and medical expenses using 108-dimensional demographic data (XGBoost, SVM + SMOTE, R²=0.32, Accuracy=0.85) to analyzing chemical compounds for lipophilicity prediction using influence-based data selection and LoRA fine-tuning.

I don’t just build models — I stress-test them. I’ve implemented Membership Inference Attacks (MIA) to expose model privacy leaks, executed Model Stealing via API to demonstrate deployment vulnerabilities, and hardened models using Robust Adversarial Training (CIFAR-10). To ensure transparency, I’ve dissected what models “see” using Network Dissection, Grad-CAM, ScoreCAM, AblationCAM, and LIME on ResNet architectures, quantifying their agreement with IoU metrics — a rare blend of security and explainability.

Proficient in Python, PyTorch, Scikit-learn, Hugging Face, AWS, SQL, and MATLAB, I’m comfortable across the stack — from data cleaning and cloud deployment to model interpretation and security auditing. My projects aren’t just academic — they’re practical, documented, and deployed on GitHub, ready for industry or research use.

I speak fluent English and am actively learning German (A1), embracing the challenge of building a global career in AI. Whether it’s making models more robust, more interpretable, or more efficient — I believe in code that works, reports that explain, and models that can be trusted.

Let’s connect and build the next generation of responsible AI! together!

---

## 📬 Let’s Connect

I’m actively seeking opportunities in **ML Engineering, AI Research, or AI Security**. If you’re working on trustworthy AI, adversarial ML, or applied GenAI — let’s talk!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/hajrahrehman/)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/hajraRehman)  
✉️ hafizahajra6@gmail.com

---

_Last updated: September 2025_



<div align="center">
  <img src="https://komarev.com/ghpvc/?username=hajraRehman&label=Profile%20Views&color=0e75b6&style=flat" alt="Profile Views" />
</div>
