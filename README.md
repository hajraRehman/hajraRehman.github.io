  <img src="hajrahahah.jpg" width="150" style="border-radius: 50%; border: 3px solid #4A90E2; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
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

📄 [Download My CV (PDF)](https://github.com/hajraRehman/hajraRehman.github.io/blob/main/Rehman_HafizaHajrah_CV.pdf)

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

# Hafiza Hajrah Rehman

<div align="center">

[![Data Analyst | Data Scientist | AI Security & Interpretability Specialist | GenAI Enthusiast](https://img.shields.io/badge/Data%20Analyst%20%26%20AI%20Specialist-%23121011?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.linkedin.com/in/hafizahajrah-rehman/)  
**M.Sc. Data Science & AI @ Saarland University, Germany 🇩🇪 | B.Sc. Computer Science @ UCP, Pakistan 🇵🇰**

[![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:hafizahajra6@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hafizahajrah-rehman/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah)

</div>

---

## 👋 About Me

I'm Hafiza Hajrah Rehman, a passionate **Data Scientist and AI Researcher** with a focus on **trustworthy, interpretable, and secure AI systems**. Currently pursuing my Master's in Data Science and Artificial Intelligence at Saarland University, Germany, I specialize in adversarial machine learning, explainable AI (XAI), and parameter-efficient fine-tuning of large language models (LLMs).

I thrive at the intersection of **theory and practice**—blending mathematical rigor with hands-on projects to build models that not only perform but can be trusted in real-world applications like healthcare, IoT security, and autonomous driving. Beyond code, I'm a mentor, volunteer, and advocate for accessible AI education, drawing from my roots in Pakistan to foster inclusive tech communities.

> *"I believe in code that works, reports that explain, and models that can be trusted."*

Fluent in English, intermediate in German, and always eager to collaborate on ethical AI innovations. Open to opportunities in **AI Research, ML Engineering, and Trustworthy AI** roles in Europe or beyond (yes, even dreaming of NASA 🚀).

---

## 🎓 Education

### M.Sc. Data Science and Artificial Intelligence  
**Saarland University**, Saarbrücken, Germany  
*March 2024 – March 2026 (Expected)*  
- **Key Courses**: Machine Learning, Neural Networks: Theory and Implementation, Generative AI, Statistical NLP, Advances in AI for Autonomous Driving, German A1  

### B.Sc. Computer Science (Honors)  
**University of Central Punjab**, Lahore, Pakistan  
*December 2019 – July 2023*  
- **Final Year Project**: Deep Learning for Pedestrian Danger Estimation (3rd Place in University Competition)  
- **Relevant Coursework**: Artificial Intelligence, Data Analysis Techniques, Mathematics for Machine Learning, Introduction to Data Science  

---

## 💼 Professional Experience

### ML/Data Analyst  
**Zeeoutsourcing UK** (Remote)  
*November 2022 – February 2024*  
- Contributed to AI research projects on IoT security and energy-efficient intrusion detection.  
- Developed **SFlexCrypt**, a ML framework for detecting Sinkhole attacks in wireless sensor networks using Scikit-learn and Contiki-Cooja datasets—improving detection accuracy and energy efficiency.  
- Performed data preprocessing, model training, and evaluation; collaborated on publications for smart city applications.  

### AI Engineering Intern  
**Zeeoutsourcing UK** (Remote)  
*August 2022 – October 2022*  
- Assisted in ML model development for IoT applications, focusing on performance tuning and evaluation.  
- Conducted literature reviews, data preparation, and report visualization for energy-efficient IoT systems.  

---

## 🚀 Featured Projects

Here are some of my standout projects showcasing expertise in AI security, interpretability, and generative models. All built with open-source tools—check the repos for code, demos, and reports!

| Project | Description | Tech Stack | Links |
|---------|-------------|------------|-------|
| **🔐 Membership Inference Attack (MIA) on ResNet18**<br/>*Evaluating Privacy Leakage in Pretrained Models* | Implemented shadow model training and attack classifier in PyTorch to detect training set membership. Achieved >80% success rate on CIFAR-10 subset, highlighting privacy risks. | PyTorch, CIFAR-10, Adversarial ML | [![GitHub](https://img.shields.io/badge/View_Code-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah/mia-resnet18) |
| **🕵️ Model Stealing via Mock API**<br/>*Reverse-Engineering a Protected Encoder* | Black-box extraction pipeline using query synthesis and fine-tuning to replicate target model behavior with <5% accuracy loss—exposing API vulnerabilities. | PyTorch, Transfer Learning, API Security | [![GitHub](https://img.shields.io/badge/View_Code-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah/model-stealing-api) |
| **🛡️ Robust Adversarial Training for CIFAR-10**<br/>*Building Models That Survive Attacks* | Applied FGSM and PGD for hardening ResNet18: Boosted robust accuracy from 0% to 48% under PGD while keeping 78% clean accuracy. | PyTorch, FGSM, PGD, Model Defense | [![GitHub](https://img.shields.io/badge/View_Code-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah/robust-adversarial-cifar) |
| **🧾 Interpretability of ResNet using Grad-CAM & Network Dissection**<br/>*What Does Your Model Actually “See”?* | Visualized activations in ResNet18/50 on ImageNet/Places365; used Grad-CAM and LIME for transparent decision-making in safety-critical apps. | XAI, Grad-CAM, LIME, OpenCV | [![View Visuals](https://img.shields.io/badge/View_Visuals-blue?style=for-the-badge&logo=figma&logoColor=white)](https://hafizahajrah.notion.site/ResNet-Interpretability-Demo) [![GitHub](https://img.shields.io/badge/Code-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah/resnet-interpretability) |
| **🧪 Fine-tuning Chemical LLMs with LoRA & Influence Sampling**<br/>*Parameter-Efficient Learning for Scientific Domains* | Compared LoRA, BitFit, iA3 on chemical datasets via Hugging Face; LoRA proved most stable for influence-based sampling. | Hugging Face, LoRA, PyTorch | [![GitHub](https://img.shields.io/badge/View_Code-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah/chemical-llm-finetuning) |
| **🩻 Label-Efficient Tumor Detection using SimCLR + Grad-CAM**<br/>*Self-Supervised Learning for Medical Anomaly Detection* | Trained SimCLR on unlabeled X-rays, fine-tuned for tumors: AUC 0.65 (vs. 0.58 baseline), reducing annotations by ~70%. Grad-CAM validated alignments. | SimCLR, PyTorch, Grad-CAM | [![View Report](https://img.shields.io/badge/View_Report-green?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/tumor-detection-report) |
| **🏥 Medical Expense Prediction with XGBoost & SMOTE**<br/>*Healthcare Analytics with Real-World Impact* | End-to-end pipeline for cost prediction: R²=0.89, F1=0.92; tackled imbalance with SMOTE. | XGBoost, Scikit-learn, Pandas | [![GitHub](https://img.shields.io/badge/Code%20%2B_Dataset-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah/medical-expense-prediction) |

---

## 📁 Archived & Conceptual Projects

- **🌐 SFlexCrypt — Sinkhole Attack Detection in IoT Networks**  
  ML framework for energy-efficient detection in wireless sensor networks (Contiki-Cooja sim).  
  *Skills*: Python, Scikit-learn, IoT Security  

- **🏙️ Low-Power IoT Challenges in Smart Cities**  
  Research on energy-optimized data transmission for urban IoT.  
  *Skills*: IoT Analytics, Power Optimization  

- **🤖 Intellivision — AI Content Creation & Video Lecture Generator** *(FICS NUST 2023 Finalist)*  
  Web app using GPT-3 + Synthesia for auto-generating lectures.  
  *Skills*: GPT-3, NLP, Flask, EdTech AI  

- **💬 Conversational AI Help-Desk Bot**  
  University chatbot with intent recognition, reducing manual queries by 60%.  
  *Skills*: Python, Dialogflow, NLP  

- **🧠 Explainable AI for Medical Diagnosis**  
  Integrated LIME, SHAP, and counterfactuals for interpretable healthcare models.  
  *Skills*: XAI, SHAP, Ethical AI  

- **🧮 Research Collaboration — NUCES-FAST Lahore**  
  Worked on Theory of Automata, Mathematical Computing, and ML integration.  
  *Skills*: Theoretical CS, Automata  

---

## 🏆 Achievements & Certifications

### Achievements
- 🥉 **3rd Place** — UCP Final Year Project Competition (2023) for Pedestrian Detection with Danger Estimation  
- 🧩 **Selected Project** — Represented UCP at FICS NUST 2023 with *Intellivision*  
- 📊 **Research Contribution** — SFlexCrypt integrated into IoT security research at Zeeoutsourcing UK (2023)  
- 🔬 **Research Collaboration** — With Prof. Liaqat Majeed on Automata & ML (NUCES-FAST Lahore, 2022)  
- 💡 **Samsung Innovation Campus AI Cohort-II Graduate** (2022)  

### Certifications
- ☁️ **AWS Academy Machine Learning Foundations** — AWS Graduate  
- ☁️ **AWS Academy Data Engineering** — AWS Graduate  
- ☁️ **AWS Academy Natural Language Processing** — AWS Graduate  
- 🎓 **Artificial Intelligence Course (SIC Cohort-II)** — Samsung Innovation Campus  

---

## 🤝 Leadership & Volunteering

- **President — Pakistan Student Association Saarland**  
  Led academic, cultural, and community initiatives for Pakistani students at Saarland University.  

- **ML Instructor — LIVEX-UMT** (Volunteer)  
  Taught foundational ML to undergraduates; mentored capstone projects.  

- **Management Head — TAAKRA’23**  
  Coordinated Pakistan’s largest multi-category inter-university competition.  

- **Volunteer — Al-Khidmat Foundation**  
  Participated in plantation and ration drives for community welfare (2023).  

- **Director Linguistics & Writer’s Group — UCP Societies**  
  Mentored creative/technical writing teams; organized workshops and debates.  

---

## 🛠️ Technical Skills

### Programming Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=postgresql&logoColor=white) ![MATLAB](https://img.shields.io/badge/MATLAB-FF0000?style=flat&logo=matlab&logoColor=white) ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) ![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)

### Frameworks & Libraries
- **ML/DL**: PyTorch, Scikit-learn, XGBoost, TensorFlow, LoRA, iA3  
- **NLP/GenAI**: Hugging Face, Transformers, Prompt Engineering  
- **Data Viz**: Matplotlib, Seaborn, Tableau  
- **Web/Other**: React.js, Django, HTML/CSS  

### Techniques & Tools
- **ML Techniques**: Supervised/Unsupervised Learning, SMOTE, Grad-CAM, LIME, Contrastive Learning (SimCLR)  
- **Data Analysis**: Pandas, NumPy, EDA, Feature Engineering  
- **Cloud/DevOps**: AWS (SageMaker, S3, EC2), Git, Docker, Jupyter  
- **Domains**: AI Security, Trustworthy AI, Computer Vision, NLP, Generative AI, Healthcare, Autonomous Driving  

---

## 🌍 Languages
- **English** (Fluent) 🇬🇧  
- **German** (Intermediate) 🇩🇪  
- **Urdu** (Native) 🇵🇰  
- **Punjabi** (Native) 🇵🇰  
- **Arabic** (Basic) 🇸🇦  

---

## 📬 Let's Connect!

I'm always up for chatting about AI , collaborative projects, or just sharing recipes from Lahore. Reach out!

<div align="center">
  
[![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:hafizahajra6@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hafizahajrah-rehman/)  
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafizahajrah)

</div>

---

*Last Updated: October 2025*  
<div align="center">  
  <img src="https://img.shields.io/badge/Built%20with-%E2%9D%A4%EF%B8%8F%20and%20Markdown-1f425f.svg" alt="Built with Love">  
</div>
