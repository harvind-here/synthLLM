# synthLLM (using Llama 3.3)
Using LLM for synthetic data generation for ML training.

This repository provides a tool to generate synthetic medical billing datasets using a Large Language Model (LLM) Llama 3.3. The generated data closely resembles real-world billing records, making it ideal for machine learning training, data analysis, or research purposes. The repository also includes tools to visualize and analyze the generated data.

## Skills:
<img src="https://img.shields.io/badge/LLMs-3776AB?style=flat-square&logo=ML&logoColor=white" alt="LLMs"> <img src="https://img.shields.io/badge/Synthetic Data Generation-3776AB?style=flat-square&logo=finetuning&logoColor=white" alt="Synthetic Data"> <img src="https://img.shields.io/badge/Data Analytics-3776AB?style=flat-square&logo=Data&logoColor=white" alt="Data Analytics">

## Features
Synthetic Data Generation: Mimics patterns and distributions of real medical billing data.
Schema Validation: Ensures each generated record conforms to a predefined schema using Pydantic.
Visualization Tools: Analyze the generated data with summary statistics and visualizations.
Exportable Outputs: Save the synthetic dataset as JSON or process it into CSV format.
Repository Structure
synthllm2.py: Script to generate the synthetic dataset.
analysis.py: Tool to visualize and analyze the synthetic dataset.
medical_billing_dataset.json: Sample real dataset used as input.
gen_dataset.json: Generated synthetic dataset.
requirements.txt: Contains the required dependencies.

## Prerequisites
-Python 3.8 or later
-API key for the Groq LLM (set as the environment variable GROQ_API_KEY).
-Install dependencies listed in requirements.txt.

Installation
Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```
Install dependencies:
```
pip install -r requirements.txt
```
Set your Groq API Key:
```
export GROQ_API_KEY='your-api-key-here'
```
Ensure the medical_billing_dataset.json file is present in the repository root.

Generate Synthetic Data
Run the data generation script:
```
python synthllm2.py
```
Output:

Synthetic dataset saved as gen_dataset.json.
Summary of the generated data displayed in the console.
Analyze Data
Run the analysis script:
```
python analysis.py
```
## Features:

Plots distributions of key features.
Highlights patterns in the generated data using statistical summaries.

