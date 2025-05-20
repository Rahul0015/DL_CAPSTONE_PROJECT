# NESY Threat Project: Hybrid Neural-Symbolic Approach to Terrorism Prediction in India

## Project Overview

This project investigates a hybrid approach combining the strengths of neural networks and symbolic logic to predict terrorist events specifically within India, using the Global Terrorism Database (GTD). Our methodology leverages the ability of a Long Short-Term Memory (LSTM) network to learn complex patterns from textual summaries of past events and integrates it with a rule-based system derived from expert knowledge and logical reasoning. This hybrid approach aims to improve the accuracy and interpretability of terrorism prediction.

This project was a collaborative effort by a team of three members.

## Project Structure

The repository is organized as follows:
DEEPLCAPSTONE/
├── data/
│ ├── raw/ # Raw Global Terrorism Database (GTD) datasets
│ ├── interim/ # Cleaned and filtered GTD data, focusing on incidents in India
│ ├── processed/ # Vectorized textual summaries and corresponding labels, ready for model training
│ └── external/ # Directory for any additional external datasets used (currently optional)
├── notebooks/ # Jupyter notebooks for Exploratory Data Analysis (EDA) and development stages
│ ├── 1_eda.ipynb # Initial exploration of the raw GTD data
│ ├── 2_feature_engineering.ipynb # Steps for cleaning, filtering (India), and preparing data
│ ├── 3_model_dev.ipynb # Development and training of the LSTM-based classification model
│ ├── 4_symbolic_logic.ipynb # Design and implementation of the rule-based symbolic logic system
│ └── 5_hybrid_testing.ipynb # Experiments and evaluation of the combined neural-symbolic approach
├── models/ # Directory for saving trained models and necessary artifacts
│ ├── lstm_model.h5 # Trained LSTM model file (in HDF5 format)
│ ├── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer (serialized using pickle)
│ └── label_encoder.pkl # Trained Label Encoder (serialized using pickle)
├── src/ # Source code for the project's modules
│ ├── init.py # Initializes the src directory as a Python package
│ ├── preprocessing.py # Contains functions for loading the raw GTD data, cleaning it, and saving the India-specific subset
│ ├── ml_model.py # Implements the LSTM-based neural network classifier
│ ├── symbolic_rules.py # Defines the rule-based decision logic using symbolic reasoning
│ └── hybrid_predictor.py # Integrates the predictions from the LSTM model and the symbolic rules
├── tests/ # Optional directory for unit tests
│ └── test_pipeline.py # Contains unit tests for different parts of the pipeline (currently focusing on basic structure)
├── main.py # Orchestrates the entire data processing, model training, and hybrid prediction pipeline
├── requirements.txt # Lists the Python dependencies required to run the project
├── README.md # This file, providing an overview of the project
└── .gitignore # Specifies intentionally untracked files that Git should ignore

## Project Workflow and Key Contributions

Our project followed these key stages:

1.  **Data Acquisition and Exploration (Notebooks/1_eda.ipynb):**

    - The raw Global Terrorism Database (GTD) was acquired.
    - Initial Exploratory Data Analysis (EDA) was performed to understand the dataset's structure, identify key features, and assess data quality.

2.  **Data Preprocessing and Feature Engineering (src/preprocessing.py, notebooks/2_feature_engineering.ipynb):**

    - The GTD dataset was filtered to include only terrorist incidents that occurred in India.
    - Textual summaries of the terrorist events were identified as the primary input for the neural network.
    - Relevant features were selected and prepared for both the neural network and the symbolic logic components.
    - The processed India-specific data was saved in the `data/interim/` directory.

3.  **Neural Network Model Development (src/ml_model.py, notebooks/3_model_dev.ipynb):**

    - An LSTM-based neural network architecture was designed for classifying terrorist events based on their summaries.
    - Textual data was vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to convert it into a numerical format suitable for the LSTM.
    - Labels for the events were encoded.
    - The LSTM model was trained on the processed data.
    - The trained LSTM model, TF-IDF vectorizer, and label encoder were saved in the `models/` directory.

4.  **Symbolic Logic Implementation (src/symbolic_rules.py, notebooks/4_symbolic_logic.ipynb):**

    - A rule-based system was developed based on expert knowledge, logical reasoning, and potentially patterns observed during EDA.
    - These rules were designed to capture specific conditions or characteristics of terrorist events in India that might be indicative of certain outcomes or patterns, potentially complementing the neural network's learning.

5.  **Hybrid Prediction System (src/hybrid_predictor.py, notebooks/5_hybrid_testing.ipynb):**

    - A mechanism was implemented to combine the predictions from the LSTM model and the symbolic rule-based system. This could involve weighted averaging, applying rules based on the confidence of the neural network, or other integration strategies.
    - The performance of the hybrid system was evaluated and compared to the individual components.

6.  **Pipeline Orchestration (main.py):**

    - The `main.py` script orchestrates the entire workflow, calling the necessary functions from the `src/` directory to load data, preprocess it, train the model, apply symbolic rules, and evaluate the hybrid predictor.

7.  **Testing (tests/test_pipeline.py):**

    - Basic unit tests were implemented to ensure the fundamental components of the pipeline (like data loading and preprocessing) function as expected.

8.  **Documentation (README.md):**

    - This `README.md` file provides a comprehensive overview of the project, its structure, and the work undertaken.

9.  **Dependencies (requirements.txt):**
    - The `requirements.txt` file lists all the necessary Python libraries and their versions required to run the project, ensuring reproducibility.

## Team Contributions

This project was a collaborative effort by three team members. While specific task allocation details might have been managed internally, the overall success of this project relied on the combined skills and efforts of each member across the different stages of development, analysis, and implementation.

## Getting Started

To run this project, please follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/nesy_threat_project.git](https://github.com/YOUR_USERNAME/nesy_threat_project.git)
    cd nesy_threat_project
    ```

    (Replace `YOUR_USERNAME/nesy_threat_project` with the actual repository URL)

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main pipeline script:**
    ```bash
    python main.py
    ```
5. **Thier is one file missing from the raw because of the size exceed the limit here is the link for the same https://1drv.ms/x/c/eb74c1a68ca4f4ed/EVSM0IqoFD5DrDO3EvgdChMBwAZnWNzHYe66aBPMh-PvIA?e=mSXsi9
This will execute the entire pipeline, from data loading and preprocessing to model training and hybrid prediction. The output and evaluation results will be printed to the console.

## Further Work

Potential future directions for this project include:

- Exploring more sophisticated neural network architectures.
- Developing a more comprehensive and nuanced set of symbolic rules.
- Implementing more advanced techniques for integrating neural and symbolic predictions.
- Evaluating the model on a more recent and unseen portion of the GTD.
- Investigating the interpretability of the hybrid system to understand the reasons behind its predictions.
- Deploying the prediction system as a usable application.
