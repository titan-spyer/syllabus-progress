# 📚 Syllabus Progress Tracker

An AI-powered web application to help students effortlessly track their academic progress.

[![Syllabus Progress Tracker](https://drive.google.com/uc?id=1x2tmehhO60H29yIOaZiXgAj_IvmN2_X8)](https://syllabus-progress-with-gemini.streamlit.app/?embed_options=dark_theme)

---

## ✨ Features

* **Intelligent Syllabus Analysis**: Upload your syllabus PDF, and the app uses the Gemini AI API to automatically parse and break down the content into modules and topics.
* **Dynamic Progress Tracking**: Visually monitor your progress with intuitive dashboards and graphs. Mark topics as complete and see your overall progress update in real-time.
* **AI-Powered Study Assistant**: Get personalized insights and assistance powered by Gemini 1.5 Flash:
    * **Syllabus DNA**: A unique radar chart that analyzes the syllabus and shows its composition in terms of difficulty, numerics, diagrams, and theoretical concepts.
    * **Upcoming Topic Analysis**: The AI provides a brief, encouraging analysis of your next uncompleted topics.
    * **Practice Question Generation**: Generate relevant practice questions for any topic to test your understanding.
    * **AI Chat Tutor**: A conversational assistant that can answer questions based **only** on the content of your uploaded syllabus.
* **Personalized Profile**: Create and manage your user profile, academic details, and track your progress across multiple subjects and semesters.
* **Intuitive UI**: A clean, responsive interface with a toggle for dark and light themes for a comfortable user experience.

---

## 🚀 Live Demo

Experience the app live in both its dark and light themes.

* **Dark Theme**: [https://syllabus-progress-with-gemini.streamlit.app/?embed_options=dark_theme](https://syllabus-progress-with-gemini.streamlit.app/?embed_options=dark_theme)
* **Light Theme**: [https://syllabus-progress-with-gemini.streamlit.app/?embed_options=light_theme](https://syllabus-progress-with-gemini.streamlit.app/?embed_options=light_theme)

---

## 📸 Screenshots

| Profile Page with Syllabus Setup | Dashboard with Syllabus DNA |
| :---: |:---:|
| ![Profile Page](https://drive.google.com/uc?id=1G6mlSw7EDeRMmSFV_rWhglY--0GpaY-4) | ![Dashboard](https://drive.google.com/uc?id=1dPCnmZvxTL179dX50ZdAO96W9I2vflgj) |

| AI-Powered Recommendations | Topic Progress Tracking |
| :---: |:---:|
| ![AI Recommendations](https://drive.google.com/uc?id=1BKdyIXSCDr5miF2vOwrbdddwU-V9epD_) | ![Topics Progress](https://drive.google.com/uc?id=1x2tmehhO60H29yIOaZiXgAj_IvmN2_X8) |

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.9+
* A Google AI Studio API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/titan-spyer/syllabus-progress.git](https://github.com/titan-spyer/syllabus-progress.git)
    cd syllabus-progress
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root directory and add your Google AI Studio API key.
    ```env
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

### Running the App

Run the following command in your terminal to start the Streamlit application:
```bash
streamlit run streamlit_app.py
