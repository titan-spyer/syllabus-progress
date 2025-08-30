import os
import json
import urllib.parse

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import numpy as np

load_dotenv()
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate

# --- Theme Configuration ---

DARK_MODE_CSS = """
<style>
:root {
    --primary-color: #19A7CE;
    --background-color: #0E1117;
    --secondary-background-color: #161A21;
    --text-color: #FAFAFA;
    --font: "Source Sans Pro", sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    --background-color: #161A21;
}

/* Containers with border */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--secondary-background-color);
    border-color: #3c3f44;
    border-radius: 0.5rem;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border: 1px solid #3c3f44;
    border-radius: 0.5rem;
    padding: 1rem;
}

/* Expanders */
[data-testid="stExpander"] summary {
    background-color: var(--secondary-background-color);
    border-radius: 0.5rem;
}
</style>
"""

LIGHT_MODE_CSS = """
<style>
:root {
    --primary-color: #19A7CE;
    --background-color: #FFFFFF;
    --secondary-background-color: #F0F2F6;
    --text-color: #31333F;
    --font: "Source Sans Pro", sans-serif;
}

/* Containers with border */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e6eaf1;
    border-radius: 0.5rem;
}

/* Metrics */
[data-testid="stMetric"] {
    border: 1px solid #e6eaf1;
    border-radius: 0.5rem;
    padding: 1rem;
}
</style>
"""

def profile_page():
    """
    Displays the user profile page where users can create, view, and edit their information.
    """
    st.header("👤 User Profile")

    # Initialize session state for user profile if it doesn't exist
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {"is_complete": False}

    # If profile is not complete, force user to create it
    if not st.session_state.user_profile.get("is_complete"):
        st.info("Welcome! Please create your profile to get started.")
        with st.form("create_profile_form"):
            st.subheader("Create Your Profile")
            name = st.text_input("Full Name *")
            email = st.text_input("Email Address *")
            university_name = st.text_input("University Name *")
            university_roll_no = st.text_input("University Roll Number")
            profile_pic = st.file_uploader(
                "Upload a profile picture", type=["png", "jpg", "jpeg"]
            )

            submitted = st.form_submit_button("Create Profile")

            if submitted:
                if not name or not email or not university_name:
                    st.error("Please fill in all required fields (*).")
                else:
                    st.session_state.user_profile = {
                        "name": name,
                        "email": email,
                        "university_name": university_name,
                        "university_roll_no": university_roll_no,
                        "profile_pic": profile_pic.getvalue() if profile_pic else None,
                        "is_complete": True,
                    }
                    st.success("Profile created successfully!")
                    st.rerun()
        return  # Stop execution here until profile is created

    # --- Display the Profile Dashboard ---
    profile = st.session_state.user_profile

    with st.container(border=True):
        col1, col2 = st.columns([1, 3])

        with col1:
            if profile["profile_pic"]:
                st.image(profile["profile_pic"], use_container_width="auto")
            else:
                # A simple placeholder avatar
                st.markdown(
                    f"""<div style="
                        width: 120px;
                        height: 120px;
                        background-color: #ddd;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 48px;
                        font-weight: bold;
                        color: #fff;
                        background-image: linear-gradient(45deg, #6a11cb, #2575fc);
                        ">
                        {profile['name'][0].upper()}
                    </div>""",
                    unsafe_allow_html=True,
                )

        with col2:
            st.subheader(profile["name"])
            st.write(f"**🎓 University:** {profile['university_name']}")
            st.write(
                f"**🆔 Roll Number:** {profile.get('university_roll_no', 'N/A')}"
            )
            st.write(f"**✉️ Email:** {profile['email']}")

    # --- Syllabus Summary ---
    if "upload_data" in st.session_state and "subjects" in st.session_state.upload_data:
        st.subheader("Academic Overview")
        total_subjects = len(st.session_state.upload_data["subjects"])
        total_credits = sum(
            s["credits"] for s in st.session_state.upload_data["subjects"]
        )
        semester = st.session_state.upload_data.get("semester", "N/A")

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Semester", semester)
        col2.metric("Subjects Tracking", f"{total_subjects}")
        col3.metric("Total Credits", f"{total_credits}")

    st.divider()

    # --- Edit Profile Section ---
    with st.expander("✏️ Edit Your Profile"):
        with st.form("profile_form"):
            st.subheader("Update Your Information")

            # Inputs for editing
            new_name = st.text_input("Name", value=profile["name"])
            new_email = st.text_input("Email", value=profile["email"])
            new_university_name = st.text_input(
                "University Name", value=profile["university_name"]
            )
            new_university_roll_no = st.text_input(
                "University Roll Number", value=profile.get("university_roll_no", "")
            )
            new_profile_pic = st.file_uploader(
                "Upload a new profile picture", type=["png", "jpg", "jpeg"]
            )

            submitted = st.form_submit_button("Save Changes")

            if submitted:
                profile["name"] = new_name
                profile["email"] = new_email
                profile["university_name"] = new_university_name
                profile["university_roll_no"] = new_university_roll_no
                if new_profile_pic is not None:
                    profile["profile_pic"] = new_profile_pic.getvalue()
                st.success("Profile updated successfully!")
                st.rerun()


def upload_syllabus_page():
    """
    Displays the page for uploading and processing syllabus PDF files.
    """
    st.title("📄 Upload Syllabus")

    # Check if the initial syllabus info has been submitted
    if "upload_data" not in st.session_state:
        st.header("Step 1: Enter Syllabus Details")
        # Use a form to group inputs and have a single submit button
        with st.form("syllabus_details_form"):
            semester = st.selectbox(
                "Select Semester", options=[f"Semester {i}" for i in range(1, 9)]
            )
            num_subjects = st.number_input(
                "Enter Number of Subjects", min_value=1, max_value=6, value=1
            )

            subjects = []
            for i in range(num_subjects):
                st.markdown("---")
                name = st.text_input(f"Subject {i + 1} Name", key=f"subject_name_{i}")
                credits = st.number_input(
                    f"Credits for Subject {i + 1}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"credit_{i}",
                )
                subjects.append({"name": name, "credits": credits, "pdf_content": None})

            submitted = st.form_submit_button("Next: Upload PDFs")

            if submitted:
                # A simple validation to ensure names are entered
                if any(not s["name"].strip() for s in subjects):
                    st.error("Please provide a name for every subject.")
                else:
                    st.session_state.upload_data = {
                        "semester": semester,
                        "subjects": subjects,
                    }
                    st.rerun()  # Rerun to show the next step
    else:
        # This block runs after the user has submitted the details
        st.header("Step 2: Upload PDF for each subject")

        upload_data = st.session_state.upload_data

        # Defensive check for old data structure from previous runs
        if "subjects" not in upload_data:
            st.warning("It looks like your saved data is from an older version. Clearing it now.")
            del st.session_state.upload_data
            st.rerun()
            
        st.info(
            f"Now, please upload the syllabus PDF for each of the {len(upload_data['subjects'])} subjects in {upload_data['semester']}."
        )

        # Create a list to hold the uploaded file objects
        uploaded_files = []
        for i, subject in enumerate(upload_data["subjects"]):
            st.subheader(f"Upload for: {subject['name']} ({subject['credits']} credits)")
            uploaded_file = st.file_uploader(
                f"Syllabus for {subject['name']}", type="pdf", key=f"uploader_{i}"
            )
            uploaded_files.append(uploaded_file)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Go Back and Edit Details"):
                # Clear the uploaded data to return to step 1
                del st.session_state.upload_data
                # Also clear any processed flag if it exists
                if "syllabus_processed" in st.session_state:
                    del st.session_state.syllabus_processed
                # Clear progress data as well
                if "progress" in st.session_state:
                    del st.session_state.progress
                st.rerun()

        with col2:
            if st.button("✅ Finish & Process Syllabus"):
                # Check if a file has been uploaded for every subject
                if any(file is None for file in uploaded_files):
                    st.error("Please upload a PDF for every subject before proceeding.")
                else:
                    with st.spinner("Processing PDFs... This may take a moment."):
                        for i, file in enumerate(uploaded_files):
                            # Move file pointer to the beginning before reading
                            file.seek(0)
                            pdf_reader = PdfReader(file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text() or ""
                            # Store the extracted text in the session state
                            st.session_state.upload_data["subjects"][i][
                                "pdf_content"
                            ] = text

                    # Set a flag to indicate processing is complete
                    st.session_state.syllabus_processed = True
                    st.success("All syllabuses have been processed successfully!")
                    # Set the next page to navigate to and rerun.
                    st.session_state.page_to_navigate = "Track Progress"
                    st.rerun()


def get_modules_from_syllabus(syllabus_text: str) -> list[str]:
    """
    Uses Google's Generative AI to extract module names from syllabus text.

    Args:
        syllabus_text: The raw text extracted from the syllabus PDF.

    Returns:
        A list of strings, where each string is a module name.
    """
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Based on the following syllabus text, please identify and list the main modules or units.
        Each module/unit should be a separate item in the list.
        Return ONLY the list of module/unit names, separated by newlines. Do not add any other text, titles, or introductions.

        Example output:
        Module 1: Introduction to Programming
        Module 2: Data Structures
        Module 3: Algorithms
        Module 4: Object-Oriented Programming
        Module 5: Databases

        Syllabus Text:
        ---
        {syllabus_text[:8000]}
        ---
        """
        response = model.generate_content(prompt)
        if not hasattr(response, "text"):
            st.warning(
                "The AI model did not return any text. The syllabus might be empty or in an unreadable format."
            )
            return []
        modules = [
            line.strip() for line in response.text.strip().split("\n") if line.strip()
        ]
        return modules
    except Exception as e:
        st.error(f"An error occurred while processing with AI: {e}")
        return []


@st.cache_data
def get_topics_for_module(syllabus_text: str, module_name: str) -> list[str]:
    """
    Uses Google's Generative AI to extract topics from a specific module in the syllabus.

    Args:
        syllabus_text: The raw text extracted from the syllabus PDF.
        module_name: The name of the module to find topics for.

    Returns:
        A list of strings, where each string is a topic name.
    """
    if not syllabus_text or not module_name:
        return []
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Based on the following syllabus text, please identify and list the topics for the module named "{module_name}".
        Return ONLY the list of topic names, separated by newlines. Do not add any other text, titles, or introductions.
        List only the specific topics within that module, not the module name itself.

        Syllabus Text:
        ---
        {syllabus_text[:8000]}
        ---
        """
        response = model.generate_content(prompt)
        return [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    except Exception as e:
        st.error(f"An error occurred while fetching topics for {module_name}: {e}")
        return []


@st.cache_data
def get_syllabus_composition_analysis(_model, syllabus_text: str) -> dict:
    """
    Uses AI to analyze the composition of a syllabus.

    Args:
        _model: The generative AI model instance.
        syllabus_text: The raw text from the syllabus PDF.

    Returns:
        A dictionary with scores for different syllabus characteristics.
    """
    if not syllabus_text:
        return {}
    try:
        prompt = f"""
        Analyze the following syllabus text. On a scale of 1 to 10, rate its content based on the following criteria:
        1. "difficulty": The overall conceptual difficulty of the subject matter.
        2. "numerics": The emphasis on numeric problems, calculations, and mathematical derivations.
        3. "diagrams": How important diagrams, charts, and visual aids are for understanding the content.
        4. "definitions": The amount of theoretical concepts, definitions, and memorization required.

        Return the answer ONLY as a single, minified JSON object. Do not include any other text, explanations, or markdown formatting.
        Example: {{"difficulty": 7, "numerics": 8, "diagrams": 5, "definitions": 6}}

        Syllabus Text:
        ---
        {syllabus_text[:8000]}
        ---
        """
        response = _model.generate_content(prompt)
        # Clean up the response to extract only the JSON part
        json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        analysis = json.loads(json_text)
        return analysis
    except Exception as e:
        st.error(f"An error occurred during syllabus composition analysis: {e}")
        # Return a default structure on error
        return {"difficulty": 0, "numerics": 0, "diagrams": 0, "definitions": 0}


@st.cache_data
def get_upcoming_topic_analysis(_model, syllabus_text: str, all_topics: tuple, completed_topics: tuple) -> str:
    """
    Uses AI to analyze the next uncompleted topics for difficulty.

    Args:
        _model: The generative AI model instance.
        syllabus_text: The raw text from the syllabus PDF.
        all_topics: A tuple of all topics in the subject.
        completed_topics: A tuple of completed topics.

    Returns:
        A string containing an AI-generated analysis of upcoming topics.
    """
    if not syllabus_text or not all_topics:
        return "No topics available for analysis."

    uncompleted_topics = [topic for topic in all_topics if topic not in completed_topics]

    if not uncompleted_topics:
        return "Congratulations! You have completed all topics for this subject."

    # Analyze the next 1-3 topics
    next_topics_to_analyze = uncompleted_topics[:3]
    
    try:
        prompt = f"""
        Based on the provided syllabus text, analyze the upcoming topics and provide a brief summary of what to expect.
        Focus on the potential difficulty, whether they are more theoretical or practical, and any key concepts.
        Keep the analysis concise and encouraging.

        Upcoming topics to analyze: {', '.join(next_topics_to_analyze)}

        Syllabus Text:
        ---
        {syllabus_text[:8000]}
        ---
        """
        response = _model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred during upcoming topic analysis: {e}")
        return "Could not analyze upcoming topics due to an error."


@st.cache_data
def get_important_topics(_model, syllabus_text: str, all_topics: tuple) -> list[str]:
    """
    Uses AI to identify the most important topics from a list.
    """
    if not syllabus_text or not all_topics:
        return []
    try:
        prompt = f"""
        From the following list of topics for a subject, identify the 3 most important, foundational, or core concepts based on the provided syllabus text.
        These are topics that are likely prerequisites for others or are central to the subject's theme.
        Return ONLY the list of the 3 topic names, each on a new line. Do not add any other text, titles, or introductions.

        List of all topics:
        {', '.join(all_topics)}

        Syllabus Text:
        ---
        {syllabus_text[:8000]}
        ---
        """
        response = _model.generate_content(prompt)
        # Filter response to ensure topics are valid and from the original list
        important = [line.strip() for line in response.text.strip().split("\n") if line.strip() and line.strip() in all_topics]
        return important[:3] # Ensure we only return max 3
    except Exception as e:
        st.error(f"An error occurred while identifying important topics: {e}")
        return []


@st.cache_data
def get_practice_questions(_model, syllabus_text: str, topic_name: str) -> str:
    """
    Uses AI to generate practice questions for a specific topic.
    """
    if not syllabus_text or not topic_name:
        return "Cannot generate questions without a topic."
    try:
        prompt = f"""
        You are a helpful study assistant. Based on the provided syllabus text, generate 2-3 practice questions for the specific topic: "{topic_name}".
        The questions should be designed to test a student's understanding of the core concepts of this topic.
        The questions can be conceptual, problem-solving, or definition-based, as appropriate for the topic.
        Format the output as a numbered list. Do not include any other text, titles, or introductions.

        Syllabus Text:
        ---
        {syllabus_text[:8000]}
        ---
        """
        response = _model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating practice questions: {e}")
        return "Could not generate questions due to an error."


def get_chat_response(_model, syllabus_text: str, chat_history: list, user_question: str) -> str:
    """
    Generates a response from the AI based on the syllabus and chat history.
    """
    history_str = "\n".join([f"**{msg['role'].title()}**: {msg['content']}" for msg in chat_history])

    prompt = f"""
    **System Instruction:**
    You are an expert tutor for the subject described in the syllabus below. Your role is to answer a student's questions clearly and concisely.
    Base your answers STRICTLY on the provided syllabus text. If the answer cannot be found in the syllabus, politely state that the information is not available in the document.
    Do not answer questions that are outside the scope of the syllabus content.

    **Syllabus Text:**
    ---
    {syllabus_text[:8000]}
    ---
    
    **Conversation History:**
    {history_str}
    
    **New User Question:**
    {user_question}
    
    **Your Answer:**
    """
    
    try:
        response = _model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Log error for debugging
        print(f"An error occurred while getting a chat response: {e}")
        return "Sorry, I encountered an error while trying to respond. The AI service may be temporarily unavailable. Please try again later."


def get_youtube_search_links(topic_name: str, subject_name: str) -> list[str]:
    """
    Generates YouTube search links for a given topic.
    """
    base_url = "https://www.youtube.com/results?search_query="
    query1 = urllib.parse.quote(f"{topic_name} {subject_name} tutorial")
    link1 = f"- [Search for '{topic_name}' in '{subject_name}']({base_url}{query1})"
    query2 = urllib.parse.quote(f"{topic_name} explained")
    link2 = f"- [Search for '{topic_name} Explained']({base_url}{query2})"
    return [link1, link2]


def create_radar_chart(stats: dict, completed_percentage: float):
    labels = list(stats.keys())
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    total_values = list(stats.values())
    total_values += total_values[:1]

    completion_ratio = completed_percentage / 100.0
    completed_values = [v * completion_ratio for v in total_values]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, total_values, color='red', alpha=0.25, label='Total Syllabus')
    ax.plot(angles, total_values, color='red', linewidth=2)
    ax.fill(angles, completed_values, color='green', alpha=0.4, label='Completed')
    ax.plot(angles, completed_values, color='green', linewidth=2)
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label.replace('_', ' ').title() for label in labels])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig

def track_progress_page():
    """
    Displays the syllabus progress tracking page with pie charts for each subject.
    """
    st.title("📊 Track Progress")

    if not (
        "syllabus_processed" in st.session_state
        and st.session_state.syllabus_processed
    ):
        st.info(
            "You haven't uploaded and processed a syllabus yet. Please go to the 'Upload Syllabus' page."
        )
        return

    # Initialize progress tracking in session state if it doesn't exist
    if "progress" not in st.session_state:
        st.session_state.progress = {}

    # Defensive check for session state
    if "upload_data" not in st.session_state or "subjects" not in st.session_state.upload_data:
        st.error("Syllabus data is missing or corrupt. Please re-upload.")
        if "upload_data" in st.session_state:
            del st.session_state.upload_data
        if "syllabus_processed" in st.session_state:
            del st.session_state.syllabus_processed
        if "progress" in st.session_state:
            del st.session_state.progress
        st.rerun()
        return

    upload_data = st.session_state.upload_data
    st.header(f"Syllabus Overview for {upload_data['semester']}")

    # Check if modules have been extracted. If not, do it now.
    if "modules" not in upload_data["subjects"][0]:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY not found in .env file. Cannot analyze syllabus with AI.")
            return
        with st.spinner("🤖 AI is analyzing your syllabus to identify modules..."):
            for i, subject in enumerate(upload_data["subjects"]):
                if subject.get("pdf_content"):
                    modules = get_modules_from_syllabus(subject["pdf_content"])
                    st.session_state.upload_data["subjects"][i]["modules"] = modules
                else:
                    st.session_state.upload_data["subjects"][i]["modules"] = []
        st.success("AI analysis complete! Here is your syllabus breakdown:")
        st.rerun()

    progress_data = st.session_state.progress

    # Create columns for each subject to display them horizontally
    cols = st.columns(len(upload_data["subjects"]))

    # Display progress for each subject
    for i, subject in enumerate(upload_data["subjects"]):
        with cols[i]:
            subject_name = subject["name"]
            with st.container(border=True):
                st.subheader(f"{subject_name} ({subject['credits']} credits)")

                # Initialize progress for this subject if not present
                if subject_name not in progress_data:
                    progress_data[subject_name] = {}

                modules = subject.get("modules", [])
                if not modules:
                    st.warning("Could not identify modules for this subject.")
                    continue

                # Data for the horizontal bar chart
                progress_df_data = {"module": [], "Completion (%)": []}

                # Display each module and its progress tracking
                for module_name in modules:
                    # Initialize progress for this module if not present
                    if module_name not in progress_data[subject_name]:
                        progress_data[subject_name][module_name] = {
                            "topics": [],
                            "completed_topics": [],
                        }

                    # Get topics for the module using the cached AI function
                    all_topics = get_topics_for_module(subject.get("pdf_content", ""), module_name)

                    # Store the fetched topics if we haven't already
                    if not progress_data[subject_name][module_name]["topics"] and all_topics:
                         progress_data[subject_name][module_name]["topics"] = all_topics

                    # Use an expander for each module to not clutter the UI
                    with st.expander(f"Log progress for: {module_name}"):
                        if not all_topics:
                            st.write("No topics found by AI for this module.")
                            completed_percentage = 0
                        else:
                            # Multiselect for marking topics as complete
                            completed_topics = st.multiselect(
                                "Mark completed topics:",
                                options=all_topics,
                                default=progress_data[subject_name][module_name].get("completed_topics", []),
                                key=f"multiselect_{subject_name}_{module_name}"
                            )
                            # Update the session state immediately on change
                            progress_data[subject_name][module_name]["completed_topics"] = completed_topics

                            completed_percentage = (len(completed_topics) / len(all_topics)) * 100 if all_topics else 0
                            st.progress(int(completed_percentage), text=f"{len(completed_topics)} / {len(all_topics)} topics completed.")

                    progress_df_data["module"].append(module_name.split(":")[0]) # Shorten label for chart
                    progress_df_data["Completion (%)"].append(completed_percentage)

                # Create and display the horizontal bar chart for the subject
                if progress_df_data["module"]:
                    st.write("**Module Completion Overview**")
                    progress_df = pd.DataFrame(progress_df_data)
                    st.bar_chart(progress_df.set_index('module'))


def dashboard_page():
    """
    Displays the main dashboard with an overview of progress.
    """
    st.title("📊 Dashboard")

    # Check if syllabus has been processed
    if not ("syllabus_processed" in st.session_state and st.session_state.syllabus_processed):
        st.info("Welcome! Please upload a syllabus to get started.")
        if st.button("Go to Upload Page"):
            st.session_state.page_to_navigate = "Upload Syllabus"
            st.rerun()
        return

    # All data should be available if syllabus is processed
    upload_data = st.session_state.upload_data
    progress_data = st.session_state.progress
    subject_list = [s["name"] for s in upload_data["subjects"]]

    # 1. Subject Selection Dropdown
    selected_subject_name = st.selectbox("Select a subject to view details:", subject_list)

    if not selected_subject_name:
        st.warning("No subjects found. Please upload a syllabus.")
        return

    # Find the selected subject's data
    selected_subject = next((s for s in upload_data["subjects"] if s["name"] == selected_subject_name), None)
    if not selected_subject:
        st.error("Could not find data for the selected subject.")
        return

    # --- Calculate Overall Progress ---
    total_topics_count = 0
    completed_topics_count = 0
    subject_progress = progress_data.get(selected_subject_name, {})

    for module_name in selected_subject.get("modules", []):
        # Ensure module exists in progress data
        if module_name not in subject_progress:
            subject_progress[module_name] = {"topics": [], "completed_topics": []}

        module_progress = subject_progress[module_name]
        all_topics_in_module = module_progress.get("topics", [])
        
        if not all_topics_in_module:
            all_topics_in_module = get_topics_for_module(selected_subject.get("pdf_content", ""), module_name)
            st.session_state.progress[selected_subject_name][module_name]["topics"] = all_topics_in_module

        completed_topics_in_module = module_progress.get("completed_topics", [])
        
        total_topics_count += len(all_topics_in_module)
        completed_topics_count += len(completed_topics_in_module)

    overall_completion_percentage = (completed_topics_count / total_topics_count) * 100 if total_topics_count > 0 else 0

    # 2. Syllabus Coverage Display
    st.subheader(f"Progress for {selected_subject_name}")
    st.progress(int(overall_completion_percentage), text=f"{overall_completion_percentage:.1f}% Complete")
    st.metric(label="Topics Covered", value=f"{completed_topics_count} / {total_topics_count}")

    st.divider()

    # --- AI-Powered Insights ---
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Syllabus DNA")
        st.write("An AI-generated breakdown of the subject's characteristics.")
        
        with st.spinner("🔬 Analyzing syllabus composition..."):
            composition_stats = get_syllabus_composition_analysis(model, selected_subject.get("pdf_content", ""))

        if composition_stats and any(composition_stats.values()):
            radar_fig = create_radar_chart(composition_stats, overall_completion_percentage)
            st.pyplot(radar_fig)
        else:
            st.warning("Could not generate the syllabus DNA chart.")

    with col2:
        st.subheader("What's Next?")
        st.write("A look at your upcoming topics.")
        
        with st.expander("View Analysis", expanded=False):
            all_subject_topics = tuple(topic for module in subject_progress.values() for topic in module.get("topics", []))
            completed_subject_topics = tuple(topic for module in subject_progress.values() for topic in module.get("completed_topics", []))

            with st.spinner("🤖 Preparing your next challenge..."):
                upcoming_analysis = get_upcoming_topic_analysis(model, selected_subject.get("pdf_content", ""), all_subject_topics, completed_subject_topics)
            
            st.info(f"**Pacing:** You have completed **{int(overall_completion_percentage)}%** of this subject.")
            st.markdown("**Upcoming Topic Analysis:**")
            if upcoming_analysis:
                st.success(upcoming_analysis)

    st.divider()

    # --- AI Chat Assistant ---
    st.subheader("💬 AI Study Assistant")
    st.write(f"Ask questions about **{selected_subject_name}** and get answers based on the syllabus.")

    # Initialize chat history in session state
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if selected_subject_name not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[selected_subject_name] = [
            {"role": "assistant", "content": f"Hello! How can I help you with {selected_subject_name} today?"}
        ]

    # Display chat messages from history
    for message in st.session_state.chat_sessions[selected_subject_name]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(f"Ask about {selected_subject_name}..."):
        # Add user message to chat history and display it
        st.session_state.chat_sessions[selected_subject_name].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_for_model = st.session_state.chat_sessions[selected_subject_name][:-1]
                response = get_chat_response(
                    model,
                    selected_subject.get("pdf_content", ""),
                    history_for_model,
                    prompt
                )
                st.markdown(response)
        
        st.session_state.chat_sessions[selected_subject_name].append({"role": "assistant", "content": response})

    st.divider()

    with st.expander("🎓 View AI-Powered Study Recommendations", expanded=False):
        # We already have all_subject_topics calculated above
        if all_subject_topics:
            with st.spinner("🔍 Identifying key topics and generating recommendations..."):
                important_topics = get_important_topics(model, selected_subject.get("pdf_content", ""), all_subject_topics)

            if not important_topics:
                st.info("Could not identify key topics for recommendations at this time.")
            else:
                st.write("Here are some resources for topics the AI identified as most critical:")
                # Display recommendations for each important topic
                for topic in important_topics:
                    with st.container(border=True):
                        st.markdown(f"#### Key Topic: **{topic}**")
                        
                        rec_col1, rec_col2 = st.columns(2)

                        with rec_col1:
                            st.markdown("**🤔 Practice Questions**")
                            with st.spinner(f"Generating questions for '{topic}'..."):
                                questions = get_practice_questions(model, selected_subject.get("pdf_content", ""), topic)
                            st.info(questions)

                        with rec_col2:
                            st.markdown("**📺 Video Recommendations**")
                            st.write("Click to search for this topic on YouTube:")
                            video_links = get_youtube_search_links(topic, selected_subject_name)
                            for link in video_links:
                                st.markdown(link)
        else:
            st.warning("No topics found for this subject to generate recommendations.")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Syllabus Tracker", page_icon="📚", layout="centered")

    # --- Theme Toggle ---
    st.sidebar.header("Display Options")
    # Default to True for dark mode
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True, key="dark_mode")

    css = DARK_MODE_CSS if dark_mode else LIGHT_MODE_CSS
    st.markdown(css, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page_options = ["Dashboard", "Profile", "Upload Syllabus", "Track Progress"]
    st.sidebar.markdown("---") # Separator

    # If a programmatic navigation was requested, update the state for the radio
    # button BEFORE it is rendered.
    if "page_to_navigate" in st.session_state:
        st.session_state.navigation = st.session_state.page_to_navigate
        # Clean up the temporary state variable
        del st.session_state.page_to_navigate

    # Initialize navigation state if it doesn't exist to handle programmatic changes
    if "navigation" not in st.session_state:
        st.session_state.navigation = "Dashboard"
    page = st.sidebar.radio("Go to", page_options, key="navigation",)

    if page == "Dashboard":
        dashboard_page()
    elif page == "Profile":
        profile_page()
    elif page == "Upload Syllabus":
        upload_syllabus_page()
    elif page == "Track Progress":
        track_progress_page()


if __name__ == "__main__":
    main()