import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

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


def profile_page():
    """
    Displays the user profile page where users can view and edit their information.
    """
    st.header("👤 User Profile")

    # Initialize session state for user profile if it doesn't exist
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "Your Name",
            "email": "your.email@example.com",
            "profile_pic": None,
        }

    col1, col2 = st.columns([1, 2])

    with col1:
        # Display current profile picture if it exists, otherwise a placeholder
        if st.session_state.user_profile["profile_pic"]:
            st.image(
                st.session_state.user_profile["profile_pic"],
                caption="Profile Picture",
                width=150,
            )
        else:
            st.markdown("**(No profile picture)**")

    with col2:
        st.subheader("Current Profile Information")
        st.write(f"**Name:** {st.session_state.user_profile['name']}")
        st.write(f"**Email:** {st.session_state.user_profile['email']}")

    st.divider()

    # Form to edit profile information
    with st.form("profile_form"):
        st.subheader("Edit Your Profile")

        # Input for name
        new_name = st.text_input("Name", value=st.session_state.user_profile["name"])

        # Input for email
        new_email = st.text_input("Email", value=st.session_state.user_profile["email"])

        # Input for profile picture
        new_profile_pic = st.file_uploader(
            "Upload a new profile picture", type=["png", "jpg", "jpeg"]
        )

        submitted = st.form_submit_button("Save Changes")

        if submitted:
            st.session_state.user_profile["name"] = new_name
            st.session_state.user_profile["email"] = new_email
            if new_profile_pic is not None:
                # To read the file, we need to get its bytes
                st.session_state.user_profile["profile_pic"] = new_profile_pic.getvalue()
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
        model = genai.GenerativeModel("gemini-pro")
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

    # Defensive check for session state
    if "upload_data" not in st.session_state or "subjects" not in st.session_state.upload_data:
        st.error("Syllabus data is missing or corrupt. Please re-upload.")
        if "upload_data" in st.session_state:
            del st.session_state.upload_data
        if "syllabus_processed" in st.session_state:
            del st.session_state.syllabus_processed
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

    # Display a pie chart for each subject
    for subject in upload_data["subjects"]:
        with st.container(border=True):
            st.subheader(f"{subject['name']} ({subject['credits']} credits)")
            modules = subject.get("modules", [])

            if not modules:
                st.warning("Could not identify distinct modules for this subject from the provided PDF.")
                continue

            fig, ax = plt.subplots()
            ax.pie([1] * len(modules), labels=modules, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Syllabus Tracker", page_icon="📚", layout="centered")

    st.sidebar.title("Navigation")
    page_options = ["Home", "Profile", "Upload Syllabus", "Track Progress"]

    # If a programmatic navigation was requested, update the state for the radio
    # button BEFORE it is rendered.
    if "page_to_navigate" in st.session_state:
        st.session_state.navigation = st.session_state.page_to_navigate
        # Clean up the temporary state variable
        del st.session_state.page_to_navigate

    # Initialize navigation state if it doesn't exist to handle programmatic changes
    if "navigation" not in st.session_state:
        st.session_state.navigation = "Home"
    page = st.sidebar.radio("Go to", page_options, key="navigation",)

    if page == "Home":
        st.title("📚 AI Syllabus Tracker")
        st.header("Welcome!")
        st.write("This application helps you track your syllabus progress using AI.")
        st.write("Use the sidebar to navigate to different sections.")
    elif page == "Profile":
        profile_page()
    elif page == "Upload Syllabus":
        upload_syllabus_page()
    elif page == "Track Progress":
        track_progress_page()


if __name__ == "__main__":
    main()