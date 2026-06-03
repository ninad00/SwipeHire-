import asyncio
import json
import os
from dotenv import load_dotenv

from browser_use import Agent, Browser, Tools
from browser_use.tools.views import UploadFileAction
from browser_use.llm import ChatGroq
from browser_use.llm import ChatGoogle

# Load environment variables (if any exist)
load_dotenv()

def map_resume_to_info(resume_path: str) -> dict:
    """Parses resume.json and maps it to the structure needed for form filling."""
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f'Resume JSON file not found at: {resume_path}')

    with open(resume_path, 'r', encoding='utf-8') as f:
        resume_data = json.load(f)

    personal_info = resume_data.get('personal_info', {})
    
    # Split name into first and last name
    full_name = personal_info.get('name', '')
    name_parts = full_name.strip().split(maxsplit=1)
    first_name = name_parts[0] if name_parts else ''
    last_name = name_parts[1] if len(name_parts) > 1 else ''

    # Get email and phone numbers
    email = personal_info.get('personal_email') or personal_info.get('institution_email') or ''
    phone = personal_info.get('phone', '')

    # Map candidate's digital healthcare experience into a strong answer for:
    # "What drew you to healthcare?"
    charak_experience = ""
    for exp in resume_data.get('experience', []):
        if 'Charak' in exp.get('organization', ''):
            points = " ".join(exp.get('points', []))
            charak_experience = (
                f"my machine learning role at {exp.get('organization')}, "
                f"where I worked on retinal fundus image generalization using ViT/CLIP "
                f"and built risk prediction models."
            )
            break

    if charak_experience:
        why_healthcare = (
            f"I want to apply my machine learning and data science background to build impactful digital "
            f"health systems, inspired by {charak_experience}"
        )
    else:
        why_healthcare = (
            "I am highly motivated to leverage data science, deep learning, and advanced AI to solve "
            "pressing medical and clinical challenges."
        )

    # Compile the final structured info dictionary matching reference.py format
    info = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "age": "21",
        "US_citizen": False,
        "sponsorship_needed": True,
        "resume": "resume.pdf",
        # Default fallback values for IIT Indore student location
        "postal_code": "453552",
        "country": "India",
        "state": "Madhya Pradesh",
        "city": "Indore",
        "address": "IIT Indore Campus, Khandwa Road, Simrol",
        # Demographics & Disclosures
        "gender": "Male",
        "race": "Asian",
        "Veteran_status": "Not a veteran",
        "disability_status": "No disability",
        # Dynamic customized fields
        "why_healthcare": why_healthcare
    }
    return info

async def main():
    resume_json_path = 'resume.json'
    resume_pdf_path = 'resume.pdf'

    # 1. Parse and prepare structured application details from resume.json
    try:
        info = map_resume_to_info(resume_json_path)
        print("Successfully parsed and mapped candidate information:")
        print(json.dumps(info, indent=2))
    except Exception as e:
        print(f"Error parsing resume.json: {e}")
        return

    # 2. Configure the LLM
    # Use Gemma 4 31B via browser-use's optimized native ChatGoogle wrapper
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    print("Initializing Gemma 4 31B via optimized native ChatGoogle...")
    llm = ChatGoogle(
        model="gemma-4-31b-it",
        api_key=gemini_api_key,
    )

    # 3. Create browser actions and tools
    tools = Tools()

    @tools.action(description='Upload resume file')
    async def upload_resume(browser_session, path: str = resume_pdf_path):
        try:
            page = await browser_session.get_current_page()
            file_input = page.locator("input[type=file]").first
            await file_input.set_input_files(path)
            return f"Successfully uploaded resume from {path}"
        except Exception:
            return f"File registered. Please use the built-in upload_file_to_element action to attach the file."

    # Configure Browser
    browser = Browser(cross_origin_iframes=True)

    # 4. Formulate the comprehensive job application task
    task = f"""
    - Your goal is to fill out and submit a job application form with the provided information.
    - Navigate to https://apply.appcast.io/jobs/50590620606/applyboard/apply/
    - Scroll through the entire application and use extract_structured_data action to extract all the relevant information needed to fill out the job application form. use this information and return a structured output that can be used to fill out the entire form: {info}. Use the done action to finish the task. Fill out the job application form with the following information.
        - Before completing every step, refer to this information for accuracy. It is structured in a way to help you fill out the form and is the source of truth.
    - Follow these instructions carefully:
        - if anything pops up that blocks the form, close it out and continue filling out the form.
        - Do not skip any fields, even if they are optional. If you do not have the information, make your best guess based on the information provided.
        Fill out the form from top to bottom, never skip a field to come back to it later. When filling out a field, only focus on one field per step. For each of these steps, scroll to the related text. These are the steps:
            1) use input_text action to fill out the following:
                - "First name" (from first_name in info)
                - "Last name" (from last_name in info)
                - "Email" (from email in info)
                - "Phone number" (from phone in info)
            2) use the upload_file_to_element action to fill out the following:
                - Resume upload field (using the upload_resume action tool with resume.pdf)
            3) use input_text action to fill out the following:
                - "Postal code" (from postal_code in info)
                - "Country" (from country in info)
                - "State" (from state in info)
                - "City" (from city in info)
                - "Address" (from address in info)
                - "Age" (from age in info)
            4) use click action to select the following options:
                - "Are you legally authorized to work in the country for which you are applying?" (Select NO/False based on US_citizen in info)
                - "Will you now or in the future require sponsorship for employment visa status (e.g., H-1B visa status, etc.) to work legally for Rochester Regional Health?" (Select YES/True based on sponsorship_needed in info)
                - "Do you have, or are you in the process of obtaining, a professional license?"
                    - SELECT NO FOR THIS FIELD
            5) use input_text action to fill out the following:
                - "What drew you to healthcare?" (Use the detailed why_healthcare description from info)
            6) use click action to select the following options:
                - "How many years of experience do you have in a related role?"
                - "Gender" (Select Male based on gender in info)
                - "Race" (Select Asian based on race in info)
                - "Hispanic/Latino" (Select No)
                - "Veteran status" (Select Not a veteran based on Veteran_status in info)
                - "Disability status" (Select No disability based on disability_status in info)
            7) use input_text action to fill out the following:
                - "Today's date" (Use today's date)
            8) CLICK THE SUBMIT BUTTON AND CHECK FOR A SUCCESS SCREEN. Once there is a success screen, complete your end task of writing final_result and outputting it.
    - Before you start, create a step-by-step plan to complete the entire task. Make sure to delegate a step for each field to be filled out.
    *** IMPORTANT ***: 
        - You are not done until you have filled out every field of the form.
        - When you have completed the entire form, press the submit button to submit the application and use the done action once you have confirmed that the application is submitted
        - PLACE AN EMPHASIS ON STEP 4, the click action. That section should be filled out.
        - At the end of the task, structure your final_result as 1) a human-readable summary of all detections and actions performed on the page with 2) a list with all questions encountered in the page. Do not say "see above." Include a fully written out, human-readable summary at the very end.
    """

    available_file_paths = [os.path.abspath(resume_pdf_path)]

    # 5. Initialize and run the agent
    print("Starting browser-use job application agent...")
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        tools=tools,
        available_file_paths=available_file_paths,
        use_vision=False,
    )

    try:
        history = await agent.run()
        print("\n=== Submission Result ===")
        print(history.final_result())
    except Exception as e:
        print(f"An execution error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())