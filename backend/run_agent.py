import sys
import os
import json
import asyncio
import base64
import firebase_admin
from firebase_admin import credentials, firestore

# Force output to use UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Ensure event loop policy is set for subprocesses on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def main():
    if len(sys.argv) < 3:
        print(json.dumps({"type": "error", "message": "Missing uid or job_id arguments"}))
        return
        
    uid = sys.argv[1]
    job_id = sys.argv[2]
    
    # 1. Initialize Firebase
    try:
        cred_path = os.path.join(os.path.dirname(__file__), "firebase.json")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Firebase initialization failed: {str(e)}"}))
        return
        
    try:
        # 2. Fetch Job Details from Firestore (collection "resumes")
        job_doc = db.collection("resumes").document(job_id).get()
        if not job_doc.exists:
            print(json.dumps({"type": "error", "message": "Job not found"}))
            return
            
        job_data = job_doc.to_dict()
        
        apply_link = None
        apply_options = job_data.get("apply_options", [])
        if apply_options and isinstance(apply_options, list):
            apply_link = apply_options[0].get("link")
        if not apply_link:
            apply_link = job_data.get("share_link")
            
        if not apply_link:
            print(json.dumps({"type": "error", "message": "No application link found"}))
            return
            
        # 3. Fetch User Profile Details from Firestore (collection "users")
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            print(json.dumps({"type": "error", "message": "User profile not found"}))
            return
            
        user_data = user_doc.to_dict()
        info_dict = user_data.get("info_dict", {})
        job_dict = user_data.get("job_dict", {})
        dynamic_keys = user_data.get("dynamic_keys", {})
        
        # 4. Check for persistent resume file
        resumes_dir = os.path.join(os.path.dirname(__file__), "resumes")
        resume_path = None
        for ext in [".pdf", ".docx"]:
            p = os.path.join(resumes_dir, f"{uid}{ext}")
            if os.path.exists(p):
                resume_path = p
                break
                
        # 5. Build dynamic schema profile JSON
        profile_context = {
            "info_dict": info_dict,
            "job_dict": job_dict,
            "dynamic_keys": dynamic_keys
        }
        profile_json_str = json.dumps(profile_context, indent=2)
        
        # Configure Gemini API Key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print(json.dumps({"type": "error", "message": "GEMINI_API_KEY is not configured on server"}))
            return
            
        from browser_use import Agent, Browser
        from browser_use.llm import ChatGoogle
        
        # Configure ChatGoogle
        llm = ChatGoogle(
            model="gemma-4-31b-it",
            api_key=gemini_api_key,
        )
        
        # Start browser in headful mode (headless=False) for stable execution
        browser = Browser(headless=False, disable_security=True, cross_origin_iframes=False)
        
        # Construct the task prompt
        task_prompt = f"""
Your goal is to fill out and submit a job application form using the user's details.
Here is the user's detailed information stored in our database:
{profile_json_str}

Since the format and keys of the user's information vary across different users, you should locate the matching values from the provided JSON. For example, look for first name, last name, email, phone, address, gender, veteran status, disability, work authorization, sponsorship, resume, etc. in any of the keys or sub-keys of the JSON. If a value is missing, make a reasonable guess or leave it default.

Navigate to the job application URL: {apply_link}

Scroll through the entire application to extract the required fields, fill them out from top to bottom, upload the resume if required.
"""
        
        available_files = []
        if resume_path:
            abs_resume = os.path.abspath(resume_path)
            available_files.append(abs_resume)
            task_prompt += f"\n- Upload the user's resume when prompted. The resume file is registered and available for selection at: {abs_resume} (use upload_file_to_element action to attach it)."
        else:
            task_prompt += f"\n- If the form asks for a resume, since none is uploaded, try to proceed without it or generate a dummy text resume if required."
            
        task_prompt += """
- Complete the form and click the submit button. Once you see a success screen or confirming message, call the done action.
- Be thorough. Do not skip fields.
"""
        
        async def on_step_end(agent_instance):
            try:
                screenshot_url = None
                resumes_dir = os.path.join(os.path.dirname(__file__), "resumes")
                os.makedirs(resumes_dir, exist_ok=True)
                screenshot_path = os.path.join(resumes_dir, f"{uid}_live.png")
                
                try:
                    # Capture screenshot safely using browser_session
                    screenshot_bytes = None
                    if agent_instance.browser_session:
                        screenshot_bytes = await agent_instance.browser_session.take_screenshot(
                            path=screenshot_path,
                            format='jpeg',
                            quality=45
                        )
                    if screenshot_bytes:
                        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                        screenshot_url = f"data:image/jpeg;base64,{screenshot_b64}"
                except Exception as e_screenshot:
                    sys.stderr.write(f"Screenshot capture failed: {e_screenshot}\n")
                    
                current_url = ""
                if agent_instance.browser_session:
                    try:
                        current_url = await agent_instance.browser_session.get_current_page_url()
                    except Exception:
                        pass
                        
                thinking = "Analyzing page..."
                next_goal = ""
                action_info = ""
                
                history_list = agent_instance.history.history
                if history_list:
                    last_item = history_list[-1]
                    if last_item.model_output:
                        thinking = getattr(last_item.model_output, 'thinking', "") or ""
                        next_goal = getattr(last_item.model_output, 'next_goal', "") or ""
                        action_info = str(getattr(last_item.model_output, 'action', "")) or ""
                        
                msg = {
                    "type": "step",
                    "step": agent_instance.state.n_steps,
                    "url": current_url,
                    "screenshot": screenshot_url,
                    "thinking": thinking,
                    "next_goal": next_goal,
                    "action": action_info,
                    "message": f"Completed step {agent_instance.state.n_steps}"
                }
                print(json.dumps(msg), flush=True)
            except Exception as e:
                sys.stderr.write(f"Error in on_step_end inside subprocess: {e}\n")
                
        agent = Agent(
            task=task_prompt,
            llm=llm,
            browser=browser,
            available_file_paths=available_files,
        )
        
        async def periodic_screenshot_loop():
            await asyncio.sleep(5)  # Let browser launch first
            resumes_dir = os.path.join(os.path.dirname(__file__), "resumes")
            os.makedirs(resumes_dir, exist_ok=True)
            screenshot_path = os.path.join(resumes_dir, f"{uid}_live.png")
            while True:
                try:
                    if agent.browser_session:
                        await agent.browser_session.take_screenshot(
                            path=screenshot_path,
                            format='jpeg',
                            quality=45
                        )
                except Exception as e_periodic:
                    sys.stderr.write(f"Periodic screenshot failed: {e_periodic}\n")
                await asyncio.sleep(10)

        screenshot_task = asyncio.create_task(periodic_screenshot_loop())
        try:
            history = await agent.run(on_step_end=on_step_end)
        finally:
            screenshot_task.cancel()
            try:
                await screenshot_task
            except asyncio.CancelledError:
                pass
        final_res = history.final_result() or "Form submitted successfully"
        
        print(json.dumps({
            "type": "success",
            "message": "Form automation successfully finished!",
            "result": final_res
        }), flush=True)
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({
            "type": "error",
            "message": f"Application failed: {str(e)}"
        }), flush=True)

if __name__ == '__main__':
    asyncio.run(main())
