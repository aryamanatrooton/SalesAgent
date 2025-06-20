import streamlit as st
import datetime
import json
import requests
import uuid
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import BaseTool
from typing import Dict, List, Optional

# Configure page
st.set_page_config(page_title="Canadian Immigration Assistant", layout="wide")

# Define services and their questions
services = {
    "Study Visa": [
        "\n \nWhat program are you interested in studying?",
        "\n \nWhich province/territory do you plan to study in?",
        "\n \nWhat is your highest level of education?",
        "\n \nWhat is your English proficiency level (IELTS score if available)?",
        "\n \nHow much funds can you show for your application?",
        "\n \nWhen do you want to submit your application?"
    ],
    "SOWP": {
        "initial": [
            "\n \nAre you married or planning to get married? If planning, when?",
            "\n \nWhat is your highest level of education?",
            "\n \nDo you have any work experience?",
            "\n \nWhat is your spouse's current status in Canada?"
        ],
        "study_permit": [
            "\n \nWhat program is your spouse studying in Canada?",
            "\n \nHow long is the program?",
            "\n \nWhen does your spouse's study permit expire?",
            "\n \nHow much funds can you show for your application?",
            "\n \nWhen do you want to submit your application?"
        ],
        "work_permit": [
            "\n \nWhen does your spouse's work permit expire?",
            "\n \nDoes your spouse have a job in Canada? (If yes, what is the NOC or job title?)",
            "\n \nHow much funds can you show for your application?",
            "\n \nWhen do you want to submit your application?"
        ]
    },
    "Express Entry": [
        "\n \nWhat is your age?",
        "\n \nWhat is your highest level of education?",
        "\n \nHow many years of skilled work experience do you have?",
        "\n \nWhat is your English proficiency level (IELTS score if available)?",
        "\n \nDo you have any French language skills?",
        "\n \nHave you ever worked or studied in Canada before?",
        "\n \nWhen do you want to submit your application?"
    ],
    "PNP": [
        "\n \nWhich province are you interested in?",
        "\n \nWhat is your current occupation?",
        "\n \nHow many years of work experience do you have?",
        "\n \nWhat is your highest level of education?",
        "\n \nWhat is your English proficiency level?",
        "\n \nDo you have any connections to the province (job offer, family, etc.)?",
        "\n \nWhen do you want to submit your application?"
    ]
}

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial message immediately
    initial_message = "Welcome to the Canadian Immigration Assistant! I'll help determine which immigration pathways might be best for you. \n \nWhat is your full name and nationality?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False
if "is_sql" not in st.session_state:
    st.session_state.is_sql = None
if "calendly_scheduled" not in st.session_state:
    st.session_state.calendly_scheduled = False
if "service_selected" not in st.session_state:
    st.session_state.service_selected = False
if "selected_service" not in st.session_state:
    st.session_state.selected_service = None
if "spouse_status" not in st.session_state:
    st.session_state.spouse_status = None
if "qualification_questions" not in st.session_state:
    st.session_state.qualification_questions = []
if "in_conditional_flow" not in st.session_state:
    st.session_state.in_conditional_flow = False
if "show_service_buttons" not in st.session_state:
    st.session_state.show_service_buttons = False
if "show_spouse_status_buttons" not in st.session_state:
    st.session_state.show_spouse_status_buttons = False
if "first_question_answered" not in st.session_state:
    st.session_state.first_question_answered = False
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

# Get API keys from Streamlit secrets
api_key = st.secrets["OPEN_AI_KEY"]
calendly_token = st.secrets["CALENDLY_PAT"]

# Function to generate LLM feedback on responses
def generate_llm_feedback(question, response):
    # Connect to OpenAI for feedback generation  
    prompt = f"""
    I am a Canadian immigration consultant chatbot. The user has just answered this question:
    "{question}"
    
    Their response was:
    "{response}"
    
    Please generate a thoughtful, helpful, and encouraging 1-2 sentence response that:
    1. Acknowledges their answer
    2. Provides some useful context or insight related to Canadian immigration based on their response
    3. Maintains a professional but friendly tone
    
    Response:
    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            feedback = result["choices"][0]["message"]["content"].strip()
            return feedback
        else:
            return "Thank you for sharing that information."
            
    except Exception as e:
        return "I appreciate your response. That's helpful context."

# Function to evaluate if lead qualifies for immigration consultation
def evaluate_sql(responses):
    # Connect to OpenAI for evaluation
    prompt = f"""
    Based on the following responses to Canadian immigration qualification questions, determine if this person is a good candidate for personalized immigration consultation services.
    
    A qualified candidate typically:
    - Has education and work experience that may qualify for Canadian immigration programs
    - Has adequate language proficiency (English/French)
    - Has specific interest in a Canadian immigration program
    - Would benefit from personalized consultation rather than just general information
    
    Responses:
    {json.dumps(responses, indent=2)}
    
    Return your evaluation as a JSON with these fields:
    - is_qualified: (true/false)
    - reasoning: (brief explanation of why they qualify or don't qualify)
    - next_steps: (recommended immigration pathway or resources if not qualified)
    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            evaluation = json.loads(result["choices"][0]["message"]["content"])
            return evaluation
        else:
            st.error(f"Error calling OpenAI API: {response.status_code}")
            return {"is_qualified": False, "reasoning": "Error in evaluation", "next_steps": "Please contact our immigration team directly."}
            
    except Exception as e:
        st.error(f"Error evaluating candidacy: {str(e)}")
        return {"is_qualified": False, "reasoning": "Error in evaluation", "next_steps": "Please contact our immigration team directly."}

# Calendly integration tool
# Modify the CalendlyTool class to better handle the response
class CalendlyTool(BaseTool):
    name: str = "calendly_scheduler"
    description: str = "Schedule an immigration consultation using Calendly"
    
    def _run(self, query: str) -> str:
        try:
            # Get the user's information from session state
            user_name = ""
            user_email = ""
            
            # Try to extract name from responses
            if "name_nationality" in st.session_state.responses:
                name_info = st.session_state.responses["name_nationality"]
                # Simple extraction - in production you might want to use NLP
                name_parts = name_info.split()
                if len(name_parts) >= 1:
                    user_name = name_parts[0]  # Just take first word as name
            
            # Try to find email in the conversation
            for q, a in st.session_state.responses.items():
                if "@" in a and "." in a:  # Very basic email detection
                    user_email = a
                    break
            
            # Get available scheduling links from Calendly
            headers = {
                "Authorization": f"Bearer {calendly_token}",
                "Content-Type": "application/json"
            }
            
            event_api = st.secrets["EVENT_API"]

            querystring = {
                "max_event_count": 1,
                "owner": "{}".format(event_api),
                "owner_type": "EventType"
            }

            print(querystring)
            
            # Get the user's scheduling links
            scheduling_response = requests.post(
                "https://api.calendly.com/scheduling_links",
                headers=headers,
                params=querystring
            )
            print(scheduling_response.status_code)
            
            if scheduling_response.status_code != 201:
                return json.dumps({
                    "success": False,
                    "message": f"Failed to retrieve scheduling links: {scheduling_response.status_code}"
                })
            
            scheduling_data = scheduling_response.json()
            print(scheduling_data)
            
            # Extract the booking URL from the response
            # The correct structure is {'resource': {'booking_url': 'url', ...}}
            scheduling_link = None
            if "resource" in scheduling_data and "booking_url" in scheduling_data["resource"]:
                scheduling_link = scheduling_data["resource"]["booking_url"]
            
            if not scheduling_link:
                return json.dumps({
                    "success": False,
                    "message": "No scheduling links available in the response"
                })
            
            # For direct API scheduling, you would use the Event Types API:
            # 1. GET https://api.calendly.com/event_types to find the appropriate event type
            # 2. GET https://api.calendly.com/event_types/{uuid}/available_times to get available slots
            # 3. POST https://api.calendly.com/scheduled_events to create an event
            
            # For simplicity and better user experience, we'll return the scheduling link
            # This allows the user to complete booking on Calendly's optimized scheduling page
            
            scheduling_link = f"{scheduling_link}?email={user_email}&name={user_name}"


            print(scheduling_link)

            return json.dumps({
                "success": True,
                "scheduling_link": scheduling_link,
                "user_info": {
                    "name": user_name,
                    "email": user_email
                },
                "message": "Scheduling link retrieved successfully. Please direct the user to complete their booking."
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "message": f"Error retrieving Calendly scheduling information: {str(e)}"
            })
        

# Add this function to get user email if not already provided
def request_email():
    response = "To complete your booking, I'll need your email address. Please provide it so we can send you the confirmation details."
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.booking_step = "email_collection"


# Initialize LangChain components for the agent
def initialize_booking_agent():
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", api_key=api_key)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    tools = [
        CalendlyTool(),
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent="chat-conversational-react-description",
        verbose=True,
        memory=memory,
    )
    
    return agent

# Function to handle service selection
def select_service(service):
    # Add user's selection to chat history
    st.session_state.messages.append({"role": "user", "content": f"I'm interested in {service}"})
    
    st.session_state.selected_service = service
    st.session_state.service_selected = True
    st.session_state.show_service_buttons = False
    
    if service == "SOWP":
        # For SOWP, first add initial questions
        st.session_state.qualification_questions = services[service]["initial"]
        st.session_state.in_conditional_flow = True
    else:
        # For other services, add all questions
        st.session_state.qualification_questions = services[service]
        st.session_state.in_conditional_flow = False
    
    st.session_state.current_question_index = 0
    response = f"Thank you for selecting {service}. {st.session_state.qualification_questions[0]}"
    st.session_state.messages.append({"role": "assistant", "content": response})

# Function to handle spouse status selection
def select_spouse_status(status):
    # Add user's selection to chat history
    display_status = "Study Permit" if status == "study_permit" else "Work Permit"
    st.session_state.messages.append({"role": "user", "content": f"My spouse is on a {display_status}"})
    
    st.session_state.spouse_status = status
    st.session_state.show_spouse_status_buttons = False
    
    # Add the appropriate questions based on spouse status
    if status == "study_permit":
        st.session_state.qualification_questions.extend(services["SOWP"]["study_permit"])
    else:  # work_permit
        st.session_state.qualification_questions.extend(services["SOWP"]["work_permit"])
    
    st.session_state.in_conditional_flow = False
    
    # Move to the next question
    st.session_state.current_question_index += 1
    response = f"Thank you. {st.session_state.qualification_questions[st.session_state.current_question_index]}"
    st.session_state.messages.append({"role": "assistant", "content": response})

# UI Components
st.title("Canadian Immigration Qualification Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Show loading spinner when waiting for response
if st.session_state.waiting_for_response:
    with st.chat_message("assistant"):
        st.write("Thinking...")

# Function to process user input
# Modified process_input function for production-ready Calendly integration
def process_input():
    st.session_state.waiting_for_response = True
    
    if st.session_state.interview_complete:
        # Interview is complete - handle scheduling if qualified
        if st.session_state.is_sql:
            if "booking_step" not in st.session_state:
                st.session_state.booking_step = "initial"
                
            if st.session_state.booking_step == "initial":
                # Check if user wants to proceed with booking
                user_input_lower = st.session_state.user_input.lower()
                if any(word in user_input_lower for word in ["yes", "sure", "book", "schedule", "appointment", "consult"]):
                    # Check if we have an email
                    has_email = False
                    for q, a in st.session_state.responses.items():
                        if "@" in a and "." in a:
                            has_email = True
                            break
                    
                    if not has_email:
                        request_email()
                    else:
                        # Get Calendly scheduling link
                        calendly_tool = CalendlyTool()
                        result = calendly_tool._run("")
                        calendly_data = json.loads(result)
                        
                        if calendly_data["success"]:
                            # Store the scheduling link
                            st.session_state.scheduling_link = calendly_data["scheduling_link"]
                            st.session_state.booking_step = "link_provided"
                            
                            response = f"Great! I've set up a scheduling link for your immigration consultation. Please click the link below to select a time that works best for you:\n\n[Schedule Your Immigration Consultation]({calendly_data['scheduling_link']})\n\nAfter scheduling, you'll receive a confirmation email with the meeting details."
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            response = f"I'm sorry, there was an issue setting up your appointment scheduling. Please try again later or contact our support team directly. Error: {calendly_data['message']}"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = "No problem. If you decide you'd like to schedule a consultation later, just let me know. In the meantime, is there anything else I can help you with regarding Canadian immigration?"
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            elif st.session_state.booking_step == "email_collection":
                # Process the user's email
                user_email = st.session_state.user_input.strip()
                if "@" in user_email and "." in user_email:
                    # Store the email
                    st.session_state.responses["email"] = user_email
                    
                    # Get Calendly scheduling link
                    calendly_tool = CalendlyTool()
                    result = calendly_tool._run("")
                    calendly_data = json.loads(result)
                    
                    if calendly_data["success"]:
                        # Store the scheduling link
                        st.session_state.scheduling_link = calendly_data["scheduling_link"]
                        st.session_state.booking_step = "link_provided"
                        
                        response = f"Thank you for providing your email. I've set up a scheduling link for your immigration consultation. Please click the link below to select a time that works best for you:\n\n[Schedule Your Immigration Consultation]({calendly_data['scheduling_link']})\n\nAfter scheduling, you'll receive a confirmation email with the meeting details."
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = f"I'm sorry, there was an issue setting up your appointment scheduling. Please try again later or contact our support team directly. Error: {calendly_data['message']}"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = "That doesn't appear to be a valid email address. Please provide a valid email so we can send you the confirmation details."
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            elif st.session_state.booking_step == "link_provided":
                # After link is provided, just handle general conversation
                response = "Your scheduling link has been provided above. Once you've completed your booking, you'll receive a confirmation email. Is there anything else you'd like to know about the immigration process before your consultation?"
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Not qualified, provide resources
            response = "Thank you for providing your information. Based on what you've shared, we recommend exploring our free resources and guides first. Would you like me to suggest some specific immigration resources for your situation?"
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    elif not st.session_state.first_question_answered:
        # First question was about name and nationality
        st.session_state.responses["name_nationality"] = st.session_state.user_input
        st.session_state.first_question_answered = True
        
        # Now ask about service selection
        response = "Thank you! Which immigration service are you interested in?"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_service_buttons = True
    
    elif st.session_state.in_conditional_flow and st.session_state.selected_service == "SOWP" and st.session_state.current_question_index == 3:
        # We're at the spouse status question for SOWP
        st.session_state.responses[st.session_state.qualification_questions[st.session_state.current_question_index]] = st.session_state.user_input
        
        # Ask about spouse status with buttons
        response = "Is your spouse on a study permit or work permit?"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_spouse_status_buttons = True
    
    else:
        # Handle normal question flow
        current_q = st.session_state.current_question_index
        current_question = st.session_state.qualification_questions[current_q]
        
        # Save the response to the current question
        st.session_state.responses[current_question] = st.session_state.user_input
        
        # If this is an email question, store it specifically
        if "email" in current_question.lower():
            st.session_state.responses["email"] = st.session_state.user_input
        
        # Generate feedback for the current response
        feedback = generate_llm_feedback(
            current_question, 
            st.session_state.user_input
        )
        
        # Move to the next question
        st.session_state.current_question_index += 1
        
        # Check if we've gone through all questions
        if st.session_state.current_question_index >= len(st.session_state.qualification_questions):
            st.session_state.interview_complete = True
            
            # Evaluate if candidate qualifies for consultation
            evaluation = evaluate_sql(st.session_state.responses)
            st.session_state.is_sql = evaluation["is_qualified"]
            
            if evaluation["is_qualified"]:
                response = f"{feedback} Thank you for sharing your immigration details! {evaluation['reasoning']} Would you like to schedule a personalized consultation with one of our Canadian immigration specialists to discuss your options in detail?"
            else:
                response = f"{feedback} Thank you for your interest in Canadian immigration. {evaluation['reasoning']} {evaluation['next_steps']}"
                
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Provide LLM feedback on previous response and ask the next question
            next_question = st.session_state.qualification_questions[st.session_state.current_question_index]
            response = f"{feedback} {next_question}"
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state.waiting_for_response = False

# Add this to the main UI section to show the scheduling link as a clickable button when appropriate
if "booking_step" in st.session_state and st.session_state.booking_step == "link_provided" and "scheduling_link" in st.session_state:
    st.markdown(f"<a href='{st.session_state.scheduling_link}' target='_blank'><button style='background-color:#4CAF50;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;'>Schedule Your Consultation</button></a>", unsafe_allow_html=True)

# Display service selection buttons when needed
if st.session_state.show_service_buttons:
    st.write("Please select a service:")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.button("Study Visa", on_click=select_service, args=("Study Visa",), key="btn_study")
    with col2:
        st.button("SOWP", on_click=select_service, args=("SOWP",), key="btn_sowp")
    with col3:
        st.button("Express Entry", on_click=select_service, args=("Express Entry",), key="btn_ee")
    with col4:
        st.button("PNP", on_click=select_service, args=("PNP",), key="btn_pnp")

# Display spouse status buttons when needed
if st.session_state.show_spouse_status_buttons:
    st.write("Please select your spouse's permit type:")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Study Permit", on_click=select_spouse_status, args=("study_permit",), key="btn_study_permit")
    with col2:
        st.button("Work Permit", on_click=select_spouse_status, args=("work_permit",), key="btn_work_permit")

# User input field
if not st.session_state.calendly_scheduled and not (st.session_state.show_service_buttons or st.session_state.show_spouse_status_buttons):
    user_input = st.chat_input("Your response")
    if user_input:
        # Immediately display the user message
        st.session_state.user_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()  # Rerun to display the user message immediately

# Check if we need to process user input (separate from display)
if "user_input" in st.session_state and not st.session_state.waiting_for_response:
    # Process the input after displaying it
    process_input()
    # Clear the input to avoid processing it again
    if "user_input" in st.session_state:
        del st.session_state.user_input
    st.rerun()  # Update UI with bot response

elif st.session_state.calendly_scheduled:
    st.success("Thank you for scheduling! Our immigration specialist will meet with you at the scheduled time.")