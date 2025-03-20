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

# Define qualification questions for Canadian immigration
# qualification_questions = [
#     "What is your full name and nationality?",
#     "What is your age and highest level of education?",
#     "What is your current occupation and how many years of work experience do you have?", 
#     "What is your English proficiency level (IELTS score if available)?",
#     "Do you have any French language skills?",
#     "Have you ever visited or lived in Canada before?",
#     "Which Canadian immigration program are you most interested in (Express Entry, Provincial Nominee, etc.)?"
# ]

qualification_questions = [
    "Are you married or planning to get married? If planning, when?",
    "What is your highest level of education?",
    "Do you have any work experience?",
    "What is your spouse's current status in Canada?",
    "When does your spouse's permit expire?",
    "Does your spouse have a job in Canada? (If yes, what is the NOC or job title?)",
    "How much funds can you show for your application?",
    "When do you want to submit your application?"
]



# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial message immediately
    initial_message = "Welcome to the Canadian Immigration Assistant! I'll help determine which immigration pathways might be best for you and if you qualify for a consultation. " + qualification_questions[0]
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
    st.write({json.dumps(responses, indent=2)})
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
        
        print("teri maa ka --> ", payload)

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )

        print("teri maa --> ", response.json())
        
        if response.status_code == 200:
            result = response.json()
            print("teri maa ki --> ", result)
            evaluation = json.loads(result["choices"][0]["message"]["content"])
            return evaluation
        else:
            st.error(f"Error calling OpenAI API: {response.status_code}")
            return {"is_qualified": False, "reasoning": "Error in evaluation", "next_steps": "Please contact our immigration team directly."}
            
    except Exception as e:
        st.error(f"Error evaluating candidacy: {str(e)}")
        return {"is_qualified": False, "reasoning": "Error in evaluation", "next_steps": "Please contact our immigration team directly."}

# Calendly integration tool
class CalendlyTool(BaseTool):
    name: str = "calendly_scheduler"
    description: str = "Schedule an immigration consultation using Calendly"
    
    def _run(self, query: str) -> str:
        # This is a simplified example - in production, you would:
        # 1. Get available time slots from Calendly API
        # 2. Present options to the user
        # 3. Create the appointment with their selection
        
        # For demo purposes, we'll simulate success
        available_slots = [
            {"datetime": "2025-03-20T10:00:00", "duration": "45min"},
            {"datetime": "2025-03-20T14:00:00", "duration": "45min"},
            {"datetime": "2025-03-21T11:00:00", "duration": "45min"}
        ]
        
        # In a real implementation, you'd call the Calendly API:
        # GET https://api.calendly.com/scheduling_links
        # with proper authentication in the header: Authorization: Bearer {token}
        # Details at: https://developer.calendly.com/api-docs/
        
        # For now, simulate success
        appointment_id = str(uuid.uuid4())
        return json.dumps({
            "success": True,
            "appointment_id": appointment_id,
            "available_slots": available_slots,
            "message": "Immigration consultation times retrieved. Please ask the user to select a time."
        })

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

# UI Components
st.title("Canadian Immigration Qualification Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to process user input
def process_input():
    if st.session_state.interview_complete:
        # Interview is complete - handle scheduling if qualified
        if st.session_state.is_sql:
            # Use LangChain agent for booking
            agent = initialize_booking_agent()
            
            # Add callback for streaming in Streamlit
            st_callback = StreamlitCallbackHandler(st.container())
            
            with st.chat_message("assistant"):
                response_container = st.empty()
                response = agent.run(st.session_state.user_input, callbacks=[st_callback])
                response_container.write(response)
                
                # Check if booking was made
                if "appointment" in response.lower() and "scheduled" in response.lower():
                    st.session_state.calendly_scheduled = True
                    st.success("Immigration consultation successfully scheduled!")
        else:
            # Not qualified, provide resources
            response = "Thank you for providing your information. Based on what you've shared, we recommend exploring our free resources and guides first. Would you like me to suggest some specific immigration resources for your situation?"
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    else:
        # Still going through qualification questions
        current_q = st.session_state.current_question_index
        
        # Save the response to the current question
        st.session_state.responses[qualification_questions[current_q]] = st.session_state.user_input
        
        # Generate feedback for the current response
        feedback = generate_llm_feedback(
            qualification_questions[current_q], 
            st.session_state.user_input
        )
        
        # Move to the next question
        st.session_state.current_question_index += 1
        
        # Check if we've gone through all questions
        if st.session_state.current_question_index >= len(qualification_questions):
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
            next_question = qualification_questions[st.session_state.current_question_index]
            response = f"{feedback} {next_question}"
            st.session_state.messages.append({"role": "assistant", "content": response})

# User input field
if not st.session_state.calendly_scheduled:
    user_input = st.chat_input("Your response")
    if user_input:
        st.session_state.user_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        process_input()
        st.rerun()
else:
    st.success("Thank you for scheduling! Our immigration specialist will meet with you at the scheduled time.")