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
st.set_page_config(page_title="Lead Qualification Assistant", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
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

# Define qualification questions
qualification_questions = [
    "What is your name and company?",
    "What industry are you in and what is your role?",
    "What specific challenges are you facing that our solution might address?",
    "What is your timeline for implementing a solution?",
    "What's your budget range for this project?",
    "Do you have decision-making authority or who else is involved in the decision?",
    "How did you hear about our company?"
]

# Function to evaluate if lead is sales qualified
def evaluate_sql(responses):
    # Connect to OpenAI for evaluation
    # REPLACE WITH YOUR API KEY
    api_key = "YOUR_OPENAI_API_KEY"  # <-- Fill in your API key here
    
    prompt = f"""
    Based on the following responses to our qualification questions, determine if this is a Sales Qualified Lead (SQL).
    A Sales Qualified Lead typically:
    - Has a clear problem our solution can solve
    - Has budget available
    - Has decision-making authority or access to decision makers
    - Has a reasonable timeline (within 1-6 months)
    
    Responses:
    {json.dumps(responses, indent=2)}
    
    Return your evaluation as a JSON with these fields:
    - is_qualified: (true/false)
    - reasoning: (brief explanation)
    - next_steps: (recommended approach)
    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4",
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
            return {"is_qualified": False, "reasoning": "Error in evaluation", "next_steps": "Please contact our sales team directly."}
            
    except Exception as e:
        st.error(f"Error evaluating lead: {str(e)}")
        return {"is_qualified": False, "reasoning": "Error in evaluation", "next_steps": "Please contact our sales team directly."}

# Calendly integration tool
class CalendlyTool(BaseTool):
    name = "calendly_scheduler"
    description = "Schedule an appointment using Calendly"
    
    def _run(self, query: str) -> str:
        # REPLACE WITH YOUR CALENDLY API TOKEN
        calendly_token = "YOUR_CALENDLY_TOKEN"  # <-- Fill in your Calendly token here
        
        # This is a simplified example - in production, you would:
        # 1. Get available time slots from Calendly API
        # 2. Present options to the user
        # 3. Create the appointment with their selection
        
        # For demo purposes, we'll simulate success
        available_slots = [
            {"datetime": "2025-03-20T10:00:00", "duration": "30min"},
            {"datetime": "2025-03-20T14:00:00", "duration": "30min"},
            {"datetime": "2025-03-21T11:00:00", "duration": "30min"}
        ]
        
        # In a real implementation, you'd call the Calendly API:
        # GET https://api.calendly.com/scheduling_links
        # with proper authentication
        
        # For now, simulate success
        appointment_id = str(uuid.uuid4())
        return json.dumps({
            "success": True,
            "appointment_id": appointment_id,
            "available_slots": available_slots,
            "message": "Appointment options retrieved successfully. Please ask the user to select a time."
        })

# Initialize LangChain components for the agent
def initialize_booking_agent():
    # REPLACE WITH YOUR API KEY
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", api_key="YOUR_OPENAI_API_KEY")
    
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
st.title("Sales Qualification Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to process user input
def process_input():
    if st.session_state.interview_complete:
        # Interview is complete - handle free-form chat with agent if SQL
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
                    st.success("Appointment successfully scheduled!")
        else:
            # Not SQL, just respond nicely
            response = "Thank you for your interest. Based on your needs, we recommend exploring our self-service options. Would you like more information about those?"
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    else:
        # Still going through qualification questions
        current_q = st.session_state.current_question_index
        
        # Save the response to the current question
        st.session_state.responses[qualification_questions[current_q]] = st.session_state.user_input
        
        # Move to the next question
        st.session_state.current_question_index += 1
        
        # Check if we've gone through all questions
        if st.session_state.current_question_index >= len(qualification_questions):
            st.session_state.interview_complete = True
            
            # Evaluate if this is a Sales Qualified Lead
            evaluation = evaluate_sql(st.session_state.responses)
            st.session_state.is_sql = evaluation["is_qualified"]
            
            if evaluation["is_qualified"]:
                response = f"Thanks for sharing those details! Based on what you've told me, I think we can definitely help with your needs. {evaluation['reasoning']} Would you like to schedule a call with one of our sales representatives to discuss further?"
            else:
                response = f"Thank you for your interest. {evaluation['reasoning']} {evaluation['next_steps']}"
                
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Provide feedback on previous response and ask the next question
            if current_q > 0:
                feedback = generate_feedback(current_q, st.session_state.responses[qualification_questions[current_q-1]])
                next_question = qualification_questions[st.session_state.current_question_index]
                response = f"{feedback} {next_question}"
            else:
                response = qualification_questions[st.session_state.current_question_index]
                
            st.session_state.messages.append({"role": "assistant", "content": response})

# Function to generate contextual feedback
def generate_feedback(question_index, response):
    # Simple feedback patterns
    feedback_templates = [
        "Thanks for sharing that information.",
        "I appreciate your response about {}.",
        "That's helpful context about {}.",
        "Great, that gives us a better understanding of your situation.",
        "Thank you for that insight."
    ]
    
    import random
    
    # Extract key topic from the previous question
    prev_question = qualification_questions[question_index-1]
    topic = prev_question.split("What")[1].split("?")[0].strip() if "What" in prev_question else "that"
    
    feedback = random.choice(feedback_templates)
    if "{}" in feedback:
        feedback = feedback.format(topic)
        
    return feedback

# Display initial question or continue conversation
if len(st.session_state.messages) == 0:
    initial_message = "Hello! I'd like to learn more about your needs to see how we can help. " + qualification_questions[0]
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# User input field
if not st.session_state.calendly_scheduled:
    user_input = st.chat_input("Your response")
    if user_input:
        st.session_state.user_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        process_input()
        st.rerun()
else:
    st.success("Thank you for scheduling! Our sales representative will meet with you at the scheduled time.")