import streamlit as st
import os
import random
from datetime import datetime
import json
import time
from typing import Optional, Dict, List, Any

# Import your modules
from LearningBuddy import LearningBuddy, InterviewState
from FileReader import load_prompt, load_context, load_patient_profile, load_random_case_profile

# Streamlit App Configuration
st.set_page_config(
    page_title="Clinical Learning Buddy",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for chat interface
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Chat container */
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Therapist messages */
    [data-testid="chat-message-user"] {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    
    /* Patient messages */
    [data-testid="chat-message-assistant"] {
        background-color: #f3e5f5;
        margin-right: 20%;
    }
    
    /* Evaluation card */
    .evaluation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Critical feedback card */
    .critical-feedback {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%) !important;
    }
    
    /* Feedback sections */
    .feedback-section {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid rgba(255,255,255,0.5);
    }
    
    /* Score display */
    .score-display {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .score-high {
        background: linear-gradient(135deg, #51cf66 0%, #339933 100%);
        color: white;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        color: #333;
    }
    
    .score-low {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
    }
    
    /* Session info cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-block;
        background-color: #e0e0e0;
        padding: 10px 15px;
        border-radius: 20px;
        margin: 10px 0;
    }
    
    .typing-indicator span {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #999;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(1);
            opacity: 0.5;
        }
        40% {
            transform: scale(1.3);
            opacity: 1;
        }
    }
    
    /* Session header */
    .session-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-active {
        background-color: #4caf50;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'learning_buddy' not in st.session_state:
    st.session_state.learning_buddy = None
if 'current_state' not in st.session_state:
    st.session_state.current_state = None
if 'roleplay_active' not in st.session_state:
    st.session_state.roleplay_active = False
if 'roleplay_messages' not in st.session_state:
    st.session_state.roleplay_messages = []
if 'completed_sessions' not in st.session_state:
    st.session_state.completed_sessions = []
if 'phase_results' not in st.session_state:
    st.session_state.phase_results = {}
if 'typing' not in st.session_state:
    st.session_state.typing = False
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = None

# Helper functions
def simulate_typing(duration=2.0):
    """Simulate typing with variable duration"""
    placeholder = st.empty()
    placeholder.markdown(
        '<div class="typing-indicator"><span></span><span></span><span></span></div>',
        unsafe_allow_html=True
    )
    time.sleep(duration)
    placeholder.empty()

def calculate_response_delay(message_length):
    """Calculate realistic response delay based on message length"""
    # Base delay + additional time based on message length
    base_delay = random.uniform(0.8, 1.5)
    reading_time = len(message_length) * 0.01  # 0.01 seconds per character
    thinking_time = random.uniform(0.5, 1.0)
    return base_delay + reading_time + thinking_time

def format_session_duration(start_time):
    """Format session duration in a readable format"""
    if not start_time:
        return "0 minutes"
    duration = datetime.now() - start_time
    minutes = int(duration.total_seconds() / 60)
    seconds = int(duration.total_seconds() % 60)
    return f"{minutes}m {seconds}s"

# Sidebar
st.sidebar.title("🏥 Clinical Learning Buddy")
st.sidebar.markdown("---")

# API Key Input
if not st.session_state.api_key:
    st.sidebar.warning("⚠️ Please enter your OpenAI API key")
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        placeholder="sk-..."
    )
    if st.sidebar.button("Set API Key"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.session_state.learning_buddy = LearningBuddy(api_key_input)
            st.sidebar.success("✅ API Key set successfully!")
            st.rerun()
else:
    st.sidebar.success("✅ API Key configured")
    if st.sidebar.button("🔑 Change API Key"):
        st.session_state.api_key = None
        st.session_state.learning_buddy = None
        st.rerun()

# Feature Selection
feature = st.sidebar.radio(
    "Navigate",
    ["💬 Therapy Session", "📊 Session Analytics", "🎯 Performance Review"],
    index=0
)

# Session Status in Sidebar
if st.session_state.roleplay_active:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Session Status")
    st.sidebar.markdown(f'<span class="status-indicator status-active"></span> **Active Session**', unsafe_allow_html=True)
    st.sidebar.info(f"Duration: {format_session_duration(st.session_state.session_start_time)}")
    st.sidebar.warning("Messages: " + str(len(st.session_state.roleplay_messages) // 2))

# Main Content
if not st.session_state.learning_buddy and st.session_state.api_key:
    st.session_state.learning_buddy = LearningBuddy(st.session_state.api_key)

if feature == "💬 Therapy Session":
    # Session header
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    st.markdown("# 💬 Virtual Therapy Session")
    st.markdown("Practice your clinical skills with AI-powered patient simulations")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.api_key:
        st.error("🔐 Please enter your OpenAI API key in the sidebar to continue.")
    else:
        # Session Controls
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            if st.button("🎭 New Patient", use_container_width=True, disabled=st.session_state.roleplay_active, type="primary"):
                with st.spinner("🔄 Preparing new patient case..."):
                    try:
                        # Initialize new session
                        st.session_state.current_state = {
                            "case_study": None,
                            "diagnosis_summary": "",
                            "clinical_summary": "",
                            "chatlog": [],
                            "feedback": ""
                        }
                        
                        # Phase 1: Profile Verification
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Verifying patient profile...")
                        progress_bar.progress(30)
                        result = st.session_state.learning_buddy._phase_1_profile_verification(st.session_state.current_state)
                        st.session_state.current_state.update(result)
                        
                        # Phase 2: Clinical Summary
                        status_text.text("Generating clinical summary...")
                        progress_bar.progress(70)
                        result = st.session_state.learning_buddy._phase_2_summary(st.session_state.current_state)
                        st.session_state.current_state.update(result)
                        
                        progress_bar.progress(100)
                        status_text.text("Ready to begin!")
                        time.sleep(0.5)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.session_state.roleplay_active = True
                        st.session_state.roleplay_messages = []
                        st.session_state.session_start_time = datetime.now()
                        st.success("✅ Patient is ready! You may begin the session.")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🏁 End Session", use_container_width=True, disabled=not st.session_state.roleplay_active, type="secondary"):
                if st.session_state.roleplay_active:
                    # Add confirmation dialog
                    st.session_state.show_end_confirmation = True
        
        with col3:
            if st.button("📋 Patient Info", use_container_width=True, disabled=not st.session_state.current_state):
                st.session_state.show_patient_info = not getattr(st.session_state, 'show_patient_info', False)
        
        with col4:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.roleplay_active = False
                st.session_state.roleplay_messages = []
                st.session_state.current_state = None
                st.session_state.phase_results = {}
                st.session_state.session_start_time = None
                st.rerun()
        
        # End session confirmation
        if getattr(st.session_state, 'show_end_confirmation', False):
            st.warning("⚠️ Are you sure you want to end this session?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, End Session", use_container_width=True):
                    with st.spinner("🤔 Generating comprehensive feedback..."):
                        simulate_typing(3.0)  # Simulate processing time
                        try:
                            # Update chatlog
                            st.session_state.current_state['chatlog'] = st.session_state.roleplay_messages
                            
                            # Show additional processing message for strict evaluation
                            st.info("🔍 Conducting thorough clinical evaluation...")
                            time.sleep(1.5)
                            
                            # Generate feedback
                            result = st.session_state.learning_buddy._phase_4_feedback(st.session_state.current_state)
                            st.session_state.current_state.update(result)
                            
                            # Save session
                            st.session_state.completed_sessions.append({
                                "timestamp": datetime.now(),
                                "duration": format_session_duration(st.session_state.session_start_time),
                                "state": st.session_state.current_state.copy(),
                                "messages": st.session_state.roleplay_messages.copy()
                            })
                            
                            st.session_state.roleplay_active = False
                            st.session_state.show_end_confirmation = False
                            st.success("✅ Session completed successfully!")
                            st.info("📊 Check the feedback below and visit Session Analytics for detailed review.")
                        except Exception as e:
                            st.error(f"❌ Error generating feedback: {str(e)}")
            with col2:
                if st.button("❌ Continue Session", use_container_width=True):
                    st.session_state.show_end_confirmation = False
                    st.rerun()
        
        # Patient Information Display
        if st.session_state.current_state and getattr(st.session_state, 'show_patient_info', False):
            with st.container():
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("### 📋 Patient Information")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text_area("Case Details", st.session_state.current_state.get('case_study', ''), height=150, disabled=True)
                with col2:
                    st.markdown("**Quick Stats:**")
                    st.markdown(f"- Session: {'Active' if st.session_state.roleplay_active else 'Ended'}")
                    st.markdown(f"- Duration: {format_session_duration(st.session_state.session_start_time)}")
                    st.markdown(f"- Exchanges: {len(st.session_state.roleplay_messages) // 2}")
                
                if st.session_state.current_state.get('clinical_summary'):
                    st.markdown("### 📄 Clinical Summary")
                    st.info(st.session_state.current_state['clinical_summary'])
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat Interface
        if st.session_state.roleplay_active:
            st.markdown("---")
            st.markdown("### 💭 Therapy Conversation")
            
            # Chat messages container
            chat_container = st.container(height=400)
            
            with chat_container:
                # Welcome message if no messages yet
                if not st.session_state.roleplay_messages:
                    st.chat_message("assistant").markdown(
                        "👋 *The patient has entered the room. Please begin the session when you're ready.*"
                    )
                
                # Display conversation history
                for i, msg in enumerate(st.session_state.roleplay_messages):
                    if msg.get('type') == 'human' or msg.get('role') == 'user':
                        st.chat_message("user").markdown(msg.get("content", ""))
                    elif msg.get('type') == 'ai' or msg.get('role') == 'assistant':
                        st.chat_message("assistant").markdown(msg.get("content", ""))
            
            # Chat input
            if prompt := st.chat_input("Type your response as therapist...", disabled=st.session_state.typing):
                # Add therapist message
                st.session_state.roleplay_messages.append({
                    "type": "human",
                    "role": "user",
                    "content": prompt
                })
                
                # Set typing state
                st.session_state.typing = True
                st.rerun()
            
            # Generate patient response if typing
            if st.session_state.typing and st.session_state.roleplay_messages:
                with chat_container:                    
                        with st.empty():
                            # Calculate realistic delay
                            delay = calculate_response_delay(st.session_state.roleplay_messages[-1]["content"])
                            simulate_typing(delay)
                            
                            try:
                                # Prepare messages
                                instruction_prompt = load_prompt('Virtual Patient')
                                messages = [
                                    {"type": "system", "content": instruction_prompt},
                                    {"type": "human", "content": st.session_state.current_state['case_study']}
                                ] + st.session_state.roleplay_messages
                                
                                # Convert to LangChain format
                                from langchain.schema import SystemMessage, HumanMessage, AIMessage
                                langchain_messages = []
                                for m in messages:
                                    if m.get('type') == 'system':
                                        langchain_messages.append(SystemMessage(content=m['content']))
                                    elif m.get('type') == 'human':
                                        langchain_messages.append(HumanMessage(content=m['content']))
                                    elif m.get('type') == 'ai':
                                        langchain_messages.append(AIMessage(content=m['content']))
                                
                                # Get response
                                response = st.session_state.learning_buddy.llm_mini_virtualPatient.invoke(langchain_messages)
                                
                                # Add patient response
                                st.session_state.roleplay_messages.append({
                                    "type": "ai",
                                    "role": "assistant",
                                    "content": response.content
                                })
                                
                                st.session_state.typing = False
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                st.session_state.typing = False
        
        # Display Feedback After Session
        if st.session_state.current_state and st.session_state.current_state.get('feedback') and not st.session_state.roleplay_active:
            st.markdown("---")
            st.markdown("### 🎯 Clinical Supervision Feedback")
            
            # Warning banner for strict feedback
            st.warning("⚠️ **Note**: This feedback is intentionally strict and critical to accelerate your learning. Each point of criticism is an opportunity for growth.")
            
            # Parse feedback for critical errors (if present)
            feedback_text = st.session_state.current_state['feedback']
            
            # Check for critical errors in feedback
            if "CRITICAL ERROR" in feedback_text or "critical error" in feedback_text.lower():
                st.error("🚨 **CRITICAL ERRORS DETECTED** - Review immediately for patient safety")
            
            # Display feedback with enhanced formatting
            feedback_container = st.container()
            with feedback_container:
                # Use columns for better layout
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Main feedback content
                    st.markdown('<div class="evaluation-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);">', unsafe_allow_html=True)
                    st.markdown("#### 📋 Detailed Clinical Evaluation")
                    st.markdown(feedback_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Quick stats from feedback
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Quick Review")
                    
                    # Extract score if present in feedback
                    import re
                    score_match = re.search(r'SCORE:\s*(\d+)/100', feedback_text, re.IGNORECASE)
                    if score_match:
                        score = int(score_match.group(1))
                        if score >= 80:
                            st.success(f"Score: {score}/100")
                        elif score >= 70:
                            st.warning(f"Score: {score}/100")
                        else:
                            st.error(f"Score: {score}/100")
                    
                    # Session info
                    st.markdown(f"**Exchanges**: {len(st.session_state.roleplay_messages) // 2}")
                    st.markdown(f"**Duration**: {format_session_duration(st.session_state.session_start_time)}")
                    
                    # Performance indicator
                    if "exceptional" in feedback_text.lower():
                        st.success("Performance: Exceptional")
                    elif "competent" in feedback_text.lower():
                        st.info("Performance: Competent")
                    elif "developing" in feedback_text.lower():
                        st.warning("Performance: Developing")
                    else:
                        st.error("Performance: Needs Work")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📥 Save Feedback", use_container_width=True):
                    st.success("✅ Feedback saved to session history!")
            with col2:
                if st.button("🎭 New Patient", use_container_width=True):
                    st.session_state.current_state = None
                    st.rerun()
            with col3:
                if st.button("📊 View Analytics", use_container_width=True):
                    st.session_state.selected_feature = "📊 Session Analytics"
                    st.rerun()

elif feature == "📊 Session Analytics":
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    st.markdown("# 📊 Session Analytics & History")
    st.markdown("Track your progress and review past sessions")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.completed_sessions:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Sessions", len(st.session_state.completed_sessions))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            avg_messages = sum(len(s['messages']) for s in st.session_state.completed_sessions) / len(st.session_state.completed_sessions)
            st.metric("Avg Exchanges", f"{avg_messages/2:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_time = sum(int(s['duration'].split('m')[0]) for s in st.session_state.completed_sessions if 'duration' in s)
            st.metric("Total Time", f"{total_time} min")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if st.session_state.completed_sessions:
                last_session = st.session_state.completed_sessions[-1]['timestamp']
                st.metric("Last Session", last_session.strftime("%m/%d %H:%M"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Session History
        st.markdown("### 📚 Session History")
        
        # Filter options
        col1, col2 = st.columns([3, 1])
        with col2:
            sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First"])
        
        sessions_to_display = st.session_state.completed_sessions.copy()
        if sort_order == "Newest First":
            sessions_to_display.reverse()
        
        for i, session in enumerate(sessions_to_display):
            session_num = len(st.session_state.completed_sessions) - i if sort_order == "Newest First" else i + 1
            
            with st.expander(f"📁 Session {session_num} - {session['timestamp'].strftime('%Y-%m-%d %H:%M')} ({session.get('duration', 'N/A')})"):
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Patient", "💬 Conversation", "📝 Feedback", "📊 Stats"])
                
                with tab1:
                    st.markdown("**Patient Profile:**")
                    st.text_area("", session['state'].get('case_study', ''), height=200, disabled=True, key=f"case_{i}")
                    
                    if session['state'].get('clinical_summary'):
                        st.markdown("**Clinical Summary:**")
                        st.info(session['state'].get('clinical_summary'))
                
                with tab2:
                    st.markdown("**Therapy Conversation:**")
                    for j, msg in enumerate(session['messages']):
                        if msg.get('type') == 'human' or msg.get('role') == 'user':
                            st.chat_message("user").markdown(msg.get('content', ''))
                        elif msg.get('type') == 'ai' or msg.get('role') == 'assistant':
                            st.chat_message("assistant").markdown(msg.get('content', ''))
                
                with tab3:
                    st.markdown("**AI Supervisor Feedback:**")
                    if session['state'].get('feedback'):
                        st.markdown('<div class="info-card">', unsafe_allow_html=True)
                        st.markdown(session['state'].get('feedback'))
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No feedback available for this session")
                
                with tab4:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Session Metrics:**")
                        st.markdown(f"- Duration: {session.get('duration', 'N/A')}")
                        st.markdown(f"- Total Exchanges: {len(session['messages']) // 2}")
                        st.markdown(f"- Messages: {len(session['messages'])}")
                    with col2:
                        st.markdown("**Timing:**")
                        st.markdown(f"- Date: {session['timestamp'].strftime('%Y-%m-%d')}")
                        st.markdown(f"- Time: {session['timestamp'].strftime('%H:%M:%S')}")
    else:
        st.info("📭 No completed sessions yet. Start a therapy session to begin tracking your progress!")
        if st.button("🎭 Start First Session", use_container_width=True):
            st.session_state.selected_feature = "💬 Therapy Session"
            st.rerun()

elif feature == "🎯 Performance Review":
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    st.markdown("# 🎯 Performance Review")
    st.markdown("Analyze your clinical performance and growth")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.completed_sessions:
        # Performance Overview
        st.markdown("### 📈 Performance Overview")
        
        # Calculate performance metrics
        total_sessions = len(st.session_state.completed_sessions)
        total_exchanges = sum(len(s['messages']) // 2 for s in st.session_state.completed_sessions)
        avg_session_length = sum(int(s.get('duration', '0m').split('m')[0]) for s in st.session_state.completed_sessions) / total_sessions
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 🎓 Experience Level")
            if total_sessions < 5:
                st.markdown("**Beginner** - Keep practicing!")
            elif total_sessions < 15:
                st.markdown("**Intermediate** - Good progress!")
            else:
                st.markdown("**Advanced** - Expert level!")
            st.progress(min(total_sessions / 20, 1.0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### ⏱️ Session Management")
            st.markdown(f"**Avg Duration:** {avg_session_length:.1f} minutes")
            if avg_session_length < 10:
                st.markdown("Consider longer sessions for deeper therapy")
            elif avg_session_length > 30:
                st.markdown("Good session length for therapy practice")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 💬 Engagement")
            avg_exchanges = total_exchanges / total_sessions
            st.markdown(f"**Avg Exchanges:** {avg_exchanges:.1f}")
            if avg_exchanges < 5:
                st.markdown("Try more back-and-forth dialogue")
            else:
                st.markdown("Good conversational engagement!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Learning Progress Chart
        st.markdown("### 📊 Learning Progress")
        
        # Create simple progress visualization
        if len(st.session_state.completed_sessions) > 1:
            import pandas as pd
            
            # Prepare data
            dates = [s['timestamp'].date() for s in st.session_state.completed_sessions]
            durations = [int(s.get('duration', '0m').split('m')[0]) for s in st.session_state.completed_sessions]
            exchanges = [len(s['messages']) // 2 for s in st.session_state.completed_sessions]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Duration (min)': durations,
                'Exchanges': exchanges
            })
            
            # Display charts
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df.set_index('Date')['Duration (min)'])
            with col2:
                st.line_chart(df.set_index('Date')['Exchanges'])
        
        # Tips and Recommendations
        st.markdown("### 💡 Tips for Improvement")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("Based on your session history, here are some recommendations:")
        
        tips = []
        if avg_session_length < 15:
            tips.append("• Try to extend your sessions to allow for deeper therapeutic exploration")
        if avg_exchanges < 8:
            tips.append("• Engage in more back-and-forth dialogue to practice active listening")
        if total_sessions < 10:
            tips.append("• Regular practice will help you develop better clinical intuition")
        tips.append("• Review the AI feedback after each session to identify areas for improvement")
        tips.append("• Practice with different patient profiles to broaden your experience")
        
        for tip in tips:
            st.markdown(tip)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("📭 Complete some therapy sessions to see your performance review!")
        if st.button("🎭 Start First Session", use_container_width=True):
            st.session_state.selected_feature = "💬 Therapy Session"
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Clinical Learning Buddy v3.0 - Powered by ZACK!! and chatGPT")
st.sidebar.caption("© 2025 - Practice Makes Perfect 🎯")