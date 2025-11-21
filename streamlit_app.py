import streamlit as st
from datetime import datetime
import time
from LearningBuddy_Agent import LearningBuddy, STACER_MODE,  MI_MODE, CBT_MODE
from utils.Voice import *
import base64
import markdown2
import uuid
import io
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
from utils.api_key_loader import load_elevenlabs_api_key, load_openai_api_key
from utils.speech_to_text import STT
############################################################
# Streamlit App Configuration
############################################################
st.set_page_config(
    page_title="Learning Buddy: MI & CBT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

selected_mode = None
PASSWORD = "AI@Psychiatry2025"
############################################################
# CSS for enhanced chat interface
############################################################
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
    
    /* LangGraph status indicators */
    .langgraph-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .phase-indicator {
        background: rgba(255,255,255,0.2);
        padding: 5px 15px;
        border-radius: 15px;
        display: inline-block;
        margin: 5px;
        font-size: 0.9em;
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
    
    /* Session info cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* LangGraph workflow indicator */
    .workflow-indicator {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #2e7d32;
    }
    
    /* Processing overlay */
    .processing-overlay {
        background: rgba(102, 126, 234, 0.1);
        padding: 30px;
        border-radius: 15px;
        border: 2px solid #667eea;
        text-align: center;
        color: #667eea;
        font-weight: 600;
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
    
    /* Session header */
    .session-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #4caf50;
        animation: pulse 2s infinite;
    }
    
    .status-processing {
        background-color: #ff9800;
        animation: pulse 2s infinite;
    }
    
    .status-complete {
        background-color: #2196f3;
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
    
    /* Patient selection styling */
    .patient-selection {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 4px solid #4caf50;
    }
    
    /* LangGraph branding */
    .langgraph-branding {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 0.9em;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

############################################################
# Helper functions
############################################################
def autoplay_audio(audio_bytes: bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    unique_id = f"audio_{uuid.uuid4().hex}"

    html = f"""
    <html>
        <body>
            <audio id="{unique_id}" autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
            <script>
                const audio = document.getElementById("{unique_id}");
                if (audio) {{
                    audio.volume = 1.0;
                    audio.play().catch((e) => {{
                        console.warn("Autoplay blocked:", e);
                    }});
                }}
            </script>
        </body>
    </html>
    """

    # This embeds it in an iframe so Streamlit doesn't re-render it
    components.html(html, height=0)

def get_available_patient_profiles(directory='Patient_Profiles', mode=None):
    """Get list of available patient profile files"""

    # Select Patients available in the current mode
    if mode is None:
        mode = MI_MODE
    elif mode == MI_MODE:
        directory += "/MI"
    elif mode == CBT_MODE:
        directory += "/CBT"
    # Add more modes ...

    try:
        if not os.path.exists(directory):
            return []
        files = [f for f in os.listdir(directory) if f.endswith('.txt') or not '.' in f]
        return sorted(files)
    except Exception:
        return []

def get_patient_profile_preview(filepath):
    """Get a preview of the patient profile content"""
    try:
        content = st.session_state.learning_buddy.load_patient_profile(filepath)
        return content[:200] + "..." if len(content) > 200 else content
    except Exception:
        return "Unable to load preview"

def format_session_duration(start_time_iso):
    """Format session duration from ISO string"""
    try:
        start_time = datetime.fromisoformat(start_time_iso)
        duration = datetime.now() - start_time
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{minutes}m {seconds}s"
    except:
        return "0m 0s"

def reset_session_state():
    """Reset all session-related state variables"""
    st.session_state.session_data = None
    st.session_state.chat_messages = []
    st.session_state.session_active = False
    st.session_state.selected_patient_profile = None
    st.session_state.processing_phase = None
    st.session_state.show_patient_info = False
    st.session_state.show_end_confirmation = False
    st.session_state.presentation_messages = []
    st.session_state.presentation_started = False
    st.session_state.presentation_phase = False
    st.session_state.evaluator_chat_messages = []
    st.presentation_start_time = None
    st.session_start_time = None
    st.session_state.evaluator_intro_shown = False

def get_phase_display_name(phase):
    """Get user-friendly phase names"""
    phase_names = {
        "initialize_node": "🔄 Initializing",
        "verify_profile": "✅ Profile Verification", 
        "generate_summary": "📋 Clinical Summary",
        "setup_roleplay": "🎭 Roleplay Setup",
        "active_roleplay": "💬 Active Session",
        "evaluate_session": "🎯 Generating Feedback",
        "complete_session": "✅ Session Complete",
        "completed": "✅ Completed",
        "error": "❌ Error"
    }
    return phase_names.get(phase, phase)

def save_evaluator_transcript():
    filepath = st.session_state.session_data.get('folder_name')
    if not filepath:
        return  # Exit if no filepath is defined

    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, "evaluator_transcript.txt")

    with open(full_path, "w", encoding="utf-8") as f:
        for msg in st.session_state.evaluator_chat_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            f.write(f"{role}: {content}\n\n")


############################################################
# Initialize session state
############################################################
if 'api_key' not in st.session_state:
    st.session_state.api_key = load_openai_api_key()
if 'elevenlabs_api_key' not in st.session_state:
    st.session_state.elevenlabs_api_key = load_elevenlabs_api_key()
if 'learning_buddy' not in st.session_state:
    st.session_state.learning_buddy = None
if 'session_data' not in st.session_state:
    st.session_state.session_data = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'completed_sessions' not in st.session_state:
    st.session_state.completed_sessions = []
if 'selected_patient_profile' not in st.session_state:
    st.session_state.selected_patient_profile = None
if 'available_profiles' not in st.session_state:
    st.session_state.available_profiles = get_available_patient_profiles()
if 'processing_phase' not in st.session_state:
    st.session_state.processing_phase = None
if 'show_patient_info' not in st.session_state:
    st.session_state.show_patient_info = False
if 'show_end_confirmation' not in st.session_state:
    st.session_state.show_end_confirmation = False
if "muted" not in st.session_state:
    st.session_state.muted = False
if 'presentation_phase' not in st.session_state:
    st.session_state.presentation_phase = False
if 'presentation_messages' not in st.session_state:
    st.session_state.presentation_messages = []
if 'presentation_started' not in st.session_state:
    st.session_state.presentation_started = False
if "evaluator_chat_messages" not in st.session_state:
    st.session_state.evaluator_chat_messages = []
if "evaluator_muted" not in st.session_state:
    st.session_state.evaluator_muted = False
if "password" not in st.session_state:
    st.session_state.password = False
if "evaluator_intro_shown" not in st.session_state:
    st.session_state.evaluator_intro_shown = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "mode_label" not in st.session_state:
    st.session_state.mode_label = None
if 'voice_transcriber' not in st.session_state:
        try:
            st.session_state.voice_transcriber = STT()
        except ValueError as e:
            st.error(f"Failed to initialize voice services: {e}")
            st.stop()


############################################################
# Sidebar Configuration
############################################################
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 350px !important;  
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 350px !important;
        }
    </style>
""", unsafe_allow_html=True)
st.sidebar.title("Learning Buddy for MI & CBT")
st.sidebar.markdown("---")

############################################################
# Mode Management
############################################################
if not st.session_state.get("mode"):
    if not st.session_state.get("password"):
        st.sidebar.warning("⚠️Select a Mode and Enter Password⚠️")
    else:
        st.sidebar.warning("⚠️ Please Select a Mode")

    # Optional extra info in an expander
    with st.sidebar.expander("ℹ️ More about mode selection"):
        st.markdown(
            """
            **Selecting a mode** changes the style of patient interview you conduct and the criteria used for evaluation and feedback.
             
            **MI (Motivational Interviewing):**  
            Practice using motivational interviewing techniques with the patient.

            **CBT (Cognitive Behavioral Therapy):**  
            Practice using cognitive behavioral therapy techniques with the patient.
            """
        )

    # Mode selection dropdown
    mode_options = {
        "Motivational Interviewing (MI)": MI_MODE,
        "Cognitive Behavioural Therapy (CBT)": CBT_MODE
        # APPEND MORE OPTIONS HERE
    }
    selected_mode_label = st.sidebar.selectbox("Select Mode", list(mode_options.keys()))
    selected_mode = mode_options[selected_mode_label]
    if not st.session_state.get("password"):
        password_input = st.sidebar.text_input(
            "Password",
            type="password",
            placeholder="Enter Password"
        )

    if st.sidebar.button("🟢 Get Started"):
        if st.session_state.password or password_input == PASSWORD:
            try:
                st.session_state.password = True
                st.session_state.mode = selected_mode
                st.session_state.mode_label = selected_mode_label
                st.session_state.available_profiles = get_available_patient_profiles(mode = st.session_state.mode)
                st.session_state.learning_buddy = LearningBuddy(st.session_state.api_key, st.session_state.elevenlabs_api_key, selected_mode)

                st.sidebar.success("✅ Login Success!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error initializing LearningBuddy: {str(e)}")
        else:
            st.sidebar.error("❌ Incorrect Password")

else:
    st.sidebar.success(f"✅ Mode Configured")
    # Optional extra info in an expander
    with st.sidebar.expander("ℹ️ More about mode selection"):
        st.markdown(
            """
            **Selecting a mode** changes the style of patient interview you conduct and the criteria used for evaluation and feedback.
             
            **MI (Motivational Interviewing):**  
            Practice using motivational interviewing techniques with the patient.

            **CBT (Cognitive Behavioral Therapy):**  
            Practice using cognitive behavioral therapy techniques with the patient.
            """
        )
    if st.sidebar.button("🔄 Change Mode", disabled=st.session_state.processing_phase is not None or st.session_state.session_active):
        st.session_state.learning_buddy = None
        st.session_state.mode = None
        st.session_state.mode_label = None
        reset_session_state()
        st.rerun()


############################################################
# Patient Profile Selection
############################################################

if st.session_state.api_key and st.session_state.mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Patient Selection")
    
    # Patient profile selection
    profile_options = ["Random"] + st.session_state.available_profiles
    
    selected_option = st.sidebar.selectbox(
        "Choose Patient Profile:",
        options=profile_options,
        index=0,
        disabled=st.session_state.session_active or st.session_state.processing_phase is not None
    )
    
    if selected_option != "Random":
        print("Selected option",selected_option)
        if st.session_state.learning_buddy.mode == STACER_MODE:
            st.session_state.selected_patient_profile = os.path.join('Patient_Profiles/CBT', selected_option)
        elif st.session_state.learning_buddy.mode == MI_MODE:
            st.session_state.selected_patient_profile = os.path.join('Patient_Profiles/MI', selected_option)
        elif st.session_state.learning_buddy.mode == CBT_MODE:
            st.session_state.selected_patient_profile = os.path.join('Patient_Profiles/CBT', selected_option)

        # Show preview
        with st.sidebar.expander("📋 Profile Preview"):
            preview = get_patient_profile_preview(st.session_state.selected_patient_profile)
            st.text_area("Profile Preview", preview, height=100, disabled=True, key="profile_preview", label_visibility="collapsed")
    else:
        st.session_state.selected_patient_profile = None
        st.sidebar.info("🎲 Random patient will be selected")

    # Optional extra info in an expander
    with st.sidebar.expander("ℹ️ More about patient profiles"):
        st.markdown(
            """
            Each patient profile contains a detailed case study including symptoms, history, and behavior patterns.  
            Selecting a specific profile allows you to practice on a particular type of patient or condition.  
            Using 'Random' can help simulate a variety of clinical scenarios.
            """
        )


############################################################
# LangGraph Session Status
############################################################
if st.session_state.session_data or st.session_state.processing_phase:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 System Status")
    
    if st.session_state.processing_phase:
        phase_name = get_phase_display_name(st.session_state.processing_phase)
        st.sidebar.markdown(f'<span class="status-indicator status-processing"></span> **{phase_name}**', unsafe_allow_html=True)
        st.sidebar.warning("⏳ Core workflows running...")
    elif st.session_state.session_data:
        current_phase = st.session_state.session_data.get('current_phase', 'unknown')
        phase_name = get_phase_display_name(current_phase)
        session_active = st.session_state.session_data.get('session_active', False)
        
        if session_active:
            st.sidebar.markdown(f'<span class="status-indicator status-active"></span> **{phase_name}**', unsafe_allow_html=True)
            # Show session stats
            if st.session_state.learning_buddy:
                session_stats = st.session_state.learning_buddy.get_session_statistics()
                st.sidebar.info(f"Exchanges: {session_stats.get('total_exchanges', 0)}")
                
                # Show duration if available
                session_metadata = st.session_state.session_data.get('session_metadata', {})
                if session_metadata.get('start_time'):
                    duration = format_session_duration(session_metadata['start_time'])
                    st.sidebar.info(f"Duration: {duration}")
        else:
            st.sidebar.markdown(f'<span class="status-indicator status-complete"></span> **{phase_name}**', unsafe_allow_html=True)


############################################################
# Main Content
############################################################
if True:
    # Session header
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    if st.session_state.mode_label:
        st.markdown(
            "<h1 style='text-align: center;'>Agentic Patient Interview & Evaluation:</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h2 style='text-align: center;'>{st.session_state.mode_label}</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h1 style='text-align: center;'>Agentic Patient Interview & Evaluation</h1>",
            unsafe_allow_html=True
        )
    st.markdown(
        "<h5 style='text-align: center;'>Practice clinical skills with AI-powered patient simulations and evaluations</h5>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.password:
        st.error("🔐 Please enter the password in the sidebar to continue.")
    
    elif not st.session_state.learning_buddy:
        st.error("❌ Please select a mode in the sidebar to continue.")
    
    else:
        ############################################################
        # Session Controls
        ############################################################
        col1, col2, col3 = st.columns([2, 2, 2])
        
        controls_disabled = st.session_state.processing_phase is not None
        
        with col1:
            if st.button("🎭 Start Session", 
                       use_container_width=True, 
                       disabled=st.session_state.session_active or controls_disabled, 
                       type="primary"):
                
                # Set processing phase
                reset_session_state()
                st.session_state.processing_phase = "initialize_node"
                st.rerun()
        
        with col2:
            show_end_button = ((st.session_state.session_active or st.session_state.presentation_phase) and
                               not st.session_state.show_end_confirmation and
                               not controls_disabled)
            
            if st.button("🏁 End Session", 
                       use_container_width=True, 
                       disabled=not show_end_button, 
                       type="secondary"):
                st.session_state.presentation_phase = False
                st.session_state.show_end_confirmation = True
                st.rerun()

        with col3:
            if st.button("🔄 Reset All", 
                       use_container_width=True,
                       disabled=controls_disabled):
                if st.session_state.learning_buddy:
                    st.session_state.learning_buddy.reset_session()
                reset_session_state()
                st.rerun()

        ############################################################
        # Process session initialization
        ############################################################
        if st.session_state.processing_phase == "initialize_node":
            st.markdown('<div class="processing-overlay">', unsafe_allow_html=True)
            st.markdown("### 🔄 Initializing AI-centric Workflow...")
            st.markdown("Running through automated phases: Profile Verification → Clinical Summary → Roleplay Setup")
            
            # Show progress
            progress_placeholder = st.empty()
            with progress_placeholder:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Phase indicators
                status_text.text("🔄 Starting workflow...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                status_text.text("✅ Profile verification...")
                progress_bar.progress(30)
                time.sleep(0.3)
                
                status_text.text("📋 Generating clinical summary...")
                progress_bar.progress(60)
                time.sleep(0.3)
                
                status_text.text("🎭 Setting up roleplay...")
                progress_bar.progress(90)
                time.sleep(0.3)
                
                try:
                    session_result = st.session_state.learning_buddy.start_session(
                        case_study=None,
                        selected_profile=st.session_state.selected_patient_profile
                    )
                    
                    if session_result.get("error_message"):
                        st.error(f"❌ Error: {session_result['error_message']}")
                        st.session_state.processing_phase = None
                    else:
                        progress_bar.progress(100)
                        status_text.text("✅ Session ready!")
                        time.sleep(0.5)
                        
                        # Update session state
                        st.session_state.session_data = session_result
                        st.session_state.session_active = session_result.get('session_active', False)
                        st.session_state.chat_messages = []
                        st.session_state.processing_phase = None
                        
                        st.success("✅ Initial setup completed successfully!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ System error encountered: {str(e)}")
                    st.session_state.processing_phase = None
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(1)
            st.rerun()


        ############################################################
        # End session confirmation
        ############################################################
        if st.session_state.show_end_confirmation and not st.session_state.processing_phase:
            st.warning("⚠️ Are you sure you want to end this session?")
            st.info("🔍 This will trigger AI feedback evaluation workflow.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, End Session", use_container_width=True, type="primary"):
                    st.session_state.processing_phase = "evaluate_session"
                    st.session_state.show_end_confirmation = False
                    st.rerun()
            with col2:
                if st.button("❌ Continue Session", use_container_width=True):
                    st.session_state.show_end_confirmation = False
                    st.rerun()
        
        # Process session ending
        if st.session_state.processing_phase == "evaluate_session":
            st.markdown('<div class="processing-overlay">', unsafe_allow_html=True)
            st.markdown("### 🎯 Running AI Feedback Evaluation...")
            st.markdown("AI supervisor analyzing session using comprehensive clinical criteria. This may take a moment(Around 3-5 minutes).")
            
            progress_placeholder = st.empty()
            with progress_placeholder:
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                progress_bar.empty()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            try:
                # End session using LangGraph
                result = st.session_state.learning_buddy.end_session()
                
                if result.get("error"):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    # Save completed session
                    session_export = st.session_state.learning_buddy.export_session_data()
                    if session_export:
                        st.session_state.completed_sessions.append({
                            "timestamp": datetime.now(),
                            "langgraph_data": session_export,
                            "profile_used": st.session_state.selected_patient_profile or "Random"
                        })
                    
                    # Update session state
                    st.session_state.session_data = result
                    st.session_state.session_active = False
                    st.session_state.processing_phase = None
                    
                    st.success("✅ AI feedback evaluation completed!")
                    st.info("📊 Check the feedback below and visit Session Analytics for detailed review.")
                
            except Exception as e:
                st.error(f"❌ AI feedback error: {str(e)}")
                st.session_state.processing_phase = None
            
            time.sleep(1)
            st.rerun()


        ############################################################
        #  Before Starting
        ############################################################
        if (not st.session_state.session_active
                and st.session_state.learning_buddy
                and not st.session_state.show_end_confirmation
                and not st.session_state.processing_phase
                and not st.session_state.session_data
                ):
            st.info(
                f"Press **'Start Session'** to begin your **{st.session_state.mode_label}** evaluation. \n\n"
                "You may optionally select a **patient profile** before starting. "
                "**First-time users** are encouraged to leave it on the default **'Random'** option."
            )

        ############################################################
        # Active Chat Interface (After pressing Start Session)
        ############################################################
        if (st.session_state.session_active and 
            not st.session_state.processing_phase and 
            not st.session_state.show_end_confirmation):

            st.info(
                f"Use the **chat window** below to conduct your virtual patient inverview. \n\n"
                f"Press **'End Session'** to end the interview and receive your **{st.session_state.mode_label}** evaluation."
            )
            # Clinical Summary Reference
            if st.session_state.session_data.get("patient_context", {}).get("clinical_summary"):
                with st.expander("📋 **Clinical Summary** - Therapist Reference", expanded=False):
                    st.markdown(st.session_state.session_data['patient_context']['clinical_summary'])
                    
                    # Session stats
                    col1, col2, col3 = st.columns(3)
                    if st.session_state.learning_buddy:
                        session_stats = st.session_state.learning_buddy.get_session_statistics()
                        with col1:
                            profile_type = "Selected" if st.session_state.selected_patient_profile else "Random"
                            st.markdown(f"**Profile**: {profile_type}")
                        with col2:
                            st.markdown(f"**Exchanges**: {session_stats.get('total_exchanges', 0)}")
                        with col3:
                            session_metadata = st.session_state.session_data.get('session_metadata', {})
                            if session_metadata.get('start_time'):
                                duration = format_session_duration(session_metadata['start_time'])
                                st.markdown(f"**Duration**: {duration}")

            #Chat Set up
            # --- Initialize session start time ---
            if 'session_start_time' not in st.session_state:
                st.session_state.session_start_time = datetime.now()

            # --- Initialize chat messages ---
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []

            # --- Add Evaluator Welcome Message (only once) ---
            if not st.session_state.evaluator_intro_shown:
                mode = st.session_state.mode
                if mode == STACER_MODE:
                    st.session_state.chat_messages.append({
                        "role": "evaluator",
                        "content": "Hello and welcome to your simulated CBT evaluation. I will be evaluating your patient interviewing skills today. You may use this chat interface to speak with the virtual patient. When you have finished speaking with the patient, press the 'End Session' button. "
                    })
                elif mode == MI_MODE:
                       st.session_state.chat_messages.append({
                        "role": "evaluator",
                        "content": "Hello and welcome to your simulated MI evaluation. I will be evaluating your motivational interviewing skills today. You may use this chat interface to speak with the virtual patient. When you have finished speaking with the patient, press the 'End Session' button. "
                    })
                elif mode == CBT_MODE:
                       st.session_state.chat_messages.append({
                        "role": "evaluator",
                        "content": "Hello and welcome to your simulated CBT evaluation. I will be evaluating your cognitive behavioural therapy skills today. You may use this chat interface to speak with the virtual patient. When you have finished speaking with the patient, press the 'End Session' button. "
                    })
                st.session_state.chat_messages.append({
                    "role": "evaluator",
                    "content": "The virtual patient is in the chatroom with us. You may now begin."
                })
                st.session_state.evaluator_intro_shown = True

            # Chat Interface
            chat_container = st.container(height=400)
            
            with chat_container:

                # Display chat history
                for msg in st.session_state.chat_messages:
                    role = msg.get('role')
                    content = msg.get('content', '')

                    if role == 'user':  # Therapist (Right side)
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="background-color: #e3f2fd; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                <strong>You:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif role == 'assistant':  # Patient (Left side)
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #f3e5f5; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                <strong>Patient:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif role == 'evaluator':
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #f1f8e9; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                <strong>Evaluator:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Mute Toggle
            col1, col2, col_filler, col3 = st.columns([0.06, 1, 0.5, 0.2])

            with col1:
                st.toggle("Muted", key="muted", label_visibility="collapsed")

            with col2:
                if st.session_state.muted:
                    st.markdown(
                        "<span style='color:red; font-weight:bold; font-size:20px;'>🔇 Audio playback is off</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:green; font-weight:bold; font-size:20px;'>🔊 Audio playback is on</span>",
                        unsafe_allow_html=True,
                    )
            with col3:
                # Get voice input
                audio_info = mic_recorder(
                    start_prompt="🎤 Speak",
                    stop_prompt="⏹️ Stop",
                    just_once=True,
                    key='mic_interview'
                )

            # Chat Input
            # NEW: Voice and Text Input Section
            def handle_interview_message(prompt_text):
                """Central function to process a user's message."""
                if not prompt_text:
                    return

                # Add user message to display
                st.session_state.chat_messages.append({"role": "user", "content": prompt_text})

                # Send message through LangGraph
                with st.spinner("🔄 System processing response..."):
                    try:
                        response = st.session_state.learning_buddy.send_message(prompt_text)

                        if response.get("error"):
                            st.error(f"❌ Error: {response['error']}")
                        else:
                            ai_response = response.get("ai_response", "")
                            if ai_response:
                                st.session_state.chat_messages.append(
                                    {"role": "assistant", "content": ai_response})
                                if not st.session_state.muted:
                                    try:
                                        st.session_state._pending_audio_text = ai_response
                                        st.session_state._trigger_audio_playback = True
                                    except Exception as e:
                                        st.warning(f"⚠️ Audio playback error: {e}")

                            # Update session data
                            st.session_state.session_data = st.session_state.learning_buddy.get_session_state()
                            st.session_state.session_active = response.get("session_active", False)

                    except Exception as e:
                        st.error(f"❌ LangGraph communication error: {str(e)}")
                st.rerun()

            # Get text input
            prompt = st.chat_input("Type your response as therapist...")

            # Process voice input if available
            transcribed_text = None
            if audio_info and audio_info['bytes']:
                with st.spinner("Transcribing your voice..."):
                    transcribed_text = st.session_state.voice_transcriber.speech_to_text(audio_info['bytes'])

            # Handle whichever input was provided
            if transcribed_text:
                handle_interview_message(transcribed_text)
            elif prompt:
                handle_interview_message(prompt)

            #################################################
            # Audio Playback
            #################################################
            if st.session_state.get('_trigger_audio_playback') and not st.session_state.get("muted", False):
                try:
                    audio_bytes = st.session_state.learning_buddy.voice.text_to_speech(
                        st.session_state._pending_audio_text, speaker="patient"
                    )
                    st.session_state.latest_audio = audio_bytes
                except Exception as e:
                    st.warning(f"⚠️ Audio generation error: {e}")
                finally:
                    del st.session_state['_trigger_audio_playback']
                    del st.session_state['_pending_audio_text']

            if "latest_audio" in st.session_state:
                autoplay_audio(st.session_state.latest_audio)
                del st.session_state["latest_audio"]

        ############################################################
        # Display Feedback After Session-- (After Pressing End Session)
        ############################################################

        if (st.session_state.session_data and
                st.session_state.session_data.get('feedback_messages') and
                st.session_state.session_data.get('cod_stacer') and
                not st.session_state.session_active and
                not st.session_state.processing_phase):

            st.info(
                "Use the **tabs** below to review your **session and feedback** from the evaluator.\n\n"
                "Use the chat interface in the  **'Evaluator Chat'** tab to speak with the evaluator and ask specific questions about your performance. \n\n"
                "Click the **Evaluation Form** tab to see your filled in evaluation form based on your performance.\n\n "
                "Use the **Detailed Feedback** tab to see a more detailed explanation behind every score and decision on your evaluation. \n\n"
                "The **Patient Interview Transcript** tab has the full patient interview for your reference.\n\n"
                "Review the **Patient Profile** tab to see the full case details of the patient you interviewed."
            )
            # Use tabs for Feedback and CBT/MI form
            mode = st.session_state.learning_buddy.mode
            # Use tabs for Feedback and CBT/MI form
            if mode == STACER_MODE:
                tab1, tab2, tab3, tab4,  tab6 = st.tabs(["💬 Evaluator Chat", "📝 STACER Evaluation Form", " 📊 Detailed Feedback" , "🎤 Patient Interview Transcript",  "👤 Patient Profile"])
            elif mode == MI_MODE:
                tab1, tab2, tab3, tab4,  tab6 = st.tabs(["💬 Evaluator Chat", "📝 MI Evaluation Form", " 📊 Detailed Feedback" , "🎤 Patient Interview Transcript", "👤 Patient Profile"])
            elif mode == CBT_MODE:
                tab1, tab2, tab3, tab4,  tab6 = st.tabs(["💬 Evaluator Chat", "📝 CBT Evaluation Form", " 📊 Detailed Feedback" , "🎤 Patient Interview Transcript", "👤 Patient Profile"])
            
            # Tab 1: Evaluator Chat
            with tab1:
                st.markdown("### 💬 Interactive Evaluator")
                evaluator_chat_container = st.container(height=400)

                # Only allow access if evaluator session is active
                evaluator_active = (
                        st.session_state.learning_buddy
                )

                if not evaluator_active:
                    st.warning("⚠️ Evaluator session not available. Complete your interview and presentation first.")
                else:
                    # Initialize chat messages if missing
                    if "evaluator_chat_messages" not in st.session_state:
                        st.session_state.evaluator_chat_messages = []

                    # Initialize muted state if missing
                    if "evaluator_muted" not in st.session_state:
                        st.session_state.evaluator_muted = False

                    #Append the first message
                    if st.session_state.evaluator_chat_messages == []:
                        try:
                            response = st.session_state.learning_buddy.send_evaluator_message("Please Begin the Evaluation")

                            if response.get("error"):
                                st.error(f"❌ Error: {response['error']}")
                            else:
                                ai_response = response.get("ai_response", "")
                                if ai_response:
                                    st.session_state.evaluator_chat_messages.append({
                                        "role": "evaluator",
                                        "content": ai_response
                                    })
                                    save_evaluator_transcript()

                                    if not st.session_state.evaluator_muted:
                                        try:
                                            audio_bytes = st.session_state.learning_buddy.voice.text_to_speech(
                                                ai_response, speaker="evaluator")
                                            st.session_state.latest_evaluator_audio = audio_bytes
                                        except Exception as e:
                                            st.warning(f"⚠️ Audio playback error: {e}")

                        except Exception as e:
                            st.error(f"❌ Evaluator processing error: {e}")

                    with evaluator_chat_container:
                        for msg in st.session_state.evaluator_chat_messages:
                            role = msg.get("role")
                            content = msg.get("content", "")

                            if role == "user":
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                        <div style="background-color: #e3f2fd; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                            <strong>You:</strong><br>{content}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                            elif role == "evaluator":
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                                        <div style="background-color: #f1f8e9; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                            <strong>Evaluator:</strong><br>{content}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                    # Mute Toggle
                    col1, col2, col_filler, col3 = st.columns([0.06, 1, 0.5, 0.2])

                    with col1:
                        st.toggle("Muted", key="evaluator_muted", label_visibility="collapsed")

                    with col2:
                        if st.session_state.evaluator_muted:
                            st.markdown(
                                "<span style='color:red; font-weight:bold; font-size:20px;'>🔇 Audio playback is off</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<span style='color:green; font-weight:bold; font-size:20px;'>🔊 Audio playback is on</span>",
                                unsafe_allow_html=True,
                                        )
                    with col3:
                        audio_info = mic_recorder(
                            start_prompt="🎤 Speak",
                            stop_prompt="⏹️ Stop",
                            just_once=True,
                            key='mic_evaluator'
                        )

                    # NEW: Voice and Text input for Evaluator
                    # ---------------------------------------------------------------------
                    def handle_evaluator_message(prompt_text):
                        if not prompt_text:
                            return

                        st.session_state.evaluator_chat_messages.append({"role": "user", "content": prompt_text})
                        with st.spinner("💬 Evaluator preparing response..."):
                            try:
                                response = st.session_state.learning_buddy.send_evaluator_message(prompt_text)
                                if response.get("error"):
                                    st.error(f"❌ Error: {response['error']}")
                                else:
                                    ai_response = response.get("ai_response", "")
                                    if ai_response:
                                        st.session_state.evaluator_chat_messages.append(
                                            {"role": "evaluator", "content": ai_response})
                                        save_evaluator_transcript()
                                        if not st.session_state.evaluator_muted:
                                            try:
                                                audio_bytes = st.session_state.learning_buddy.voice.text_to_speech(
                                                    ai_response, speaker="evaluator")
                                                st.session_state.latest_evaluator_audio = audio_bytes
                                            except Exception as e:
                                                st.warning(f"⚠️ Audio playback error: {e}")
                            except Exception as e:
                                st.error(f"❌ Evaluator processing error: {e}")
                        st.rerun()

                    prompt = st.chat_input("Ask a question about your evaluation...")

                    transcribed_text = None
                    if audio_info and audio_info['bytes']:
                        with st.spinner("Transcribing your voice..."):
                            transcribed_text = st.session_state.voice_transcriber.speech_to_text(audio_info['bytes'])

                    if transcribed_text:
                        handle_evaluator_message(transcribed_text)
                    elif prompt:
                        handle_evaluator_message(prompt)

                    # Audio Playback
                    if 'latest_evaluator_audio' in st.session_state:
                        audio_base64 = base64.b64encode(st.session_state.latest_evaluator_audio).decode('utf-8')
                        audio_html = f'''
                                <audio autoplay>
                                  <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                                </audio>
                                '''
                        st.markdown(audio_html, unsafe_allow_html=True)

                        del st.session_state['latest_evaluator_audio']

            # Tab 2: COD CBT Form
            with tab2:
                STACER_text = st.session_state.session_data.get('cod_stacer', "")

                if mode == STACER_MODE:
                    st.markdown("### 📄 Completed COD CBT Evaluation Form")
                    st.info("This is a structured Core CBT form based on your most recent session.")
                elif mode == MI_MODE:
                    st.markdown("### 📄 Completed MI Evaluation Form")
                    st.info("This is a structured Motivational Interviewing form based on your most recent session.")
                elif mode == CBT_MODE:
                    st.markdown("### 📄 Completed CBT Evaluation Form")
                    st.info(
                        "This is a structured Cognitive Behavioural Therapy form based on your most recent session.")

                st.markdown("---")

                # Display preserving spacing and formatting
                st.markdown(STACER_text, unsafe_allow_html=True)

            # Tab 3: Feedback
            with tab3:

                feedback_full_text = st.session_state.session_data['feedback_messages']

                # Extract detailed feedback from messages
                feedback_text = ' \n '.join(
                    item.content for item in feedback_full_text if hasattr(item, 'content')
                )
                phrase = "DETAILED CLINICAL EVALUATION"
                parts = feedback_text.split(phrase)
                if len(parts) >= 3:
                    # After splitting, parts[2] is everything *after* the second occurrence
                    feedback_text = parts[2].strip()
                else:
                    # Phrase appears fewer than 2 times — keep original or handle differently
                    feedback_text = feedback_text

                feedback_container = st.container()
                with feedback_container:
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown('<div class="evaluation-card">', unsafe_allow_html=True)
                        st.markdown("#### 📋 AI Clinical Evaluation")
                        st.markdown(feedback_text)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="workflow-indicator">', unsafe_allow_html=True)
                        st.markdown("#### 🔬 AI Generated Summary")

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

                        if st.session_state.learning_buddy:
                            session_stats = st.session_state.learning_buddy.get_session_statistics()
                            st.markdown(f"**Exchanges**: {session_stats.get('total_exchanges', 0)}")

                            session_metadata = st.session_state.session_data.get('session_metadata', {})
                            if session_metadata.get('session_duration_minutes'):
                                st.markdown(f"**Duration**: {session_metadata['session_duration_minutes']:.1f} min")

                            profile_type = "Selected" if st.session_state.selected_patient_profile else "Random"
                            st.markdown(f"**Profile**: {profile_type}")

                        st.markdown('</div>', unsafe_allow_html=True)

            # Tab 4: Transcript
            with tab4:
                # Chat Interface
                chat_container = st.container(height=400)

                with chat_container:
                    # Welcome message
                    if not st.session_state.chat_messages:
                        st.chat_message("assistant").markdown(
                            "👋 *Patient simulation ready. The session is now active and ready for therapeutic interaction.*"
                        )

                    # Display chat history
                    for msg in st.session_state.chat_messages:
                        role = msg.get('role')
                        content = msg.get('content', '')

                        if role == 'user':  # Therapist (Right side)
                            st.markdown(f"""
                                                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                                    <div style="background-color: #e3f2fd; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                                        <strong>You:</strong><br>{content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)

                        elif role == 'assistant':  # Patient (Left side)
                            st.markdown(f"""
                                                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                                                    <div style="background-color: #f3e5f5; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                                        <strong>Patient:</strong><br>{content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)

            #Tab 6: Patient Profile
            with tab6:

                clinical_summary = st.session_state.session_data.get('clinical_summary', '')
                if clinical_summary:
                    st.markdown("**AI-Generated Clinical Summary:**")
                    st.info(clinical_summary)

                case_study = (
                    st.session_state.session_data
                    .get("patient_context", {})
                    .get("case_study", "")
                )
                st.info("**Patient Prompt and Profile Used During Simulation:**")
                st.markdown(case_study)
