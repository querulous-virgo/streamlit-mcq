import streamlit as st
import io
import base64
import time
import threading
from queue import Queue, Empty
from together import Together
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid
from datetime import datetime
import hashlib

# Configure Streamlit page
st.set_page_config(
    page_title="MCQ Image Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
.question-card {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
    background-color: #f9f9f9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.status-waiting {
    color: #6c757d;
    font-weight: bold;
}

.status-extracting {
    color: #ff6b35;
    font-weight: bold;
}

.status-solving {
    color: #007bff;
    font-weight: bold;
}

.status-completed {
    color: #28a745;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}

.queue-info {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

@dataclass
class QuestionData:
    id: str
    image: Image.Image
    filename: str
    extracted_text: Optional[str] = None
    solution: Optional[str] = None
    status: str = "waiting"  # waiting, extracting, solving, completed, error
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

class ProcessingManager:
    def __init__(self, api_key: str, max_concurrent=3):
        self.client = Together(api_key=api_key)
        self.max_concurrent = max_concurrent
        self.processing_queue = Queue()
        self.active_tasks = {}  # task_id -> thread
        self.lock = threading.Lock()
        self.running = True
        
        # Start the queue processor
        self.processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
    
    def add_question(self, question_data: QuestionData):
        """Add a question to the processing queue"""
        self.processing_queue.put(question_data)
    
    def get_active_count(self):
        """Get number of currently active processing tasks"""
        with self.lock:
            # Clean up completed threads
            completed = []
            for task_id, thread in self.active_tasks.items():
                if not thread.is_alive():
                    completed.append(task_id)
            
            for task_id in completed:
                del self.active_tasks[task_id]
            
            return len(self.active_tasks)
    
    def get_queue_size(self):
        """Get number of items waiting in queue"""
        return self.processing_queue.qsize()
    
    def _process_queue(self):
        """Background thread that processes the queue"""
        while self.running:
            try:
                # Check if we can start a new task
                if self.get_active_count() < self.max_concurrent:
                    try:
                        question_data = self.processing_queue.get(timeout=1)
                        
                        # Start processing this question
                        task_id = question_data.id
                        thread = threading.Thread(
                            target=self._process_question, 
                            args=(question_data,),
                            daemon=True
                        )
                        
                        with self.lock:
                            self.active_tasks[task_id] = thread
                        
                        thread.start()
                        
                    except Empty:
                        continue
                else:
                    time.sleep(1)  # Wait before checking again
                    
            except Exception as e:
                st.error(f"Queue processing error: {e}")
                time.sleep(1)
    
    def _process_question(self, question_data: QuestionData):
        """Process a single question (extract text and solve)"""
        try:
            # Step 1: Extract text
            question_data.status = "extracting"
            extracted_text = self._extract_text_from_image(question_data.image)
            question_data.extracted_text = extracted_text
            
            # Step 2: Solve question
            question_data.status = "solving"
            solution = self._solve_question(extracted_text)
            question_data.solution = solution
            
            # Mark as completed
            question_data.status = "completed"
            
        except Exception as e:
            question_data.status = "error"
            question_data.error_message = str(e)
        finally:
            # Remove from active tasks
            with self.lock:
                if question_data.id in self.active_tasks:
                    del self.active_tasks[question_data.id]
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using vision model"""
        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            prompt = "Extract all text from the image. Focus on the question, description, examples, and constraints. Ignore UI elements like buttons or unrelated text."
            
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[
                    {"role": "system", "content": "You are an expert OCR reader for coding platforms."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ]},
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Text extraction failed: {str(e)}")
    
    def _solve_question(self, extracted_text: str) -> str:
        """Solve the MCQ question"""
        try:
            prompt = f"""
            You are an expert problem solver. Analyze the following MCQ question and provide the correct answer.

            Question Text:
            {extracted_text}

            Instructions:
            1. First, clearly state the correct answer (A, B, C, D, etc.)
            2. Then provide a detailed explanation of why this answer is correct
            3. Explain why other options are incorrect if applicable
            4. Show any calculations or reasoning steps

            Format your response as:
            **ANSWER: [Letter]**

            **EXPLANATION:**
            [Detailed explanation here]
            """
            
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Question solving failed: {str(e)}")

def get_file_hash(uploaded_file):
    """Generate a unique hash for uploaded file"""
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    return hashlib.md5(file_content).hexdigest()

def get_status_display(status):
    """Get display text and color for status"""
    status_map = {
        'waiting': ('⏳ Waiting in Queue', 'status-waiting'),
        'extracting': ('🔄 Extracting Text', 'status-extracting'),
        'solving': ('🧠 Solving Question', 'status-solving'),
        'completed': ('✅ Completed', 'status-completed'),
        'error': ('❌ Error', 'status-error')
    }
    return status_map.get(status, ('❓ Unknown', 'status-waiting'))

def display_question_details(question_data: QuestionData, question_number: int):
    """Display detailed view of a question"""
    st.markdown(f"### Question {question_number}: {question_data.filename}")
    
    # Two columns: image and content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Image")
        st.image(question_data.image, use_column_width=True)
        
        status_text, status_class = get_status_display(question_data.status)
        st.markdown(f"**Status:** <span class='{status_class}'>{status_text}</span>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.subheader("📝 Extracted Text")
        if question_data.extracted_text:
            st.text_area("", question_data.extracted_text, height=200, disabled=True, key=f"text_{question_data.id}")
        else:
            if question_data.status == 'waiting':
                st.info("Waiting to process...")
            elif question_data.status == 'extracting':
                st.info("🔄 Extracting text from image...")
            elif question_data.status == 'error':
                st.error(f"Error: {question_data.error_message}")
            else:
                st.warning("No text extracted yet")
    
    st.markdown("---")
    st.subheader("💡 Solution")
    if question_data.solution:
        st.markdown(question_data.solution)
    else:
        if question_data.status == 'waiting':
            st.info("⏳ Waiting to process...")
        elif question_data.status == 'extracting':
            st.info("🔄 Extracting text...")
        elif question_data.status == 'solving':
            st.info("🧠 Solving question...")
        elif question_data.status == 'error':
            st.error(f"❌ Error: {question_data.error_message}")
        else:
            st.warning("No solution available")

def main():
    st.title("🧠 MCQ Image Solver")
    st.markdown("Upload images of MCQ questions and get AI-powered solutions!")
    
    # Initialize session state
    if 'questions' not in st.session_state:
        st.session_state.questions = {}
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = None
    
    # API Key configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Together AI API Key", type="password", key="api_key")
        
        if api_key and not st.session_state.processor:
            try:
                st.session_state.processor = ProcessingManager(api_key)
                st.success("✅ API Key configured!")
            except Exception as e:
                st.error(f"❌ API Key error: {e}")
        elif not api_key:
            st.warning("Please enter your Together AI API Key")
            st.info("Get your key from: https://api.together.xyz/")
        
        # Queue status
        if st.session_state.processor:
            st.header("📊 Processing Status")
            active_count = st.session_state.processor.get_active_count()
            queue_size = st.session_state.processor.get_queue_size()
            
            st.metric("Active Processing", active_count)
            st.metric("Waiting in Queue", queue_size)
            
            if active_count > 0 or queue_size > 0:
                st.info("🔄 Processing in progress...")
    
    if not st.session_state.processor:
        st.warning("⚠️ Please configure your API key in the sidebar to continue")
        return
    
    # File uploader
    st.header("📤 Upload Questions")
    uploaded_files = st.file_uploader(
        "Choose MCQ images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images of MCQ questions (PNG, JPG, JPEG)"
    )
    
    # Process newly uploaded files
    if uploaded_files:
        new_files_added = False
        for uploaded_file in uploaded_files:
            # Create unique ID based on file content
            file_hash = get_file_hash(uploaded_file)
            question_id = f"{uploaded_file.name}_{file_hash[:8]}"
            
            # Only add if not already processed
            if question_id not in st.session_state.questions:
                try:
                    image = Image.open(uploaded_file)
                    question_data = QuestionData(
                        id=question_id,
                        image=image,
                        filename=uploaded_file.name
                    )
                    
                    st.session_state.questions[question_id] = question_data
                    st.session_state.processor.add_question(question_data)
                    new_files_added = True
                    
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        if new_files_added:
            st.success(f"✅ Added {len([f for f in uploaded_files if f'{f.name}_{get_file_hash(f)[:8]}' in st.session_state.questions])} new questions to queue!")
    
    # Display questions if any exist
    if st.session_state.questions:
        st.markdown("---")
        st.header(f"📋 Questions ({len(st.session_state.questions)})")
        
        # Grid layout for question cards
        questions_list = list(st.session_state.questions.items())
        
        # Display in rows of 3
        for i in range(0, len(questions_list), 3):
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if i + j < len(questions_list):
                    question_id, question_data = questions_list[i + j]
                    question_number = i + j + 1
                    
                    with col:
                        # Create thumbnail
                        thumbnail = question_data.image.copy()
                        thumbnail.thumbnail((250, 250))
                        
                        # Display image
                        st.image(thumbnail, caption=f"Q{question_number}: {question_data.filename}")
                        
                        # Status
                        status_text, status_class = get_status_display(question_data.status)
                        st.markdown(f"<div class='{status_class}'>{status_text}</div>", 
                                   unsafe_allow_html=True)
                        
                        # View button
                        if st.button(f"View Question {question_number}", 
                                   key=f"view_{question_id}",
                                   use_container_width=True):
                            st.session_state.selected_question = (question_id, question_number)
    
    # Display selected question details
    if st.session_state.selected_question:
        question_id, question_number = st.session_state.selected_question
        if question_id in st.session_state.questions:
            st.markdown("---")
            question_data = st.session_state.questions[question_id]
            display_question_details(question_data, question_number)
            
            # Close button
            if st.button("❌ Close Details", key="close_details"):
                st.session_state.selected_question = None
                st.rerun()
    
    # Auto-refresh for live updates
    if st.session_state.questions and st.session_state.processor:
        # Check if any questions are still processing
        processing = any(q.status in ['waiting', 'extracting', 'solving'] 
                        for q in st.session_state.questions.values())
        
        if processing:
            # Add refresh button
            if st.button("🔄 Refresh Status", key="refresh"):
                st.rerun()
            
            # Auto-refresh every 3 seconds if processing
            time.sleep(0.1)  # Small delay to prevent excessive refreshing
            st.rerun()

if __name__ == "__main__":
    main()
