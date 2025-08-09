import streamlit as st
import io
import base64
import time
import threading
from queue import Queue, Empty
from together import Together
from PIL import Image, ImageOps
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid
from datetime import datetime
import hashlib
import traceback

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

.error-details {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    font-family: monospace;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

@dataclass
class QuestionData:
    id: str
    image: Optional[Image.Image]
    filename: str
    image_bytes: Optional[bytes] = None
    extracted_text: Optional[str] = None
    solution: Optional[str] = None
    status: str = "waiting"  # waiting, extracting, solving, completed, error
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

def safe_image_operations(func):
    """Decorator for safe image operations with proper error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Image operation failed: {str(e)}")
            return None
    return wrapper

def load_and_validate_image(uploaded_file):
    """Safely load and validate an image file"""
    try:
        # Read file bytes
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        if len(file_bytes) == 0:
            raise ValueError("File is empty")
        
        # Try to open with PIL
        image = Image.open(io.BytesIO(file_bytes))
        
        # Verify image can be loaded
        image.verify()
        
        # Reload image for actual use (verify() closes the file)
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Validate image size
        if image.size[0] < 10 or image.size[1] < 10:
            raise ValueError("Image is too small")
        
        if image.size[0] > 10000 or image.size[1] > 10000:
            # Resize very large images
            image.thumbnail((5000, 5000), Image.Resampling.LANCZOS)
        
        return image, file_bytes
        
    except Exception as e:
        error_msg = f"Failed to load image {uploaded_file.name}: {str(e)}"
        st.error(error_msg)
        return None, None

@safe_image_operations
def create_thumbnail(image, size=(250, 250)):
    """Safely create a thumbnail"""
    if image is None:
        return None
    
    try:
        # Create a copy first
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    except Exception as e:
        # If thumbnail creation fails, try a different approach
        try:
            # Use resize instead of thumbnail
            thumbnail = image.resize(size, Image.Resampling.LANCZOS)
            return thumbnail
        except Exception as e2:
            st.error(f"Thumbnail creation failed: {str(e2)}")
            return None

class ProcessingManager:
    def __init__(self, api_key: str, max_concurrent=3):
        self.client = Together(api_key=api_key)
        self.max_concurrent = max_concurrent
        self.processing_queue = Queue()
        self.active_tasks = {}  # task_id -> thread
        self.lock = threading.Lock()
        self.running = True
        
        # Test API connection
        try:
            # Make a simple test call to validate API key
            self._test_api_connection()
        except Exception as e:
            raise Exception(f"API connection failed: {str(e)}")
        
        # Start the queue processor
        self.processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
    
    def _test_api_connection(self):
        """Test API connection with a simple call"""
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.1,
                max_tokens=10
            )
            return True
        except Exception as e:
            raise Exception(f"API test failed: {str(e)}")
    
    def add_question(self, question_data: QuestionData):
        """Add a question to the processing queue"""
        if question_data.image is None:
            question_data.status = "error"
            question_data.error_message = "Invalid image data"
            return
        
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
                print(f"Queue processing error: {e}")
                time.sleep(1)
    
    def _process_question(self, question_data: QuestionData):
        """Process a single question (extract text and solve)"""
        try:
            if question_data.image is None:
                raise Exception("No valid image data")
            
            # Step 1: Extract text
            question_data.status = "extracting"
            extracted_text = self._extract_text_from_image(question_data.image)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                raise Exception("No meaningful text extracted from image")
            
            question_data.extracted_text = extracted_text
            
            # Step 2: Solve question
            question_data.status = "solving"
            solution = self._solve_question(extracted_text)
            question_data.solution = solution
            
            # Mark as completed
            question_data.status = "completed"
            
        except Exception as e:
            question_data.status = "error"
            error_trace = traceback.format_exc()
            question_data.error_message = f"{str(e)}\n\nFull trace:\n{error_trace}"
        finally:
            # Remove from active tasks
            with self.lock:
                if question_data.id in self.active_tasks:
                    del self.active_tasks[question_data.id]
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using vision model"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure image is in RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG', optimize=True)
                img_bytes.seek(0)
                
                # Check image size
                if img_bytes.tell() > 20 * 1024 * 1024:  # 20MB limit
                    raise Exception("Image file too large (>20MB)")
                
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
                    ],
                    timeout=60
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Text extraction failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _solve_question(self, extracted_text: str) -> str:
        """Solve the MCQ question"""
        max_retries = 3
        for attempt in range(max_retries):
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
                    timeout=60
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Question solving failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

def get_file_hash(uploaded_file):
    """Generate a unique hash for uploaded file"""
    try:
        current_position = uploaded_file.tell()
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        uploaded_file.seek(current_position)  # Reset to original position
        return hashlib.md5(file_content).hexdigest()
    except Exception as e:
        # Fallback to filename + timestamp if hashing fails
        return hashlib.md5(f"{uploaded_file.name}_{time.time()}".encode()).hexdigest()

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
        if question_data.image:
            try:
                st.image(question_data.image, use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
                st.info("Image data is corrupted or invalid")
        else:
            st.error("No valid image data")
        
        status_text, status_class = get_status_display(question_data.status)
        st.markdown(f"**Status:** <span class='{status_class}'>{status_text}</span>", 
                   unsafe_allow_html=True)
        
        # Show error details if present
        if question_data.status == 'error' and question_data.error_message:
            with st.expander("🔍 Error Details"):
                st.markdown(f'<div class="error-details">{question_data.error_message}</div>', 
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
                st.error("Failed to extract text")
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
            st.error("❌ Processing failed")
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
                with st.spinner("Testing API connection..."):
                    st.session_state.processor = ProcessingManager(api_key)
                st.success("✅ API Key configured and tested!")
            except Exception as e:
                st.error(f"❌ API Configuration failed: {str(e)}")
                st.info("Please check your API key and internet connection")
        elif not api_key:
            st.warning("Please enter your Together AI API Key")
            st.info("Get your key from: https://api.together.xyz/")
        
        # Queue status
        if st.session_state.processor:
            st.header("📊 Processing Status")
            try:
                active_count = st.session_state.processor.get_active_count()
                queue_size = st.session_state.processor.get_queue_size()
                
                st.metric("Active Processing", active_count)
                st.metric("Waiting in Queue", queue_size)
                
                if active_count > 0 or queue_size > 0:
                    st.info("🔄 Processing in progress...")
            except Exception as e:
                st.error(f"Error getting status: {str(e)}")
    
    if not st.session_state.processor:
        st.warning("⚠️ Please configure your API key in the sidebar to continue")
        return
    
    # File uploader
    st.header("📤 Upload Questions")
    uploaded_files = st.file_uploader(
        "Choose MCQ images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images of MCQ questions (PNG, JPG, JPEG). Max 50 images.",
        key="file_uploader"
    )
    
    # Process newly uploaded files
    if uploaded_files:
        if len(uploaded_files) > 50:
            st.error("❌ Maximum 50 images allowed. Please select fewer images.")
            return
        
        new_files_processed = 0
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Create unique ID based on file content
                file_hash = get_file_hash(uploaded_file)
                question_id = f"{uploaded_file.name}_{file_hash[:8]}"
                
                # Only add if not already processed
                if question_id not in st.session_state.questions:
                    image, image_bytes = load_and_validate_image(uploaded_file)
                    
                    if image is not None:
                        question_data = QuestionData(
                            id=question_id,
                            image=image,
                            filename=uploaded_file.name,
                            image_bytes=image_bytes
                        )
                        
                        st.session_state.questions[question_id] = question_data
                        st.session_state.processor.add_question(question_data)
                        new_files_processed += 1
                    else:
                        errors.append(uploaded_file.name)
                        
            except Exception as e:
                errors.append(f"{uploaded_file.name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if new_files_processed > 0:
            st.success(f"✅ Successfully added {new_files_processed} questions to processing queue!")
        
        if errors:
            st.error(f"❌ Failed to process {len(errors)} files:")
            for error in errors:
                st.text(f"• {error}")
    
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
                        try:
                            # Create and display thumbnail
                            if question_data.image:
                                thumbnail = create_thumbnail(question_data.image)
                                if thumbnail:
                                    st.image(thumbnail, caption=f"Q{question_number}: {question_data.filename}")
                                else:
                                    st.error(f"Q{question_number}: Cannot display image")
                                    st.text(question_data.filename)
                            else:
                                st.error(f"Q{question_number}: No image data")
                                st.text(question_data.filename)
                            
                            # Status
                            status_text, status_class = get_status_display(question_data.status)
                            st.markdown(f"<div class='{status_class}'>{status_text}</div>", 
                                       unsafe_allow_html=True)
                            
                            # View button
                            if st.button(f"View Question {question_number}", 
                                       key=f"view_{question_id}",
                                       use_container_width=True):
                                st.session_state.selected_question = (question_id, question_number)
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error displaying Question {question_number}: {str(e)}")
    
    # Display selected question details
    if st.session_state.selected_question:
        question_id, question_number = st.session_state.selected_question
        if question_id in st.session_state.questions:
            st.markdown("---")
            question_data = st.session_state.questions[question_id]
            display_question_details(question_data, question_number)
            
            # Close button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("❌ Close Details", key="close_details", use_container_width=True):
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
            
            # Auto-refresh every 5 seconds if processing
            time.sleep(0.1)  # Small delay to prevent excessive refreshing
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.text("Please refresh the page and try again")
        if st.checkbox("Show detailed error information"):
            st.code(traceback.format_exc())
