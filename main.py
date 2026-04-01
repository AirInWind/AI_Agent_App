import sys
from PyQt5.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QLineEdit, QTextEdit, QVBoxLayout, QWidget
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import json
import os

load_dotenv() # Load environment variables from .env file

# memory settings
MEMORY_FILE = "memory.json"
MAX_MEMORY_SIZE = 150 # total user+assistant messages
MASCOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assests", "mascot")
MASCOT_FRAMES = sorted(
    [
        os.path.join(MASCOT_DIR, file_name)
        for file_name in os.listdir(MASCOT_DIR)
        if file_name.lower().endswith(".png")
    ]
) if os.path.isdir(MASCOT_DIR) else []

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(mem):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    
memory = load_memory()

@tool
def calculator(a: float, b:float) -> str:
    """Useful for performing basic calculations with numbers"""
    return f"The sum of {a} and {b} is {a + b}"

@tool
def joshuaRamirez() -> str:
    """Returns a brief bio of Joshua Ramirez when the user asks about him."""
    return "Joshua Ramirez is a Demon King Manipulator, known for his erratic behavior and sus nature."

model = ChatOpenAI(model="gpt-4o-mini")
tools = [calculator, joshuaRamirez]
agent_executor = create_react_agent(model, tools)

class AgentWorker(QObject):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, messages):
        super().__init__()
        self.messages = messages

    def run(self):
        response = ""

        try:
            for chunk in agent_executor.stream({"messages": self.messages}):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for message in chunk["agent"]["messages"]:
                        response += message.content
            self.finished.emit(response or "I couldn't generate a response.")
        except Exception as exc:
            self.failed.emit(f"Sorry, I ran into an error: {exc}")

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Agent Chat")
        self.setGeometry(100, 100, 600, 400)
        self.pending_user_input = None
        self.worker = None
        self.worker_thread = None
        self.mascot_index = 0
        self.mascot_pixmaps = [QPixmap(path) for path in MASCOT_FRAMES]

        layout = QVBoxLayout()

        self.mascot_label = QLabel()
        self.mascot_label.setAlignment(Qt.AlignCenter)
        self.mascot_label.setMinimumHeight(170)
        layout.addWidget(self.mascot_label)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your message here...")
        layout.addWidget(self.input_box)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.input_box.returnPressed.connect(self.send_button.click) # allows "Enter" as submit
        layout.addWidget(self.send_button)

        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(180)
        self.animation_timer.timeout.connect(self.show_next_mascot_frame)

        self.setLayout(layout)
        self.show_idle_mascot()
        
        if memory:
            self.chat_display.append("Loaded memory (recent):")
            for item in memory[-150:]:
                role = "You" if item.get("role") == "user" else "Assistant"
                self.chat_display.append(f"{role}: {item.get('content')}")
            self.chat_display.append("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_mascot_pixmap()

    def mascot_target_size(self):
        return QSize(
            max(self.mascot_label.width(), 180),
            max(self.mascot_label.height(), 170),
        )

    def update_mascot_pixmap(self):
        if not self.mascot_pixmaps:
            self.mascot_label.setText("Mascot images not found")
            self.mascot_label.setPixmap(QPixmap())
            return

        current_pixmap = self.mascot_pixmaps[self.mascot_index]
        scaled_pixmap = current_pixmap.scaled(
            self.mascot_target_size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.mascot_label.setText("")
        self.mascot_label.setPixmap(scaled_pixmap)

    def show_idle_mascot(self):
        self.mascot_index = 0
        self.update_mascot_pixmap()

    def show_next_mascot_frame(self):
        if not self.mascot_pixmaps:
            return
        self.mascot_index = (self.mascot_index + 1) % len(self.mascot_pixmaps)
        self.update_mascot_pixmap()

    def start_thinking_state(self):
        self.send_button.setEnabled(False)
        self.input_box.setEnabled(False)
        if len(self.mascot_pixmaps) > 1:
            self.animation_timer.start()

    def stop_thinking_state(self):
        self.animation_timer.stop()
        self.send_button.setEnabled(True)
        self.input_box.setEnabled(True)
        self.input_box.setFocus()
        self.show_idle_mascot()

    def build_messages(self, user_input):
        messages = []
        for item in memory:
            if item.get("role") == "user":
                messages.append(HumanMessage(content=item.get("content", "")))
            else:
                messages.append(AIMessage(content=item.get("content", "")))
        messages.append(HumanMessage(content=user_input))
        return messages

    def start_agent_worker(self, messages):
        self.worker_thread = QThread(self)
        self.worker = AgentWorker(messages)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_agent_response)
        self.worker.failed.connect(self.handle_agent_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def finish_message_cycle(self, response):
        self.chat_display.append(f"Assistant: {response}")

        if self.pending_user_input is not None:
            memory.append({"role": "user", "content": self.pending_user_input})
            memory.append({"role": "assistant", "content": response})

            if len(memory) > MAX_MEMORY_SIZE:
                memory[:] = memory[-MAX_MEMORY_SIZE:]
            save_memory(memory)

        self.pending_user_input = None
        self.stop_thinking_state()

    def handle_agent_response(self, response):
        self.finish_message_cycle(response)

    def handle_agent_error(self, error_message):
        self.finish_message_cycle(error_message)

    def cleanup_worker(self):
        self.worker = None
        self.worker_thread = None

    def send_message(self):
        user_input = self.input_box.text().strip()
        if not user_input or self.worker_thread is not None:
            return

        self.chat_display.append(f"You: {user_input}")
        self.input_box.clear()

        self.pending_user_input = user_input
        self.start_thinking_state()
        self.start_agent_worker(self.build_messages(user_input))
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())
