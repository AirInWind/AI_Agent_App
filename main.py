import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
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
MAX_MEMORY_SIZE = 50 # total user+assistant messages

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
    
class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Agent Chat")
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()
        
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
        
        self.setLayout(layout)
        
        if memory:
            self.chat_display.append("Loaded memory (recent):")
            for item in memory[-10:]:
                role = "You" if item.get("role") == "user" else "Assistant"
                self.chat_display.append(f"{role}: {item.get('content')}")
            self.chat_display.append("")
        
    def send_message(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return
        self.chat_display.append(f"You: {user_input}")
        self.input_box.clear()
        
        # build message list from memory + current user message
        messages = []
        for item in memory:
            if item.get("role") == "user":
                messages.append(HumanMessage(content=item.get("content","")))
            else:
                messages.append(AIMessage(content=item.get("content","")))
        messages.append(HumanMessage(content=user_input))
        
        response = ""
        
        for chunk in agent_executor.stream ({"messages": [HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response += message.content
        self.chat_display.append(f"Assistant: {response}")
        
        memory.append({"role": "user", "content": user_input})
        memory.append({"role": "assistant", "content": response})
        
        if len(memory) > MAX_MEMORY_SIZE:
            memory[:] = memory[-MAX_MEMORY_SIZE:]
        save_memory(memory)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())