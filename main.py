import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

@tool
def calculator(a: float, b:float) -> str:
    """Useful for performing basic calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"

@tool
def joshuaRamirez() -> str:
    """Returns a brief bio of Joshua Ramirez when the user asks about him."""
    return "Joshua Ramirez is a Demon King Manipulator, known for his erratic behavior and sus nature."

model = ChatOpenAI(temperature=0)
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
        layout.addWidget(self.send_button)
        
        self.setLayout(layout)
        
    def send_message(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return
        self.chat_display.append(f"You: {user_input}")
        self.input_box.clear()
        
        response = ""
        
        for chunk in agent_executor.stream ({"messages": [HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response += message.content
        self.chat_display.append(f"Assistant: {response}")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())