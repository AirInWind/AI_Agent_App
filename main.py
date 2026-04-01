import html
import json
import os
import sys

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from PyQt5.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

load_dotenv()  # Load environment variables from .env file

# memory settings
MEMORY_FILE = "memory.json"
MAX_MEMORY_SIZE = 150  # total user+assistant messages
MASCOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assests", "mascot")
MASCOT_FRAMES = (
    sorted(
        [
            os.path.join(MASCOT_DIR, file_name)
            for file_name in os.listdir(MASCOT_DIR)
            if file_name.lower().endswith(".png")
        ]
    )
    if os.path.isdir(MASCOT_DIR)
    else []
)


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
def calculator(a: float, b: float) -> str:
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
        self.setGeometry(120, 80, 980, 680)
        self.setMinimumSize(820, 560)
        self.setFont(QFont("Trebuchet MS", 10))

        self.pending_user_input = None
        self.worker = None
        self.worker_thread = None
        self.mascot_index = 0
        self.mascot_pixmaps = [QPixmap(path) for path in MASCOT_FRAMES]

        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(180)
        self.animation_timer.timeout.connect(self.show_next_mascot_frame)

        self.setup_ui()
        self.apply_styles()
        self.show_idle_mascot()
        self.load_chat_history()

    def setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(16)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(18)
        outer_layout.addLayout(body_layout, 1)

        self.sidebar_card = QFrame()
        self.sidebar_card.setObjectName("sidebarCard")
        self.sidebar_card.setFixedWidth(290)
        self.add_shadow(self.sidebar_card, blur_radius=32, offset_y=8)
        sidebar_layout = QVBoxLayout(self.sidebar_card)
        sidebar_layout.setContentsMargins(22, 22, 22, 22)
        sidebar_layout.setSpacing(14)

        mascot_eyebrow = QLabel("Mascot Mode")
        mascot_eyebrow.setObjectName("eyebrow")
        sidebar_layout.addWidget(mascot_eyebrow)

        mascot_title = QLabel("Agent Buddy")
        mascot_title.setObjectName("sidebarTitle")
        sidebar_layout.addWidget(mascot_title)

        mascot_subtitle = QLabel("A chat companion that comes alive while your assistant thinks.")
        mascot_subtitle.setObjectName("sidebarSubtitle")
        mascot_subtitle.setWordWrap(True)
        sidebar_layout.addWidget(mascot_subtitle)

        self.status_badge = QLabel("Ready to help")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(self.status_badge)

        self.mascot_label = QLabel()
        self.mascot_label.setAlignment(Qt.AlignCenter)
        self.mascot_label.setMinimumHeight(280)
        self.mascot_label.setObjectName("mascotDisplay")
        sidebar_layout.addWidget(self.mascot_label, 1)

        tip_card = QFrame()
        tip_card.setObjectName("tipCard")
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(14, 14, 14, 14)
        tip_layout.setSpacing(6)

        tip_title = QLabel("How it behaves")
        tip_title.setObjectName("tipTitle")
        tip_layout.addWidget(tip_title)

        self.tip_body = QLabel("Idle: calm pose. Thinking: cycles through all mascot poses like a little dance.")
        self.tip_body.setObjectName("tipBody")
        self.tip_body.setWordWrap(True)
        tip_layout.addWidget(self.tip_body)

        sidebar_layout.addWidget(tip_card)
        body_layout.addWidget(self.sidebar_card)

        self.chat_card = QFrame()
        self.chat_card.setObjectName("chatCard")
        self.add_shadow(self.chat_card, blur_radius=38, offset_y=10)
        chat_layout = QVBoxLayout(self.chat_card)
        chat_layout.setContentsMargins(22, 20, 22, 22)
        chat_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        header_copy = QVBoxLayout()
        header_copy.setSpacing(2)

        conversation_eyebrow = QLabel("Conversation")
        conversation_eyebrow.setObjectName("eyebrow")
        header_copy.addWidget(conversation_eyebrow)

        conversation_title = QLabel("Talk to your AI agent")
        conversation_title.setObjectName("chatTitle")
        header_copy.addWidget(conversation_title)

        conversation_subtitle = QLabel("Ask a question, get a reply, and watch the mascot react in real time.")
        conversation_subtitle.setObjectName("chatSubtitle")
        conversation_subtitle.setWordWrap(True)
        header_copy.addWidget(conversation_subtitle)

        header_layout.addLayout(header_copy, 1)

        self.header_pill = QLabel("Online")
        self.header_pill.setObjectName("headerPill")
        self.header_pill.setAlignment(Qt.AlignCenter)
        self.header_pill.setFixedHeight(34)
        self.header_pill.setMinimumWidth(88)
        header_layout.addWidget(self.header_pill, 0, Qt.AlignTop)

        chat_layout.addLayout(header_layout)

        self.chat_display = QTextBrowser()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.setFrameShape(QFrame.NoFrame)
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display, 1)

        self.input_card = QFrame()
        self.input_card.setObjectName("inputCard")
        input_layout = QHBoxLayout(self.input_card)
        input_layout.setContentsMargins(12, 12, 12, 12)
        input_layout.setSpacing(12)

        self.input_box = QLineEdit()
        self.input_box.setObjectName("messageInput")
        self.input_box.setPlaceholderText("Message your agent...")
        self.input_box.returnPressed.connect(self.send_button_click)
        input_layout.addWidget(self.input_box, 1)

        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setFixedHeight(46)
        self.send_button.setMinimumWidth(120)
        input_layout.addWidget(self.send_button)

        chat_layout.addWidget(self.input_card)
        body_layout.addWidget(self.chat_card, 1)

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #fff6ed, stop:0.55 #fffdf8, stop:1 #ffe9d4);
                color: #37251a;
            }
            QFrame#sidebarCard, QFrame#chatCard {
                border: 1px solid rgba(174, 122, 76, 0.18);
                border-radius: 28px;
            }
            QFrame#sidebarCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fff3e2, stop:1 #ffe2c6);
            }
            QFrame#chatCard {
                background: rgba(255, 252, 247, 0.96);
            }
            QLabel#eyebrow {
                color: #c56b2b;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
                text-transform: uppercase;
                background: transparent;
            }
            QLabel#sidebarTitle {
                font-size: 26px;
                font-weight: 700;
                color: #432918;
                background: transparent;
            }
            QLabel#sidebarSubtitle, QLabel#chatSubtitle {
                font-size: 13px;
                line-height: 1.45;
                color: #76533e;
                background: transparent;
            }
            QLabel#chatTitle {
                font-size: 24px;
                font-weight: 700;
                color: #432918;
                background: transparent;
            }
            QLabel#statusBadge {
                background: #fffaf4;
                color: #ba5c1f;
                border: 1px solid #f1c9a6;
                border-radius: 18px;
                font-size: 12px;
                font-weight: 700;
                min-height: 34px;
                padding: 0 14px;
            }
            QLabel#headerPill {
                background: #fff3df;
                color: #b25a25;
                border: 1px solid #f3d0ae;
                border-radius: 17px;
                font-size: 12px;
                font-weight: 700;
                padding: 0 12px;
            }
            QLabel#mascotDisplay {
                background: rgba(255, 255, 255, 0.42);
                border: 1px dashed rgba(176, 117, 68, 0.35);
                border-radius: 24px;
                color: #8b5e44;
            }
            QFrame#tipCard {
                background: rgba(255, 250, 242, 0.92);
                border: 1px solid rgba(208, 164, 123, 0.42);
                border-radius: 20px;
            }
            QLabel#tipTitle {
                font-size: 14px;
                font-weight: 700;
                color: #5c3a25;
                background: transparent;
            }
            QLabel#tipBody {
                font-size: 12px;
                line-height: 1.45;
                color: #775744;
                background: transparent;
            }
            QTextBrowser#chatDisplay {
                background: #fff9f3;
                border: 1px solid #f0dfcf;
                border-radius: 24px;
                padding: 18px;
                selection-background-color: #ffd7b2;
            }
            QTextBrowser#chatDisplay QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 10px 4px 10px 0;
            }
            QTextBrowser#chatDisplay QScrollBar::handle:vertical {
                background: #efc49d;
                border-radius: 6px;
                min-height: 32px;
            }
            QTextBrowser#chatDisplay QScrollBar::add-line:vertical,
            QTextBrowser#chatDisplay QScrollBar::sub-line:vertical {
                height: 0;
            }
            QFrame#inputCard {
                background: #fffaf5;
                border: 1px solid #efd9c3;
                border-radius: 22px;
            }
            QLineEdit#messageInput {
                background: transparent;
                border: none;
                font-size: 15px;
                color: #452b1d;
                padding: 4px 8px;
            }
            QLineEdit#messageInput::placeholder {
                color: #af8e78;
            }
            QPushButton#sendButton {
                background: #f4742b;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 14px;
                font-weight: 700;
                padding: 0 20px;
            }
            QPushButton#sendButton:hover {
                background: #e56720;
            }
            QPushButton#sendButton:pressed {
                background: #cf5817;
            }
            QPushButton#sendButton:disabled {
                background: #e6baa0;
                color: #fff6f0;
            }
            """
        )

    def add_shadow(self, widget, blur_radius, offset_y):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setOffset(0, offset_y)
        shadow.setColor(QColor(104, 61, 30, 45))
        widget.setGraphicsEffect(shadow)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_mascot_pixmap()

    def mascot_target_size(self):
        return QSize(
            max(self.mascot_label.width() - 24, 220),
            max(self.mascot_label.height() - 24, 240),
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

    def set_agent_status(self, is_thinking):
        if is_thinking:
            self.status_badge.setText("Thinking and dancing")
            self.header_pill.setText("Working")
            self.tip_body.setText("Your mascot is cycling through the pose frames while the assistant prepares a response.")
            self.send_button.setText("Thinking...")
        else:
            self.status_badge.setText("Ready to help")
            self.header_pill.setText("Online")
            self.tip_body.setText("Idle: calm pose. Thinking: cycles through all mascot poses like a little dance.")
            self.send_button.setText("Send")

    def start_thinking_state(self):
        self.send_button.setEnabled(False)
        self.input_box.setEnabled(False)
        self.set_agent_status(True)
        if len(self.mascot_pixmaps) > 1:
            self.animation_timer.start()

    def stop_thinking_state(self):
        self.animation_timer.stop()
        self.send_button.setEnabled(True)
        self.input_box.setEnabled(True)
        self.input_box.setFocus()
        self.show_idle_mascot()
        self.set_agent_status(False)

    def format_message_html(self, role, content):
        if role == "user":
            bubble_color = "#f4742b"
            role_color = "#ffe9d8"
            text_color = "#ffffff"
            align = "right"
        else:
            bubble_color = "#fff0df"
            role_color = "#c05d24"
            text_color = "#4b2d1d"
            align = "left"

        safe_content = html.escape(content).replace("\n", "<br>")
        safe_role = html.escape(role)

        return f"""
            <div style="margin: 8px 0 12px 0; text-align: {align};">
                <div style="
                    display: inline-block;
                    max-width: 78%;
                    background: {bubble_color};
                    color: {text_color};
                    padding: 14px 16px;
                    border-radius: 18px;
                    border: 1px solid rgba(140, 88, 51, 0.10);
                ">
                    <div style="
                        font-size: 11px;
                        font-weight: 700;
                        letter-spacing: 0.5px;
                        color: {role_color};
                        margin-bottom: 6px;
                        text-transform: uppercase;
                    ">{safe_role}</div>
                    <div style="
                        font-size: 14px;
                        line-height: 1.45;
                    ">{safe_content}</div>
                </div>
            </div>
        """

    def append_chat_message(self, role, content):
        self.chat_display.append(self.format_message_html(role, content))
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def load_chat_history(self):
        if not memory:
            self.append_chat_message(
                "Assistant",
                "Hi, I'm ready whenever you are. Ask me anything and I'll dance while I think.",
            )
            return

        for item in memory[-MAX_MEMORY_SIZE:]:
            role = "You" if item.get("role") == "user" else "Assistant"
            self.append_chat_message(role, item.get("content", ""))

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
        self.append_chat_message("Assistant", response)

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

    def send_button_click(self):
        if self.send_button.isEnabled():
            self.send_button.click()

    def send_message(self):
        user_input = self.input_box.text().strip()
        if not user_input or self.worker_thread is not None:
            return

        self.append_chat_message("You", user_input)
        self.input_box.clear()

        self.pending_user_input = user_input
        self.start_thinking_state()
        self.start_agent_worker(self.build_messages(user_input))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())
