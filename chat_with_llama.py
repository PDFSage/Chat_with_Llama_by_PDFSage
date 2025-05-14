import sys
import subprocess
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, TextIteratorStreamer, GenerationConfig
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QSpinBox, QDoubleSpinBox
from PyQt6.QtCore import QThread, pyqtSignal

def load_model():
    """Model_loaded ∧ GPU_allocated → Ready_to_infer"""
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    return model, tokenizer

def generate_text(model, tokenizer, prompt: str) -> str:
    """Prompt ∧ Model_loaded → Generated_text"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    print(generate_text(model, tokenizer, "Hello, world!"))

def git_clone():
    """GitHub_repo ∧ Acceptance → Cloned"""
    subprocess.run(["git", "clone", "https://github.com/meta-llama/llama"], check=True)

class GenerateThread(QThread):
    """Prompt ∧ Model → Tokens_emitted"""
    token_emitted = pyqtSignal(str)
    generation_finished = pyqtSignal()

    def __init__(self, model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def run(self):
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.model.device)
        _ = self.model.generate(**inputs, generation_config=config, streamer=streamer)
        for token in streamer:
            self.token_emitted.emit(token)
        self.generation_finished.emit()

class MainWindow(QMainWindow):
    """UI ∧ Buttons → Chat_application"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat with Llama")
        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.clone_button = QPushButton("Clone Repository")
        self.clone_button.clicked.connect(self.clone_repo)
        self.layout.addWidget(self.clone_button)
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model_wrapper)
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(QLabel("Prompt"))
        self.prompt_input = QLineEdit()
        self.layout.addWidget(self.prompt_input)
        self.layout.addWidget(QLabel("Max New Tokens"))
        self.max_new_tokens_spin = QSpinBox()
        self.max_new_tokens_spin.setRange(1, 2048)
        self.max_new_tokens_spin.setValue(50)
        self.layout.addWidget(self.max_new_tokens_spin)
        self.layout.addWidget(QLabel("Temperature"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 5.0)
        self.temp_spin.setValue(1.0)
        self.temp_spin.setSingleStep(0.1)
        self.layout.addWidget(self.temp_spin)
        self.layout.addWidget(QLabel("Top K"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 10000)
        self.top_k_spin.setValue(50)
        self.layout.addWidget(self.top_k_spin)
        self.layout.addWidget(QLabel("Top P"))
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setValue(0.95)
        self.top_p_spin.setSingleStep(0.01)
        self.layout.addWidget(self.top_p_spin)
        self.layout.addWidget(QLabel("Repetition Penalty"))
        self.rep_pen_spin = QDoubleSpinBox()
        self.rep_pen_spin.setRange(1.0, 10.0)
        self.rep_pen_spin.setValue(1.0)
        self.rep_pen_spin.setSingleStep(0.1)
        self.layout.addWidget(self.rep_pen_spin)
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_text)
        self.layout.addWidget(self.generate_button)
        self.output_area = QTextEdit()
        self.layout.addWidget(self.output_area)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.model = None
        self.tokenizer = None

    def clone_repo(self):
        git_clone()

    def load_model_wrapper(self):
        self.model, self.tokenizer = load_model()

    def generate_text(self):
        if not self.model or not self.tokenizer:
            return
        self.output_area.clear()
        prompt = self.prompt_input.text()
        max_new_tokens = self.max_new_tokens_spin.value()
        temperature = self.temp_spin.value()
        top_k = self.top_k_spin.value()
        top_p = self.top_p_spin.value()
        repetition_penalty = self.rep_pen_spin.value()
        thread = GenerateThread(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty
        )
        thread.token_emitted.connect(self.append_token)
        thread.generation_finished.connect(self.finish_generation)
        thread.start()
        self.thread = thread

    def append_token(self, token):
        self.output_area.insertPlainText(token)

    def finish_generation(self):
        pass

def main_pyqt():
    """App_launched ∧ Sys_args → GUI_initialized"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
