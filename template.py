import os

# === Define the project root ===
ROOT_DIR = "AI-VOICE-CLONING-AGENT"

# === Define the folder structure ===
structure = {
    "api": [
        "__init__.py",
        "main.py",
        "auth.py",
        "utils.py"
    ],
    "core": [
        "__init__.py",
        "preprocess.py",
        "embedding.py",
        "inference.py",
        "watermark.py"
    ],
    "tests": [
        "__init__.py",
        "test_preprocess.py",
        "test_embedding.py",
        "test_inference.py",
        "test_api.py"
    ],
    ".": [  # Root-level files
        "Dockerfile",
        "requirements.txt",
        "config.py"
    ]
}

# === Helper function to create file safely ===
def create_file(path: str, content: str = ""):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"📄 Created file: {path}")
    else:
        print(f"⚠️ Skipped (already exists): {path}")

# === Create project structure ===
print(f"\n🚀 Initializing AI Voice Cloning Engine inside: {ROOT_DIR}\n")

os.makedirs(ROOT_DIR, exist_ok=True)

for folder, files in structure.items():
    folder_path = os.path.join(ROOT_DIR, folder) if folder != "." else ROOT_DIR

    # Create directories
    if folder != ".":
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 Created directory: {folder_path}")

    # Create files
    for file in files:
        filepath = os.path.join(folder_path, file)
        if file == "__init__.py":
            content = f"# {folder} package\n"
        elif file == "main.py":
            content = (
                '"""\nFastAPI main app for /api/clone endpoint.\n"""\n\n'
                'from fastapi import FastAPI\n\n'
                'app = FastAPI(title="AI Voice Cloning Engine")\n\n\n'
                '@app.get("/")\n'
                'def root():\n'
                '    return {"message": "AI Voice Cloning Engine API is running."}\n'
            )
        elif file == "requirements.txt":
            content = "fastapi\nuvicorn\nrequests\npydantic\n"
        elif file == "Dockerfile":
            content = (
                "FROM python:3.10-slim\n"
                "WORKDIR /app\n"
                "COPY . .\n"
                "RUN pip install -r requirements.txt\n"
                'CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
            )
        elif file == "config.py":
            content = (
                '"""\nConfiguration file for API keys, endpoints, and constants.\n"""\n\n'
                'API_VERSION = "v1"\n'
                'FREE_TTS_API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"\n'
                'SUPPORTED_LANGUAGES = ["en", "ur", "de"]\n'
                'DEFAULT_SAMPLE_RATE = 16000\n'
            )
        else:
            content = f"# {file} - implementation coming soon\n"

        create_file(filepath, content)

print("\n✅ Project structure successfully created inside AI-VOICE-CLONING-AGENT/")
