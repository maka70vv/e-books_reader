import datetime
import os
import uuid

from flask import Flask, request, render_template, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename

from parsers import extract_text_from_pdf, extract_text_from_fb2
from tts import generate_speech, audio_data

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["book"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif filename.endswith(".fb2"):
                text = extract_text_from_fb2(file_path)
            else:
                return "Unsupported format", 400

            session["text"] = text
            return redirect(url_for("read_book"))

    return render_template("index.html")

@app.route("/read", methods=["GET", "POST"])
def read_book():
    text = session.get("text", "")
    total_pages = len(text) // 1500 + 1
    audio_url = None

    if request.method == "POST":
        range_type = request.form.get("range_type")
        if range_type == "all":
            selected_text = text
        else:
            start = int(request.form.get("start", 1)) - 1
            end = int(request.form.get("end", 1))
            chars_per_page = 1500
            selected_text = text[start * chars_per_page:end * chars_per_page]

        # Генерация уникального имени
        unique_id = str(uuid.uuid4())
        filename = f"audiobook_{unique_id}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        generate_speech(selected_text, audio_data, output_path)
        audio_url = url_for("serve_audio", filename=filename)

    return render_template("read.html", text=text[:5000], pages=total_pages, audio_url=audio_url)

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="audio/wav")



if __name__ == "__main__":
    app.run(debug=True)
