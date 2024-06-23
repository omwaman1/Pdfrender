from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained transformer model
qa_pipeline = pipeline("question-answering")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                try:
                    pdf_reader = PdfReader(file)
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()
                    return jsonify({'context': text})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
        elif 'context' in request.json and 'question' in request.json:
            context = request.json.get('context')
            question = request.json.get('question')
            if context and question:
                try:
                    result = qa_pipeline(question=question, context=context)
                    answer = result['answer']
                    return jsonify({'context': context, 'answer': answer})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            else:
                return jsonify({'error': 'Missing context or question'}), 400
    return jsonify({'message': 'Send a POST request with a file or context and question.'})

if __name__ == '__main__':
    app.run(debug=True)
