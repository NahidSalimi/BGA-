import os
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, session, redirect, url_for

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and process your document once at startup
loader = TextLoader("C:/Users/HP-PC/PycharmProjects/pythonProject/company_faq.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

# Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
    retriever=vectorstore.as_retriever()
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")  # set in .env ideally

# Pretty chat HTML template with Bootstrap 5
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>Brave FAQ Chatbot</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    <style>
        body {
            background: #f5f8fa;
        }
        .chat-container {
            max-width: 700px;
            margin: 40px auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgb(0 0 0 / 0.1);
            padding: 30px;
            display: flex;
            flex-direction: column;
            height: 80vh;
        }
        .chat-box {
            flex-grow: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 15px;
            background: #fff;
        }
        .message {
            padding: 10px 15px;
            border-radius: 20px;
            margin-bottom: 10px;
            max-width: 75%;
            word-wrap: break-word;
        }
        .user-message {
            background: #E4FF00;
            color: #333;
            align-self: flex-end;
            border-bottom-right-radius: 0;
        }
        .bot-message {
            background: #ffffff;
            color: #1a1a1a;
            align-self: flex-start;
            border-bottom-left-radius: 0;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
        }
        form {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 12px 15px;
            border-radius: 25px;
            border: 1px solid #B3CC00;
            font-size: 1rem;
        }
        button {
            padding: 12px 25px;
            border-radius: 25px;
            background: #E4FF00;
            border: none;
            color: #333;
            font-weight: bold;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        button:hover {
            background: #B3CC00;
            color: white;
        }
    </style>
</head>
<body>
    <div class="chat-container d-flex flex-column">
        <h2 class="mb-3 text-center">💚Brave Internal Chatbot</h2>
        <div class="chat-box" id="chat-box">
            {% for entry in chat_history %}
                {% if entry.user %}
                <div class="message user-message">{{ entry.user }}</div>
                {% endif %}
                {% if entry.bot %}
                <div class="message bot-message">{{ entry.bot }}</div>
                {% endif %}
            {% endfor %}
        </div>
        <form method="POST" action="{{ url_for('chat') }}">
            <input type="text" name="query" autocomplete="off" placeholder="🧠 Your Question..." required autofocus />
            <button type="submit">Ask</button>
        </form>
    </div>

    <script>
        // Scroll to bottom on page load to show latest messages
        const chatBox = document.getElementById('chat-box');
        chatBox.scrollTop = chatBox.scrollHeight;
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    session.clear()  # Start fresh on root access
    return redirect(url_for("chat"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form.get("query")
        if query:
            chat_history = session["chat_history"]
            chat_history.append({"user": query, "bot": None})
            session["chat_history"] = chat_history

            # Use invoke() and extract 'result' key if dict
            result = qa.invoke(query)
            answer = result.get('result') if isinstance(result, dict) else str(result)

            chat_history[-1]["bot"] = answer
            session["chat_history"] = chat_history

    return render_template_string(HTML_TEMPLATE, chat_history=session.get("chat_history", []))

if __name__ == "__main__":
    app.run(debug=True)


