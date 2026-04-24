#🚖 <SIKKA RAG CHATBOT>
> *AI-powered chatbot using RAG and vector search for transportation*

---

## 📌 Overview

SIKKA is a transportation app focused on Mansoura, Egypt, and its routes between surrounding governorates (Alexandria, Damietta, Dakahlia (internal), Sharqia, Gharbia). SIKKA has an intelligent chatbot that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware responses about routes between governorates in Egypt (internal or external).

It allows users to:

* Ask for routes between governorates
* Ask for the routes' prices

---

## 🚀 Features

* 💬 Interactive chatbot interface
* 📄 Document ingestion & processing
* 🔍 Semantic search using Vector Database
* ⚡ Fast responses using LLM APIs
* 🖥️ Simple UI built with Streamlit

---

## 🏗️ Tech Stack

* **Language:** Python
* **LLM:** GEMINI api
* **Framework:** Streamlit
* **Vector DB:** Pinecone
* **Other:** LangChain / custom pipeline

---

## 📁 Project Structure

```bash
├── data
      ├──SIKKA_data.xlsx   # Egypt transportation data
├── app2.py              # Streamlit UI
├── bot.py              # Backend / API logic
├── pipeline_fixed.py           # Data processing & embeddings
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
* to install the project:
git clone https://github.com/<NahlaAbdElmohsen>/<SIKKA--Rag-Chatbot>.git
cd <SIKKA--Rag-Chatbot>

* to install requirements for the project:
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory and add:

```env
OPENAI_API_KEY=your_api_key_here
PINECONE_API_KEY=your_api_key_here
```

---

## ▶️ Running the App

```bash
streamlit run app2.py
```

---

## 📸 Screenshots

![Alt Text](<img width="1257" height="489" alt="chat1" src="https://github.com/user-attachments/assets/f70f7187-4f36-4a73-8124-7713c173b395" />
)
![Alt Text](<img width="1236" height="318" alt="chat2" src="https://github.com/user-attachments/assets/8112ebc5-3af3-4ebf-8659-e09a062d81b3" />
)
![Alt Text](<img width="1226" height="551" alt="chat3" src="https://github.com/user-attachments/assets/1130d044-9922-4b79-8341-f8bdade2e203" />
)

---

## 🛠️ Future Improvements

* Add chat history
* Support multiple file types
* Improve retrieval accuracy
* build a custom travel plan for the user
* Deploy to cloud (Streamlit Cloud / Docker)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

Built in the Class of 2026, Faculty of Computers & Information, Mansoura University.

---

## 👩‍💻 Author

**<Nahla Mohamed>**

* GitHub: https://github.com/<NahlaAbdElmohsen>
