# 🔥 RAG-powered AI Chatbot on OpenShift AI 🚀

![banner](https://user-images.githubusercontent.com/placeholder/banner.png)

Welcome to this cutting-edge *RAG (Retrieval-Augmented Generation)* chatbot project!  
This app combines the best of *open-source LLMs, OpenShift AI, LangChain, and vector search* to give you fast, accurate answers from your own data — all wrapped in a sleek Flask API!

---

## ✨ Tech Stack

| 🔧 Component         | 💡 Description |
|----------------------|----------------|
| 🧠 *Mistral LLM*   | Lightweight, high-performance open-source LLM |
| 📦 *vLLM*          | Fast, efficient model serving engine |
| ☁️ *OpenShift AI*  | Scalable deployment of AI workloads |
| 🔍 *LangChain*     | RAG framework for retrieval + generation |
| 🗃️ *Elasticsearch*| Vector DB for embedding-based search |
| 🌐 *Flask*         | REST API framework to interact with the model |

---

## 🔧 Architecture

![architecture](https://user-images.githubusercontent.com/placeholder/architecture-diagram.png)

*Flow*:
1.⁠ ⁠📝 User sends a question to Flask API.
2.⁠ ⁠🔎 LangChain retrieves relevant chunks from Elasticsearch.
3.⁠ ⁠🧠 Mistral LLM (served via vLLM on OpenShift AI) generates the answer.
4.⁠ ⁠🚀 Response is returned to the user.

---
