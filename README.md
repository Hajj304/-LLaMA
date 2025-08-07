إليك محتوى ملف `README.md` الموجود داخل المشروع المضغوط، جاهز للنشر على GitHub:

---

```markdown
# 🧠 n8n + LLaMA + Ollama + RAG

مشروع محلي يربط بين [n8n](https://n8n.io) و [LLaMA](https://ollama.com/library/llama2) عبر [Ollama](https://ollama.com)، ويستخدم تقنيات RAG (استرجاع معزز بالسياق) لتوفير إجابات ذكية مبنية على بياناتك الخاصة.

---

## 📁 بنية المشروع

```

n8n-llama-custom-rag/
├── data/                  ← ملفات النصوص أو الدروس الخاصة بك
│   └── my\_knowledge.txt
├── db/                    ← قاعدة بيانات Chroma لتخزين الـ embeddings
├── embedding.py           ← سكربت لتحويل البيانات إلى vectors
├── query\_context.py       ← سكربت لاسترجاع السياق المرتبط بالسؤال
├── .github/workflows/     ← ملفات CI (إن وُجدت)
└── README.md              ← هذا الملف

````

---

## 🚀 طريقة الاستخدام

### 1. إعداد البيئة

```bash
pip install langchain chromadb
ollama run llama2
````

### 2. تجهيز قاعدة المعرفة

```bash
python3 embedding.py
```

سيقوم السكربت بقراءة `data/my_knowledge.txt`، تقسيمه، وتحويله إلى قاعدة بيانات vectors في مجلد `db/`.

---

### 3. استخدام في n8n

داخل n8n:

* **Chat Trigger** يستقبل سؤال المستخدم.
* Node من نوع **Execute Command** يشغّل:

  ```bash
  python3 query_context.py "{{ $json['userMessage'] }}"
  ```
* النتيجة تُمرر إلى **HTTP Request Node** نحو Ollama:

```json
POST http://localhost:11434/api/generate
{
  "model": "llama2",
  "prompt": "المعرفة:\n{{ السياق }}\n\nالسؤال: {{ السؤال }}",
  "stream": false
}
```

---

## 📬 مثال عملي

**my\_knowledge.txt**:

```
الكفاءة المهنية هي مجموع المهارات والقدرات التي يحتاجها المدرس لإدارة القسم.
الديداكتيك يركز على طرق التدريس الفعالة، مثل التعليم بالأمثلة والتعلم النشط.
```

**سؤال المستخدم**:

> ما هي الكفاءة المهنية؟

**النموذج LLaMA سيرد** بإجابة مستندة على هذه المعارف.

---

## 💡 أفكار للتوسعة

* دعم ملفات PDF وتحويلها إلى نصوص.
* استخدام واجهة دردشة حية React + n8n.
* ربط بـ Google Sheets لتحديث البيانات تلقائيًا.
* استخدام Vector DB سحابي (Pinecone، Weaviate...).

---

## 📜 رخصة

مشروع مفتوح المصدر لغرض تعليمي.

```

هل ترغب أن أترجمه أيضًا للإنجليزية ليكون ثنائي اللغة على GitHub؟
```


