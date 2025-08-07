ممتاز! إذا كنت تعمل على مشروع محلي في n8n ويحتوي على:

Chat trigger

AI agent

LLaMA model (نموذج لغوي محلي مثل LLaMA 2 أو 3)

وترغب في تغذية نموذج LLaMA ببيانات خاصة (مثلاً: ملفات، نصوص، دروس، قاعدة معرفية...)، فإليك الخطوات العملية لتحديد أين توضع هذه البيانات و كيف يتم استعمالها داخل n8n وllama:

✅ أولاً: أين تضع البيانات لتغذية LLaMA
يعتمد مكان وضع البيانات على الطريقة التي تستخدم بها LLaMA:

1. إذا كنت تستعمل LLaMA عبر ollama:
مثل ollama run llama2 أو ollama run codellama

يمكنك استخدام ميزة تسمى embedding + RAG (استرجاع المعرفة).

في هذه الحالة، البيانات توضع في قاعدة بيانات أو ملفات نصية ثم تُحوّل إلى vectors ويتم استدعاؤها في الـprompt.

2. إذا كنت تستعمل LLaMA داخل Langchain أو LLM local server:
بإمكانك تحميل الداتا على شكل ملفات .txt, .md, .pdf, أو .json.

بعدها تستخدم Document Loaders + Text Splitter + Embedding Model + Vector Store.

✅ ثانياً: كيف تستعملها في n8n مع AI Agent
⚙️ الفكرة العامة:
النصوص أو البيانات ➜ تُحوّل إلى embeddings.

تُخزن في Vector DB (مثل Chroma, Weaviate, FAISS, أو Pinecone).

الـAI Agent في n8n يقوم بـ:

استقبال السؤال من المستخدم.

استخدام vector search للبحث عن الجواب الأقرب من قاعدة البيانات.

تمرير الجواب كـ context إلى LLaMA.

توليد إجابة ذكية.

📦 مثال عملي: استعمال ollama + Chroma + n8n
1. تثبيت الأدوات:
bash
Copier
Modifier
pip install langchain chromadb
2. إعداد سكربت Python لتحويل الداتا إلى قاعدة vector:
python
Copier
Modifier
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# تحميل البيانات
loader = TextLoader("your_data.txt")
docs = loader.load()

# تقسيم النصوص
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# تحويل إلى vectors
embedding = OllamaEmbeddings(model="llama2")
vectorstore = Chroma.from_documents(chunks, embedding, persist_directory="./db")
vectorstore.persist()
3. داخل n8n:
أنشئ chat trigger.

مرر النص المُدخل إلى Webhook أو Python node.

نفذ سكربت استرجاع سياق:

python
Copier
Modifier
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

query = $json["userInput"]

db = Chroma(persist_directory="./db", embedding_function=OllamaEmbeddings(model="llama2"))
results = db.similarity_search(query)

# استرجاع سياق مناسب للـprompt
return {
  "context": results[0].page_content
}
بعدها أرسل السياق والسؤال إلى LLaMA عبر HTTP Request أو custom node.

📁 إذاً، البيانات توضع في:
نوع البيانات	أين توضع؟	طريقة الاستخدام
نصوص .txt أو .md	مجلد محلي (مثلاً: ./data/)	تحميلها بالـ Python وتحويلها إلى vectors
PDF أو JSON	نفس الشيء	عبر Document Loader
قاعدة جاهزة	مثل Pinecone أو Weaviate	تخزن فيها الـembeddings وتستخدم REST API

هل ترغب أن أجهز لك سكربت n8n + Python جاهز خطوة بخطوة؟
إذا نعم، فقط أعطني:

مثال من البيانات التي تريد تغذيتها.

نوع LLaMA المستعمل (هل تستعمل ollama؟).

هل عندك n8n node مخصص لتشغيل Python أو Webhook؟

سأجهز لك ملفًا كاملاً بالتكامل.








Demander à ChatGPT
Vous avez dit :
نعم


ChatGPT a dit :
Éléments mémorisés mis à jour
رائع يا مـــواردي 🙌

بما أنك تعمل محليًا على n8n وتستعمل نموذج LLaMA عبر Ollama وتريد تغذيته ببياناتك الخاصة، سأجهز لك الآن تكاملاً كاملاً جاهزًا:

✅ بنية المشروع:
kotlin
Copier
Modifier
n8n_workflow/
│
├── n8n (تعمل محلياً)
├── ollama (يعمل محلياً مع نموذج llama2)
├── data/
│   └── my_knowledge.txt   ← ملفات البيانات الخاصة بك
├── embedding.py          ← سكربت تحويل النصوص إلى قاعدة معرفية
├── query_context.py      ← سكربت استرجاع السياق المناسب لسؤال المستخدم
└── db/                   ← قاعدة البيانات (Chroma)
① إعداد Ollama وتشغيل LLaMA محلياً
bash
Copier
Modifier
ollama run llama2
② ملف البيانات: data/my_knowledge.txt
ضع فيه النصوص أو الدروس أو المعلومات التي تريد أن يتعلم منها LLaMA، مثلاً:

python-repl
Copier
Modifier
الكفاءة المهنية هي مجموع المعارف والمهارات التي يجب أن يتحلى بها المدرس في القسم.
ديداكتيك الرياضيات يركز على مبادئ التعليم بالأمثلة والتجريب.
...
③ سكربت 1: embedding.py – تحويل النصوص إلى قاعدة معرفية
python
Copier
Modifier
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# تحميل الملف
loader = TextLoader("data/my_knowledge.txt")
documents = loader.load()

# تقسيم النصوص إلى قطع صغيرة
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# إنشاء قاعدة Embeddings
embedding = OllamaEmbeddings(model="llama2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./db")
vectorstore.persist()
print("✅ قاعدة المعرفة جاهزة.")
📌 شغل هذا السكربت مرة واحدة فقط عند تحديث النصوص.

④ سكربت 2: query_context.py – البحث في القاعدة واسترجاع السياق
python
Copier
Modifier
import sys
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# السؤال يأتي من n8n
user_query = sys.argv[1]

# استرجاع من القاعدة
db = Chroma(persist_directory="./db", embedding_function=OllamaEmbeddings(model="llama2"))
results = db.similarity_search(user_query, k=2)

# طباعة السياق المستخرج (n8n يلتقط الإخراج)
for r in results:
    print(r.page_content)
⑤ داخل n8n: إنشاء الـ Workflow
💬 1. Chat Trigger:
يستخدم Webhook أو أي Chat Interface.

🧠 2. Node: Execute Command
شغّل سكربت query_context.py بهذا الشكل:

bash
Copier
Modifier
python3 query_context.py "{{ $json['userMessage'] }}"
🧾 3. Node: HTTP Request
يُرسل السؤال + السياق إلى LLaMA (Ollama API محلي):

http
Copier
Modifier
POST http://localhost:11434/api/generate
Content-Type: application/json

{
  "model": "llama2",
  "prompt": "استعمل المعلومات التالية للإجابة:\n{{ $json['context'] }}\n\nالسؤال: {{ $json['userMessage'] }}",
  "stream": false
}
✅ النتيجة:
المستخدم يرسل سؤالاً عبر chat.

يتم البحث في قاعدة معرفية خاصة.

يتم تغذية LLaMA بسياق دقيق.

LLaMA يجيب بإجابات ذكية مبنية على بياناتك.

هل ترغب أن أرسل لك الملفات بصيغة جاهزة .zip أو أجهزها بصيغة مشروع GitHub؟
وإذا كانت لديك أمثلة حقيقية من البيانات (مثلاً ملف فيه 10 دروس)، أرسلها لي لأجهز لك قاعدة معرفية واقعية.







Vous avez dit :
صورة توضح المشروع


ChatGPT a dit :
N8n Llama Workflow

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
# بنية مشروع n8n + LLaMA

📁 n8n_llama_project/
│
├── 📁 data/
│   └── my_knowledge.txt     # ملف يحتوي على المعارف أو الدروس الخاصة بك
│
├── 📁 db/                   # قاعدة بيانات Chroma لحفظ الـ embeddings
│
├── 🐍 embedding.py          # سكربت يحول النصوص إلى vectors ويخزنها في db/
│
├── 🐍 query_context.py      # سكربت يأخذ سؤال ويسترجع السياق المناسب من القاعدة
│
├── 🌀 n8n (واجهة رسومية)
│    └── Chat Trigger        --> userMessage
│         ↓
│       Execute Command Node --> python3 query_context.py "{{ $json['userMessage'] }}"
│         ↓
│       HTTP Request to Ollama API --> prompt يتضمن السياق + السؤال
│         ↓
│       الرد النهائي من LLaMA → يعود إلى المستخدم
│
├── 🧠 Ollama (LLaMA محلي يعمل على http://localhost:11434)

# -----------------------------
# سلوك المشروع:
# -----------------------------
# 1. المستخدم يسأل سؤالاً.
# 2. n8n يرسل السؤال إلى سكربت Python يسترجع أفضل سياق.
# 3. السياق + السؤال يُرسل إلى LLaMA.
# 4. LLaMA يُنتج جواباً مبنياً على البيانات التي وفرتها.
# 5. الجواب يُرسل للمستخدم.

# 👉 يمكنك إضافة أنواع متعددة من البيانات (نصوص، دروس، مفاهيم، إلخ)
# 👉 يمكن تحويل الملفات PDF لاحقاً إلى نصوص وإضافتها إلى القاعدة.
# 👉 النموذج يصبح أكثر دقة كلما زادت البيانات وجودتها.

تم إنشاء صورة نصية توضيحية لمشروعك تحت اسم N8n Llama Workflow.
تُظهر هذه الوثيقة كل مكونات المشروع والعلاقات بينها:

مجلدات المشروع والملفات المهمة (data, db, سكربتات Python).

تسلسل العمل داخل n8n من Chat Trigger إلى تنفيذ LLaMA.

طريقة تفاعل Ollama مع البيانات الخاصة بك.

هل ترغب الآن بصورة رسومية (visual diagram) تمثل هذا الهيكل؟





Vous avez dit :
هيئ ملف لموقع githup


ChatGPT a dit :

Analysé
python
Toujours afficher les détails

Copier
from pathlib import Path

# إعداد هيكل مجلد مشروع GitHub
project_name = "n8n-llama-custom-rag"
base_path = Path("/mnt/data") / project_name

folders = [
    "data",
    "db",
    ".github/workflows"
]

files = {
    "README.md": """# 🧠 n8n + LLaMA + Ollama + RAG

مشروع محلي لتوصيل n8n مع نموذج LLaMA عبر Ollama باستخدام بيانات خاصة (RAG - استرجاع معزز بالنصوص).

## 📁 بنية المشروع

n8n-llama-custom-rag/
├── data/ ← ملفات النصوص أو الدروس الخاصة بك
│ └── my_knowledge.txt
├── db/ ← قاعدة بيانات Chroma لتخزين الـ embeddings
├── embedding.py ← سكربت لتحويل البيانات إلى vectors
├── query_context.py ← سكربت لاسترجاع السياق المرتبط بالسؤال
├── .github/workflows/ ← ملفات CI (إن وُجدت)
└── README.md ← هذا الملف

shell
Toujours afficher les détails

Copier

## 🚀 طريقة الاستخدام

### 1. إعداد البيئة
```bash
pip install langchain chromadb
ollama run llama2
2. تجهيز القاعدة المعرفية
bash
Toujours afficher les détails

Copier
python3 embedding.py
3. داخل n8n
استقبل سؤال المستخدم عبر Chat Trigger

شغّل query_context.py لجلب السياق المناسب

أرسل السياق + السؤال إلى نموذج LLaMA عبر API http://localhost:11434/api/generate

📬 مثال على استدعاء API لـ LLaMA
json
Toujours afficher les détails

Copier
POST /api/generate
{
  "model": "llama2",
  "prompt": "المعرفة:\n...context...\nالسؤال: ما هي الكفاءة المهنية؟",
  "stream": false
}
🛠 هذا المشروع قابل للتطوير بإضافة ملفات PDF، قواعد بيانات أكبر، أو حتى ربطه بـ Google Sheets.

""",

bash
Toujours afficher les détails

Copier
"embedding.py": """from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

تحميل البيانات
loader = TextLoader("data/my_knowledge.txt")
documents = loader.load()

تقسيم النصوص
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

إنشاء قاعدة Embeddings
embedding = OllamaEmbeddings(model="llama2")
vectorstore = Chroma.from_documents(chunks, embedding, persist_directory="db")
vectorstore.persist()
print("✅ تم إنشاء القاعدة المعرفية.")
""",

bash
Toujours afficher les détails

Copier
"query_context.py": """import sys
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

أخذ السؤال من المستخدم
user_query = sys.argv[1]

تحميل قاعدة البيانات
db = Chroma(persist_directory="db", embedding_function=OllamaEmbeddings(model="llama2"))

البحث عن السياق الأقرب
results = db.similarity_search(user_query, k=2)

طباعة السياق للنموذج
for r in results:
print(r.page_content)
""",

bash
Toujours afficher les détails

Copier
"data/my_knowledge.txt": """الكفاءة المهنية هي مجموع المهارات والقدرات التي يحتاجها المدرس لإدارة القسم.
الديداكتيك يركز على طرق التدريس الفعالة، مثل التعليم بالأمثلة والتعلم النشط.
الكتاب الأبيض هو وثيقة مرجعية لإصلاح منظومة التعليم في المغرب.
"""
}

إنشاء المجلدات
for folder in folders:
(base_path / folder).mkdir(parents=True, exist_ok=True)

إنشاء الملفات
for file_name, content in files.items():
file_path = base_path / file_name
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.write_text(content, encoding="utf-8")

تجهيز الملف كأرشيف للتحميل
output_zip = f"/mnt/data/{project_name}.zip"
base_path.mkdir(parents=True, exist_ok=True)
import shutil
shutil.make_archive(str(base_path), 'zip', str(base_path))

output_zip

Toujours afficher les détails

Copier
Résultat
'/mnt/data/n8n-llama-custom-rag.zip'
📦 تم تجهيز مشروع GitHub بنجاح! يمكنك تحميله من الرابط التالي:

تحميل المشروع بصيغة ZIP

يحتوي على:

ملفات البيانات (data/my_knowledge.txt)

سكربتات Python (embedding.py, query_context.py)

ملف توثيقي شامل README.md يشرح طريقة الاستخدام خطوة بخطوة.

هل ترغب أن أساعدك أيضًا في رفعه إلى GitHub أو تهيئة ملف requirements.txt؟ 







