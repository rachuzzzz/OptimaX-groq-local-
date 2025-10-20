<div align="center">
  <img src="LOGO.png" alt="OptimaX Logo" width="200"/>

  # OptimaX v4.0 - SQL Chat Application

  **AI-Powered Natural Language to SQL with Simplified Architecture**

  [![Angular](https://img.shields.io/badge/Angular-20.3-DD0031?logo=angular)](https://angular.io/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
  [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?logo=postgresql)](https://www.postgresql.org/)
  [![Groq](https://img.shields.io/badge/Groq-Cloud_LLM-FF6B6B)](https://groq.com/)

</div>

---

## 📖 About OptimaX

OptimaX v4.0 is an intelligent SQL chat application that allows users to query traffic accident data using natural language. Built with a clean, simplified architecture using a single Groq LLM and LlamaIndex tools for fast, intelligent database interactions.

### 🎯 Key Features

- **Natural Language Queries** - Ask questions in plain English, get SQL results
- **Single LLM Architecture** - Groq llama-3.3-70b for all tasks (chat, SQL, reasoning)
- **Smart Tools** - SQL execution + Chart recommendations
- **Modern Glass UI** - Beautiful glass morphism design with smooth animations
- **Real-time Charts** - Automatic chart detection and visualization
- **Session Memory** - Multi-turn conversation context
- **Developer Friendly** - Clean, simple codebase (2 backend files)

### 📊 Dataset

- **7.7+ Million US Traffic Accident Records** (2016-2023)
- Geographic data (state, city, coordinates)
- Weather conditions (temperature, visibility, precipitation)
- Severity levels (1-4 scale)
- Road features (traffic signals, junctions, crossings)

**Data Source:** [US Accidents Dataset on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

---

## 🏗️ Architecture (v4.0 Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│                OptimaX 4.0 - Single-LLM Architecture         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Frontend (Angular 20)                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ • Glass Morphism UI                                 │    │
│  │ • Chart.js Visualization                            │    │
│  │ • Real-time Chat Interface                          │    │
│  │ • Session Management                                │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ↓ HTTP/REST                        │
│                                                               │
│  Backend (FastAPI + LlamaIndex)                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ReActAgent (Groq llama-3.3-70b)                     │    │
│  │         ↓                                            │    │
│  │ Tools:                                               │    │
│  │   • execute_sql - SQL query execution               │    │
│  │   • get_schema - Database schema info               │    │
│  │   • recommend_chart - Chart suggestions             │    │
│  │         ↓                                            │    │
│  │ Session Memory - Multi-turn Context                 │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ↓ SQL                              │
│                                                               │
│  Database (PostgreSQL)                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 7.7M+ Accident Records                              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Node.js** 18+ and npm ([Download](https://nodejs.org/))
- **Python** 3.9+ ([Download](https://www.python.org/downloads/))
- **PostgreSQL** 14+ ([Download](https://www.postgresql.org/download/))
- **Groq API Key** ([Get free key](https://console.groq.com/keys))
- **Git** ([Download](https://git-scm.com/))

### Step 1: Clone the Repository

```bash
git clone https://github.com/rachuzzzz/OptimaX-groq-local-.git
cd OptimaX
```

### Step 2: Setup Database

#### 2.1 Create PostgreSQL Database

```bash
# Open PostgreSQL command line
psql -U postgres

# Create database
CREATE DATABASE traffic_db;

# Exit psql
\q
```

#### 2.2 Load Database Schema

```bash
# Load the schema
psql -U postgres -d traffic_db -f create_accidents_table.sql
```

#### 2.3 Import Dataset

1. Download the US Accidents dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
2. Extract the CSV file
3. Import data into PostgreSQL:

```bash
# Using psql COPY command
psql -U postgres -d traffic_db

# Inside psql:
\COPY us_accidents FROM 'path/to/US_Accidents.csv' DELIMITER ',' CSV HEADER;
```

**Note:** Import may take 30-60 minutes depending on your system.

### Step 3: Setup Backend

#### 3.1 Navigate to Backend Directory

```bash
cd sql-chat-backend
```

#### 3.2 Install Python Dependencies

```bash
# Using a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3.3 Configure Environment Variables

Create a `.env` file in the `sql-chat-backend` directory:

```env
# .env file
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/traffic_db
```

**Get your Groq API key at:** https://console.groq.com/keys

#### 3.4 Start Backend Server

```bash
python main.py
```

Backend will be available at: `http://localhost:8000`

### Step 4: Setup Frontend

#### 4.1 Open New Terminal and Navigate to Frontend

```bash
cd sql-chat-app
```

#### 4.2 Install Node Dependencies

```bash
npm install
```

#### 4.3 Start Development Server

```bash
ng serve
# Or use npm
npm start
```

Frontend will be available at: `http://localhost:4200`

### Step 5: Access the Application

1. Open your browser and navigate to `http://localhost:4200`
2. Start chatting with your database!

---

## 💻 Usage Examples

### Example Queries

#### Geographic Analysis
```
"Show me the top 10 states with the most accidents"
"Which city has the most accidents?"
"Find accidents in California"
```

#### Weather Analysis
```
"Show accidents during rain"
"Which weather conditions cause the most accidents?"
```

#### Severity Analysis
```
"Count accidents by severity level"
"Show me severe accidents in New York"
```

#### Temporal Analysis
```
"Show accidents by month in 2021"
"What time of day has the most accidents?"
```

---

## 🛠️ Technology Stack

### Frontend
- **Framework:** Angular 20.3
- **Language:** TypeScript 5.9
- **Styling:** SCSS with Glass Morphism
- **Charts:** Chart.js + ng2-charts

### Backend
- **Framework:** FastAPI 0.104.1
- **Language:** Python 3.9+
- **LLM:** Groq llama-3.3-70b-versatile
- **Agent Framework:** LlamaIndex
- **Database:** PostgreSQL with SQLAlchemy

### AI/ML
- **LLM Provider:** Groq Cloud
- **Model:** llama-3.3-70b-versatile
- **Agent:** LlamaIndex ReActAgent

---

## 📁 Project Structure

```
OptimaX/
├── sql-chat-app/                    # Angular Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/
│   │   │   │   ├── chat-interface/       # Main chat UI
│   │   │   │   ├── chart-visualization/  # Chart rendering
│   │   │   │   └── loading-screen/       # Loading animation
│   │   │   └── services/
│   │   │       └── chat.service.ts       # API communication
│   │   └── styles.scss                   # Global styles
│   └── package.json
│
├── sql-chat-backend/                # FastAPI Backend
│   ├── main.py                      # FastAPI app + Agent (v4.0)
│   ├── tools.py                     # SQL + Chart tools
│   ├── requirements.txt             # Python dependencies
│   └── .env                         # Environment config
│
├── V4_ARCHITECTURE.md               # Detailed architecture docs
├── LOGO.png                         # Application logo
└── README.md                        # This file
```

---

## 🔧 Configuration

### Backend Configuration

Edit `sql-chat-backend/.env`:

```env
# Groq API
GROQ_API_KEY=your_groq_api_key_here

# Database connection
DATABASE_URL=postgresql://user:password@host:port/database
```

### Frontend Configuration

The frontend is pre-configured to connect to `http://localhost:8000` (backend).

To change, edit `sql-chat-app/src/app/services/chat.service.ts`:

```typescript
private apiUrl = 'http://localhost:8000';  // Update if needed
```

---

## 🧪 Testing

### Backend Health Check

```bash
curl http://localhost:8000/health
```

### Test Query

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show me top 5 states"}'
```

---

## 📈 API Endpoints

### Core Endpoints

- `POST /chat` - Send message, get response
- `GET /health` - Health check
- `GET /sessions` - List active sessions
- `DELETE /sessions/{id}` - Delete session
- `GET /models` - Model information
- `GET /table-info` - Database schema

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error:** `GROQ_API_KEY not found`

**Solution:**
```bash
# Add to .env file
GROQ_API_KEY=your_key_here
```

### Database Connection Failed

**Error:** `could not connect to server`

**Solution:**
- Verify PostgreSQL is running
- Check `.env` has correct credentials
- Test: `psql -U postgres -d traffic_db`

### Frontend Build Errors

**Solution:**
```bash
rm -rf node_modules package-lock.json
npm install
```

---

## 📚 Documentation

See [V4_ARCHITECTURE.md](./V4_ARCHITECTURE.md) for:
- Complete architecture overview
- Tool descriptions
- System prompt details
- Migration from v3.0
- Enhancement ideas

---

## 🚢 Deployment

### Production Build

#### Frontend
```bash
cd sql-chat-app
ng build --configuration production
```

#### Backend
```bash
cd sql-chat-backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is for educational and demonstration purposes.

**Dataset License:** [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

---

## 🙏 Acknowledgments

- **Groq** - Fast LLM inference platform
- **LlamaIndex** - Agent framework
- **Angular Team** - Frontend framework
- **FastAPI** - Python web framework
- **Sobhan Moosavi** - Dataset creator

---

## 📞 Support

### Documentation
- [V4 Architecture Guide](V4_ARCHITECTURE.md)

### Need Help?
1. Check V4_ARCHITECTURE.md
2. Review API endpoints at `/docs` (FastAPI auto-docs)
3. Check application logs
4. Test with health check endpoint

---

## 🎉 Success Indicators

You're ready when you see:
- ✅ Backend starts at port 8000
- ✅ Frontend loads at `http://localhost:4200`
- ✅ Health check returns `{"status": "healthy"}`
- ✅ Database connection successful
- ✅ Groq LLM initialized
- ✅ Charts rendering correctly

---

<div align="center">

**Version:** 4.0.0
**Status:** ✅ Production Ready
**Architecture:** Single-LLM (Groq) + LlamaIndex Tools

**Built with ❤️ using Angular, FastAPI, LlamaIndex, and Groq**

[⬆ Back to Top](#optimax-v40---sql-chat-application)

</div>
