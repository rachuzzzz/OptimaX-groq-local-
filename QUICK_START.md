# OptimaX v4.0 - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites Check

Run the setup checker first:
```bash
CHECK-SETUP.bat
```

This will verify you have:
- ✅ Python 3.9+
- ✅ Node.js 18+
- ✅ npm
- ✅ Required configuration files

---

## Step 1: Configure Environment

### Create `.env` file in `sql-chat-backend` folder:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/traffic_db
```

**Get Groq API Key:** https://console.groq.com/keys (Free!)

---

## Step 2: Install Dependencies

### Backend:
```bash
cd sql-chat-backend
pip install -r requirements.txt
```

### Frontend:
```bash
cd sql-chat-app
npm install
```

---

## Step 3: Launch Application

### Option A: Use the Launcher (Easiest)
Double-click: **START-OPTIMAX-V4.bat**

This will:
1. ✅ Start backend on port 8000
2. ✅ Start frontend on port 4200
3. ✅ Open browser automatically

### Option B: Manual Launch

**Terminal 1 (Backend):**
```bash
cd sql-chat-backend
python main.py
```

**Terminal 2 (Frontend):**
```bash
cd sql-chat-app
npm start
```

**Browser:**
```
http://localhost:4200
```

---

## Step 4: Test the Application

### Try These Queries:

**Greeting:**
```
Hi
```
Expected: Friendly introduction

**Data Query:**
```
Show me the top 10 states with most accidents
```
Expected: SQL query + results + chart

**Analysis:**
```
How many severe accidents in California?
```
Expected: SQL query + count + natural language answer

---

## Stop the Application

### Option A: Use the Stop Script
Double-click: **STOP-OPTIMAX.bat**

### Option B: Manual Stop
Close both terminal windows or press `Ctrl+C` in each

---

## Troubleshooting

### Backend Won't Start

**Problem:** `GROQ_API_KEY not found`
```bash
# Solution: Add to sql-chat-backend/.env
GROQ_API_KEY=your_actual_key_here
```

**Problem:** `DATABASE_URL connection failed`
```bash
# Solution: Check PostgreSQL is running
# Verify credentials in .env file
psql -U postgres -d traffic_db
```

### Frontend Won't Start

**Problem:** `npm ERR! code ENOENT`
```bash
# Solution: Install dependencies
cd sql-chat-app
npm install
```

**Problem:** Port 4200 already in use
```bash
# Solution: Kill existing Angular process
taskkill /F /IM node.exe
# Or change port in angular.json
```

### Backend Running but No Response

**Problem:** `Cannot connect to backend`
```bash
# Solution: Check backend is on port 8000
curl http://localhost:8000/health

# Should return: {"status": "healthy", ...}
```

### No SQL Query Returned

**Problem:** Response has no SQL
- Check backend logs for errors
- Verify Groq API key is valid
- Check Groq API rate limits
- Try simpler query first: "show me 5 states"

---

## Port Reference

| Service  | Port | URL                      |
|----------|------|--------------------------|
| Frontend | 4200 | http://localhost:4200    |
| Backend  | 8000 | http://localhost:8000    |
| API Docs | 8000 | http://localhost:8000/docs |

---

## Files Reference

### Batch Scripts
- **START-OPTIMAX-V4.bat** - Launch entire system
- **STOP-OPTIMAX.bat** - Stop all services
- **CHECK-SETUP.bat** - Verify setup

### Documentation
- **README.md** - Complete documentation
- **V4_ARCHITECTURE.md** - Architecture details
- **FRONTEND_INTEGRATION.md** - Integration guide
- **QUICK_START.md** - This file

### Backend
- **sql-chat-backend/main.py** - FastAPI app + Agent
- **sql-chat-backend/tools.py** - SQL + Chart tools
- **sql-chat-backend/.env** - Configuration (you create this)

### Frontend
- **sql-chat-app/** - Angular application

---

## Success Checklist

Before reporting issues, verify:

- [ ] Python 3.9+ installed: `python --version`
- [ ] Node.js 18+ installed: `node --version`
- [ ] PostgreSQL running: `psql -U postgres`
- [ ] `.env` file created in `sql-chat-backend/`
- [ ] GROQ_API_KEY set in `.env`
- [ ] DATABASE_URL set in `.env`
- [ ] Backend dependencies installed: `pip install -r requirements.txt`
- [ ] Frontend dependencies installed: `npm install`
- [ ] Backend starts: `http://localhost:8000/health` returns healthy
- [ ] Frontend starts: `http://localhost:4200` loads
- [ ] Can send "hi" and get response
- [ ] Can query data and see results

---

## First-Time Setup (Complete Workflow)

### 1. Check Prerequisites
```bash
CHECK-SETUP.bat
```

### 2. Create Database (If Needed)
```bash
psql -U postgres
CREATE DATABASE traffic_db;
\q
```

### 3. Import Data (If Needed)
```bash
# Download US Accidents dataset from Kaggle
# Import using psql COPY command
psql -U postgres -d traffic_db -f create_accidents_table.sql
```

### 4. Configure Backend
```bash
# Create sql-chat-backend/.env
notepad sql-chat-backend\.env

# Add these lines:
GROQ_API_KEY=gsk_your_key_here
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/traffic_db
```

### 5. Install Dependencies
```bash
# Backend
cd sql-chat-backend
pip install -r requirements.txt
cd ..

# Frontend
cd sql-chat-app
npm install
cd ..
```

### 6. Launch
```bash
START-OPTIMAX-V4.bat
```

### 7. Test
- Open http://localhost:4200
- Send: "hi"
- Send: "show me top 5 states"

### 8. Enjoy! 🎉

---

## Example Session

```
You: Hi

OptimaX: Hello! I'm OptimaX, your AI assistant for US traffic accident
analysis. I can analyze 7.7M accident records to help you understand
patterns by location, weather, time, and severity. What would you like
to explore?

You: Show me the top 5 states by accident count

OptimaX: Here are the top 5 states by accident count:

SQL Query:
SELECT state, COUNT(*) as accident_count
FROM us_accidents
GROUP BY state
ORDER BY accident_count DESC
LIMIT 5;

Results:
1. California (CA) - 1,741,433 accidents
2. Florida (FL) - 880,192 accidents
3. Texas (TX) - 582,837 accidents
4. Oregon (OR) - 371,279 accidents
5. New York (NY) - 357,294 accidents

[Chart appears showing bar graph]

You: What about weather conditions?

OptimaX: [Continues conversation with context...]
```

---

## Quick Reference Commands

### Start System
```bash
START-OPTIMAX-V4.bat
```

### Stop System
```bash
STOP-OPTIMAX.bat
```

### Check Health
```bash
curl http://localhost:8000/health
```

### View Logs
- Backend: Check terminal running `python main.py`
- Frontend: Check terminal running `npm start`
- Browser: Press F12 for console

### Restart Backend Only
1. Close backend terminal
2. Reopen: `cd sql-chat-backend && python main.py`

### Restart Frontend Only
1. Close frontend terminal
2. Reopen: `cd sql-chat-app && npm start`

---

## Need Help?

1. Run `CHECK-SETUP.bat` to verify configuration
2. Check `FRONTEND_INTEGRATION.md` for detailed integration info
3. Review `V4_ARCHITECTURE.md` for architecture details
4. Check backend terminal for error messages
5. Check browser console (F12) for frontend errors

---

## What's Next?

Once running successfully:
- Explore different query types
- Try the debug panel (developer mode)
- View session information
- Export chat history
- Customize system prompts (advanced)

---

**Version:** 4.0
**Status:** Ready to Use
**Last Updated:** October 2024

**Enjoy OptimaX! 🚀**
