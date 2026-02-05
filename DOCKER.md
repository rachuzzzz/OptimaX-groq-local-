# OptimaX Docker Setup Guide

This guide will help you run OptimaX using Docker containers for easy deployment and sharing.

## 📋 Prerequisites

- **Docker** 20.10+ ([Download](https://www.docker.com/get-started))
- **Docker Compose** 2.0+ (included with Docker Desktop)
- **Groq API Key** ([Get free key](https://console.groq.com/keys))

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/rachuzzzz/OptimaX-groq-local-.git
cd OptimaX
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Groq API Key (required)
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration (optional - defaults provided)
POSTGRES_DB=optimax_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Use internal Docker network database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/optimax_db

# Or use external database
# DATABASE_URL=postgresql://user:pass@external-host:5432/database_name
```

### 3. Start the Application

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 4. Access the Application

- **Frontend:** http://localhost:4200
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **PostgreSQL:** localhost:5432 (if using included database)

### 5. Stop the Application

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes database data)
docker-compose down -v
```

---

## 🏗️ Architecture

The Docker setup includes three services:

```
┌─────────────────────────────────────────────────┐
│                  Docker Network                  │
│                                                   │
│  ┌──────────────┐      ┌──────────────┐        │
│  │   Frontend   │      │   Backend    │        │
│  │  (Angular)   │─────▶│  (FastAPI)   │        │
│  │  Port: 4200  │      │  Port: 8000  │        │
│  └──────────────┘      └──────┬───────┘        │
│                                │                 │
│                                ▼                 │
│                       ┌──────────────┐          │
│                       │  PostgreSQL  │          │
│                       │  Port: 5432  │          │
│                       └──────────────┘          │
│                                                   │
└─────────────────────────────────────────────────┘
```

### Service Details

1. **Frontend (optimax-frontend)**
   - Angular 20 application
   - Nginx web server
   - Port: 4200 → 80 (internal)

2. **Backend (optimax-backend)**
   - FastAPI Python application
   - Groq LLM integration
   - Port: 8000

3. **PostgreSQL (optimax-postgres)** - Optional
   - PostgreSQL 14 database
   - Port: 5432
   - Persistent volume for data

---

## 📝 Usage Scenarios

### Scenario 1: Demo with Included Database

Perfect for demos and presentations. Uses the included PostgreSQL container.

```bash
# 1. Set environment variables
echo "GROQ_API_KEY=your_key_here" > .env
echo "DATABASE_URL=postgresql://postgres:postgres@postgres:5432/optimax_db" >> .env

# 2. Start all services
docker-compose up -d

# 3. Access the application
# Frontend: http://localhost:4200
# Backend: http://localhost:8000
```

The database starts empty. You can:
- Use OptimaX with any PostgreSQL database by updating DATABASE_URL
- Load sample data (see Data Loading section below)

### Scenario 2: Connect to External Database

Use OptimaX with your existing database.

```bash
# 1. Set environment variables
echo "GROQ_API_KEY=your_key_here" > .env
echo "DATABASE_URL=postgresql://user:pass@external-host:5432/your_db" >> .env

# 2. Start only backend and frontend (exclude postgres)
docker-compose up -d backend frontend

# 3. Access the application
# OptimaX will connect to your external database
```

### Scenario 3: Development with Hot Reload

For active development (not using Docker for code changes).

```bash
# Use Docker only for PostgreSQL
docker-compose up -d postgres

# Run backend and frontend locally
cd sql-chat-backend
python main.py

# In another terminal
cd sql-chat-app
ng serve
```

---

## 🗄️ Data Loading

### Option A: Use Your Own Database

Simply point DATABASE_URL to your existing PostgreSQL database. OptimaX will auto-detect the schema.

### Option B: Load Sample Data (US Traffic Accidents)

If using the included PostgreSQL container:

```bash
# 1. Download dataset from Kaggle
# https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

# 2. Create table schema
docker exec -i optimax-postgres psql -U postgres -d optimax_db < create_accidents_table.sql

# 3. Import CSV data
docker cp US_Accidents.csv optimax-postgres:/tmp/
docker exec -i optimax-postgres psql -U postgres -d optimax_db -c "\COPY us_accidents FROM '/tmp/US_Accidents.csv' DELIMITER ',' CSV HEADER;"

# 4. Verify data
docker exec -i optimax-postgres psql -U postgres -d optimax_db -c "SELECT COUNT(*) FROM us_accidents;"
```

---

## 🔧 Docker Commands Reference

### Build and Start

```bash
# Build images and start services
docker-compose up -d

# Build without cache (force rebuild)
docker-compose build --no-cache
docker-compose up -d

# Start specific service
docker-compose up -d backend
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Service Management

```bash
# Stop services
docker-compose stop

# Start stopped services
docker-compose start

# Restart services
docker-compose restart

# Restart specific service
docker-compose restart backend
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove containers and volumes (deletes database data)
docker-compose down -v

# Remove containers, volumes, and images
docker-compose down -v --rmi all
```

### Debugging

```bash
# Access backend container shell
docker exec -it optimax-backend bash

# Access frontend container shell
docker exec -it optimax-frontend sh

# Access PostgreSQL shell
docker exec -it optimax-postgres psql -U postgres -d optimax_db

# Check service health
docker-compose ps

# Inspect container
docker inspect optimax-backend
```

---

## 🐛 Troubleshooting

### Issue: Backend can't connect to database

**Error:** `could not connect to server: Connection refused`

**Solution:**
```bash
# Check if postgres is running
docker-compose ps postgres

# Check postgres logs
docker-compose logs postgres

# Verify DATABASE_URL uses service name 'postgres'
# Correct: postgresql://postgres:postgres@postgres:5432/optimax_db
# Wrong:    postgresql://postgres:postgres@localhost:5432/optimax_db
```

### Issue: Port already in use

**Error:** `Bind for 0.0.0.0:4200 failed: port is already allocated`

**Solution:**
```bash
# Option 1: Stop the process using the port
# On Windows
netstat -ano | findstr :4200
taskkill /PID <PID> /F

# On Linux/Mac
lsof -ti:4200 | xargs kill -9

# Option 2: Change port in docker-compose.yml
# Change "4200:80" to "8080:80" for example
```

### Issue: Frontend shows "Backend connection failed"

**Solution:**
```bash
# 1. Check backend is running
docker-compose ps backend

# 2. Check backend logs
docker-compose logs backend

# 3. Test backend health
curl http://localhost:8000/health

# 4. Verify frontend can reach backend
docker exec optimax-frontend wget -O- http://backend:8000/health
```

### Issue: GROQ_API_KEY not found

**Solution:**
```bash
# 1. Verify .env file exists
cat .env

# 2. Ensure GROQ_API_KEY is set
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 3. Restart services
docker-compose down
docker-compose up -d
```

### Issue: Out of disk space

**Solution:**
```bash
# Clean up unused Docker resources
docker system prune -a

# Remove old images
docker image prune -a

# Remove unused volumes
docker volume prune
```

---

## 📦 Sharing with Your Team

### Method 1: Docker Compose (Recommended)

Share the repository with docker-compose.yml:

```bash
# Your team member clones the repo
git clone https://github.com/rachuzzzz/OptimaX-groq-local-.git
cd OptimaX

# They add their API key
echo "GROQ_API_KEY=their_key_here" > .env

# Start the application
docker-compose up -d

# Access at http://localhost:4200
```

### Method 2: Docker Images (Registry)

Build and push to Docker Hub:

```bash
# Login to Docker Hub
docker login

# Tag images
docker tag optimax-backend:latest yourusername/optimax-backend:latest
docker tag optimax-frontend:latest yourusername/optimax-frontend:latest

# Push images
docker push yourusername/optimax-backend:latest
docker push yourusername/optimax-frontend:latest

# Team members pull and run
docker pull yourusername/optimax-backend:latest
docker pull yourusername/optimax-frontend:latest
docker-compose up -d
```

### Method 3: Export/Import Images

For offline sharing:

```bash
# Export images to tar files
docker save optimax-backend:latest | gzip > optimax-backend.tar.gz
docker save optimax-frontend:latest | gzip > optimax-frontend.tar.gz

# Share the .tar.gz files

# Team member imports
docker load < optimax-backend.tar.gz
docker load < optimax-frontend.tar.gz
docker-compose up -d
```

---

## 🔒 Production Deployment

### Security Checklist

```env
# Use strong passwords
POSTGRES_PASSWORD=your_strong_password_here

# Use production database
DATABASE_URL=postgresql://user:secure_pass@production-db:5432/app_db

# Never commit .env file
# Add to .gitignore
```

### Docker Compose Production Override

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    environment:
      - ENVIRONMENT=production
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 1G

  frontend:
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

Deploy:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 📊 Monitoring

### View Resource Usage

```bash
# Container stats
docker stats

# Specific container
docker stats optimax-backend

# Service logs with timestamps
docker-compose logs -f --timestamps backend
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:4200

# Database health
docker exec optimax-postgres pg_isready -U postgres
```

---

## 🎯 Best Practices

1. **Always use .env for secrets** - Never commit API keys
2. **Use volumes for persistent data** - Database data survives container restarts
3. **Monitor logs** - Use `docker-compose logs -f` during development
4. **Health checks** - Included in docker-compose.yml for service dependencies
5. **Resource limits** - Add in production to prevent resource exhaustion
6. **Regular cleanup** - Run `docker system prune` periodically
7. **Use .dockerignore** - Reduces build context and speeds up builds

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [OptimaX README](./README.md)
- [OptimaX Architecture](./V4_ARCHITECTURE.md)

---

## 🆘 Support

If you encounter issues:

1. Check this documentation
2. Review logs: `docker-compose logs -f`
3. Verify health: `docker-compose ps`
4. Check environment: `cat .env`
5. Restart services: `docker-compose restart`

---

## ✅ Build-Time Verification

The Docker build now includes a **fail-fast verification step** that ensures LlamaIndex NL-SQL is properly installed before the image can be built.

### What Gets Verified at Build Time

```
[1/3] Verifying imports...
  [OK] llama_index
  [OK] llama_index.core
  [OK] llama_index.core.query_engine
  [OK] llama_index.llms.groq
  [OK] fastapi
  [OK] uvicorn
  [OK] sqlalchemy

[2/3] Verifying NL-SQL classes...
  [OK] NLSQLTableQueryEngine
  [OK] SQLDatabase
  [OK] Settings
  [OK] Groq

[3/3] Verifying package versions...
  [OK] llama-index==0.12.0
  [OK] llama-index-core==0.12.0
  [OK] llama-index-llms-groq==0.3.0
```

If ANY check fails, the Docker build fails immediately.

### Runtime Verification Endpoints

After the container is running:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed LlamaIndex verification
curl http://localhost:8000/verify/llamaindex
```

### What Each Endpoint Proves

| Endpoint | What It Proves |
|----------|----------------|
| `/health` returns `healthy` | Backend running, DB connected, NL-SQL engine initialized |
| `/verify/llamaindex` returns `pass` | All imports work, versions correct, classes callable |

---

## 🔒 Security Features

The production Dockerfile includes:

- **Non-root user**: Container runs as `optimax` user (UID 1000)
- **No secrets baked in**: Environment variables injected at runtime
- **Minimal attack surface**: Using `python:3.11-slim` base image
- **No dev artifacts**: `.venv`, `.env`, `.vscode` excluded via `.dockerignore`

---

<div align="center">

**OptimaX Docker Setup**
**Version:** 6.1.0

Built with Docker + Docker Compose

[⬆ Back to Top](#optimax-docker-setup-guide)

</div>
