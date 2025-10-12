# OptimaX Optimized Backend - Quick Start

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.10+
- PostgreSQL database with `us_accidents` table
- Ollama running locally with these models:
  - `phi3:mini` (for intent routing)
  - `qwen2.5-coder:3b` (for SQL generation)

### Step 1: Install Dependencies
```bash
cd sql-chat-backend
pip install -r requirements.txt
```

### Step 2: Configure Environment
Create `.env` file:
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/your_database
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 3: Test Optimizations
Run the test suite to verify everything works:
```bash
python test_optimizations.py
```

Expected output:
```
✅ Heuristic routing accuracy: 80-90%
✅ Cache hits: <10ms response time
✅ Async inference: Working
```

### Step 4: Run Optimized Backend
```bash
python main_optimized.py
```

The server will start on `http://localhost:8002`

### Step 5: Test the API
```bash
# Test health
curl http://localhost:8002/health

# Test chat endpoint
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me top 10 states with accidents"}'

# View performance metrics
curl http://localhost:8002/performance
```

## 📊 Performance Monitoring

Access performance dashboard:
```bash
curl http://localhost:8002/performance | jq
```

You should see:
```json
{
  "total_requests": 100,
  "heuristic_routing": {
    "hits": 75,
    "percentage": 75.0
  },
  "cache_stats": {
    "hit_rate": 0.68
  },
  "response_times": {
    "average_ms": 650
  }
}
```

## 🎯 Key Metrics to Monitor

1. **Heuristic Hit Rate**: Should be 60-80%
   - Lower? Add more keywords to `heuristic_router.py`

2. **Cache Hit Rate**: Should reach 70-80% over time
   - Lower? Increase cache size or TTL

3. **Average Response Time**: Should be 400-800ms
   - Higher? Check Ollama performance or database queries

## 🔧 Troubleshooting

### Issue: "Ollama connection error"
**Solution:** Make sure Ollama is running:
```bash
ollama serve
ollama pull phi3:mini
ollama pull qwen2.5-coder:3b
```

### Issue: "Database connection failed"
**Solution:** Check your DATABASE_URL in `.env`:
```bash
# Test database connection
psql $DATABASE_URL -c "SELECT COUNT(*) FROM us_accidents;"
```

### Issue: "Low heuristic hit rate"
**Solution:** Customize keywords in `heuristic_router.py`:
```python
# Add domain-specific keywords
router.SQL_KEYWORDS.update({
    'your_custom_keyword',
    'another_keyword'
})
```

### Issue: "Slow response times"
**Solutions:**
1. Check Ollama model loading time:
   ```bash
   time curl -X POST http://localhost:11434/api/generate \
     -d '{"model":"phi3:mini","prompt":"test"}'
   ```

2. Monitor database query performance:
   ```sql
   EXPLAIN ANALYZE SELECT state, COUNT(*) FROM us_accidents GROUP BY state;
   ```

3. Increase cache size:
   ```python
   cache = get_query_cache(max_size=1000)  # Default: 500
   ```

## 📈 Optimization Checklist

After setup, verify these optimizations are working:

- [ ] Heuristic router classifying 60-80% of queries
- [ ] Cache hit rate improving over time (check `/performance`)
- [ ] Response times under 1 second for most queries
- [ ] No LLM timeouts or connection errors
- [ ] Database queries using proper indexes

## 🎓 Next Steps

1. **Review the full guide:** See `OPTIMIZATION_GUIDE.md` for detailed explanations

2. **Customize for your domain:**
   - Add keywords to `heuristic_router.py`
   - Adjust cache TTL in `query_cache.py`
   - Tune model parameters in `async_inference.py`

3. **Monitor in production:**
   - Set up `/performance` endpoint monitoring
   - Track response times and error rates
   - Analyze query patterns

4. **Compare with original:**
   ```bash
   # Run original backend
   python main.py

   # Run optimized backend
   python main_optimized.py

   # Compare /performance metrics
   ```

## 📞 Support

Having issues? Check:
1. Test results: `python test_optimizations.py`
2. Logs: Check console output for errors
3. Performance: `curl http://localhost:8002/performance`

## 🎉 Success Indicators

You're ready for production when:
- ✅ Test suite passes all tests
- ✅ Heuristic hit rate > 60%
- ✅ Cache hit rate > 50% (and growing)
- ✅ Average response time < 1000ms
- ✅ No connection errors to Ollama or database
- ✅ `/performance` endpoint shows healthy metrics

## 🚢 Switching from Original to Optimized

```bash
# Backup original
mv main.py main_original.py

# Use optimized version
mv main_optimized.py main.py

# Restart service
# (your service restart command here)
```

To rollback:
```bash
mv main.py main_optimized.py
mv main_original.py main.py
```

---

**Need help?** Check the full documentation in `OPTIMIZATION_GUIDE.md`
