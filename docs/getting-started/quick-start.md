# Quick Start Guide

Get the Medical Multi-Agent Chat System running in 5 minutes!

## 🚀 Fast Track (TL;DR)

```bash
# 1. Clone & Install
git clone <repo> && cd projects/multiagent_chat
pip install -r requirements.txt

# 2. Setup LLM (choose one)
# Option A: Use LM Studio (download from lmstudio.ai, start server)
# Option B: Use Ollama (ollama serve)
# Option C: Use OpenAI (set API key in .env)

# 3. Configure
cp env.example .env
# Edit .env with your LLM_BASE_URL

# 4. Launch
poetry run python launch_a2a_agents.py

# 5. Chat
# Open browser to http://localhost:8080
```

## 📋 Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] 8GB+ RAM available
- [ ] LLM provider ready (LM Studio/Ollama/OpenAI)
- [ ] Port 9100-9102 available for agents

## Step 1: Install Dependencies

```bash
# Install dependencies with Poetry
poetry install
```

## Step 2: Start LM Studio

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Download a model (e.g., Llama 3 8B)
3. Start the server:
```bash
# In LM Studio UI: Click "Start Server"
# Or headless: lms server start
```

Your LM Studio will run on `http://localhost:1234` by default.

## Step 3: Configure Environment

```bash
# Copy template
cp env.example .env

# Edit with your settings (only if needed)
nano .env  # or vim, code, etc.
```

**Minimal `.env` configuration**:
```env
# Just this one line if using default model names!
LLM_BASE_URL=http://localhost:1234

# Optional: specify your model names if different
GENERAL_MODEL=llama-3-8b-instruct
MED_MODEL=medgemma-2
```

## Step 4: Launch the Agents

### All-in-One Launch (Easiest)

```bash
poetry run python launch_a2a_agents.py
```

You should see:
```
Starting all A2A agents...
✓ MedGemma agent ready: MedGemma Medical Assistant
✓ Clinical agent ready: Clinical Research Assistant
✓ Router agent ready: Medical Query Router
```

### Individual Launch (For Development)

```bash
# Terminal 1: MedGemma Agent
poetry run python launch_a2a_agents.py medgemma

# Terminal 2: Clinical Agent
poetry run python launch_a2a_agents.py clinical

# Terminal 3: Router Agent
poetry run python launch_a2a_agents.py router
```

## Step 5: Test Your Setup

### Using the Web UI

1. Open your browser to `http://localhost:8080`
2. Type a medical question
3. Watch agents collaborate!

### Using the Command Line

```bash
# Quick test
poetry run python test_a2a_sdk.py

# Or use curl
curl -X POST http://localhost:9100 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "route_query",
    "params": {"query": "What is diabetes?"},
    "id": 1
  }'
```

### Using Python

```python
# test_query.py
from a2a.client import AgentClient
import asyncio

async def test():
    router = AgentClient("http://localhost:9100")
    result = await router.invoke_skill(
        "route_query",
        query="What are symptoms of hypertension?"
    )
    print(f"Response: {result['response']}")
    print(f"Handled by: {result['agent_used']}")

asyncio.run(test())
```

## 🎯 Try These Example Queries

Test different agent capabilities:

### Medical Questions (→ MedGemma)
- "What are the symptoms of diabetes?"
- "How is hypertension treated?"
- "What are the side effects of metformin?"

### Clinical Data (→ Clinical Agent)
- "Show recent blood pressure readings"
- "Find patients with diabetes"
- "Get lab results from last week"

### Complex Queries (→ Multiple Agents)
- "What are normal blood pressure ranges and show me patients outside that range"
- "Explain diabetes and find diabetic patients in facility F001"

## 🔍 Verify Everything is Working

### Check Agent Health

```bash
# Test each agent's health
curl http://localhost:9100/.well-known/agent.json  # Router
curl http://localhost:9101/.well-known/agent.json  # MedGemma
curl http://localhost:9102/.well-known/agent.json  # Clinical
```

### View Logs

```bash
# If using combined launcher
# Logs appear in terminal

# If using Docker
docker logs multiagent-chat-router
docker logs multiagent-chat-medgemma
docker logs multiagent-chat-clinical
```

## ⚠️ Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| **"Connection refused"** | Check LLM server is running |
| **"Port already in use"** | Kill existing process: `lsof -ti:9100 \| xargs kill` |
| **"Module not found"** | Run `pip install -r requirements.txt` |
| **"Timeout error"** | Increase timeout: `CHAT_TIMEOUT_SECONDS=180` |
| **Slow responses** | Check GPU acceleration in LM Studio |

## 🎉 Success! What's Next?

You now have a working multi-agent medical AI system! Here's what to explore:

### Customize Your Agents
- Modify agent prompts in `server/sdk_agents/`
- Add new skills to existing agents
- Create entirely new specialist agents

### Add Data Sources
- [Configure FHIR](configuration.md#fhir-setup) for real patient data
- [Setup SQL-on-FHIR](configuration.md#spark-setup) for analytics
- Add custom data sources

### Deploy to Production
- [Docker deployment](../deployment/docker.md)
- [Monitoring setup](../deployment/monitoring.md)
- Scale with Kubernetes

### Learn More
- [Architecture Overview](../architecture/overview.md)
- [Creating New Agents](../development/creating-agents.md)
- [API Reference](../development/api-reference.md)

## Need Help?

- 📖 Check the [FAQ](../faq.md)
- 💬 Ask in [Discussions](https://github.com/...)
- 🐛 Report [Issues](https://github.com/...)

---

**Congratulations!** You're now running a sophisticated A2A protocol-based medical AI system. The agents are discovering each other, routing queries intelligently, and collaborating to provide comprehensive medical insights. 🏥🤖
