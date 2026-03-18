#!/bin/bash
# setup.sh

echo "🔧 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "📦 Pulling qwen2.5-coder:7b..."
ollama pull qwen2.5-coder:7b

echo "🐍 Installing Python deps..."
pip install -r pr_police/requirements.txt

echo "🚀 Starting Ollama..."
ollama serve &

echo "🚀 Starting FastAPI..."
nohup uvicorn pr_police.main:app --host 0.0.0.0 --port 8000 &

# --- GitHub Actions Runner Setup ---
echo "🏃 Setting up GitHub Actions Runner..."

# Download latest runner
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64-2.322.0.tar.gz
tar xzf ./actions-runner-linux-x64.tar.gz

# Prompt for inputs
read -p "Enter your GitHub repo URL (e.g. https://github.com/yourname/pr-police): " REPO_URL
read -p "Enter your runner token (from repo → Settings → Actions → Runners): " RUNNER_TOKEN

# Configure and start
./config.sh --url "$REPO_URL" --token "$RUNNER_TOKEN" --unattended
./run.sh &

echo ""
echo "✅ All done! Your machine is now a GitHub Actions runner."
echo "   Check status at: $REPO_URL/settings/actions/runners"
