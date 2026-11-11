#!/bin/bash
# SaigonBot Multi-Tenant Chatbot API - Automated Setup Script (Linux / macOS)

set -e  # Exit on error

echo "=========================================="
echo "SaigonBot Multi-Tenant Chatbot API - Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Check Python version (3.10+)
echo "1. Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo -e "${RED}✗ Python 3.10+ is required (found: $(python3 --version))${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python version: $(python3 --version)"
echo ""

# 2. Check Ollama (required for embeddings regardless of LLM_PROVIDER)
echo "2. Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama is not installed${NC}"
    echo "   Ollama is required for local embeddings (mxbai-embed-large)."
    echo "   Install from: https://ollama.com"
    echo "   Or run: curl -fsSL https://ollama.com/install.sh | sh"
    read -p "   Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Ollama is installed"
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo "   Version: $OLLAMA_VERSION"
fi
echo ""

# 3. Create virtual environment (.venv)
echo "3. Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment (.venv) already exists${NC}"
    read -p "   Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        python3 -m venv .venv
        echo -e "${GREEN}✓${NC} Virtual environment created"
    else
        echo -e "${GREEN}✓${NC} Using existing virtual environment"
    fi
else
    python3 -m venv .venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi
echo ""

# 4. Activate virtual environment
echo "4. Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# 5. Upgrade pip
echo "5. Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓${NC} pip upgraded"
echo ""

# 6. Install dependencies
echo "6. Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi
echo ""

# 7. Create .env file from config.env
echo "7. Setting up configuration (.env)..."
if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
    read -p "   Overwrite with config.env? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp config.env .env
        echo -e "${GREEN}✓${NC} .env file created from config.env"
    else
        echo -e "${GREEN}✓${NC} Keeping existing .env file"
    fi
elif [ -f "config.env" ]; then
    cp config.env .env
    echo -e "${GREEN}✓${NC} .env file created from config.env"
else
    echo -e "${RED}✗ config.env not found${NC}"
    exit 1
fi

# Read active settings from .env
eval "$(python -c '
from dotenv import dotenv_values
import shlex
vals = dotenv_values(".env")
print("ACTIVE_TENANT=" + shlex.quote((vals.get("TENANT") or "chikku").strip().lower()))
print("ACTIVE_PROVIDER=" + shlex.quote((vals.get("LLM_PROVIDER") or "openai").strip().lower()))
print("ACTIVE_PORT=" + shlex.quote((vals.get("API_PORT") or "8000").strip()))
print("HAS_OPENAI_KEY=" + ("1" if (vals.get("OPENAI_API_KEY") or "").strip() else "0"))
')"
echo "   Active TENANT=$ACTIVE_TENANT | LLM_PROVIDER=$ACTIVE_PROVIDER | API_PORT=$ACTIVE_PORT"
echo ""

# 8. Check Ollama models
echo "8. Checking Ollama models..."
if command -v ollama &> /dev/null; then
    echo "   Checking for mxbai-embed-large (required for embeddings)..."
    if ollama list 2>/dev/null | grep -q "mxbai-embed-large"; then
        echo -e "${GREEN}✓${NC} mxbai-embed-large found"
    else
        echo -e "${YELLOW}⚠ mxbai-embed-large not found (or Ollama server not running)${NC}"
        read -p "   Pull mxbai-embed-large now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "   Pulling mxbai-embed-large..."
            ollama pull mxbai-embed-large
            echo -e "${GREEN}✓${NC} mxbai-embed-large pulled"
        fi
    fi

    if [ "$ACTIVE_PROVIDER" = "ollama" ]; then
        echo "   Checking for llama3:8b-instruct-q4_0 (LLM_PROVIDER=ollama)..."
        if ollama list 2>/dev/null | grep -q "llama3:8b-instruct-q4_0"; then
            echo -e "${GREEN}✓${NC} llama3:8b-instruct-q4_0 found"
        else
            read -p "   Pull llama3:8b-instruct-q4_0 now? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "   Pulling llama3:8b-instruct-q4_0 (this may take several minutes)..."
                ollama pull llama3:8b-instruct-q4_0
                echo -e "${GREEN}✓${NC} llama3:8b-instruct-q4_0 pulled"
            fi
        fi
    else
        echo -e "${GREEN}✓${NC} LLM_PROVIDER=openai — local Ollama chat model is optional"
        read -p "   (Optional) Also pull llama3:8b-instruct-q4_0 for offline fallback? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ollama pull llama3:8b-instruct-q4_0
            echo -e "${GREEN}✓${NC} llama3:8b-instruct-q4_0 pulled"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Skipping Ollama model check (Ollama not installed)${NC}"
fi
echo ""

# 9. Check tenant vector store & optional re-ingestion
echo "9. Checking vector store for TENANT=$ACTIVE_TENANT..."
if [ "$ACTIVE_TENANT" = "saigon" ]; then
    if [ -f "data/processed/menu_items.json" ]; then
        ITEM_COUNT=$(python -c "import json; print(len(json.load(open('data/processed/menu_items.json', encoding='utf-8'))))" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓${NC} Saigon menu data found (items: $ITEM_COUNT)"
    else
        echo -e "${YELLOW}⚠ Menu data not found at data/processed/menu_items.json${NC}"
        echo "   Run: python src/extract_menu.py"
    fi

    if [ -d "chroma_db" ] && [ "$(ls -A chroma_db 2>/dev/null)" ]; then
        echo -e "${GREEN}✓${NC} Pre-built chroma_db/ found"
        read -p "   Rebuild Saigon menu index now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python src/ingestion.py
            echo -e "${GREEN}✓${NC} Saigon ingestion complete"
        fi
    elif [ -f "data/processed/menu_items.json" ]; then
        read -p "   Build Saigon vector index now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python src/ingestion.py
            echo -e "${GREEN}✓${NC} Saigon ingestion complete"
        fi
    fi
else
    if [ -d "chikku_db" ] && [ "$(ls -A chikku_db 2>/dev/null)" ]; then
        echo -e "${GREEN}✓${NC} Pre-built chikku_db/ found (330 chunks shipped in repo)"
        read -p "   Rebuild Chikku KB index with --reset now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python src/ingest_chikku.py --reset
            python src/audit_chikku_coverage.py
            echo -e "${GREEN}✓${NC} Chikku ingestion & audit complete"
        fi
    else
        read -p "   Build Chikku KB index now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python src/ingest_chikku.py --reset
            python src/audit_chikku_coverage.py
            echo -e "${GREEN}✓${NC} Chikku ingestion & audit complete"
        fi
    fi
fi
echo ""

# 10. Summary
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
if [ "$ACTIVE_PROVIDER" = "openai" ] && [ "$HAS_OPENAI_KEY" = "0" ]; then
    echo -e "${YELLOW}IMPORTANT: Set OPENAI_API_KEY in .env before starting the API server!${NC}"
    echo ""
fi
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Ensure Ollama is running (needed for embeddings):"
echo "   ollama serve"
echo ""
echo "3. Start the API server:"
echo "   python run_api.py"
echo ""
echo "4. Access the API:"
echo "   - API:  http://localhost:$ACTIVE_PORT"
echo "   - Docs: http://localhost:$ACTIVE_PORT/docs"
echo ""
echo "For detailed instructions, see README.md"
echo ""
