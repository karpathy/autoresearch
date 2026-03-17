#!/bin/bash
# Docker entrypoint script for AutoResearch

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print startup banner
echo -e "${BLUE}"
echo "======================================"
echo "    AutoResearch Docker Container"
echo "======================================"
echo -e "${NC}"

# Wait for dependencies
echo -e "${YELLOW}Waiting for dependencies...${NC}"

# Wait for PostgreSQL
echo "Checking PostgreSQL..."
while ! nc -z postgres 5432 2>/dev/null; do
    echo "  PostgreSQL not ready, waiting..."
    sleep 2
done
echo -e "${GREEN}✓ PostgreSQL ready${NC}"

# Wait for Redis
echo "Checking Redis..."
while ! nc -z redis 6379 2>/dev/null; do
    echo "  Redis not ready, waiting..."
    sleep 2
done
echo -e "${GREEN}✓ Redis ready${NC}"

# Initialize database (if needed)
echo -e "${YELLOW}Initializing database...${NC}"
python3 -c "
import os
from sqlalchemy import create_engine, text

db_url = 'postgresql://autoresearch:{}@postgres/autoresearch_knowledge'.format(
    os.getenv('POSTGRES_PASSWORD', 'secure_password_change_me')
)
try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    print('${GREEN}✓ Database connection successful${NC}')
except Exception as e:
    print(f'Database initialization: {e}')
" 2>/dev/null || echo -e "${YELLOW}Database initialization skipped${NC}"

# Download data if needed
if [ ! -f "/app/data/train.bin" ]; then
    echo -e "${YELLOW}Preparing data (first run only)...${NC}"
    python3 prepare.py
    echo -e "${GREEN}✓ Data preparation complete${NC}"
fi

# Create logs directory if it doesn't exist
mkdir -p /app/logs

# Print environment info
echo -e "${BLUE}Environment:${NC}"
echo "  Python: $(python3 --version)"
echo "  GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'None')"
echo "  Mode: ${AUTORESEARCH_MODE:-docker}"
echo "  Log Level: ${LOG_LEVEL:-INFO}"
echo ""

# Execute the command
echo -e "${GREEN}Starting AutoResearch...${NC}"
echo "Command: $@"
echo ""

exec "$@"
