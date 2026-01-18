#!/bin/bash
set -e

# =============================================================================
# Deploy script for Jupyter Book → DDI mirror repository
# 
# Usage: ./deploy.sh [commit message]
# 
# This script:
#   1. Builds the Jupyter Book locally
#   2. Syncs _build/html to the DDI mirror repo
#   3. Commits and pushes to GitHub Pages
# =============================================================================

# Configuration
MIRROR_REPO="../DDI"
BUILD_DIR="_build/html"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📚 Building Jupyter Book...${NC}"
jupyter-book build .

# Check if build succeeded
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ Build failed: $BUILD_DIR not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build complete${NC}"

# Check if mirror repo exists
if [ ! -d "$MIRROR_REPO" ]; then
    echo -e "${RED}❌ Mirror repository not found at $MIRROR_REPO${NC}"
    echo -e "${YELLOW}Please run:${NC}"
    echo "  cd ~/Documents/GitHub"
    echo "  git clone https://github.com/amoreira2/DDI.git"
    exit 1
fi

# Check if mirror repo is a git repository
if [ ! -d "$MIRROR_REPO/.git" ]; then
    echo -e "${RED}❌ $MIRROR_REPO is not a git repository${NC}"
    exit 1
fi

echo -e "${YELLOW}📤 Syncing to mirror repository...${NC}"

# Sync built HTML to mirror repo (delete files not in source)
rsync -av --delete \
    --exclude='.git' \
    --exclude='.nojekyll' \
    "$BUILD_DIR/" "$MIRROR_REPO/"

# Ensure .nojekyll exists (tells GitHub Pages not to process with Jekyll)
touch "$MIRROR_REPO/.nojekyll"

# Commit and push
cd "$MIRROR_REPO"

# Get commit message from argument or use default with timestamp
COMMIT_MSG="${1:-Deploy: $(date '+%Y-%m-%d %H:%M')}"

git add -A

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}ℹ️  No changes to deploy${NC}"
else
    git commit -m "$COMMIT_MSG"
    echo -e "${YELLOW}🚀 Pushing to GitHub...${NC}"
    git push origin main
    echo -e "${GREEN}✅ Deployed successfully to https://amoreira2.github.io/DDI/${NC}"
fi
