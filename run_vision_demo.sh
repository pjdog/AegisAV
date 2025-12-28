#!/bin/bash
#
# AegisAV Computer Vision System Demo
# Quick launcher for impressive demonstration
#

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🎥 AegisAV Computer Vision System Demo                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Run demo
echo "🚀 Launching vision system demonstration..."
echo ""
echo "This will:"
echo "  - Initialize simulated camera with defect injection"
echo "  - Run 5 asset inspections with image capture"
echo "  - Perform client-side quick detection"
echo "  - Execute server-side detailed analysis"
echo "  - Create anomalies in world model"
echo "  - Display comprehensive statistics"
echo ""
echo "Press Enter to continue..."
read -r

python examples/demo_vision_system.py

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           ✅ Demo Complete!                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📂 Images saved to: data/vision/demo/"
echo "📄 Full documentation: vision/README.md"
echo "📋 Summary: VISION_SYSTEM_SUMMARY.md"
echo ""
echo "Next steps:"
echo "  - Run tests: pytest tests/vision/ -v"
echo "  - Check images: ls -lh data/vision/demo/"
echo "  - Read docs: cat vision/README.md"
echo ""
