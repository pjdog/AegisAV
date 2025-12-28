#!/bin/bash
#
# AegisAV Visual Demo Runner
#
# Runs the integrated vision system demo and opens the HTML report
#

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              🎥 AegisAV Integrated Vision System - Visual Demo              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
python -c "import PIL" 2>/dev/null || {
    echo "⚠️  Pillow not installed. Installing..."
    pip install Pillow
}

echo "✅ All dependencies ready"
echo ""

# Run demo
echo "🚀 Launching visual demonstration..."
echo ""
echo "This will:"
echo "  - Simulate 3 asset inspections with computer vision"
echo "  - Capture and analyze images for defects"
echo "  - Create anomalies and trigger re-inspections"
echo "  - Generate annotated images with bounding boxes"
echo "  - Create an interactive HTML report"
echo ""
echo "Press Enter to continue..."
read -r

python examples/demo_integrated_vision.py

# Check if report was generated
REPORT_PATH="data/vision/demo_visual/reports/demo_report.html"

if [ -f "$REPORT_PATH" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           ✅ Demo Complete!                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Generated Files:"
    echo "   • Annotated Images: data/vision/demo_visual/annotated/"
    echo "   • HTML Report: $REPORT_PATH"
    echo ""
    echo "🎥 For Video Recording:"
    echo "   1. Open the HTML report in your browser"
    echo "   2. Use OBS Studio or similar to record screen"
    echo "   3. Annotated images show defect detection in action"
    echo ""

    # Try to open in browser (Linux)
    if command -v xdg-open &> /dev/null; then
        echo "🌐 Opening report in browser..."
        xdg-open "$REPORT_PATH" 2>/dev/null || true
    fi

    echo ""
    echo "Manual open: file://$(pwd)/$REPORT_PATH"
    echo ""
else
    echo "❌ Report not generated. Check for errors above."
    exit 1
fi
