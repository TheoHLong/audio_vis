#!/bin/bash

# Audio Visualization Switcher
# Switches between ridge visualization and heatmap visualization

FRONTEND_DIR="/Users/longtenghai/code/audio_vis/frontend"
RIDGE_FILE="$FRONTEND_DIR/main_ridge_backup.js"
HEATMAP_FILE="$FRONTEND_DIR/main.js"
TEMP_FILE="$FRONTEND_DIR/temp_main.js"

if [[ $1 == "ridge" ]]; then
    echo "Switching to ridge visualization..."
    if [ -f "$RIDGE_FILE" ]; then
        mv "$HEATMAP_FILE" "$TEMP_FILE"
        mv "$RIDGE_FILE" "$HEATMAP_FILE"
        mv "$TEMP_FILE" "$RIDGE_FILE"
        echo "✅ Switched to ridge visualization"
        echo "📝 Note: You may also want to update the HTML legend manually"
    else
        echo "❌ Ridge backup file not found"
    fi
elif [[ $1 == "heatmap" ]]; then
    echo "Switching to heatmap visualization..."
    if [ -f "$RIDGE_FILE" ]; then
        mv "$HEATMAP_FILE" "$TEMP_FILE"
        mv "$RIDGE_FILE" "$HEATMAP_FILE"
        mv "$TEMP_FILE" "$RIDGE_FILE"
        echo "✅ Switched to heatmap visualization"
    else
        echo "❌ Already using heatmap visualization"
    fi
else
    echo "Usage: $0 [ridge|heatmap]"
    echo ""
    echo "Current visualization: $(if grep -q 'drawLayerHeatmap' $HEATMAP_FILE; then echo 'heatmap'; else echo 'ridge'; fi)"
    echo ""
    echo "ridge   - Switch to original ridge plots with PCA components"
    echo "heatmap - Switch to neural activation heatmaps with jet colormap"
fi
