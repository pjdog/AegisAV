# AegisAV OBS Thought Bubble Overlay

Real-time visualization of agent thinking for video streaming/recording.

## Quick Start

1. **Start the AegisAV server:**
   ```bash
   python -m agent.server.main
   ```

2. **Open OBS Studio** and add a Browser Source:
   - Right-click Sources → Add → Browser
   - URL: `http://localhost:8080/overlay/`
   - Width: 1920, Height: 1080
   - Check "Shutdown source when not visible"

3. **Run a simulation** - thought bubbles will appear for each drone

## Configuration

Edit the overlay URL with query parameters:
- `?drone=drone_001` - Show only one specific drone
- `?ws=ws://192.168.1.100:8080/ws/unreal` - Custom WebSocket URL

## Debug Mode

Open browser console and run:
```javascript
simulateThinking()  // Generates fake thinking data for testing UI
```

## Architecture

```
┌─────────────────┐    WebSocket     ┌──────────────────┐
│  AegisAV Server │ ───────────────► │  OBS Browser     │
│  /ws/unreal     │   JSON events    │  Source          │
└─────────────────┘                  └──────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │  Unreal/AirSim   │
                                     │  (video layer)   │
                                     └──────────────────┘
```

The overlay receives:
- `thinking_start` - Agent begins reasoning
- `thinking_update` - Risk, considerations, options
- `critic_result` - Safety/Efficiency/Goal verdicts
- `thinking_complete` - Final decision

## Customization

The overlay uses CSS custom properties for easy theming:
- Edit colors in `<style>` section
- Adjust bubble positions in `dronePositions` object
- Modify animation timing in CSS keyframes
