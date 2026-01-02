# AegisAV Overlay Plugin for Unreal Engine

Native UMG overlay widgets for the AegisAV drone simulation system. Displays agent thinking, camera feeds, telemetry, critic evaluations, and battery status directly in the AirSim viewport.

## Features

- **Draggable Panels**: All panels can be moved by dragging their title bars
- **Collapsible**: Click the collapse button to minimize panels
- **Real-time Updates**: WebSocket connection to AegisAV backend for live data
- **Auto-reconnect**: Automatically reconnects if connection is lost
- **Scenario Asset Spawning**: Places solar panels, wind turbines, substations, and power lines from backend events

### Panels

| Panel | Description |
|-------|-------------|
| Thought Bubble | Agent's current thinking, considerations, options, and decisions |
| Camera | Live drone camera feed with FPS indicator |
| Critics | Safety, Efficiency, and Goal Alignment critic verdicts |
| Telemetry | Position, velocity, attitude, altitude, speed |
| Battery | Battery level, voltage, time remaining, warning indicators |

## Installation

### Prerequisites

- Unreal Engine 5.x
- Visual Studio 2022 with C++ game development workload
- Git

### Automated Setup (Windows)

Run the installation script from the AegisAV root:

```batch
scripts\install_airsim.bat
```

This will:
1. Clone AirSim if not present
2. Build AirSim
3. Copy the AegisAVOverlay plugin
4. Generate Visual Studio project files

### Manual Setup

1. Copy the `AegisAVOverlay` folder to your AirSim project's `Plugins` directory:
   ```
   AirSim/Unreal/Plugins/AegisAVOverlay/
   ```

2. Regenerate Visual Studio project files:
   - Right-click your `.uproject` file
   - Select "Generate Visual Studio project files"

3. Open the solution in Visual Studio and build

## Usage

### Starting the Overlay

The overlay automatically initializes when a level loads. To manually control it:

**Blueprint:**
```
Get Game Instance -> Get Subsystem (UAegisAVSubsystem)
  -> Connect To Backend
  -> Show Overlay
```

**C++:**
```cpp
if (UAegisAVSubsystem* Subsystem = GetGameInstance()->GetSubsystem<UAegisAVSubsystem>())
{
    Subsystem->ConnectToBackend(TEXT("ws://localhost:8080/ws/unreal"));
    Subsystem->ShowOverlay();
}
```

### Keyboard Shortcuts (Configure in Input Settings)

| Key | Action |
|-----|--------|
| F1 | Toggle entire overlay |
| F2 | Toggle Thought Bubble panel |
| F3 | Toggle Telemetry panel |
| F4 | Toggle Critics panel |
| F5 | Toggle Camera panel |

### Configuration

The subsystem has these configurable properties:

```cpp
// Auto-connect on subsystem initialization
bAutoConnectOnInit = false;

// Auto-show overlay when connected
bAutoShowOverlay = true;

// Default WebSocket server URL
DefaultServerURL = "ws://localhost:8080/ws/unreal";
```

### Asset Spawning (Solar Panels, Wind Turbines, Substations, Power Lines)

The overlay can spawn world assets based on backend scenario messages. Assets are placed by converting latitude/longitude/altitude into Unreal world space using a geo origin.

- By default, the first received asset becomes the origin (`UseFirstAssetAsOrigin=true`).
- To set a fixed origin, configure `OriginLatitude`, `OriginLongitude`, and `OriginAltitude` in `Config/AegisAV.ini`.
- Customize meshes and scale in `Config/AegisAV.ini` using the `AssetMesh.*` and `AssetScale.*` keys.

Anomaly markers are spawned as spheres when `spawn_anomaly_marker` messages arrive.

## Creating Custom Panels

1. Create a new Blueprint widget inheriting from `UAegisAVBasePanel`

2. Add required widget bindings:
   - `TitleBar` (Border) - The draggable header
   - `TitleText` (TextBlock) - Panel title display
   - `CollapseButton` (Button) - Collapse toggle
   - `ContentSlot` (NamedSlot) - Your content goes here

3. Add your custom widgets to the ContentSlot

4. Override `OnCollapsedStateChanged` to handle collapse animations

## WebSocket Messages

The plugin connects to `/ws/unreal` and receives these message types:

| Type | Data |
|------|------|
| `telemetry` | Position, velocity, attitude, battery |
| `thinking_update` | Agent's cognitive state and options |
| `critic_result` | Safety/Efficiency/Goal verdicts |
| `battery_update` | Detailed battery status |
| `camera_frame` | Base64-encoded PNG image |
| `anomaly_detected` | Anomaly alerts |
| `spawn_asset` | Spawn a world asset (solar panel, wind turbine, etc.) |
| `clear_assets` | Remove all spawned assets |
| `spawn_anomaly_marker` | Spawn an anomaly marker |
| `clear_anomaly_markers` | Remove all anomaly markers |

## File Structure

```
AegisAVOverlay/
├── AegisAVOverlay.uplugin
├── Source/AegisAVOverlay/
│   ├── AegisAVOverlay.Build.cs
│   ├── Public/
│   │   ├── AegisAVOverlay.h          # Module
│   │   ├── AegisAVDataTypes.h        # Data structures
│   │   ├── AegisAVWebSocketClient.h  # WebSocket client
│   │   ├── AegisAVSubsystem.h        # Game instance subsystem
│   │   └── Widgets/
│   │       ├── AegisAVBasePanel.h        # Draggable base
│   │       ├── AegisAVMasterHUD.h        # Container HUD
│   │       ├── AegisAVThoughtBubblePanel.h
│   │       ├── AegisAVCameraPanel.h
│   │       ├── AegisAVCriticPanel.h
│   │       ├── AegisAVTelemetryPanel.h
│   │       └── AegisAVBatteryPanel.h
│   └── Private/
│       └── [Implementations]
└── Content/
    └── Widgets/   # Blueprint widget designs
```

## Troubleshooting

### "WebSocket connection failed"

- Ensure the AegisAV server is running: `python -m agent.server.main`
- Check the server URL (default: `ws://localhost:8080/ws/unreal`)
- Verify no firewall is blocking the connection

### Panels not appearing

- Check that the MasterHUD Blueprint has all panels properly bound
- Verify `bAutoShowOverlay` is true or call `ShowOverlay()` manually
- Check the Output Log for any widget creation errors

### Dragging not working

- Ensure `bIsDraggable` is true on the panel
- Verify the TitleBar widget is properly bound in Blueprint
- Check that mouse events are reaching the widget

## License

Part of the AegisAV project.
