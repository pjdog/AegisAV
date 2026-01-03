# AegisAV Design System & Style Guide

## 1. Design Philosophy
**"Sleek, Autonomous, Transparent."**
The AegisAV aesthetic combines the rugged utility of industrial inspection with the sophisticated intelligence of agentic AI. The interface should feel like a "CommandOS" — dark, data-dense, yet extremely legible. It bridges the gap between the physical world (Unreal Engine/Real Drones) and the cognitive world (AI Reasoning/Critics).

## 2. Color Palette
Reflecting a "Dark Mode IO" feel with neon accents for high-contrast visibility.

### Base Colors (The Void)
Used for backgrounds, panels, and surfaces to reduce eye strain in low-light environments.
- **Void Black**: `#09090B` (Main Background)
- **Deep Space**: `#18181B` (Secondary Background / Sidebar)
- **Orbital Gray**: `#27272A` (Cards / Panels / borders)
- **Steel**: `#52525B` (Muted Text / Icons)
- **Starlight**: `#FAFAFA` (Primary Text)

### Primary Branding (The Core)
- **Aegis Cyan**: `#06b6d4` (Primary Action, Active State, "Predictive" Thought)
  - *Glow*: `0 0 10px rgba(6, 182, 212, 0.5)`
- **Safety Green**: `#10b981` (Success, Low Risk, "Safe" Verdict)
- **Warning Amber**: `#f59e0b` (Moderate Risk, Concerns, "Deliberative" Thought)
- **Critical Red**: `#ef4444` (High Risk, Defect Detected, "Reject" Verdict)

### Cognitive States (The AI Brain)
Consistent across Dashboard and Unreal Engine "Thought Bubbles".
- **Reactive (Blue)**: `#3b82f6` - Immediate, low-level responses.
- **Deliberative (Amber)**: `#f59e0b` - Planning, weighing options.
- **Reflective (Purple)**: `#8b5cf6` - Learning, post-analysis.
- **Predictive (Cyan)**: `#06b6d4` - Future estimation, long-term planning.

## 3. Typography
Clean, legible sans-serif for UI, and monospace for data/code.

- **Primary Font**: `Inter` or `SF Pro Display`
  - *Usage*: Headers, Body text, Labels.
- **Data Font**: `JetBrains Mono` or `Roboto Mono`
  - *Usage*: Coordinates, Telemetry values, Code snippets, Logs.

### Hierarchy
- **H1 (Page Title)**: 24px, Bold, Tracking -0.02em
- **H2 (Section Header)**: 18px, Semibold, Uppercase, Tracking 0.05em (Technical feel)
- **Body**: 14px, Regular
- **Label/Caption**: 12px, Medium, Oppacity 0.7

## 4. UI Components & Effects

### Glassmorphism (The "Heads-Up Display" Layer)
Used for floating panels in Unreal and overlays in the Web Dashboard.
```css
.glass-panel {
  background: rgba(24, 24, 27, 0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
```

### Borders & Glows
- **Active Elements**: 1px solid `Aegis Cyan` with refined box-shadow `0 0 8px rgba(6, 182, 212, 0.3)`.
- **Critical Alerts**: Pulse animation with `Critical Red`.

### Data Visualization
- **Charts**: Minimal grid lines (`#27272A`). Thin, precise strokes (2px).
- **Gradients**: Use subtle gradients for area charts (e.g., `linear-gradient(180deg, rgba(6,182,212,0.2) 0%, rgba(6,182,212,0) 100%)`).

### The "Thought Bubble" (Unreal & Web)
A signature element representing the AI's mind.
- **Shape**: Rounded Rectangle with a "terminal" aesthetic.
- **Header**: Contains the Drone ID and current Cognitive State (Color-coded badge).
- **Body**: Scrolling text of reasoning logs (Monospace).
- **Icons**: Simple, outlined SVG icons for Critics (Shield for Safety, Clock for Efficiency, Target for Goal).

## 5. Imagery & Vibe Keywords
When generating assets or directing visual design:
- **Keywords**: *Cyberpunk-lite, Aerospace, Precision, Wireframe, Telemetry, LiDAR, Blueprint, Neon, Matte Black.*
- **Avoid**: *Cartoonish colors, Flat "Corporate Memphis" art, bright white backgrounds, cluttered interfaces.*

## 6. Logo & Iconography Guidelines
**"Not Insane."** Keep it simple, geometric, and immediately recognizable.

### Design Principles
- **Geometry First**: Use simple shapes (hexagons, shields, circles) as containers.
- **Monoline/Stroke**: Use consistent stroke weights.
- **No Clutter**: Remove background noise. Logos should work in monochrome.
- **Tech/Abstract**: Avoid literal illustrations of drones. Use abstract representations (e.g., a aperture blade for "Vision", a node graph for "AI").

### Concepts
- **The "Aegis" Shield**: A modernization of the protection concept. Not a medieval shield, but a digital barrier or forcefield.
- **The Electronic Eye**: Representing Computer Vision.
- **The Node**: Representing Agentic AI/Decision making.
