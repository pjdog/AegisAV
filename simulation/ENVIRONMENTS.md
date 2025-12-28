# AegisAV Environment Assets Plan

## Overview

High-fidelity 3D environments for drone inspection simulation in Unreal Engine.

## Priority Environments

### 1. Solar Farm (Sprint 1)
**Priority: HIGH** - Core use case

```
┌─────────────────────────────────────────────────────────────────┐
│                        SOLAR FARM                                │
├─────────────────────────────────────────────────────────────────┤
│  Layout:                                                         │
│  • 10x10 grid of solar panel arrays                             │
│  • Central inverter/transformer station                          │
│  • Gravel access roads between rows                              │
│  • Perimeter fence                                               │
│                                                                   │
│  Defects to model:                                               │
│  • Cracked panels (texture overlay)                              │
│  • Hot spots (thermal discoloration)                             │
│  • Bird droppings / debris                                       │
│  • Vegetation overgrowth (encroachment)                          │
│  • Broken mounting brackets                                      │
│                                                                   │
│  Assets needed:                                                   │
│  • Solar panel mesh (with material variants)                     │
│  • Mounting structure                                            │
│  • Inverter building                                             │
│  • Transformer                                                   │
│  • Fence sections                                                │
└─────────────────────────────────────────────────────────────────┘
```

**Asset Sources:**
- Unreal Marketplace: "Industrial Structures Pack"
- Free: Sketchfab solar panel models (convert to UE4)
- Custom: Defect texture overlays

### 2. Wind Turbine Farm (Sprint 2)
**Priority: HIGH** - Dramatic visuals

```
┌─────────────────────────────────────────────────────────────────┐
│                      WIND TURBINE FARM                           │
├─────────────────────────────────────────────────────────────────┤
│  Layout:                                                         │
│  • 5-10 wind turbines (80-120m height)                          │
│  • Varied terrain (hills, valleys)                               │
│  • Access roads                                                  │
│  • Substation building                                           │
│                                                                   │
│  Defects to model:                                               │
│  • Blade cracks / erosion                                        │
│  • Lightning damage                                              │
│  • Ice accumulation                                              │
│  • Oil leaks on nacelle                                          │
│  • Bird/bat strikes                                              │
│                                                                   │
│  Inspection patterns:                                            │
│  • Blade tip orbit (close inspection)                            │
│  • Full blade scan (vertical)                                    │
│  • Nacelle inspection (360° orbit)                               │
│  • Tower inspection (vertical descent)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Electrical Substation (Sprint 2)
**Priority: MEDIUM** - Critical infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                     ELECTRICAL SUBSTATION                        │
├─────────────────────────────────────────────────────────────────┤
│  Components:                                                     │
│  • High-voltage transformers                                     │
│  • Circuit breakers                                              │
│  • Disconnect switches                                           │
│  • Bus bars                                                      │
│  • Control building                                              │
│  • Insulators                                                    │
│                                                                   │
│  Defects to model:                                               │
│  • Corrosion on equipment                                        │
│  • Oil leaks                                                     │
│  • Cracked insulators                                            │
│  • Vegetation intrusion                                          │
│  • Heat damage / discoloration                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Power Line Corridor (Sprint 3)
**Priority: MEDIUM** - Linear inspection

```
┌─────────────────────────────────────────────────────────────────┐
│                    POWER LINE CORRIDOR                           │
├─────────────────────────────────────────────────────────────────┤
│  Layout:                                                         │
│  • 1km transmission line                                         │
│  • 5-10 towers                                                   │
│  • Varied terrain                                                │
│  • Right-of-way clearing                                         │
│                                                                   │
│  Defects to model:                                               │
│  • Damaged conductors                                            │
│  • Missing/damaged insulators                                    │
│  • Tower corrosion                                               │
│  • Vegetation encroachment                                       │
│  • Nesting debris                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Requirements

### Performance Targets (6950XT)
- **Resolution**: 1920x1080 @ 60 FPS minimum
- **Draw distance**: 2km+ for environment context
- **LOD system**: 4 levels for complex assets
- **Lighting**: Dynamic sun + real-time shadows

### Camera Specifications
- **Front camera**: 90° FOV, 1920x1080, RGB
- **Bottom camera**: 90° FOV, 1920x1080, for nadir shots
- **Thermal (simulated)**: Grayscale heat map overlay

### Defect Injection System

```python
# Runtime defect spawning
class DefectSpawner:
    def spawn_defect(self, asset, defect_type, severity):
        """Spawn a defect on an asset at runtime."""
        # Attach decal/mesh to asset
        # Set material parameters for severity
        # Register in world model
```

## Asset Pipeline

### Workflow
1. **Model in Blender** (or purchase from marketplace)
2. **UV unwrap** for defect decals
3. **Export FBX** to Unreal
4. **Create materials** with defect parameters
5. **Set up LODs** for performance
6. **Add collision** for physics

### Naming Convention
```
SM_SolarPanel_01          # Static mesh
M_SolarPanel_Base         # Base material
M_SolarPanel_Cracked      # Defect variant
MI_SolarPanel_Crack_01    # Material instance
BP_SolarPanel             # Blueprint with logic
```

## Sprint Timeline

| Sprint | Environment | Status |
|--------|-------------|--------|
| 1 | Solar Farm (basic) | 🔲 Not started |
| 2 | Wind Turbines | 🔲 Not started |
| 2 | Substation | 🔲 Not started |
| 3 | Power Lines | 🔲 Not started |
| 4 | Polish + Weather | 🔲 Not started |

## Quick Start Option

For faster demo, use **AirSim pre-built environments**:
- **AirSimNH** (Neighborhood) - Available now
- **LandscapeMountains** - Scenic terrain
- **City** - Urban environment

These won't have infrastructure assets but allow immediate flight testing.

## Resources

### Unreal Marketplace (Paid)
- "Industrial Structures" - $50
- "Power Plant Pack" - $35
- "Solar Panel Set" - $25

### Free Assets
- Sketchfab (CC licensed models)
- TurboSquid (free section)
- Quixel Megascans (free with UE)

### Tutorials
- [AirSim Custom Environment](https://microsoft.github.io/AirSim/build_linux/)
- [UE4 Level Design](https://docs.unrealengine.com/en-US/Basics/Levels/)
