# Project Nora

Autonomous robot with multimodal AI capabilities - vision, voice, and creative content generation.

## License

Apache License 2.0 - See LICENSE file for details.

Patent protection included. Commercial use allowed with attribution.

## Workspace Structure
```
nora_ws/
├── src/
│   └── nora_head/          # Robot head subsystem
│       ├── nora_head/      # Python package
│       │   ├── nodes/      # ROS 2 nodes
│       │   ├── launch/     # Launch files
│       │   ├── config/     # Configuration files
│       │   └── utils/      # Utility modules
│       ├── msg/            # Custom messages
│       ├── srv/            # Custom services
│       └── setup.py
├── build/                  # Build artifacts (gitignored)
├── install/                # Install space (gitignored)
└── log/                    # Build logs (gitignored)
```

## Quick Commands
```bash
# Build workspace
nora_build

# Navigate to workspace
nora_ws

# Navigate to source
nora_src

# Clean build
nora_clean
```

## Hardware

- Compute: Jetson Nano Orin (8GB, 512GB SSD)
- Vision: 16MP autofocus camera
- Audio: ReSpeaker 4-mic array (XVF3800)
- Output: 5W dual speakers, 24× NeoPixel LEDs (2 rings)
- Movement: 2× servos (pan/tilt)
- Power: 24V 8Ah battery, mecanum wheel base

## Architecture

- Orchestration: ROS 2 Humble
- Native nodes: Hardware interfaces, coordination
- Container nodes: Heavy AI models (YOLOv8, Whisper, Llama)
- Cluster: 3× Raspberry Pi 5 (offload processing when docked)

## Wake Word

"Hey Nora" or "Nora"

## Features

- Interest-driven autonomous exploration
- Multimodal interaction (vision + voice)
- Emotion recognition and empathetic response
- Creative content generation (blog posts, poetry)
- Offline-first with opportunistic cloud offload
- Graceful degradation (works without internet)

## Dependencies

- ROS 2 Humble (Apache-2.0)
- Isaac ROS via Docker containers (Apache-2.0)
- Jetson AI Lab models (Apache-2.0)
- Porcupine wake word (Apache-2.0 free tier)

All major dependencies use compatible licenses.
