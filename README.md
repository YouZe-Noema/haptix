# haptix

> Tactile data infrastructure for the ML era.

`haptix` provides a unified format (`.hapt`), Python API, and ML framework integration for tactile sensor data. Think of it as `torchvision` for touch.

## Quick Start

```python
import haptix

# Load any supported tactile sensor file
data = haptix.load("experiment/sandpaper_80.hapt")

# Access metadata
print(data.interaction.speed_mm_s)   # 50.0
print(data.labels.material)          # "sandpaper_grit_80"

# Get data as numpy
frames = data.raw.numpy()            # ndarray [T, H, W, C]

# Feed directly to PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(data.to_torch(batch_size=32))

# Round-trip guarantee: save and reload is lossless
data.save("copy.hapt")
reloaded = haptix.load("copy.hapt")
assert data.raw.checksum == reloaded.raw.checksum
```

## Supported Sensors

| Sensor | Modality | Status |
|---|---|---|
| DIGIT / DIGIT v2 | imaging | ✅ Supported |
| GelSight | imaging | ✅ Via conversion |
| BioTac | dynamic | 🔜 Planned |
| TacTip | imaging | 🔜 Planned |

## Installation

```bash
pip install haptix           # core
pip install haptix[torch]    # with PyTorch integration
pip install haptix[all]      # everything
```

## License

MIT
