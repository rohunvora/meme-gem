# meme-gem

Bulk generate personalized memes for any person using AI image generation.

## The Concept

Creating good AI-generated memes of a specific person requires two things:

1. **Optimized reference images** - A curated set of photos showing the person in different angles, expressions, and lighting conditions
2. **Smart meme generation** - Matching the right reference image to each meme template, then using AI to merge them while preserving facial identity

This tool handles both steps: organize your reference images with metadata, then batch-generate an entire meme pack with a single command.

## How It Works

```
reference_images/           memes/templates/
├── front_neutral.jpg       ├── drake.png
├── laughing_3quarter.jpg   ├── distracted_bf.png
├── confused_front.jpg      ├── this_is_fine.png
└── metadata.json           └── ...
        │                          │
        └──────────┬───────────────┘
                   ▼
           Gemini Image API
           (gemini-3-pro-image-preview)
                   │
                   ▼
            output/meme_pack/
            ├── drake.png
            ├── distracted_bf.png
            └── this_is_fine.png
```

**Reference image selection**: Each meme template is analyzed to determine what expression/angle it needs. The generator automatically picks the best matching reference image for each meme.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Add your key from https://aistudio.google.com/apikey

# Generate meme pack
python meme_generator.py
```

## Step 1: Prepare Reference Images

Add photos to `reference_images/` and create a `metadata.json`:

```json
{
  "subject": "Person Name",
  "images": {
    "photo1.jpg": {
      "angle": "front",
      "expression": "neutral, slight smile",
      "use_for": ["default", "most memes"],
      "primary": true
    },
    "photo2.jpg": {
      "angle": "3quarter",
      "expression": "laughing",
      "use_for": ["laughing", "happy", "reaction memes"]
    },
    "photo3.jpg": {
      "angle": "front",
      "expression": "confused, skeptical",
      "use_for": ["confused", "skeptical", "frustrated"]
    }
  },
  "features": {
    "glasses": "black rectangular frames",
    "facial_hair": "beard",
    "hair": "dark, short"
  }
}
```

**Tips for reference images:**
- Include a variety of expressions: neutral, happy, laughing, confused, serious
- Multiple angles help: front-facing, 3/4 view, profile
- Good lighting and high resolution improve results
- The `features` field helps preserve distinctive characteristics

## Step 2: Add Meme Templates

Drop meme images into `memes/templates/`. The system auto-analyzes each template to:
- Identify the required expression and angle
- Generate an optimized prompt for recreation
- Cache results for fast re-runs

## Step 3: Generate

```bash
# Generate full pack (auto-resumes if interrupted)
python meme_generator.py

# Limit number of memes
python meme_generator.py --limit 5

# List available templates
python meme_generator.py --list

# Just analyze templates (no generation)
python meme_generator.py --analyze
```

## Python API

```python
from meme_generator import MemeGenerator

generator = MemeGenerator()

# Generate single meme
generator.generate_meme("drake.png", person_name="john")

# Generate full pack
generator.generate_meme_pack(person_name="john")

# List templates with their requirements
generator.list_templates()
```

## Model

Uses `gemini-3-pro-image-preview` (Gemini Nano Banana Pro) - see [docs/gemini-image-api.md](docs/gemini-image-api.md) for API details.

**Key capabilities:**
- Multi-reference image support (up to 5 human references)
- High-resolution output (1K/2K/4K)
- Reasoning for composition refinement

## Requirements

- Python 3.10+
- Gemini API key
- ~$0.02-0.05 per meme

## License

MIT
