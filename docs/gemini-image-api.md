# Gemini 3 Pro Image Preview API Reference

## Model ID
```
gemini-3-pro-image-preview
```
Also known as "Nano Banana Pro Preview"

## Key Capabilities
- **High-resolution output**: 1K, 2K, and 4K visuals
- **Advanced text rendering**: Legible, stylized text for infographics, menus, diagrams
- **Google Search grounding**: Real-time data for weather, stocks, events
- **Thinking process**: Reasoning to refine composition before final output
- **Multi-reference images**: Up to 14 reference images (6 objects, 5 humans for consistency)

## Basic Usage

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="Your prompt here",
    config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE']
    )
)

# Extract results
for part in response.parts:
    if part.text:
        print(part.text)
    elif image := part.as_image():
        image.save("output.png")
```

## Image Editing with Reference Images

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")

# Load reference image
with open("reference.jpg", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Your editing prompt here"),
            ],
        )
    ],
    config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE'],
    ),
)
```

## Configuration Options

### Image Configuration
```python
config=types.GenerateContentConfig(
    response_modalities=['TEXT', 'IMAGE'],
    image_config=types.ImageConfig(
        aspect_ratio="16:9",  # Options: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
        image_size="2K"       # Options: "1K", "2K", "4K" (uppercase required)
    ),
)
```

### With Google Search Grounding
```python
config=types.GenerateContentConfig(
    response_modalities=['TEXT', 'IMAGE'],
    tools=[{"google_search": {}}]
)
```

## Multi-Turn Chat for Iterative Editing

```python
chat = client.chats.create(
    model="gemini-3-pro-image-preview",
    config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE']
    )
)

# First message
response1 = chat.send_message("Create a base image...")

# Iterative refinement
response2 = chat.send_message("Now modify the background to...")
```

## Token Consumption

| Aspect Ratio | 1K Tokens | 2K Tokens | 4K Tokens |
|--------------|-----------|-----------|-----------|
| 1:1          | 1120      | 1120      | 2000      |
| 16:9         | 1120      | 1120      | 2000      |
| 21:9         | 1120      | 1120      | 2000      |

## Best Practices

1. **Be hyper-specific**: Provide detailed descriptions over keyword lists
2. **Provide context**: Explain image purpose and intent
3. **Iterate conversationally**: Use multi-turn chat for refinement
4. **Use step-by-step instructions**: Break complex scenes into sequences
5. **Employ semantic language**: Describe desired outcomes positively
6. **Control composition**: Use photographic/cinematic terminology

## Limitations

- Best with specific languages (EN, de-DE, es-MX, fr-FR, ja-JP, ko-KR, pt-BR, zh-CN)
- Does not support audio or video inputs
- Won't always follow exact number of image outputs explicitly requested
- All generated images include a SynthID watermark

## Response Structure

Responses include:
- **Text parts**: Accompanying descriptions
- **Inline data parts**: Base64-encoded image data with MIME type "image/png"
- **Thought signatures**: Encrypted representations of internal thought process (for multi-turn consistency)
