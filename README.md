# GPT-Image-2 ComfyUI Nodes

> **ComfyUI custom nodes for GPT-Image-2** — OpenAI's latest image generation model.
> Generate and edit images directly inside ComfyUI using the OpenAI Images API or an OpenAI-compatible provider.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![GPT-Image-2](https://img.shields.io/badge/Model-GPT--Image--2-green)](https://platform.openai.com/docs/guides/image-generation)

---

## What is GPT-Image-2?

GPT-Image-2 uses OpenAI-compatible GPT Image models with native image editing support. It supports:

- **Text-to-Image** — generate high-quality images from a text description
- **Image-to-Image** — edit or transform up to 16 reference images guided by a prompt

---

## Nodes

| Node | Description |
|------|-------------|
| 🔑 GPT-Image-2 API Key | Set your key once — wire to all nodes |
| 🌐 GPT-Image-2 Base URL | Set an OpenAI-compatible API base URL for third-party providers |
| 🖼️ GPT-Image-2 Text to Image | Generate an image from a text prompt |
| 🖼️ GPT-Image-2 Image to Image | Edit or transform up to 16 reference images |
| 🔑 Jimeng API Key | Set a Volcano Ark API key for Jimeng/Seedream nodes |
| 🌐 Jimeng Base URL | Set a Volcano Ark-compatible API base URL |
| 🖼️ Jimeng Seedream Image | Generate or edit images with exposed Seedream image parameters |
| 🔑 Mango AIGC Credentials | Set Mango AIGC access and secret keys for Wan2.7 nodes |
| 🌐 Mango AIGC Base URL | Set the Mango AIGC API base URL |
| 🔑 Hello Image Host Token | Set a Hello image host token for Wan2.7 reference uploads |
| 🖼️ Mango Wan2.7 Text to Image | Generate images with Mango AIGC Wan2.7 image models |
| ✂️ Grid Crop Images | Split images into a configurable cols x rows grid |

---

## Installation

### Via ComfyUI Manager (recommended)
1. Open **ComfyUI Manager** → **Install via Git URL**
2. Paste: `https://github.com/Anil-matcha/gpt-image-2-comfyui`
3. Restart ComfyUI

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Anil-matcha/gpt-image-2-comfyui
pip install -r gpt-image-2-comfyui/requirements.txt
```

---

## Quick Start

1. Create an OpenAI API key, or get a key from an OpenAI-compatible provider
2. Right-click the ComfyUI canvas → **Add Node** → **🖼️ GPT-Image-2**
3. Add a **🔑 GPT-Image-2 API Key** node, paste your key, and wire its output to any generation node
4. Write a prompt and hit **Queue Prompt**

> **Tip:** You can put your key and base URL in this plugin's `config.json` so they are easy to find.
> **Third-party API:** Add a **🌐 GPT-Image-2 Base URL** node or fill the `base_url` field on the generation nodes to use an OpenAI-compatible provider. The default is `https://api.openai.com/v1`.

---

## Node Reference

### 🔑 GPT-Image-2 API Key

Set your OpenAI or OpenAI-compatible API key once and wire the output to all generation nodes. Alternatively, leave every `api_key` field blank — nodes automatically read from `OPENAI_API_KEY`, this plugin's `config.json`, then fall back to `~/.muapi/config.json` for older MuAPI CLI compatibility.

| Field | Description |
|-------|-------------|
| `api_key` | Your OpenAI or OpenAI-compatible provider API key |

**Output:** `api_key` string (wire to generation nodes)

---

### 🌐 GPT-Image-2 Base URL

Set an OpenAI-compatible API base URL once and wire the output to generation nodes. Leave blank to use `https://api.openai.com/v1`.

The easiest persistent setup is to create `config.json` in this plugin directory:

```json
{
  "api_key": "YOUR_KEY",
  "base_url": "https://api.example.com/v1",
  "jimeng_api_key": "YOUR_VOLCENGINE_ARK_KEY",
  "jimeng_base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "mango_access_key": "YOUR_MANGO_AIGC_ACCESS_KEY",
  "mango_secret_key": "YOUR_MANGO_AIGC_SECRET_KEY",
  "mango_base_url": "https://aigc.mgtv.com",
  "helloimg_token": "YOUR_HELLOIMG_TOKEN",
  "helloimg_base_url": "https://www.helloimg.com/api/v1",
  "user_agent": "gpt-image-2-comfyui/1.0"
}
```

The plugin reads config in this order:

1. Node field input
2. Environment variables
3. This plugin's `config.json`
4. `~/.muapi/config.json`
5. Default OpenAI base URL

You can also configure the base URL with either environment variable:

```bash
OPENAI_BASE_URL=https://api.example.com/v1
GPT_IMAGE2_BASE_URL=https://api.example.com/v1
MUAPI_BASE_URL=https://api.example.com/v1
OPENAI_USER_AGENT=gpt-image-2-comfyui/1.0
JIMENG_API_KEY=your-volcano-ark-key
JIMENG_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
MANGO_ACCESS_KEY=your-mango-aigc-access-key
MANGO_SECRET_KEY=your-mango-aigc-secret-key
MANGO_BASE_URL=https://aigc.mgtv.com
HELLOIMG_TOKEN=your-helloimg-token
HELLOIMG_BASE_URL=https://www.helloimg.com/api/v1
```

**Output:** `base_url` string (wire to generation nodes)

The optional `user_agent` config value is sent as the `User-Agent` header on API requests and image URL downloads. Set `OPENAI_USER_AGENT` to override it without editing the config file.

---

### 🖼️ GPT-Image-2 Text to Image

Generate a high-quality image from a text prompt.

| Field | Description |
|-------|-------------|
| `prompt` | Describe the image you want to generate |
| `api_key` | *(optional)* API key — wire from the API Key node or leave blank |
| `base_url` | *(optional)* OpenAI-compatible API base URL — wire from the Base URL node or leave blank |
| `model` | GPT Image model: `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, or `gpt-image-1-mini` |
| `n` | Number of images to request, 1-10 |
| `size` | `auto`, `1024x1024`, `1536x1024`, `1024x1536`, `2048x2048`, `2048x1152`, `1152x2048`, `3840x2160`, or `2160x3840` |
| `quality` | `auto`, `low`, `medium`, or `high` |
| `background` | `auto`, `transparent`, or `opaque` |
| `output_format` | `png`, `jpeg`, or `webp` |
| `output_compression` | Compression level, 0-100 |
| `moderation` | `auto` or `low` |
| `stream` | Request streaming mode from providers that support it |
| `partial_images` | Number of partial images to request while streaming |
| `style` | Optional DALL-E 3 style passthrough for compatible providers |
| `user` | Optional end-user identifier |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image as a ComfyUI tensor |
| `image_url` | STRING | Generated image URL or data URL |
| `request_id` | STRING | OpenAI request ID or created timestamp |

**Example prompt:**
```
A photorealistic image of a red fox sitting in a snowy forest at dusk.
```

---

### 🖼️ GPT-Image-2 Image to Image

Edit or transform up to 16 reference images guided by a text prompt.

| Field | Description |
|-------|-------------|
| `prompt` | Describe the desired transformation or style |
| `api_key` | *(optional)* API key — wire from the API Key node or leave blank |
| `base_url` | *(optional)* OpenAI-compatible API base URL — wire from the Base URL node or leave blank |
| `model` | GPT Image model: `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, or `gpt-image-1-mini` |
| `n` | Number of images to request, 1-10 |
| `size` | `auto`, `1024x1024`, `1536x1024`, `1024x1536`, `2048x2048`, `2048x1152`, `1152x2048`, `3840x2160`, or `2160x3840` |
| `quality` | `auto`, `low`, `medium`, or `high` |
| `background` | `auto`, `transparent`, or `opaque` |
| `input_fidelity` | `auto`, `low`, or `high` |
| `output_format` | `png`, `jpeg`, or `webp` |
| `output_compression` | Compression level, 0-100 |
| `moderation` | `auto` or `low` |
| `stream` | Request streaming mode from providers that support it |
| `partial_images` | Number of partial images to request while streaming |
| `user` | Optional end-user identifier |
| `mask_image` | Optional mask image for inpainting-compatible providers |
| `image_1` … `image_16` | *(optional)* Reference images to edit |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image as a ComfyUI tensor |
| `image_url` | STRING | Generated image URL or data URL |
| `request_id` | STRING | OpenAI request ID or created timestamp |

**Example prompt:**
```
Transform this product image into a premium e-commerce poster style.
```

---

### 🖼️ Jimeng Seedream Image

Generate or edit images through Volcano Ark's Seedream image generation API. The node exposes the core API parameters directly, including model, size, response format, output format, watermark, seed, streaming, sequential image generation, web search, prompt optimization mode, and raw `extra_json` passthrough.

| Field | Description |
|-------|-------------|
| `prompt` | Text instruction for generation or image editing |
| `jimeng_api_key` | *(optional)* Volcano Ark API key — wire from the Jimeng API Key node or leave blank |
| `base_url` | *(optional)* Volcano Ark-compatible base URL |
| `model` | Seedream model ID, default `doubao-seedream-5-0-260128` |
| `size` | `2K`, `3K`, `4K`, preset pixel sizes, or `custom` |
| `custom_width`, `custom_height` | Used when `size` is `custom` |
| `response_format` | `url` or `b64_json` |
| `output_format` | `jpeg` or `png` |
| `watermark` | Add the provider watermark when enabled |
| `seed` | Reproducibility seed; `-1` omits the parameter |
| `stream` | Request streaming image events |
| `sequential_image_generation` | `disabled` or `auto` for grouped image output |
| `max_images` | Maximum generated images for sequential output |
| `enable_web_search` | Sends `tools: [{"type": "web_search"}]` |
| `optimize_prompt_mode` | `standard`, `fast`, `auto`, or `disabled` |
| `extra_json` | Raw JSON object merged into the request payload |
| `image_1` … `image_14` | Optional reference images |

**Outputs:** generated images as an IMAGE batch, newline-separated image URLs/data URLs, request ID, and raw JSON response.

---

### 🖼️ Mango Wan2.7 Text to Image

Generate images through Mango AIGC's storyboard image API. The node submits `POST /openapi/v1/storyboard/generateByPromptV2`, reads the returned `aseetRecordId`, then polls `POST /openapi/v1/storyboard/getAssetInfo` with `recordIds` until `images[].imgUrl` is available.

| Field | Description |
|-------|-------------|
| `prompt` | Text prompt for Wan2.7 image generation |
| `mango_access_key`, `mango_secret_key` | *(optional)* Mango AIGC credentials — wire from the Credentials node or leave blank |
| `base_url` | *(optional)* Mango AIGC API base URL, default `https://aigc.mgtv.com` |
| `helloimg_token` | *(optional)* Hello image host token used when local IMAGE inputs are connected |
| `helloimg_base_url` | *(optional)* Hello API base URL, default `https://www.helloimg.com/api/v1` |
| `model` | `Wan2.7-image-pro (35)` or `Wan2.7-image (34)` |
| `style_id` | Manual style ID fallback; model selection takes precedence |
| `ratio` | Output aspect ratio such as `16:9`, `1:1`, or `9:16` |
| `resolution` | `1K` or `2K` |
| `nums` | Number of images to request, 1-4 |
| `seed` | Reproducibility seed; `-1` omits the parameter |
| `img_urls` | Optional reference image URLs, one per line, sent as `imgUrls` |
| `image_1` … `image_6` | Optional local reference images. The node uploads them to Hello image host first, with a fixed 1-hour expiry, then sends the returned URLs as `imgUrls` |
| `prompt_args_json` | Optional JSON array sent as `prompt.args` |
| `poll_interval`, `timeout_seconds` | Asset polling cadence and timeout |
| `extra_json` | Raw JSON object merged into the request payload |

**Outputs:** generated images as an IMAGE batch, newline-separated image URLs, Mango asset record ID, and raw JSON response.

---

### ✂️ Grid Crop Images

Split each input image into a configurable grid. The output `images` is an IMAGE batch ordered left-to-right, top-to-bottom.

| Field | Description |
|-------|-------------|
| `image` | Input IMAGE or IMAGE batch |
| `mask` | Optional MASK input, for workflows where alpha is carried separately from IMAGE |
| `cols` | Number of columns |
| `rows` | Number of rows |
| `fit_mode` | `pad_edge` pads the right/bottom edge so every tile has the same size; `crop_remainder` drops non-divisible right/bottom pixels |
| `col_start_offset` | Pixels to skip from the left before splitting the grid |
| `col_end_offset` | Pixels to skip from the right before splitting the grid |
| `row_start_offset` | Pixels to skip from the top before splitting the grid |
| `row_end_offset` | Pixels to skip from the bottom before splitting the grid |

**Outputs:** split images as an IMAGE batch, plus `tile_width`, `tile_height`, and cropped `masks`.

---

## API Compatibility

By default, requests go through the OpenAI API. You can override the base URL for any provider that implements the same OpenAI Images API request format:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/images/generations` | POST | Create an image from a prompt |
| `/v1/images/edits` | POST | Edit or generate from one or more input images |

Text-to-image requests are sent as JSON. Image-to-image requests are sent as `multipart/form-data` with `image[]` file fields and an optional `mask` file field.

The generation nodes expose the current OpenAI Images API option names and pass them through in the request payload. The nodes read `data[0].b64_json` responses from GPT Image models, and also support URL responses from compatible providers.

`gpt-image-2` is the default model. For edit requests, `input_fidelity` is omitted for `gpt-image-2` because GPT Image 2 always processes image inputs at high fidelity.

---

## License

MIT — see [LICENSE](LICENSE).
