"""
OpenAI GPT Image ComfyUI Nodes
==============================
ComfyUI nodes for GPT Image generation via the OpenAI Images API.

  GPTImage2TextToImage     — POST /v1/images/generations
  GPTImage2ImageToImage    — POST /v1/images/edits

Auth:     Authorization: Bearer <api key>
"""

import base64
import hashlib
import hmac
import io
import os
import json
import time
import uuid

import numpy as np
import requests
import torch
from PIL import Image

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_CONFIG_PATH = os.path.join(PLUGIN_DIR, "config.json")
LEGACY_CONFIG_PATH = os.path.expanduser("~/.muapi/config.json")
CONFIG_PATHS = (PLUGIN_CONFIG_PATH, LEGACY_CONFIG_PATH)
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JIMENG_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MANGO_BASE_URL = "https://aigc.mgtv.com"
DEFAULT_HELLOIMG_BASE_URL = "https://www.helloimg.com/api/v1"
DEFAULT_USER_AGENT = "gpt-image-2-comfyui/1.0"
TOOLBOX_CATEGORY = "⭐ Toolbox"
GPT_IMAGE_CATEGORY = f"{TOOLBOX_CATEGORY}/GPT-Image-2"
JIMENG_CATEGORY = f"{TOOLBOX_CATEGORY}/Jimeng"
MANGO_CATEGORY = f"{TOOLBOX_CATEGORY}/Mango AIGC"
IMAGE_CATEGORY = f"{TOOLBOX_CATEGORY}/Image"

MODEL_OPTIONS = ["gpt-image-2", "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"]
SIZE_OPTIONS = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "1152x2048",
    "3840x2160",
    "2160x3840",
]
QUALITY_OPTIONS = ["auto", "low", "medium", "high"]
BACKGROUND_OPTIONS = ["auto", "transparent", "opaque"]
OUTPUT_FORMAT_OPTIONS = ["png", "jpeg", "webp"]
MODERATION_OPTIONS = ["auto", "low"]
INPUT_FIDELITY_OPTIONS = ["auto", "low", "high"]
JIMENG_MODEL_OPTIONS = [
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
    "doubao-seedream-3-0-t2i-250415",
]
JIMENG_SIZE_OPTIONS = [
    "2K",
    "3K",
    "4K",
    "2048x2048",
    "2304x1728",
    "1728x2304",
    "2848x1600",
    "1600x2848",
    "2496x1664",
    "1664x2496",
    "3136x1344",
    "3072x3072",
    "3456x2592",
    "2592x3456",
    "4096x2304",
    "2304x4096",
    "2496x3744",
    "3744x2496",
    "4704x2016",
    "custom",
]
JIMENG_RESPONSE_FORMAT_OPTIONS = ["url", "b64_json"]
JIMENG_OUTPUT_FORMAT_OPTIONS = ["jpeg", "png"]
JIMENG_SEQUENTIAL_OPTIONS = ["disabled", "auto"]
JIMENG_OPTIMIZE_PROMPT_OPTIONS = ["auto", "standard", "fast", "disabled"]
MANGO_WAN27_STYLE_OPTIONS = [
    "Wan2.7-image-pro (35)",
    "Wan2.7-image (34)",
]
MANGO_RATIO_OPTIONS = ["16:9", "1:1", "9:16", "3:4", "4:3", "3:2", "2:3", "21:9"]
MANGO_RESOLUTION_OPTIONS = ["1K", "2K"]


# ── Helpers ────────────────────────────────────────────────────────────────────

_CONFIG_ERRORS = {}


def _read_config(config_path):
    if not os.path.isfile(config_path):
        return None

    _CONFIG_ERRORS.pop(config_path, None)
    try:
        with open(config_path, encoding="utf-8-sig") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("top-level JSON value must be an object")
        return config
    except Exception as exc:
        _CONFIG_ERRORS[config_path] = str(exc)
        return None


def _clean_config_value(value):
    if value is None:
        return ""
    return str(value).strip()


def _config_lookup(config, key):
    for candidate in (key, key.upper()):
        if candidate in config:
            value = _clean_config_value(config.get(candidate))
            if value:
                return value

    if "_" in key:
        section_name, nested_key = key.split("_", 1)
        section = config.get(section_name) or config.get(section_name.upper())
        if isinstance(section, dict):
            for candidate in (nested_key, nested_key.upper()):
                if candidate in section:
                    value = _clean_config_value(section.get(candidate))
                    if value:
                        return value

    for section_name in ("openai", "gpt_image", "gpt_image2"):
        section = config.get(section_name) or config.get(section_name.upper())
        if isinstance(section, dict):
            for candidate in (key, key.upper()):
                if candidate in section:
                    value = _clean_config_value(section.get(candidate))
                    if value:
                        return value

    return ""


def _load_config_value(key):
    for config_path in CONFIG_PATHS:
        config = _read_config(config_path)
        if config:
            value = _config_lookup(config, key)
            if value:
                return value
    return ""


def _config_status_message():
    parts = []
    for config_path in CONFIG_PATHS:
        if config_path in _CONFIG_ERRORS:
            parts.append(f"{config_path} (invalid: {_CONFIG_ERRORS[config_path]})")
        elif os.path.isfile(config_path):
            parts.append(f"{config_path} (loaded)")
        else:
            parts.append(f"{config_path} (missing)")
    return "; ".join(parts)


def _load_api_key(api_key_input):
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    key = _load_config_value("api_key")
    if key:
        return key
    raise RuntimeError(
        "No API key found. Either paste your key into the api_key field, "
        "set OPENAI_API_KEY, or add api_key to config.json. "
        f"Checked config files: {_config_status_message()}"
    )


def _load_base_url(base_url_input=""):
    if base_url_input and base_url_input.strip():
        return base_url_input.strip().rstrip("/")

    for env_name in ("OPENAI_BASE_URL", "GPT_IMAGE2_BASE_URL", "MUAPI_BASE_URL"):
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url.rstrip("/")

    base_url = _load_config_value("base_url")
    if base_url:
        return base_url.rstrip("/")

    return DEFAULT_BASE_URL


def _load_jimeng_api_key(api_key_input):
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()

    for env_name in ("JIMENG_API_KEY", "VOLCENGINE_API_KEY", "ARK_API_KEY"):
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return env_key

    for key in ("jimeng_api_key", "volcengine_api_key", "ark_api_key"):
        value = _load_config_value(key)
        if value:
            return value

    raise RuntimeError(
        "No Jimeng API key found. Paste it into jimeng_api_key, "
        "set JIMENG_API_KEY/VOLCENGINE_API_KEY/ARK_API_KEY, or add "
        "jimeng_api_key, volcengine_api_key, or ark_api_key to config.json. "
        f"Checked config files: {_config_status_message()}"
    )


def _load_jimeng_base_url(base_url_input=""):
    if base_url_input and base_url_input.strip():
        return base_url_input.strip().rstrip("/")

    for env_name in ("JIMENG_BASE_URL", "VOLCENGINE_ARK_BASE_URL", "ARK_BASE_URL"):
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url.rstrip("/")

    for key in ("jimeng_base_url", "volcengine_base_url", "ark_base_url"):
        value = _load_config_value(key)
        if value:
            return value.rstrip("/")

    return DEFAULT_JIMENG_BASE_URL


def _load_mango_access_key(access_key_input):
    if access_key_input and access_key_input.strip():
        return access_key_input.strip()

    for env_name in ("MANGO_ACCESS_KEY", "MGTV_ACCESS_KEY", "MANGO_AIGC_ACCESS_KEY"):
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return env_key

    for key in ("mango_access_key", "mgtv_access_key", "mango_aigc_access_key"):
        value = _load_config_value(key)
        if value:
            return value

    raise RuntimeError(
        "No Mango AIGC access key found. Paste it into mango_access_key, "
        "set MANGO_ACCESS_KEY/MGTV_ACCESS_KEY/MANGO_AIGC_ACCESS_KEY, or add "
        "mango_access_key to config.json. "
        f"Checked config files: {_config_status_message()}"
    )


def _load_mango_secret_key(secret_key_input):
    if secret_key_input and secret_key_input.strip():
        return secret_key_input.strip()

    for env_name in ("MANGO_SECRET_KEY", "MGTV_SECRET_KEY", "MANGO_AIGC_SECRET_KEY"):
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return env_key

    for key in ("mango_secret_key", "mgtv_secret_key", "mango_aigc_secret_key"):
        value = _load_config_value(key)
        if value:
            return value

    raise RuntimeError(
        "No Mango AIGC secret key found. Paste it into mango_secret_key, "
        "set MANGO_SECRET_KEY/MGTV_SECRET_KEY/MANGO_AIGC_SECRET_KEY, or add "
        "mango_secret_key to config.json. "
        f"Checked config files: {_config_status_message()}"
    )


def _load_mango_base_url(base_url_input=""):
    if base_url_input and base_url_input.strip():
        return base_url_input.strip().rstrip("/")

    for env_name in ("MANGO_BASE_URL", "MGTV_BASE_URL", "MANGO_AIGC_BASE_URL"):
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url.rstrip("/")

    for key in ("mango_base_url", "mgtv_base_url", "mango_aigc_base_url"):
        value = _load_config_value(key)
        if value:
            return value.rstrip("/")

    return DEFAULT_MANGO_BASE_URL


def _load_helloimg_token(token_input):
    if token_input and token_input.strip():
        return token_input.strip()

    for env_name in ("HELLOIMG_TOKEN", "HELLO_IMAGE_TOKEN"):
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return env_key

    for key in ("helloimg_token", "hello_image_token"):
        value = _load_config_value(key)
        if value:
            return value

    raise RuntimeError(
        "No Hello image host token found. Paste it into helloimg_token, "
        "set HELLOIMG_TOKEN/HELLO_IMAGE_TOKEN, or add helloimg_token to config.json. "
        f"Checked config files: {_config_status_message()}"
    )


def _load_helloimg_base_url(base_url_input=""):
    if base_url_input and base_url_input.strip():
        return base_url_input.strip().rstrip("/")

    for env_name in ("HELLOIMG_BASE_URL", "HELLO_IMAGE_BASE_URL"):
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url.rstrip("/")

    for key in ("helloimg_base_url", "hello_image_base_url"):
        value = _load_config_value(key)
        if value:
            return value.rstrip("/")

    return DEFAULT_HELLOIMG_BASE_URL


def _load_user_agent():
    env_user_agent = os.environ.get("OPENAI_USER_AGENT", "").strip()
    if env_user_agent:
        return env_user_agent

    user_agent = _load_config_value("user_agent")
    if user_agent:
        return user_agent

    return DEFAULT_USER_AGENT


def _auth_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": _load_user_agent(),
    }


def _image_options_payload(
    model,
    n,
    size,
    quality,
    background,
    output_format,
    output_compression,
    moderation,
    user,
    stream,
    partial_images,
    style=None,
    input_fidelity=None,
):
    payload = {
        "model": model,
        "n": int(n),
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "moderation": moderation,
    }
    if output_format in ("jpeg", "webp"):
        payload["output_compression"] = int(output_compression)
    if user and user.strip():
        payload["user"] = user.strip()
    if stream:
        payload["stream"] = True
    if stream and partial_images:
        payload["partial_images"] = int(partial_images)
    if style and style.strip():
        payload["style"] = style.strip()
    if input_fidelity and input_fidelity != "auto" and model != "gpt-image-2":
        payload["input_fidelity"] = input_fidelity
    return payload


def _json_to_image_result(data):
    url = _output_image_url(data)
    return _download_image(url), url


def _read_stream_response(resp):
    _check(resp)
    completed = None
    last_image_event = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", ""))
        if event.get("b64_json"):
            last_image_event = event
        if event_type.endswith(".completed"):
            completed = event
            break
    if completed:
        return completed
    if last_image_event:
        return last_image_event
    raise RuntimeError("Streaming response finished without a generated image.")


def _post_json(api_key, base_url, endpoint, payload):
    use_stream = bool(payload.get("stream"))
    resp = requests.post(
        f"{base_url}/{endpoint}",
        headers={**_auth_headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=600,
        stream=use_stream,
    )
    if use_stream:
        return _read_stream_response(resp), _request_id(resp)
    _check(resp)
    return resp.json(), _request_id(resp)


def _post_multipart(api_key, base_url, endpoint, data, files, stream_response=False):
    resp = requests.post(
        f"{base_url}/{endpoint}",
        headers=_auth_headers(api_key),
        data=data,
        files=files,
        timeout=600,
        stream=stream_response,
    )
    if stream_response:
        return _read_stream_response(resp), _request_id(resp)
    _check(resp)
    return resp.json(), _request_id(resp)


def _read_jimeng_stream_response(resp):
    _check(resp)
    images = []
    usage = {}
    final_event = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", ""))
        if event.get("usage"):
            usage = event.get("usage") or {}
        if event.get("error"):
            raise RuntimeError(f"Jimeng stream error: {event['error']}")
        if event.get("b64_json") or event.get("url"):
            images.append(event)
        if event_type.endswith(".completed"):
            final_event = event
            break
    if final_event and final_event.get("images"):
        images.extend(final_event["images"])
    if not images and final_event:
        _append_image_url([], final_event)
        images.append(final_event)
    if not images:
        raise RuntimeError("Jimeng streaming response finished without an image.")
    return {"data": images, "usage": usage, "stream_event": final_event or {}}


def _post_jimeng_json(api_key, base_url, payload):
    use_stream = bool(payload.get("stream"))
    resp = requests.post(
        f"{base_url}/images/generations",
        headers={
            **_auth_headers(api_key),
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=900,
        stream=use_stream,
    )
    if use_stream:
        return _read_jimeng_stream_response(resp), _request_id(resp)
    _check(resp)
    return resp.json(), _request_id(resp)


def _mango_signature(method, path, timestamp, nonce, query_params, secret_key):
    query_parts = []
    for key in sorted(query_params.keys()):
        query_parts.append(f"{key}={query_params[key]}")
    sorted_query_string = "&".join(query_parts)
    message = f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{sorted_query_string}".encode("utf-8")
    secret = secret_key.encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _mango_headers(access_key, secret_key, method, path, query_params=None):
    query_params = query_params or {}
    nonce = uuid.uuid4().hex[:16]
    timestamp = str(int(time.time()))
    return {
        "Content-Type": "application/json",
        "User-Agent": _load_user_agent(),
        "X-Access-Key": access_key,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": _mango_signature(
            method,
            path,
            timestamp,
            nonce,
            query_params,
            secret_key,
        ),
    }


def _mango_response_data(result):
    if not isinstance(result, dict):
        raise RuntimeError(f"Mango AIGC response is not a JSON object: {result}")

    code = result.get("code")
    if code not in (None, 0, "0", 200, "200", "success", "SUCCESS"):
        message = result.get("msg") or result.get("message") or result.get("error") or result
        raise RuntimeError(f"Mango AIGC API error {code}: {message}")

    data = result.get("data", result)
    if isinstance(data, dict):
        data_code = data.get("code")
        if data_code not in (None, 0, "0", 200, "200", "success", "SUCCESS"):
            message = data.get("msg") or data.get("message") or data.get("error") or data
            raise RuntimeError(f"Mango AIGC API error {data_code}: {message}")

    return data


def _post_mango_json(access_key, secret_key, base_url, endpoint, payload):
    path = "/" + endpoint.lstrip("/")
    resp = requests.post(
        f"{base_url}{path}",
        headers=_mango_headers(access_key, secret_key, "POST", path),
        json=payload,
        timeout=120,
    )
    _check(resp)
    return resp.json(), _request_id(resp)


def _helloimg_expired_at(hours=1):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + hours * 3600))


def _helloimg_image_url(result):
    if not isinstance(result, dict):
        raise RuntimeError(f"Hello image host response is not a JSON object: {result}")

    status = result.get("status")
    success = result.get("success")
    if status in (False, "false") or success in (False, "false"):
        message = result.get("message") or result.get("msg") or result.get("error") or result
        raise RuntimeError(f"Hello image host upload failed: {message}")
    if status not in (None, True, "true", 200, "200") and success not in (None, True, "true"):
        message = result.get("message") or result.get("msg") or result.get("error") or result
        raise RuntimeError(f"Hello image host upload failed: {message}")

    data = result.get("data") or result
    if isinstance(data, dict):
        links = data.get("links")
        if isinstance(links, dict):
            for key in ("url", "html", "markdown", "delete_url"):
                value = links.get(key)
                if value and key == "url":
                    return str(value)
        for key in ("url", "image_url", "src"):
            value = data.get(key)
            if value:
                return str(value)

    raise RuntimeError(f"No uploaded image URL in Hello image host response: {result}")


def _upload_helloimg_image(image_tensor, token, base_url):
    filename, file_obj, mime_type = _image_tensor_to_file_tuple(image_tensor, "reference.png")
    try:
        resp = requests.post(
            f"{base_url}/upload",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": _load_user_agent(),
            },
            data={
                "permission": 1,
                "expired_at": _helloimg_expired_at(1),
            },
            files={
                "file": (filename, file_obj, mime_type),
            },
            timeout=120,
        )
    finally:
        file_obj.close()
    _check(resp)
    return _helloimg_image_url(resp.json()), resp.json()


def _upload_helloimg_images(image_tensors, token, base_url):
    urls = []
    responses = []
    for image_tensor in image_tensors:
        url, response = _upload_helloimg_image(image_tensor, token, base_url)
        urls.append(url)
        responses.append(response)
    return urls, responses


def _mango_record_id(result):
    data = _mango_response_data(result)
    if isinstance(data, dict):
        for key in ("aseetRecordId", "assetRecordId", "recordId", "id"):
            value = data.get(key)
            if value:
                return str(value)
    raise RuntimeError(f"No Mango AIGC asset record id in result: {result}")


def _mango_asset_urls(asset_result):
    data = _mango_response_data(asset_result)
    urls = []

    def walk(item):
        if isinstance(item, str):
            if item:
                urls.append(item)
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        for key in ("images", "imageList", "outputs", "output"):
            value = item.get(key)
            if value:
                walk(value)
        for key in ("imgUrl", "imageUrl", "url", "wmImgUrl"):
            value = item.get(key)
            if value:
                urls.append(str(value))
                return

    walk(data)

    seen = []
    for url in urls:
        if url not in seen:
            seen.append(url)
    return seen


def _mango_asset_status(asset_result):
    data = _mango_response_data(asset_result)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return ""
    for key in ("generateStatus", "status", "state"):
        value = data.get(key)
        if value is not None:
            if isinstance(value, dict) and value.get("status") is not None:
                return str(value["status"])
            return str(value)
    return ""


def _poll_mango_asset(access_key, secret_key, base_url, record_id, poll_interval, timeout_seconds):
    deadline = time.time() + int(timeout_seconds)
    last_result = None
    while True:
        last_result, request_id = _post_mango_json(
            access_key,
            secret_key,
            base_url,
            "openapi/v1/storyboard/getAssetInfo",
            {"recordIds": [int(record_id) if str(record_id).isdigit() else record_id]},
        )
        urls = _mango_asset_urls(last_result)
        if urls:
            return last_result, request_id

        status = _mango_asset_status(last_result).lower()
        if status in ("failed", "fail", "error", "canceled", "cancelled", "3", "-1"):
            raise RuntimeError(f"Mango AIGC generation failed: {last_result}")
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Mango AIGC asset {record_id}. Last response: {last_result}"
            )

        time.sleep(max(1, int(poll_interval)))


def _request_id(resp):
    return (
        resp.headers.get("x-request-id")
        or resp.headers.get("openai-request-id")
        or ""
    )


def _image_tensor_to_file_tuple(image_tensor, filename):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    buf.seek(0)
    return (filename, buf, "image/png")


def _image_tensor_to_data_url(image_tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _multipart_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _output_image_url(result):
    out = result.get("outputs") or result.get("output") or []
    if isinstance(out, list) and out:
        return str(out[0])
    if isinstance(out, str):
        return out
    data = result.get("data")
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            if item.get("url"):
                return str(item["url"])
            if item.get("b64_json"):
                output_format = result.get("output_format") or "png"
                return f"data:image/{output_format};base64,{item['b64_json']}"
    if result.get("b64_json"):
        output_format = result.get("output_format") or "png"
        return f"data:image/{output_format};base64,{result['b64_json']}"
    for k in ("image_url", "url"):
        if result.get(k):
            return str(result[k])
    raise RuntimeError(f"No output image URL in result: {result}")


def _append_image_url(urls, item, default_format="png"):
    if isinstance(item, str) and item:
        urls.append(item)
        return
    if not isinstance(item, dict):
        return
    if item.get("url"):
        urls.append(str(item["url"]))
    elif item.get("image_url"):
        urls.append(str(item["image_url"]))
    elif item.get("b64_json"):
        output_format = item.get("output_format") or default_format
        urls.append(f"data:image/{output_format};base64,{item['b64_json']}")


def _output_image_urls(result):
    output_format = str(result.get("output_format") or "png")
    urls = []
    for key in ("data", "images", "outputs", "output", "choices"):
        value = result.get(key)
        if isinstance(value, list):
            for item in value:
                _append_image_url(urls, item, output_format)
        elif value:
            _append_image_url(urls, value, output_format)
    _append_image_url(urls, result, output_format)
    seen = []
    for url in urls:
        if url not in seen:
            seen.append(url)
    if not seen:
        raise RuntimeError(f"No output image URL in result: {result}")
    return seen


def _download_image(url):
    if url.startswith("data:image/") and ";base64," in url:
        _, b64_data = url.split(";base64,", 1)
        img = _normalize_pil_image(Image.open(io.BytesIO(base64.b64decode(b64_data))))
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    r = requests.get(url, headers={"User-Agent": _load_user_agent()}, timeout=120)
    r.raise_for_status()
    img = _normalize_pil_image(Image.open(io.BytesIO(r.content)))
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _normalize_pil_image(img):
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        return img.convert("RGBA")
    return img.convert("RGB")


def _download_images(urls):
    tensors = [_download_image(url) for url in urls]
    if not tensors:
        raise RuntimeError("No generated images to download.")
    channels = max(t.shape[-1] for t in tensors)
    if channels == 4:
        tensors = [
            t if t.shape[-1] == 4 else torch.cat((t, torch.ones_like(t[..., :1])), dim=-1)
            for t in tensors
        ]
    return torch.cat(tensors, dim=0)


def _check(resp):
    if resp.status_code == 401:
        raise RuntimeError("Auth failed — check your API key.")
    if resp.status_code == 402:
        raise RuntimeError("Insufficient credits or billing issue.")
    if resp.status_code == 429:
        raise RuntimeError("Rate limited — please retry later.")
    if not resp.ok:
        print(f"[GPTImage2] API ERROR {resp.status_code}: {resp.text[:500]}")
        try:
            err = resp.json()
            raise RuntimeError(f"API {resp.status_code}: {err}")
        except Exception:
            raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")


# ── Nodes ──────────────────────────────────────────────────────────────────────

class GPTImage2ApiKey:
    """
    Store your OpenAI API key once and wire it to any GPT-Image-2 node.
    Alternatively, leave all api_key fields empty — nodes auto-read from
    OPENAI_API_KEY, this plugin's config.json, then ~/.muapi/config.json.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Your OpenAI API key or OpenAI-compatible provider API key.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(self, api_key):
        return (_load_api_key(api_key),)


class GPTImage2BaseUrl:
    """
    Provide an OpenAI-compatible API base URL.
    Defaults to https://api.openai.com/v1 when left blank.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_BASE_URL,
                        "tooltip": "OpenAI-compatible API base URL, for example https://api.example.com/v1",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base_url",)
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(self, base_url):
        return (_load_base_url(base_url),)


class GPTImage2TextToImage:
    """
    GPT-Image-2 Text-to-Image
    --------------------------
    Generate a high-quality image from a text prompt using GPT-Image-2.

    Endpoint: POST /v1/images/generations
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A photorealistic image of a red fox sitting in a snowy forest at dusk.",
                    },
                ),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "size": (SIZE_OPTIONS, {"default": "auto"}),
                "quality": (QUALITY_OPTIONS, {"default": "auto"}),
                "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
                "output_format": (OUTPUT_FORMAT_OPTIONS, {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "moderation": (MODERATION_OPTIONS, {"default": "auto"}),
                "stream": ("BOOLEAN", {"default": False}),
                "partial_images": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                "style": ("STRING", {"multiline": False, "default": ""}),
                "user": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(
        self,
        prompt,
        api_key="",
        base_url="",
        model="gpt-image-2",
        n=1,
        size="auto",
        quality="auto",
        background="auto",
        output_format="png",
        output_compression=100,
        moderation="auto",
        stream=False,
        partial_images=0,
        style="",
        user="",
    ):
        api_key = _load_api_key(api_key)
        base_url = _load_base_url(base_url)
        payload = {"prompt": prompt}
        payload.update(
            _image_options_payload(
                model=model,
                n=n,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                output_compression=output_compression,
                moderation=moderation,
                user=user,
                stream=stream,
                partial_images=partial_images,
                style=style,
            )
        )
        print(f"[GPTImage2 T2I] Submitting to {base_url}/images/generations...")
        result, request_id = _post_json(api_key, base_url, "images/generations", payload)
        image, url = _json_to_image_result(result)
        request_id = request_id or str(result.get("created", ""))
        print(f"[GPTImage2 T2I] Done -> {request_id or url[:80]}")
        return (image, url, request_id)


class GPTImage2ImageToImage:
    """
    GPT-Image-2 Image-to-Image
    ---------------------------
    Transform or edit up to 16 reference images guided by a text prompt.
    Common uses: style transfer, product shots, scene editing.

    Endpoint: POST /v1/images/edits

    Example prompt:
        "Transform this product image into a premium e-commerce poster style."
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Transform this product image into a premium e-commerce poster style.",
                    },
                ),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "size": (SIZE_OPTIONS, {"default": "auto"}),
                "quality": (QUALITY_OPTIONS, {"default": "auto"}),
                "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
                "input_fidelity": (INPUT_FIDELITY_OPTIONS, {"default": "auto"}),
                "output_format": (OUTPUT_FORMAT_OPTIONS, {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "moderation": (MODERATION_OPTIONS, {"default": "auto"}),
                "stream": ("BOOLEAN", {"default": False}),
                "partial_images": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                "user": ("STRING", {"multiline": False, "default": ""}),
                "mask_image": ("IMAGE",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
                "image_15": ("IMAGE",),
                "image_16": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(
        self,
        prompt,
        api_key="",
        base_url="",
        model="gpt-image-2",
        n=1,
        size="auto",
        quality="auto",
        background="auto",
        input_fidelity="auto",
        output_format="png",
        output_compression=100,
        moderation="auto",
        stream=False,
        partial_images=0,
        user="",
        mask_image=None,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        image_11=None,
        image_12=None,
        image_13=None,
        image_14=None,
        image_15=None,
        image_16=None,
    ):
        api_key = _load_api_key(api_key)
        base_url = _load_base_url(base_url)
        tensors = [
            image_1, image_2, image_3, image_4, image_5, image_6, image_7,
            image_8, image_9, image_10, image_11, image_12, image_13,
            image_14, image_15, image_16,
        ]
        files = []
        for i, img in enumerate(tensors, 1):
            if img is not None:
                files.append(("image[]", _image_tensor_to_file_tuple(img, f"image_{i}.png")))
        if not files:
            raise ValueError("At least one input image is required.")

        payload = {"prompt": prompt}
        payload.update(
            _image_options_payload(
                model=model,
                n=n,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                output_compression=output_compression,
                moderation=moderation,
                user=user,
                stream=stream,
                partial_images=partial_images,
                input_fidelity=input_fidelity,
            )
        )
        if mask_image is not None:
            files.append(("mask", _image_tensor_to_file_tuple(mask_image, "mask.png")))

        data = {key: _multipart_value(value) for key, value in payload.items()}
        print(f"[GPTImage2 I2I] Submitting ({len(files)} file(s)) to {base_url}/images/edits...")
        result, request_id = _post_multipart(
            api_key,
            base_url,
            "images/edits",
            data,
            files,
            stream_response=bool(payload.get("stream")),
        )
        image, url = _json_to_image_result(result)
        request_id = request_id or str(result.get("created_at") or result.get("created") or "")
        print(f"[GPTImage2 I2I] Done -> {request_id or url[:80]}")
        return (image, url, request_id)


class JimengApiKey:
    """
    Store a Volcano Ark/Jimeng API key once and wire it to Jimeng nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jimeng_api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Volcano Ark API key for Seedream/Jimeng models.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("jimeng_api_key",)
    FUNCTION = "run"
    CATEGORY = JIMENG_CATEGORY

    def run(self, jimeng_api_key):
        return (_load_jimeng_api_key(jimeng_api_key),)


class JimengBaseUrl:
    """
    Provide a Volcano Ark-compatible base URL.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_JIMENG_BASE_URL,
                        "tooltip": "Default: https://ark.cn-beijing.volces.com/api/v3",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base_url",)
    FUNCTION = "run"
    CATEGORY = JIMENG_CATEGORY

    def run(self, base_url):
        return (_load_jimeng_base_url(base_url),)


class JimengSeedreamImage:
    """
    Jimeng/Seedream image generation via Volcano Ark Image generation API.
    Exposes the API parameters documented by Volcano Engine instead of hiding
    them behind presets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for generation or editing.",
                    },
                ),
            },
            "optional": {
                "jimeng_api_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "model": (JIMENG_MODEL_OPTIONS, {"default": "doubao-seedream-5-0-260128"}),
                "size": (JIMENG_SIZE_OPTIONS, {"default": "2K"}),
                "custom_width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "custom_height": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "response_format": (JIMENG_RESPONSE_FORMAT_OPTIONS, {"default": "b64_json"}),
                "output_format": (JIMENG_OUTPUT_FORMAT_OPTIONS, {"default": "jpeg"}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
                "stream": ("BOOLEAN", {"default": False}),
                "sequential_image_generation": (
                    JIMENG_SEQUENTIAL_OPTIONS,
                    {"default": "disabled"},
                ),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "optimize_prompt_mode": (
                    JIMENG_OPTIMIZE_PROMPT_OPTIONS,
                    {"default": "auto"},
                ),
                "extra_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional raw JSON merged into the request payload.",
                    },
                ),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "request_id", "raw_response")
    FUNCTION = "run"
    CATEGORY = JIMENG_CATEGORY

    def run(
        self,
        prompt,
        jimeng_api_key="",
        base_url="",
        model="doubao-seedream-5-0-260128",
        size="2K",
        custom_width=2048,
        custom_height=2048,
        response_format="b64_json",
        output_format="jpeg",
        watermark=False,
        seed=-1,
        stream=False,
        sequential_image_generation="disabled",
        max_images=1,
        enable_web_search=False,
        optimize_prompt_mode="auto",
        extra_json="",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        image_11=None,
        image_12=None,
        image_13=None,
        image_14=None,
    ):
        api_key = _load_jimeng_api_key(jimeng_api_key)
        base_url = _load_jimeng_base_url(base_url)

        size_value = (
            f"{int(custom_width)}x{int(custom_height)}"
            if size == "custom"
            else size.split(" ", 1)[0]
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_value,
            "response_format": response_format,
            "output_format": output_format,
            "watermark": bool(watermark),
            "stream": bool(stream),
        }
        if int(seed) >= 0:
            payload["seed"] = int(seed)

        tensors = [
            image_1, image_2, image_3, image_4, image_5, image_6, image_7,
            image_8, image_9, image_10, image_11, image_12, image_13, image_14,
        ]
        image_inputs = [_image_tensor_to_data_url(img) for img in tensors if img is not None]
        if image_inputs:
            payload["image"] = image_inputs[0] if len(image_inputs) == 1 else image_inputs

        if sequential_image_generation != "disabled":
            payload["sequential_image_generation"] = sequential_image_generation
            payload["sequential_image_generation_options"] = {
                "max_images": int(max_images),
            }

        if enable_web_search:
            payload["tools"] = [{"type": "web_search"}]

        if optimize_prompt_mode != "disabled":
            mode = "standard" if optimize_prompt_mode == "auto" else optimize_prompt_mode
            payload["optimize_prompt_options"] = {"mode": mode}

        if extra_json and extra_json.strip():
            try:
                extra_payload = json.loads(extra_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"extra_json is not valid JSON: {exc}") from exc
            if not isinstance(extra_payload, dict):
                raise ValueError("extra_json must decode to a JSON object.")
            payload.update(extra_payload)

        if payload.get("sequential_image_generation") == "auto":
            requested_total = len(image_inputs) + int(
                payload.get("sequential_image_generation_options", {}).get("max_images", 1)
            )
            if requested_total > 15:
                raise ValueError(
                    "Jimeng group generation supports reference images + generated images <= 15."
                )

        print(f"[Jimeng Seedream] Submitting to {base_url}/images/generations...")
        result, request_id = _post_jimeng_json(api_key, base_url, payload)
        urls = _output_image_urls(result)
        images = _download_images(urls)
        raw_response = json.dumps(result, ensure_ascii=False)
        request_id = request_id or str(result.get("id") or result.get("created") or "")
        print(f"[Jimeng Seedream] Done -> {len(urls)} image(s)")
        return (images, "\n".join(urls), request_id, raw_response)


class MangoAIGCCredentials:
    """
    Store Mango AIGC access/secret keys once and wire them to Wan2.7 nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mango_access_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Mango AIGC Access-Key.",
                    },
                ),
                "mango_secret_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Mango AIGC Secret-Key used to sign requests.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mango_access_key", "mango_secret_key")
    FUNCTION = "run"
    CATEGORY = MANGO_CATEGORY

    def run(self, mango_access_key, mango_secret_key):
        return (
            _load_mango_access_key(mango_access_key),
            _load_mango_secret_key(mango_secret_key),
        )


class MangoAIGCBaseUrl:
    """
    Provide a Mango AIGC API base URL.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_MANGO_BASE_URL,
                        "tooltip": "Default: https://aigc.mgtv.com",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base_url",)
    FUNCTION = "run"
    CATEGORY = MANGO_CATEGORY

    def run(self, base_url):
        return (_load_mango_base_url(base_url),)


class HelloImgToken:
    """
    Store a Hello image host token once and wire it to Wan2.7 uploads.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "helloimg_token": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Hello image host API token.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("helloimg_token",)
    FUNCTION = "run"
    CATEGORY = MANGO_CATEGORY

    def run(self, helloimg_token):
        return (_load_helloimg_token(helloimg_token),)


class MangoWan27TextToImage:
    """
    Mango AIGC Wan2.7 text-to-image.

    Submit POST /openapi/v1/storyboard/generateByPromptV2 and poll
    /openapi/v1/storyboard/getAssetInfo until image URLs are available.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for Wan2.7 image generation.",
                    },
                ),
            },
            "optional": {
                "mango_access_key": ("STRING", {"multiline": False, "default": ""}),
                "mango_secret_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "helloimg_token": ("STRING", {"multiline": False, "default": ""}),
                "helloimg_base_url": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
                "model": (MANGO_WAN27_STYLE_OPTIONS, {"default": "Wan2.7-image-pro (35)"}),
                "style_id": ("INT", {"default": 35, "min": 1, "max": 9999, "step": 1}),
                "ratio": (MANGO_RATIO_OPTIONS, {"default": "16:9"}),
                "resolution": (MANGO_RESOLUTION_OPTIONS, {"default": "2K"}),
                "nums": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
                "img_urls": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional reference image URLs, one per line. Sent as imgUrls.",
                    },
                ),
                "prompt_args_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "Optional prompt args JSON array merged into prompt.args.",
                    },
                ),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1}),
                "timeout_seconds": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 30}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "extra_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional raw JSON merged into the generateByPromptV2 payload.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "asset_record_id", "raw_response")
    FUNCTION = "run"
    CATEGORY = MANGO_CATEGORY

    def run(
        self,
        prompt,
        mango_access_key="",
        mango_secret_key="",
        base_url="",
        helloimg_token="",
        helloimg_base_url="",
        model="Wan2.7-image-pro (35)",
        style_id=35,
        ratio="16:9",
        resolution="2K",
        nums=1,
        seed=-1,
        img_urls="",
        prompt_args_json="[]",
        poll_interval=3,
        timeout_seconds=600,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        extra_json="",
    ):
        access_key = _load_mango_access_key(mango_access_key)
        secret_key = _load_mango_secret_key(mango_secret_key)
        base_url = _load_mango_base_url(base_url)

        style_from_model = str(model).rsplit("(", 1)[-1].rstrip(")")
        resolved_style_id = int(style_from_model) if style_from_model.isdigit() else int(style_id)

        reference_urls = [
            line.strip()
            for line in str(img_urls).replace(",", "\n").splitlines()
            if line.strip()
        ]

        upload_images = [
            image
            for image in (image_1, image_2, image_3, image_4, image_5, image_6)
            if image is not None
        ]
        upload_responses = []
        if upload_images:
            token = _load_helloimg_token(helloimg_token)
            upload_base_url = _load_helloimg_base_url(helloimg_base_url)
            print(f"[Mango Wan2.7] Uploading {len(upload_images)} reference image(s) to Hello image host...")
            uploaded_urls, upload_responses = _upload_helloimg_images(
                upload_images,
                token,
                upload_base_url,
            )
            reference_urls.extend(uploaded_urls)

        if len(reference_urls) > 6:
            raise ValueError("Mango Wan2.7 supports at most 6 reference images.")

        if prompt_args_json and prompt_args_json.strip():
            try:
                prompt_args = json.loads(prompt_args_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"prompt_args_json is not valid JSON: {exc}") from exc
            if not isinstance(prompt_args, list):
                raise ValueError("prompt_args_json must decode to a JSON array.")
        else:
            prompt_args = []

        payload = {
            "styleId": resolved_style_id,
            "ratio": ratio,
            "resolution": resolution,
            "nums": int(nums),
            "imgUrls": reference_urls,
            "prompt": {
                "args": prompt_args,
                "prompt": prompt,
            },
        }
        if int(seed) >= 0:
            payload["seed"] = int(seed)

        if extra_json and extra_json.strip():
            try:
                extra_payload = json.loads(extra_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"extra_json is not valid JSON: {exc}") from exc
            if not isinstance(extra_payload, dict):
                raise ValueError("extra_json must decode to a JSON object.")
            payload.update(extra_payload)

        print(f"[Mango Wan2.7] Submitting to {base_url}/openapi/v1/storyboard/generateByPromptV2...")
        submit_result, submit_request_id = _post_mango_json(
            access_key,
            secret_key,
            base_url,
            "openapi/v1/storyboard/generateByPromptV2",
            payload,
        )
        record_id = _mango_record_id(submit_result)
        print(f"[Mango Wan2.7] Asset record id: {record_id}. Polling asset info...")
        asset_result, asset_request_id = _poll_mango_asset(
            access_key,
            secret_key,
            base_url,
            record_id,
            poll_interval,
            timeout_seconds,
        )
        urls = _mango_asset_urls(asset_result)
        images = _download_images(urls)
        raw_response = json.dumps(
            {
                "submit": submit_result,
                "asset": asset_result,
                "uploaded_reference_images": upload_responses,
                "request_id": asset_request_id or submit_request_id,
            },
            ensure_ascii=False,
        )
        print(f"[Mango Wan2.7] Done -> {len(urls)} image(s)")
        return (images, "\n".join(urls), record_id, raw_response)


class GridCropImages:
    """
    Split each input image into a row-major grid and return all tiles as one IMAGE batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cols": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "rows": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "fit_mode": (["pad_edge", "crop_remainder"], {"default": "pad_edge"}),
                "col_start_offset": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "col_end_offset": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "row_start_offset": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "row_end_offset": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("images", "tile_width", "tile_height", "masks")
    FUNCTION = "crop"
    CATEGORY = IMAGE_CATEGORY

    def crop(
        self,
        image,
        cols=4,
        rows=4,
        fit_mode="pad_edge",
        col_start_offset=0,
        col_end_offset=0,
        row_start_offset=0,
        row_end_offset=0,
        mask=None,
    ):
        cols = int(cols)
        rows = int(rows)
        if cols < 1 or rows < 1:
            raise ValueError("cols and rows must be >= 1.")

        col_start_offset = int(col_start_offset)
        col_end_offset = int(col_end_offset)
        row_start_offset = int(row_start_offset)
        row_end_offset = int(row_end_offset)
        offsets = (col_start_offset, col_end_offset, row_start_offset, row_end_offset)
        if any(offset < 0 for offset in offsets):
            raise ValueError("Grid crop offsets must be >= 0.")

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"Expected IMAGE tensor with 3 or 4 dimensions, got {image.dim()}.")

        batch, height, width, channels = image.shape
        mask = self._normalize_mask(mask, batch, height, width, image.device, image.dtype)

        crop_left = col_start_offset
        crop_right = width - col_end_offset
        crop_top = row_start_offset
        crop_bottom = height - row_end_offset
        if crop_left >= crop_right or crop_top >= crop_bottom:
            raise ValueError(
                f"Grid crop offsets leave no image area: image size {width}x{height}, "
                f"col offsets {col_start_offset}/{col_end_offset}, "
                f"row offsets {row_start_offset}/{row_end_offset}."
            )

        if any(offsets):
            image = image[:, crop_top:crop_bottom, crop_left:crop_right, :]
            mask = mask[:, crop_top:crop_bottom, crop_left:crop_right]
            batch, height, width, channels = image.shape

        if fit_mode == "crop_remainder":
            if height < rows or width < cols:
                raise ValueError(
                    f"Image size {width}x{height} is too small for a {cols}x{rows} grid "
                    "when fit_mode is crop_remainder."
                )
            tile_width = width // cols
            tile_height = height // rows
            fit_width = tile_width * cols
            fit_height = tile_height * rows
            image = image[:, :fit_height, :fit_width, :]
            mask = mask[:, :fit_height, :fit_width]
        elif fit_mode == "pad_edge":
            tile_width = (width + cols - 1) // cols
            tile_height = (height + rows - 1) // rows
            fit_width = tile_width * cols
            fit_height = tile_height * rows
            pad_width = fit_width - width
            pad_height = fit_height - height
            if pad_width or pad_height:
                nchw = image.movedim(-1, 1)
                image = torch.nn.functional.pad(
                    nchw,
                    (0, pad_width, 0, pad_height),
                    mode="replicate",
                ).movedim(1, -1)
                mask = torch.nn.functional.pad(
                    mask.unsqueeze(1),
                    (0, pad_width, 0, pad_height),
                    mode="replicate",
                ).squeeze(1)
        else:
            raise ValueError(f"Unsupported fit_mode: {fit_mode}")

        tiles = (
            image.contiguous()
            .view(batch, rows, tile_height, cols, tile_width, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(batch * rows * cols, tile_height, tile_width, channels)
        )
        mask_tiles = (
            mask.contiguous()
            .view(batch, rows, tile_height, cols, tile_width)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(batch * rows * cols, tile_height, tile_width)
        )
        return (tiles, tile_width, tile_height, mask_tiles)

    @staticmethod
    def _normalize_mask(mask, batch, height, width, device, dtype):
        if mask is None:
            return torch.zeros((batch, height, width), device=device, dtype=dtype)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        if mask.dim() != 3:
            raise ValueError(f"Expected MASK tensor with 2 or 3 dimensions, got {mask.dim()}.")

        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[1] != height or mask.shape[2] != width:
            raise ValueError(
                f"Mask size {mask.shape[2]}x{mask.shape[1]} does not match image size {width}x{height}."
            )

        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1)
        elif mask.shape[0] != batch:
            raise ValueError(f"Mask batch size {mask.shape[0]} does not match image batch size {batch}.")

        return mask


NODE_CLASS_MAPPINGS = {
    "GPTImage2ApiKey":        GPTImage2ApiKey,
    "GPTImage2BaseUrl":       GPTImage2BaseUrl,
    "GPTImage2TextToImage":   GPTImage2TextToImage,
    "GPTImage2ImageToImage":  GPTImage2ImageToImage,
    "JimengApiKey":           JimengApiKey,
    "JimengBaseUrl":          JimengBaseUrl,
    "JimengSeedreamImage":    JimengSeedreamImage,
    "MangoAIGCCredentials":   MangoAIGCCredentials,
    "MangoAIGCBaseUrl":       MangoAIGCBaseUrl,
    "HelloImgToken":          HelloImgToken,
    "MangoWan27TextToImage":  MangoWan27TextToImage,
    "GridCropImages":         GridCropImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2ApiKey":        "🔑 GPT-Image-2 API Key",
    "GPTImage2BaseUrl":       "🌐 GPT-Image-2 Base URL",
    "GPTImage2TextToImage":   "🖼️ GPT-Image-2 Text to Image",
    "GPTImage2ImageToImage":  "🖼️ GPT-Image-2 Image to Image",
    "JimengApiKey":           "🔑 Jimeng API Key",
    "JimengBaseUrl":          "🌐 Jimeng Base URL",
    "JimengSeedreamImage":    "🖼️ Jimeng Seedream Image",
    "MangoAIGCCredentials":   "🔑 Mango AIGC Credentials",
    "MangoAIGCBaseUrl":       "🌐 Mango AIGC Base URL",
    "HelloImgToken":          "🔑 Hello Image Host Token",
    "MangoWan27TextToImage":  "🖼️ Mango Wan2.7 Text to Image",
    "GridCropImages":         "✂️ Grid Crop Images",
}
