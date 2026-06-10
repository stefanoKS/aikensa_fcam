import atexit
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

try:
    import imagingcontrol4 as ic4
except ImportError:
    ic4 = None


CAMERA_CONFIG_PATH = Path(__file__).resolve().parent.parent / "camera_serial" / "cameraSerial.yaml"

MAIN_CAMERA_DEFAULTS = {
    "width": 3072,
    "height": 2048,
    "fps": 5,
    "color": True,
    "rotate_180": False,
    "exposure_us": 10000,
    "gain_db": 10,
    "wb_temperature": 4500,
    "wb_ratio_r": 2.2,
    "wb_ratio_g": 1.0,
    "wb_ratio_b": 2.2,
    "auto_exposure": False,
    "auto_gain": False,
    "auto_wb": False,
}

HOLE_CAMERA_DEFAULTS = {
    **MAIN_CAMERA_DEFAULTS,
    "width": 1280,
    "height": 720,
    "fps": 30,
}


_IC4_CTX = None


def _ensure_ic4_context() -> bool:
    global _IC4_CTX

    if ic4 is None:
        return False

    if _IC4_CTX is None:
        _IC4_CTX = ic4.Library.init_context(
            api_log_level=ic4.LogLevel.WARNING,
            log_targets=ic4.LogTarget.STDERR,
        )
        _IC4_CTX.__enter__()
        atexit.register(_IC4_CTX.__exit__, None, None, None)

    return True


def _load_camera_config() -> dict:
    if not CAMERA_CONFIG_PATH.exists():
        return {}

    with CAMERA_CONFIG_PATH.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    return loaded if isinstance(loaded, dict) else {}


def _coerce_identifier(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return stripped
    return value


def _default_camera_settings(logical_camera_id: int) -> dict:
    base = HOLE_CAMERA_DEFAULTS if logical_camera_id == 0 else MAIN_CAMERA_DEFAULTS
    return {
        **base,
        "identifier": logical_camera_id,
    }


def get_camera_settings(logical_camera_id: int) -> dict:
    config = _load_camera_config()
    key = f"cam{logical_camera_id}"
    loaded_entry = config.get(key, {})
    if not isinstance(loaded_entry, dict):
        loaded_entry = {"identifier": loaded_entry}

    shared_inspection_settings = config.get("inspection_defaults", {}) if logical_camera_id != 0 else {}
    if not isinstance(shared_inspection_settings, dict):
        shared_inspection_settings = {}

    normalized_entry = dict(loaded_entry)
    if "serial" in normalized_entry and "identifier" not in normalized_entry:
        serial_value = normalized_entry["serial"]
        normalized_entry["identifier"] = None if serial_value is None else str(serial_value).strip()

    for alias in ("id", "camera_id"):
        if alias in normalized_entry and "identifier" not in normalized_entry:
            normalized_entry["identifier"] = normalized_entry[alias]

    top_level_rotate = bool(config.get("ic4_rotate_180", False))
    settings = _default_camera_settings(logical_camera_id)
    settings["rotate_180"] = top_level_rotate
    settings.update({k: v for k, v in shared_inspection_settings.items() if v is not None})
    settings.update({k: v for k, v in normalized_entry.items() if v is not None})
    settings["identifier"] = _coerce_identifier(settings.get("identifier"))

    if logical_camera_id != 0 and settings["identifier"] is not None:
        settings["identifier"] = str(settings["identifier"]).strip()

    return settings


def _find_device(identifier: Union[int, str]) -> Optional[object]:
    if not _ensure_ic4_context():
        return None

    try:
        devices = ic4.DeviceEnum.devices()
    except Exception as error:
        print(f"[IC4] Failed to enumerate devices: {error}")
        return None

    if not devices:
        print("[IC4] No devices found.")
        return None

    if isinstance(identifier, int):
        if 0 <= identifier < len(devices):
            return devices[identifier]
        print(f"[IC4] Camera index {identifier} out of range.")
        return None

    for device in devices:
        if getattr(device, "serial", None) == identifier:
            return device

    for device in devices:
        model_name = getattr(device, "model_name", "") or ""
        if identifier.lower() in model_name.lower():
            return device

    print(f"[IC4] No device matched '{identifier}'.")
    return None


def _try_set(property_map, prop_ids: Iterable, value) -> bool:
    if ic4 is None:
        return False

    for prop_id in prop_ids:
        if prop_id is None:
            continue
        try:
            property_map.set_value(prop_id, value)
            return True
        except ic4.IC4Exception:
            continue
    return False


def _try_get_str(property_map, prop_ids: Iterable) -> Optional[str]:
    if ic4 is None:
        return None

    for prop_id in prop_ids:
        if prop_id is None:
            continue
        try:
            return property_map.get_value_str(prop_id)
        except ic4.IC4Exception:
            continue
    return None


def _apply_manual_controls(property_map, settings: dict) -> None:
    if ic4 is None:
        return

    exposure_auto = getattr(ic4.PropId, "EXPOSURE_AUTO", None)
    exposure_time = getattr(ic4.PropId, "EXPOSURE_TIME", None)
    gain_auto = getattr(ic4.PropId, "GAIN_AUTO", None)
    gain = getattr(ic4.PropId, "GAIN", None)
    wb_auto = getattr(ic4.PropId, "BALANCE_WHITE_AUTO", None)
    wb_temp = getattr(ic4.PropId, "WHITEBALANCE_TEMPERATURE", None)
    balance_selector = getattr(ic4.PropId, "BALANCE_RATIO_SELECTOR", None)
    balance_ratio = getattr(ic4.PropId, "BALANCE_RATIO", None)

    _try_set(property_map, (exposure_auto,), "Off")
    _try_set(property_map, (gain_auto,), "Off")
    _try_set(property_map, (wb_auto,), "Off")
    _try_set(property_map, (exposure_time,), float(settings["exposure_us"]))
    _try_set(property_map, (gain,), float(settings["gain_db"]))

    applied_ratio = False
    if balance_selector is not None and balance_ratio is not None:
        for selector_name, ratio_key in (("Red", "wb_ratio_r"), ("Green", "wb_ratio_g"), ("Blue", "wb_ratio_b")):
            try:
                property_map.set_value(balance_selector, selector_name)
                property_map.set_value(balance_ratio, float(settings[ratio_key]))
                applied_ratio = True
            except ic4.IC4Exception:
                continue

    if not applied_ratio:
        _try_set(property_map, (wb_temp,), int(settings["wb_temperature"]))


class DummyCapture:
    def __init__(self, width: int, height: int, label: str, rotate_180: bool = False):
        self._width = int(width)
        self._height = int(height)
        self._label = label
        self._rotate_180 = bool(rotate_180)
        self._open = False

    def isOpened(self) -> bool:
        return self._open

    def release(self):
        self._open = False

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open:
            return False, None

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        cv2.putText(frame, "IC4 camera unavailable", (40, max(80, self._height // 2 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, self._label, (40, max(130, self._height // 2 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
        if self._rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id == getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3):
            return float(self._width)
        if prop_id == getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4):
            return float(self._height)
        if prop_id == getattr(cv2, "CAP_PROP_FPS", 5):
            return 0.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return True


class IC4Capture:
    def __init__(self, settings: dict):
        if not _ensure_ic4_context():
            raise RuntimeError("imagingcontrol4 module is not available")

        device = _find_device(settings["identifier"])
        if device is None:
            raise RuntimeError(f"No IC4 device found for '{settings['identifier']}'")

        self._width = int(settings["width"])
        self._height = int(settings["height"])
        self._fps = float(settings["fps"])
        self._rotate_180 = bool(settings.get("rotate_180", False))
        self._open = True

        self._grabber = ic4.Grabber(device)
        property_map = self._grabber.device_property_map

        pixel_format = getattr(ic4.PropId, "PIXEL_FORMAT", None)
        width_prop = getattr(ic4.PropId, "WIDTH", None)
        height_prop = getattr(ic4.PropId, "HEIGHT", None)
        offset_x = getattr(ic4.PropId, "OFFSET_X", None)
        offset_y = getattr(ic4.PropId, "OFFSET_Y", None)
        offset_auto_center = getattr(ic4.PropId, "OFFSET_AUTO_CENTER", None)
        reverse_x = getattr(ic4.PropId, "REVERSE_X", None)
        reverse_y = getattr(ic4.PropId, "REVERSE_Y", None)
        acquisition_frame_rate = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE", None)
        acquisition_frame_rate_enable = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE_ENABLE", None)
        frame_rate = getattr(ic4.PropId, "FRAME_RATE", None)

        pixel_format_value = getattr(ic4.PixelFormat, "BayerRG8", None) if settings.get("color", True) else getattr(ic4.PixelFormat, "Mono8", None)
        if pixel_format_value is not None:
            _try_set(property_map, (pixel_format,), pixel_format_value)

        _try_set(property_map, (offset_auto_center,), "Off")
        _try_set(property_map, (offset_x,), 0)
        _try_set(property_map, (offset_y,), 0)
        _try_set(property_map, (width_prop,), self._width)
        _try_set(property_map, (height_prop,), self._height)

        rx_ok = _try_set(property_map, (reverse_x,), self._rotate_180)
        ry_ok = _try_set(property_map, (reverse_y,), self._rotate_180)
        self._software_rotate = self._rotate_180 and not (rx_ok and ry_ok)

        _apply_manual_controls(property_map, settings)
        _try_set(property_map, (acquisition_frame_rate_enable,), True)
        _try_set(property_map, (acquisition_frame_rate, frame_rate), self._fps)

        self._sink = ic4.SnapSink()
        self._grabber.stream_setup(self._sink)

        pixel_format_name = _try_get_str(property_map, (pixel_format,))
        self._conversion_code = None
        if pixel_format_name == "RGB8":
            self._conversion_code = cv2.COLOR_RGB2BGR
        elif pixel_format_name == "BayerRG8":
            self._conversion_code = cv2.COLOR_BayerRG2BGR
        elif pixel_format_name == "BayerBG8":
            self._conversion_code = cv2.COLOR_BayerBG2BGR
        elif pixel_format_name == "BayerGR8":
            self._conversion_code = cv2.COLOR_BayerGR2BGR
        elif pixel_format_name == "BayerGB8":
            self._conversion_code = cv2.COLOR_BayerGB2BGR
        elif pixel_format_name in {"YUV422Packed", "YUY2"}:
            self._conversion_code = cv2.COLOR_YUV2BGR_YUY2

    def isOpened(self) -> bool:
        return self._open

    def release(self):
        if not self._open:
            return

        try:
            self._grabber.stream_stop()
        finally:
            self._grabber.device_close()
            self._open = False

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open:
            return False, None

        try:
            buffer = self._sink.snap_single(int(timeout_ms))
        except ic4.IC4Exception:
            return False, None

        if buffer is None:
            return False, None

        frame = buffer.numpy_copy()
        if self._conversion_code is not None:
            frame = cv2.cvtColor(frame, self._conversion_code)
        if self._software_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id == getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3):
            return float(self._width)
        if prop_id == getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4):
            return float(self._height)
        if prop_id == getattr(cv2, "CAP_PROP_FPS", 5):
            return float(self._fps)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        property_map = self._grabber.device_property_map
        acquisition_frame_rate = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE", None)
        frame_rate = getattr(ic4.PropId, "FRAME_RATE", None)
        acquisition_frame_rate_enable = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE_ENABLE", None)

        if prop_id == getattr(cv2, "CAP_PROP_FPS", 5):
            _try_set(property_map, (acquisition_frame_rate_enable,), True)
            success = _try_set(property_map, (acquisition_frame_rate, frame_rate), float(value))
            if success:
                self._fps = float(value)
            return success

        return False


def initialize_camera_ic4(logical_camera_id: int):
    settings = get_camera_settings(logical_camera_id)

    try:
        capture = IC4Capture(settings)
        print(
            f"[IC4] Initialized camera cam{logical_camera_id} using identifier {settings['identifier']} "
            f"(fps={settings['fps']}, exposure_us={settings['exposure_us']})"
        )
        return capture
    except Exception as error:
        print(f"[IC4] Failed to initialize cam{logical_camera_id}: {error}")
        return DummyCapture(
            settings["width"],
            settings["height"],
            f"cam{logical_camera_id}",
            rotate_180=settings.get("rotate_180", False),
        )