import cv2

from aikensa.camscripts.ic4_camera import get_camera_settings


def _resolve_opencv_source(identifier, fallback_id):
    if isinstance(identifier, str):
        stripped = identifier.strip()
        if stripped.isdigit():
            return int(stripped)
        if stripped:
            return stripped
        return fallback_id

    if identifier is None:
        return fallback_id

    return identifier


def initialize_hole_camera(camNum):
    settings = get_camera_settings(camNum)
    source = _resolve_opencv_source(settings.get("identifier"), camNum)
    api_preferences = []

    if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW"):
        api_preferences.append(cv2.CAP_DSHOW)
    api_preferences.append(cv2.CAP_ANY)

    for api_preference in api_preferences:
        capture = cv2.VideoCapture(source, api_preference)
        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(settings["width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings["height"]))
        capture.set(cv2.CAP_PROP_FPS, float(settings["fps"]))

        print(f"[OpenCV] Initialized hole camera cam{camNum} using source {source}")
        return capture

    print(f"[OpenCV] Failed to initialize hole camera cam{camNum} using source {source}")
    return cv2.VideoCapture()
