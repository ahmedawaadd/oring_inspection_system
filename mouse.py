"""Mouse state and OpenCV mouse callback for drawing reference regions.

OpenCV's mouse callback is a separate function that can't return values to
the main loop directly. A shared dictionary is the workaround: both the
callback and the main loop read and write the same object, so changes in one
are immediately visible in the other. The state is created by the main loop
and handed to OpenCV via the callback's ``param`` argument (rather than a
module-level global), so ownership stays explicit."""

import cv2


def new_mouse_state():
    """Return a fresh mouse-state dictionary."""
    return {
        "active_slot": None,  # which slot (1 or 2) the user is drawing for right now
        "drawing":     False, # true while the left mouse button is held down
        "pt1":         (0, 0),
        "pt2":         (0, 0),
        "roi_ready":   False, # flipped to True on mouse-up so the main loop knows to act
    }


def on_mouse(event, x, y, flags, param):
    mouse = param
    # Ignore mouse events if no slot is active (user hasn't pressed 1 or 2 yet)
    if mouse["active_slot"] is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.update(drawing=True, roi_ready=False, pt1=(x, y), pt2=(x, y))
    elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
        # Keep updating pt2 as the mouse moves
        mouse["pt2"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and mouse["drawing"]:
        # signal the main loop on mouse release to capture the reference
        mouse.update(drawing=False, pt2=(x, y), roi_ready=True)
