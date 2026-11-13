"""Fast shaded mannequin overlays built from 2D pose landmarks.

The renderer uses only OpenCV primitives, so it stays inexpensive enough for
the live camera loop and does not need a 3D engine or GPU.
"""

from __future__ import annotations

import numpy as np


def draw_mannequin(image, points, color=(135, 220, 175), alpha=0.58,
                   visibility=None, width_scale=1.0) -> None:
    """Draw a softly shaded, translucent body over ``image`` in place.

    ``points`` is a 33x2 array of image pixel coordinates in MediaPipe order.
    Invisible or off-screen joints are skipped; the drawing is clipped to the
    image bounds and composited only inside the mannequin's small bounding box.
    """
    import cv2

    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 29:
        return
    h, w = image.shape[:2]
    visible = (np.ones(len(pts), dtype=bool) if visibility is None
               else np.asarray(visibility, dtype=bool))
    if len(visible) < len(pts):
        visible = np.pad(visible, (0, len(pts) - len(visible)), constant_values=False)
    valid_ids = [i for i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
                 if i < len(pts) and visible[i] and np.isfinite(pts[i]).all()]
    if len(valid_ids) < 6:
        return

    shoulder_mid = (pts[11] + pts[12]) * 0.5
    hip_mid = (pts[23] + pts[24]) * 0.5
    torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
    shoulder_width = float(np.linalg.norm(pts[11] - pts[12]))
    if torso_len < 5 or shoulder_width < 3:
        return

    # Estimate a padded local patch; all geometry is drawn in patch coordinates.
    joint_radius = max(2, int(round(shoulder_width * 0.095 * width_scale)))
    widths = {
        "upper_arm": max(3, int(round(shoulder_width * 0.105 * width_scale))),
        "forearm": max(3, int(round(shoulder_width * 0.082 * width_scale))),
        "thigh": max(4, int(round(shoulder_width * 0.16 * width_scale))),
        "shin": max(3, int(round(shoulder_width * 0.105 * width_scale))),
    }
    head_r = max(4, int(round(shoulder_width * 0.22 * width_scale)))
    pad = max(head_r * 2, int(shoulder_width * 0.22)) + 4
    low = np.nanmin(pts[valid_ids], axis=0) - pad
    high = np.nanmax(pts[valid_ids], axis=0) + pad
    x0, y0 = max(0, int(low[0])), max(0, int(low[1]))
    x1, y1 = min(w, int(high[0]) + 1), min(h, int(high[1]) + 1)
    if x1 <= x0 or y1 <= y0:
        return

    def local(i):
        return tuple(np.rint(pts[i] - (x0, y0)).astype(int))

    patch = image[y0:y1, x0:x1]
    overlay = np.zeros_like(patch)
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    base = tuple(int(np.clip(c, 0, 255)) for c in color)
    shade = tuple(int(c * 0.58) for c in base)
    highlight = tuple(int(min(255, c * 1.23 + 18)) for c in base)

    def capsule(a, b, width):
        if not (visible[a] and visible[b]):
            return
        p1, p2 = local(a), local(b)
        outer = width + max(2, width // 3)
        cv2.line(overlay, p1, p2, shade, outer, cv2.LINE_AA)
        cv2.circle(overlay, p1, outer // 2, shade, -1, cv2.LINE_AA)
        cv2.circle(overlay, p2, outer // 2, shade, -1, cv2.LINE_AA)
        cv2.line(overlay, p1, p2, base, width, cv2.LINE_AA)
        cv2.circle(overlay, p1, width // 2, base, -1, cv2.LINE_AA)
        cv2.circle(overlay, p2, width // 2, base, -1, cv2.LINE_AA)
        # Narrow offset highlight gives the rounded limb a simple 3D sheen.
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max(1.0, float(np.hypot(dx, dy)))
        ox, oy = int(round(-dy / length * max(1, width // 5))), int(round(dx / length * max(1, width // 5)))
        cv2.line(overlay, (p1[0] + ox, p1[1] + oy), (p2[0] + ox, p2[1] + oy),
                 highlight, max(1, width // 5), cv2.LINE_AA)
        cv2.line(mask, p1, p2, 255, outer, cv2.LINE_AA)
        cv2.circle(mask, p1, outer // 2, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, p2, outer // 2, 255, -1, cv2.LINE_AA)

    def joint(i, radius=joint_radius):
        if visible[i]:
            center = local(i)
            cv2.circle(overlay, center, radius + 1, shade, -1, cv2.LINE_AA)
            cv2.circle(overlay, center, radius, base, -1, cv2.LINE_AA)
            cv2.ellipse(overlay, (center[0] - radius // 3, center[1] - radius // 3),
                        (max(1, radius // 3), max(1, radius // 3)), 0, 0, 360,
                        highlight, -1, cv2.LINE_AA)
            cv2.circle(mask, center, radius + 1, 255, -1, cv2.LINE_AA)

    # Limbs go behind the torso for a more natural silhouette.
    for a, b, key in ((23, 25, "thigh"), (25, 27, "shin"),
                      (24, 26, "thigh"), (26, 28, "shin"),
                      (11, 13, "upper_arm"), (13, 15, "forearm"),
                      (12, 14, "upper_arm"), (14, 16, "forearm")):
        capsule(a, b, widths[key])

    # Torso and pelvis are softly beveled polygons based on shoulder/hip width.
    if all(visible[i] for i in (11, 12, 23, 24)):
        shoulders = [np.asarray(local(11)), np.asarray(local(12))]
        hips = [np.asarray(local(23)), np.asarray(local(24))]
        sh_mid = (shoulders[0] + shoulders[1]) * 0.5
        hp_mid = (hips[0] + hips[1]) * 0.5
        sh_vec = shoulders[1] - shoulders[0]
        hp_vec = hips[1] - hips[0]
        sh_vec = sh_vec / max(1.0, np.linalg.norm(sh_vec)) * shoulder_width * 0.40
        hp_vec = hp_vec / max(1.0, np.linalg.norm(hp_vec)) * shoulder_width * 0.31
        torso = np.asarray([sh_mid - sh_vec, sh_mid + sh_vec,
                            hp_mid + hp_vec, hp_mid - hp_vec], dtype=np.int32)
        cv2.fillConvexPoly(overlay, torso, base, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, torso, 255, cv2.LINE_AA)
        # Center highlight and side shadow suggest a rounded chest and abdomen.
        mid_top = tuple(np.rint(sh_mid).astype(int))
        mid_bottom = tuple(np.rint(hp_mid).astype(int))
        cv2.line(overlay, mid_top, mid_bottom, highlight,
                 max(1, int(shoulder_width * 0.045)), cv2.LINE_AA)
        pelvis_radius = max(3, int(shoulder_width * 0.13))
        cv2.ellipse(overlay, tuple(np.rint(hp_mid).astype(int)),
                    (int(shoulder_width * 0.34), pelvis_radius), 0, 0, 360,
                    shade, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, tuple(np.rint(hp_mid).astype(int)),
                    (int(shoulder_width * 0.31), max(2, pelvis_radius - 1)), 0, 0, 360,
                    base, -1, cv2.LINE_AA)
        cv2.ellipse(mask, tuple(np.rint(hp_mid).astype(int)),
                    (int(shoulder_width * 0.34), pelvis_radius), 0, 0, 360,
                    255, -1, cv2.LINE_AA)

    # Neck joins the clavicle to a head centered just above the nose landmark.
    head_center = (pts[0] if visible[0] else shoulder_mid + np.array([0, -0.55 * torso_len]))
    head_center = head_center + np.array([0, -head_r * 0.35])
    head_center_local = tuple(np.rint(head_center - (x0, y0)).astype(int))
    neck_top = tuple(np.rint(shoulder_mid + (head_center - shoulder_mid) * 0.68 - (x0, y0)).astype(int))
    neck_bottom = tuple(np.rint(shoulder_mid - (x0, y0)).astype(int))
    neck_width = max(3, int(shoulder_width * 0.16))
    cv2.line(overlay, neck_bottom, neck_top, base, neck_width, cv2.LINE_AA)
    cv2.line(mask, neck_bottom, neck_top, 255, neck_width, cv2.LINE_AA)
    cv2.ellipse(overlay, head_center_local, (head_r, int(head_r * 1.22)), 0, 0, 360,
                shade, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (head_center_local[0] - head_r // 4,
                          head_center_local[1] - head_r // 4),
                (max(2, head_r - 2), int(head_r * 1.12)), 0, 0, 360,
                base, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (head_center_local[0] - head_r // 3,
                          head_center_local[1] - head_r // 2),
                (max(1, head_r // 4), max(1, head_r // 3)), 0, 0, 360,
                highlight, -1, cv2.LINE_AA)
    cv2.ellipse(mask, head_center_local, (head_r, int(head_r * 1.22)), 0, 0, 360,
                255, -1, cv2.LINE_AA)

    for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        joint(i, max(2, int(joint_radius * (1.05 if i in (11, 12, 23, 24) else 0.78))))

    blended = cv2.addWeighted(patch, 1.0 - float(alpha), overlay, float(alpha), 0)
    np.copyto(patch, blended, where=(mask > 0)[..., None])
