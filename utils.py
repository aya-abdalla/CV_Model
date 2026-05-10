
import io

import gradio as gr
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor

def generate_bounding_boxes(image):
    with open(image, "rb") as f:
             image_content = f.read() 
    result = client.analyze(
    image_data=image_content,
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    gender_neutral_caption=True,  # Optional (default is False)
)
    if result.caption is not None:
       caption=result.caption.text
       print(f"Caption: '{result.caption.text}', Confidence {result.caption.confidence:.4f}")
    else:
         caption="No caption generated."
         print("No caption generated.")

    bounding_boxes = parse_blocks(result)
    img=draw_bounding_boxes(image, bounding_boxes)
    return img ,caption
def parse_blocks(results):
    
    bounding_boxes = []
    for line in results.read.blocks[0].lines:
        bounding_box = {
            "box_2d": line.bounding_polygon,
            "label": line.text
        }
        bounding_boxes.append(bounding_box)
    return bounding_boxes
import cv2
import numpy as np
from typing import List, Dict, Any

def draw_bounding_boxes(
    image: np.ndarray,
    bounding_boxes: List[Dict[str, Any]],
    box_color: tuple = (0, 0, 255),
    text_color: tuple = (255, 255, 255),
    line_width: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes with labels on a Gradio image (numpy array RGB).

    Args:
        image:          Numpy array in RGB format (as Gradio provides)
        bounding_boxes: List of dicts with 'box_2d' (4 {x,y} points) and 'label'
        box_color:      BGR tuple for box outline color
        text_color:     BGR tuple for label text color
        line_width:     Thickness of the bounding box lines
        font_scale:     Scale of the label font

    Returns:
        Annotated image as RGB numpy array (ready for Gradio output)
    """
    # Gradio gives RGB — convert to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for item in bounding_boxes:
        points = item.get("box_2d", [])
        label  = str(item.get("label", ""))

        if len(points) < 2:
            continue

        # Build numpy array of shape (N, 1, 2) as OpenCV expects
        pts = np.array(
            [[int(p["x"]), int(p["y"])] for p in points],
            dtype=np.int32
        ).reshape((-1, 1, 2))

        # Draw the polygon
        cv2.polylines(image_bgr, [pts], isClosed=True, color=box_color, thickness=line_width)

        # --- Label ---
        if label:
            font      = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            xs = [int(p["x"]) for p in points]
            ys = [int(p["y"]) for p in points]
            tx = min(xs)
            ty = max(min(ys) - 4, text_h + 4)

            # Filled background rectangle
            cv2.rectangle(
                image_bgr,
                (tx, ty - text_h - baseline),
                (tx + text_w, ty + baseline),
                box_color,
                thickness=cv2.FILLED
            )

            # Label text
            cv2.putText(
                image_bgr, label,
                (tx, ty),
                font, font_scale,
                text_color, thickness,
                lineType=cv2.LINE_AA
            )

    # Convert back to RGB for Gradio
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
import cv2
import numpy as np
from typing import List, Dict, Any

def draw_bounding_boxes(
    image,  # accepts both numpy array (RGB) or file path string
    bounding_boxes: List[Dict[str, Any]],
    box_color: tuple = (0, 0, 255),
    text_color: tuple = (255, 255, 255),
    line_width: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:

    # Handle both string path and numpy array
    if isinstance(image, str):
        image_bgr = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    for item in bounding_boxes:
        points = item.get("box_2d", [])
        label  = str(item.get("label", ""))

        if len(points) < 2:
            continue

        pts = np.array(
            [[int(p["x"]), int(p["y"])] for p in points],
            dtype=np.int32
        ).reshape((-1, 1, 2))

        cv2.polylines(image_bgr, [pts], isClosed=True, color=box_color, thickness=line_width)

        if label:
            font      = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            xs = [int(p["x"]) for p in points]
            ys = [int(p["y"]) for p in points]
            tx = min(xs)
            ty = max(min(ys) - 4, text_h + 4)

            cv2.rectangle(
                image_bgr,
                (tx, ty - text_h - baseline),
                (tx + text_w, ty + baseline),
                box_color,
                cv2.FILLED
            )

            cv2.putText(
                image_bgr, label,
                (tx, ty),
                font, font_scale,
                text_color, thickness,
                lineType=cv2.LINE_AA
            )

    # Always return RGB for Gradio
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def plot_bounding_boxes(image_data, bounding_boxes):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'cyan',
        'lime', 'magenta', 'violet', 'gold', 'silver'
    ] + additional_colors
         
    text_color = "white",
    line_width= 2,
    font_size= 12
    try:
        # Use a default font if NotoSansCJK is not available
        try:
            font = ImageFont.load_default()
        except OSError:
            print("NotoSansCJK-Regular.ttc not found. Using default font.")
            font = ImageFont.load_default()
    # Assuming bounding_boxes is your list: [{'box_2d': [...], 'label': '-'}]
        print(f"Bounding boxes: {bounding_boxes}")
        for i, item in enumerate(bounding_boxes):
            box_color = colors[i % len(colors)]
       
            points = item.get("box_2d", [])
            label = item.get("label", "")

            if len(points) < 2:
                continue

            # Extract plain integers explicitly
            xs = [int(p["x"]) for p in points]
            ys = [int(p["y"]) for p in points]
            coords = list(zip(xs, ys))  # [(x0,y0), (x1,y1), ...]

            # Draw each edge as a separate line using flat [x1, y1, x2, y2] format
            n = len(coords)
            for i in range(n):
                x1, y1 = coords[i]
                x2, y2 = coords[(i + 1) % n]
                draw.line(
                    xy=[int(x1), int(y1), int(x2), int(y2)],
                    fill=box_color,
                    width=int(line_width)
                )

        # Label
        if label:
            text_x = int(min(xs))
            text_y = int(max(min(ys) - font_size - 4, 0))

            try:
                # Pillow >= 8.0
                bbox = draw.textbbox((text_x, text_y), label, font=font)
                rect = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            except AttributeError:
                # Pillow < 8.0 fallback
                w, h = draw.textsize(label, font=font)
                rect = [text_x, text_y, text_x + w, text_y + h]

            draw.rectangle(rect, fill=box_color)
            draw.text((text_x, text_y), label, fill=text_color, font=font)

        # for i, item in enumerate(bounding_boxes):
        #         box_color = colors[i % len(colors)]
        #         points = item.get("box_2d", [])
        #         label = item.get("label", "")

        #         if len(points) < 2:
        #             continue

        #         # Extract all x and y values to get the bounding rectangle
        #         xs = [p["x"] for p in points]
        #         ys = [p["y"] for p in points]

        #         # Draw polygon if 4 points, else draw rectangle
        #         if len(points) == 4:
        #             polygon = [(p["x"], p["y"]) for p in points]
        #             draw.polygon(polygon, outline=box_color)
        #             # Redraw outline with line_width (polygon doesn't support width in older Pillow)
        #             for i in range(len(polygon)):
        #                 draw.line([polygon[i], polygon[(i + 1) % len(polygon)]], fill=box_color, width=line_width)
        #         else:
        #             draw.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=box_color, width=line_width)

        #         # Draw label background + text at the top-left of the bounding box
        #         if label:
        #             text_x, text_y = min(xs), min(ys) - font_size - 4
        #             # Keep label inside image bounds
        #             text_y = max(text_y, 0)

        #             # Background rectangle for text
        #             bbox = draw.textbbox((text_x, text_y), label, font=font)
        #             draw.rectangle(bbox, fill=box_color)
        #             draw.text((text_x, text_y), label, fill=text_color, font=font)

        # for i, bounding_box in enumerate(bounding_boxes):
        #     color = colors[i % len(colors)]
            
        #     # Extract x and y coordinates from the list of points
        #     points = bounding_box["box_2d"]
        #     x_coords = [p['x'] for p in points]
        #     y_coords = [p['y'] for p in points]
            
        #     # Calculate min/max and scale (Azure usually uses 0-1000 normalized)
        #     abs_x1 = int(min(x_coords) / 1000 * width)
        #     abs_y1 = int(min(y_coords) / 1000 * height)
        #     abs_x2 = int(max(x_coords) / 1000 * width)
        #     abs_y2 = int(max(y_coords) / 1000 * height)

        #     # Draw bounding box and label
        #     draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
            
        #     label = bounding_box.get("label", "-")
        #     if label:
        #         draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color, font=font)

    #    # bounding_boxes = json.loads(bounding_boxes)
    #     print(f"Bounding boxes: {bounding_boxes}")
    #     for i, bounding_box in enumerate(bounding_boxes):
    #         color = colors[i % len(colors)]
    #         abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
    #         abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
    #         abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
    #         abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

    #         if abs_x1 > abs_x2:
    #             abs_x1, abs_x2 = abs_x2, abs_x1

    #         if abs_y1 > abs_y2:
    #             abs_y1, abs_y2 = abs_y2, abs_y1

    #         # Draw bounding box and label
    #         draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
    #         if "label" in bounding_box:
    #             draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
    image.save("output_image.jpg")
    print(f"Annotated image saved as output_image.jpg")
    return image

import cv2
import numpy as np
from typing import List, Dict, Any

# Distinct color palette for different labels (BGR format)
COLORS = [
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 128, 255),   # Sky blue
    (0, 215, 255),   # Gold
    (144, 238, 144), # Light green
]

def get_color_for_label(label: str, label_color_map: dict) -> tuple:
    """Assign a consistent color per unique label."""
    if label not in label_color_map:
        label_color_map[label] = COLORS[len(label_color_map) % len(COLORS)]
    return label_color_map[label]

def draw_bounding_boxes(
    image,
    bounding_boxes: List[Dict[str, Any]],
    line_width: int = 2,
    font_scale: float = 0.55,
    text_padding: int = 6,
    text_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Draw bounding boxes with non-overlapping labels on a Gradio image.

    Args:
        image:          File path (str) or numpy RGB array from Gradio
        bounding_boxes: [{'box_2d': [{'x':..,'y':..}, ...], 'label': '...'}]
        line_width:     Box edge thickness
        font_scale:     Label font size
        text_padding:   Padding inside label background
        text_color:     Label text color (BGR)

    Returns:
        Annotated image as RGB numpy array
    """
    # ── Load image ──────────────────────────────────────────────────────────
    if isinstance(image, str):
        image_bgr = cv2.imread(image)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image}")
    elif isinstance(image, np.ndarray):
        image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    img_h, img_w = image_bgr.shape[:2]

    # ── Draw on a copy so original is untouched ──────────────────────────────
    overlay     = image_bgr.copy()
    label_color_map = {}
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2

    # Track placed label rectangles to resolve overlaps
    placed_labels: List[tuple] = []  # list of (x1, y1, x2, y2)

    def find_non_overlapping_position(tx, ty, tw, th, margin=2):
        """Shift label vertically until it doesn't overlap any placed label."""
        for _ in range(20):  # max shift attempts
            candidate = (tx, ty, tx + tw, ty + th)
            overlap = any(
                not (candidate[2] + margin < r[0] or candidate[0] > r[2] + margin or
                     candidate[3] + margin < r[1] or candidate[1] > r[3] + margin)
                for r in placed_labels
            )
            if not overlap:
                return tx, ty
            ty -= (th + margin)  # shift upward
            if ty < 0:
                ty = candidate[3] + margin  # if no room above, go below
        return tx, ty

    for item in bounding_boxes:
        points = item.get("box_2d", [])
        label  = str(item.get("label", "")).strip()

        if len(points) < 2:
            continue

        color = get_color_for_label(label, label_color_map)

        # ── Draw polygon with thick outline + semi-transparent fill ──────────
        pts = np.array(
            [[int(p["x"]), int(p["y"])] for p in points],
            dtype=np.int32
        ).reshape((-1, 1, 2))

        # Semi-transparent fill
        mask = image_bgr.copy()
        cv2.fillPoly(mask, [pts], color)
        cv2.addWeighted(mask, 0.15, overlay, 0.85, 0, overlay)

        # Solid border — draw twice for a "halo" effect (white outline + color)
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=(255, 255, 255), thickness=line_width + 3)
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=color, thickness=line_width)

        # ── Label size ───────────────────────────────────────────────────────
        if not label:
            continue

        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        xs = [int(p["x"]) for p in points]
        ys = [int(p["y"]) for p in points]

        # Start label just above the topmost point
        tx = max(min(xs), 0)
        ty = min(ys) - text_padding - baseline

        label_w = text_w + text_padding * 2
        label_h = text_h + text_padding * 2 + baseline

        # Clamp to image bounds
        tx = min(tx, img_w - label_w)
        ty = max(ty, label_h)

        # Resolve overlaps
        tx, ty = find_non_overlapping_position(tx, ty, label_w, label_h)

        # Register this label's rect
        placed_labels.append((tx, ty, tx + label_w, ty + label_h))

        # ── Draw label background (rounded feel via filled rect) ─────────────
        # Dark shadow for depth
        cv2.rectangle(
            overlay,
            (tx + 2, ty + 2),
            (tx + label_w + 2, ty + label_h + 2),
            (0, 0, 0),
            cv2.FILLED
        )
        # Colored background
        cv2.rectangle(
            overlay,
            (tx, ty),
            (tx + label_w, ty + label_h),
            color,
            cv2.FILLED
        )
        # White border around label
        cv2.rectangle(
            overlay,
            (tx, ty),
            (tx + label_w, ty + label_h),
            (255, 255, 255),
            1
        )

        # ── Draw label text ──────────────────────────────────────────────────
        cv2.putText(
            overlay,
            label,
            (tx + text_padding, ty + text_h + text_padding),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
def gradio_interface():
    """
    Gradio app interface for bounding box generation with example pairs.
    """
    # Example image + prompt pairs
    examples = [
        ["images/cookies.jpg"],
        ["images/messed_room.jpg"],
        ["images/yoga.jpg"],
        ["images/foreign_menu.png"]
    ]

    with gr.Blocks(gr.themes.Glass(secondary_hue= "rose")) as demo:
        gr.Markdown("# Gemini Bounding Box Generator")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Section")
                input_image = gr.Image(type="filepath")
                submit_btn = gr.Button("Generate")

            with gr.Column():
                gr.Markdown("### Output Section")
                output_image = gr.Image(type="pil", label="Output Image")
                output_caption = gr.Textbox(label="Generated Caption")

        gr.Markdown("### Examples")
        gr.Examples(
            examples=examples,
            inputs=input_image, 
            label="Example Images"
        )

        # Event to generate bounding boxes
        submit_btn.click(
            generate_bounding_boxes,
            inputs= input_image,
            outputs=[output_image , output_caption]
        )

    return demo

if __name__ == "__main__":
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )


    demo = gradio_interface()
    demo.launch()