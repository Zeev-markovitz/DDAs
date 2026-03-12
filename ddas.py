from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import cv2

def create_dda_grid(
    image_lists,
    row_labels,
    col_labels,
    out_file,
    padding=10,
    bg_color=(255, 255, 255),
    font_size=20,
    transpose=False,
    crop=None,
):
    """
    Creates a composite image grid from two or three lists of image file paths.

    Parameters:
    - image_lists - list of lists of images
    - row_labels: List of strain labels.
    - col_labels: List of column labels.
    - out_file: String, path to save the output PNG.
    - padding: Integer, space between images.
    - bg_color: Tuple, background color in (R, G, B).
    - font_size: Integer, size of the font for labels.
    - transpose: Bool, if True swap rows/cols of image_lists and labels.
    - crop: Optional int. If set, crop each image to a square of size crop×crop
            centered on the image center before further processing.
    """

    if transpose:
        image_lists = [list(x) for x in zip(*image_lists)]
        row_labels, col_labels = col_labels, row_labels

    def center_crop(img, size):
        w, h = img.size
        half = size // 2
        cx, cy = w // 2, h // 2
        left = max(0, cx - half)
        upper = max(0, cy - half)
        right = min(w, left + size)
        lower = min(h, upper + size)
        return img.crop((left, upper, right, lower))

    # Load images (and crop if requested)
    all_lists = []
    for L in image_lists:
        row = []
        for f in L:
            if f is None:
                row.append(None)
            else:
                img = Image.open(f)
                if crop is not None:
                    img = center_crop(img, crop)
                row.append(img)
        all_lists.append(row)

    # Determine grid layout
    num_rows = len(row_labels)
    num_cols = len(col_labels)

    # Resize images to match the smallest height in each row
    for row_idx in range(num_rows):
        min_height = min(
            img_list[row_idx].height
            for img_list in all_lists
            if img_list[row_idx] is not None
        )
        for img_list in all_lists:
            img = img_list[row_idx]
            if img is None:
                continue
            scale_factor = min_height / img.height
            new_width = int(img.width * scale_factor)
            img_list[row_idx] = img.resize((new_width, min_height), Image.LANCZOS)

    # Column-wise max widths and heights
    max_widths = [
        max((img.width for img in lst if img is not None), default=0)
        for lst in all_lists
    ]
    max_heights = [
        max((img.height for img in lst if img is not None), default=0)
        for lst in all_lists
    ]

    try:
        label_font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        try:
            label_font = ImageFont.truetype("Keyboard.ttf", font_size)
        except IOError:
            label_font = ImageFont.load_default()

    # Compute label area sizes
    strain_label_width = (
        max(label_font.getbbox(s)[2] - label_font.getbbox(s)[0] for s in row_labels)
        + padding
    )
    time_label_height = (
        max(label_font.getbbox(t)[3] - label_font.getbbox(t)[1] for t in col_labels)
        + padding
    )

    # Determine total image size
    grid_width = sum(max_widths) + padding * (len(all_lists) - 1) + strain_label_width
    grid_height = (
        num_rows * max(max_heights)
        + padding * (num_rows - 1)
        + time_label_height
    )

    # Create blank image
    grid_img = Image.new("RGB", (grid_width, grid_height), bg_color)
    draw = ImageDraw.Draw(grid_img)

    # Draw column labels
    x_offset = strain_label_width
    for col_idx, time_label in enumerate(col_labels):
        bbox = label_font.getbbox(time_label)
        label_width = bbox[2] - bbox[0]
        label_x = x_offset + (max_widths[col_idx] // 2) - (label_width // 2)
        draw.text((label_x, 5), time_label, fill=(0, 0, 0), font=label_font)
        x_offset += max_widths[col_idx] + padding

    # Paste images and row labels
    x_offset, y_offset = strain_label_width, time_label_height
    row_area_height = max(max_heights)

    for row_idx, strain_label in enumerate(row_labels):
        s_bbox = label_font.getbbox(strain_label)
        s_height = s_bbox[3] - s_bbox[1]
        label_y = y_offset + (row_area_height - s_height) // 2
        draw.text((5, label_y), strain_label, fill=(0, 0, 0), font=label_font)

        x_offset = strain_label_width
        for col_idx, img_list in enumerate(all_lists):
            img = img_list[row_idx]
            if img is not None:
                paste_x = x_offset + (max_widths[col_idx] - img.width) // 2
                paste_y = y_offset
                grid_img.paste(img, (paste_x, paste_y))
            x_offset += max_widths[col_idx] + padding

        y_offset += row_area_height + padding

    grid_img.save(out_file)

def create_stacked_image(
    input_files1, input_files2, labels, height, out_file,
    sep_size=3, sep_color=(0, 0, 0),
    max_slice_width=None, max_radius=50, max_dev_from_center=100,
    kind='stacked',
    orientation='vertical',
    horizontal_label_rotation=None,
    layout_order=('labels', 'files1', 'files2'),
    font_size=12,
    min_thresh_intensity=200,
):
    from PIL import Image, ImageDraw, ImageFont

    assert len(input_files1) == len(input_files2) == len(labels)
    assert kind in ('stacked', 'opposite')
    assert orientation in ('vertical', 'horizontal')
    assert horizontal_label_rotation in (None, 'left', 'right')
    assert set(layout_order) == {'labels', 'files1', 'files2'}

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # -------------------------------
    # Pre-render labels
    # -------------------------------
    label_images = []
    max_label_height = 0
    dummy = Image.new("RGB", (10, 10))
    ddraw = ImageDraw.Draw(dummy)
    ascent, descent = font.getmetrics()

    for label in labels:
        bbox = ddraw.textbbox((0, 0), label, font=font)
        lw = bbox[2] - bbox[0]
        lh = ascent + descent
        lbl = Image.new("RGB", (lw + 4, lh + 4), "white")
        ld = ImageDraw.Draw(lbl)
        ld.text((2, 2), label, fill="black", font=font)
        if orientation == 'horizontal':
            if horizontal_label_rotation == 'left':
                lbl = lbl.rotate(90, expand=True)
            elif horizontal_label_rotation == 'right':
                lbl = lbl.rotate(-90, expand=True)
        label_images.append(lbl)
        max_label_height = max(max_label_height, lbl.height)

    label_band_height = max_label_height + 6

    # -------------------------------
    # Disk detection
    # -------------------------------
    sample = Image.open(input_files1[0])
    img_width, _ = sample.size
    centers_x, centers_y = [], []
    for f1, f2 in zip(input_files1, input_files2):
        (cx1, cy1), _ = find_dda_disk(f1, max_radius=max_radius,
                                     max_dev_from_center=max_dev_from_center,
                                     min_thresh_intensity=min_thresh_intensity)
        (cx2, cy2), _ = find_dda_disk(f2, max_radius=max_radius,
                                     max_dev_from_center=max_dev_from_center,
                                     min_thresh_intensity=min_thresh_intensity)
        centers_x.extend([cx1, cx2])
        centers_y.extend([cy1, cy2])
    x_offsets = [0] + [centers_x[0] - cx for cx in centers_x[1:]]
    display_width = (max_slice_width if max_slice_width else img_width) // 2
    left_margin = 100
    n = len(labels)

    # -------------------------------
    # Canvas sizing
    # -------------------------------
    if orientation == 'vertical':
        if kind == 'stacked':
            # Each pair: slice1 + sep + slice2
            final_height = n * (2 * height + sep_size) + (n - 1) * sep_size
            final_width = left_margin + sep_size + display_width
        else:  # opposite
            col_widths = {'labels': left_margin, 'files1': display_width, 'files2': display_width}
            final_width = sum(col_widths[k] for k in layout_order) + sep_size * (len(layout_order) - 1)
            final_height = n * height + (n - 1) * sep_size
    else:
        if kind == 'stacked':
            final_width = n * (2 * height + sep_size) + (n - 1) * sep_size
            final_height = display_width + label_band_height
        else:
            row_heights = {'labels': label_band_height, 'files1': display_width, 'files2': display_width}
            final_width = n * height + (n - 1) * sep_size
            final_height = sum(row_heights[k] for k in layout_order) + sep_size * (len(layout_order) - 1)

    final_img = Image.new("RGB", (final_width, final_height), "white")
    draw = ImageDraw.Draw(final_img)

    x_offset_pos = 0
    y_offset = 0

    for i, (f1, f2) in enumerate(zip(input_files1, input_files2)):
        img1 = Image.open(f1)
        img2 = Image.open(f2)
        cx1, cy1 = centers_x[2*i], centers_y[2*i]
        cx2, cy2 = centers_x[2*i+1], centers_y[2*i+1]
        top1 = max(0, cy1 - height // 2)
        top2 = max(0, cy2 - height // 2)
        c1 = crop_and_pad(img1, top1, img_width, height, x_offsets[2*i])
        c2 = crop_and_pad(img2, top2, img_width, height, x_offsets[2*i+1])
        if max_slice_width:
            c1 = slice_image_by_center(c1, max_slice_width, cx1 + x_offsets[2*i])
            c2 = slice_image_by_center(c2, max_slice_width, cx2 + x_offsets[2*i+1])
        c1 = keep_left_half(c1)
        c2 = keep_left_half(c2)
        label_img = label_images[i]

        # -------------------------------
        # STACKED VERTICAL
        # -------------------------------
        if kind == 'stacked' and orientation == 'vertical':
            # vertical separator between label and slices
            draw.rectangle([(left_margin, 0), (left_margin + sep_size, final_height)], fill=sep_color)

            pair_top = y_offset

            # slice1
            final_img.paste(c1, (left_margin + sep_size, y_offset))
            y_offset += height

            # separator between slice1 and slice2 — always drawn
            draw.rectangle([(left_margin + sep_size, y_offset), (final_width, y_offset + sep_size)], fill=sep_color)
            y_offset += sep_size

            # slice2
            final_img.paste(c2, (left_margin + sep_size, y_offset))
            y_offset += height
            pair_bottom = y_offset

            # label centered across two slices
            center_y = (pair_top + pair_bottom) // 2
            ly = center_y - label_img.height // 2
            lx = left_margin // 2 - label_img.width // 2
            final_img.paste(label_img, (lx, ly))

            # separator after pair — only internal
            if i != n - 1:
                draw.rectangle([(0, y_offset), (final_width, y_offset + sep_size)], fill=sep_color)
                y_offset += sep_size

        # -------------------------------
        # STACKED HORIZONTAL
        # -------------------------------
        elif kind == 'stacked' and orientation == 'horizontal':
            c1r = c1.rotate(90, expand=True)
            c2r = c2.rotate(90, expand=True)
            pair_left = x_offset_pos

            # slice1
            final_img.paste(c1r, (x_offset_pos, label_band_height))
            x_offset_pos += c1r.width

            # separator between slice1 and slice2 — always drawn
            draw.rectangle([(x_offset_pos, label_band_height), (x_offset_pos + sep_size, final_height)], fill=sep_color)
            x_offset_pos += sep_size

            # slice2
            final_img.paste(c2r, (x_offset_pos, label_band_height))
            x_offset_pos += c2r.width
            pair_right = x_offset_pos

            # label centered across two slices
            center_x = (pair_left + pair_right) // 2
            lx = center_x - label_img.width // 2
            ly = (label_band_height - label_img.height) // 2
            final_img.paste(label_img, (lx, ly))

            # separator between label layer and slice layer
            draw.rectangle([(pair_left, label_band_height - sep_size), (pair_right, label_band_height)], fill=sep_color)

            # separator after pair — only internal
            if i != n - 1:
                draw.rectangle([(x_offset_pos, 0), (x_offset_pos + sep_size, final_height)], fill=sep_color)
                x_offset_pos += sep_size

        # -------------------------------
        # OPPOSITE
        # -------------------------------
        else:
            if orientation == 'vertical':
                c2 = c2.transpose(Image.FLIP_LEFT_RIGHT)
                col_widths = {'labels': left_margin, 'files1': display_width, 'files2': display_width}
                x = 0
                x_positions = {}
                for key in layout_order:
                    x_positions[key] = x
                    x += col_widths[key]
                    if key != layout_order[-1]:
                        draw.rectangle([(x, y_offset), (x + sep_size, y_offset + height)], fill=sep_color)
                        x += sep_size
                lx = x_positions['labels'] + (col_widths['labels'] - label_img.width)//2
                ly = y_offset + (height - label_img.height)//2
                final_img.paste(label_img, (lx, ly))
                final_img.paste(c1, (x_positions['files1'], y_offset))
                final_img.paste(c2, (x_positions['files2'], y_offset))
                y_offset += height
                if i != n - 1:
                    draw.rectangle([(0, y_offset), (final_width, y_offset + sep_size)], fill=sep_color)
                    y_offset += sep_size
            else:
                c1r = c1.rotate(90, expand=True)
                c2r = c2.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
                if layout_order.index('files1') < layout_order.index('files2'):
                    c1r = c1r.transpose(Image.FLIP_TOP_BOTTOM)
                    c2r = c2r.transpose(Image.FLIP_TOP_BOTTOM)
                row_heights = {'labels': label_band_height, 'files1': display_width, 'files2': display_width}
                y = 0
                y_positions = {}
                for key in layout_order:
                    y_positions[key] = y
                    y += row_heights[key]
                    if key != layout_order[-1]:
                        draw.rectangle([(x_offset_pos, y), (x_offset_pos + c1r.width, y + sep_size)], fill=sep_color)
                        y += sep_size
                lx = x_offset_pos + (c1r.width - label_img.width)//2
                ly = y_positions['labels']
                if horizontal_label_rotation == 'left':
                    ly += row_heights['labels'] - label_img.height - 2
                else:
                    ly += (row_heights['labels'] - label_img.height)//2
                final_img.paste(label_img, (lx, ly))
                final_img.paste(c2r, (x_offset_pos, y_positions['files2']))
                final_img.paste(c1r, (x_offset_pos, y_positions['files1']))
                x_offset_pos += c1r.width
                if i != n - 1:
                    draw.rectangle([(x_offset_pos, 0), (x_offset_pos + sep_size, final_height)], fill=sep_color)
                    x_offset_pos += sep_size

    final_img.save(out_file)

# -------------------------------
# Helpers
# -------------------------------
def slice_image_by_center(image, width, center_x):
    half = width // 2
    return image.crop((max(0, center_x - half), 0,
                       min(image.width, center_x + half), image.height))

def keep_left_half(image):
    return image.crop((0, 0, image.width // 2, image.height))

def crop_and_pad(image, top, width, height, x_offset):
    return pad_image_with_x_offset(image.crop((0, top, image.width, top + height)), x_offset)

def pad_image_with_x_offset(image, x_offset):
    w, h = image.size
    new = Image.new("RGB", (w + abs(x_offset), h), "white")
    new.paste(image, (x_offset, 0))
    return new

def find_dda_disk(in_file, out_file=None, max_radius=50, max_dev_from_center=50, debug=False, min_thresh_intensity=200):
    # pil_image = Image.open(in_file)
    # print(pil_image.getexif().get(274)) 

    # 1. Read with Pillow
    pil_img = Image.open(in_file)

    # 2. Apply EXIF orientation correction (no-op if none present)
    # Apparenly, PIL already rotates the image as it should be rotated
    # pil_img = ImageOps.exif_transpose(pil_img)

    # 3. Ensure 3-channel RGB (convert if grayscale / RGBA / etc.)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 4. Convert to NumPy array (RGB)
    img_rgb = np.array(pil_img)

    # 5. Convert RGB → BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Load the image and convert to grayscale
    # image = cv2.imread(in_file, cv2.IMREAD_COLOR)
    image = img_bgr
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    image_center = (w // 2, h // 2)

    # Threshold the image to detect bright regions
    _, thresh = cv2.threshold(gray, min_thresh_intensity, 255, cv2.THRESH_BINARY)

    # ------------------------------------------------------------
    # Restrict contour detection to inner square
    # ------------------------------------------------------------
    half_side = max_radius + max_dev_from_center

    x0 = max(0, image_center[0] - half_side)
    x1 = min(w, image_center[0] + half_side)
    y0 = max(0, image_center[1] - half_side)
    y1 = min(h, image_center[1] + half_side)

    thresh_inner = thresh[y0:y1, x0:x1]
    # ------------------------------------------------------------

    # Find contours only in the inner square
    contours, _ = cv2.findContours(
        thresh_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if debug:
        cv2.imwrite('/tmp/debug_gray.png', gray)
        cv2.imwrite('/tmp/debug_thresh.png', thresh)
        cv2.imwrite('/tmp/debug_thresh_inner.png', thresh_inner)

    # Find the largest circular contour
    best_center, best_radius = None, 0

    for cnt in contours:
        # Shift contour coordinates back to full-image space
        cnt = cnt + [x0, y0]

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        if (
            10 < radius <= max_radius
            and abs(center[0] - image_center[0]) < max_dev_from_center
            and abs(center[1] - image_center[1]) < max_dev_from_center
        ):
            if radius > best_radius:
                best_center, best_radius = center, radius

    # Draw and save the marked image if out_file is provided
    if out_file:
        pil_image = Image.open(in_file)
        draw = ImageDraw.Draw(pil_image)

        if debug:
            # Draw all contours in red
            for cnt in contours:
                cnt = cnt + [x0, y0]
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                draw.ellipse(
                    (
                        center[0] - radius,
                        center[1] - radius,
                        center[0] + radius,
                        center[1] + radius,
                    ),
                    outline="red",
                    width=1,
                )

            # Draw max_radius
            draw.ellipse(
                (
                    image_center[0] - max_radius,
                    image_center[1] - max_radius,
                    image_center[0] + max_radius,
                    image_center[1] + max_radius,
                ),
                outline="blue",
                width=1,
            )

            # Draw max_dev_from_center square
            draw.rectangle(
                (
                    image_center[0] - max_dev_from_center,
                    image_center[1] - max_dev_from_center,
                    image_center[0] + max_dev_from_center,
                    image_center[1] + max_dev_from_center,
                ),
                outline="blue",
                width=1,
            )

            # Draw inner contour search square
            draw.rectangle((x0, y0, x1, y1), outline="green", width=1)

            if best_center is not None:
                draw.ellipse(
                    (
                        best_center[0] - best_radius,
                        best_center[1] - best_radius,
                        best_center[0] + best_radius,
                        best_center[1] + best_radius,
                    ),
                    outline="blue",
                    width=1,
                )

            # Draw image center
            x_size = 10
            draw.line(
                (
                    image_center[0] - x_size,
                    image_center[1] - x_size,
                    image_center[0] + x_size,
                    image_center[1] + x_size,
                ),
                fill="blue",
                width=1,
            )
            draw.line(
                (
                    image_center[0] - x_size,
                    image_center[1] + x_size,
                    image_center[0] + x_size,
                    image_center[1] - x_size,
                ),
                fill="blue",
                width=1,
            )
        else:
            draw.ellipse(
                (
                    best_center[0] - best_radius,
                    best_center[1] - best_radius,
                    best_center[0] + best_radius,
                    best_center[1] + best_radius,
                ),
                outline="red",
                width=3,
            )

        pil_image.save(out_file)

    if best_center is None:
        raise ValueError(f"No disk detected in the image: {in_file}")

    return best_center, best_radius