from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import cv2
from pathlib import Path

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
    Create a labeled grid image from multiple lists of image files.

    Each inner list in ``image_lists`` represents a column in the grid and
    contains images corresponding to the entries in ``row_labels``. Images are
    resized within each row so that all images in that row share the smallest
    height, preserving aspect ratio. Optional center cropping can be applied
    prior to resizing. Row and column labels are rendered along the left and
    top margins, respectively.

    Parameters
    ----------
    image_lists : list of list of str
        Nested list of image file paths. Each inner list corresponds to a
        column, and each element corresponds to a row. Entries may be ``None``
        to leave a cell empty.
    row_labels : list of str
        Labels displayed along the left side of the grid.
    col_labels : list of str
        Labels displayed along the top of the grid.
    out_file : str
        Path to the output image file.
    padding : int, default=10
        Number of pixels used as spacing between images and between grid
        elements.
    bg_color : tuple of int, default=(255, 255, 255)
        Background color of the canvas in ``(R, G, B)`` format.
    font_size : int, default=20
        Font size used when rendering row and column labels.
    transpose : bool, default=False
        If ``True``, transpose the grid by swapping rows and columns of
        ``image_lists`` and exchanging ``row_labels`` and ``col_labels``.
    crop : int or None, default=None
        If provided, each image is center-cropped to a square of size
        ``crop × crop`` pixels before resizing and placement.

    Returns
    -------
    None
        The resulting grid image is written to ``out_file``.

    Notes
    -----
    Images are aligned within their column cells and centered horizontally
    relative to the maximum width of that column. Rows share a uniform height
    determined by the tallest image within the row after resizing.
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
    max_slice_width=None,
    layout='stacked',
    orientation='vertical',
    horizontal_label_rotation=None,
    layout_order=('labels', 'files1', 'files2'),
    font_size=12,

    rads=None,

    # Disk detection parameters
    max_radius=50, max_dev_from_center=100, min_thresh_intensity=230,
):
    """
    Construct a composite image from paired input images with optional labels.

    The function extracts vertically centered slices from paired images,
    aligns them based on automatically detected disk centers, and arranges
    them either stacked or opposite each other. Labels corresponding to each
    image pair are rendered alongside or above the slices depending on the
    chosen orientation.

    Parameters
    ----------
    input_files1 : list of str
        File paths for the first image in each pair.
    input_files2 : list of str
        File paths for the second image in each pair.
    labels : list of str
        Labels corresponding to each pair of images.
    height : int
        Height in pixels of the vertical slice extracted from each image.
    out_file : str
        Path to the output image file.
    sep_size : int, default=3
        Thickness in pixels of separator lines drawn between slices or
        sections.
    sep_color : tuple of int, default=(0, 0, 0)
        Color of separator lines in ``(R, G, B)`` format.
    max_slice_width : int or None, default=None
        Maximum width of the cropped slice centered on the detected disk.
        If ``None``, the full image width is used before halving.
    layout : {'stacked', 'opposite'}, default='stacked'
        Layout mode. ``'stacked'`` places slices sequentially for each pair,
        while ``'opposite'`` positions slices from the two files side-by-side
        or above/below one another depending on orientation.
    orientation : {'vertical', 'horizontal'}, default='vertical'
        Orientation of the overall layout.
    horizontal_label_rotation : {None, 'left', 'right'}, default=None
        Optional rotation applied to labels when ``orientation='horizontal'``.
    layout_order : tuple of {'labels', 'files1', 'files2'}, default=('labels', 'files1', 'files2')
        Ordering of layout elements when ``kind='opposite'``. Determines how
        labels and slices are arranged in rows or columns.
    font_size : int, default=12
        Font size used for label rendering.
    max_radius : int, default=50
        Maximum radius of the detected disk used for alignment.
    max_dev_from_center : int, default=100
        Maximum allowed deviation in pixels of the detected disk center from
        the image center.
    min_thresh_intensity : int, default=230
        Threshold intensity used when detecting the disk center.

    Returns
    -------
    None
        The resulting composite image is written to ``out_file``.

    Notes
    -----
    Disk centers are detected using :func:`find_dda_disk`, and slices are
    horizontally aligned using offsets derived from these centers. Each slice
    is optionally restricted to the left half after cropping to emphasize the
    region of interest.
    """

    assert len(input_files1) == len(input_files2) == len(labels)
    assert layout in ('stacked', 'opposite')
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
    centers_x, centers_y, radii = [], [], []
    for f1, f2 in zip(input_files1, input_files2):
        (cx1, cy1), r1 = find_dda_disk(f1, max_radius=max_radius,
                                      max_dev_from_center=max_dev_from_center,
                                      min_thresh_intensity=min_thresh_intensity)
        (cx2, cy2), r2 = find_dda_disk(f2, max_radius=max_radius,
                                     max_dev_from_center=max_dev_from_center,
                                     min_thresh_intensity=min_thresh_intensity)
        centers_x.extend([cx1, cx2])
        centers_y.extend([cy1, cy2])
        radii.extend([r1, r2])
    x_offsets = [0] + [centers_x[0] - cx for cx in centers_x[1:]]
    display_width = (max_slice_width if max_slice_width else img_width) // 2
    left_margin = 100
    n = len(labels)

    # -------------------------------
    # Canvas sizing
    # -------------------------------
    if orientation == 'vertical':
        if layout == 'stacked':
            # Each pair: slice1 + sep + slice2
            final_height = n * (2 * height + sep_size) + (n - 1) * sep_size
            final_width = left_margin + sep_size + display_width
        else:  # opposite
            col_widths = {'labels': left_margin, 'files1': display_width, 'files2': display_width}
            final_width = sum(col_widths[k] for k in layout_order) + sep_size * (len(layout_order) - 1)
            final_height = n * height + (n - 1) * sep_size
    else:
        if layout == 'stacked':
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
        
        if rads:
            r1, r2 = radii[2*i], radii[2*i+1]

            intensities, _ = plot_pixel_intensities_from_plate(img1, (cx1, cy1), r1, 500)
            rads_ = get_rads_from_pixels(intensities, rads)
            img1 = draw_circles(Image.open(f1), (cx1, cy1), [r+r1 for r in rads_], line_width=1)

            intensities, _ = plot_pixel_intensities_from_plate(img2, (cx2, cy2), r2, 500)
            rads_ = get_rads_from_pixels(intensities, rads)
            img2 = draw_circles(Image.open(f2), (cx2, cy2), [r+r2 for r in rads_], line_width=1)

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
        if layout == 'stacked' and orientation == 'vertical':
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
        elif layout == 'stacked' and orientation == 'horizontal':
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
# TODO: these functions can probably be in-lined into the main function.
def slice_image_by_center(image, width, center_x):
    """
    Extract a horizontal slice centered at a specified x-coordinate.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image to be cropped.
    width : int
        Width of the slice to extract.
    center_x : int
        Horizontal coordinate representing the center of the slice.

    Returns
    -------
    PIL.Image.Image
        Cropped image containing the requested horizontal slice.

    Notes
    -----
    The crop bounds are clamped to the image boundaries to avoid indexing
    outside the image.
    """

    half = width // 2
    return image.crop((max(0, center_x - half), 0,
                       min(image.width, center_x + half), image.height))

def keep_left_half(image):
    """
    Return the left half of an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.

    Returns
    -------
    PIL.Image.Image
        Cropped image containing only the left half of the input.
    """
    return image.crop((0, 0, image.width // 2, image.height))

def crop_and_pad(image, top, width, height, x_offset):
    """
    Crop a vertical region from an image and apply horizontal padding on the
    left side.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    top : int
        Top pixel coordinate of the crop region.
    width : int
        Width of the region to crop (unused but preserved for API
        compatibility).
    height : int
        Height of the cropped region.
    x_offset : int
        Horizontal offset used when padding the cropped image.

    Returns
    -------
    PIL.Image.Image
        Cropped image padded horizontally according to ``x_offset``.
    """

    return pad_image_with_x_offset(image.crop((0, top, image.width, top + height)), x_offset)

def pad_image_with_x_offset(image, x_offset):
    """
    Pad an image horizontally by shifting it within a larger (white) canvas,
    effectively padding it with white on its left side.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    x_offset : int
        Horizontal offset applied when pasting the image into the padded
        canvas. Positive values shift the image to the right.

    Returns
    -------
    PIL.Image.Image
        New image containing the original image placed within a padded
        background.
    """
    w, h = image.size
    new = Image.new("RGB", (w + abs(x_offset), h), "white")
    new.paste(image, (x_offset, 0))
    return new

_DDA_DISK_CACHE = {}

def clear_dda_disk_cache():
    """Clear the DDA disk detection cache."""
    global _DDA_DISK_CACHE
    _DDA_DISK_CACHE.clear()

def find_dda_disk(in_file, max_radius=50, max_dev_from_center=50, min_thresh_intensity=230, debug_folder=None):
    """
    Detect a circular disk feature near the center of an image.

    The function thresholds the grayscale version of the image to identify
    bright regions, restricts contour detection to a square region around the
    image center, and selects the largest circular contour whose radius and
    position satisfy the specified constraints.

    Parameters
    ----------
    in_file : str
        Path to the input image file.
    max_radius : int, default=50
        Maximum allowed radius for the detected disk.
    max_dev_from_center : int, default=50
        Maximum allowed deviation in pixels between the disk center and the
        image center in both x and y directions.
    min_thresh_intensity : int, default=230
        Intensity threshold used to binarize the grayscale image before
        contour detection.
    debug_folder : str or None, default=None
        If provided, diagnostic images will be saved to this folder for debugging
        purposes.

    Returns
    -------
    best_center : tuple of int
        Coordinates ``(x, y)`` of the detected disk center.
    best_radius : int
        Radius of the detected disk.

    Raises
    ------
    ValueError
        If no valid disk satisfying the constraints is detected.

    Notes
    -----
    Images are loaded with Pillow and converted to NumPy arrays for processing
    with OpenCV. Contour detection is restricted to a square region around the
    image center defined by ``max_radius + max_dev_from_center`` to reduce
    spurious detections.
    """

    in_file = str(Path(in_file).expanduser().resolve())

    key = (in_file, max_radius, max_dev_from_center, min_thresh_intensity)

    if key in _DDA_DISK_CACHE and not debug_folder:
        return _DDA_DISK_CACHE[key]

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

    if debug_folder:
        prefix = Path(in_file).stem
        pil_image = Image.open(in_file)
        draw = ImageDraw.Draw(pil_image)

        cv2.imwrite(f'{debug_folder}/{prefix}_gray.png', gray)
        cv2.imwrite(f'{debug_folder}/{prefix}_thresh.png', thresh)
        cv2.imwrite(f'{debug_folder}/{prefix}_thresh_inner.png', thresh_inner)

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
                outline="green",
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

        pil_image.save(f"{debug_folder}/{prefix}_debug.png")

    if best_center is None:
        raise ValueError(f"No disk detected in the image: {in_file}")

    _DDA_DISK_CACHE[key] = (best_center, best_radius)
    return best_center, best_radius

def plot_pixel_intensities_from_plate(img, center, start_radius, max_length, plot=False, out_file=None, debug=False):
    """
    Computes the average pixel intensity in concentric circles from a given center in the image
    and plots the intensities as a scatter plot.
    
    Parameters:
    - img: A PIL Image object.
    - center: Tuple (cx, cy) representing the center of the concentric circles.
    - start_radius: Integer, radius from which to start measuring intensity.
    - max_length: Integer, width and height of the cropped square around the center.
    - out_file: String, path to save the output PNG file.
    
    Returns:
    - A tuple (intensities, fig), where:
      - intensities is a NumPy array of average pixel intensities per radius.
      - fig is the matplotlib figure object.
    """
    cx, cy = center
    half_length = max_length // 2
    img_cropped = img.crop((cx - half_length, cy - half_length, cx + half_length, cy + half_length))
    img_gray = img_cropped.convert("L")
    img_array = np.array(img_gray)
    
    max_radius = min(half_length, half_length)
    intensities = []
    
    for r in range(start_radius, max_radius):
        y, x = np.ogrid[:max_length, :max_length]
        mask = ((x - half_length) ** 2 + (y - half_length) ** 2 >= r ** 2) & ((x - half_length) ** 2 + (y - half_length) ** 2 < (r + 1) ** 2)
        pixel_values = img_array[mask]
        if pixel_values.size > 0:
            intensities.append(np.mean(pixel_values))
            # intensities.append(np.median(pixel_values))
        else:
            intensities.append(0)

        if debug:
            apply_mask_to_image(img_cropped, mask, f"/tmp/debug/{r}.png")
    
    # Create scatter plot
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(range(start_radius, start_radius + len(intensities)), intensities, s=5, color='black')
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Average Intensity")
        ax.set_title("Pixel Intensity Profile from Plate Center")
        
        # Save the plot
        if out_file is not None:
            fig.savefig(out_file, dpi=300)
        plt.close(fig)
    
    return np.array(intensities), fig

def get_rads_from_pixels(
    pixels, thresholds=(0.2, 0.5, 0.8)
):
    pixels = np.array(pixels) # Copy to avoid modifying the original

    # The disk detection can be somewhat wrong and the first few pixels can be of high intesnity.
    # We solve this by nullifying the first pixels until the minimum value.
    pixels[:(pixels > pixels.min()).argmin()] = pixels.min()

    pixels = pixels - pixels.min()
    max_y = np.median(np.sort(pixels)[-50:])

    result = []
    for t in thresholds: # Percent inhibition
        t_y = max_y * (1-t) # 1-t because at, e.g., 20% inhibition we'll be looking for the first (100%-20%=80%) intensity.
        x = (pixels >= t_y).argmax()
        result.append(x)

    return result

def plot_pixel_intensities(
    intensities, rads=None, ax=None, out_file=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = None
    ax.scatter(range(len(intensities)), intensities, s=5, color='black')

    if rads is not None:
        for r in rads:
            ax.axvline(r, color='red', linestyle='--')

    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Average Intensity")
    ax.set_title("Pixel Intensity Profile from Plate Center")
    
    # Save the plot
    if out_file is not None:
        fig.savefig(out_file, dpi=300)
    plt.close(fig)

    return fig, ax

def draw_circles(
    in_image: Image.Image,
    center: tuple[float, float],
    radii: list[float],
    line_width: int = 2,
    circle_color="red",
) -> Image.Image:
    """
    Draw concentric circles on a copy of a PIL image.

    Parameters
    ----------
    in_image : PIL.Image.Image
        Input image.
    center : (x, y)
        Circle center in pixel coordinates.
    radii : list of float
        Radii of circles to draw (in pixels).
    line_width : int, default=2
        Width of the circle outlines.
    circle_color : color spec, default="red"
        Any PIL-compatible color (e.g., "red", (255, 0, 0), "#FF0000").

    Returns
    -------
    PIL.Image.Image
        A copy of the image with circles drawn.
    """
    if not radii:
        return in_image.copy()

    cx, cy = center
    out = in_image.copy()
    draw = ImageDraw.Draw(out)

    for r in radii:
        if r <= 0:
            continue

        # Bounding box for the circle
        bbox = [
            cx - r,
            cy - r,
            cx + r,
            cy + r,
        ]

        draw.ellipse(
            bbox,
            outline=circle_color,
            width=line_width,
        )

    return out