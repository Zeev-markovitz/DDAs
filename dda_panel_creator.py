#!/usr/bin/env python3

import re
import itertools
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
from gooey import Gooey, GooeyParser
from ddas import create_dda_grid, create_stacked_image

@dataclass
class PlateImage:
    file_path: Path
    plate_id: int
    timepoint_label: str
    filename_prefix: str
    metadata_label: Optional[str] = None
    genotype: Optional[str] = None

    @property
    def display_name(self) -> str:
        base_name = self.metadata_label or self.genotype or self.filename_prefix
        return f"{base_name} ({self.plate_id})" if base_name else str(self.plate_id)

def discover_plate_images(folder_paths: List[str]) -> List[PlateImage]:
    discovered_images = []
    
    plate_pattern = re.compile(r"Run-\d+-Plate-(\d+)\.png$")
    
    for folder in map(Path, folder_paths):
        for img_file in folder.glob("*.png"):
            match = plate_pattern.search(img_file.name)
            if match:
                plate_id = int(match.group(1))
                prefix = img_file.name.split("Run-")[0].rstrip("-")
                discovered_images.append(
                    PlateImage(img_file, plate_id, folder.name, prefix)
                )
    return discovered_images

def apply_metadata_from_excel(
    images: List[PlateImage],
    excel_path: str,
    sheet_name: str
):
    if not excel_path: 
        return
        
    try:
        metadata_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        metadata_df["Plate number"] = (
            metadata_df["Plate number"]
            .astype(str)
            .str.split(",")
            .apply(lambda ids: [plt_id.strip() for plt_id in ids])
        )
        metadata_df = metadata_df.explode("Plate number")
        metadata_df["Plate number"] = pd.to_numeric(metadata_df["Plate number"])
        
        metadata_records = metadata_df.sort_values("Plate number").to_dict("records")
        unique_plate_ids = sorted({img.plate_id for img in images})
        id_to_meta_map = dict(zip(unique_plate_ids, metadata_records))

        for img in images:
            if img.plate_id in id_to_meta_map:
                record = id_to_meta_map[img.plate_id]
                img.metadata_label = str(record.get("Label", "")) or None
                img.genotype = str(record.get("Genotype", "")) or None
                
    except Exception as error:
        warnings.warn(f"Metadata mapping failed: {error}")

def process_and_save_grids(
    images: List[PlateImage], 
    timepoints: List[str], 
    output_dirs: Dict[str, Path], 
    group_name: str = "all_plates",
    slice_height: int = 40,
    slice_width: int = 400,
):
    unique_ids = sorted({img.plate_id for img in images})

    id_to_label = {}
    for pid in unique_ids:
        representative_img = next(img for img in images if img.plate_id == pid)
        id_to_label[pid] = representative_img.display_name

    ordered_labels = [id_to_label[pid] for pid in unique_ids]
    
    image_grid_paths = []
    for tp in timepoints:
        row = [
            next((str(img.file_path) for img in images 
                  if img.timepoint_label == tp and img.plate_id == pid), None)
            for pid in unique_ids
        ]
        image_grid_paths.append(row)

    # vertical default
    create_dda_grid(
        image_grid_paths,
        ordered_labels,
        timepoints,
        output_dirs["grid_vertical"] / f"{group_name}.jpg"
    )
    
    create_dda_grid(
        image_grid_paths,
        ordered_labels,
        timepoints,
        output_dirs["grid_horizontal"] / f"{group_name}.jpg",
        orientation='horizontal'
    )
    
    for tp_start, tp_end in itertools.combinations(timepoints, 2):
        start_paths, end_paths, active_labels = [], [], []
        
        for pid in unique_ids:
            img_start = next((img for img in images if img.plate_id == pid and img.timepoint_label == tp_start), None)
            img_end = next((img for img in images if img.plate_id == pid and img.timepoint_label == tp_end), None)
            
            if img_start and img_end:
                start_paths.append(str(img_start.file_path))
                end_paths.append(str(img_end.file_path))
                active_labels.append(img_start.display_name)
        
        if not start_paths:
            continue
        
        # stacked & opposite
        layout_variants = itertools.product(["stacked", "opposite"], ["vertical", "horizontal"])
        for kind, orientation in layout_variants:
            file_name = f"{group_name}_{tp_start}_vs_{tp_end}.png"
            target_path = output_dirs[f"{kind}_{orientation}"] / file_name
            
            create_stacked_image(
                start_paths, end_paths, active_labels, 
                slice_height, 
                target_path, 
                layout=kind, 
                orientation=orientation,
                max_slice_width=slice_width
            )

def run_pipeline(args):
    timepoints = [Path(folder).name for folder in args.input_folders]
    root_output = Path(args.output_folder)
    
    sub_folders = [
        "grid_vertical", "grid_horizontal", 
        "stacked_vertical", "stacked_horizontal", 
        "opposite_vertical", "opposite_horizontal"
    ]
    dir_map = {name: root_output / name for name in sub_folders}
    for folder in dir_map.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    all_images = discover_plate_images(args.input_folders)
    apply_metadata_from_excel(all_images, args.excel, args.sheet)
    
    sort_order = [val.strip() for val in (args.order or "").split(",")]
    all_images.sort(key=lambda img: (
        sort_order.index(img.genotype) if img.genotype in sort_order else len(sort_order), 
        img.plate_id
    ))
    
    process_and_save_grids(all_images, timepoints, dir_map, slice_height=args.slice_height, slice_width=args.slice_width)
    
    unique_genotypes = sorted({img.genotype for img in all_images if img.genotype})
    for genotype in unique_genotypes:
        genotype_subset = [img for img in all_images if img.genotype == genotype]
        process_and_save_grids(genotype_subset, timepoints, dir_map, group_name=f"genotype_{genotype}", slice_height=args.slice_height, slice_width=args.slice_width)

@Gooey(
    program_name="DDA Grid Creator",
    default_size=(700, 720),
)
def main():
    parser = GooeyParser(description="Generate image grids and comparisons for DDAs.")
    
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "input_folders",
        nargs='+',
        widget="MultiDirChooser",
        metavar="Input folders",
        help="Input folders (two or three, corresponding to 24, 48 and potentially 72 hours)."\
        " When providing them manually, separate them using a colon (':')."
    )
    
    io_group.add_argument(
        "output_folder",
        widget="DirChooser",
        metavar="Output folder",
    )
    
    meta_group = parser.add_argument_group("Metadata")
    meta_group.add_argument(
        "--excel",
        widget="FileChooser",
        metavar="Excel with metadata (optional)",
        help="Excel with plate metadata. Three columns are possible: \n"+\
        "1) 'Plate number'\n"+\
        "2) 'Label' (optional, used to label each plate)\n"+\
        "3) 'Genotype' (optional, used for creating grouped panels and will be used as the label if no 'Label' column is provided)"
    )
    
    meta_group.add_argument(
        "--sheet",
        default="Sheet1",
        metavar="Sheet name",
        help="Sheet name in the Excel file"
    )
    
    meta_group.add_argument(
        "--order",
        default="",
        metavar="Sort order (optional)",
        help="Comma-separated list of Genotypes to define sort order"
    )

    visualization_group = parser.add_argument_group("Visualization")
    visualization_group.add_argument(
        "--slice-height",
        type=int,
        default=40,
        metavar="Disk diameter",
        help="Slice height in pixels - should be approximately the diameter of the disk.",
    )

    visualization_group.add_argument(
        "--slice-width",
        type=int,
        default=400,
        metavar="Distance from the disk to the edge of the plate",
        help="Slice width in pixels - the distance from the center of the disk to the edge of the plate you want to be displayed" +\
            " (doesn't have to be the whole plate).",
    )
    
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
