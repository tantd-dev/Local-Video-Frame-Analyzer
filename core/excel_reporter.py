"""
Excel report writer for multi-video batch processing.

Writes a summary row per video into an Excel file, including
batch timing, total duration, path to final_result.json,
and the final analysis summary.
"""
import os
from datetime import datetime

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


HEADER_COLUMNS = [
    "No.",
    "Video Name",
    "Frame Count",
    "Batch Size",
    "Total Batches",
    "Successful Batches",
    "Failed Batches",
    "Batches Duration (s)",
    "Aggregation Duration (s)",
    "Total Duration (s)",
    "Final Result Path",
    "Summary",
    "Processed At",
]


def _apply_header_style(ws):
    """Apply styling to the header row."""
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    for col_idx, header in enumerate(HEADER_COLUMNS, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    # Set column widths
    widths = [6, 30, 12, 10, 12, 16, 14, 20, 22, 18, 60, 80, 22]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w


def create_or_load_workbook(excel_path: str):
    """Create a new Excel file or load existing one.

    Returns:
        (Workbook, Worksheet) tuple.
    """
    if not HAS_OPENPYXL:
        raise ImportError(
            "openpyxl is required for Excel export.\n"
            "Install it with: pip install openpyxl"
        )

    if os.path.exists(excel_path):
        wb = load_workbook(excel_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Multi-Video Results"
        _apply_header_style(ws)
        wb.save(excel_path)

    return wb, ws


def append_video_result(
    excel_path: str,
    video_name: str,
    frame_count: int,
    batch_size: int,
    total_batches: int,
    successful_batches: int,
    failed_batches: int,
    batches_duration: float,
    aggregation_duration: float,
    total_duration: float,
    final_result_path: str,
    summary_text: str,
):
    """Append a result row for one video to the Excel file.

    Args:
        excel_path: Path to the .xlsx file.
        video_name: Name of the video (folder name).
        frame_count: Number of frames in the video.
        batch_size: Frames per batch.
        total_batches: Total number of batches.
        successful_batches: Number of successful batches.
        failed_batches: Number of failed batches.
        batches_duration: Total batch processing time in seconds.
        aggregation_duration: Aggregation time in seconds.
        total_duration: Total time in seconds.
        final_result_path: Absolute path to final_result.json.
        summary_text: The final_analysis.summary text.
    """
    wb, ws = create_or_load_workbook(excel_path)

    row_num = ws.max_row + 1
    seq_num = row_num - 1  # Sequence number (excluding header)

    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    row_data = [
        seq_num,
        video_name,
        frame_count,
        batch_size,
        total_batches,
        successful_batches,
        failed_batches,
        round(batches_duration, 2),
        round(aggregation_duration, 2),
        round(total_duration, 2),
        final_result_path,
        summary_text,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]

    for col_idx, value in enumerate(row_data, 1):
        cell = ws.cell(row=row_num, column=col_idx, value=value)
        cell.border = thin_border
        if col_idx in (11, 12):  # Long text columns
            cell.alignment = Alignment(wrap_text=True)

    wb.save(excel_path)
