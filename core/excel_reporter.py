"""
Excel report writer for multi-video batch processing.

Writes a summary row per video into an Excel file, including
batch timing, total duration, path to final_result.json,
and the final analysis summary.

Includes robust handling for Windows file locks (PermissionError)
when files are opened in Microsoft Excel or other programs:
- Pre-validation of writability
- Automatic retry on lock
- Automatic fallback saving to timestamped backup files to prevent data loss
"""
import os
import time
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


def check_excel_file_writable(excel_path: str) -> tuple[bool, str]:
    """Check if the Excel file is writable and not locked by Microsoft Excel or another program.

    Returns:
        (is_writable: bool, reason_message: str)
    """
    if not excel_path:
        return False, "Đường dẫn file Excel không hợp lệ (trống)."

    abs_path = os.path.abspath(excel_path)
    dir_path = os.path.dirname(abs_path)

    # 1. Check/create parent directory
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            return False, f"Không thể tạo thư mục chứa file Excel:\n{dir_path}\nLỗi: {e}"

    # 2. Check if Excel temporary lock file exists (~$filename)
    base_name = os.path.basename(abs_path)
    lock_file = os.path.join(dir_path, f"~${base_name}")
    if os.path.exists(lock_file):
        return False, (
            f"Tệp Excel đang được mở trong Microsoft Excel:\n{abs_path}\n"
            f"(Phát hiện tệp khóa tạm: ~${base_name})\n"
            f"Vui lòng lưu và đóng file Excel trước khi tiếp tục."
        )

    # 3. Check existing file writability
    if os.path.exists(abs_path):
        if not os.access(abs_path, os.W_OK):
            return False, (
                f"Tệp Excel đang ở chế độ chỉ đọc (Read-Only) hoặc không có quyền ghi:\n{abs_path}\n"
                f"Vui lòng kiểm tra thuộc tính tệp."
            )
        try:
            with open(abs_path, "r+b"):
                pass
        except PermissionError:
            return False, (
                f"Tệp Excel đang được mở trong Microsoft Excel hoặc một chương trình khác:\n{abs_path}\n"
                f"Vui lòng đóng file Excel trước khi tiếp tục."
            )
        except Exception as e:
            return False, f"Không thể mở tệp Excel để ghi:\n{e}"
    else:
        # File doesn't exist yet, test directory writability
        test_file = os.path.join(dir_path, f".test_perm_{os.getpid()}.tmp")
        try:
            with open(test_file, "wb") as f:
                f.write(b"")
        except PermissionError:
            return False, (
                f"Không có quyền tạo file trong thư mục:\n{dir_path}\n"
                f"Vui lòng chọn thư mục khác hoặc cấp quyền ghi cho thư mục này."
            )
        except Exception as e:
            return False, f"Thư mục không thể ghi:\n{e}"
        finally:
            if os.path.exists(test_file):
                try:
                    os.remove(test_file)
                except Exception:
                    pass

    return True, ""


def save_workbook_safe(
    wb,
    excel_path: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    log_fn=None,
) -> tuple[bool, str, str]:
    """Save an openpyxl Workbook with retry and fallback handling for locked files.

    If the file is locked (PermissionError, e.g. open in Excel), it will retry
    max_retries times, waiting retry_delay seconds between attempts.

    If saving to excel_path still fails after retries, it saves to a timestamped
    backup/fallback file (e.g. filename_backup_YYYYMMDD_HHMMSS.xlsx) to ensure
    NO DATA IS LOST.

    Returns:
        (success: bool, actual_saved_path: str, message: str)
    """
    abs_path = os.path.abspath(excel_path)
    dir_name = os.path.dirname(abs_path)
    if dir_name and not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            err = f"Không thể tạo thư mục: {dir_name} ({e})"
            if log_fn:
                log_fn(err)
            return False, "", err

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            wb.save(abs_path)
            if attempt > 1 and log_fn:
                log_fn(f"Đã lưu thành công vào '{abs_path}' (sau lần thử {attempt}).")
            return True, abs_path, ""
        except PermissionError as e:
            last_err = e
            if attempt < max_retries:
                if log_fn:
                    log_fn(
                        f"⚠️ Tệp '{os.path.basename(abs_path)}' đang bị khóa (đang mở trong Excel). "
                        f"Đang thử lại sau {retry_delay}s... (Lần {attempt}/{max_retries}). "
                        f"Vui lòng đóng file Excel!"
                    )
                time.sleep(retry_delay)
        except Exception as e:
            last_err = e
            break

    # If saving to abs_path failed, save to fallback file to prevent data loss!
    base_name = os.path.splitext(os.path.basename(abs_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_path = os.path.join(dir_name, f"{base_name}_backup_{timestamp}.xlsx")

    try:
        wb.save(fallback_path)
        warn_msg = (
            f"Không thể lưu vào '{abs_path}' do tệp đang mở trong Excel hoặc bị khóa.\n"
            f"-> ĐÃ LƯU DỰ PHÒNG TOÀN BỘ KẾT QUẢ VÀO:\n   {fallback_path}\n"
            f"Vui lòng đóng file Excel để các video tiếp theo ghi vào file chính."
        )
        if log_fn:
            log_fn(warn_msg)
        return True, fallback_path, warn_msg
    except Exception as fallback_err:
        # Last resort: try user's home directory
        home_fallback = os.path.expanduser(f"~/{base_name}_backup_{timestamp}.xlsx")
        try:
            wb.save(home_fallback)
            warn_msg = (
                f"Không thể ghi vào thư mục đích. Đã lưu dự phòng vào:\n   {home_fallback}"
            )
            if log_fn:
                log_fn(warn_msg)
            return True, home_fallback, warn_msg
        except Exception as home_err:
            err_msg = f"Lỗi lưu file Excel ({last_err}). Lưu dự phòng cũng thất bại: {home_err}"
            if log_fn:
                log_fn(err_msg)
            return False, "", err_msg


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

    abs_path = os.path.abspath(excel_path)
    dir_path = os.path.dirname(abs_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(abs_path):
        wb = load_workbook(abs_path)
        ws = wb.active
        # If sheet is empty, initialize header
        if ws.max_row <= 1 and ws.cell(row=1, column=1).value is None:
            ws.title = "Multi-Video Results"
            _apply_header_style(ws)
            try:
                wb.save(abs_path)
            except Exception:
                pass
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Multi-Video Results"
        _apply_header_style(ws)
        wb.save(abs_path)

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
    wb=None,
    ws=None,
    log_fn=None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Append a result row for one video to the Excel file with lock protection.

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
        wb: Optional existing Workbook instance (to avoid reloading and preserve state).
        ws: Optional existing Worksheet instance.
        log_fn: Optional callable for logging retry/fallback status.
        max_retries: Number of retries on PermissionError.
        retry_delay: Delay between retries in seconds.

    Returns:
        dict with:
            'workbook': Workbook,
            'worksheet': Worksheet,
            'saved_path': str,
            'is_fallback': bool,
            'success': bool,
            'warning': str,
    """
    if wb is None or ws is None:
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

    success, saved_path, warning_msg = save_workbook_safe(
        wb,
        excel_path,
        max_retries=max_retries,
        retry_delay=retry_delay,
        log_fn=log_fn,
    )

    abs_excel = os.path.abspath(excel_path)
    is_fallback = (os.path.abspath(saved_path) != abs_excel) if saved_path else True

    return {
        'workbook': wb,
        'worksheet': ws,
        'saved_path': saved_path,
        'is_fallback': is_fallback,
        'success': success,
        'warning': warning_msg,
    }
