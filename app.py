"""
Local Video Frame Analyzer — Entry Point

A desktop GUI tool for testing local Vision/Video AI models.
Handles frame batching → AI analysis → JSON results → final aggregation.
"""
import sys
import os

# Ensure the project root is on sys.path so package imports work
# when running directly with `python app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Local Video Frame Analyzer")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
