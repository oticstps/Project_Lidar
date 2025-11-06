import sys
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
    QGroupBox, QTextEdit, QProgressBar, QCheckBox,
    QDoubleSpinBox, QFrame, QTabWidget, QSlider,
    QGridLayout, QSplitter, QStatusBar, QMessageBox,
    QFileDialog, QMenu, QAction, QStyle, QToolButton
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QSettings
from PyQt5.QtGui import QFont, QPalette, QColor, QKeySequence, QIcon, QLinearGradient, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
from rplidar import RPLidar
import serial.tools.list_ports
import json
import csv
from datetime import datetime

class LidarWorker(QThread):
    data_ready = pyqtSignal(list)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.lidar = None
        self.running = False
        self.min_quality = 0
        self.scan_mode = 'normal'
        
    def set_min_quality(self, quality):
        self.min_quality = quality
        
    def set_scan_mode(self, mode):
        self.scan_mode = mode
        
    def run(self):
        try:
            self.lidar = RPLidar(self.port)
            info = self.lidar.get_info()
            self.status_update.emit(f"LIDAR connected: {info}")
            self.running = True
            
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break
                
                # Filter data based on quality
                filtered_scan = [point for point in scan if point[0] >= self.min_quality]
                self.data_ready.emit(filtered_scan)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.stop_lidar()
            
    def stop_lidar(self):
        self.running = False
        if self.lidar:
            self.lidar.stop()
            self.lidar.disconnect()
            self.status_update.emit("LIDAR disconnected")

class CADPlot(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create subplots
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-8000, 8000)
        self.ax.set_ylim(-8000, 8000)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_title('CAD View - LIDAR Scanner')
        
        # Initialize scatter plot
        self.scatter = self.ax.scatter([], [], s=2, c='blue', alpha=0.6, picker=True)
        self.x_data = []
        self.y_data = []
        self.distances = []
        self.angles = []
        
        # Zoom and pan variables
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.drag_start = None
        self.is_panning = False
        
        # Connect events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        
        # Measurement tools
        self.measurement_line = None
        self.measurement_start = None
        self.measurement_end = None
        
        # ROI (Region of Interest)
        self.roi_rectangle = None
        self.roi_start = None
        self.roi_active = False

    def update_plot(self, scan_data):
        if not scan_data:
            return
            
        self.x_data = []
        self.y_data = []
        self.distances = []
        self.angles = []
        
        for quality, angle, distance in scan_data:
            angle_rad = math.radians(angle)
            x = distance * math.cos(angle_rad)
            y = distance * math.sin(angle_rad)
            
            self.x_data.append(x)
            self.y_data.append(y)
            self.distances.append(distance)
            self.angles.append(angle)
        
        # Update scatter plot
        if self.x_data and self.y_data:
            self.scatter.set_offsets(np.column_stack([self.x_data, self.y_data]))
            self.scatter.set_array(np.array(self.distances))
            self.scatter.set_clim(0, 8000)
            
        self.draw()

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            # Get current limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Calculate zoom factor
            zoom_factor = 1.2 if event.button == 'up' else 1/1.2
            
            # Calculate new limits centered on mouse position
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            new_xlim = [event.xdata - (event.xdata - xlim[0]) * zoom_factor,
                       event.xdata + (xlim[1] - event.xdata) * zoom_factor]
            new_ylim = [event.ydata - (event.ydata - ylim[0]) * zoom_factor,
                       event.ydata + (ylim[1] - event.ydata) * zoom_factor]
            
            # Apply new limits
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.draw()

    def on_press(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:  # Left click
                if self.roi_active:
                    self.start_roi_selection(event)
                else:
                    self.start_measurement(event)
            elif event.button == 3:  # Right click
                self.is_panning = True
                self.press = (event.x, event.y)
                self.cur_xlim = self.ax.get_xlim()
                self.cur_ylim = self.ax.get_ylim()

    def on_motion(self, event):
        if self.is_panning and self.press:
            dx = (event.x - self.press[0]) / self.fig.bbox.width
            dy = (event.y - self.press[1]) / self.fig.bbox.height
            dx = (self.cur_xlim[1] - self.cur_xlim[0]) * dx
            dy = (self.cur_ylim[1] - self.cur_ylim[0]) * dy
            self.ax.set_xlim(self.cur_xlim[0] - dx, self.cur_xlim[1] - dx)
            self.ax.set_ylim(self.cur_ylim[0] - dy, self.cur_ylim[1] - dy)
            self.fig.canvas.draw()

    def on_release(self, event):
        self.is_panning = False
        self.press = None

    def on_pick(self, event):
        # Handle point selection
        if event.mouseevent.button == 1:
            ind = event.ind[0]
            x = self.x_data[ind]
            y = self.y_data[ind]
            dist = self.distances[ind]
            angle = self.angles[ind]
            
            # Highlight selected point
            self.ax.scatter([x], [y], s=50, c='red', marker='x')
            self.draw()
            
            # Show point info
            print(f"Point selected: X={x:.2f}, Y={y:.2f}, Distance={dist:.2f}mm, Angle={angle:.2f}°")

    def start_measurement(self, event):
        if self.measurement_start is None:
            self.measurement_start = (event.xdata, event.ydata)
            self.ax.scatter([event.xdata], [event.ydata], s=30, c='red', marker='o')
        else:
            self.measurement_end = (event.xdata, event.ydata)
            self.ax.scatter([event.xdata], [event.ydata], s=30, c='red', marker='o')
            
            # Draw measurement line
            if self.measurement_line:
                self.measurement_line.remove()
            self.measurement_line, = self.ax.plot(
                [self.measurement_start[0], self.measurement_end[0]],
                [self.measurement_start[1], self.measurement_end[1]],
                'r--', linewidth=1
            )
            
            # Calculate distance
            dist = np.sqrt(
                (self.measurement_end[0] - self.measurement_start[0])**2 +
                (self.measurement_end[1] - self.measurement_start[1])**2
            )
            
            # Add distance annotation
            mid_x = (self.measurement_start[0] + self.measurement_end[0]) / 2
            mid_y = (self.measurement_start[1] + self.measurement_end[1]) / 2
            self.ax.annotate(f'{dist:.2f} mm', (mid_x, mid_y), 
                            color='red', fontsize=10, ha='center')
            
            self.draw()
            self.measurement_start = None
            self.measurement_end = None

    def start_roi_selection(self, event):
        if self.roi_start is None:
            self.roi_start = (event.xdata, event.ydata)
        else:
            # Create ROI rectangle
            x1, y1 = self.roi_start
            x2, y2 = (event.xdata, event.ydata)
            
            if self.roi_rectangle:
                self.roi_rectangle.remove()
            
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            
            self.roi_rectangle = Rectangle((x_min, y_min), width, height, 
                                          linewidth=2, edgecolor='red', 
                                          facecolor='none', linestyle='--')
            self.ax.add_patch(self.roi_rectangle)
            self.draw()
            self.roi_start = None
            self.roi_active = False

    def reset_view(self):
        self.ax.set_xlim(-8000, 8000)
        self.ax.set_ylim(-8000, 8000)
        self.draw()

    def toggle_roi_mode(self):
        self.roi_active = not self.roi_active
        return self.roi_active

class PolarPlot(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_ylim(0, 8000)
        self.ax.grid(True)
        
        # Initialize scatter plot
        self.scatter = self.ax.scatter([], [], s=2, c='blue', alpha=0.6)
        self.angles = []
        self.distances = []
        
        # Connect events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def update_plot(self, scan_data):
        if not scan_data:
            return
            
        self.angles = []
        self.distances = []
        
        for quality, angle, distance in scan_data:
            # Convert to radians for polar plot
            angle_rad = math.radians(angle)
            self.angles.append(angle_rad)
            self.distances.append(distance)
        
        # Update scatter plot
        if self.angles and self.distances:
            self.scatter.set_offsets(np.column_stack([self.angles, self.distances]))
            self.scatter.set_array(np.array(self.distances))
            self.scatter.set_clim(0, 8000)
            
        self.draw()

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            # Get current radial limit
            rlim = self.ax.get_ylim()
            
            # Calculate zoom factor
            zoom_factor = 1.2 if event.button == 'up' else 1/1.2
            
            # Apply new limit
            new_rlim = [rlim[0], rlim[1] * zoom_factor]
            self.ax.set_ylim(new_rlim)
            self.draw()

class ObstacleDetector:
    def __init__(self):
        self.threshold = 1000
        self.sector_start = 0
        self.sector_end = 90
        self.obstacles = []
        
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def set_sector(self, start, end):
        self.sector_start = start
        self.sector_end = end
        
    def detect(self, scan_data):
        obstacles = []
        for quality, angle, distance in scan_data:
            if (distance < self.threshold and distance > 0 and 
                self.sector_start <= angle <= self.sector_end):
                obstacles.append((angle, distance))
        
        self.obstacles = obstacles
        return obstacles

class LidarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lidar_worker = None
        self.scan_count = 0
        self.obstacle_detector = ObstacleDetector()
        self.settings = QSettings('LIDAR_GUI', 'Settings')
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("Advanced LIDAR CAD Viewer")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for left/right panels
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left panel for controls
        self.left_panel = self.create_control_panel()
        self.left_panel.setMaximumWidth(400)
        self.left_panel.setMinimumWidth(300)
        
        # Right panel for plots
        self.right_panel = self.create_plot_panel()
        
        # Add panels to splitter
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([400, 1200])
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready to connect")
        self.status_bar.addWidget(self.status_label)
        
        # Add scan count to status bar
        self.scan_progress = QProgressBar()
        self.scan_progress.setMaximum(100)
        self.scan_progress.setValue(0)
        self.status_bar.addPermanentWidget(QLabel("Scan:"))
        self.status_bar.addPermanentWidget(self.scan_progress)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Update available ports
        self.update_ports()
        
        # Connect signals
        self.quality_spin.valueChanged.connect(self.on_quality_change)
        self.dist_spin.valueChanged.connect(self.on_distance_change)
        self.obstacle_check.stateChanged.connect(self.on_obstacle_toggle)
        self.sector_start.valueChanged.connect(self.on_sector_change)
        self.sector_end.valueChanged.connect(self.on_sector_change)
        self.threshold_spin.valueChanged.connect(self.on_threshold_change)
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        save_action = QAction('Save Scan Data', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_scan_data)
        file_menu.addAction(save_action)
        
        export_action = QAction('Export Plot', self)
        export_action.triggered.connect(self.export_plot)
        file_menu.addAction(export_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('Ctrl+R')
        reset_view_action.triggered.connect(self.reset_plot_view)
        view_menu.addAction(reset_view_action)
        
        toggle_roi_action = QAction('Toggle ROI Mode', self)
        toggle_roi_action.setShortcut('Ctrl+I')
        toggle_roi_action.triggered.connect(self.toggle_roi_mode)
        view_menu.addAction(toggle_roi_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        measure_action = QAction('Measurement Tool', self)
        measure_action.setShortcut('Ctrl+M')
        measure_action.triggered.connect(self.toggle_measurement_mode)
        tools_menu.addAction(measure_action)
        
        calibrate_action = QAction('Calibrate', self)
        calibrate_action.triggered.connect(self.calibrate_lidar)
        tools_menu.addAction(calibrate_action)
        
    def create_control_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #cccccc;
            }
            QLabel {
                color: black;
            }
            QGroupBox {
                font-weight: bold;
                color: black;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #f0f8ff;
                border: 1px solid #a0c4e8;
                border-radius: 5px;
                padding: 5px;
                color: black;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d0e6ff;
            }
            QPushButton:pressed {
                background-color: #b0d0ff;
            }
            QComboBox {
                border: 1px solid #a0c4e8;
                border-radius: 3px;
                padding: 3px;
                background-color: white;
                color: black;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #a0c4e8;
                border-radius: 3px;
                padding: 3px;
                background-color: white;
                color: black;
            }
            QTextEdit {
                border: 1px solid #a0c4e8;
                border-radius: 3px;
                background-color: white;
                color: black;
            }
            QCheckBox {
                color: black;
            }
            QSlider::groove:horizontal {
                border: 1px solid #a0c4e8;
                height: 8px;
                background: white;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #a0c4e8;
                border: 1px solid #7a9bc7;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("LIDAR Scanner Control")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: #add8e6; padding: 10px; color: black; border-radius: 5px;")
        layout.addWidget(title)
        
        # Connection Group
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_group)
        
        # Port selection
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.refresh_ports_btn = QPushButton("Refresh")
        self.refresh_ports_btn.clicked.connect(self.update_ports)
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.refresh_ports_btn)
        conn_layout.addLayout(port_layout)
        
        # Connection buttons
        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_lidar)
        self.disconnect_btn.setEnabled(False)
        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.disconnect_btn)
        conn_layout.addLayout(btn_layout)
        
        layout.addWidget(conn_group)
        
        # Scan Settings Group
        scan_group = QGroupBox("Scan Settings")
        scan_layout = QVBoxLayout(scan_group)
        
        # Minimum quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Min Quality:"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(0, 15)
        self.quality_spin.setValue(5)
        quality_layout.addWidget(self.quality_spin)
        scan_layout.addLayout(quality_layout)
        
        # Max distance
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Max Distance (mm):"))
        self.dist_spin = QSpinBox()
        self.dist_spin.setRange(1000, 12000)
        self.dist_spin.setValue(8000)
        dist_layout.addWidget(self.dist_spin)
        scan_layout.addLayout(dist_layout)
        
        # Scan mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Scan Mode:"))
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["Normal", "Express", "Boost"])
        mode_layout.addWidget(self.scan_mode_combo)
        scan_layout.addLayout(mode_layout)
        
        layout.addWidget(scan_group)
        
        # Data Display Group
        data_group = QGroupBox("Scan Information")
        data_layout = QVBoxLayout(data_group)
        
        self.data_display = QTextEdit()
        self.data_display.setMaximumHeight(150)
        self.data_display.setReadOnly(True)
        data_layout.addWidget(self.data_display)
        
        # Statistics
        stats_layout = QGridLayout()
        self.points_label = QLabel("Points: 0")
        self.quality_label = QLabel("Avg Quality: 0.0")
        self.distance_label = QLabel("Avg Distance: 0.0 mm")
        self.angle_label = QLabel("Angle Range: 0°-360°")
        stats_layout.addWidget(QLabel("Points:"), 0, 0)
        stats_layout.addWidget(self.points_label, 0, 1)
        stats_layout.addWidget(QLabel("Avg Quality:"), 1, 0)
        stats_layout.addWidget(self.quality_label, 1, 1)
        stats_layout.addWidget(QLabel("Avg Distance:"), 2, 0)
        stats_layout.addWidget(self.distance_label, 2, 1)
        stats_layout.addWidget(QLabel("Angle Range:"), 3, 0)
        stats_layout.addWidget(self.angle_label, 3, 1)
        data_layout.addLayout(stats_layout)
        
        layout.addWidget(data_group)
        
        # Obstacle Detection Group
        obstacle_group = QGroupBox("Obstacle Detection")
        obstacle_layout = QVBoxLayout(obstacle_group)
        
        self.obstacle_check = QCheckBox("Enable Obstacle Detection")
        obstacle_layout.addWidget(self.obstacle_check)
        
        # Sector settings
        sector_layout = QHBoxLayout()
        sector_layout.addWidget(QLabel("Sector:"))
        self.sector_start = QDoubleSpinBox()
        self.sector_start.setRange(0, 359)
        self.sector_start.setValue(0)
        sector_layout.addWidget(self.sector_start)
        sector_layout.addWidget(QLabel("to"))
        self.sector_end = QDoubleSpinBox()
        self.sector_end.setRange(0, 359)
        self.sector_end.setValue(90)
        sector_layout.addWidget(self.sector_end)
        obstacle_layout.addLayout(sector_layout)
        
        # Obstacle threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold (mm):"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(100, 5000)
        self.threshold_spin.setValue(1000)
        threshold_layout.addWidget(self.threshold_spin)
        obstacle_layout.addLayout(threshold_layout)
        
        self.obstacle_label = QLabel("No obstacles detected")
        self.obstacle_label.setStyleSheet("color: green; font-weight: bold;")
        obstacle_layout.addWidget(self.obstacle_label)
        
        layout.addWidget(obstacle_group)
        
        # Tools Group
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 100)
        self.zoom_slider.setValue(50)
        self.zoom_slider.valueChanged.connect(self.on_zoom_change)
        zoom_layout.addWidget(self.zoom_slider)
        tools_layout.addLayout(zoom_layout)
        
        # Reset view button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_plot_view)
        tools_layout.addWidget(reset_btn)
        
        layout.addWidget(tools_group)
        
        # Spacer at the bottom
        layout.addStretch()
        
        return panel
        
    def create_plot_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: black;
                padding: 8px;
                border: 1px solid #cccccc;
                border-bottom-color: #e0e0e0;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                font-weight: bold;
            }
        """)
        
        # Cartesian plot tab
        cartesian_tab = QWidget()
        cartesian_layout = QVBoxLayout(cartesian_tab)
        self.cartesian_plot = CADPlot(self, width=10, height=8)
        cartesian_layout.addWidget(self.cartesian_plot)
        self.tabs.addTab(cartesian_tab, "CAD View")
        
        # Polar plot tab
        polar_tab = QWidget()
        polar_layout = QVBoxLayout(polar_tab)
        self.polar_plot = PolarPlot(self, width=10, height=8)
        polar_layout.addWidget(self.polar_plot)
        self.tabs.addTab(polar_tab, "Polar View")
        
        layout.addWidget(self.tabs)
        
        # Add control buttons for splitter
        control_layout = QHBoxLayout()
        
        # Button to show/hide control panel
        self.toggle_panel_btn = QPushButton("<<")
        self.toggle_panel_btn.clicked.connect(self.toggle_panel)
        control_layout.addWidget(self.toggle_panel_btn)
        
        # Spacer
        control_layout.addStretch()
        
        # Button to maximize view
        maximize_btn = QPushButton("Maximize View")
        maximize_btn.clicked.connect(self.maximize_view)
        control_layout.addWidget(maximize_btn)
        
        layout.addLayout(control_layout)
        
        return panel
        
    def toggle_panel(self):
        if self.left_panel.isVisible():
            self.left_panel.hide()
            self.toggle_panel_btn.setText(">>")
        else:
            self.left_panel.show()
            self.toggle_panel_btn.setText("<<")
            
    def maximize_view(self):
        # Toggle between normal and maximized view
        if self.left_panel.isVisible():
            self.splitter.setSizes([0, self.width()])
            self.left_panel.hide()
            self.toggle_panel_btn.setText(">>")
        else:
            self.splitter.setSizes([400, self.width() - 400])
            self.left_panel.show()
            self.toggle_panel_btn.setText("<<")
        
    def update_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)
        
        if not ports:
            self.port_combo.addItem("No ports found")
            
    def toggle_connection(self):
        if self.lidar_worker and self.lidar_worker.isRunning():
            self.disconnect_lidar()
        else:
            self.connect_lidar()
            
    def connect_lidar(self):
        port = self.port_combo.currentText()
        if not port or port == "No ports found":
            self.status_label.setText("No valid port selected")
            return
            
        try:
            self.lidar_worker = LidarWorker(port)
            self.lidar_worker.set_min_quality(self.quality_spin.value())
            self.lidar_worker.set_scan_mode(self.scan_mode_combo.currentText().lower())
            self.lidar_worker.data_ready.connect(self.update_data)
            self.lidar_worker.status_update.connect(self.status_label.setText)
            self.lidar_worker.error_occurred.connect(self.handle_error)
            
            self.lidar_worker.start()
            
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.status_label.setText(f"Connecting to {port}...")
            
        except Exception as e:
            self.handle_error(str(e))
            
    def disconnect_lidar(self):
        if self.lidar_worker:
            self.lidar_worker.stop_lidar()
            self.lidar_worker.wait(1000)  # Wait max 1 second
            self.lidar_worker = None
            
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.status_label.setText("Disconnected")
        
    def update_data(self, scan_data):
        self.scan_count += 1
        self.scan_progress.setValue(self.scan_count % 100)
        
        # Update plots
        self.cartesian_plot.update_plot(scan_data)
        self.polar_plot.update_plot(scan_data)
        
        # Calculate statistics
        if scan_data:
            qualities = [point[0] for point in scan_data]
            distances = [point[2] for point in scan_data]
            
            avg_quality = sum(qualities) / len(qualities)
            avg_distance = sum(distances) / len(distances)
            
            self.points_label.setText(f"{len(scan_data)}")
            self.quality_label.setText(f"{avg_quality:.2f}")
            self.distance_label.setText(f"{avg_distance:.1f} mm")
            
            # Update data display
            self.data_display.clear()
            self.data_display.append(f"Scan #{self.scan_count} - {len(scan_data)} points")
            for i, (quality, angle, distance) in enumerate(scan_data[:5]):  # Show first 5 points
                self.data_display.append(f"  {angle:6.2f}° | {distance:6.0f} mm | Quality: {quality}")
            if len(scan_data) > 5:
                self.data_display.append(f"  ... and {len(scan_data) - 5} more points")
            
            # Obstacle detection
            if self.obstacle_check.isChecked():
                obstacles = self.obstacle_detector.detect(scan_data)
                if obstacles:
                    closest = min(obstacles, key=lambda x: x[1])
                    self.obstacle_label.setText(
                        f"🚨 Obstacle at {closest[0]:.1f}° | {closest[1]:.0f} mm"
                    )
                    self.obstacle_label.setStyleSheet("color: red; font-weight: bold; background-color: yellow;")
                else:
                    self.obstacle_label.setText("No obstacles detected")
                    self.obstacle_label.setStyleSheet("color: green; font-weight: bold;")
        
    def on_quality_change(self, value):
        if self.lidar_worker:
            self.lidar_worker.set_min_quality(value)
            
    def on_distance_change(self, value):
        # Update plot limits
        self.cartesian_plot.ax.set_xlim(-value, value)
        self.cartesian_plot.ax.set_ylim(-value, value)
        self.polar_plot.ax.set_ylim(0, value)
        self.cartesian_plot.draw()
        self.polar_plot.draw()
        
    def on_obstacle_toggle(self, state):
        # Reset obstacle label when disabled
        if not state:
            self.obstacle_label.setText("No obstacles detected")
            self.obstacle_label.setStyleSheet("color: green; font-weight: bold;")
        
    def on_sector_change(self, value):
        self.obstacle_detector.set_sector(
            self.sector_start.value(),
            self.sector_end.value()
        )
        
    def on_threshold_change(self, value):
        self.obstacle_detector.set_threshold(value)
        
    def on_zoom_change(self, value):
        # Calculate zoom factor from slider (1-100)
        zoom_factor = value / 50.0  # 50 is neutral
        current_xlim = self.cartesian_plot.ax.get_xlim()
        current_ylim = self.cartesian_plot.ax.get_ylim()
        
        center_x = (current_xlim[0] + current_xlim[1]) / 2
        center_y = (current_ylim[0] + current_ylim[1]) / 2
        
        range_x = (current_xlim[1] - current_xlim[0]) * (1/zoom_factor)
        range_y = (current_ylim[1] - current_ylim[0]) * (1/zoom_factor)
        
        new_xlim = [center_x - range_x/2, center_x + range_x/2]
        new_ylim = [center_y - range_y/2, center_y + range_y/2]
        
        self.cartesian_plot.ax.set_xlim(new_xlim)
        self.cartesian_plot.ax.set_ylim(new_ylim)
        self.cartesian_plot.draw()
        
    def reset_plot_view(self):
        self.cartesian_plot.reset_view()
        self.polar_plot.ax.set_ylim(0, 8000)
        self.polar_plot.draw()
        self.zoom_slider.setValue(50)
        
    def toggle_roi_mode(self):
        roi_active = self.cartesian_plot.toggle_roi_mode()
        if roi_active:
            self.status_label.setText("ROI Selection Mode: Click and drag to select region")
        else:
            self.status_label.setText("ROI Selection Mode: Deactivated")
            
    def toggle_measurement_mode(self):
        self.status_label.setText("Measurement Mode: Click two points to measure distance")
        
    def save_scan_data(self):
        if not self.lidar_worker or not self.lidar_worker.isRunning():
            QMessageBox.warning(self, "Warning", "No active scan data to save")
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Scan Data", "", 
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)", 
            options=options
        )
        
        if file_name:
            try:
                if file_name.endswith('.json'):
                    with open(file_name, 'w') as f:
                        json.dump({
                            'scan_data': [
                                {'quality': q, 'angle': a, 'distance': d} 
                                for q, a, d in self.cartesian_plot.scan_data
                            ],
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                elif file_name.endswith('.csv'):
                    with open(file_name, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Quality', 'Angle', 'Distance'])
                        for q, a, d in self.cartesian_plot.scan_data:
                            writer.writerow([q, a, d])
                
                self.status_label.setText(f"Data saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")
                
    def export_plot(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)", 
            options=options
        )
        
        if file_name:
            try:
                self.cartesian_plot.fig.savefig(file_name)
                self.status_label.setText(f"Plot exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")
                
    def calibrate_lidar(self):
        # Simple calibration - reset all settings to defaults
        reply = QMessageBox.question(
            self, "Calibrate LIDAR", 
            "This will reset all settings to defaults. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.quality_spin.setValue(5)
            self.dist_spin.setValue(8000)
            self.sector_start.setValue(0)
            self.sector_end.setValue(90)
            self.threshold_spin.setValue(1000)
            self.scan_mode_combo.setCurrentIndex(0)
            self.obstacle_check.setChecked(False)
            self.reset_plot_view()
            self.status_label.setText("LIDAR calibrated")
            
    def handle_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        self.disconnect_lidar()
        QMessageBox.critical(self, "LIDAR Error", f"An error occurred: {error_msg}")
        
    def closeEvent(self, event):
        self.disconnect_lidar()
        self.save_settings()
        event.accept()
        
    def save_settings(self):
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        self.settings.setValue('quality', self.quality_spin.value())
        self.settings.setValue('distance', self.dist_spin.value())
        self.settings.setValue('threshold', self.threshold_spin.value())
        
    def load_settings(self):
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
            
        quality = self.settings.value('quality', 5, type=int)
        distance = self.settings.value('distance', 8000, type=int)
        threshold = self.settings.value('threshold', 1000, type=int)
        
        self.quality_spin.setValue(quality)
        self.dist_spin.setValue(distance)
        self.threshold_spin.setValue(threshold)

def main():
    app = QApplication(sys.argv)
    
    # Set professional light theme
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(255, 255, 255))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(240, 240, 240))
    palette.setColor(QPalette.AlternateBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 0, 255))
    palette.setColor(QPalette.Highlight, QColor(173, 216, 230))  # Light blue
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = LidarGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
