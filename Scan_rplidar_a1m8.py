import sys
import math
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
                             QGroupBox, QTextEdit, QProgressBar, QCheckBox,
                             QDoubleSpinBox, QSplitter, QFrame, QTabWidget)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from rplidar import RPLidar
import serial.tools.list_ports

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
        
    def set_min_quality(self, quality):
        self.min_quality = quality
        
    def run(self):
        try:
            self.lidar = RPLidar(self.port)
            self.status_update.emit(f"LIDAR connected to {self.port}")
            self.running = True
            
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break
                
                # Filter data berdasarkan quality
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

class RealTimePlot(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=100):
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

class CartesianPlot(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-8000, 8000)
        self.ax.set_ylim(-8000, 8000)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_title('Cartesian View')
        
        # Initialize scatter plot
        self.scatter = self.ax.scatter([], [], s=2, c='red', alpha=0.6)
        self.x_data = []
        self.y_data = []
        self.distances = []
        
    def update_plot(self, scan_data):
        if not scan_data:
            return
            
        self.x_data = []
        self.y_data = []
        self.distances = []
        
        for quality, angle, distance in scan_data:
            angle_rad = math.radians(angle)
            x = distance * math.cos(angle_rad)
            y = distance * math.sin(angle_rad)
            
            self.x_data.append(x)
            self.y_data.append(y)
            self.distances.append(distance)
        
        # Update scatter plot
        if self.x_data and self.y_data:
            self.scatter.set_offsets(np.column_stack([self.x_data, self.y_data]))
            self.scatter.set_array(np.array(self.distances))
            self.scatter.set_clim(0, 8000)
            
        self.draw()

class LidarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lidar_worker = None
        self.init_ui()
        self.scan_count = 0
        
    def init_ui(self):
        self.setWindowTitle("RPLIDAR A1M8 GUI Controller")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = self.create_control_panel()
        left_panel.setMaximumWidth(350)
        
        # Right panel for plots
        right_panel = self.create_plot_panel()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Status bar
        self.status_label = QLabel("Ready to connect")
        self.statusBar().addWidget(self.status_label)
        
        # Progress bar for scan count
        self.scan_progress = QProgressBar()
        self.scan_progress.setMaximum(100)
        self.statusBar().addPermanentWidget(QLabel("Scan Count:"))
        self.statusBar().addPermanentWidget(self.scan_progress)
        
        # Update available ports
        self.update_ports()
        
    def create_control_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("RPLIDAR A1M8 Controller")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Connection Group
        conn_group = QGroupBox("Connection Settings")
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
        
        layout.addWidget(scan_group)
        
        # Data Display Group
        data_group = QGroupBox("Scan Information")
        data_layout = QVBoxLayout(data_group)
        
        self.data_display = QTextEdit()
        self.data_display.setMaximumHeight(150)
        self.data_display.setReadOnly(True)
        data_layout.addWidget(self.data_display)
        
        # Statistics
        stats_layout = QVBoxLayout()
        self.points_label = QLabel("Points: 0")
        self.quality_label = QLabel("Avg Quality: 0.0")
        self.distance_label = QLabel("Avg Distance: 0.0 mm")
        stats_layout.addWidget(self.points_label)
        stats_layout.addWidget(self.quality_label)
        stats_layout.addWidget(self.distance_label)
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
        self.obstacle_label.setStyleSheet("color: red; font-weight: bold;")
        obstacle_layout.addWidget(self.obstacle_label)
        
        layout.addWidget(obstacle_group)
        
        # Spacer at the bottom
        layout.addStretch()
        
        return panel
        
    def create_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        self.tabs = QTabWidget()
        
        # Polar plot tab
        polar_tab = QWidget()
        polar_layout = QVBoxLayout(polar_tab)
        self.polar_plot = RealTimePlot(self, width=10, height=8)
        polar_layout.addWidget(self.polar_plot)
        self.tabs.addTab(polar_tab, "Polar View")
        
        # Cartesian plot tab
        cartesian_tab = QWidget()
        cartesian_layout = QVBoxLayout(cartesian_tab)
        self.cartesian_plot = CartesianPlot(self, width=10, height=8)
        cartesian_layout.addWidget(self.cartesian_plot)
        self.tabs.addTab(cartesian_tab, "Cartesian View")
        
        layout.addWidget(self.tabs)
        
        return panel
        
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
        self.polar_plot.update_plot(scan_data)
        self.cartesian_plot.update_plot(scan_data)
        
        # Calculate statistics
        if scan_data:
            qualities = [point[0] for point in scan_data]
            distances = [point[2] for point in scan_data]
            
            avg_quality = sum(qualities) / len(qualities)
            avg_distance = sum(distances) / len(distances)
            
            self.points_label.setText(f"Points: {len(scan_data)}")
            self.quality_label.setText(f"Avg Quality: {avg_quality:.2f}")
            self.distance_label.setText(f"Avg Distance: {avg_distance:.1f} mm")
            
            # Update data display
            self.data_display.clear()
            self.data_display.append(f"Scan #{self.scan_count} - {len(scan_data)} points")
            for i, (quality, angle, distance) in enumerate(scan_data[:10]):  # Show first 10 points
                self.data_display.append(f"  {angle:6.2f}° | {distance:6.0f} mm | Quality: {quality}")
            if len(scan_data) > 10:
                self.data_display.append(f"  ... and {len(scan_data) - 10} more points")
            
            # Obstacle detection
            if self.obstacle_check.isChecked():
                self.detect_obstacles(scan_data)
        
    def detect_obstacles(self, scan_data):
        threshold = self.threshold_spin.value()
        start_angle = self.sector_start.value()
        end_angle = self.sector_end.value()
        
        obstacles = []
        for quality, angle, distance in scan_data:
            if (distance < threshold and distance > 0 and 
                start_angle <= angle <= end_angle):
                obstacles.append((angle, distance))
        
        if obstacles:
            closest = min(obstacles, key=lambda x: x[1])
            self.obstacle_label.setText(
                f"🚨 Obstacle at {closest[0]:.1f}° | {closest[1]:.0f} mm"
            )
            self.obstacle_label.setStyleSheet("color: red; font-weight: bold; background-color: yellow;")
        else:
            self.obstacle_label.setText("No obstacles detected")
            self.obstacle_label.setStyleSheet("color: green; font-weight: bold;")
            
    def handle_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        self.disconnect_lidar()
        
    def closeEvent(self, event):
        self.disconnect_lidar()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = LidarGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
