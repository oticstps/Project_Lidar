import sys
import math
import numpy as np
from collections import deque
from datetime import datetime
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
                             QGroupBox, QTextEdit, QProgressBar, QCheckBox,
                             QDoubleSpinBox, QSplitter, QFrame, QTabWidget,
                             QSlider, QFileDialog, QMessageBox, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPointF
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Circle, Wedge
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from rplidar import RPLidar
import serial.tools.list_ports

class MotionAnalyzer:
    """Analisis gerakan dan perubahan environment"""
    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)
        self.prev_scan = None
        
    def add_scan(self, scan_data):
        """Tambah scan baru ke history"""
        points = {}
        for quality, angle, distance in scan_data:
            angle_key = int(angle)
            points[angle_key] = distance
        self.history.append(points)
        
    def detect_motion(self, threshold=200):
        """Deteksi gerakan dengan membandingkan scan"""
        if len(self.history) < 2:
            return []
        
        current = self.history[-1]
        previous = self.history[-2]
        
        motion_zones = []
        for angle in current:
            if angle in previous:
                diff = abs(current[angle] - previous[angle])
                if diff > threshold and current[angle] > 0 and previous[angle] > 0:
                    motion_zones.append({
                        'angle': angle,
                        'distance': current[angle],
                        'change': diff
                    })
        
        return motion_zones
    
    def get_velocity_map(self):
        """Hitung peta kecepatan perubahan"""
        if len(self.history) < 2:
            return {}
        
        velocity = {}
        current = self.history[-1]
        previous = self.history[-2]
        
        for angle in current:
            if angle in previous:
                velocity[angle] = current[angle] - previous[angle]
        
        return velocity

class MapBuilder:
    """Builder untuk membuat map 2D dari scan LIDAR"""
    def __init__(self, resolution=50):
        self.resolution = resolution  # mm per cell
        self.occupancy_grid = {}
        self.scan_history = deque(maxlen=50)
        
    def add_scan(self, scan_data):
        """Tambah scan ke map"""
        self.scan_history.append(scan_data)
        
        for quality, angle, distance in scan_data:
            if distance > 0:
                angle_rad = math.radians(angle)
                x = distance * math.cos(angle_rad)
                y = distance * math.sin(angle_rad)
                
                # Convert ke grid coordinates
                grid_x = int(x / self.resolution)
                grid_y = int(y / self.resolution)
                
                key = (grid_x, grid_y)
                if key not in self.occupancy_grid:
                    self.occupancy_grid[key] = {'hits': 0, 'total': 0}
                
                self.occupancy_grid[key]['hits'] += 1
                self.occupancy_grid[key]['total'] += 1
    
    def get_occupancy_map(self):
        """Dapatkan occupancy map"""
        return self.occupancy_grid
    
    def get_obstacles(self, threshold=0.5):
        """Ekstrak obstacle dari map"""
        obstacles = []
        for (x, y), data in self.occupancy_grid.items():
            if data['total'] > 0:
                occupancy = data['hits'] / data['total']
                if occupancy > threshold:
                    obstacles.append({
                        'x': x * self.resolution,
                        'y': y * self.resolution,
                        'confidence': occupancy
                    })
        return obstacles
    
    def clear_map(self):
        """Reset map"""
        self.occupancy_grid = {}
        self.scan_history.clear()

class ClusterAnalyzer:
    """Analisis cluster untuk deteksi objek"""
    @staticmethod
    def find_clusters(scan_data, distance_threshold=300):
        """Temukan cluster dari point cloud"""
        if not scan_data:
            return []
        
        # Convert to cartesian
        points = []
        for quality, angle, distance in scan_data:
            if distance > 0:
                angle_rad = math.radians(angle)
                x = distance * math.cos(angle_rad)
                y = distance * math.sin(angle_rad)
                points.append([x, y, angle, distance])
        
        if len(points) < 3:
            return []
        
        # Simple clustering by distance
        points = sorted(points, key=lambda p: p[2])  # Sort by angle
        clusters = []
        current_cluster = [points[0]]
        
        for i in range(1, len(points)):
            prev = current_cluster[-1]
            curr = points[i]
            
            # Calculate distance between consecutive points
            dist = math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
            
            if dist < distance_threshold:
                current_cluster.append(curr)
            else:
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                current_cluster = [curr]
        
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)
        
        # Calculate cluster properties
        cluster_info = []
        for cluster in clusters:
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            cluster_info.append({
                'center': (center_x, center_y),
                'size': len(cluster),
                'points': cluster,
                'bbox': (min(xs), max(xs), min(ys), max(ys))
            })
        
        return cluster_info

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

class EnhancedPolarPlot(FigureCanvas):
    """Enhanced polar plot dengan motion detection"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 10), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_ylim(0, 8000)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_facecolor('#1a1a1a')
        
        # Multiple layers
        self.scatter_main = self.ax.scatter([], [], s=3, c='cyan', alpha=0.7, label='Current')
        self.scatter_motion = self.ax.scatter([], [], s=50, c='red', marker='x', alpha=0.9, label='Motion')
        self.scatter_history = self.ax.scatter([], [], s=1, c='gray', alpha=0.3, label='History')
        
        # Add legend
        self.ax.legend(loc='upper right', fontsize=8)
        
        self.history_angles = deque(maxlen=1000)
        self.history_distances = deque(maxlen=1000)
        
    def update_plot(self, scan_data, motion_zones=None):
        if not scan_data:
            return
        
        angles = []
        distances = []
        
        for quality, angle, distance in scan_data:
            angle_rad = math.radians(angle)
            angles.append(angle_rad)
            distances.append(distance)
            
            # Add to history
            self.history_angles.append(angle_rad)
            self.history_distances.append(distance)
        
        # Update main scatter
        if angles and distances:
            self.scatter_main.set_offsets(np.column_stack([angles, distances]))
            colors = plt.cm.viridis(np.array(distances) / 8000)
            self.scatter_main.set_color(colors)
        
        # Update history
        if len(self.history_angles) > 0:
            self.scatter_history.set_offsets(
                np.column_stack([list(self.history_angles), list(self.history_distances)])
            )
        
        # Update motion zones
        if motion_zones:
            motion_angles = [math.radians(m['angle']) for m in motion_zones]
            motion_distances = [m['distance'] for m in motion_zones]
            self.scatter_motion.set_offsets(np.column_stack([motion_angles, motion_distances]))
        else:
            self.scatter_motion.set_offsets(np.column_stack([[], []]))
        
        self.draw()

class MapVisualization(FigureCanvas):
    """Advanced map visualization dengan occupancy grid"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 10), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-8000, 8000)
        self.ax.set_ylim(-8000, 8000)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2, linestyle='--')
        self.ax.set_xlabel('X (mm)', fontsize=10)
        self.ax.set_ylabel('Y (mm)', fontsize=10)
        self.ax.set_title('Occupancy Map & Object Detection', fontsize=12, fontweight='bold')
        self.ax.set_facecolor('#1a1a1a')
        
        # Robot position
        self.robot = Circle((0, 0), 200, color='green', alpha=0.5, label='Robot')
        self.ax.add_patch(self.robot)
        
        # Scatter plots
        self.scatter_points = self.ax.scatter([], [], s=2, c='cyan', alpha=0.6, label='Scan Points')
        self.scatter_obstacles = self.ax.scatter([], [], s=30, c='red', marker='s', alpha=0.8, label='Obstacles')
        
        self.ax.legend(loc='upper right', fontsize=8)
        
        self.cluster_patches = []
        
    def update_map(self, scan_data, obstacles=None, clusters=None):
        if not scan_data:
            return
        
        x_data = []
        y_data = []
        
        for quality, angle, distance in scan_data:
            angle_rad = math.radians(angle)
            x = distance * math.cos(angle_rad)
            y = distance * math.sin(angle_rad)
            x_data.append(x)
            y_data.append(y)
        
        # Update scan points
        if x_data and y_data:
            self.scatter_points.set_offsets(np.column_stack([x_data, y_data]))
        
        # Update obstacles
        if obstacles:
            obs_x = [o['x'] for o in obstacles]
            obs_y = [o['y'] for o in obstacles]
            self.scatter_obstacles.set_offsets(np.column_stack([obs_x, obs_y]))
        
        # Clear old cluster patches
        for patch in self.cluster_patches:
            patch.remove()
        self.cluster_patches = []
        
        # Draw clusters
        if clusters:
            for i, cluster in enumerate(clusters):
                if len(cluster['points']) >= 3:
                    points = [(p[0], p[1]) for p in cluster['points']]
                    try:
                        hull = ConvexHull(points)
                        hull_points = [points[i] for i in hull.vertices]
                        polygon = Polygon(hull_points, fill=False, edgecolor='yellow', 
                                        linewidth=2, linestyle='--', alpha=0.7)
                        self.ax.add_patch(polygon)
                        self.cluster_patches.append(polygon)
                        
                        # Add cluster center
                        cx, cy = cluster['center']
                        circle = Circle((cx, cy), 100, color='orange', alpha=0.5)
                        self.ax.add_patch(circle)
                        self.cluster_patches.append(circle)
                    except:
                        pass
        
        self.draw()

class HeatmapVisualization(FigureCanvas):
    """Heatmap visualization untuk density analysis"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 10), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Density Heatmap', fontsize=12, fontweight='bold')
        self.grid_size = 160  # 8000mm / 50mm resolution
        self.heatmap_data = np.zeros((self.grid_size, self.grid_size))
        self.im = None
        
    def update_heatmap(self, scan_data):
        if not scan_data:
            return
        
        # Decay existing data
        self.heatmap_data *= 0.95
        
        for quality, angle, distance in scan_data:
            if distance > 0:
                angle_rad = math.radians(angle)
                x = distance * math.cos(angle_rad)
                y = distance * math.sin(angle_rad)
                
                # Convert to grid
                grid_x = int((x + 8000) / 100)
                grid_y = int((y + 8000) / 100)
                
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    self.heatmap_data[grid_y, grid_x] += 1
        
        # Apply gaussian filter for smoothing
        smoothed = gaussian_filter(self.heatmap_data, sigma=2)
        
        if self.im is None:
            self.im = self.ax.imshow(smoothed, cmap='hot', interpolation='bilinear',
                                    extent=[-8000, 8000, -8000, 8000], origin='lower')
            self.fig.colorbar(self.im, ax=self.ax, label='Density')
        else:
            self.im.set_data(smoothed)
            self.im.set_clim(vmin=0, vmax=np.max(smoothed))
        
        self.draw()

class LidarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lidar_worker = None
        self.motion_analyzer = MotionAnalyzer(history_size=10)
        self.map_builder = MapBuilder(resolution=50)
        self.scan_count = 0
        self.recording = False
        self.recorded_scans = []
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("🚀 RPLIDAR A1M8 Advanced Analytics Suite")
        self.setGeometry(50, 50, 1600, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left control panel
        left_panel = self.create_advanced_control_panel()
        left_panel.setMaximumWidth(380)
        
        # Right visualization panel
        right_panel = self.create_advanced_plot_panel()
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)
        
        # Enhanced status bar
        self.status_label = QLabel("🟢 Ready | Waiting for connection...")
        self.statusBar().addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: 0")
        self.statusBar().addPermanentWidget(self.fps_label)
        
        self.scan_progress = QProgressBar()
        self.scan_progress.setMaximum(100)
        self.statusBar().addPermanentWidget(QLabel("Scans:"))
        self.statusBar().addPermanentWidget(self.scan_progress)
        
        self.update_ports()
        
    def create_advanced_control_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("⚡ LIDAR Control Center")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #00d4ff; padding: 10px;")
        layout.addWidget(header)
        
        # Connection Group
        conn_group = self.create_connection_group()
        layout.addWidget(conn_group)
        
        # Advanced Settings
        settings_group = self.create_advanced_settings_group()
        layout.addWidget(settings_group)
        
        # Analysis Controls
        analysis_group = self.create_analysis_group()
        layout.addWidget(analysis_group)
        
        # Recording Controls
        record_group = self.create_recording_group()
        layout.addWidget(record_group)
        
        # Live Statistics
        stats_group = self.create_statistics_group()
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
    def create_connection_group(self):
        group = QGroupBox("🔌 Connection")
        layout = QVBoxLayout(group)
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.refresh_ports_btn = QPushButton("🔄")
        self.refresh_ports_btn.setMaximumWidth(40)
        self.refresh_ports_btn.clicked.connect(self.update_ports)
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.refresh_ports_btn)
        layout.addLayout(port_layout)
        
        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("▶ Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.disconnect_btn = QPushButton("⏹ Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_lidar)
        self.disconnect_btn.setEnabled(False)
        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.disconnect_btn)
        layout.addLayout(btn_layout)
        
        return group
    
    def create_advanced_settings_group(self):
        group = QGroupBox("⚙️ Advanced Settings")
        layout = QGridLayout(group)
        
        layout.addWidget(QLabel("Min Quality:"), 0, 0)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(0, 15)
        self.quality_spin.setValue(5)
        layout.addWidget(self.quality_spin, 0, 1)
        
        layout.addWidget(QLabel("Max Distance:"), 1, 0)
        self.dist_spin = QSpinBox()
        self.dist_spin.setRange(1000, 12000)
        self.dist_spin.setValue(8000)
        self.dist_spin.setSuffix(" mm")
        layout.addWidget(self.dist_spin, 1, 1)
        
        layout.addWidget(QLabel("Update Rate:"), 2, 0)
        self.update_rate = QSlider(Qt.Horizontal)
        self.update_rate.setRange(1, 10)
        self.update_rate.setValue(5)
        layout.addWidget(self.update_rate, 2, 1)
        
        return group
    
    def create_analysis_group(self):
        group = QGroupBox("🔍 Analysis Tools")
        layout = QVBoxLayout(group)
        
        self.motion_check = QCheckBox("🎯 Motion Detection")
        self.motion_check.setChecked(True)
        layout.addWidget(self.motion_check)
        
        motion_layout = QHBoxLayout()
        motion_layout.addWidget(QLabel("Threshold:"))
        self.motion_threshold = QSpinBox()
        self.motion_threshold.setRange(50, 1000)
        self.motion_threshold.setValue(200)
        self.motion_threshold.setSuffix(" mm")
        motion_layout.addWidget(self.motion_threshold)
        layout.addLayout(motion_layout)
        
        self.cluster_check = QCheckBox("🎲 Object Clustering")
        self.cluster_check.setChecked(True)
        layout.addWidget(self.cluster_check)
        
        cluster_layout = QHBoxLayout()
        cluster_layout.addWidget(QLabel("Cluster Dist:"))
        self.cluster_distance = QSpinBox()
        self.cluster_distance.setRange(100, 1000)
        self.cluster_distance.setValue(300)
        self.cluster_distance.setSuffix(" mm")
        cluster_layout.addWidget(self.cluster_distance)
        layout.addLayout(cluster_layout)
        
        self.mapping_check = QCheckBox("🗺️ Build Occupancy Map")
        self.mapping_check.setChecked(True)
        layout.addWidget(self.mapping_check)
        
        clear_map_btn = QPushButton("🧹 Clear Map")
        clear_map_btn.clicked.connect(self.clear_map)
        layout.addWidget(clear_map_btn)
        
        return group
    
    def create_recording_group(self):
        group = QGroupBox("⏺️ Recording")
        layout = QVBoxLayout(group)
        
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        
        self.recorded_count = QLabel("Recorded: 0 scans")
        layout.addWidget(self.recorded_count)
        
        save_btn = QPushButton("💾 Save Data")
        save_btn.clicked.connect(self.save_recorded_data)
        layout.addWidget(save_btn)
        
        return group
    
    def create_statistics_group(self):
        group = QGroupBox("📊 Live Statistics")
        layout = QVBoxLayout(group)
        
        self.points_label = QLabel("Points: 0")
        self.quality_label = QLabel("Avg Quality: 0.0")
        self.distance_label = QLabel("Avg Distance: 0 mm")
        self.motion_label = QLabel("Motion Zones: 0")
        self.clusters_label = QLabel("Detected Objects: 0")
        
        for label in [self.points_label, self.quality_label, self.distance_label,
                     self.motion_label, self.clusters_label]:
            label.setStyleSheet("padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
            layout.addWidget(label)
        
        return group
    
    def create_advanced_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        
        # Enhanced Polar View
        polar_tab = QWidget()
        polar_layout = QVBoxLayout(polar_tab)
        self.polar_plot = EnhancedPolarPlot(self)
        polar_layout.addWidget(self.polar_plot)
        self.tabs.addTab(polar_tab, "🎯 Enhanced Polar View")
        
        # Map View
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        self.map_plot = MapVisualization(self)
        map_layout.addWidget(self.map_plot)
        self.tabs.addTab(map_tab, "🗺️ Object Detection Map")
        
        # Heatmap View
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        self.heatmap_plot = HeatmapVisualization(self)
        heatmap_layout.addWidget(self.heatmap_plot)
        self.tabs.addTab(heatmap_tab, "🔥 Density Heatmap")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def update_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
        
        if not ports:
            self.port_combo.addItem("No ports found")
    
    def toggle_connection(self):
        if self.lidar_worker and self.lidar_worker.isRunning():
            self.disconnect_lidar()
        else:
            self.connect_lidar()
    
    def connect_lidar(self):
        port_text = self.port_combo.currentText()
        if not port_text or "No ports" in port_text:
            QMessageBox.warning(self, "Error", "No valid port selected!")
            return
        
        port = port_text.split(" - ")[0]
        
        try:
            self.lidar_worker = LidarWorker(port)
            self.lidar_worker.set_min_quality(self.quality_spin.value())
            self.lidar_worker.data_ready.connect(self.update_data)
            self.lidar_worker.status_update.connect(self.update_status)
            self.lidar_worker.error_occurred.connect(self.handle_error)
            
            self.lidar_worker.start()
            
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.status_label.setText(f"🟡 Connecting to {port}...")
            
        except Exception as e:
            self.handle_error(str(e))
    
    def disconnect_lidar(self):
        if self.lidar_worker:
            self.lidar_worker.stop_lidar()
            self.lidar_worker.wait(1000)
            self.lidar_worker = None
        
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.status_label.setText("🔴 Disconnected")
    
    def update_data(self, scan_data):
        self.scan_count += 1
        self.scan_progress.setValue(self.scan_count % 100)
        
        if not scan_data:
            return
        
        # Update analyzers
        self.motion_analyzer.add_scan(scan_data)
        
        if self.mapping_check.isChecked():
            self.map_builder.add_scan(scan_data)
        
        # Detect motion
        motion_zones = []
        if self.motion_check.isChecked():
            motion_zones = self.motion_analyzer.detect_motion(self.motion_threshold.value())
        
        # Detect clusters
        clusters = []
        if self.cluster_check.isChecked():
            clusters = ClusterAnalyzer.find_clusters(scan_data, self.cluster_distance.value())
        
        # Get obstacles from map
        obstacles = self.map_builder.get_obstacles() if self.mapping_check.isChecked() else []
        
        # Update plots
        self.polar_plot.update_plot(scan_data, motion_zones)
        self.map_plot.update_map(scan_data, obstacles, clusters)
        self.heatmap_plot.update_heatmap(scan_data)
        
        # Update statistics
        self.update_statistics(scan_data, motion_zones, clusters)
        
        # Recording
        if self.recording:
            self.recorded_scans.append({
                'timestamp': datetime.now().isoformat(),
                'scan': scan_data,
                'motion': motion_zones,
                'clusters': len(clusters)
            })
            self.recorded_count.setText(f"Recorded: {len(self.recorded_scans)} scans")
    
    def update_statistics(self, scan_data, motion_zones, clusters):
        if scan_data:
            qualities = [p[0] for p in scan_data]
            distances = [p[2] for p in scan_data]
            
            avg_quality = sum(qualities) / len(qualities)
            avg_distance = sum(distances) / len(distances)
            
            self.points_label.setText(f"📍 Points: {len(scan_data)}")
            self.quality_label.setText(f"⭐ Avg Quality: {avg_quality:.2f}")
            self.distance_label.setText(f"📏 Avg Distance: {avg_distance:.0f} mm")
            self.motion_label.setText(f"🎯 Motion Zones: {len(motion_zones)}")
            self.clusters_label.setText(f"🎲 Detected Objects: {len(clusters)}")
            
            # Update FPS
            self.fps_label.setText(f"FPS: {self.scan_count % 10}")
    
    def update_status(self, message):
        if "connected" in message.lower():
            self.status_label.setText(f"🟢 {message}")
        elif "disconnected" in message.lower():
            self.status_label.setText(f"🔴 {message}")
        else:
            self.status_label.setText(message)
    
    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.record_btn.setText("⏹ Stop Recording")
            self.record_btn.setStyleSheet("background-color: #ff4444;")
            self.recorded_scans = []
        else:
            self.record_btn.setText("🔴 Start Recording")
            self.record_btn.setStyleSheet("")
    
    def save_recorded_data(self):
        if not self.recorded_scans:
            QMessageBox.warning(self, "No Data", "No recorded data to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Recorded Data", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.recorded_scans, f, indent=2)
                QMessageBox.information(self, "Success", 
                    f"Saved {len(self.recorded_scans)} scans to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
    
    def clear_map(self):
        self.map_builder.clear_map()
        QMessageBox.information(self, "Map Cleared", "Occupancy map has been reset!")
    
    def handle_error(self, error_msg):
        self.status_label.setText(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "LIDAR Error", error_msg)
        self.disconnect_lidar()
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit Confirmation',
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.disconnect_lidar()
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    
    # Modern dark theme
    app.setStyle('Fusion')
    dark_palette = QPalette()
    
    # Colors
    dark_palette.setColor(QPalette.Window, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Text, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 60, 60))
    dark_palette.setColor(QPalette.Link, QColor(0, 212, 255))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(dark_palette)
    
    # Custom stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #232323;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #3a3a3a;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #4a4a4a;
            border: 1px solid #5a5a5a;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QPushButton:pressed {
            background-color: #3a3a3a;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666666;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #3a3a3a;
            border: 1px solid #5a5a5a;
            padding: 5px;
            border-radius: 3px;
        }
        QProgressBar {
            border: 1px solid #5a5a5a;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #00d4ff;
        }
        QTabWidget::pane {
            border: 1px solid #3a3a3a;
            background-color: #1a1a1a;
        }
        QTabBar::tab {
            background-color: #3a3a3a;
            color: #cccccc;
            padding: 8px 15px;
            margin: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #00d4ff;
            color: #000000;
            font-weight: bold;
        }
        QTabBar::tab:hover {
            background-color: #4a4a4a;
        }
    """)
    
    window = LidarGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
