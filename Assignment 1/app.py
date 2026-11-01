import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QLineEdit, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QFrame, QMessageBox, QDialog, QFileDialog, QTabWidget,
)

from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ALGO_DISPATCH = {}

import search_algos as backend

if hasattr(backend, "BFS"): ALGO_DISPATCH["BFS"] = backend.BFS
if hasattr(backend, "DFS"): ALGO_DISPATCH["DFS"] = backend.DFS
if hasattr(backend, "IDFS"): ALGO_DISPATCH["IDFS"] = backend.IDFS
if hasattr(backend, "AStar_Manhattan"): ALGO_DISPATCH["AStar_Manhattan"] = backend.AStar_Manhattan
if hasattr(backend, "AStar_Euclidean"): ALGO_DISPATCH["AStar_Euclidean"] = backend.AStar_Euclidean

GOAL_STATE = (0, 1, 2, 3, 4, 5, 6, 7, 8)

def validate_board_input(raw_input) -> Tuple[bool, str, List[int]]:
    """Validates that input is an array of 9 distinct integers from 0 to 8."""
    if isinstance(raw_input, str):
        parts = raw_input.replace(",", " ").strip().split()
    elif isinstance(raw_input, (list, tuple)):
        parts = [str(x) for x in raw_input]
    else:
        return False, "Input must be a string or list.", []

    if len(parts) != 9:
        return False, f"Expected 9 elements, got {len(parts)}.", []
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        return False, "All elements must be integers.", []

    if sorted(numbers) != list(range(9)):
        return False, "Must contain digits 0 to 8 with no duplicates.", []

    return True, "", numbers


def count_inversions(state: List[int]) -> int:
    tiles = [x for x in state if x != 0]
    inv = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inv += 1
    return inv


def is_solvable(state: List[int]) -> bool:
    return count_inversions(state) % 2 == 0


def deduce_move_direction(s_prev: List[int], s_curr: List[int]) -> Tuple[str, str]:
    """Deduces move direction and moved tile between two consecutive states."""
    idx_prev = s_prev.index(0)
    idx_curr = s_curr.index(0)
    swapped_tile = str(s_curr[idx_prev])
    diff = idx_curr - idx_prev

    if diff == -3:
        direction = "Up"
    elif diff == 3:
        direction = "Down"
    elif diff == -1:
        direction = "Left"
    elif diff == 1:
        direction = "Right"
    else:
        direction = "Move"
    return direction, f"Tile {swapped_tile}"


def normalize_solution_path(initial_state: List[int], raw_path: List[List[int]]) -> List[List[int]]:
    """Ensures full sequence of states includes initial state at index 0."""
    if not raw_path:
        return [list(initial_state)]
    if list(raw_path[0]) == list(initial_state):
        return [list(s) for s in raw_path]
    return [list(initial_state)] + [list(s) for s in raw_path]


class SearchWorker(QThread):
    done = pyqtSignal(dict)
    fail = pyqtSignal(str)

    def __init__(self, algo_func, board: List[int], algo_name: str):
        super().__init__()
        self.algo_func = algo_func
        self.board = list(board)
        self.algo_name = algo_name

    def run(self):
        try:
            t0 = time.perf_counter()
            path, explored_l = self.algo_func(self.board)
            elapsed = time.perf_counter() - t0

            states = normalize_solution_path(self.board, path)
            cost = max(0, len(states) - 1)

            self.done.emit({
                "states": states,
                "cost": cost,
                "explored": explored_l,
                "depth": cost,
                "time": elapsed,
                "algo_name": self.algo_name,
            })
        except Exception as e:
            self.fail.emit(str(e))


class TileWidget(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(80, 80)
        self.setFont(QFont("sans-serif", 20, QFont.Weight.Bold))

    def set_tile(self, val: int, is_highlighted: bool = False):
        if val == 0:
            self.setText("")
            self.setStyleSheet("QLabel { background-color: #1A1A1A; border: 1px dashed #353535; border-radius: 6px; }")
        else:
            self.setText(str(val))
            border = "#3584E4" if is_highlighted else "#454545"
            bg = "#3A3A3A" if not is_highlighted else "#2C4A6F"
            self.setStyleSheet(f"""
                QLabel {{
                    background-color: {bg};
                    border: 1px solid {border};
                    border-radius: 6px;
                    color: #FFFFFF;
                }}
                QLabel:hover {{
                    background-color: #484848;
                }}
            """)

    def mousePressEvent(self, e):
        self.clicked.emit(self.idx)


class BoardWidget(QWidget):
    board_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.tiles = []
        layout = QGridLayout(self)
        layout.setSpacing(6)
        for i in range(9):
            tile = TileWidget(i)
            tile.clicked.connect(self._on_tile_clicked)
            r, c = divmod(i, 3)
            layout.addWidget(tile, r, c)
            self.tiles.append(tile)
        self.set_state(self.state)

    def set_state(self, state: List[int], highlight_idx: Optional[int] = None):
        self.state = list(state)
        for i, val in enumerate(self.state):
            self.tiles[i].set_tile(val, is_highlighted=(i == highlight_idx))

    def _on_tile_clicked(self, clicked_idx: int):
        b_idx = self.state.index(0)
        cr, cc = divmod(clicked_idx, 3)
        br, bc = divmod(b_idx, 3)
        if (abs(cr - br) == 1 and cc == bc) or (abs(cc - bc) == 1 and cr == br):
            new_s = list(self.state)
            new_s[b_idx], new_s[clicked_idx] = new_s[clicked_idx], new_s[b_idx]
            self.set_state(new_s, highlight_idx=b_idx)
            self.board_changed.emit(new_s)

class BenchmarkDialog(QDialog):
    """Benchmark window showing side-by-side comparison table & charts."""

    def __init__(self, initial_state: List[int], parent=None):
        super().__init__(parent)
        self.state = list(initial_state)
        self.setWindowTitle("Benchmark")
        self.resize(800, 560)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        state_str = ", ".join(str(x) for x in self.state)
        lbl_info = QLabel(f"State: [{state_str}]")
        lbl_info.setStyleSheet("color: #909090; font-size: 12px;")
        layout.addWidget(lbl_info)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Algorithm", "Cost", "Nodes Explored", "Depth", "Time (ms)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.fig = Figure(figsize=(7, 2.3), facecolor="#242424")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        btn_row = QHBoxLayout()
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.results = []
        self._run_benchmarks()

    def _run_benchmarks(self):
        algos = [
            ("BFS", ALGO_DISPATCH.get("BFS")),
            ("DFS", ALGO_DISPATCH.get("DFS")),
            ("IDFS", ALGO_DISPATCH.get("IDFS")),
            ("A* (Manhattan)", ALGO_DISPATCH.get("AStar_Manhattan")),
            ("A* (Euclidean)", ALGO_DISPATCH.get("AStar_Euclidean")),
        ]
        self.results = []
        labels, nodes, times = [], [], []

        for name, func in algos:
            if func is None:
                continue
            try:
                t0 = time.perf_counter()
                path, explored = func(list(self.state))
                elapsed = time.perf_counter() - t0
                states = normalize_solution_path(self.state, path)
                cost = max(0, len(states) - 1)
                self.results.append({
                    "name": name, "cost": cost, "explored": explored,
                    "depth": cost, "time_ms": elapsed * 1000.0, "success": True,
                })
                labels.append(name)
                nodes.append(explored)
                times.append(elapsed * 1000.0)
            except Exception as e:
                self.results.append({
                    "name": name, "cost": 0, "explored": 0, "depth": 0, "time_ms": 0.0, "success": False, "error": str(e)
                })

        self.table.setRowCount(len(self.results))
        for r, res in enumerate(self.results):
            self.table.setItem(r, 0, QTableWidgetItem(res["name"]))
            self.table.setItem(r, 1, QTableWidgetItem(str(res["cost"]) if res["success"] else "Error"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{res['explored']:,}" if res["success"] else "-"))
            self.table.setItem(r, 3, QTableWidgetItem(str(res["depth"]) if res["success"] else "-"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{res['time_ms']:.2f}" if res["success"] else "-"))

        if self.fig and labels:
            self.fig.clear()
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2)

            ax1.bar(range(len(labels)), nodes, color="#3584E4", edgecolor="#1C71D8")
            ax1.set_title("Nodes Explored", color="#DFDFDF", fontsize=9, fontweight="bold")
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=20, ha="right", color="#A0A0A0", fontsize=8)
            ax1.set_facecolor("#1E1E1E")
            ax1.tick_params(colors="#A0A0A0")
            ax1.grid(True, linestyle="--", alpha=0.15, color="#5E5E5E")

            ax2.bar(range(len(labels)), times, color="#9141AC", edgecolor="#772B94")
            ax2.set_title("Time (ms)", color="#DFDFDF", fontsize=9, fontweight="bold")
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=20, ha="right", color="#A0A0A0", fontsize=8)
            ax2.set_facecolor("#1E1E1E")
            ax2.tick_params(colors="#A0A0A0")
            ax2.grid(True, linestyle="--", alpha=0.15, color="#5E5E5E")

            self.fig.tight_layout()
            self.canvas.draw()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Benchmark CSV", "benchmark.csv", "CSV (*.csv)")
        if path:
            with open(path, "w") as f:
                f.write("Algorithm,Cost,Nodes,Depth,Time(ms)\n")
                for r in self.results:
                    f.write(f'"{r["name"]}",{r["cost"]},{r["explored"]},{r["depth"]},{r["time_ms"]:.3f}\n')
            QMessageBox.information(self, "Saved", "Benchmark CSV saved successfully.")


def _create_card(title: str, val: str) -> QFrame:
    f = QFrame()
    f.setStyleSheet("background-color: #2F2F2F; border: 1px solid #3D3D3D; border-radius: 5px; padding: 4px;")
    l = QVBoxLayout(f)
    l.setContentsMargins(6, 4, 6, 4)
    t = QLabel(title.upper())
    t.setStyleSheet("color: #8C8C8C; font-size: 10px; font-weight: 600;")
    v = QLabel(val)
    v.setObjectName("val")
    v.setStyleSheet("color: #FFFFFF; font-size: 15px; font-weight: bold;")
    l.addWidget(t)
    l.addWidget(v)
    return f


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.solution_states: List[List[int]] = []
        self.current_step = 0
        self.interval_ms = 300

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.setWindowTitle("8-Puzzle")
        self.resize(980, 660)

        qss_file = Path(__file__).parent / "theme.qss"
        self.setStyleSheet(qss_file.read_text(encoding="utf-8"))

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header_row = QHBoxLayout()
        lbl_app_name = QLabel()
        lbl_app_name.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        header_row.addWidget(lbl_app_name)
        header_row.addStretch()

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #F6D32D; font-weight: 500;")
        header_row.addWidget(self.lbl_status)
        root.addLayout(header_row)

        body = QHBoxLayout()
        body.setSpacing(14)

        left = QWidget()
        left.setFixedWidth(330)
        l_layout = QVBoxLayout(left)
        l_layout.setContentsMargins(0, 0, 0, 0)
        l_layout.setSpacing(10)

        grp_config = QGroupBox()
        gc_layout = QVBoxLayout(grp_config)
        gc_layout.setSpacing(10)

        gc_layout.addWidget(QLabel("Initial Board (9 numbers, 0 is blank):"))
        inp_row = QHBoxLayout()
        self.inp_array = QLineEdit("1,2,5,3,4,0,6,7,8")
        self.inp_array.textChanged.connect(self._on_input_changed)
        inp_row.addWidget(self.inp_array, 1)

        btn_rand = QPushButton("Random")
        btn_rand.clicked.connect(self._randomize)
        inp_row.addWidget(btn_rand)
        gc_layout.addLayout(inp_row)

        self.badge_solv = QLabel("Solvable (Inversions: 2)")
        self.badge_solv.setStyleSheet("background-color: #264A31; color: #8FF0A4; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
        gc_layout.addWidget(self.badge_solv)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine if hasattr(QFrame, "Shape") else QFrame.HLine)
        sep.setStyleSheet("color: #3D3D3D; margin: 4px 0;")
        gc_layout.addWidget(sep)

        gc_layout.addWidget(QLabel("Algorithm:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["A*", "BFS", "IDFS", "DFS"])
        self.combo_algo.currentIndexChanged.connect(self._on_algo_changed)
        gc_layout.addWidget(self.combo_algo)

        # Heuristic dropdown
        self.lbl_heur = QLabel("Heuristic:")
        gc_layout.addWidget(self.lbl_heur)
        self.combo_heur = QComboBox()
        self.combo_heur.addItems(["Manhattan Distance", "Euclidean Distance"])
        gc_layout.addWidget(self.combo_heur)

        gc_layout.addSpacing(6)

        self.btn_solve = QPushButton("Solve")
        self.btn_solve.clicked.connect(self._solve)
        gc_layout.addWidget(self.btn_solve)

        self.btn_bench = QPushButton("Benchmark")
        self.btn_bench.clicked.connect(self._benchmark)
        gc_layout.addWidget(self.btn_bench)

        l_layout.addWidget(grp_config)
        l_layout.addStretch()

        body.addWidget(left)

        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(10)

        m_row = QHBoxLayout()
        m_row.setSpacing(8)
        self.card_cost = _create_card("Cost", "-")
        self.card_nodes = _create_card("Explored", "-")
        self.card_depth = _create_card("Depth", "-")
        self.card_time = _create_card("Time", "- ms")
        m_row.addWidget(self.card_cost)
        m_row.addWidget(self.card_nodes)
        m_row.addWidget(self.card_depth)
        m_row.addWidget(self.card_time)
        r_layout.addLayout(m_row)

        self.tabs = QTabWidget()

        tab_board = QWidget()
        tb_layout = QVBoxLayout(tab_board)
        tb_layout.setContentsMargins(12, 12, 12, 12)
        tb_layout.setSpacing(10)

        self.board = BoardWidget()
        self.board.setFixedSize(280, 280)
        self.board.board_changed.connect(self._on_board_clicked)
        tb_layout.addWidget(self.board, 0, Qt.AlignmentFlag.AlignCenter)

        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #242424; border: 1px solid #363636; border-radius: 6px; padding: 6px;")
        cf_layout = QVBoxLayout(ctrl_frame)
        cf_layout.setSpacing(8)

        step_slider_row = QHBoxLayout()
        step_slider_label = QLabel("Step:")
        step_slider_label.setStyleSheet("color: #A0A0A0; font-size: 11px; min-width: 45px;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_scrub)
        self.slider.setEnabled(False)
        self.lbl_step_num = QLabel("0/0")
        self.lbl_step_num.setStyleSheet("color: #A0A0A0; font-size: 11px; min-width: 45px;")
        step_slider_row.addWidget(step_slider_label)
        step_slider_row.addWidget(self.slider, 1)
        step_slider_row.addWidget(self.lbl_step_num)
        cf_layout.addLayout(step_slider_row)

        speed_slider_row = QHBoxLayout()
        speed_slider_label = QLabel("Speed:")
        speed_slider_label.setStyleSheet("color: #A0A0A0; font-size: 11px; min-width: 45px;")
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(50, 1000)
        self.slider_speed.setValue(self.interval_ms)
        self.slider_speed.setInvertedAppearance(True)  # Left faster, right slower
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        self.lbl_speed_val = QLabel(f"{self.interval_ms} ms")
        self.lbl_speed_val.setStyleSheet("color: #A0A0A0; font-size: 11px; min-width: 45px;")
        speed_slider_row.addWidget(speed_slider_label)
        speed_slider_row.addWidget(self.slider_speed, 1)
        speed_slider_row.addWidget(self.lbl_speed_val)
        cf_layout.addLayout(speed_slider_row)

        btn_ctrl_row = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self._prev)
        btn_ctrl_row.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._toggle_play)
        btn_ctrl_row.addWidget(self.btn_play)

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._next)
        btn_ctrl_row.addWidget(self.btn_next)
        cf_layout.addLayout(btn_ctrl_row)

        tb_layout.addWidget(ctrl_frame)
        self.tabs.addTab(tab_board, "Board")

        tab_trace = QWidget()
        tt_layout = QVBoxLayout(tab_trace)
        self.table_trace = QTableWidget()
        self.table_trace.setColumnCount(4)
        self.table_trace.setHorizontalHeaderLabels(["Step", "Move", "Swapped", "Grid"])
        self.table_trace.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_trace.cellClicked.connect(lambda r, c: self.slider.setValue(r))
        tt_layout.addWidget(self.table_trace)

        btn_exp_trace = QPushButton("Export Trace")
        btn_exp_trace.clicked.connect(self._export_trace)
        tt_layout.addWidget(btn_exp_trace)

        self.tabs.addTab(tab_trace, "Trace Table")
        r_layout.addWidget(self.tabs, 1)

        body.addWidget(right, 1)
        root.addLayout(body, 1)

        self._on_input_changed()

    def _update_card(self, frame: QFrame, val: str):
        lbl = frame.findChild(QLabel, "val")
        if lbl:
            lbl.setText(val)

    def _on_algo_changed(self):
        is_astar = (self.combo_algo.currentText() == "A*")
        self.combo_heur.setEnabled(is_astar)
        self.lbl_heur.setEnabled(is_astar)

    def _on_speed_changed(self, value: int):
        self.interval_ms = value
        self.lbl_speed_val.setText(f"{value} ms")
        if self.timer.isActive():
            self.timer.setInterval(self.interval_ms)

    def _on_input_changed(self):
        valid, err, state = validate_board_input(self.inp_array.text())
        if not valid:
            self.badge_solv.setText("Invalid Format")
            self.badge_solv.setStyleSheet("background-color: #4D2626; color: #FFA0A0; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
            self.btn_solve.setEnabled(False)
            self.btn_bench.setEnabled(False)
            return

        solv = is_solvable(state)
        inv = count_inversions(state)
        if solv:
            self.badge_solv.setText(f"Solvable (Inversions: {inv})")
            self.badge_solv.setStyleSheet("background-color: #264A31; color: #8FF0A4; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
            self.btn_solve.setEnabled(True)
            self.btn_bench.setEnabled(True)
            self.board.set_state(state)
        else:
            self.badge_solv.setText(f"Unsolvable (Inversions: {inv})")
            self.badge_solv.setStyleSheet("background-color: #4D2626; color: #FFA0A0; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
            self.btn_solve.setEnabled(False)
            self.btn_bench.setEnabled(False)

    def _on_board_clicked(self, new_state: List[int]):
        self.inp_array.blockSignals(True)
        self.inp_array.setText(",".join(str(x) for x in new_state))
        self.inp_array.blockSignals(False)
        self._on_input_changed()

    def _randomize(self):
        curr = list(GOAL_STATE)
        for _ in range(25):
            idx = curr.index(0)
            r, c = divmod(idx, 3)
            adj = []
            if r > 0: adj.append((r - 1) * 3 + c)
            if r < 2: adj.append((r + 1) * 3 + c)
            if c > 0: adj.append(r * 3 + c - 1)
            if c < 2: adj.append(r * 3 + c + 1)
            swap_with = random.choice(adj)
            curr[idx], curr[swap_with] = curr[swap_with], curr[idx]
        self.inp_array.setText(",".join(str(x) for x in curr))

    def _solve(self):
        valid, _, state = validate_board_input(self.inp_array.text())
        if not valid: return

        algo_label = self.combo_algo.currentText()
        if algo_label == "A*":
            heur_label = self.combo_heur.currentText()
            algo_key = "AStar_Manhattan" if "Manhattan" in heur_label else "AStar_Euclidean"
        else:
            algo_key = algo_label

        func = ALGO_DISPATCH.get(algo_key)

        self.lbl_status.setText(f"Solving with {algo_key}...")
        self.btn_solve.setEnabled(False)

        self.worker = SearchWorker(func, state, algo_key)
        self.worker.done.connect(self._on_solved)
        self.worker.fail.connect(lambda err: QMessageBox.critical(self, "Error", err))
        self.worker.start()

    def _on_solved(self, res: dict):
        self.lbl_status.setText("")
        self.btn_solve.setEnabled(True)

        self.solution_states = res["states"]
        self._update_card(self.card_cost, str(res["cost"]))
        self._update_card(self.card_nodes, f"{res['explored']:,}")
        self._update_card(self.card_depth, str(res["depth"]))
        self._update_card(self.card_time, f"{res['time'] * 1000.0:.2f} ms")

        # Setup step scrubber
        total_steps = max(0, len(self.solution_states) - 1)
        self.slider.setEnabled(total_steps > 0)
        self.slider.setRange(0, total_steps)
        self.slider.setValue(0)
        self.current_step = 0
        self._update_step_view(0)

        # Populate Trace Table
        self.table_trace.setRowCount(len(self.solution_states))
        for i, s in enumerate(self.solution_states):
            self.table_trace.setItem(i, 0, QTableWidgetItem(str(i)))
            if i == 0:
                direction, tile = "START", "-"
            else:
                direction, tile = deduce_move_direction(self.solution_states[i - 1], s)
            self.table_trace.setItem(i, 1, QTableWidgetItem(direction))
            self.table_trace.setItem(i, 2, QTableWidgetItem(tile))
            grid = f"[{s[0]} {s[1]} {s[2]}] [{s[3]} {s[4]} {s[5]}] [{s[6]} {s[7]} {s[8]}]"
            self.table_trace.setItem(i, 3, QTableWidgetItem(grid))

    def _update_step_view(self, step: int):
        if not self.solution_states: return
        self.current_step = step
        s = self.solution_states[step]
        prev_blank = self.solution_states[step - 1].index(0) if step > 0 else None
        self.board.set_state(s, highlight_idx=prev_blank)
        total = max(0, len(self.solution_states) - 1)
        self.lbl_step_num.setText(f"{step}/{total}")
        self.table_trace.selectRow(step)

    def _on_scrub(self, val: int):
        self._update_step_view(val)

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            if self.current_step >= len(self.solution_states) - 1:
                self.current_step = 0
            self.timer.start(self.interval_ms)
            self.btn_play.setText("Pause")

    def _on_tick(self):
        if self.current_step < len(self.solution_states) - 1:
            self.slider.setValue(self.current_step + 1)
        else:
            self._toggle_play()

    def _prev(self):
        if self.current_step > 0:
            self.slider.setValue(self.current_step - 1)

    def _next(self):
        if self.solution_states and self.current_step < len(self.solution_states) - 1:
            self.slider.setValue(self.current_step + 1)

    def _export_trace(self):
        if not self.solution_states: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Trace File", "trace.txt", "Text (*.txt)")
        if path:
            with open(path, "w") as f:
                f.write(f"8-Puzzle Solution Trace ({len(self.solution_states) - 1} steps)\n\n")
                for i, s in enumerate(self.solution_states):
                    d, tile = ("START", "") if i == 0 else deduce_move_direction(self.solution_states[i - 1], s)
                    f.write(f"Step {i}: {d} {tile}\n+---+---+---+\n| {s[0]} | {s[1]} | {s[2]} |\n| {s[3]} | {s[4]} | {s[5]} |\n| {s[6]} | {s[7]} | {s[8]} |\n+---+---+---+\n\n")
            QMessageBox.information(self, "Exported", "Trace saved successfully.")

    def _benchmark(self):
        valid, _, state = validate_board_input(self.inp_array.text())
        if valid:
            BenchmarkDialog(state, parent=self).exec()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()