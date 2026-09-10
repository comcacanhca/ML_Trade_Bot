"""
Simple pyqtgraph-based interactive candlestick chart renderer.
This provides much smoother pan/zoom for large datasets using Qt/OpenGL.

Dependencies:
  pip install pyqt5 pyqtgraph

Usage:
  from plot_chart.renderer_pyqtgraph import show_interactive_chart
  show_interactive_chart(hist, range_list=[5,20,40], deals=..., results=..., BB=200)
"""
from typing import List, Optional
import pandas as pd

import numpy as np

try:
    # pyqtgraph bundles Qt abstraction helpers; import QtWidgets for QApplication
    from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
    import pyqtgraph as pg
except Exception:
    try:
        # fallback to direct PyQt5 imports if pyqtgraph.Qt doesn't expose QtWidgets
        from PyQt5 import QtGui, QtCore, QtWidgets
        import pyqtgraph as pg
    except Exception:
        raise ImportError('pyqtgraph and PyQt5 (or PySide2) are required for the interactive renderer.\nInstall with: pip install pyqt5 pyqtgraph')

from data_handler import DATE, OPEN, HIGH, LOW, CLOSE, SENKOUA, SENKUB



BULLISH_CANDLE_COLOR = '#2ecc71'
BEARISH_CANDLE_COLOR = '#ff4d4f'
CANDLE_WICK_COLOR = '#2b2b2b'
CANDLE_BORDER_COLOR = '#111111'
CANDLE_HALF_WIDTH = 0.32
CANDLE_WICK_WIDTH = 1.1
CANDLE_BODY_BORDER_WIDTH = 0.9
MIN_BODY_HEIGHT_RATIO = 0.00015
MIN_BODY_HEIGHT_ABSOLUTE = 1e-6
DAY_SEPARATOR_COLOR = '#808080'
DAY_SEPARATOR_ALPHA = 0.35
DAY_SEPARATOR_WIDTH = 1
DAY_SEPARATOR_STYLE = QtCore.Qt.DashLine
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 700
INDICATOR_PANEL_RATIO = 0.2
INDICATOR_GUIDE_COLOR = '#ff6b6b'
INDICATOR_GUIDE_ALPHA = 0.5
INDICATOR_Y_MIN = 0.0
INDICATOR_Y_MAX = 100.0





class CandlestickItem(pg.GraphicsObject):
    """A GraphicsObject that draws candlesticks from provided data_handler.

    Data format: list/array of tuples (x, open, close, low, high)
    """

    def __init__(self, data):
        super().__init__()
        self.data = np.asarray(data)
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        if len(self.data) == 0:
            p.end()
            return

        price_range = float(np.max(self.data[:, 4]) - np.min(self.data[:, 3]))
        min_body_height = max(price_range * MIN_BODY_HEIGHT_RATIO, MIN_BODY_HEIGHT_ABSOLUTE)

        wick_pen = QtGui.QPen(QtGui.QColor(CANDLE_WICK_COLOR))
        wick_pen.setWidthF(CANDLE_WICK_WIDTH)
        wick_pen.setCosmetic(True)

        body_border_pen = QtGui.QPen(QtGui.QColor(CANDLE_BORDER_COLOR))
        body_border_pen.setWidthF(CANDLE_BODY_BORDER_WIDTH)
        body_border_pen.setCosmetic(True)

        for (x, open_, close_, low, high) in self.data:
            is_bullish = close_ >= open_
            fill_color = QtGui.QColor(BULLISH_CANDLE_COLOR if is_bullish else BEARISH_CANDLE_COLOR)
            fill_color.setAlpha(235)

            body_low = float(min(open_, close_))
            body_high = float(max(open_, close_))
            body_height = body_high - body_low

            if body_height < min_body_height:
                body_center = float(open_ + close_) / 2.0
                body_low = body_center - (min_body_height / 2.0)
                body_height = min_body_height

            p.setPen(wick_pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawLine(QtCore.QPointF(x, low), QtCore.QPointF(x, high))

            p.setPen(body_border_pen)
            p.setBrush(QtGui.QBrush(fill_color))
            p.drawRect(QtCore.QRectF(x - CANDLE_HALF_WIDTH, body_low, 2 * CANDLE_HALF_WIDTH, body_height))

        p.end()

    def paint(self, p, *args):
        if self.picture is not None:
            p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if len(self.data) == 0:
            return QtCore.QRectF()
        xs = self.data[:, 0]
        lows = self.data[:, 3]
        highs = self.data[:, 4]
        return QtCore.QRectF(float(xs.min()) - 1, float(lows.min()) - 1,
                             float(xs.max() - xs.min()) + 2, float(highs.max() - lows.min()) + 2)


class YAxisScaleViewBox(pg.ViewBox):
    """Support Ctrl + drag on the Y axis to scale vertically without changing X."""

    @staticmethod
    def _has_ctrl_modifier(ev):
        ctrl_modifier = getattr(getattr(QtCore.Qt, 'KeyboardModifier', QtCore.Qt), 'ControlModifier')
        return bool(ev.modifiers() & ctrl_modifier)

    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis=axis)

    def mouseDragEvent(self, ev, axis=None):
        if axis == 1 and self._has_ctrl_modifier(ev) and ev.button() in [
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.MiddleButton,
        ]:
            ev.accept()
            screen_delta = ev.screenPos() - ev.lastScreenPos()
            scale_y = 1.02 ** screen_delta.y()
            self._resetTarget()
            self.scaleBy(y=scale_y)
            self.sigRangeChangedManually.emit([False, True])
            return

        if axis is None and ev.button() & QtCore.Qt.MouseButton.RightButton:
            axis = 1
        super().mouseDragEvent(ev, axis=axis)


def _get_xy_from_hist(hist):
    # Use integer indices for x for performance, map to date labels separately
    x = np.arange(len(hist), dtype=float)
    try:
        o = hist[OPEN].astype(float).to_numpy()
        c = hist[CLOSE].astype(float).to_numpy()
        h = hist[HIGH].astype(float).to_numpy()
        l = hist[LOW].astype(float).to_numpy()
    except Exception:
        # Fallback by column position assumption
        cols = list(hist.columns)
        o = hist[cols[1]].astype(float).to_numpy()
        l = hist[cols[2]].astype(float).to_numpy()
        h = hist[cols[3]].astype(float).to_numpy()
        c = hist[cols[4]].astype(float).to_numpy()
    return x, o, h, l, c


def _get_day_boundary_indices(hist: 'pd.DataFrame') -> list[int]:
    if DATE not in hist.columns or hist.empty:
        return []

    date_values = pd.to_datetime(hist[DATE], errors='coerce')
    day_boundaries = []
    for i in range(1, len(date_values)):
        previous_date = date_values.iloc[i - 1]
        current_date = date_values.iloc[i]
        if pd.isna(previous_date) or pd.isna(current_date):
            continue
        if current_date.date() != previous_date.date():
            day_boundaries.append(i)
    return day_boundaries


def _configure_plot_interaction(plot_widget: 'pg.PlotWidget', lock_y_autorange: bool = True) -> None:
    view_box = plot_widget.getViewBox()
    plot_widget.setMouseEnabled(x=True, y=True)
    view_box.setMouseEnabled(x=True, y=True)
    view_box.setMenuEnabled(True)
    view_box.setMouseMode(pg.ViewBox.PanMode)
    if lock_y_autorange:
        plot_widget.enableAutoRange(axis='y', enable=False)


def show_interactive_chart(hist: 'pd.DataFrame', range_list: Optional[List[int]] = None, deals: Optional[list] = None,
                            results: Optional[list] = None, BB: Optional[int] = None,
                            indicator_lines: Optional[List[str]] = None,
                            price_indicator_lines: Optional[List[str]] = None,
                            mfivas: tuple = (20, 80),
                            indicator_avg_period: int = 15) -> None:



    """Show an interactive chart using pyqtgraph.

    This function will create a native Qt window and block (exec_) until closed.
    """
    # prepare data_handler
    if range_list is None:
        range_list = []

    x, o, h, l, c = _get_xy_from_hist(hist)
    data = [(float(xi), float(o[i]), float(c[i]), float(l[i]), float(h[i])) for i, xi in enumerate(x)]

    # Create Qt application if not present
    app = QtWidgets.QApplication.instance()
    app_owner = False
    if app is None:
        app = QtWidgets.QApplication([])
        app_owner = True

    if indicator_lines is None:
        indicator_lines = []
    if price_indicator_lines is None:
        price_indicator_lines = []

    has_indicator_panel = len(indicator_lines) > 0

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Interactive Candlestick Chart (pyqtgraph)")
    window.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

    if has_indicator_panel:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plot = pg.PlotWidget(viewBox=YAxisScaleViewBox())
        indicator_plot = pg.PlotWidget(viewBox=YAxisScaleViewBox())
        indicator_plot.addLegend(offset=(10, 10))
        splitter.addWidget(plot)
        splitter.addWidget(indicator_plot)
        main_panel_height = int(DEFAULT_WINDOW_HEIGHT * (1 - INDICATOR_PANEL_RATIO))
        indicator_panel_height = int(DEFAULT_WINDOW_HEIGHT * INDICATOR_PANEL_RATIO)
        splitter.setSizes([main_panel_height, indicator_panel_height])
        splitter.setChildrenCollapsible(False)
        window.setCentralWidget(splitter)
        indicator_plot.showGrid(x=False, y=True, alpha=0.3)
        indicator_plot.setXLink(plot)
        indicator_plot.setYRange(INDICATOR_Y_MIN, INDICATOR_Y_MAX, padding=0)
        _configure_plot_interaction(indicator_plot)

    else:
        plot = pg.PlotWidget(viewBox=YAxisScaleViewBox())
        indicator_plot = None
        window.setCentralWidget(plot)


    plot.showGrid(x=False, y=True, alpha=0.3)
    plot.addLegend(offset=(10, 10))
    _configure_plot_interaction(plot, lock_y_autorange=False)

    # Add candlesticks


    item = CandlestickItem(data)
    plot.addItem(item)

    # Plot moving averages
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    for i, period in enumerate(range_list):
        colname = f'MA{period}'
        avg_colname = f'MA{period}AVG'
        muot_col = f"MA{period}Div"


        if colname in hist.columns:
            ys = hist[colname].astype(float).to_numpy()
            plot.plot(x, ys, pen=pg.mkPen(colors[i % len(colors)], width=1.2))

        if avg_colname in hist.columns:
            ys = hist[avg_colname].astype(float).to_numpy()
            plot.plot(x, ys, pen=pg.mkPen(colors[i % len(colors)], width=2.4))

        for col in hist.columns:
            if muot_col in col:
                ys = hist[col].astype(float).to_numpy()
                plot.plot(x, ys, pen=pg.mkPen(colors[i % len(colors)], width=1.2))
                break

    # Bollinger bands
    if isinstance(BB, int):
        upper_col = f'Upper{BB}'
        lower_col = f'Lower{BB}'
        if upper_col in hist.columns and lower_col in hist.columns:
            plot.plot(x, hist[upper_col].astype(float).to_numpy(), pen=pg.mkPen('#888888', width=1))
            plot.plot(x, hist[lower_col].astype(float).to_numpy(), pen=pg.mkPen('#888888', width=1))

    def _add_cloud_segment(start_index, end_index, is_bullish_cloud, senkou_a_values, senkou_b_values):
        if end_index - start_index < 1:
            return
        segment_x = x[start_index:end_index + 1]
        segment_a = senkou_a_values[start_index:end_index + 1]
        segment_b = senkou_b_values[start_index:end_index + 1]
        upper_line = pg.PlotDataItem(segment_x, segment_a, pen=pg.mkPen(None))
        lower_line = pg.PlotDataItem(segment_x, segment_b, pen=pg.mkPen(None))
        cloud_brush = pg.mkBrush(46, 204, 113, 45) if is_bullish_cloud else pg.mkBrush(231, 76, 60, 45)
        cloud_fill = pg.FillBetweenItem(upper_line, lower_line, brush=cloud_brush)
        cloud_fill.setZValue(-5)
        upper_line.setZValue(-6)
        lower_line.setZValue(-6)
        plot.addItem(upper_line)
        plot.addItem(lower_line)
        plot.addItem(cloud_fill)

    if SENKOUA in hist.columns and SENKUB in hist.columns:
        senkou_a = pd.to_numeric(hist[SENKOUA], errors='coerce').to_numpy(dtype=float)
        senkou_b = pd.to_numeric(hist[SENKUB], errors='coerce').to_numpy(dtype=float)
        valid_mask = ~np.isnan(senkou_a) & ~np.isnan(senkou_b)
        cloud_direction = senkou_a >= senkou_b
        segment_start = None
        previous_direction = None
        for index in range(len(hist)):
            if not valid_mask[index]:
                if segment_start is not None:
                    _add_cloud_segment(segment_start, index - 1, previous_direction, senkou_a, senkou_b)
                segment_start = None
                previous_direction = None
                continue

            current_direction = bool(cloud_direction[index])
            if segment_start is None:
                segment_start = index
                previous_direction = current_direction
                continue

            if current_direction != previous_direction:
                _add_cloud_segment(segment_start, index, previous_direction, senkou_a, senkou_b)
                segment_start = index
                previous_direction = current_direction

        if segment_start is not None:
            _add_cloud_segment(segment_start, len(hist) - 1, previous_direction, senkou_a, senkou_b)


    price_indicator_colors = ['#f39c12', '#3498db', '#9b59b6', '#16a085', '#ffffff', '#e67e22']
    for index, indicator_name in enumerate(price_indicator_lines):
        if indicator_name not in hist.columns:
            continue
        indicator_values = pd.to_numeric(hist[indicator_name], errors='coerce').to_numpy(dtype=float)
        indicator_color = price_indicator_colors[index % len(price_indicator_colors)]
        if indicator_name == 'ParabolicSAR':
            scatter = pg.ScatterPlotItem(x, indicator_values, symbol='o', brush=pg.mkBrush(indicator_color), size=4)
            plot.addItem(scatter)
        else:
            plot.plot(x, indicator_values, pen=pg.mkPen(indicator_color, width=1.2), name=indicator_name)

    def _resolve_deal_end_index(result, start_index):
        """Resolve the x-position where a deal should stop being drawn."""
        if isinstance(result, dict):
            for key in ('end_index', 'end', 'close_index'):
                value = result.get(key)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        pass

            end_date = result.get('end_date')
            if end_date is not None and DATE in hist.columns:
                hist_dates = hist[DATE].astype(str).to_numpy()
                matches = np.where(hist_dates == str(end_date))[0]
                if len(matches) > 0:
                    return int(matches[0])

        return int(start_index)

    # Plot deals with entry, SL, TP lines bounded by each deal's real end index.
    if deals and results:
        chart_last_index = len(hist) - 1
        for idx, deal in enumerate(deals):
            if idx >= len(results) or not isinstance(deal, dict):
                break

            if deal.get('res')=='No res':
                continue

            result = results[idx]
            start_idx = deal.get('start_index')
            entry = deal.get('entry')
            sl = deal.get('sl')
            tp = deal.get('tp')

            if start_idx is None or entry is None:
                continue

            try:
                start_idx = int(start_idx)
                entry_y = float(entry)
            except (TypeError, ValueError):
                continue

            end_idx = _resolve_deal_end_index(result, start_idx)
            start_idx = max(0, min(start_idx, chart_last_index))
            end_idx = max(start_idx, min(end_idx, chart_last_index))

            start_x = float(start_idx)
            end_x = float(end_idx)
            line_xs = np.array([start_x, end_x])

            # Plot entry line (white color, dashed)
            plot.plot(line_xs, np.array([entry_y, entry_y]), pen=pg.mkPen('#ffffff', width=2, style=2))

            # Plot stop loss line (red dashed)
            if sl is not None:
                try:
                    sl_y = float(sl)
                    plot.plot(line_xs, np.array([sl_y, sl_y]), pen=pg.mkPen('#d62728', width=1.5, style=2))
                except (TypeError, ValueError):
                    pass

            # Plot take profit line (green dashed)
            if tp is not None:
                try:
                    tp_y = float(tp)
                    plot.plot(line_xs, np.array([tp_y, tp_y]), pen=pg.mkPen('#2ca02c', width=1.5, style=2))
                except (TypeError, ValueError):
                    pass

            # Plot entry point as scatter (blue circle)
            scatter = pg.ScatterPlotItem([start_x], [entry_y], symbol='o', brush=pg.mkBrush('#0000ff'), size=8)
            plot.addItem(scatter)

    indicator_colors = ['#f1c40f', '#00bcd4', '#e91e63', '#9c27b0', '#4caf50']
    has_bounded_indicator_line = False
    avg_line_style = QtCore.Qt.DashLine
    avg_line_alpha = 180
    avg_line_suffix = f' AVG({indicator_avg_period})'
    if indicator_plot is not None:
        for index, indicator_name in enumerate(indicator_lines):
            if indicator_name not in hist.columns:
                continue
            indicator_values = hist[indicator_name].astype(float)
            indicator_color = indicator_colors[index % len(indicator_colors)]
            indicator_plot.plot(x, indicator_values.to_numpy(), pen=pg.mkPen(indicator_color, width=1.4), name=indicator_name)
            indicator_upper_name = indicator_name.upper()
            if 'MFI' in indicator_upper_name or 'RSI' in indicator_upper_name:
                avg_values = indicator_values.rolling(window=indicator_avg_period, min_periods=1).mean().to_numpy()
                avg_pen = pg.mkPen(indicator_color, width=1.2, style=avg_line_style)
                avg_pen.setColor(QtGui.QColor('lightblue'))
                avg_pen.color().setAlpha(avg_line_alpha)
                indicator_plot.plot(x, avg_values, pen=avg_pen, name=f'{indicator_name}{avg_line_suffix}')
                has_bounded_indicator_line = True

        if indicator_lines:
            lower_line = pg.InfiniteLine(pos=float(mfivas[0]), angle=0, pen=pg.mkPen(INDICATOR_GUIDE_COLOR, width=1, style=DAY_SEPARATOR_STYLE), movable=False)
            upper_line = pg.InfiniteLine(pos=float(mfivas[1]), angle=0, pen=pg.mkPen(INDICATOR_GUIDE_COLOR, width=1, style=DAY_SEPARATOR_STYLE), movable=False)
            lower_line.setOpacity(INDICATOR_GUIDE_ALPHA)
            upper_line.setOpacity(INDICATOR_GUIDE_ALPHA)
            indicator_plot.addItem(lower_line)
            indicator_plot.addItem(upper_line)

        indicator_plot.setYRange(INDICATOR_Y_MIN, INDICATOR_Y_MAX, padding=0)
        _configure_plot_interaction(indicator_plot)




    # Draw vertical separators between trading days
    separator_pen = pg.mkPen(DAY_SEPARATOR_COLOR, width=DAY_SEPARATOR_WIDTH, style=DAY_SEPARATOR_STYLE)
    separator_pen.setCosmetic(True)
    for boundary_index in _get_day_boundary_indices(hist):
        separator_line = pg.InfiniteLine(
            pos=float(boundary_index),
            angle=90,
            pen=separator_pen,
            movable=False
        )
        separator_line.setZValue(-10)
        separator_line.setOpacity(DAY_SEPARATOR_ALPHA)
        plot.addItem(separator_line)
        if indicator_plot is not None:
            indicator_separator_line = pg.InfiniteLine(
                pos=float(boundary_index),
                angle=90,
                pen=separator_pen,
                movable=False
            )
            indicator_separator_line.setZValue(-10)
            indicator_separator_line.setOpacity(DAY_SEPARATOR_ALPHA)
            indicator_plot.addItem(indicator_separator_line)

    # Configure X-axis to show date labels every N ticks

    axis = plot.getAxis('bottom')
    dates = hist[DATE].astype(str).to_list()
    n = max(1, int(len(dates) / 10))
    ticks = [(int(i), dates[i]) for i in range(0, len(dates), n)]
    axis.setTicks([ticks])

    if len(x) > 0:
        plot.setXRange(float(x[0]), float(x[-1]), padding=0)
        finite_lows = l[np.isfinite(l)]
        finite_highs = h[np.isfinite(h)]
        if len(finite_lows) > 0 and len(finite_highs) > 0:
            y_min = float(np.min(finite_lows))
            y_max = float(np.max(finite_highs))
            y_padding = max((y_max - y_min) * 0.02, 1e-6)
            plot.setYRange(y_min - y_padding, y_max + y_padding, padding=0)
        _configure_plot_interaction(plot)
    if indicator_plot is not None:
        indicator_plot.setYRange(INDICATOR_Y_MIN, INDICATOR_Y_MAX, padding=0)
        _configure_plot_interaction(indicator_plot)


    window.show()
    if app_owner:
        app.exec_()


