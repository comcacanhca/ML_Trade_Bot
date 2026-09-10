import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.colors as colors
from data_handler.download_data_mt5 import *

figwidth = 14
figheight = 6.5

col1 = 'green'
col2 = 'red'
# colors = ['tomato','orange','gold','green','lightgreen','coral','skyblue','aliceblue','blueviolet','violet','purple','indigo','gray','teal','royalblue','navy','darkblue','black']
rainbow_colors_rgb = np.array([[1.0, 0.0, 0.0],  # Đỏ
                               [1.0, 0.65, 0.0],  # Cam
                               [1.0, 1.0, 0.0],  # Vàng
                               [0.0, 1.0, 0.0],  # Xanh lá
                               [0.0, 0.0, 1.0],  # Xanh dương
                               [0.5, 0.0, 1.0]])  # Tím


# Tạo colormap tùy chỉnh
def get_color(quan=20):
    my_cmap = colors.LinearSegmentedColormap.from_list('my_cmap', rainbow_colors_rgb, N=quan)

    # Tạo danh sách các giá trị từ 0 đến 1
    colors_values = np.linspace(0, 1, quan)

    # Lấy các màu tương ứng từ colormap
    return [colors.rgb2hex(my_cmap(color)) for color in colors_values]


def draw(X_axis, Yaxis, Y1axis=None):
    import seaborn as sns
    # Tạo dữ liệu mẫu

    # Tạo figure và các subplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    data1 = {'Buy': [i for i in range(len(Yaxis))], 'y': Yaxis}
    data2 = {'Sell': [i for i in range(len(Y1axis))], 'y1': Y1axis}
    # Vẽ biểu đồ đầu tiên
    sns.lineplot(data=data1, x='Buy', y='y', ax=axes[0], markers='o')

    # Vẽ biểu đồ thứ hai
    sns.lineplot(data=data2, x='Sell', y='y1', ax=axes[1], markers='o')

    # Hiển thị biểu đồ
    plt.show()


def draw_vung_gia(hist, pair='GBPUSD'):
    '''forx_data_minute = data_handler
    # Set the index to a datetime object
    hist.index = pd.to_datetime(hist.index)
    # Display the last five rows
    hist.tail()
    hist['dates'] = hist.index.strftime(
        '%Y-%m-%d %H:%M:%S')'''
    # Plot the series
    print("*************Draw chart******************")

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    up = hist[hist[CLOSE] >= hist[OPEN]]
    down = hist[hist[CLOSE] < hist[OPEN]]
    ax.vlines(x=hist[DATE], ymax=hist[HIGH], ymin=hist[LOW], linewidth=0)  # rau tren
    ax.bar(up[DATE], abs(up[CLOSE] - up[OPEN]), bottom=up[OPEN], color=col1)  # than nen
    ax.vlines(x=up[DATE], ymax=up[HIGH], ymin=up[LOW], color=col1)
    ax.bar(down[DATE], abs(down[CLOSE] - down[OPEN]), bottom=down[CLOSE], color=col2)
    ax.vlines(x=down[DATE], ymax=down[HIGH], ymin=down[LOW], color=col2)

    key_points_index_up, key_points_index_down, key_points_value_up, key_points_value_down, ob_points_index, ob_points_value = [], [], [], [], [], []
    if KEY in hist.columns and TREND in hist.columns:
        for i in range(len(hist[DATE])):
            if hist[KEY][i] != 0 and hist[TREND][i] == 'tang':
                key_points_index_up.append(hist[DATE][i])
                key_points_value_up.append(hist[KEY][i])
            elif hist[KEY][i] != 0 and hist[TREND][i] == 'giam':
                key_points_index_down.append(hist[DATE][i])
                key_points_value_down.append(hist[KEY][i])

        ax.plot(key_points_index_up, key_points_value_up, marker='o', color='#006400', linewidth=0)
        ax.plot(key_points_index_down, key_points_value_down, marker='o', color='darkred', linewidth=0)
    elif KEY in hist.columns:
        for i in range(len(hist[DATE])):
            if hist[KEY][i] != 0:
                key_points_index_up.append(hist[DATE][i])
                key_points_value_up.append(hist[KEY][i])
        ax.plot(key_points_index_up, key_points_value_up, marker='o', color='purple', linewidth=0)

    if 'Order Block' in hist.columns:
        for i in range(len(hist[DATE])):
            if hist['Order Block'][i] != 0:
                ob_points_index.append(hist[DATE][i])
                ob_points_value.append(hist['Order Block'][i])

        ax.plot(ob_points_index, ob_points_value, marker='o', color='blue', linewidth=0)

    '''ax.plot(hist['dates'], hist['Adj Close'])
    ax.plot(hist['dates'], hist['Open'])'''

    # Set title and axis label
    plt.title(str(pair), fontsize=16)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Price', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Set maximum number of tick locators
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xticks(rotation=20)
    pts = plt.ginput(2)
    print(pts)
    # Show the plot
    plt.show()
    return pts


# get_vung_gia(download_data(quanlity=300), pair='EURUSD')

def draw_MA(
        hist: pd.DataFrame,
        deals: pd.DataFrame = None,
        results: pd.DataFrame = None,
        range_list: list = [],
        colors: list = None,
        vectors: list = [],
        limitvec: int = 300,
        BB: int | bool = False,
        mfivas: tuple = (20, 80),
        indicator_lines: list | None = None,
        price_indicator_lines: list | None = None,
        indicator_avg_period: int = 15,
        interactive: bool = False,
        backend: str = 'pyqtgraph'
) -> None:


    """
    Draw price chart with multiple technical indicators.

    Args:
        hist: DataFrame containing price data_handler with columns [date, open, high, low, close]
        deals: Optional DataFrame containing trade deals
        results: Optional DataFrame containing results
        range_list: List of periods for Moving Averages
        colors: Color list for different indicators
        vectors: List of vectors to plot
        limitvec: Vector limit
        BB: Whether to plot Bollinger Bands
        mfivas: Tuple containing MFI values range (low, high)
        indicator_lines: Optional list of oscillator column names to plot in the lower panel (e.g. ['RSI14', 'MFI'])
        price_indicator_lines: Optional list of price-level indicator columns to plot on the main chart
        indicator_avg_period: Rolling average period for RSI/MFI lines

    """

    def _resolve_indicator_columns(hist, indicator_lines):
        if indicator_lines is None:
            return [col for col in hist.columns if 'RSI' in col or 'MFI' in col or 'WTO' in col]

        resolved_columns = []
        for indicator_name in indicator_lines:
            if indicator_name in hist.columns:
                resolved_columns.append(indicator_name)
        return resolved_columns

    indicator_columns = _resolve_indicator_columns(hist, indicator_lines)
    price_indicator_columns = [col for col in (price_indicator_lines or []) if col in hist.columns]

    def _setup_subplots():
        has_indicators = len(indicator_columns) > 0 or len(vectors) > 0


        if has_indicators:
            fig, (ax1, ax2) = plt.subplots(figsize=(figwidth, figheight), nrows=2, ncols=1, sharex=True)
            return fig, ax1, ax2
        fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
        return fig, ax1, None

    def _plot_candlesticks(ax, hist):
        up = hist[hist[CLOSE] >= hist[OPEN]]
        down = hist[hist[CLOSE] < hist[OPEN]]

        # Plot candlesticks
        ax.vlines(x=hist[DATE], ymax=hist[HIGH], ymin=hist[LOW], linewidth=0)
        ax.bar(up[DATE], abs(up[CLOSE] - up[OPEN]), bottom=up[OPEN], color=col1)
        ax.vlines(x=up[DATE], ymax=up[HIGH], ymin=up[LOW], color=col1)
        ax.bar(down[DATE], abs(down[CLOSE] - down[OPEN]), bottom=down[CLOSE], color=col2)
        ax.vlines(x=down[DATE], ymax=down[HIGH], ymin=down[LOW], color=col2)

    def _plot_moving_averages(ax, hist, range_list):
        ma_colors = get_color(len(range_list))
        for i, period in enumerate(range_list):
            ma_col = f'MA{period}'
            muot_col = f"{MA}{period}Div"

            if ma_col in hist.columns:
                start_idx = period + 1
                ax.plot(hist[DATE][start_idx:], hist[ma_col][start_idx:],
                        color=ma_colors[i], linewidth=1, linestyle='solid')
            for col in hist.columns:
                if muot_col in col:
                    start_idx = period + 1
                    ax.plot(hist[DATE][start_idx:], hist[muot_col][start_idx:],
                            color=ma_colors[i], linewidth=1, linestyle='solid')
                    break

    def _plot_key_levels(ax, hist):
        if KEY not in hist.columns:
            return

        key_points = {
            'up': {'dates': [], 'values': []},
            'down': {'dates': [], 'values': []}
        }

        for i, row in hist.iterrows():
            if row[KEY] == 0:
                continue

            trend_type = 'up' if TREND in hist.columns and row[TREND] == 'tang' else 'down'
            key_points[trend_type]['dates'].append(row[DATE])
            key_points[trend_type]['values'].append(row[KEY])

        if key_points['up']['dates']:
            ax.plot(key_points['up']['dates'], key_points['up']['values'],
                    marker='o', color='#006400', linewidth=0)
        if key_points['down']['dates']:
            ax.plot(key_points['down']['dates'], key_points['down']['values'],
                    marker='o', color='darkred', linewidth=0)

    def _plot_peaks_troughs(ax, hist, range_list):
        for period in range_list:
            peak_col = f'dinh{period}'
            trough_col = f'day{period}'

            if peak_col in hist.columns:
                ax.plot(hist[DATE], hist[peak_col], marker='o', color='darkred', linewidth=0)
                ax.plot(hist[DATE], hist[trough_col], marker='o', color='darkgreen', linewidth=0)

    def _plot_ma_crossovers(ax, hist, range_list):
        for period in range_list:
            cut_col = f'CUTMA{period}'
            ma_col = f'MA{period}'

            if cut_col not in hist.columns:
                continue

            cut_points = hist[hist[cut_col] == 'cut']
            if not cut_points.empty:
                ax.plot(cut_points[DATE], cut_points[ma_col],
                        marker='o', color=colors[range_list.index(period)], linewidth=0)

    def _plot_h1_levels(ax, hist):
        if 'H1' not in hist.columns:
            return

        h1_points = hist[hist[DATE].str.contains('00:00')]
        if not h1_points.empty:
            ax.plot(h1_points[DATE], h1_points['H1'],
                    color='black', linewidth=2, linestyle='solid')

    def _plot_deals_column(ax, hist):
        """Plot deals from the DEALS column in the histogram data_handler"""
        if "DEALS" not in hist.columns:
            return

        deals_data = hist[hist['DEALS'].notna()]
        if not deals_data.empty:
            ax.plot(deals_data[DATE], deals_data['DEALS'],
                    marker='o', color='blue', linewidth=0)

    def _plot_deals_data(ax, hist, deals, results):
        """Plot deals from the separate deals and results data_handler"""
        if deals is None:
            return

        for index, deal in enumerate(deals):
            start_index = deal['start_index']
            entry = deal['entry']
            stop_loss = deal['sl']
            best_take_profit = deal['tp']

            if results[index]['res'] != 'No res':
                entry_date = results[index]['entry_date']
            else:
                entry_date = hist[DATE][start_index] if start_index < len(hist) else hist[DATE][len(hist) - 1]

            end_date = results[index]['end_date']

            # Plot entry line
            ax.plot([entry_date, end_date], [entry, entry],
                    linewidth=1.5, alpha=1, linestyle='-', color='black')

            # Plot stop loss line
            ax.plot([hist[DATE][start_index - 1], end_date], [stop_loss, stop_loss],
                    linewidth=1, alpha=0.5, linestyle='solid', color='red')

            # Plot take profit line
            ax.plot([hist[DATE][start_index - 1], end_date], [best_take_profit, best_take_profit],
                    linewidth=1, alpha=0.5, linestyle='solid', color='purple')

    def _plot_bollinger_bands(ax, hist, BB):
        """Plot Bollinger Bands if enabled"""
        if not isinstance(BB, int):
            return
        bb_columns = [col for col in hist.columns if any(i == col for i in [f'Upper{BB}', f'Lower{BB}'])]

        for col in bb_columns:
            ax.plot(hist[DATE], hist[col], color='gray', linewidth=1, linestyle='solid')

    def _plot_ichimoku_cloud(ax, hist):
        if SENKOUA not in hist.columns or SENKUB not in hist.columns:
            return

        senkou_a = pd.to_numeric(hist[SENKOUA], errors='coerce').reset_index(drop=True)
        senkou_b = pd.to_numeric(hist[SENKUB], errors='coerce').reset_index(drop=True)
        date_values = hist[DATE].reset_index(drop=True)
        valid_mask = senkou_a.notna() & senkou_b.notna()
        if not valid_mask.any():
            return

        cloud_direction = senkou_a >= senkou_b
        segment_start = None
        previous_direction = None
        for index in range(len(hist)):
            if not valid_mask.iloc[index]:
                if segment_start is not None and index - segment_start > 1:
                    segment_direction = previous_direction
                    segment_slice = slice(segment_start, index)
                    cloud_color = 'mediumseagreen' if segment_direction else 'lightcoral'
                    ax.fill_between(date_values.iloc[segment_slice], senkou_a.iloc[segment_slice], senkou_b.iloc[segment_slice], color=cloud_color, alpha=0.18)
                segment_start = None
                previous_direction = None
                continue

            current_direction = bool(cloud_direction.iloc[index])
            if segment_start is None:
                segment_start = index
                previous_direction = current_direction
                continue

            if current_direction != previous_direction:
                segment_slice = slice(segment_start, index + 1)
                cloud_color = 'mediumseagreen' if previous_direction else 'lightcoral'
                ax.fill_between(date_values.iloc[segment_slice], senkou_a.iloc[segment_slice], senkou_b.iloc[segment_slice], color=cloud_color, alpha=0.18)
                segment_start = index
                previous_direction = current_direction

        if segment_start is not None and len(hist) - segment_start > 1:
            segment_slice = slice(segment_start, len(hist))
            cloud_color = 'mediumseagreen' if previous_direction else 'lightcoral'
            ax.fill_between(date_values.iloc[segment_slice], senkou_a.iloc[segment_slice], senkou_b.iloc[segment_slice], color=cloud_color, alpha=0.18)


    def _plot_price_indicator_lines(ax, hist, price_indicator_columns):
        if not price_indicator_columns:
            return

        line_colors = get_color(len(price_indicator_columns))
        for index, column in enumerate(price_indicator_columns):
            line_style = 'None' if column == 'ParabolicSAR' else 'solid'
            marker = '.' if column == 'ParabolicSAR' else None
            ax.plot(
                hist[DATE],
                hist[column],
                color=line_colors[index],
                linewidth=1,
                linestyle=line_style,
                marker=marker,
                markersize=2,
                label=column,
            )
        ax.legend(loc='upper left', fontsize=9)

    def _plot_rsi_vectors(ax2, hist, vectors, limitvec, indicator_columns, mfivas, indicator_avg_period):
        if ax2 is None:
            return

        indicator_colors = get_color(max(len(indicator_columns), 1))
        has_mfi_line = False
        has_rsi_line = False
        avg_line_suffix = f' AVG({indicator_avg_period})'
        avg_line_style = '--'
        avg_line_alpha = 0.85

        if indicator_columns:
            for index, indicator_column in enumerate(indicator_columns):
                indicator_series = hist[indicator_column]
                indicator_color = indicator_colors[index % len(indicator_colors)]
                ax2.plot(hist[DATE], indicator_series, color=indicator_color, label=indicator_column)

                indicator_upper_name = indicator_column.upper()
                if 'MFI' in indicator_upper_name or 'RSI' in indicator_upper_name:
                    avg_series = indicator_series.rolling(window=indicator_avg_period, min_periods=1).mean()
                    ax2.plot(
                        hist[DATE],
                        avg_series,
                        color=indicator_color,
                        linestyle=avg_line_style,
                        alpha=avg_line_alpha,
                        label=f'{indicator_column}{avg_line_suffix}'
                    )

                if 'MFI' in indicator_upper_name:
                    has_mfi_line = True
                if 'RSI' in indicator_upper_name:
                    has_rsi_line = True

        if indicator_columns:
            ax2.axhline(y=mfivas[0], color='r', linestyle='-')
            ax2.axhline(y=mfivas[1], color='r', linestyle='-')

        if vectors and len(vectors) > 0:
            for i in range(min(len(vectors), limitvec)):
                ax2.plot(hist[DATE], vectors[i])

        if indicator_columns:
            ax2.legend(loc='upper left', fontsize=9)



    def _plot_day_separators(ax, hist, extra_axes=None):
        if DATE not in hist.columns or hist.empty:
            return

        day_separator_color = '#808080'
        day_separator_alpha = 0.35
        day_separator_width = 0.8
        day_separator_style = '--'

        date_values = pd.to_datetime(hist[DATE], errors='coerce')
        day_boundaries = []
        for i in range(1, len(date_values)):
            previous_date = date_values.iloc[i - 1]
            current_date = date_values.iloc[i]
            if pd.isna(previous_date) or pd.isna(current_date):
                continue
            if current_date.date() != previous_date.date():
                day_boundaries.append(hist[DATE].iloc[i])

        target_axes = [ax]
        if extra_axes is not None:
            target_axes.extend([extra_ax for extra_ax in extra_axes if extra_ax is not None])

        for boundary in day_boundaries:
            for target_ax in target_axes:
                target_ax.axvline(
                    x=boundary,
                    color=day_separator_color,
                    linewidth=day_separator_width,
                    linestyle=day_separator_style,
                    alpha=day_separator_alpha
                )

    def _setup_chart_style(ax):

        """Setup the chart style and labels"""
        plt.title("Chart", fontsize=16)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.xticks(fontsize=15, rotation=20)
        plt.yticks(fontsize=15)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.grid(axis='x', visible=False)

    print('--------Draw chart--------')
    # Main execution

    # If interactive rendering requested, delegate to pyqtgraph renderer (fast native)

    if interactive:
        if backend == 'pyqtgraph':
            try:
                # Import here to keep optional dependency
                from plot_chart.renderer_pyqtgraph import show_interactive_chart
                show_interactive_chart(
                    hist=hist,
                    range_list=range_list,
                    deals=deals,
                    results=results,
                    BB=BB,
                    indicator_lines=indicator_columns,
                    price_indicator_lines=price_indicator_columns,
                    mfivas=mfivas,
                    indicator_avg_period=indicator_avg_period
                )


                return
            except Exception as e:
                # Fallback to matplotlib rendering if pyqtgraph is unavailable
                print(f'Interactive renderer (pyqtgraph) failed: {e}. Falling back to matplotlib.')
        else:
            print(f'Interactive backend "{backend}" not supported. Falling back to matplotlib.')

    fig, ax1, ax2 = _setup_subplots()

    _plot_candlesticks(ax1, hist)
    _plot_moving_averages(ax1, hist, range_list)
    #_plot_key_levels(ax1, hist)
    _plot_peaks_troughs(ax1, hist, range_list)
    _plot_ma_crossovers(ax1, hist, range_list)
    _plot_h1_levels(ax1, hist)
    _plot_deals_column(ax1, hist)
    _plot_deals_data(ax1, hist, deals, results)
    _plot_bollinger_bands(ax1, hist, BB)
    _plot_ichimoku_cloud(ax1, hist)
    _plot_price_indicator_lines(ax1, hist, price_indicator_columns)
    _plot_rsi_vectors(ax2, hist, vectors, limitvec, indicator_columns, mfivas, indicator_avg_period)
    _plot_day_separators(ax1, hist, extra_axes=[ax2])



    # Adjust subplot positions if RSI or vectors are present

    if ax2 is not None:
        ax1.set_position([0.125, 0.30, 0.775, 0.56])
        ax2.set_position([0.125, 0.10, 0.775, 0.14])
        plt.setp(ax1.get_xticklabels(), visible=False)

    _setup_chart_style(ax1)

    plt.show()


def draw_lan_can(
        df: pd.DataFrame,
        position: int,
        dis: list = [],
        deal: dict = None,
        mfivas: tuple = (20, 80),
        area: list = None,
        save: bool = False
) -> None:
    """
    Draw candlestick chart with technical indicators for a specific position and range.

    Args:
        df: DataFrame containing price data_handler
        position: Center position for the chart window
        dis: List containing [left_distance, right_distance] from position
        deal: Dictionary or list containing deal information
        mfivas: Tuple containing MFI values range (low, high)
        area: List containing [lower_bound, upper_bound] for area highlighting
        save: Boolean to save the chart instead of displaying
    """

    def _prepare_data(df, position, dis):
        """Prepare and validate data_handler window"""
        dis[0] = min(position, dis[0])
        dis[1] = min(len(df) - position - 1, dis[1])

        hist = df[(position - dis[0]):(position + dis[1])].copy()
        return hist.reset_index(drop=True), dis[0], dis[1]

    def _setup_subplots(hist):
        """Setup subplots based on indicators present"""
        has_indicators = any('MFI' in col or 'WTO' in col or 'tick_volume' in col for col in hist.columns)

        if has_indicators:
            fig, (ax1, ax2) = plt.subplots(figsize=(figwidth, figheight), nrows=2, ncols=1, sharex=True)
            return fig, ax1, ax2

        fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
        return fig, ax1, None

    def _plot_candlesticks(ax, hist, pre):
        """Plot candlestick chart"""
        up = hist[hist[CLOSE] >= hist[OPEN]]
        down = hist[hist[CLOSE] < hist[OPEN]]

        ax.vlines(x=hist[DATE], ymax=hist[HIGH], ymin=hist[LOW], linewidth=0)
        ax.bar(up[DATE], abs(up[CLOSE] - up[OPEN]), bottom=up[OPEN], color=col1)
        ax.vlines(x=up[DATE], ymax=up[HIGH], ymin=up[LOW], color=col1)
        ax.bar(down[DATE], abs(down[CLOSE] - down[OPEN]), bottom=down[CLOSE], color=col2)
        ax.vlines(x=down[DATE], ymax=down[HIGH], ymin=down[LOW], color=col2)

        # Plot midpoint marker
        midpoint = (hist[OPEN][pre] + hist[CLOSE][pre]) / 2
        ax.plot(hist[DATE][pre], midpoint, color='blue', marker='o')

    def _plot_area_bounds(ax, hist, area):
        """Plot area bounds if specified"""
        if area is None:
            return

        x_range = [hist[DATE].iloc[0], hist[DATE].iloc[-1]]
        for bound in area:
            ax.plot(x_range, [bound, bound], color='brown', linewidth=0.5, linestyle='--')

    def _plot_moving_averages(ax, hist):
        """Plot moving averages"""
        ma_cols = [col for col in hist.columns if 'MA' in col and 'Del' not in col]
        if not ma_cols:
            return

        colors = get_color(len(ma_cols))
        for i, col in enumerate(ma_cols):
            ax.plot(hist[DATE], hist[col], color=colors[i], linewidth=1, linestyle='solid')

    def _plot_ichimoku(ax, hist):
        """Plot Ichimoku cloud indicators"""
        ichi_colors = {TENKAN: 'red', KIJUN: 'blue', SENKOUA: 'purple',
                       SENKUB: 'pink', CHIKOU: 'green'}

        for col in hist.columns:
            if col in ichi_colors:
                ax.plot(hist[DATE], hist[col], color=ichi_colors[col],
                        linewidth=1, linestyle='--')

        if SENKUB in hist.columns:
            ax.fill_between(hist[DATE], hist[SENKOUA], hist[SENKUB], alpha=0.2)

    def _plot_bands(ax, hist):
        """Plot upper and lower bands"""
        band_cols = [col for col in hist.columns
                     if ('Upper' in col or 'Lower' in col) and 'DelMAs' not in col]

        for col in band_cols:
            ax.plot(hist[DATE], hist[col], color='gray', linewidth=1, linestyle='solid')

    def _plot_zigzag(ax, hist):
        """Plot ZigZag indicator"""
        if 'ZigZag' not in hist.columns:
            return

        # Lấy các điểm ZigZag không phải None
        zigzag_points = hist[hist['ZigZag'].notna()]

        if len(zigzag_points) > 0:
            # Vẽ đường nối các điểm ZigZag
            ax.plot(zigzag_points[DATE], zigzag_points['ZigZag'],
                    color='blue', linewidth=2, linestyle='-', alpha=0.7, label='ZigZag')

            # Đánh dấu các điểm đỉnh và đáy nếu có cột ZigZag_Type
            if 'ZigZag_Type' in hist.columns:
                highs = zigzag_points[zigzag_points['ZigZag_Type'] == 'high']
                lows = zigzag_points[zigzag_points['ZigZag_Type'] == 'low']

                if len(highs) > 0:
                    ax.scatter(highs[DATE], highs['ZigZag'],
                              color='red', marker='v', s=100, zorder=5, label='ZigZag High')

                if len(lows) > 0:
                    ax.scatter(lows[DATE], lows['ZigZag'],
                              color='green', marker='^', s=100, zorder=5, label='ZigZag Low')

    def _plot_deals(ax, hist, deal, pre, next_):
        """Plot deal information"""
        if deal is None:
            return

        if isinstance(deal, list):
            _plot_multiple_deals(ax, hist, deal, pre, next_)
        else:
            _plot_single_deal(ax, hist, deal, pre, next_)

    def _plot_single_deal(ax, hist, deal, pre, next_):
        """Plot single deal information"""
        entry = deal['entry']
        stop_loss = deal['sl']
        take_profit = deal['tp']

        date_range = [hist[DATE][pre], hist[DATE][pre + next_ - 1]]

        ax.plot(date_range, [entry, entry], linewidth=1.5, alpha=0.3,
                linestyle='-', color='black')
        ax.plot(date_range, [stop_loss, stop_loss], linewidth=1, alpha=0.5,
                linestyle='solid', color='red')
        ax.plot(date_range, [take_profit, take_profit], linewidth=1, alpha=0.5,
                linestyle='solid', color='purple')

    def _plot_multiple_deals(ax, hist, deals, pre, next_):
        """Plot multiple deals information"""
        date_range = [hist[DATE][pre], hist[DATE][pre + next_ - 1]]

        for i, deal in enumerate(deals):
            if i == 0:
                _plot_single_deal(ax, hist, deal, pre, next_)
            else:
                ax.plot(date_range, [deal['sl'], deal['sl']],
                        linewidth=2, alpha=0.5, linestyle='solid', color='black')

    def _plot_indicators(ax2, hist, mfivas):
        """Plot technical indicators"""
        if ax2 is None:
            return

        for col in hist.columns:
            if 'MFI' in col:
                ax2.plot(hist[DATE], hist[col])
                ax2.plot(hist[DATE], [mfivas[1]] * len(hist), color='gray', linestyle='--')
                ax2.plot(hist[DATE], [mfivas[0]] * len(hist), color='gray', linestyle='--')
            elif 'WTO' in col:
                ax2.plot(hist[DATE], hist[col])
                ax2.plot(hist[DATE], [0] * len(hist), color='gray', linestyle='--')
            elif VOLUME in col:
                try:
                    ax2.bar(hist[DATE], hist[VOLUME], color='gray', alpha=0.6, width=0.6, label=VOLUME)
                    ax2.legend(loc='upper right', fontsize='small')
                except Exception:
                    # fallback to line if bar fails for date type
                    ax2.plot(hist[DATE], hist[VOLUME], color='gray', alpha=0.6, label=VOLUME)

    def _setup_chart_style(ax, save):
        """Setup chart style and save if required"""
        if not save:
            plt.title("Chart", fontsize=16)
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Price', fontsize=15)
            plt.xticks(fontsize=15, rotation=20)
            plt.yticks(fontsize=15)
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            return

        plt.ioff()
        for spine in ax.spines.values():
            spine.set_visible(False)

        for spine_name in ['left', 'right', 'bottom', 'top']:
            ax.spines[spine_name].set_position(('data_handler', 0.1 if spine_name in ['left', 'bottom'] else 0.9))

        ax.set_xticks([])
        ax.set_yticks([])

    # Main execution
    hist, pre, next_ = _prepare_data(df, position, dis)
    fig, ax1, ax2 = _setup_subplots(hist)
    print(f'Plotting chart... {position}: {df[DATE][position]}')

    _plot_candlesticks(ax1, hist, pre)
    _plot_area_bounds(ax1, hist, area)
    _plot_moving_averages(ax1, hist)
    _plot_ichimoku(ax1, hist)
    _plot_bands(ax1, hist)
    _plot_zigzag(ax1, hist)  # Thêm vẽ ZigZag
    _plot_deals(ax1, hist, deal, pre, next_)
    _plot_indicators(ax2, hist, mfivas)

    if ax2 is not None:
        ax1.set_position([0.125, 0.30, 0.775, 0.56])
        ax2.set_position([0.125, 0.10, 0.775, 0.14])
        plt.setp(ax1.get_xticklabels(), visible=False)

    _setup_chart_style(ax1, save)


    if save:
        plt.savefig(save)
        plt.close(fig)
    else:
        plt.show()





import seaborn as sns


def draw_histGram(dat, cols, target, bin=5):
    print(f'BIN: {bin}')
    print('LENGTH:', len(dat.index))

    df = dat.groupby([pd.cut(dat[i], bins=bin) for index, i in enumerate(cols)])[target].value_counts().unstack()
    df = df.dropna()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    sns.heatmap(df.pivot(index=cols[0], columns=cols[1], values='rate'), cmap="Blues", annot=True, ax=ax1)
    sns.heatmap(df.pivot(index=cols[0], columns=cols[1], values='Win'), cmap="Greens", annot=True, ax=ax2)

    plt.title(f'{cols} and {target}')
    plt.show()
