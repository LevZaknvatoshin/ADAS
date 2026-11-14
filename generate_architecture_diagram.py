import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set font family
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

# Canvas setup - high resolution widescreen
fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
ax = fig.add_subplot(111)
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)
ax.axis('off')

# Color palette (Modern Dark / Slate Tech Theme)
BG_COLOR = "#0F172A"       # Slate 900
PANEL_BG = "#1E293B"       # Slate 800
PANEL_BORDER = "#334155"   # Slate 700
HEADER_BG = "#0284C7"      # Sky 600

# Group Colors
C_INPUT = {"bg": "#1E293B", "box": "#0F172A", "border": "#38BDF8", "title": "#38BDF8"}       # Sky
C_PERCEPTION = {"bg": "#1E293B", "box": "#1E1E38", "border": "#818CF8", "title": "#818CF8"}  # Indigo
C_TWIN = {"bg": "#1E293B", "box": "#172554", "border": "#60A5FA", "title": "#60A5FA"}        # Blue
C_SIM = {"bg": "#1E293B", "box": "#064E3B", "border": "#34D399", "title": "#34D399"}         # Emerald
C_EVAL = {"bg": "#1E293B", "box": "#451A03", "border": "#FBBF24", "title": "#FBBF24"}        # Amber

TEXT_MAIN = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
TEXT_ACCENT = "#38BDF8"

fig.patch.set_facecolor(BG_COLOR)

# Title and Header Banner
title_box = FancyBboxPatch((40, 980), 1840, 80, boxstyle="round,pad=0,rounding_size=12",
                           facecolor="#1E293B", edgecolor="#334155", linewidth=1.5)
ax.add_patch(title_box)

ax.text(70, 1025, "АРХИТЕКТУРА СИСТЕМЫ МОДЕЛИРОВАНИЯ И ВАЛИДАЦИИ ДОРОЖНЫХ ИНЦИДЕНТОВ",
        fontsize=16, fontweight='bold', color="#FFFFFF", va='center')
ax.text(70, 998, "Проект: «Валидация инцидентов» (P119) | Заказчик: ФАУ «РОСДОРНИИ» | Исполнитель: ВИШ РУТ (МИИТ)",
        fontsize=11, color="#38BDF8", va='center')

# Function to draw container card
def draw_container(x, y, w, h, title, subtitle, color_cfg):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                         facecolor=color_cfg["bg"], edgecolor=color_cfg["border"], linewidth=1.8)
    ax.add_patch(box)
    
    # Header bar
    hbar = FancyBboxPatch((x, y + h - 42), w, 42, boxstyle="round,pad=0,rounding_size=12",
                          facecolor=color_cfg["border"], edgecolor="none", alpha=0.15)
    ax.add_patch(hbar)
    
    ax.text(x + 18, y + h - 22, title, fontsize=12, fontweight='bold', color=color_cfg["title"], va='center')
    if subtitle:
        ax.text(x + w - 18, y + h - 22, subtitle, fontsize=9.5, color=TEXT_MUTED, ha='right', va='center')

# Function to draw item inside container
def draw_block(x, y, w, h, title, desc_lines, tag=None, bg_override=None, border_color="#475569"):
    bg = bg_override if bg_override else PANEL_BG
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                         facecolor=bg, edgecolor=border_color, linewidth=1.2)
    ax.add_patch(box)
    
    # Title
    ax.text(x + 14, y + h - 18, title, fontsize=11, fontweight='bold', color=TEXT_MAIN, va='center')
    
    # Tag
    if tag:
        tag_w = len(tag) * 6.5 + 14
        tbox = FancyBboxPatch((x + w - tag_w - 10, y + h - 26), tag_w, 18,
                              boxstyle="round,pad=0,rounding_size=4", facecolor="#0284C7", edgecolor="none")
        ax.add_patch(tbox)
        ax.text(x + w - 10 - tag_w/2, y + h - 17, tag, fontsize=8.5, fontweight='bold', color="#FFFFFF", ha='center', va='center')
        
    # Text lines
    start_y = y + h - 38
    for i, line in enumerate(desc_lines):
        ax.text(x + 14, start_y - (i * 15.5), line, fontsize=9.2, color=TEXT_MUTED, va='center')

# Function to draw arrow
def draw_connection(x1, y1, x2, y2, label=None, label_side='top', color="#38BDF8", style='->', rad=0.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", mutation_scale=16,
                            connectionstyle=f"arc3,rad={rad}",
                            color=color, linewidth=2.0, linestyle='-')
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        offset_y = 10 if label_side == 'top' else -12
        ax.text(mx, my + offset_y, label, fontsize=9, fontweight='bold', color="#E2E8F0",
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_COLOR, edgecolor="#475569", alpha=0.9, lw=0.8))

# ==================== 1. ВХОДНЫЕ ДАННЫЕ ====================
draw_container(40, 110, 280, 840, "1. ВХОДНЫЕ ДАННЫЕ", "Источники", C_INPUT)

draw_block(58, 720, 244, 180, "Дорожная камера", [
    "• Стационарные камеры наблюдения",
    "• Разрешение: Full HD / 4K",
    "• Одиночные фото / видеокадры",
    "• Сложные условия (день/ночь)"
], tag="CCTV / RTSP", bg_override="#132034", border_color="#38BDF8")

draw_block(58, 480, 244, 210, "Параметры съемки", [
    "• Фокусное расстояние (мм)",
    "• Горизонтальный угол HFOV",
    "• Ориентация и ракурс камеры",
    "• Соотношение сторон (W / H)",
    "• Оптические константы"
], tag="Метаданные", bg_override="#132034", border_color="#38BDF8")

draw_block(58, 140, 244, 310, "Каталог ассетов", [
    "• Транспортные средства:",
    "  легковые, грузовые, автобусы",
    "• Пешеходы и спецтехника",
    "• Типовые препятствия:",
    "  конусы, барьеры, обломки",
    "• Текстуры дорожного покрытия",
    "  (сухой/мокрый асфальт, лед)"
], tag="3D Models", bg_override="#132034", border_color="#38BDF8")

# ==================== 2. МОДУЛЬ ВОСПРИЯТИЯ ====================
draw_container(360, 110, 360, 840, "2. МОДУЛЬ ВОСПРИЯТИЯ", "Perception Engine", C_PERCEPTION)

draw_block(380, 680, 320, 220, "Оценка монокулярной глубины", [
    "• Модель: Depth Anything V2 Small",
    "• Инференс: PyTorch / Hugging Face",
    "• Оптимизация под CPU и GPU (CUDA)",
    "• Бикубическая интерполяция сетки",
    "• Формирование карты z-координат"
], tag="AI Neural Net", bg_override="#1A1D36", border_color="#818CF8")

draw_block(380, 470, 320, 180, "Нормализация и фильтрация", [
    "• Перцентильное отсечение [2%, 98%]",
    "• Устранение артефактов неба и шумов",
    "• Масштабирование градиента глубины",
    "• Сохранение матрицы в depth.npy",
    "• Тепловая карта: depth_preview.png"
], tag="Data Cleaning", bg_override="#1A1D36", border_color="#818CF8")

draw_block(380, 260, 320, 180, "Сегментация полотна и полос", [
    "• Автоматическая маска полотна дороги",
    "• Трассировка линий разметки",
    "• Формирование векторного контура",
    "• Экспорт полигонов: road_polygon.json",
    "• Логическое деление проезжей части"
], tag="CV / Segmentation", bg_override="#1A1D36", border_color="#818CF8")

draw_block(380, 140, 320, 95, "Веб-инструмент (Annotator App)", [
    "• Ручная валидация масок полотна оператором",
    "• Локальный сервер на FastAPI/HTTP"
], tag="Web GUI", bg_override="#1A1D36", border_color="#818CF8")

# ==================== 3. 2.5D РЕКОНСТРУКЦИЯ СЦЕНЫ ====================
draw_container(760, 110, 370, 840, "3. 2.5D РЕКОНСТРУКЦИЯ СЦЕНЫ", "Scene Builder (Blender)", C_TWIN)

draw_block(780, 710, 330, 190, "Математическая обратная проекция", [
    "• Точная формула обратной перспективы:",
    "  X = (u - 0.5) · 2 · d · tan(HFOV/2)",
    "  Y = (0.5 - v) · 2 · d · tan(HFOV/2) · (H/W)",
    "  Z = -d",
    "• Преобразование 2D-пикселей в 3D-вершины"
], tag="Pin-Hole Math", bg_override="#0D1E3A", border_color="#60A5FA")

draw_block(780, 480, 330, 200, "Генерация 3D-рельефа (Mesh)", [
    "• Сглаженная триангулированная сетка",
    "• Параметрическая плотность (grid: 240+)",
    "• Деформация геометрии по карте глубины",
    "• Сглаживание нормалей рельефа полотна",
    "• Объект: RoadScene_2.5D_Reconstruction"
], tag="Parametric Mesh", bg_override="#0D1E3A", border_color="#60A5FA")

draw_block(780, 290, 330, 165, "Камерное текстурирование", [
    "• Наложение UV-проекции исходного фото",
    "• 100% оптическое соответствие кадру",
    "• Коррекция теней и бликов покрытия",
    "• Раздельные слои для разметки"
], tag="UV Project", bg_override="#0D1E3A", border_color="#60A5FA")

draw_block(780, 140, 330, 125, "Система виртуальных камер", [
    "• ReferenceCamera (50mm) — точка обзора CCTV",
    "• DriverViewCamera — уровень глаз водителя",
    "• Ортогональная камера для топ-плана (Birds-Eye)"
], tag="Camera Rig", bg_override="#0D1E3A", border_color="#60A5FA")

# ==================== 4. СИМУЛЯТОР ИНЦИДЕНТОВ ====================
draw_container(1170, 110, 360, 840, "4. МОДЕЛИРОВАНИЕ ИНЦИДЕНТОВ", "Blender ADAS Simulator", C_SIM)

draw_block(1190, 700, 320, 200, "Аддон Blender (cv_builder)", [
    "• Боковая панель управления (N-Panel)",
    "• Быстрое переключение режимов построения:",
    "  - RoadCam 2.5D (непрерывный меш)",
    "  - Сегментированный Twin (слои полотна)",
    "• Управление привязкой к рельефу дороги"
], tag="Blender 4.1 GUI", bg_override="#063228", border_color="#34D399")

draw_block(1190, 460, 320, 210, "Сценарный спавн объектов", [
    "• Параметрическое размещение препятствий",
    "• Имитация ДТП: столкновения, заносы",
    "• Пешеходы на проезжей части",
    "• Посторонние габаритные предметы",
    "• Физический контроллер коллизий"
], tag="Incident Engine", bg_override="#063228", border_color="#34D399")

draw_block(1190, 270, 320, 165, "Моделирование среды и погоды", [
    "• Суточный цикл освещения (день / сумерки / ночь)",
    "• Система частиц: дождь, снегопад",
    "• Объемный туман (Volumetric Fog)",
    "• Блики фар и мокрый асфальт"
], tag="Environment", bg_override="#063228", border_color="#34D399")

draw_block(1190, 140, 320, 105, "Мастер-проект сцены", [
    "• Полная сериализация состояния в .blend",
    "• Возможность 100% повтора эксперимента"
], tag="Project Blend", bg_override="#063228", border_color="#34D399")

# ==================== 5. ВАЛИДАЦИЯ И СИНТЕЗ DATASET ====================
draw_container(1570, 110, 310, 840, "5. ВАЛИДАЦИЯ И ВЫХОД", "GT & Validation", C_EVAL)

draw_block(1588, 710, 274, 190, "Пакетный рендеринг", [
    "• Движок Cycles (фотореализм)",
    "• Движок EEVEE (высокая скорость)",
    "• Фоновый headless-рендеринг",
    "• Генерация серий кадров инцидентов",
    "• Экспорт в 1080p / 4K"
], tag="Batch Render", bg_override="#2E1304", border_color="#FBBF24")

draw_block(1588, 480, 274, 205, "Авто-разметка Ground Truth", [
    "• Проекция 3D BBox объектов в 2D",
    "• Учет окклюзии (перекрытия)",
    "• Генерация меток YOLO (.txt)",
    "  <class> <x> <y> <w> <h>",
    "• Сегментационные маски инцидентов"
], tag="Auto Annotate", bg_override="#2E1304", border_color="#FBBF24")

draw_block(1588, 280, 274, 170, "Готовый датасет", [
    "• Папка /images (синтетические кадры)",
    "• Папка /labels (эталонная разметка)",
    "• Метаданные условий (погода, свет)",
    "• Сбалансированные редкие классы ДТП"
], tag="Synthetic Dataset", bg_override="#2E1304", border_color="#FBBF24")

draw_block(1588, 140, 274, 115, "Валидация CV-систем", [
    "• Тестирование детекторов заказчика",
    "• Расчет метрик точности mAP@50, F1",
    "• Анализ ложных срабатываний"
], tag="Metrics / mAP", bg_override="#2E1304", border_color="#FBBF24")

# ==================== ARROWS / DATA FLOWS ====================
# 1 -> 2
draw_connection(302, 810, 380, 810, "Кадр (RGB)")
draw_connection(302, 580, 780, 790, "HFOV, W/H, Focal", rad=-0.25)
draw_connection(302, 290, 1190, 560, "3D Модели (GLB)", rad=-0.3)

# 2 -> 3
draw_connection(700, 560, 780, 560, "depth.npy")
draw_connection(700, 350, 780, 360, "road_polygon.json")

# 3 -> 4
draw_connection(1110, 580, 1190, 580, "2.5D Mesh")

# 4 -> 5
draw_connection(1510, 800, 1588, 800, "Сцена")
draw_connection(1510, 570, 1588, 570, "BBox (3D coords)")

# Bottom feedback / export arrows
draw_connection(1588, 350, 1510, 200, "Верификация", rad=0.15, color="#FBBF24")

plt.tight_layout()

# Save paths
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_schema.png")
plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Schema image successfully generated at: {output_file}")
