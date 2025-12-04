import streamlit as st
import streamlit.image as st_image
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from fpdf import FPDF
import tempfile
import os

# =============================================================================
# 🚑 MONKEY PATCH: SOLUCIÓN DE EMERGENCIA PARA STREAMLIT CLOUD
# =============================================================================
# Este bloque arregla el error "AttributeError: image_to_url" si el servidor
# instala una versión demasiado nueva de Streamlit.
if not hasattr(st_image, "image_to_url"):
    try:
        # Intentamos buscar la función donde se movió en versiones nuevas (1.35+)
        from streamlit.elements.image import image_to_url
        st_image.image_to_url = image_to_url
    except ImportError:
        # Si falla, definimos una función dummy para que no rompa la app
        def image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=False):
            return "" # Esto podría fallar visualmente pero evita el crash inicial
        st_image.image_to_url = image_to_url
# =============================================================================

# -----------------------------------------------------------------------------
# 1. BASE DE DATOS DE ASPERSORES
# -----------------------------------------------------------------------------
CATALOGO_ASPERSORES = {
    "Pequeños (Jardines/Residencial)": {
        "Hunter MP Rotator 1000 (Eficiente)": {
            "radio": 4.0, "caudal_lpm": 2.3, "presion_psi": 40, "tipo": "Rotator"
        },
        "Rain Bird 1804 - Tobera 15VAN (Spray)": {
            "radio": 4.5, "caudal_lpm": 14.0, "presion_psi": 30, "tipo": "Difusor"
        },
        "Hunter PSU - Tobera 10A (Spray Corto)": {
            "radio": 3.0, "caudal_lpm": 9.5, "presion_psi": 30, "tipo": "Difusor"
        }
    },
    "Medianos (Parques/Comercial)": {
        "Rain Bird 3500 (Rotor Corto)": {
            "radio": 7.0, "caudal_lpm": 8.5, "presion_psi": 35, "tipo": "Rotor"
        },
        "Rain Bird 5004 - Boquilla 3.0 (Estándar)": {
            "radio": 11.0, "caudal_lpm": 13.5, "presion_psi": 45, "tipo": "Rotor"
        },
        "Hunter PGP Ultra - Boquilla 2.5": {
            "radio": 10.5, "caudal_lpm": 11.0, "presion_psi": 45, "tipo": "Rotor"
        }
    },
    "Grandes (Canchas/Deportivo)": {
        "Rain Bird 8005 (Largo Alcance)": {
            "radio": 18.0, "caudal_lpm": 55.0, "presion_psi": 60, "tipo": "Cañón"
        },
        "Hunter I-40 (Acero Inoxidable)": {
            "radio": 16.0, "caudal_lpm": 48.0, "presion_psi": 60, "tipo": "Cañón"
        }
    }
}

# -----------------------------------------------------------------------------
# 2. LÓGICA Y FUNCIONES
# -----------------------------------------------------------------------------

def auto_detectar_verde(pil_image, h_min, s_min, v_min, h_max, s_max):
    img_np = np.array(pil_image)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([h_min, s_min, v_min])
    upper_green = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_clean

def calcular_posiciones_aspersores(modo, datos_geo, radio_metros, escala_pix_por_metro):
    aspersores_validos = []
    R_pix = radio_metros * escala_pix_por_metro
    S_pix = R_pix 
    row_height = S_pix * (np.sqrt(3) / 2)

    if modo == 'manual':
        poly_shape = Polygon(datos_geo)
        min_x, min_y, max_x, max_y = poly_shape.bounds
    else:
        mask = datos_geo
        h, w = mask.shape
        min_x, min_y, max_x, max_y = 0, 0, w, h

    y = min_y
    row_count = 0
    while y <= max_y:
        x = min_x
        offset_x = (S_pix / 2) if row_count % 2 != 0 else 0
        while x <= max_x:
            cx, cy = int(x + offset_x), int(y)
            es_valido = False
            if modo == 'manual':
                if poly_shape.contains(Point(cx, cy)):
                    es_valido = True
            elif modo == 'auto':
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    if mask[cy, cx] == 255:
                        es_valido = True
            if es_valido:
                aspersores_validos.append((cx, cy))
            x += S_pix
        y += row_height
        row_count += 1
    return aspersores_validos, R_pix

def dibujar_resultado_opencv(pil_image, aspersores_coords, radio_pix):
    img_np = np.array(pil_image)
    r_int = int(round(radio_pix))
    for (cx, cy) in aspersores_coords:
        cv2.circle(img_np, (cx, cy), r_int, (0, 150, 255), 2)
        cv2.circle(img_np, (cx, cy), 3, (255, 0, 0), -1)
    return img_np

def calcular_hidraulica(n_aspersores, q_unit_lpm, diametro_mm, longitud_m, zonas=1):
    q_total_lpm = n_aspersores * q_unit_lpm
    q_zona_lpm = q_total_lpm / zonas
    q_m3s = q_zona_lpm / 60000.0
    d_m = diametro_mm / 1000.0
    C = 150 
    hf_psi = 0
    if d_m > 0 and q_m3s > 0:
        hf_m = 10.67 * longitud_m * ((q_m3s / C) ** 1.852) * (d_m ** -4.87)
        hf_psi = hf_m * 1.422
    return q_total_lpm, q_zona_lpm, hf_psi

def generar_pdf_reporte(img_resultado, n_asp, q_total, q_zona, hf_psi, materiales_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Reporte de Diseno de Riego", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, "1. Visualizacion del Diseno:", ln=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        from PIL import Image
        img_pil = Image.fromarray(img_resultado)
        img_pil.save(tmpfile.name)
        pdf.image(tmpfile.name, x=15, w=180)
        tmp_path = tmpfile.name

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Calculos Hidraulicos:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Total de Aspersores: {n_asp}", ln=True)
    pdf.cell(0, 8, f"Caudal Total del Sistema: {q_total:.2f} L/min", ln=True)
    pdf.cell(0, 8, f"Caudal por Zona: {q_zona:.2f} L/min", ln=True)
    pdf.cell(0, 8, f"Perdida de Carga Estimada: {hf_psi:.2f} PSI", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Lista Estimada de Materiales (BOM):", ln=True)
    pdf.set_font("Arial", size=11)
    for item, cantidad in materiales_dict.items():
        pdf.cell(0, 8, f"- {item}: {cantidad}", ln=True)
        
    pdf.ln(10)
    pdf.cell(0, 10, "Generado por Gemini Irrigation App", align='C')
    return pdf.output(dest='S').encode('latin-1'), tmp_path

# -----------------------------------------------------------------------------
# 3. INTERFAZ DE USUARIO (STREAMLIT)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Master Riego Pro", layout="wide")
st.title("💧 Sistema Profesional de Diseño de Riego")

# --- SIDEBAR ---
st.sidebar.header("1. Configuración de Terreno")
ancho_real = st.sidebar.number_input("Ancho real imagen (m):", value=40.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("2. Selección de Aspersor")

cat_keys = list(CATALOGO_ASPERSORES.keys())
categoria = st.sidebar.selectbox("Tipo de Proyecto:", cat_keys)
mod_keys = list(CATALOGO_ASPERSORES[categoria].keys())
modelo_seleccionado = st.sidebar.selectbox("Modelo de Aspersor:", mod_keys)
datos_modelo = CATALOGO_ASPERSORES[categoria][modelo_seleccionado]
st.sidebar.caption(f"Tipo: {datos_modelo['tipo']} | Presión: {datos_modelo['presion_psi']} PSI")

col1, col2 = st.sidebar.columns(2)
radio_asp = col1.number_input("Radio (m):", value=float(datos_modelo["radio"]), format="%.1f", key=f"rad_{modelo_seleccionado}")
q_asp = col2.number_input("Caudal (L/min):", value=float(datos_modelo["caudal_lpm"]), format="%.1f", key=f"caud_{modelo_seleccionado}")

st.sidebar.markdown("---")
st.sidebar.header("3. Tubería y Zonas")
diametro_tubo = st.sidebar.number_input("Diámetro tubería (mm):", value=40.0)
largo_tubo = st.sidebar.number_input("Longitud tubería (m):", value=60.0)
num_zonas = st.sidebar.selectbox("Zonas de riego:", [1, 2, 3, 4], index=1)

# --- APP ---
uploaded_file = st.file_uploader("Sube imagen aérea (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Convertimos a RGB siempre
    image = Image.open(uploaded_file).convert("RGB")
    
    canvas_width = 700
    w_percent = (canvas_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    bg_image = image.resize((canvas_width, h_size))
    
    ESCALA = canvas_width / ancho_real
    
    tab_manual, tab_auto, tab_res = st.tabs(["✍️ 1. Dibujo Manual", "🤖 2. Detección Auto", "📊 3. Resultados y PDF"])
    
    poly_manual = []
    
    with tab_manual:
        st.info("Dibuja el polígono del área a regar.")
        
        # El canvas funcionará ahora gracias al "Monkey Patch" del inicio
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)", stroke_width=2, stroke_color="green",
            background_image=bg_image, 
            height=h_size, width=canvas_width,
            drawing_mode="polygon", key="cv_manual"
        )
        if canvas.json_data and canvas.json_data["objects"]:
            path = canvas.json_data["objects"][-1]["path"]
            poly_manual = [(float(p[1]), float(p[2])) for p in path if len(p) >= 3]
            if st.button("Usar Diseño Manual"):
                st.session_state['modo_activo'] = 'manual'
                st.session_state['datos_geo'] = poly_manual
                st.success("✅ Diseño manual seleccionado.")

    with tab_auto:
        st.info("Ajusta sliders para aislar el verde. Negro = Obstáculo.")
        c1, c2, c3 = st.columns(3)
        h_min = c1.slider("H Min", 0, 179, 30)
        h_max = c1.slider("H Max", 0, 179, 90)
        s_min = c2.slider("S Min", 0, 255, 40)
        s_max = c2.slider("S Max", 0, 255, 255)
        v_min = c3.slider("V Min", 0, 255, 40)
        v_max = c3.slider("V Max", 0, 255, 255)
        
        mask_result = auto_detectar_verde(bg_image, h_min, s_min, v_min, h_max, s_max)
        col_a, col_b = st.columns(2)
        col_a.image(bg_image, caption="Original", use_column_width=True)
        col_b.image(mask_result, caption="Máscara", use_column_width=True)
        
        if st.button("Usar Detección Automática"):
            st.session_state['modo_activo'] = 'auto'
            st.session_state['datos_geo'] = mask_result
            st.success("✅ Diseño automático seleccionado.")

    with tab_res:
        st.write("---")
        if 'modo_activo' in st.session_state and 'datos_geo' in st.session_state:
            st.markdown(f"### Modo: **{st.session_state['modo_activo'].upper()}**")
            
            if st.button("⚙️ Generar Distribución y Cálculo", type="primary"):
                aspersores, r_pix = calcular_posiciones_aspersores(
                    st.session_state['modo_activo'], 
                    st.session_state['datos_geo'], 
                    radio_asp, 
                    ESCALA
                )
                
                img_final = dibujar_resultado_opencv(bg_image, aspersores, r_pix)
                st.image(img_final, caption="Propuesta de Distribución", channels="RGB")
                
                num_asp = len(aspersores)
                q_tot, q_zona, hf = calcular_hidraulica(num_asp, q_asp, diametro_tubo, largo_tubo, num_zonas)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Aspersores", num_asp)
                m2.metric("Caudal Total", f"{q_tot:.1f}")
                m3.metric("Caudal/Zona", f"{q_zona:.1f}")
                m4.metric("Pérdida Presión", f"{hf:.2f} PSI", delta_color="inverse")
                
                materiales = {
                    f"Aspersores ({modelo_seleccionado})": num_asp,
                    "Tubería": f"{largo_tubo} m",
                    "Válvulas": num_zonas,
                    "Conectores": int(num_asp * 2),
                    "Controlador": 1
                }
                pdf_bytes, tmp_path = generar_pdf_reporte(img_final, num_asp, q_tot, q_zona, hf, materiales)
                
                st.download_button(
                    label="📄 Descargar PDF",
                    data=pdf_bytes,
                    file_name="Proyecto_Riego.pdf",
                    mime="application/pdf"
                )
                if os.path.exists(tmp_path): os.remove(tmp_path)
        else:
            st.info("👈 Confirma un diseño primero.")
else:
    st.info("👆 Sube una imagen.")
