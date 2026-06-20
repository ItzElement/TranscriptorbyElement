import os
import site

# --- PARCHE PARA WINDOWS: Forzar a que encuentre las DLLs de NVIDIA ---
if os.name == 'nt':
    # Buscar en todas las carpetas donde pip instala paquetes
    carpetas_pip = site.getsitepackages() + [site.getusersitepackages()]
    for carpeta in carpetas_pip:
        cublas_bin = os.path.join(carpeta, "nvidia", "cublas", "bin")
        cudnn_bin = os.path.join(carpeta, "nvidia", "cudnn", "bin")
        
        # Si encuentra la carpeta, la añade a las variables de entorno de Windows
        if os.path.exists(cublas_bin):
            os.add_dll_directory(cublas_bin)
            os.environ["PATH"] += os.pathsep + cublas_bin
        if os.path.exists(cudnn_bin):
            os.add_dll_directory(cudnn_bin)
            os.environ["PATH"] += os.pathsep + cudnn_bin
# ------------------------------------------------------------------------
import customtkinter as ctk
from tkinter import filedialog
import threading
import math
import os
import sys
import subprocess
import json
import urllib.request
import gc
import re
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

def format_timestamp(seconds: float):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def format_ui_time(seconds: float):
    minutes = math.floor(seconds / 60)
    secs = math.floor(seconds % 60)
    return f"{minutes}:{secs:02d}"

def capitalizar_primera_letra(texto):
    texto = texto.strip()
    for i, char in enumerate(texto):
        if char.isalpha(): 
            return texto[:i] + char.upper() + texto[i+1:]
    return texto

# --- CONEXIÓN A OLLAMA (CON ESCUDO ANTI-DUPLICADOS) ---
def corregir_lote_con_llama(lote_dicts, guion_referencia):
    url = "http://127.0.0.1:11434/api/generate"
    
    lineas_entrada = [f"{sub['index']}|{sub['text']}" for sub in lote_dicts]
    texto_entrada = "\n".join(lineas_entrada)
    
    prompt = f"""Eres un corrector ortográfico de subtítulos. Vas a comparar la ENTRADA (lo que escuchó la IA) con el GUION original.

REGLAS DE CORRECCIÓN (¡Síguelas estrictamente!):
1. ERRORES DE ESCUCHA: Si la ENTRADA dice algo erróneo (ej. "conseguir") pero el GUION dice la palabra correcta (ej. "construir"), OBLIGATORIAMENTE cámbialo por la palabra del guion.
2. REGIONALISMOS (VOSEO): Si la ENTRADA dice "querés", "tenés", OBLIGATORIAMENTE cámbialo a "querías", "tienes" según el guion.
3. CERO SINÓNIMOS: Si la entrada está bien, NO cambies palabras por sinónimos (ej. NUNCA cambies "estaba" por "había"). Respeta la palabra original.
4. IMPROVISACIONES: Si la entrada tiene palabras que no están en el guion, déjalas INTACTAS. El orador improvisó.
5. NO DUPLIQUES: Jamás repitas la oración de arriba.

Devuelve la lista con el formato exacto "ID|texto". Cero plática.

GUION ORIGINAL:
{guion_referencia}

ENTRADA A CORREGIR:
{texto_entrada}

SALIDA:
"""

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0 
        }
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            texto_corregido = result.get("response", "").strip()
            
            # --- EL EXTRACTOR DESTRUCTIVO DE PYTHON ---
            dict_respuestas = {}
            for linea in texto_corregido.split('\n'):
                partes = linea.split('|', 1)
                if len(partes) == 2:
                    idx_str = re.sub(r'\D', '', partes[0])
                    if idx_str:
                        idx = int(idx_str)
                        txt = partes[1].strip()
                        
                        txt = txt.split('->')[0].strip()
                        txt = re.sub(r'\(.*\)', '', txt).strip() 
                        txt = re.sub(r'(?i)no hay error.*$', '', txt).strip()
                        txt = re.sub(r'(?i)se mantiene igual.*$', '', txt).strip()
                        
                        dict_respuestas[idx] = txt
            
            # --- EL ESCUDO ESTRICTO DE PYTHON ---
            textos_nuevos_aceptados = set()
            
            for sub in lote_dicts:
                texto_viejo = sub['text']
                
                if sub['index'] in dict_respuestas:
                    texto_nuevo = dict_respuestas[sub['index']]
                    
                    if texto_viejo.lower() != texto_nuevo.lower():
                        pal_viejas = len(texto_viejo.split())
                        pal_nuevas = len(texto_nuevo.split())
                        
                        if pal_nuevas > 0 and abs(pal_nuevas - pal_viejas) <= 3:
                            similitud = SequenceMatcher(None, texto_viejo.lower(), texto_nuevo.lower()).ratio()
                            
                            # MAGIA: Escudo Anti-Duplicados
                            # Si Llama intenta usar un texto que YA usamos, y no era así originalmente, lo bloqueamos
                            if texto_nuevo.lower() in textos_nuevos_aceptados and texto_viejo.lower() not in textos_nuevos_aceptados:
                                print(f"[*] Escudo Anti-Duplicados en ID {sub['index']}: Llama repitió una línea. Se mantuvo original.")
                                textos_nuevos_aceptados.add(texto_viejo.lower())
                            elif similitud >= 0.55: # Permite corregir "conseguir" por "construir"
                                sub['text'] = texto_nuevo
                                textos_nuevos_aceptados.add(texto_nuevo.lower())
                            else:
                                print(f"[*] Escudo de Similitud en ID {sub['index']}: (Ratio {similitud:.2f}) Llama alucinó. Se mantuvo original.")
                                textos_nuevos_aceptados.add(texto_viejo.lower())
                        else:
                            print(f"[*] Escudo de Longitud en ID {sub['index']}: Llama alteró demasiadas palabras. Se mantuvo original.")
                            textos_nuevos_aceptados.add(texto_viejo.lower())
                    else:
                        textos_nuevos_aceptados.add(texto_viejo.lower())
                else:
                    textos_nuevos_aceptados.add(texto_viejo.lower())
                    
    except Exception as e:
        print(f"[ERROR OLLAMA]: {str(e)}")
        
    return lote_dicts

ctk.set_appearance_mode("dark")

class TranscriptorProApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Transcriptor IA")
        self.geometry("950x850")
        self.configure(fg_color="#0A0A0A") 

        self.archivo_entrada = None
        self.modelo_cargado = None
        self.total_segmentos = 0
        self.ultimo_srt_generado = None
        self.altura_guion_actual = 120  

        self.crear_interfaz()
        self.verificar_estado_ollama()

    def crear_interfaz(self):
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(header_frame, text="← Transcripción", font=("Helvetica", 18, "bold"), text_color="#FFFFFF").pack(side="left")
        self.btn_editar = ctk.CTkButton(header_frame, text="Editar SRT", width=100, fg_color="transparent", border_width=1, border_color="#333333", text_color="#A0A0A0", hover_color="#1A1A1A", command=self.editar_srt)
        self.btn_editar.pack(side="right")

        file_frame = ctk.CTkFrame(self, fg_color="#121212", border_width=1, border_color="#222222", corner_radius=6)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        file_info_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        file_info_frame.pack(fill="x", padx=15, pady=15)
        
        self.lbl_archivo = ctk.CTkLabel(file_info_frame, text="Ningún archivo seleccionado", font=("Courier", 13), text_color="#666666")
        self.lbl_archivo.pack(side="left")
        
        btn_seleccionar = ctk.CTkButton(file_info_frame, text="Seleccionar archivo", width=130, fg_color="#1F2A22", border_width=1, border_color="#2C4532", text_color="#73A47A", hover_color="#27382B", command=self.seleccionar_archivo)
        btn_seleccionar.pack(side="right")

        meta_frame = ctk.CTkFrame(self, fg_color="transparent")
        meta_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        font_meta = ("Helvetica", 11, "bold")
        self.lbl_idioma = ctk.CTkLabel(meta_frame, text="IDIOMA --", font=font_meta, text_color="#555555")
        self.lbl_idioma.pack(side="left", padx=(0, 20))
        self.lbl_duracion = ctk.CTkLabel(meta_frame, text="DURACIÓN --", font=font_meta, text_color="#555555")
        self.lbl_duracion.pack(side="left", padx=20)
        self.lbl_segmentos = ctk.CTkLabel(meta_frame, text="SEGMENTOS 0", font=font_meta, text_color="#555555")
        self.lbl_segmentos.pack(side="left", padx=20)

        script_frame = ctk.CTkFrame(self, fg_color="#100D14", border_width=1, border_color="#2A1E35", corner_radius=6)
        script_frame.pack(fill="x", padx=20, pady=10)
        
        script_header = ctk.CTkFrame(script_frame, fg_color="transparent")
        script_header.pack(fill="x", padx=15, pady=(10, 0))
        
        ctk.CTkLabel(script_header, text="Guion original", font=("Helvetica", 13, "bold"), text_color="#8E7BB3").pack(side="left")
        self.btn_generar = ctk.CTkButton(script_header, text="Generar SRT", width=120, fg_color="#36225B", hover_color="#4B2F7E", text_color="#E0D4F5", command=self.iniciar_transcripcion)
        self.btn_generar.pack(side="right")
        
        self.combo_motor = ctk.CTkComboBox(script_header, values=["Whisper (Crudo)", "Llama 3 Local"], width=130, fg_color="#151515", border_color="#333333", text_color="#AAAAAA")
        self.combo_motor.set("Whisper (Crudo)")
        self.combo_motor.pack(side="right", padx=10)
        
        self.lbl_ollama_status = ctk.CTkLabel(script_header, text="⚪ Verificando...", font=("Helvetica", 11, "bold"), text_color="#888888")
        self.lbl_ollama_status.pack(side="right", padx=10)
        
        ctk.CTkLabel(script_header, text="MOTOR", font=("Helvetica", 10, "bold"), text_color="#555555").pack(side="right", padx=5)

        self.txt_guion = ctk.CTkTextbox(script_frame, height=self.altura_guion_actual, fg_color="#151515", border_width=0, text_color="#B0B0B0", font=("Helvetica", 13))
        self.txt_guion.pack(fill="x", padx=15, pady=(15, 0))

        self.resizer_frame = ctk.CTkFrame(script_frame, height=10, fg_color="#36225B", cursor="sb_v_double_arrow", corner_radius=3)
        self.resizer_frame.pack(fill="x", padx=15, pady=(2, 10))
        lbl_dots = ctk.CTkLabel(self.resizer_frame, text="•••", font=("Arial", 9, "bold"), text_color="#A0A0A0", height=10)
        lbl_dots.place(relx=0.5, rely=0.5, anchor="center")

        self.resizer_frame.bind("<ButtonPress-1>", self.iniciar_redimension)
        self.resizer_frame.bind("<B1-Motion>", self.redimensionar_guion)
        lbl_dots.bind("<ButtonPress-1>", self.iniciar_redimension)
        lbl_dots.bind("<B1-Motion>", self.redimensionar_guion)

        self.lbl_titulo_timeline = ctk.CTkLabel(self, text="Línea de tiempo", font=("Helvetica", 12, "bold"), text_color="#777777")
        self.lbl_titulo_timeline.pack(anchor="w", padx=20, pady=(10, 0))
        
        timeline_frame = ctk.CTkFrame(self, fg_color="#121212", border_width=1, border_color="#222222", corner_radius=6)
        timeline_frame.pack(fill="both", expand=True, padx=20, pady=(5, 20))

        self.txt_timeline = ctk.CTkTextbox(timeline_frame, fg_color="transparent", text_color="#DDDDDD", font=("Helvetica", 13), spacing1=5, spacing3=5)
        self.txt_timeline.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_timeline._textbox.tag_config("timestamp", foreground="#5DADE2", font=("Courier", 11, "bold")) 
        self.txt_timeline._textbox.tag_config("text", foreground="#B3B3B3", font=("Helvetica", 13))            
        self.txt_timeline._textbox.tag_config("sys_msg", foreground="#F39C12", font=("Helvetica", 12, "italic")) 
        self.txt_timeline._textbox.tag_config("llama_msg", foreground="#9B59B6", font=("Helvetica", 12, "italic")) 
        self.txt_timeline.configure(state="disabled")

    def iniciar_redimension(self, event):
        self._start_y = event.y_root

    def redimensionar_guion(self, event):
        delta = event.y_root - self._start_y
        self.altura_guion_actual += delta
        if self.altura_guion_actual < 60: self.altura_guion_actual = 60
        if self.altura_guion_actual > 450: self.altura_guion_actual = 450
        self.txt_guion.configure(height=self.altura_guion_actual)
        self._start_y = event.y_root

    def verificar_estado_ollama(self):
        threading.Thread(target=self._hilo_ping_ollama, daemon=True).start()
        self.after(4000, self.verificar_estado_ollama)

    def _hilo_ping_ollama(self):
        try:
            url = "http://127.0.0.1:11434/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
                modelos = [m["name"] for m in data.get("models", [])]
                if any(m.startswith("llama3") for m in modelos):
                    self.after(0, lambda: self.lbl_ollama_status.configure(text="🟢 Llama ON", text_color="#2ECC71"))
                else:
                    self.after(0, lambda: self.lbl_ollama_status.configure(text="🟠 Falta Modelo", text_color="#F39C12"))
        except Exception:
            self.after(0, lambda: self.lbl_ollama_status.configure(text="🔴 Ollama OFF", text_color="#E74C3C"))

    def editar_srt(self):
        if self.ultimo_srt_generado and os.path.exists(self.ultimo_srt_generado):
            try:
                if os.name == 'nt': os.startfile(self.ultimo_srt_generado)
                elif sys.platform == "darwin": subprocess.call(('open', self.ultimo_srt_generado))
            except Exception as e:
                self.log_system(f"⚠️ Error al abrir el archivo: {str(e)}")

    def seleccionar_archivo(self):
        filepath = filedialog.askopenfilename(title="Selecciona el archivo", filetypes=[("Audio/Video", "*.mp4 *.mp3 *.wav *.mkv *.mov *.m4a")])
        if filepath:
            self.archivo_entrada = filepath
            nombre_corto = filepath.split("/")[-1]
            self.lbl_archivo.configure(text=nombre_corto, text_color="#EEEEEE")
            self.lbl_idioma.configure(text="IDIOMA --", text_color="#555555")
            self.lbl_duracion.configure(text="DURACIÓN --", text_color="#555555")
            self.lbl_segmentos.configure(text="SEGMENTOS 0", text_color="#555555")
            self.total_segmentos = 0
            self.lbl_titulo_timeline.configure(text="Línea de tiempo")
            self.txt_timeline.configure(state="normal")
            self.txt_timeline.delete("1.0", "end")
            self.txt_timeline.configure(state="disabled")

    def log_system(self, msg, color_tag="sys_msg"):
        self.after(0, self._insertar_texto, None, None, msg, color_tag)

    def log_segment(self, start_sec, end_sec, text):
        time_str = f"{format_ui_time(start_sec)} - {format_ui_time(end_sec)}"
        self.after(0, self._insertar_texto, time_str, text, None, None)

    def _insertar_texto(self, time_str, text_str, sys_msg, tag):
        self.txt_timeline.configure(state="normal")
        if sys_msg:
            self.txt_timeline.insert("end", f"{sys_msg}\n\n", tag)
        else:
            self.txt_timeline.insert("end", f"{time_str:<15}", "timestamp")
            self.txt_timeline.insert("end", f"{text_str}\n\n", "text")
        self.txt_timeline.see("end")
        self.txt_timeline.configure(state="disabled")

    def actualizar_metricas(self, idioma, duracion):
        mins = math.floor(duracion / 60)
        secs = math.floor(duracion % 60)
        self.lbl_idioma.configure(text=f"IDIOMA {idioma.upper()}", text_color="#E0E0E0")
        self.lbl_duracion.configure(text=f"DURACIÓN {mins}m {secs}s", text_color="#E0E0E0")

    def actualizar_contador_segmentos(self):
        self.total_segmentos += 1
        self.lbl_segmentos.configure(text=f"SEGMENTOS {self.total_segmentos}", text_color="#E0E0E0")
        self.lbl_titulo_timeline.configure(text=f"Línea de tiempo (Transcribiendo...)")

    def _forzar_limpieza_vram_ollama(self):
        self.log_system("🧹 Vaciando VRAM de Llama 3...", "sys_msg")
        url = "http://127.0.0.1:11434/api/generate"
        payload = {"model": "llama3", "keep_alive": 0} 
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req)
        except Exception: pass

    def iniciar_transcripcion(self):
        if not self.archivo_entrada:
            self.log_system("⚠️ ERROR: Selecciona un archivo multimedia primero.")
            return
        archivo_salida = filedialog.asksaveasfilename(defaultextension=".srt", initialfile="subtitulos.srt")
        if not archivo_salida: return 
        self.ultimo_srt_generado = archivo_salida 
        self.txt_timeline.configure(state="normal")
        self.txt_timeline.delete("1.0", "end")
        self.txt_timeline.configure(state="disabled")
        self.btn_generar.configure(state="disabled", text="Procesando...")
        self.lbl_titulo_timeline.configure(text="Línea de tiempo (Transcribiendo...)")
        self.total_segmentos = 0
        motor_seleccionado = self.combo_motor.get()
        guion_texto = self.txt_guion.get("1.0", "end").strip()
        hilo = threading.Thread(target=self.procesar_whisper_y_llama, args=(self.archivo_entrada, archivo_salida, motor_seleccionado, guion_texto))
        hilo.start()

    def procesar_whisper_y_llama(self, entrada, salida, motor, guion):
        try:
            if not self.modelo_cargado:
                self.log_system("Iniciando motor Whisper V3 (GPU). Por favor espera...")
                self.modelo_cargado = WhisperModel("large-v3", device="cuda", compute_type="float16")
            
            self.log_system("Analizando audio y calculando pausas...")
            
            segments, info = self.modelo_cargado.transcribe(
                entrada, beam_size=5, language="es",
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=250),
                word_timestamps=True
            )

            self.after(0, self.actualizar_metricas, info.language, info.duration)

            MAX_PALABRAS = 12 
            MIN_PALABRAS = 1
            sub_index = 1
            lista_srt_crudo = []
            
            for segment in segments:
                words = segment.words
                current_chunk_words = []
                chunk_start = None
                
                for idx, word_obj in enumerate(words):
                    if chunk_start is None:
                        chunk_start = word_obj.start
                    
                    palabra_cruda = word_obj.word.strip()
                    tiene_puntos = "..." in palabra_cruda or "…" in palabra_cruda
                    duracion = word_obj.end - word_obj.start
                    
                    gap = 0.0
                    if idx + 1 < len(words):
                        gap = words[idx+1].start - word_obj.end
                    pausa_por_silencio = gap >= 0.45
                    
                    palabra_limpia = palabra_cruda.lower()
                    for char in [',', '.', ';', ':', '!', '¡', '?', '¿', '…', '"', "'", '(', ')']:
                        palabra_limpia = palabra_limpia.replace(char, "")
                    
                    es_suspenso = tiene_puntos or (duracion > 1.2 and len(palabra_limpia) > 3)
                    
                    if es_suspenso:
                        palabra_limpia += "..."
                    
                    if "¿" in palabra_cruda: palabra_limpia = "¿" + palabra_limpia
                    if "?" in palabra_cruda: palabra_limpia = palabra_limpia + "?"
                    
                    current_chunk_words.append(palabra_limpia)
                    
                    pausa_fuerte = any(p in palabra_cruda for p in ['.', '?', '!', '…'])
                    pausa_suave = any(p in palabra_cruda for p in [',', ':', ';'])
                    
                    if (len(current_chunk_words) >= MAX_PALABRAS) or pausa_fuerte or (pausa_suave and len(current_chunk_words) >= MIN_PALABRAS) or es_suspenso or pausa_por_silencio:
                        chunk_end = word_obj.end
                        texto_final = " ".join(current_chunk_words)
                        texto_final = re.sub(r'\.{4,}', '...', texto_final)
                        
                        lista_srt_crudo.append({"index": sub_index, "start": format_timestamp(chunk_start), "end": format_timestamp(chunk_end), "text": texto_final})
                        
                        self.log_segment(chunk_start, chunk_end, texto_final)
                        self.after(0, self.actualizar_contador_segmentos)
                        
                        sub_index += 1
                        current_chunk_words = []
                        chunk_start = None
                
                if current_chunk_words:
                    chunk_end = segment.end
                    texto_final = " ".join(current_chunk_words)
                    texto_final = re.sub(r'\.{4,}', '...', texto_final)
                        
                    lista_srt_crudo.append({"index": sub_index, "start": format_timestamp(chunk_start), "end": format_timestamp(chunk_end), "text": texto_final})
                    self.log_segment(chunk_start, chunk_end, texto_final)
                    self.after(0, self.actualizar_contador_segmentos)
                    sub_index += 1

            self.log_system("\nLiberando memoria gráfica de Whisper...", "sys_msg")
            del self.modelo_cargado
            self.modelo_cargado = None
            gc.collect() 
                
            usar_llama = (motor == "Llama 3 Local" and len(guion) > 10)
            
            if usar_llama:
                self.log_system("🧠 Iniciando Llama 3. Corrigiendo subtítulos por lotes...", "llama_msg")
                tamano_lote = 8 # Lotes chiquitos para que no se maree Llama
                total_lotes = math.ceil(len(lista_srt_crudo) / tamano_lote)
                
                for i in range(0, len(lista_srt_crudo), tamano_lote):
                    lote_actual = lista_srt_crudo[i : i + tamano_lote]
                    numero_lote = int(i / tamano_lote) + 1
                    self.log_system(f"  -> Procesando lote {numero_lote} de {total_lotes}...", "llama_msg")
                    
                    corregir_lote_con_llama(lote_actual, guion)
                
                self.log_system("✨ Llama 3 ha finalizado la corrección.", "llama_msg")
                self._forzar_limpieza_vram_ollama()
                
            elif motor == "Llama 3 Local":
                self.log_system("⚠️ Elegiste Llama 3 pero no pegaste el guion. El archivo SRT se guardó en crudo.", "sys_msg")

            # --- ESTÉTICA FINAL DE ESCRITURA ---
            srt_final_text = ""
            capitalizar_sig = True
            
            for sub in lista_srt_crudo:
                texto_limpio = sub['text'].strip()
                texto_limpio = texto_limpio.replace("@", "o")
                
                texto_limpio = texto_limpio.replace("...", "___ELLIPSIS___").replace("…", "___ELLIPSIS___")
                texto_limpio = texto_limpio.replace("¿", "___QOPEN___").replace("?", "___QCLOSE___")
                
                for char in [',', '.', ';', ':', '!', '¡', '"', "'", '(', ')']:
                    texto_limpio = texto_limpio.replace(char, "")
                    
                texto_limpio = texto_limpio.replace("___ELLIPSIS___", "...")
                texto_limpio = texto_limpio.replace("___QOPEN___", "¿").replace("___QCLOSE___", "?")
                
                texto_limpio = texto_limpio.lower()
                
                if sub['index'] == 1:
                    texto_limpio = capitalizar_primera_letra(texto_limpio)
                elif "¿" in texto_limpio:
                    for i, char in enumerate(texto_limpio):
                        if char.isalpha():
                            texto_limpio = texto_limpio[:i] + char.upper() + texto_limpio[i+1:]
                            break
                elif capitalizar_sig and len(texto_limpio) > 0:
                    texto_limpio = capitalizar_primera_letra(texto_limpio)
                    capitalizar_sig = False
                    
                if texto_limpio.endswith("..."):
                    capitalizar_sig = True
                
                srt_final_text += f"{sub['index']}\n{sub['start']} --> {sub['end']}\n{texto_limpio}\n\n"
            
            with open(salida, "w", encoding="utf-8") as f:
                f.write(srt_final_text.strip() + "\n")

            self.log_system(f"✅ ¡PROCESO COMPLETADO EXITOSAMENTE!")
            
        except Exception as e:
            self.log_system(f"❌ ERROR FATAL: {str(e)}")
            
        finally:
            self.after(0, lambda: self.btn_generar.configure(state="normal", text="Generar SRT"))
            self.after(0, lambda: self.lbl_titulo_timeline.configure(text=f"Línea de tiempo ({self.total_segmentos} segmentos generados)"))

if __name__ == "__main__":
    app = TranscriptorProApp()
    app.mainloop()