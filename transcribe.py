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
from faster_whisper import WhisperModel
import math
import subprocess

# --- Funciones de Formato ---
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
    """Convierte todo a minúsculas y capitaliza solo la primera letra real"""
    texto = texto.strip().lower()
    for i, char in enumerate(texto):
        if char.isalpha(): # Si encuentra la primera letra (ignora signos como ¿)
            return texto[:i] + char.upper() + texto[i+1:]
    return texto

# --- CONFIGURACIÓN GLOBAL ---
ctk.set_appearance_mode("dark")

class TranscriptorProApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Transcriptor de audio by Element")
        self.geometry("950x850")
        self.configure(fg_color="#0A0A0A") 

        self.archivo_entrada = None
        self.modelo_cargado = None
        self.total_segmentos = 0
        self.ultimo_srt_generado = None  # Guardaremos la ruta aquí para el botón Editar

        self.crear_interfaz()

    def crear_interfaz(self):
        # --- HEADER ---
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(header_frame, text="Transcripción by Element V1.0", font=("Helvetica", 18, "bold"), text_color="#FFFFFF").pack(side="left")
        
        # Botón de Editar SRT ya conectado a la función
        self.btn_editar = ctk.CTkButton(header_frame, text="Editar SRT", width=100, fg_color="transparent", border_width=1, border_color="#333333", text_color="#A0A0A0", hover_color="#1A1A1A", command=self.editar_srt)
        self.btn_editar.pack(side="right")

        # --- SECCIÓN 1: AUDIO CONSOLIDADO (Archivo) ---
        file_frame = ctk.CTkFrame(self, fg_color="#121212", border_width=1, border_color="#222222", corner_radius=6)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        file_info_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        file_info_frame.pack(fill="x", padx=15, pady=15)
        
        self.lbl_archivo = ctk.CTkLabel(file_info_frame, text="Ningún archivo seleccionado", font=("Courier", 13), text_color="#666666")
        self.lbl_archivo.pack(side="left")
        
        btn_seleccionar = ctk.CTkButton(file_info_frame, text="Seleccionar archivo", width=130, fg_color="#1F2A22", border_width=1, border_color="#2C4532", text_color="#73A47A", hover_color="#27382B", command=self.seleccionar_archivo)
        btn_seleccionar.pack(side="right")

        # --- SECCIÓN METADATA ---
        meta_frame = ctk.CTkFrame(self, fg_color="transparent")
        meta_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        font_meta = ("Helvetica", 11, "bold")
        self.lbl_idioma = ctk.CTkLabel(meta_frame, text="IDIOMA --", font=font_meta, text_color="#555555")
        self.lbl_idioma.pack(side="left", padx=(0, 20))
        
        self.lbl_duracion = ctk.CTkLabel(meta_frame, text="DURACIÓN --", font=font_meta, text_color="#555555")
        self.lbl_duracion.pack(side="left", padx=20)
        
        self.lbl_segmentos = ctk.CTkLabel(meta_frame, text="SEGMENTOS 0", font=font_meta, text_color="#555555")
        self.lbl_segmentos.pack(side="left", padx=20)

        # --- SECCIÓN 2: GUION ORIGINAL (Morado oscuro) ---
        script_frame = ctk.CTkFrame(self, fg_color="#100D14", border_width=1, border_color="#2A1E35", corner_radius=6)
        script_frame.pack(fill="x", padx=20, pady=10)
        
        script_header = ctk.CTkFrame(script_frame, fg_color="transparent")
        script_header.pack(fill="x", padx=15, pady=(10, 0))
        
        ctk.CTkLabel(script_header, text="Guion original", font=("Helvetica", 13, "bold"), text_color="#8E7BB3").pack(side="left")
        
        self.btn_generar = ctk.CTkButton(script_header, text="Generar SRT", width=120, fg_color="#36225B", hover_color="#4B2F7E", text_color="#E0D4F5", command=self.iniciar_transcripcion)
        self.btn_generar.pack(side="right")
        
        combo_motor = ctk.CTkComboBox(script_header, values=["Whisper (Crudo)", "Llama 3 Local"], width=130, fg_color="#151515", border_color="#333333", text_color="#AAAAAA")
        combo_motor.set("Whisper (Crudo)")
        combo_motor.pack(side="right", padx=10)
        ctk.CTkLabel(script_header, text="MOTOR", font=("Helvetica", 10, "bold"), text_color="#555555").pack(side="right", padx=5)

        self.txt_guion = ctk.CTkTextbox(script_frame, height=120, fg_color="#151515", border_width=0, text_color="#B0B0B0", font=("Helvetica", 13))
        self.txt_guion.pack(fill="x", padx=15, pady=15)
        self.txt_guion.insert("1.0", "# Pega aquí el guion original para que la IA lo use como contexto al corregir...\n")

        # --- SECCIÓN 3: LÍNEA DE TIEMPO ---
        self.lbl_titulo_timeline = ctk.CTkLabel(self, text="Línea de tiempo", font=("Helvetica", 12, "bold"), text_color="#777777")
        self.lbl_titulo_timeline.pack(anchor="w", padx=20, pady=(10, 0))
        
        timeline_frame = ctk.CTkFrame(self, fg_color="#121212", border_width=1, border_color="#222222", corner_radius=6)
        timeline_frame.pack(fill="both", expand=True, padx=20, pady=(5, 20))

        self.txt_timeline = ctk.CTkTextbox(timeline_frame, fg_color="transparent", text_color="#DDDDDD", font=("Helvetica", 13), spacing1=5, spacing3=5)
        self.txt_timeline.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_timeline._textbox.tag_config("timestamp", foreground="#5DADE2", font=("Courier", 11, "bold")) 
        self.txt_timeline._textbox.tag_config("text", foreground="#B3B3B3", font=("Helvetica", 13))            
        self.txt_timeline._textbox.tag_config("sys_msg", foreground="#F39C12", font=("Helvetica", 12, "italic")) 
        self.txt_timeline.configure(state="disabled")

    def editar_srt(self):
        """Abre el archivo SRT en el Bloc de Notas (Windows) o editor por defecto"""
        if self.ultimo_srt_generado and os.path.exists(self.ultimo_srt_generado):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(self.ultimo_srt_generado)
                elif sys.platform == "darwin":  # macOS
                    subprocess.call(('open', self.ultimo_srt_generado))
            except Exception as e:
                self.log_system(f"⚠️ Error al abrir el archivo: {str(e)}")
        else:
            self.log_system("⚠️ Aún no has generado ningún SRT para editar.")

    def seleccionar_archivo(self):
        filepath = filedialog.askopenfilename(title="Selecciona el archivo", filetypes=[("Audio/Video", "*.mp4 *.mp3 *.wav *.mkv *.mov *.m4a")])
        if filepath:
            self.archivo_entrada = filepath
            nombre_corto = filepath.split("/")[-1]
            self.lbl_archivo.configure(text=nombre_corto, text_color="#EEEEEE")
            
            # Resetear métricas
            self.lbl_idioma.configure(text="IDIOMA --", text_color="#555555")
            self.lbl_duracion.configure(text="DURACIÓN --", text_color="#555555")
            self.lbl_segmentos.configure(text="SEGMENTOS 0", text_color="#555555")
            self.total_segmentos = 0
            self.lbl_titulo_timeline.configure(text="Línea de tiempo")
            
            # Borrar la línea de tiempo visual
            self.txt_timeline.configure(state="normal")
            self.txt_timeline.delete("1.0", "end")
            self.txt_timeline.configure(state="disabled")

    def log_system(self, msg):
        self.after(0, self._insertar_texto, None, None, msg, "sys_msg")

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
        self.lbl_titulo_timeline.configure(text=f"Línea de tiempo ({self.total_segmentos} segmentos generados)")

    def iniciar_transcripcion(self):
        if not self.archivo_entrada:
            self.log_system("⚠️ ERROR: Selecciona un archivo multimedia primero.")
            return
            
        archivo_salida = filedialog.asksaveasfilename(defaultextension=".srt", initialfile="subtitulos.srt")
        if not archivo_salida:
            return 
            
        self.ultimo_srt_generado = archivo_salida  # Guardamos la ruta para el botón de editar
            
        self.txt_timeline.configure(state="normal")
        self.txt_timeline.delete("1.0", "end")
        self.txt_timeline.configure(state="disabled")
        
        self.btn_generar.configure(state="disabled", text="Transcribiendo...")
        self.lbl_titulo_timeline.configure(text="Línea de tiempo (Transcribiendo...)")
        self.total_segmentos = 0

        hilo = threading.Thread(target=self.procesar_whisper, args=(self.archivo_entrada, archivo_salida))
        hilo.start()

    def procesar_whisper(self, entrada, salida):
        try:
            if not self.modelo_cargado:
                self.log_system("Iniciando motor Whisper V3 (GPU). Por favor espera...")
                self.modelo_cargado = WhisperModel("large-v3", device="cuda", compute_type="float16")
            
            self.log_system("Analizando audio...")
            
            segments, info = self.modelo_cargado.transcribe(
                entrada, 
                beam_size=5, 
                language="es",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250),
                word_timestamps=True
            )

            self.after(0, self.actualizar_metricas, info.language, info.duration)

            # TUS NUEVOS LÍMITES
            MAX_PALABRAS = 14
            MIN_PALABRAS = 5
            sub_index = 1
            
            with open(salida, "w", encoding="utf-8") as srt_file:
                for segment in segments:
                    current_chunk_words = []
                    chunk_start = None
                    
                    for word_obj in segment.words:
                        if chunk_start is None:
                            chunk_start = word_obj.start
                        
                        palabra_actual = word_obj.word.strip()
                        current_chunk_words.append(palabra_actual)
                        
                        pausa_fuerte = any(p in palabra_actual for p in ['.', '?', '!', '…'])
                        pausa_suave = any(p in palabra_actual for p in [',', ':', ';'])
                        
                        if (len(current_chunk_words) >= MAX_PALABRAS) or pausa_fuerte or (pausa_suave and len(current_chunk_words) >= MIN_PALABRAS):
                            chunk_end = word_obj.end
                            
                            texto_final = " ".join(current_chunk_words)
                            
                            # Borrar puntuación, PERO NO los signos de interrogación (? y ¿)
                            for char in [',', '.', ';', ':', '!', '…', '¡']:
                                texto_final = texto_final.replace(char, "")
                            
                            # Aplicar minúsculas y capitalizar solo la primera letra
                            texto_final = capitalizar_primera_letra(texto_final)
                            
                            srt_file.write(f"{sub_index}\n{format_timestamp(chunk_start)} --> {format_timestamp(chunk_end)}\n{texto_final}\n\n")
                            
                            self.log_segment(chunk_start, chunk_end, texto_final)
                            self.after(0, self.actualizar_contador_segmentos)
                            
                            sub_index += 1
                            current_chunk_words = []
                            chunk_start = None
                    
                    # Para lo que sobre al final de la línea
                    if current_chunk_words:
                        chunk_end = segment.end
                        texto_final = " ".join(current_chunk_words)
                        
                        for char in [',', '.', ';', ':', '!', '…', '¡']:
                            texto_final = texto_final.replace(char, "")
                            
                        texto_final = capitalizar_primera_letra(texto_final)
                            
                        srt_file.write(f"{sub_index}\n{format_timestamp(chunk_start)} --> {format_timestamp(chunk_end)}\n{texto_final}\n\n")
                        self.log_segment(chunk_start, chunk_end, texto_final)
                        self.after(0, self.actualizar_contador_segmentos)
                        sub_index += 1

            self.log_system(f"✅ ¡Transcripción completada con éxito!")
            self.log_system(f"💡 Tip: Puedes darle al botón 'Editar SRT' arriba a la derecha para ver el archivo.")
            
        except Exception as e:
            self.log_system(f"❌ ERROR FATAL: {str(e)}")
            
        finally:
            self.after(0, lambda: self.btn_generar.configure(state="normal", text="Generar SRT"))
            self.after(0, lambda: self.lbl_titulo_timeline.configure(text=f"Línea de tiempo ({self.total_segmentos} segmentos generados)"))

if __name__ == "__main__":
    app = TranscriptorProApp()
    app.mainloop()