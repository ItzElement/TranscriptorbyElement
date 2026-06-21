import customtkinter as ctk
from tkinter import filedialog
import threading
import math
import os
import sys
import subprocess
import json
import urllib.request
from motorIA import MotorIA

# --- FUNCIÓN FALTANTE CORREGIDA ---
def format_ui_time(seconds: float):
    minutes = math.floor(seconds / 60)
    secs = math.floor(seconds % 60)
    return f"{minutes}:{secs:02d}"

ctk.set_appearance_mode("dark")

class TranscriptorProApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Transcriptor By Element")
        self.geometry("950x850")
        self.configure(fg_color="#0A0A0A") 

        # Instanciamos el Cerebro (Motor IA)
        self.motor = MotorIA()
        
        # Variables Globales de Estado
        self.archivo_entrada = None
        self.total_segmentos = 0
        self.ultimo_srt_generado = None
        self.altura_guion_actual = 120  

        self.crear_interfaz()
        self.verificar_estado_ollama()

    def crear_interfaz(self):
        # HEADER
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(header_frame, text="Auto Transcriptor by Element", font=("Helvetica", 22, "bold"), text_color="#FFFFFF").pack(side="left")
        
        # SISTEMA DE PESTAÑAS (TABS)
        self.tabview = ctk.CTkTabview(self, fg_color="#101010", segmented_button_fg_color="#1A1A1A", segmented_button_selected_color="#36225B", segmented_button_selected_hover_color="#4B2F7E")
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        self.tab_limpiar = self.tabview.add("1. Depurador Crudo (.WAV)")
        self.tab_subtitular = self.tabview.add("2. Transcriptor (.SRT)")
        
        self.setup_tab_limpiar(self.tab_limpiar)
        self.setup_tab_subtitular(self.tab_subtitular)
        
        self.tabview.set("2. Transcriptor (.SRT)") # Empezamos mostrando esta por defecto hoy

    # ==========================================================
    # PESTAÑA 1: DEPURADOR (EL FUTURO ROUGH CUT)
    # ==========================================================
    def setup_tab_limpiar(self, parent):
        ctk.CTkLabel(parent, text="Próximamente: Corta tus audios crudos y elimina tomas falsas usando IA.", font=("Helvetica", 14), text_color="gray").pack(pady=50)

    # ==========================================================
    # PESTAÑA 2: SUBTITULADOR (LO QUE YA DOMINAMOS)
    # ==========================================================
    def setup_tab_subtitular(self, parent):
        top_frame = ctk.CTkFrame(parent, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=(5, 0))
        self.btn_editar = ctk.CTkButton(top_frame, text="Editar SRT", width=100, fg_color="transparent", border_width=1, border_color="#333333", text_color="#A0A0A0", hover_color="#1A1A1A", command=self.editar_srt)
        self.btn_editar.pack(side="right")

        file_frame = ctk.CTkFrame(parent, fg_color="#121212", border_width=1, border_color="#222222", corner_radius=6)
        file_frame.pack(fill="x", padx=10, pady=10)
        
        file_info_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        file_info_frame.pack(fill="x", padx=15, pady=15)
        
        self.lbl_archivo = ctk.CTkLabel(file_info_frame, text="Ningún archivo seleccionado", font=("Courier", 13), text_color="#666666")
        self.lbl_archivo.pack(side="left")
        
        btn_seleccionar = ctk.CTkButton(file_info_frame, text="Seleccionar archivo", width=130, fg_color="#1F2A22", border_width=1, border_color="#2C4532", text_color="#73A47A", hover_color="#27382B", command=self.seleccionar_archivo)
        btn_seleccionar.pack(side="right")

        meta_frame = ctk.CTkFrame(parent, fg_color="transparent")
        meta_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        font_meta = ("Helvetica", 11, "bold")
        self.lbl_idioma = ctk.CTkLabel(meta_frame, text="IDIOMA --", font=font_meta, text_color="#555555")
        self.lbl_idioma.pack(side="left", padx=(0, 20))
        self.lbl_duracion = ctk.CTkLabel(meta_frame, text="DURACIÓN --", font=font_meta, text_color="#555555")
        self.lbl_duracion.pack(side="left", padx=20)
        self.lbl_segmentos = ctk.CTkLabel(meta_frame, text="SEGMENTOS 0", font=font_meta, text_color="#555555")
        self.lbl_segmentos.pack(side="left", padx=20)

        script_frame = ctk.CTkFrame(parent, fg_color="#100D14", border_width=1, border_color="#2A1E35", corner_radius=6)
        script_frame.pack(fill="x", padx=10, pady=10)
        
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

        self.lbl_titulo_timeline = ctk.CTkLabel(parent, text="Línea de tiempo", font=("Helvetica", 12, "bold"), text_color="#777777")
        self.lbl_titulo_timeline.pack(anchor="w", padx=10, pady=(10, 0))
        
        timeline_frame = ctk.CTkFrame(parent, fg_color="#121212", border_width=1, border_color="#222222", corner_radius=6)
        timeline_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.txt_timeline = ctk.CTkTextbox(timeline_frame, fg_color="transparent", text_color="#DDDDDD", font=("Helvetica", 13), spacing1=5, spacing3=5)
        self.txt_timeline.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_timeline._textbox.tag_config("timestamp", foreground="#5DADE2", font=("Courier", 11, "bold")) 
        self.txt_timeline._textbox.tag_config("text", foreground="#B3B3B3", font=("Helvetica", 13))            
        self.txt_timeline._textbox.tag_config("sys_msg", foreground="#F39C12", font=("Helvetica", 12, "italic")) 
        self.txt_timeline._textbox.tag_config("llama_msg", foreground="#9B59B6", font=("Helvetica", 12, "italic")) 
        self.txt_timeline.configure(state="disabled")

    # --- FUNCIONES DE LA UI Y CALLBACKS ---
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
                self.cb_log_system(f"⚠️ Error al abrir el archivo: {str(e)}")

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

    # Callbacks inyectables para que el MotorIA modifique la Interfaz
    def cb_log_system(self, msg, color_tag="sys_msg"):
        self.after(0, self._insertar_texto, None, None, msg, color_tag)

    def cb_log_segment(self, start_sec, end_sec, text):
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

    def cb_actualizar_metricas(self, idioma, duracion):
        mins = math.floor(duracion / 60)
        secs = math.floor(duracion % 60)
        self.lbl_idioma.configure(text=f"IDIOMA {idioma.upper()}", text_color="#E0E0E0")
        self.lbl_duracion.configure(text=f"DURACIÓN {mins}m {secs}s", text_color="#E0E0E0")

    def cb_actualizar_contador(self):
        self.total_segmentos += 1
        self.lbl_segmentos.configure(text=f"SEGMENTOS {self.total_segmentos}", text_color="#E0E0E0")
        self.lbl_titulo_timeline.configure(text=f"Línea de tiempo (Transcribiendo...)")
        
    def cb_finalizar_proceso(self):
        self.btn_generar.configure(state="normal", text="Generar SRT")
        self.lbl_titulo_timeline.configure(text=f"Línea de tiempo ({self.total_segmentos} segmentos generados)")

    # --- INICIO DEL PROCESO ---
    def iniciar_transcripcion(self):
        if not self.archivo_entrada:
            self.cb_log_system("⚠️ ERROR: Selecciona un archivo multimedia primero.")
            return
            
        archivo_salida = filedialog.asksaveasfilename(defaultextension=".srt", initialfile="subtitulos.srt")
        if not archivo_salida: 
            return 
            
        self.ultimo_srt_generado = archivo_salida 
        self.txt_timeline.configure(state="normal")
        self.txt_timeline.delete("1.0", "end")
        self.txt_timeline.configure(state="disabled")
        
        self.btn_generar.configure(state="disabled", text="Procesando...")
        self.lbl_titulo_timeline.configure(text="Línea de tiempo (Transcribiendo...)")
        self.total_segmentos = 0
        
        motor_seleccionado = self.combo_motor.get()
        guion_texto = self.txt_guion.get("1.0", "end").strip()
        
        # Le enviamos al motor la información y las funciones para que nos reporte el progreso
        hilo = threading.Thread(
            target=self.motor.procesar_subtitulos, 
            args=(
                self.archivo_entrada, archivo_salida, motor_seleccionado, guion_texto,
                self.cb_log_system, self.cb_log_segment, self.cb_actualizar_metricas, 
                self.cb_actualizar_contador, self.cb_finalizar_proceso
            )
        )
        hilo.start()

if __name__ == "__main__":
    app = TranscriptorProApp()
    app.mainloop()