# -*- coding: utf-8 -*-

import os
import shutil
import time
import json
import re
import threading
import logging
import logging.handlers
from difflib import get_close_matches

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.utils import platform

# --- Constants ---
WAKE_PHRASE = "hey kala"
APP_NAME = "KalaAI"

# Guard against running on non-Android platforms

# Guard against running on non-Android platforms
if platform == 'android':
    from jnius import autoclass, PythonJavaClass, java_method, cast  # type: ignore # pylint: disable=import-error,unresolved-import


    # --- Android Class Imports ---
    try:
        # Core & Activity
        Context = autoclass('android.content.Context')
        Activity = autoclass('org.kivy.android.PythonActivity')
        Intent = autoclass('android.content.Intent')
        Uri = autoclass('android.net.Uri')

        # Permissions
        ActivityCompat = autoclass('androidx.core.app.ActivityCompat')
        Permission = autoclass('android.Manifest$permission')
        PackageManager = autoclass('android.content.pm.PackageManager')

        # Contacts
        ContactsContract = autoclass('android.provider.ContactsContract')

        # Telephony & System
        Settings = autoclass('android.provider.Settings')
        PowerManager = autoclass('android.os.PowerManager')
        PhoneNumberUtils = autoclass('android.telephony.PhoneNumberUtils')
        AlarmClock = autoclass('android.provider.AlarmClock')

        # Service & Notification
        Service = autoclass('android.app.Service')
        NotificationManager = autoclass('android.app.NotificationManager')
        NotificationChannel = autoclass('android.app.NotificationChannel')
        NotificationCompatBuilder = autoclass('androidx.core.app.NotificationCompat$Builder')
        PendingIntent = autoclass('android.app.PendingIntent')

        # UI
        WebView = autoclass('android.webkit.WebView')

        # TextToSpeech
        TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
        Locale = autoclass('java.util.Locale')

        # --- Permission Helpers ---
        def check_permission(perm):
            return ActivityCompat.checkSelfPermission(Activity.getCurrentActivity(), perm) == PackageManager.PERMISSION_GRANTED

        def request_permissions(permissions, callback):
            ActivityCompat.requestPermissions(Activity.getCurrentActivity(), permissions, 0)
            # Kivy/Pyjnius doesn't have a direct callback mechanism from requestPermissions,
            # so we check the result after a short delay.
            Clock.schedule_once(lambda dt: callback(permissions, [check_permission(perm) for perm in permissions]), 2)

    except Exception as e:
        # This will be logged once the logger is configured
        _android_error = f"Failed to load one or more Android classes: {e}"
        # Set all to None to allow the app to run and display an error
        Context = Activity = Intent = Uri = ActivityCompat = Permission = PackageManager = None
        ContactsContract = Settings = PowerManager = PhoneNumberUtils = AlarmClock = None
        Service = NotificationManager = NotificationChannel = NotificationCompatBuilder = None
        PendingIntent = WebView = TextToSpeech = Locale = None
        check_permission = request_permissions = None
else:
    _android_error = "Not running on Android platform."
    # Define dummy classes for desktop testing
    Context = Activity = Intent = Uri = ActivityCompat = Permission = PackageManager = None
    ContactsContract = Settings = PowerManager = PhoneNumberUtils = AlarmClock = None
    Service = NotificationManager = NotificationChannel = NotificationCompatBuilder = None
    PendingIntent = WebView = TextToSpeech = Locale = None
    check_permission = request_permissions = None

# Optional dependencies
try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    Model = KaldiRecognizer = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

logger = logging.getLogger(__name__)

# --- Text-to-Speech Manager ---
class TTSManager(PythonJavaClass):
    __javainterfaces__ = ['android.speech.tts.TextToSpeech$OnInitListener']

    def __init__(self):
        super().__init__()
        self.tts = None
        self.ready = False
        if Activity and TextToSpeech:
            try:
                self.tts = TextToSpeech(Activity.getCurrentActivity(), self)
            except Exception as e:
                logger.error(f"TTS Initialization failed: {e}")

    @java_method('(I)V')
    def onInit(self, status):
        if status == TextToSpeech.SUCCESS:
            self.ready = True
            logger.info("TTS initialized successfully.")
            try:
                self.tts.setLanguage(Locale.US)
            except Exception as e:
                logger.error(f"TTS setLanguage failed: {e}")
        else:
            self.ready = False
            logger.error("TTS initialization failed with status: " + str(status))

    def speak(self, text):
        if self.ready and self.tts:
            logger.info(f"TTS Speaking: {text}")
            self.tts.speak(text, TextToSpeech.QUEUE_FLUSH, None, None)
        else:
            logger.warning("TTS not ready, cannot speak.")

    def shutdown(self):
        if self.tts:
            self.tts.shutdown()
            self.ready = False

# --- WebView to Python Bridge ---
class KalaBridge(PythonJavaClass):
    __javainterfaces__ = ['KalaBridgeInterface']
    __javacontext__ = 'app'

    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance

    @java_method('(Z)V')
    def toggleListening(self, is_listening):
        Clock.schedule_once(lambda dt: self.app.toggle_listening_from_web(is_listening))

    @java_method('(Ljava/lang/String;)V')
    def selectContact(self, contact_name):
        # When a user clicks a contact from the disambiguation list in the WebView
        Clock.schedule_once(lambda dt: self.app.process_transcript(f"{WAKE_PHRASE} call {contact_name}"))

    @java_method('()Z')
    def getListeningState(self):
        return self.app.is_listening

    @java_method('()V')
    def retryInitialization(self):
        Clock.schedule_once(self.app.request_permissions_and_init, 0)

# --- Background Service ---
class KalaService(Service):
    def __init__(self):
        super().__init__()
        self.recognizer = None
        self.pyaudio_instance = None
        self.stream = None
        self.is_listening = False
        self.stop_event = threading.Event()
        self.model = None

    def onStartCommand(self, intent, flags, startId):
        logger.info("KalaService started")
        self.start_foreground_service()
        threading.Thread(target=self.initialize_vosk).start()
        return Service.START_STICKY

    def start_foreground_service(self):
        if not all([NotificationManager, NotificationChannel, NotificationCompatBuilder, PendingIntent, Activity, Context]):
            logger.error("Cannot start foreground service, Android classes unavailable.")
            return

        try:
            channel_id = "kala_service_channel"
            channel = NotificationChannel(channel_id, "Kala Voice Service", NotificationManager.IMPORTANCE_LOW)
            notification_manager = self.getSystemService(Context.NOTIFICATION_SERVICE)
            notification_manager.createNotificationChannel(channel)

            intent = Intent(self, Activity.getCurrentActivity().__class__)
            pending_intent = PendingIntent.getActivity(self, 0, intent, PendingIntent.FLAG_IMMUTABLE)

            notification_builder = NotificationCompatBuilder(self, channel_id)
            notification_builder.setContentTitle(f"{APP_NAME} Voice Assistant")
            notification_builder.setContentText(f"Listening for '{WAKE_PHRASE}'...")
            # Dynamically find the launcher icon
            icon_id = self.getResources().getIdentifier("ic_launcher", "mipmap", self.getPackageName())
            notification_builder.setSmallIcon(icon_id)
            notification_builder.setContentIntent(pending_intent)
            notification_builder.setOngoing(True)

            self.startForeground(1, notification_builder.build())
            logger.info("Foreground service started successfully.")
        except Exception as e:
            logger.error(f"Failed to start foreground service: {e}")

    def initialize_vosk(self):
        if not all([Model, KaldiRecognizer, pyaudio]):
            logger.error("Vosk or PyAudio module unavailable. Service cannot start.")
            return

        # Use the app's internal files directory for the model
        app_files_dir = Activity.getCurrentActivity().getFilesDir().getAbsolutePath()
        model_path = os.path.join(app_files_dir, "vosk-model-small-en-us")

        if not os.path.exists(model_path):
            logger.error(f"Vosk model not found at {model_path}. Please ensure it's copied by the main app first.")
            return

        try:
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            logger.info("Service: Vosk recognizer initialized.")
        except Exception as e:
            logger.error(f"Service: Vosk initialization failed: {e}")
            return

        self.initialize_audio()

    def initialize_audio(self):
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8192
            )
            self.stream.start_stream()
            logger.info("Service: PyAudio stream opened.")
            self.is_listening = True
            self.stop_event.clear()
            self.listen_loop()
        except Exception as e:
            logger.error(f"Service: PyAudio initialization failed: {e}")
            self.cleanup()

    def listen_loop(self):
        logger.info("Service: Starting wake word listening loop.")
        while self.is_listening and not self.stop_event.is_set():
            if check_permission and not check_permission(Permission.RECORD_AUDIO):
                logger.error("Service: RECORD_AUDIO permission revoked. Stopping.")
                break
            try:
                data = self.stream.read(4096, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    transcript = result.get('text', '').lower().strip()
                    if WAKE_PHRASE in transcript:
                        logger.info(f"Service: Wake phrase detected: '{transcript}'")
                        self.launch_app()
            except IOError as e:
                logger.error(f"Service: Audio read error: {e}. Reinitializing audio.")
                self.reinitialize_audio() # Attempt to recover from audio stream error
            except Exception as e:
                logger.error(f"Service: Unexpected error in listen_loop: {e}")
                break
        self.cleanup()

    def reinitialize_audio(self):
        self.cleanup()
        time.sleep(2) # Wait before retrying
        self.initialize_audio()

    def launch_app(self):
        if not all([Intent, Activity]):
            logger.error("Service: Cannot launch app, Android classes unavailable.")
            return
        try:
            package_name = Activity.getCurrentActivity().getPackageName()
            launch_intent = Activity.getCurrentActivity().getPackageManager().getLaunchIntentForPackage(package_name)
            if launch_intent:
                launch_intent.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
                self.startActivity(launch_intent)
                logger.info("Service: Launched main app activity.")
        except Exception as e:
            logger.error(f"Service: Failed to launch app: {e}")

    def cleanup(self):
        self.is_listening = False
        self.stop_event.set()
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Service: Stream cleanup error: {e}")
            self.stream = None
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                logger.error(f"Service: PyAudio cleanup error: {e}")
            self.pyaudio_instance = None

    def onDestroy(self):
        self.cleanup()
        logger.info("KalaService destroyed.")

# --- Main Kivy App ---
class KalaApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Thread lock for safe state management
        self._lock = threading.Lock()
        self.required_permissions = []
        if Permission:
             self.required_permissions = [
                Permission.RECORD_AUDIO,
                Permission.READ_CONTACTS,
                Permission.CALL_PHONE
            ]

        # App state
        self.recognizer = None
        self.pyaudio_instance = None
        self.stream = None
        self.model = None
        self.is_listening = False
        self.listening_thread = None
        self.stop_event = threading.Event()
        self.listen_timeout = 20  # Seconds of listening after activation

        # UI and helpers
        self.webview = None
        self.fallback_label = None
        self.pending_contact_matches = []
        self.tts_manager = None
        self.command_context = {}

    def build(self):
        self.root = BoxLayout(orientation='vertical')
        self.fallback_label = Label(text="Initializing Kala AI...")
        self.root.add_widget(self.fallback_label)

        # Defer initialization until the app is fully built
        Clock.schedule_once(self.post_build_init, 1)
        return self.root

    def post_build_init(self, dt):
        self.setup_logging()
        logger.info(f"Starting {APP_NAME} App")

        if platform != 'android' or _android_error:
            self.handle_critical_error(f"Unsupported Platform or Error: {_android_error}")
            return
        
        self.tts_manager = TTSManager()
        self.request_permissions_and_init()

    def setup_logging(self):
        log_file = os.path.join(self.user_data_dir, 'kala.log')
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[handler, logging.StreamHandler()]
        )
        logger.info("Logging configured.")

    def handle_critical_error(self, message):
        logger.error(message)
        self.update_web_ui('Status: CRITICAL ERROR', 'Transcript: -', f'Action: {message}')
        if self.tts_manager:
            self.tts_manager.speak(f"A critical error occurred: {message}")

    def request_permissions_and_init(self, *args):
        if not request_permissions:
            self.handle_critical_error("Permission module not loaded.")
            return

        with self._lock:
            self.update_web_ui('Status: Requesting permissions...', 'Transcript: -', 'Action: Please grant access.')
            request_permissions(self.required_permissions, self.on_permission_result)

    def on_permission_result(self, permissions, grant_results):
        denied_permissions = [p.split('.')[-1] for p, g in zip(permissions, grant_results) if not g]
        if denied_permissions:
            msg = f'Permissions denied: {", ".join(denied_permissions)}'
            logger.error(msg)
            self.update_web_ui('Status: Permissions Required', 'Transcript: -', f'Action: {msg}')
            self.speak_and_log(f"I need {', '.join(denied_permissions)} permissions to function.")
        else:
            logger.info("All required permissions granted.")
            # Initialize everything in order
            Clock.schedule_once(self.setup_webview)
            Clock.schedule_once(lambda dt: threading.Thread(target=self.initialize_vosk).start())
            Clock.schedule_once(self.start_background_service)
            Clock.schedule_interval(self.check_battery_optimization, 300) # Check every 5 mins

    def setup_webview(self, dt):
        html_path = os.path.join(os.path.dirname(__file__), 'kala.html')
        if not os.path.exists(html_path):
            self.handle_critical_error("kala.html not found.")
            return

        try:
            activity = Activity.getCurrentActivity()
            self.webview = WebView(activity)
            self.webview.getSettings().setJavaScriptEnabled(True)
            self.webview.getSettings().setMediaPlaybackRequiresUserGesture(False) # For audio cues
            self.kala_bridge = KalaBridge(self)
            self.webview.addJavascriptInterface(self.kala_bridge, 'KalaBridge')
            self.webview.loadUrl(f'file://{html_path}')
            activity.setContentView(self.webview)
            logger.info("WebView initialized successfully.")
            if self.fallback_label:
                # Fallback label is part of the Kivy tree, which is replaced by setContentView.
                # No need to remove it explicitly.
                self.fallback_label = None
        except Exception as e:
            self.handle_critical_error(f"WebView setup failed: {e}")

    def initialize_vosk(self):
        if not all([Model, KaldiRecognizer, pyaudio]):
            self.handle_critical_error("Vosk or PyAudio module unavailable.")
            return

        model_path = os.path.join(self.user_data_dir, "vosk-model-small-en-us")
        src_model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us")

        # Copy model if it doesn't exist in user data directory
        if not os.path.exists(model_path) and os.path.exists(src_model_path):
            try:
                logger.info("Copying Vosk model to user data directory...")
                shutil.copytree(src_model_path, model_path)
                logger.info(f"Copied Vosk model to {model_path}")
            except Exception as e:
                self.handle_critical_error(f"Failed to copy Vosk model: {e}")
                return
        
        if not os.path.exists(model_path):
            self.handle_critical_error(f"Vosk model not found at {src_model_path}")
            return
        
        try:
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8192
            )
            self.stream.stop_stream() # Start in a stopped state
            logger.info("Vosk and PyAudio initialized successfully for the main app.")
            Clock.schedule_once(lambda dt: self.update_web_ui('Status: Ready', 'Transcript: -', f'Action: Say "{WAKE_PHRASE}" or tap the button.'))
            Clock.schedule_once(lambda dt: self.speak_and_log("I'm ready."))
        except Exception as e:
            self.handle_critical_error(f"Vosk/PyAudio init failed: {e}")

    def toggle_listening_from_web(self, should_listen):
        with self._lock:
            if not all([self.recognizer, self.stream]):
                self.update_web_ui('Status: Not Initialized', 'Transcript: -', 'Action: Please restart the app.')
                return

            if should_listen and not self.is_listening:
                self.is_listening = True
                self.stop_event.clear()
                self.listening_thread = threading.Thread(target=self.listen_loop)
                self.listening_thread.daemon = True
                self.stream.start_stream()
                self.listening_thread.start()
                self.update_web_ui('Status: Listening...', 'Transcript: -', 'Action: Please speak your command.')

            elif not should_listen and self.is_listening:
                self.is_listening = False
                self.stop_event.set()
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.update_web_ui('Status: Ready', 'Transcript: -', 'Action: Listening stopped.')
    
    def listen_loop(self):
        start_time = time.time()
        logger.info("Starting in-app listening loop.")
        while self.is_listening and not self.stop_event.is_set():
            if (time.time() - start_time) > self.listen_timeout:
                logger.info("Listening timed out.")
                Clock.schedule_once(lambda dt: self.reset_button_ui())
                break
            try:
                data = self.stream.read(4096, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    transcript = result.get('text', '').lower().strip()
                    if transcript:
                        logger.info(f"Recognized transcript: {transcript}")
                        Clock.schedule_once(lambda dt, t=transcript: self.process_transcript(t))
                        # Reset timeout after recognition
                        start_time = time.time()
            except Exception as e:
                logger.error(f"Listening error: {e}")
                Clock.schedule_once(lambda dt: self.reset_button_ui())
                break
        
        # Ensure cleanup happens when loop exits
        with self._lock:
            if self.is_listening:
                 self.is_listening = False
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
        logger.info("Exited in-app listening loop.")

    def reset_button_ui(self):
        with self._lock:
            if self.is_listening:
                self.toggle_listening_from_web(False)
            self.update_web_ui('Status: Ready', 'Transcript: -', 'Action: Timed out. Tap button to retry.')
            if self.webview:
                self.webview.evaluateJavascript('setListeningState(false);', None)

    def process_transcript(self, transcript):
        self.update_web_ui('Status: Processing...', f'Transcript: {transcript}', 'Action: Thinking...')

        # If there's a pending question, handle the answer
        if self.command_context.get("pending_question") == "confirm_contact":
            # This logic can be expanded for other pending questions
            # For now, we assume the transcript is the selected contact name.
            self.command_context.clear() # Clear context
            transcript = f"{WAKE_PHRASE} call {transcript}" # Reconstruct command

        # Check for wake phrase only if not in an active context
        if not self.command_context and not transcript.startswith(WAKE_PHRASE):
            self.update_web_ui('Status: Ready', f'Transcript: {transcript}', f'Action: Please start with "{WAKE_PHRASE}".')
            return

        command = transcript[len(WAKE_PHRASE):].strip() if transcript.startswith(WAKE_PHRASE) else transcript
        self.parse_command(command)
        Clock.schedule_once(lambda dt: self.reset_button_ui())

    def parse_command(self, command):
        # More robust command parsing using regex
        patterns = {
            'call': re.compile(r'call (.+)', re.IGNORECASE),
            'text': re.compile(r'(text|message) (.+?) (that|to say|saying) (.+)', re.IGNORECASE),
            'open_app': re.compile(r'open (.+)', re.IGNORECASE),
            'set_alarm': re.compile(r'set an alarm for (.+)', re.IGNORECASE),
            'set_timer': re.compile(r'set a timer for (.+)', re.IGNORECASE),
            'search': re.compile(r'search for (.+)', re.IGNORECASE),
        }
        
        match = None
        intent = None

        for name, pattern in patterns.items():
            match = pattern.match(command)
            if match:
                intent = name
                break
        
        if not intent:
            self.speak_and_log(f"Sorry, I didn't understand the command: {command}")
            self.update_web_ui('Status: Ready', f'Transcript: {command}', 'Action: Command not recognized.')
            return

        # --- Execute Actions Based on Intent ---
        if intent == 'call':
            self.handle_call(match.group(1))
        elif intent == 'text':
            self.handle_text(contact_name=match.group(2), message=match.group(4))
        elif intent == 'open_app':
            self.handle_open_app(match.group(1))
        elif intent == 'set_alarm':
            self.handle_set_alarm(match.group(1))
        elif intent == 'set_timer':
            self.handle_set_timer(match.group(1))
        elif intent == 'search':
            self.handle_search(match.group(1))

    # --- Action Handlers ---
    def handle_call(self, contact_name):
        phone_number, name = self.get_contact_info(contact_name)
        if phone_number:
            self.speak_and_log(f"Calling {name} now.")
            self.update_web_ui('Status: Action', f'Transcript: Call {name}', f'Action: Initiating call to {phone_number}')
            self.make_call(phone_number)
    
    def handle_text(self, contact_name, message):
        phone_number, name = self.get_contact_info(contact_name)
        if phone_number:
            self.speak_and_log(f"Sending text to {name}: {message}")
            self.update_web_ui('Status: Action', f'Transcript: Text {name}', f'Action: Sending SMS.')
            self.send_sms(phone_number, message)

    def handle_open_app(self, app_name):
        if not all([Activity, Intent]): return
        self.speak_and_log(f"Opening {app_name}")
        self.update_web_ui('Status: Action', f'Transcript: Open {app_name}', 'Action: Launching application...')
        try:
            pm = Activity.getCurrentActivity().getPackageManager()
            intent = pm.getLaunchIntentForPackage(app_name.lower()) # Simple guess
            if not intent: # More robust search needed in a real app
                # This is a placeholder; a real implementation would query all apps
                # and match against their labels.
                raise Exception(f"Could not find a package for '{app_name}'.")
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            Activity.getCurrentActivity().startActivity(intent)
        except Exception as e:
            self.speak_and_log(f"I couldn't open {app_name}.")
            self.update_web_ui('Status: Error', f'Transcript: Open {app_name}', f'Action: {e}')

    def handle_set_alarm(self, time_str):
        if not all([Activity, Intent, AlarmClock]): return
        # Simple parsing for "7:30 AM", "8 PM", etc.
        match = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_str, re.IGNORECASE)
        if not match:
            self.speak_and_log("Sorry, I didn't understand that time format for the alarm.")
            return

        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = (match.group(3) or "").lower()

        if ampm == 'pm' and hour < 12:
            hour += 12
        if ampm == 'am' and hour == 12: # 12 AM is hour 0
            hour = 0
        
        self.speak_and_log(f"Setting an alarm for {time_str}.")
        self.update_web_ui('Status: Action', f'Transcript: Set alarm for {time_str}', 'Action: Creating alarm...')
        try:
            intent = Intent(AlarmClock.ACTION_SET_ALARM)
            intent.putExtra(AlarmClock.EXTRA_HOUR, hour)
            intent.putExtra(AlarmClock.EXTRA_MINUTES, minute)
            intent.putExtra(AlarmClock.EXTRA_MESSAGE, f"{APP_NAME} Alarm")
            Activity.getCurrentActivity().startActivity(intent)
        except Exception as e:
             self.speak_and_log(f"I couldn't set the alarm. Error: {e}")

    def handle_set_timer(self, timer_str):
        if not all([Activity, Intent, AlarmClock]): return
        seconds = 0
        minutes_match = re.search(r'(\d+)\s*minute', timer_str, re.IGNORECASE)
        seconds_match = re.search(r'(\d+)\s*second', timer_str, re.IGNORECASE)
        if minutes_match:
            seconds += int(minutes_match.group(1)) * 60
        if seconds_match:
            seconds += int(seconds_match.group(1))

        if seconds == 0:
            self.speak_and_log("Sorry, I didn't understand the timer duration.")
            return

        self.speak_and_log(f"Setting a timer for {timer_str}.")
        self.update_web_ui('Status: Action', f'Transcript: Set timer for {timer_str}', 'Action: Creating timer...')
        try:
            intent = Intent(AlarmClock.ACTION_SET_TIMER)
            intent.putExtra(AlarmClock.EXTRA_LENGTH, seconds)
            intent.putExtra(AlarmClock.EXTRA_MESSAGE, f"{APP_NAME} Timer")
            Activity.getCurrentActivity().startActivity(intent)
        except Exception as e:
            self.speak_and_log(f"I couldn't set the timer. Error: {e}")

    def handle_search(self, query):
        if not all([Activity, Intent, Uri]): return
        self.speak_and_log(f"Here are the web results for {query}")
        self.update_web_ui('Status: Action', f'Transcript: Search for {query}', 'Action: Opening browser...')
        try:
            intent = Intent(Intent.ACTION_VIEW, Uri.parse(f"https://www.google.com/search?q={query}"))
            Activity.getCurrentActivity().startActivity(intent)
        except Exception as e:
            self.speak_and_log(f"I couldn't perform the search. Error: {e}")

    # --- Utility Methods ---
    def get_contact_info(self, contact_name):
        if not check_permission(Permission.READ_CONTACTS):
            self.speak_and_log("I need permission to read your contacts.")
            return None, None
        
        if not all([ContactsContract, Activity]): return None, None
        
        all_contacts = {}
        cursor = None
        try:
            resolver = Activity.getCurrentActivity().getContentResolver()
            cursor = resolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                None, None, None, None
            )
            while cursor and cursor.moveToNext():
                name_idx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                phone_idx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                name = cursor.getString(name_idx)
                phone = cursor.getString(phone_idx)
                if name and phone:
                    all_contacts[name.lower()] = (name, phone)
        except Exception as e:
            logger.error(f"Contact query error: {e}")
            return None, None
        finally:
            if cursor:
                cursor.close()

        matches = get_close_matches(contact_name.lower(), all_contacts.keys(), n=5, cutoff=0.7)
        
        if not matches:
            self.speak_and_log(f"I couldn't find a contact named {contact_name}.")
            return None, None
        
        if len(matches) > 1:
            matched_names = [all_contacts[m][0] for m in matches]
            self.speak_and_log(f"I found a few people named {contact_name}. Which one did you mean?")
            self.command_context["pending_question"] = "confirm_contact"
            # Show options in WebView
            if self.webview:
                js_code = f'showContactMatches({json.dumps(matched_names)});'
                self.webview.evaluateJavascript(js_code, None)
            return None, None
        
        # Exact match
        original_name, phone_number = all_contacts[matches[0]]
        return phone_number, original_name

    def make_call(self, phone_number):
        if not check_permission(Permission.CALL_PHONE):
            self.speak_and_log("I need permission to make phone calls.")
            return
        if self.is_airplane_mode_on():
            self.speak_and_log("I can't make a call, airplane mode is on.")
            return
        try:
            intent = Intent(Intent.ACTION_CALL)
            intent.setData(Uri.parse(f'tel:{phone_number}'))
            Activity.getCurrentActivity().startActivity(intent)
        except Exception as e:
            logger.error(f"Call initiation error: {e}")
            self.speak_and_log(f"Sorry, I couldn't make the call. Error: {e}")
    
    def send_sms(self, phone_number, message):
        try:
            uri = Uri.parse(f"smsto:{phone_number}")
            intent = Intent(Intent.ACTION_SENDTO, uri)
            intent.putExtra("sms_body", message)
            Activity.getCurrentActivity().startActivity(intent)
        except Exception as e:
            logger.error(f"SMS initiation error: {e}")
            self.speak_and_log(f"Sorry, I couldn't send the text. Error: {e}")

    def is_airplane_mode_on(self):
        try:
            resolver = Activity.getCurrentActivity().getContentResolver()
            return Settings.Global.getInt(resolver, 'airplane_mode_on') == 1
        except Exception:
            return False # Assume off if check fails

    def check_battery_optimization(self, dt):
        if not all([Settings, Activity, PowerManager, Context]): return
        try:
            pm = Activity.getCurrentActivity().getSystemService(Context.POWER_SERVICE)
            package_name = Activity.getCurrentActivity().getPackageName()
            if not pm.isIgnoringBatteryOptimizations(package_name):
                logger.warning("Battery optimization is enabled, background service may be killed.")
                self.speak_and_log("For me to work reliably, please disable battery optimization for this app.")
                intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
                intent.setData(Uri.parse(f"package:{package_name}"))
                Activity.getCurrentActivity().startActivity(intent)
        except Exception as e:
            logger.error(f"Failed to check battery optimization: {e}")

    def start_background_service(self, dt):
        if not all([Service, Intent, Activity]): return
        try:
            intent = Intent(Activity.getCurrentActivity(), KalaService)
            Activity.getCurrentActivity().startService(intent)
            logger.info("Started KalaService in the background.")
        except Exception as e:
            logger.error(f"Failed to start background service: {e}")

    def update_web_ui(self, status, transcript, action):
        if self.fallback_label:
            self.fallback_label.text = f"{status}\n{transcript}\n{action}"
        if self.webview:
            # Escape strings for JavaScript
            status_js = json.dumps(status)
            transcript_js = json.dumps(transcript)
            action_js = json.dumps(action)
            js_code = f'updateUI({status_js}, {transcript_js}, {action_js});'
            self.webview.evaluateJavascript(js_code, None)

    def speak_and_log(self, text):
        logger.info(text)
        if self.tts_manager:
            self.tts_manager.speak(text)
    
    # --- App Lifecycle Methods ---
    def on_pause(self):
        # Stop in-app listening when app is paused to save resources
        if self.is_listening:
            self.toggle_listening_from_web(False)
        return True

    def on_resume(self):
        # Can add logic here to re-check permissions or state if needed
        pass

    def on_stop(self):
        logger.info("App stopping. Cleaning up resources.")
        if self.tts_manager:
            self.tts_manager.shutdown()
        
        with self._lock:
            self.is_listening = False
            self.stop_event.set()
            if self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()

if __name__ == '__main__':
    try:
        KalaApp().run()
    except Exception as e:
        # This top-level catch is a fallback. Logging should be configured by now.
        if 'logger' in globals():
             logger.critical(f"App crashed with unhandled exception: {e}", exc_info=True)
        else:
            print(f"App crashed before logger was configured: {e}")