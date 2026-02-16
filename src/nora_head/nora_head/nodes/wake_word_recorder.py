#!/usr/bin/env python3
# Copyright 2026 nanostation
# Licensed under the Apache License, Version 2.0

"""
Wake Word Detection + Audio Recording Node for Nora
Listens for wake word, records audio until silence, saves to disk

Hardware: reSpeaker XMOS XVF3800 (4-mic array)
Wake word: "Alexa" (temporary - will be "Nora" in Story 7)
"""

import os
import time
import uuid
import queue
import threading
import wave
import numpy as np
from collections import deque
import sounddevice as sd
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from nora_head.msg import RobotState

# Try to import openwakeword
try:
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    print("WARNING: openwakeword not installed. Wake word detection disabled.")


class WakeWordRecorderNode(Node):
    def __init__(self):
        super().__init__('wake_word_recorder')
        
        # Declare parameters
        self.declare_parameter('input_device', 'hw:0,0')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 2)
        self.declare_parameter('frame_ms', 40)
        self.declare_parameter('model_path', '/home/nanostation/models/alexa_v0.1.tflite')
        self.declare_parameter('save_dir', '/home/nanostation/nora_data/audio_recordings')
        self.declare_parameter('detection_threshold', 0.6)
        self.declare_parameter('silence_threshold', 350.0)
        self.declare_parameter('silence_duration', 1.0)
        self.declare_parameter('max_record_seconds', 8.0)
        self.declare_parameter('min_record_seconds', 0.8)
        self.declare_parameter('cooldown_seconds', 2.5)
        
        # Get parameters
        self.input_device = self.get_parameter('input_device').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.frame_ms = self.get_parameter('frame_ms').value
        self.model_path = self.get_parameter('model_path').value
        self.save_dir = self.get_parameter('save_dir').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.silence_threshold = self.get_parameter('silence_threshold').value
        self.silence_duration = self.get_parameter('silence_duration').value
        self.max_record_seconds = self.get_parameter('max_record_seconds').value
        self.min_record_seconds = self.get_parameter('min_record_seconds').value
        self.cooldown_seconds = self.get_parameter('cooldown_seconds').value
        
        # Computed parameters
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000.0)
        self.buffer_frames = 1
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Publishers
        self.wake_word_pub = self.create_publisher(Bool, '/wake_word/detected', 10)
        self.recording_pub = self.create_publisher(String, '/wake_word/recording_path', 10)
        
        # State machine
        self.STATE_IDLE = 0
        self.STATE_ACTIVE = 1
        self.state = self.STATE_IDLE
        self.last_active_time = 0
        self.hysteresis_ms = 200
        
        # Recording state
        self.recording = False
        self.record_frames = []
        self.record_start_time = None
        self.silence_start = None
        self.last_trigger_time = 0
        
        # Audio buffers
        self.buffer = deque(maxlen=self.buffer_frames)
        self.audio_queue = queue.Queue(maxsize=3)
        
        # Load wake word model
        if OPENWAKEWORD_AVAILABLE:
            try:
                self.model = Model(
                    wakeword_models=[self.model_path],
                    enable_speex_noise_suppression=False,
                    vad_threshold=0.5
                )
                self.get_logger().info(f'Wake word model loaded: {self.model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')
                self.model = None
        else:
            self.model = None
            self.get_logger().warn('Wake word detection disabled (openwakeword not available)')
        
        # Start audio stream
        self.start_audio_stream()
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()
        
        self.get_logger().info('Wake word recorder initialized')
        self.get_logger().info(f'Listening on {self.input_device} @ {self.sample_rate}Hz')
        self.get_logger().info(f'Recordings saved to: {self.save_dir}')
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass
    
    def start_audio_stream(self):
        """Start audio input stream"""
        try:
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                dtype='int16',
                callback=self.audio_callback
            )
            self.stream.start()
            self.get_logger().info('Audio stream started')
        except Exception as e:
            self.get_logger().error(f'Failed to start audio stream: {e}')
            self.stream = None
    
    def compute_rms_energy(self, frame):
        """Compute RMS energy of audio frame"""
        frame_float = frame.astype(np.float32)
        return np.sqrt(np.mean(frame_float * frame_float))
    
    def save_wav(self, frames):
        """Save recorded frames to WAV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}__{unique_id}.wav"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)  # int16
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            self.get_logger().info(f'Saved recording: {filename}')
            
            # Publish recording path
            msg = String()
            msg.data = filepath
            self.recording_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to save WAV: {e}')
    
    def stop_recording(self):
        """Stop recording and save if valid"""
        duration = time.time() - self.record_start_time if self.record_start_time else 0
        
        if duration >= self.min_record_seconds:
            self.save_wav(self.record_frames)
        else:
            self.get_logger().info(f'Discarded recording (too short: {duration:.2f}s)')
        
        self.recording = False
        self.record_frames = []
        self.record_start_time = None
        self.silence_start = None
        self.last_trigger_time = time.time()
    
    def inference_loop(self):
        """Main inference + recording loop"""
        if self.model is None:
            self.get_logger().warn('Inference loop disabled (no model)')
            return
        
        self.get_logger().info('Inference loop started. Listening for wake word...')
        
        while rclpy.ok():
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            self.buffer.append(audio_chunk)
            
            if len(self.buffer) == self.buffer_frames:
                audio_frame = np.concatenate(list(self.buffer), axis=0)
                mono = np.mean(audio_frame, axis=1)
                pcm = mono.astype(np.int16)
                
                # Wake word detection
                prediction = self.model.predict(pcm)
                now = time.time() * 1000
                detected = any(score > self.detection_threshold for score in prediction.values())
                
                # State machine
                if self.state == self.STATE_IDLE:
                    if detected and (time.time() - self.last_trigger_time) > self.cooldown_seconds:
                        self.get_logger().info('🔥 Wake word detected!')
                        
                        # Publish wake word event
                        wake_msg = Bool()
                        wake_msg.data = True
                        self.wake_word_pub.publish(wake_msg)
                        
                        self.state = self.STATE_ACTIVE
                        self.last_active_time = now
                        self.last_trigger_time = time.time()
                        
                        # Start recording
                        self.recording = True
                        self.record_frames = []
                        self.record_start_time = time.time()
                        self.silence_start = None
                
                elif self.state == self.STATE_ACTIVE:
                    if detected:
                        self.last_active_time = now
                    else:
                        if now - self.last_active_time > self.hysteresis_ms:
                            self.state = self.STATE_IDLE
                
                # Recording logic
                if self.recording and self.record_start_time is not None:
                    self.record_frames.append(pcm.tobytes())
                    energy = self.compute_rms_energy(pcm)
                    
                    # Silence detection
                    if energy < self.silence_threshold:
                        if self.silence_start is None:
                            self.silence_start = time.time()
                        elif time.time() - self.silence_start >= self.silence_duration:
                            self.stop_recording()
                            continue
                    else:
                        self.silence_start = None
                    
                    # Max duration
                    if time.time() - self.record_start_time >= self.max_record_seconds:
                        self.stop_recording()
                        continue
            
            self.audio_queue.task_done()
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            self.get_logger().info('Audio stream closed')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WakeWordRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
