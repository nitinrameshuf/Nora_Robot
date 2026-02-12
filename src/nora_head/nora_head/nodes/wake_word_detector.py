#!/usr/bin/env python3
# Copyright 2026 nanostation
# Licensed under the Apache License, Version 2.0

"""
Wake Word Detection Node for Nora
Listens for "Hey Nora" / "Alexa" wake word using ReSpeaker XVF3800
Publishes detection events to /wake_word/detected
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
import sounddevice as sd
import numpy as np
import queue
import threading
from collections import deque
from openwakeword.model import Model
import time


class WakeWordDetectorNode(Node):
    def __init__(self):
        super().__init__('wake_word_detector')
        
        # Declare parameters
        self.declare_parameter('device', 'hw:0,0')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 2)
        self.declare_parameter('model_path', '/home/nanostation/models/alexa_v0.1.tflite')
        self.declare_parameter('detection_threshold', 0.6)
        self.declare_parameter('hysteresis_ms', 200)
        self.declare_parameter('frame_duration_ms', 40)
        
        # Get parameters
        self.device = self.get_parameter('device').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.model_path = self.get_parameter('model_path').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.hysteresis_ms = self.get_parameter('hysteresis_ms').value
        self.frame_duration_ms = self.get_parameter('frame_duration_ms').value
        
        # Audio settings
        self.frame_duration = self.frame_duration_ms / 1000.0
        self.frame_size = int(self.sample_rate * self.frame_duration)
        self.buffer_frames = 1
        
        # Audio buffer
        self.buffer = deque(maxlen=self.buffer_frames)
        self.audio_queue = queue.Queue(maxsize=3)
        
        # State machine
        self.STATE_IDLE = 0
        self.STATE_ACTIVE = 1
        self.state = self.STATE_IDLE
        self.last_active_time = 0
        
        # Publisher
        self.detection_pub = self.create_publisher(
            String,
            '/wake_word/detected',
            10
        )
        
        # Load wake word model
        self.get_logger().info(f'Loading wake word model: {self.model_path}')
        
        try:
            self.model = Model(
                wakeword_models=[self.model_path],
                enable_speex_noise_suppression=False,
                vad_threshold=0.5
            )
            self.get_logger().info('Wake word model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
        
        # Start inference thread
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        # Start audio stream
        self.get_logger().info(f'Opening audio device: {self.device}')
        
        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.frame_size,
                dtype='int16',
                callback=self._audio_callback
            )
            self.stream.start()
            self.get_logger().info(f'Listening for wake word at {self.sample_rate}Hz')
            self.get_logger().info(f'Detection threshold: {self.detection_threshold}')
        
        except Exception as e:
            self.get_logger().error(f'Failed to open audio device: {e}')
            raise
        
        # Statistics
        self.detection_count = 0
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback - runs in real-time"""
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass  # Drop frame if queue is full
    
    def _inference_loop(self):
        """Inference thread - processes audio and detects wake word"""
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            self.buffer.append(audio_chunk)
            
            if len(self.buffer) == self.buffer_frames:
                # Concatenate frames
                audio_frame = np.concatenate(list(self.buffer), axis=0)
                
                # Convert to mono
                mono = np.mean(audio_frame, axis=1)
                
                # Convert to int16 PCM
                pcm = mono.astype(np.int16)
                
                # Run prediction
                prediction = self.model.predict(pcm)
                
                now = time.time() * 1000  # ms
                detected = False
                
                # Check detection threshold
                for wakeword, score in prediction.items():
                    if score > self.detection_threshold:
                        detected = True
                        self.get_logger().debug(f'{wakeword}: {score:.3f}')
                        break
                
                # State machine
                if self.state == self.STATE_IDLE:
                    if detected:
                        self._publish_detection()
                        self.state = self.STATE_ACTIVE
                        self.last_active_time = now
                
                elif self.state == self.STATE_ACTIVE:
                    if detected:
                        self.last_active_time = now
                    else:
                        # Hysteresis: wait before going back to IDLE
                        if now - self.last_active_time > self.hysteresis_ms:
                            self.state = self.STATE_IDLE
            
            self.audio_queue.task_done()
    
    def _publish_detection(self):
        """Publish wake word detection event"""
        msg = String()
        msg.data = "nora"  # Using "Alexa" model but treating as "Nora"
        
        self.detection_pub.publish(msg)
        
        self.detection_count += 1
        self.get_logger().info(f'🔥 Wake word detected! (count: {self.detection_count})')
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        self.get_logger().info('Shutting down wake word detector...')
        
        self.running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if hasattr(self, 'inference_thread'):
            self.inference_thread.join(timeout=2.0)
        
        self.get_logger().info(f'Total detections: {self.detection_count}')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WakeWordDetectorNode()
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
