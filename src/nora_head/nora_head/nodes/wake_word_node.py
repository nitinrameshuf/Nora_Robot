#!/usr/bin/env python3
# Copyright 2026 nanostation
# Licensed under the Apache License, Version 2.0

"""
Wake Word Detection Node for Nora
Detects wake word using openWakeWord (fully offline)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, Bool, String
from openwakeword.model import Model
import numpy as np


class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wake_word_node')
        
        # Declare parameters
        self.declare_parameter('model_name', 'hey_mycroft')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('chunk_size', 1280)  # 80ms at 16kHz
        
        # Get parameters
        self.model_name = self.get_parameter('model_name').value
        self.threshold = self.get_parameter('threshold').value
        self.chunk_size = self.get_parameter('chunk_size').value
        
        # Initialize openWakeWord
        try:
            self.get_logger().info(f'Loading openWakeWord: {self.model_name}')
            self.model = Model(wakeword_models=[self.model_name])
            self.get_logger().info('Model loaded successfully')
        
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
        
        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.int16)
        
        # Publishers
        self.wake_word_pub = self.create_publisher(Bool, '/wake_word/detected', 10)
        self.status_pub = self.create_publisher(String, '/wake_word/status', 10)
        
        # Subscribe to audio stream
        self.audio_sub = self.create_subscription(
            UInt8MultiArray,
            '/audio/stream',
            self.audio_callback,
            10
        )
        
        self.detection_count = 0
        self.frame_count = 0
        
        self.get_logger().info(f'Wake word node ready')
        self.get_logger().info(f'Listening for "{self.model_name}" (threshold: {self.threshold})')
    
    def audio_callback(self, msg):
        """Process incoming audio"""
        try:
            # Convert uint8 to int16
            audio_bytes = bytes(msg.data)
            audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Stereo to mono (channel 0)
            if len(audio_chunk) % 2 == 0:
                audio_mono = audio_chunk[::2]
            else:
                audio_mono = audio_chunk
            
            # Add to buffer
            self.audio_buffer = np.append(self.audio_buffer, audio_mono)
            
            # Process chunks
            while len(self.audio_buffer) >= self.chunk_size:
                chunk = self.audio_buffer[:self.chunk_size]
                self.audio_buffer = self.audio_buffer[self.chunk_size:]
                
                # Convert to float32 [-1, 1]
                chunk_float = chunk.astype(np.float32) / 32768.0
                
                # Run inference
                predictions = self.model.predict(chunk_float)
                
                self.frame_count += 1
                
                # Check threshold
                for model_name, score in predictions.items():
                    if score >= self.threshold:
                        self.on_wake_word_detected(model_name, score)
                
                if self.frame_count % 1000 == 0:
                    self.get_logger().debug(f'Processed {self.frame_count} chunks')
        
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
    
    def on_wake_word_detected(self, model_name, score):
        """Handle detection"""
        self.detection_count += 1
        
        self.get_logger().info(
            f'🎤 Wake word detected! '
            f'Model: {model_name}, Score: {score:.3f} ({self.detection_count})'
        )
        
        # Publish detection
        detection_msg = Bool()
        detection_msg.data = True
        self.wake_word_pub.publish(detection_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = f'Wake word "{model_name}" detected (score: {score:.3f})'
        self.status_pub.publish(status_msg)
    
    def destroy_node(self):
        self.get_logger().info('Wake word node shutting down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WakeWordNode()
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
