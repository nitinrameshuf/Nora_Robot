#!/usr/bin/env python3
# Copyright 2026 nanostation
# Licensed under the Apache License, Version 2.0

"""
Audio Capture Node for Nora
Captures audio from ReSpeaker XVF3800 4-mic array
Publishes audio streams to ROS 2 for speech recognition
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
import sounddevice as sd
import numpy as np
import queue


class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture')
        
        # Declare parameters
        self.declare_parameter('device', 'hw:0,0')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 2)
        self.declare_parameter('chunk_duration', 0.1)
        self.declare_parameter('publish_rate', 10.0)
        
        # Get parameters
        self.device = self.get_parameter('device').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.chunk_duration = self.get_parameter('chunk_duration').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Calculate chunk size
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Publisher for audio stream
        self.audio_pub = self.create_publisher(
            UInt8MultiArray,
            '/audio/stream',
            10
        )
        
        # Audio buffer queue
        self.audio_queue = queue.Queue()
        
        # Verify device
        self.verify_device()
        
        # Audio stream
        self.stream = None
        self.start_audio_stream()
        
        # Publisher timer
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_audio
        )
        
        self.publish_count = 0
        
        self.get_logger().info(f'Audio capture started: {self.device}')
        self.get_logger().info(f'Sample rate: {self.sample_rate} Hz, Channels: {self.channels}')
        self.get_logger().info(f'Publishing at {self.publish_rate} Hz')
    
    def verify_device(self):
        """Verify ReSpeaker is available"""
        try:
            devices = sd.query_devices()
            found = False
            
            for idx, device in enumerate(devices):
                if 'reSpeaker XVF3800' in device['name']:
                    found = True
                    self.get_logger().info(f'Found ReSpeaker at device {idx}: {device["name"]}')
                    break
            
            if not found:
                self.get_logger().warn('ReSpeaker XVF3800 not found')
        
        except Exception as e:
            self.get_logger().error(f'Error querying devices: {e}')
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback"""
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        
        self.audio_queue.put(indata.copy())
    
    def start_audio_stream(self):
        """Start audio input stream"""
        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype='int16',
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.get_logger().info('Audio stream started')
        
        except Exception as e:
            self.get_logger().error(f'Failed to start stream: {e}')
            raise
    
    def publish_audio(self):
        """Publish audio chunks"""
        if self.audio_queue.empty():
            return
        
        try:
            # Get audio chunk
            audio_chunk = self.audio_queue.get_nowait()
            
            # Convert int16 numpy array to bytes, then to list of uint8
            audio_bytes = audio_chunk.tobytes()
            
            # Create message
            msg = UInt8MultiArray()
            msg.data = list(audio_bytes)  # Convert bytes to list of uint8
            
            # Publish
            self.audio_pub.publish(msg)
            
            self.publish_count += 1
            
            if self.publish_count % 100 == 0:
                self.get_logger().info(f'Published {self.publish_count} audio chunks')
        
        except queue.Empty:
            pass
        except Exception as e:
            self.get_logger().error(f'Error publishing: {e}')
    
    def destroy_node(self):
        """Cleanup"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.get_logger().info('Audio stream stopped')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = AudioCaptureNode()
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
