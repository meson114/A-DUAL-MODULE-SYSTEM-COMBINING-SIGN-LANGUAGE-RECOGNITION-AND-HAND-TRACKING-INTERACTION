"""
Module 4: Leap Motion to Unity Socket Bridge
Captures 3D skeletal data from Leap Motion controller, encodes coordinates into JSON,
and streams over TCP socket to Unity engine for low-latency virtual hand rendering.
"""
import socket
import json
import time

HOST = '127.0.0.1'
PORT = 8080

def stream_hand_skeleton_to_unity():
    print(f"[INFO] Initializing TCP Socket Streamer on {HOST}:{PORT}...")

if __name__ == "__main__":
    stream_hand_skeleton_to_unity()
