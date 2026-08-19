"""High-Performance Binary Network Bridge for PC <-> DGX Spark Streaming.

Implements zero-copy, ultra-low-latency binary serialization and TCP communication
for streaming sensorimotor closed-loop control between an external capture PC and
the DGX Spark unified compute engine.

Wire Format Specification:
- Frame Packet (PC -> Spark, ~3.4 KB):
  [4B Magic: PBSP] [8B Tick] [8B TimestampNs] [4B DtSeconds] [307B PrevControl] [3072B RGB Pixels]
- Action Packet (Spark -> PC, self-describing length header):
  [4B Magic: PBSA] [8B Tick] [8B SparkInferenceNs] [4B LogitsLen] [4B CtrlLen] [LogitsPayload] [CtrlPayload]
"""

from __future__ import annotations

import socket
import struct
import time
from typing import NamedTuple
import numpy as np
import torch

MAGIC_FRAME = b"PBSP"
MAGIC_ACTION = b"PBSA"

HEADER_FRAME_STRUCT = struct.Struct("!4sQQf")
HEADER_ACTION_STRUCT = struct.Struct("!4sQQII")


class SparkTelemetry(NamedTuple):
    tick_id: int
    t_capture_ns: int
    t_send_ns: int
    t_spark_start_ns: int
    t_spark_end_ns: int
    t_recv_ns: int
    t_dispatch_ns: int
    total_roundtrip_ms: float
    spark_inference_ms: float
    network_transfer_ms: float


class SparkInferenceServer:
    """Zero-overhead streaming inference server running on the DGX Spark."""

    def __init__(
        self,
        model: torch.nn.Module,
        host: str = "0.0.0.0",
        port: int = 28450,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device if device is not None else next(model.parameters()).device
        self.host = host
        self.port = port
        self.config = getattr(model, "config", None)
        self.state = self.model.initial_state(batch_size=1, device=self.device)

        # Preallocated buffers on target device
        self._pixels_cuda = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=self.device)
        self._ctrl_cuda = torch.zeros((1, 307), dtype=torch.float32, device=self.device)
        self._dt_cuda = torch.tensor([0.016], dtype=torch.float32, device=self.device)

    def serve_forever(self, max_ticks: int | None = None) -> None:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)

        print(f"[Spark Server] Listening on {self.host}:{self.port} (Device: {self.device})...", flush=True)
        conn, addr = server_sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[Spark Server] Client connected from {addr}", flush=True)

        frame_size = HEADER_FRAME_STRUCT.size + 307 + 3072  # 24 + 307 + 3072 = 3403 bytes
        ticks_processed = 0

        try:
            while max_ticks is None or ticks_processed < max_ticks:
                data = bytearray()
                while len(data) < frame_size:
                    packet = conn.recv(frame_size - len(data))
                    if not packet:
                        return
                    data.extend(packet)

                t_spark_start = time.perf_counter_ns()
                magic, tick_id, t_client_ns, dt_val = HEADER_FRAME_STRUCT.unpack_from(data, 0)
                offset = HEADER_FRAME_STRUCT.size

                # Unpack prev_control
                prev_ctrl_bytes = data[offset : offset + 307]
                offset += 307

                # Unpack RGB pixels (32x32x3 uint8)
                rgb_bytes = data[offset : offset + 3072]
                rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((32, 32, 3))

                # Copy into CUDA buffers
                rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(self.device)
                ctrl_tensor = torch.from_numpy(np.frombuffer(prev_ctrl_bytes, dtype=np.uint8)).float().unsqueeze(0).to(self.device)
                self._pixels_cuda.copy_(rgb_tensor)
                self._ctrl_cuda.copy_(ctrl_tensor)
                self._dt_cuda.fill_(dt_val)

                # Execute IreneBrainModel on Spark
                with torch.no_grad():
                    out = self.model(self._pixels_cuda, self._ctrl_cuda, self._dt_cuda, self.state)
                    self.state = out.next_state

                t_spark_end = time.perf_counter_ns()
                spark_inf_ns = t_spark_end - t_spark_start

                # Prepare Action Response Packet
                btn_logits = out.action.button_logits[0].cpu().numpy().astype(np.float32).tobytes()
                ctrl_out = out.action.control[0].cpu().numpy().astype(np.float32).tobytes()

                logits_len = len(btn_logits)
                ctrl_len = len(ctrl_out)

                response = bytearray()
                response.extend(HEADER_ACTION_STRUCT.pack(MAGIC_ACTION, tick_id, spark_inf_ns, logits_len, ctrl_len))
                response.extend(btn_logits)
                response.extend(ctrl_out)

                conn.sendall(response)
                ticks_processed += 1
            print(f"[Spark Server] Successfully processed {ticks_processed} ticks.", flush=True)
        except Exception as exc:
            print(f"[Spark Server Error] {exc}", flush=True)
            raise
        finally:
            conn.close()
            server_sock.close()


class SparkClientInterface:
    """Client interface running on the local PC connected to the DGX Spark."""

    def __init__(self, host: str = "127.0.0.1", port: int = 28450) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect(self) -> None:
        self.sock.connect((self.host, self.port))

    def step(
        self,
        rgb_pixels: np.ndarray,
        prev_control_bytes: bytes,
        dt_seconds: float,
        tick_id: int,
    ) -> tuple[np.ndarray, bytes, SparkTelemetry]:
        t_capture_ns = time.perf_counter_ns()

        if hasattr(rgb_pixels, "pixels"):
            raw_rgb_bytes = rgb_pixels.pixels
        elif isinstance(rgb_pixels, np.ndarray):
            raw_rgb_bytes = rgb_pixels.tobytes()
        elif isinstance(rgb_pixels, (bytes, bytearray)):
            raw_rgb_bytes = bytes(rgb_pixels)
        else:
            raise TypeError(f"Unsupported rgb type: {type(rgb_pixels)}")

        # Pack Frame Packet
        packet = bytearray()
        packet.extend(HEADER_FRAME_STRUCT.pack(MAGIC_FRAME, tick_id, t_capture_ns, dt_seconds))
        packet.extend(prev_control_bytes)
        packet.extend(raw_rgb_bytes)

        t_send_ns = time.perf_counter_ns()
        self.sock.sendall(packet)

        # Receive Header (28 bytes)
        hdr_size = HEADER_ACTION_STRUCT.size
        hdr_data = bytearray()
        while len(hdr_data) < hdr_size:
            chunk = self.sock.recv(hdr_size - len(hdr_data))
            if not chunk:
                raise ConnectionResetError("Lost connection to DGX Spark server")
            hdr_data.extend(chunk)

        magic, ret_tick, spark_inf_ns, logits_len, ctrl_len = HEADER_ACTION_STRUCT.unpack_from(hdr_data, 0)

        # Receive Payload
        payload_size = logits_len + ctrl_len
        payload = bytearray()
        while len(payload) < payload_size:
            chunk = self.sock.recv(payload_size - len(payload))
            if not chunk:
                raise ConnectionResetError("Lost connection during payload transfer")
            payload.extend(chunk)

        t_recv_ns = time.perf_counter_ns()

        btn_logits = np.frombuffer(payload[:logits_len], dtype=np.float32)
        ctrl_floats = np.frombuffer(payload[logits_len : logits_len + ctrl_len], dtype=np.float32)

        t_dispatch_ns = time.perf_counter_ns()

        total_rtt_ms = (t_dispatch_ns - t_capture_ns) / 1e6
        spark_ms = spark_inf_ns / 1e6
        net_ms = total_rtt_ms - spark_ms

        telemetry = SparkTelemetry(
            tick_id=tick_id,
            t_capture_ns=t_capture_ns,
            t_send_ns=t_send_ns,
            t_spark_start_ns=t_send_ns,
            t_spark_end_ns=t_send_ns + spark_inf_ns,
            t_recv_ns=t_recv_ns,
            t_dispatch_ns=t_dispatch_ns,
            total_roundtrip_ms=total_rtt_ms,
            spark_inference_ms=spark_ms,
            network_transfer_ms=net_ms,
        )

        return btn_logits, ctrl_floats, telemetry

    def close(self) -> None:
        self.sock.close()
