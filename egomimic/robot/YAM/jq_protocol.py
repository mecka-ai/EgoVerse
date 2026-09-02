"""Frame parser for the JQ Precision (矩侨工业 / JQ-Industries) fabric tactile glove.

Wire format per the vendor spec "Product Specification | Fabric Electronic Skin
(Tactile Glove) v1.2" §5.5 (JQGY-YL-11, 162 sensing points, 921600 baud serial or
the Bluetooth dongle presenting the same stream):

    [seq:u8] [sensor_type:u8] [payload] [0xAA 0x55 0x03 0x99]

Each sample is TWO packets, both terminated by the 4-byte delimiter:

    seq=0x01: payload = 128 bytes  (pressure values 1..128)
    seq=0x02: payload = 144 bytes  (pressure values 129..256 + 16 bytes IMU)

The two packets combine to a 256-slot 8-bit pressure array (a 16x16 scan matrix;
162 of the slots are physical sensing points — see the mapping tables in
static/layout.js) plus an IMU quaternion (4 x float32 LE: w, x, y, z).

sensor_type: 0x01=left hand, 0x02=right hand, 0x03/0x04=feet, 0x05=whole body.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

DELIM = b"\xaa\x55\x03\x99"
PKT1_LEN = 2 + 128  # seq + type + first 128 pressure bytes
PKT2_LEN = 2 + 128 + 16  # seq + type + last 128 pressure bytes + quaternion
NUM_POINTS = 256
_QUAT = struct.Struct("<4f")

# 0x01=LH / 0x02=RH per spec §5.5.1, user-confirmed on the physical gloves.
# (The finger-band mirroring in layout.js shellTables() is still required — the
# shells sit mirrored over the fabric, so thumb<->pinky bands swap per hand.)
# 0x03/0x04 (feet) and 0x05 (body) are ignored here.
HANDS = {0x01: "left", 0x02: "right"}


@dataclass(frozen=True)
class JqFrame:
    """One assembled sample: the raw 256-slot pressure array (spec tables index it
    1-based) and the IMU quaternion (w, x, y, z)."""

    hand: str
    pressure: bytes  # 256 x u8
    quat: tuple[float, float, float, float]


class JqParser:
    """Incremental, resyncing parser: feed arbitrary serial chunks to ``push``, get
    assembled frames back. Splits on the delimiter, pairs packet 1 with the next
    packet 2 of the same sensor type; a malformed segment is counted and skipped."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._pkt1: dict[str, bytes] = {}  # hand -> first-half pressure, awaiting pkt2
        self.frames_ok = 0
        self.segment_errors = 0

    def push(self, data: bytes) -> list[JqFrame]:
        self._buf += data
        out: list[JqFrame] = []
        while True:
            i = self._buf.find(DELIM)
            if i < 0:
                # keep at most one partial segment; a garbage flood can't grow the buffer
                if len(self._buf) > 4096:
                    del self._buf[:-PKT2_LEN]
                return out
            segment = bytes(self._buf[:i])
            del self._buf[: i + len(DELIM)]
            frame = self._segment(segment)
            if frame is not None:
                out.append(frame)

    def _segment(self, seg: bytes) -> JqFrame | None:
        if len(seg) < 2:
            if seg:  # empty segment = delimiter run; not an error
                self.segment_errors += 1
            return None
        seq, stype = seg[0], seg[1]
        hand = HANDS.get(stype)
        if hand is None or (seq == 1 and len(seg) != PKT1_LEN) or (seq == 2 and len(seg) != PKT2_LEN):
            self.segment_errors += 1
            return None
        if seq == 1:
            self._pkt1[hand] = seg[2:]
            return None
        if seq == 2:
            first = self._pkt1.pop(hand, None)
            if first is None:
                self.segment_errors += 1  # pkt2 without its pkt1 (mid-stream attach)
                return None
            self.frames_ok += 1
            return JqFrame(
                hand=hand,
                pressure=first + seg[2 : 2 + 128],
                quat=_QUAT.unpack(seg[2 + 128 :]),
            )
        self.segment_errors += 1
        return None
