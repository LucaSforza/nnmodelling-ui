#!/usr/bin/env python3
"""Minimal RFB client for driving the Qt VNC platform plugin (stdlib only).

Usage: vnc.py [--host ::1] [--port 5997] STEP...
Steps (executed in order over one connection):
  shot PATH            save the full framebuffer as PNG
  crop X Y W H PATH    save a framebuffer region as PNG (2x upscaled)
  move X Y             move the pointer
  click X Y            left click          dclick X Y   double click
  rclick X Y           right click         drag X1 Y1 X2 Y2   left drag
  wheel X Y N          scroll N notches (negative = down)
  type TEXT            type printable text character by character
  key NAME             press a key or chord, e.g. Return, Escape, ctrl+z, ctrl+shift+v
  wait SECONDS         sleep
"""
import socket
import struct
import sys
import time
import zlib

KEYS = {
    'return': 0xff0d, 'enter': 0xff0d, 'tab': 0xff09, 'escape': 0xff1b, 'esc': 0xff1b,
    'backspace': 0xff08, 'delete': 0xffff, 'home': 0xff50, 'end': 0xff57,
    'left': 0xff51, 'up': 0xff52, 'right': 0xff53, 'down': 0xff54,
    'pageup': 0xff55, 'pagedown': 0xff56, 'space': 0x20,
    'shift': 0xffe1, 'ctrl': 0xffe3, 'control': 0xffe3, 'alt': 0xffe9, 'super': 0xffeb,
    **{f'f{i}': 0xffbd + i for i in range(1, 13)},
}


class Client:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=20)
        version = self.recv(12)
        if not version.startswith(b'RFB '):
            raise RuntimeError(f'not an RFB server: {version!r}')
        minor = int(version[8:11])
        self.sock.sendall(b'RFB 003.008\n' if minor >= 8 else b'RFB 003.003\n')
        if minor >= 7:
            count = self.recv(1)[0]
            types = self.recv(count)
            if 1 not in types:
                raise RuntimeError(f'no None security type: {list(types)}')
            self.sock.sendall(b'\x01')
            if minor >= 8 and struct.unpack('>I', self.recv(4))[0] != 0:
                raise RuntimeError('security handshake failed')
        else:
            if struct.unpack('>I', self.recv(4))[0] != 1:
                raise RuntimeError('server requires authentication')
        self.sock.sendall(b'\x01')  # shared
        self.width, self.height = struct.unpack('>HH', self.recv(4))
        self.recv(16)  # server pixel format, replaced below
        self.recv(struct.unpack('>I', self.recv(4))[0])
        # 32 bpp, depth 24, little endian, true colour, shifts r16 g8 b0
        self.sock.sendall(struct.pack('>B3xBBBBHHHBBB3x', 0, 32, 24, 0, 1,
                                      255, 255, 255, 16, 8, 0))
        self.sock.sendall(struct.pack('>BxHi', 2, 1, 0))  # raw encoding only
        self.buttons = 0
        self.x = self.y = 0
        self.fb = bytearray(self.width * self.height * 4)

    def recv(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise RuntimeError('connection closed')
            data += chunk
        return bytes(data)

    def update(self, incremental=False):
        self.sock.sendall(struct.pack('>BBHHHH', 3, int(incremental), 0, 0,
                                      self.width, self.height))
        while True:
            kind = self.recv(1)[0]
            if kind == 0:
                break
            if kind == 1:  # colour map
                _, _, n = struct.unpack('>BHH', self.recv(5))
                self.recv(n * 6)
            elif kind == 2:  # bell
                continue
            elif kind == 3:  # cut text
                self.recv(3)
                self.recv(struct.unpack('>I', self.recv(4))[0])
            else:
                raise RuntimeError(f'unexpected server message {kind}')
        self.recv(1)
        (rects,) = struct.unpack('>H', self.recv(2))
        for _ in range(rects):
            x, y, w, h, enc = struct.unpack('>HHHHi', self.recv(12))
            if enc != 0:
                raise RuntimeError(f'unsupported encoding {enc}')
            data = self.recv(w * h * 4)
            for row in range(h):
                start = ((y + row) * self.width + x) * 4
                self.fb[start:start + w * 4] = data[row * w * 4:(row + 1) * w * 4]

    def png(self, path, x=0, y=0, w=None, h=None, scale=1):
        w = w or self.width
        h = h or self.height
        raw = bytearray()
        for row in range(y, y + h):
            line = bytearray()
            for col in range(x, x + w):
                b, g, r = self.fb[(row * self.width + col) * 4:(row * self.width + col) * 4 + 3]
                line += bytes((r, g, b)) * scale
            for _ in range(scale):
                raw += b'\x00' + line

        def chunk(tag, body):
            return (struct.pack('>I', len(body)) + tag + body +
                    struct.pack('>I', zlib.crc32(tag + body) & 0xffffffff))
        with open(path, 'wb') as out:
            out.write(b'\x89PNG\r\n\x1a\n')
            out.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w * scale, h * scale, 8, 2, 0, 0, 0)))
            out.write(chunk(b'IDAT', zlib.compress(bytes(raw), 6)))
            out.write(chunk(b'IEND', b''))

    def pointer(self, x, y, buttons=None):
        if buttons is not None:
            self.buttons = buttons
        self.x, self.y = x, y
        self.sock.sendall(struct.pack('>BBHH', 5, self.buttons, x, y))

    def key(self, keysym, down):
        self.sock.sendall(struct.pack('>BBxxI', 4, int(down), keysym))

    def click(self, x, y, button=1, count=1):
        self.pointer(x, y, 0)
        time.sleep(0.05)
        for _ in range(count):
            self.pointer(x, y, button)
            time.sleep(0.05)
            self.pointer(x, y, 0)
            time.sleep(0.08)

    def drag(self, x1, y1, x2, y2):
        self.pointer(x1, y1, 0)
        time.sleep(0.1)
        self.pointer(x1, y1, 1)
        time.sleep(0.15)
        steps = 20
        for i in range(1, steps + 1):
            self.pointer(x1 + (x2 - x1) * i // steps, y1 + (y2 - y1) * i // steps, 1)
            time.sleep(0.03)
        time.sleep(0.15)
        self.pointer(x2, y2, 0)

    def wheel(self, x, y, notches):
        mask = 8 if notches > 0 else 16
        for _ in range(abs(notches)):
            self.pointer(x, y, mask)
            self.pointer(x, y, 0)
            time.sleep(0.05)

    def chord(self, name):
        parts = name.split('+')
        syms = []
        for part in parts:
            lower = part.lower()
            if lower in KEYS:
                syms.append(KEYS[lower])
            elif len(part) == 1:
                syms.append(ord(part))
            else:
                raise RuntimeError(f'unknown key {part}')
        for sym in syms:
            self.key(sym, True)
            time.sleep(0.03)
        for sym in reversed(syms):
            self.key(sym, False)
            time.sleep(0.03)

    def type(self, text):
        for char in text:
            self.key(ord(char), True)
            self.key(ord(char), False)
            time.sleep(0.04)


def main(argv):
    host, port = '::1', 5997
    while argv and argv[0].startswith('--'):
        flag, value, argv = argv[0], argv[1], argv[2:]
        if flag == '--host':
            host = value
        elif flag == '--port':
            port = int(value)
    if not argv:
        print(__doc__)
        return 2
    client = Client(host, port)
    i = 0
    while i < len(argv):
        step = argv[i]
        args = argv[i + 1:]
        if step == 'shot':
            time.sleep(0.3)
            client.update()
            client.png(args[0])
            print(f'saved {args[0]} ({client.width}x{client.height})')
            i += 2
        elif step == 'crop':
            time.sleep(0.3)
            client.update()
            x, y, w, h = map(int, args[:4])
            client.png(args[4], x, y, w, h, scale=2)
            print(f'saved {args[4]}')
            i += 6
        elif step in ('move', 'click', 'dclick', 'rclick'):
            x, y = int(args[0]), int(args[1])
            if step == 'move':
                client.pointer(x, y)
            else:
                client.click(x, y, 4 if step == 'rclick' else 1, 2 if step == 'dclick' else 1)
            i += 3
        elif step == 'drag':
            client.drag(*map(int, args[:4]))
            i += 5
        elif step == 'wheel':
            client.wheel(int(args[0]), int(args[1]), int(args[2]))
            i += 4
        elif step == 'type':
            client.type(args[0])
            i += 2
        elif step == 'key':
            client.chord(args[0])
            i += 2
        elif step == 'wait':
            time.sleep(float(args[0]))
            i += 2
        else:
            raise RuntimeError(f'unknown step {step}')
    time.sleep(0.2)
    client.sock.close()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, RuntimeError, ValueError, IndexError) as error:
        print(f'vnc: {error}', file=sys.stderr)
        sys.exit(1)
