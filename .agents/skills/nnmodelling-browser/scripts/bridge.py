#!/usr/bin/env python3
"""Manage the repository's loopback-only Qt VNC/noVNC test session."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time


def listening(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def managed_pid(bridge, name, expected):
    file = bridge / (name + '.pid')
    if not file.exists():
        return None
    pid = int(file.read_text().strip())
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    if not args or not args[0]:
        return None
    if expected not in args:
        raise RuntimeError(f'{file} references another process; refusing to signal it')
    return pid


def stop(bridge, executable):
    for name, expected in [('qt', os.fsencode(executable)),
                           ('browser', os.fsencode(bridge / 'server.cjs'))]:
        pid = managed_pid(bridge, name, expected)
        if pid:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if not managed_pid(bridge, name, expected):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(f'{name} did not stop; inspect its log')
        (bridge / (name + '.pid')).unlink(missing_ok=True)
    deadline = time.monotonic() + 2
    while listening('::1', 5997) or listening('127.0.0.1', 8797):
        if time.monotonic() >= deadline:
            raise RuntimeError('Bridge ports still active; inspect their process identity before stopping them')
        time.sleep(0.05)
    (bridge / 'session.json').unlink(missing_ok=True)
    path = bridge / 'nn.sock'
    if path.exists():
        info = path.lstat()
        if stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid():
            with socket.socket(socket.AF_UNIX) as probe:
                try:
                    probe.connect(str(path))
                except ConnectionRefusedError:
                    path.unlink()


def launch(command, bridge, name, env=None):
    with (bridge / (name + '.log')).open('w') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=bridge)
    (bridge / (name + '.pid')).write_text(str(process.pid) + '\n')
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    parser.add_argument('project', nargs='?')
    parser.add_argument('--build', action='store_true')
    options = parser.parse_args()
    assets = Path(__file__).resolve().parent
    root = assets.parents[3]
    bridge = root / '.computer-use'
    executable = root / 'build/qt/nnmodelling-ui'
    bridge.mkdir(exist_ok=True)
    if options.action == 'stop':
        stop(bridge, executable)
        print('Stopped NNModelling browser session')
        return
    if options.action == 'status':
        marker = bridge / 'session.json'
        recorded = json.loads(marker.read_text()) if marker.exists() else {}
        digest = hashlib.sha256(executable.read_bytes()).hexdigest() if executable.exists() else None
        print(f'VNC={listening("::1", 5997)} HTTP={listening("127.0.0.1", 8797)} '
              f'stale={bool(recorded) and recorded.get("executable_sha256") != digest}')
        return
    if options.build:
        env = os.environ.copy()
        env['CCACHE_DISABLE'] = '1'
        subprocess.run(['just', 'build'], cwd=root, env=env, check=True)
    if not executable.exists():
        raise RuntimeError('Run just build first')
    for name, expected in [('qt', os.fsencode(executable)),
                           ('browser', os.fsencode(bridge / 'server.cjs'))]:
        if managed_pid(bridge, name, expected):
            raise RuntimeError('Session running; close/save app then run bridge.py stop')
    if listening('::1', 5997) or listening('127.0.0.1', 8797):
        raise RuntimeError('Bridge ports occupied by another process')
    for name in ['server.cjs', 'index.html', 'loopback.c', 'package.json', 'package-lock.json']:
        shutil.copyfile(assets / name, bridge / name)
    if not (bridge / 'node_modules/@novnc/novnc/core/rfb.js').exists() or not (bridge / 'node_modules/ws').exists():
        subprocess.run(['npm', 'ci', '--cache', str(bridge / 'npm-cache'), '--ignore-scripts'],
                       cwd=bridge, check=True)
    subprocess.run(['cc', '-shared', '-fPIC', str(bridge / 'loopback.c'), '-ldl',
                    '-o', str(bridge / 'loopback.so')], check=True)
    path = bridge / 'nn.sock'
    if path.exists():
        info = path.lstat()
        with socket.socket(socket.AF_UNIX) as probe:
            try:
                probe.connect(str(path))
            except ConnectionRefusedError:
                if stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid():
                    path.unlink()
                else:
                    raise RuntimeError('Refusing to replace existing socket path')
            else:
                raise RuntimeError('Automation socket is active')
    env = os.environ.copy()
    env.update(LD_PRELOAD=str(bridge / 'loopback.so'),
               QT_QPA_PLATFORM='vnc:size=1600x1000:port=5997', QT_STYLE_OVERRIDE='Fusion')
    args = [str(executable), '--socket', str(path)]
    if options.project:
        args.append(str(Path(options.project).resolve()))
    qt = launch(args, bridge, 'qt', env)
    web = launch(['node', str(bridge / 'server.cjs')], bridge, 'browser')
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if qt.poll() is not None or web.poll() is not None:
            stop(bridge, executable)
            raise RuntimeError('Session failed; inspect qt.log and browser.log')
        if listening('::1', 5997) and listening('127.0.0.1', 8797) and path.exists():
            (bridge / 'session.json').write_text(json.dumps({
                'qt_pid': qt.pid, 'browser_pid': web.pid,
                'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest()
            }) + '\n')
            print('Ready: http://127.0.0.1:8797')
            return
        time.sleep(0.1)
    stop(bridge, executable)
    raise RuntimeError('Readiness timeout; inspect qt.log and browser.log')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Browser session: {error}', file=sys.stderr)
        sys.exit(1)
