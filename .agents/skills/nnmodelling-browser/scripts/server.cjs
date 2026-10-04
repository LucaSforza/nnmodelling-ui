const http = require('http');
const fs = require('fs');
const path = require('path');
const net = require('net');
const { WebSocketServer, WebSocket } = require('ws');
const root = __dirname;
const server = http.createServer((req, res) => {
  const requested = new URL(req.url, 'http://localhost').pathname;
  const file = path.resolve(root, '.' + (requested === '/' ? '/index.html' : requested));
  if (!file.startsWith(root + '/')) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (error, data) => {
    if (error) { res.writeHead(404); return res.end(); }
    res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : 'text/html');
    res.end(data);
  });
});
const wss = new WebSocketServer({ server, path: '/vnc' });
wss.on('connection', ws => {
  const tcp = net.connect({ host: '::1', port: 5997 });
  ws.on('message', data => tcp.write(data));
  tcp.on('data', data => { if (ws.readyState === WebSocket.OPEN) ws.send(data); });
  tcp.on('error', () => ws.close());
  tcp.on('close', () => ws.close());
  ws.on('close', () => tcp.destroy());
  ws.on('error', () => tcp.destroy());
});
server.listen(8797, '127.0.0.1', () => console.log('NNModelling browser bridge: http://127.0.0.1:8797'));
