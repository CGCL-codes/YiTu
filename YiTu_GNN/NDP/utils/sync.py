import socket
import sys

def server(n_trainer, ip='127.0.0.1', port=8200):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind((ip, port))
  sock.listen(8)
  clientsocks = []
  while n_trainer != 0:
    clientsocket, addr = sock.accept()
    print('Recv a connection. Waiting for connections of {} trainers'.format(n_trainer-1))
    clientsocket.setblocking(1)
    clientsocks.append(clientsocket)
    n_trainer -= 1
  return clientsocks


def trainer(ip='127.0.0.1', port=8200):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect((ip, port))
  sock.setblocking(0) # to non-blocking
  return sock


def barrier(socks, role='none'):
  if role == 'server':
    for sock in socks:
      _ = sock.recv(128)
  elif role == 'trainer':
    socks.send('barrier'.encode('utf-8'))
  else:
    print('Unknown role')

if __name__ == '__main__':
  import argparse
  import time
  parser = argparse.ArgumentParser(description='SyncTest')
  parser.add_argument("--role", type=str, default='none')
  args = parser.parse_args()
  if args.role == 'server':
    sock = server(1)
  elif args.role == 'trainer':
    sock = trainer()
  print('connect successfully.')
  if args.role == 'server':
    time.sleep(5)

  barrier(sock, args.role)
  print('{}: after barrier'.format(args.role))
