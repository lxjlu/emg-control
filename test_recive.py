import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 8888))
print("监听 UDP 8888 端口...")

while True:
    data, addr = sock.recvfrom(1024)
    print(f"来自 {addr}: {data.decode('utf-8')}")