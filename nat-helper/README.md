# Hole punching coordinator for UltraGrid

This utility serves as a meeting point for UltraGrid clients, that need to
connect to each other, but don't have publicly routable IP addresses.

Dependencies
---------
There is only one dependency - `asio`.

On Ubuntu and other Debian based distributions it can be installed like this:
```
apt install libasio-dev
```

On Fedora/RHEL:
```
dnf install asio-devel
```

Building
---------
    mkdir build && cd build
    cmake ..
    make
    
Usage
---------
    ./nat-helper [-h/--help] [-p/--port <port>]
    
If no port is specified, 12558 is used.

Protocol description
---------

Clients connect to the nat-helper server, identify themselves with a name, and
join a room. Once two clients enter the same room, nat-helper forwards name,
sdp description string, and all candidate address pairs between the two
clients.

All communication is done via messages that have the following structure:

    <HEADER><MSG_BODY>
    
`HEADER`: 5B string containing length of MSG_BODY, null-termination optional  
`MSG_BODY`: content of message, length determined by header, max 2048B
    
After establishing connection to the nat-helper server, following messages are
sent and received in that order:
1. Client sends its name
2. Client sends room name to join
3. Client sends its sdp description
4. Client receives the name of the other client in the room
5. Client receives the sdp description of the other client

After that the client sends and receives sdp candidate pairs as they are
discovered.
