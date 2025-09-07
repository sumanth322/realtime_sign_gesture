# app.py
from flask import Flask, request
from flask_socketio import SocketIO, join_room, leave_room, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # allow local testing

@socketio.on('join')
def handle_join(data):
    room = data.get('room')
    if not room:
        return
    join_room(room)
    print(f"SID {request.sid} joined room {room}")
    # let others know someone joined
    emit('peer-joined', {'sid': request.sid}, room=room, skip_sid=request.sid)

@socketio.on('leave')
def handle_leave(data):
    room = data.get('room')
    leave_room(room)
    print(f"SID {request.sid} left room {room}")
    emit('peer-left', {'sid': request.sid}, room=room)

@socketio.on('offer')
def handle_offer(data):
    room = data.get('room')
    sdp = data.get('sdp')
    print(f"offer from {request.sid} to room {room}")
    emit('offer', {'sdp': sdp, 'from': request.sid}, room=room, skip_sid=request.sid)

@socketio.on('answer')
def handle_answer(data):
    room = data.get('room')
    sdp = data.get('sdp')
    print(f"answer from {request.sid} to room {room}")
    emit('answer', {'sdp': sdp, 'from': request.sid}, room=room, skip_sid=request.sid)

@socketio.on('ice-candidate')
def handle_ice(data):
    room = data.get('room')
    cand = data.get('candidate')
    emit('ice-candidate', {'candidate': cand, 'from': request.sid}, room=room, skip_sid=request.sid)

# gesture events come from your Python inference script and are forwarded to the room
@socketio.on('gesture')
def handle_gesture(data):
    room = data.get('room')
    label = data.get('label')
    score = data.get('score', None)
    print(f"gesture '{label}' from {request.sid} to room {room}")
    # forward to everyone in the room except the sender
    emit('gesture', {'label': label, 'score': score, 'from': request.sid}, room=room, skip_sid=request.sid)

if __name__ == '__main__':
    print("🚀 Starting signaling server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

