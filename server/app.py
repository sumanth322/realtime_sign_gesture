import os
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, join_room, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# =========================
# Serve gesture.html and normal.html
# =========================
CLIENTS_DIR = os.path.join(os.path.dirname(__file__),"..","clients")

@app.route("/clients/<path:filename>")
def client_files(filename):
    return send_from_directory(CLIENTS_DIR, filename)

# =========================
# SocketIO Events
# =========================
@socketio.on("join")
def on_join(data):
    room = data["room"]
    join_room(room)
    print(f"🔗 Client joined room: {room}")

@socketio.on("offer")
def on_offer(data):
    emit("offer", data, room=data["room"], include_self=False)

@socketio.on("answer")
def on_answer(data):
    emit("answer", data, room=data["room"], include_self=False)

@socketio.on("ice-candidate")
def on_ice(data):
    emit("ice-candidate", data, room=data["room"], include_self=False)

@socketio.on("gesture")
def on_gesture(data):
    emit("gesture", data, room=data["room"], include_self=False)

# =========================
# Run server
# =========================
if __name__ == "__main__":
    print("🚀 Starting signaling server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
