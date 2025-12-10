from flask import Flask, send_file, request, jsonify
from flask_socketio import SocketIO, emit
import webbrowser

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/control-audio', methods=['GET', 'POST'])
def control_audio():
    """
    Управление аудио через HTTP запрос
    Примеры:
    GET /control-audio?action=play
    POST /control-audio с JSON {"action": "stop"}
    """
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            action = data.get('action')
        else:
            action = request.form.get('action')
    else:
        action = request.args.get('action')
    
    if action in ['play', 'stop']:
        socketio.emit('audio_command', {'action': action})
        return jsonify({'status': 'success', 'action': action})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid action. Use "play" or "stop".'}), 400

@socketio.on('connect')
def handle_connect():
    print('Клиент подключен')

@socketio.on('audio_control')
def handle_audio_control(data):
    action = data.get('action')
    if action in ['play', 'stop']:
        emit('audio_command', {'action': action}, broadcast=True)

if __name__ == '__main__':
    webbrowser.open('http://localhost:5000')
    socketio.run(app, port=5000, debug=True)

