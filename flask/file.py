from flask import Flask, render_template, request, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
import os

from os.path import split, join

import pretty_midi
from pretty_midi import Note
import numpy as np
import librosa
from pydub import AudioSegment
from tensorflow import keras
import tensorflow.python.keras.mixed_precision.policy as mixed_precision
import warnings

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

warnings.filterwarnings("ignore")
AudioSegment.converter = 'ffmpeg'

len_model = keras.models.load_model("models/offset_detector.h5")
onset_model = keras.models.load_model("models/onset_detector.h5")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

TRANS_FOLDER = 'trans'
app.config['TRANS_FOLDER'] = TRANS_FOLDER

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

def one_to_midi(notes, offsets, fileName, time):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=1)

    notes = notes.T
    offsets = offsets.T
    for pitch, hor in enumerate(notes):
        nz = np.where(hor != 0)[0]
        if len(nz) == 0:
            continue

        visit = [False] * len(hor)
        off = offsets[pitch]
        for idx in nz:
            i = idx
            while i < len(off) and off[i] != 0:
                visit[i] = True
                i += 1

        idx = 0
        while idx < len(visit):
            startTime = idx * time
            endTime = startTime

            while idx < len(visit) and visit[idx] == True:
                endTime += time
                idx += 1

            if startTime != endTime:
                instrument.notes.append(Note(velocity=100, pitch=pitch + 21, start=startTime, end=endTime))
            idx += 1

    pm.instruments.append(instrument)
    pm.write(fileName)
    
    return fileName

def transcript(X_test_path):
    y, sr = librosa.load(X_test_path, sr=16000)
    cqt = librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(21), n_bins=264, hop_length=160, bins_per_octave=36)
    cqt = np.abs(cqt)
    cqt = cqt.T

    cqt = cqt / np.max(cqt)

    # 시퀀스 길이 / 배치 사이즈 상수
    one_seq = 100
    batch_size = 10

    # 시퀀스 패딩
    pad_size = one_seq - (cqt.shape[0] % one_seq)
    cqt = np.pad(cqt, ((0, pad_size), (0, 0)), mode='constant')
    cqts = cqt.reshape(cqt.shape[0] // one_seq, one_seq, 264)

    # Batch 패딩
    desired_shape = (batch_size * ((cqts.shape[0] + (batch_size - 1)) // batch_size), one_seq, 264)
    padding_shape = (desired_shape[0] - cqts.shape[0], desired_shape[1] - cqts.shape[1], desired_shape[2] - cqts.shape[2])
    cqts = np.pad(cqts, ((0, padding_shape[0]), (0, padding_shape[1]), (0, padding_shape[2])), mode='constant')

    # 데이터 예측
    len_model.reset_states()
    len_result = len_model.predict(cqts, batch_size=batch_size)
    onset_result = onset_model.predict(cqts, batch_size=batch_size)

    onset = onset_result.reshape(onset_result.shape[0] * one_seq, 88)
    offset = len_result.reshape(len_result.shape[0] * one_seq, 88)

    onset = np.where(onset >= 0.5, 1, 0)
    offset = np.where(offset >= 0.3, 1, 0)

    to_elapse = librosa.frames_to_time(1, sr=sr, hop_length=160)
    fileName = one_to_midi(notes=onset, offsets=offset, fileName=join(app.config['TRANS_FOLDER'], split(X_test_path)[-1][:-4] + '.mid'), time=to_elapse)

    return fileName

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/fileUpload', methods=['POST'])
def file_upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # 피아노 전사 후 midi파일 다운로드 링크 생성
        filename = transcript(file_path)

        download_url = url_for('download_file', filename=filename)
        return jsonify({
            "success": True, 
            "message": "Transcription done!",
            "downloadUrl": download_url
        })
    else:
        return jsonify({"success": False, "message": "Invalid file type"}), 400

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TRANS_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', debug=False)