import os
import math

import dash
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
import pandas as pd
import plotly.express as px

from IPython.display import Audio
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import scipy
import pydub

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

with open('embeddings.json') as f:
  embeddings = json.load(f)


# celebrities = ['Kevin Hart','Morgan Freeman','Tom Cruise']
celebrities = embeddings.keys()

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

outfile = "/content/drive/My Drive/Real-Time-Voice-Cloning/samples/morgan-freeman-to-me-it's-just-a-made-up-word-a-politician's-word-so-that-young-fellas-like-yourself-can-wear-a-suit-and-a-tie-and-have-a-job.wav"
in_fpath = Path(outfile)
print("preprocessing the training audio file")
# reprocessed_wav = encoder.preprocess_wav(in_fpath)
original_wav, sampling_rate = librosa.load(in_fpath)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embed = encoder.embed_utterance(preprocessed_wav)

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H4(children="AI Speech Recognition (Planet Money podcast)"),
        dcc.Markdown(
            "Written in 100 lines of Python with [Dash](https://dash.plot.ly/)."
        ),
        dcc.Markdown(
            """
    **Instructions:** Drag the blue slider to transcribe audio from a Planet \
        Money podcast (in 5 second increments.) Transcription is done \
            in realtime with Python bindings to [Carnegie Mellon's \
                Sphinx Speech Recognition software](https://cmusphinx.github.io). \
                    Play the audio to see how well the transcription matches.
    [Code on GitHub](https://github.com/plotly/dashdub).
    """
        ),
        dcc.Markdown("**Choose your celebrity**"),
        dcc.Dropdown(id="celebrity-dropdown",options=[{'label':celebrity,'value':celebrity} for celebrity in celebrities]),
        html.Div(id="slider-output-container"),
        html.P(children="Carnegie Mellon Sphinx transcription:"),
        dcc.Textarea(id="transcription_input", cols=80),
        html.Button('Submit', id='submit', n_clicks=0),
        html.Br(),
        html.Audio(id="player", src="http://docs.google.com/uc?export=open&id=1jY1Gz9naGhvesxpm5mG1hr6Y486Wry60", controls=True, style={
          "width": "100%",
        }),
        # dcc.Graph(id="waveform", figure=fig),
    ]
)

#  Transcribe audio
@app.callback(
    dash.dependencies.Output("player", "src"),
    [dash.dependencies.Input("submit","n_clicks"),
     ],
    [dash.dependencies.State("celebrity-dropdown","value"),
     dash.dependencies.State("transcription_input", "value")],
)

def vocalize(n_clicks,celebrity,value):
    text= value
    embed = embeddings[celebrity]
    print("Synthesizing new audio...")
    # with io.capture_output() as captured:
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    audio = Audio(generated_wav, rate=synthesizer.sample_rate)
    # display(audio)
    # return json.dumps({'sample_rate':synthesizer.sample_rate,'audio':generated_wav.tolist()})

    write('generated_via_flask_api.mp3',synthesizer.sample_rate,generated_wav,normalized=True)

    folder_id = '1hOJ9GrsOHLRGe75YwLhS8tfb9zi2bvB9'
    file1 = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder_id}],'title':'audio.mp3',})
    file1.SetContentFile('generated_via_flask_api.mp3')
    file1.Upload()
    # # Fetch permissions.
    permissions = file1.GetPermissions()
    permission = file1.InsertPermission({
                    'type': 'anyone',
                    'value': 'anyone',
                    'role': 'reader'})
    token = permissions[0]['selfLink'].split('/')[-3]

    return "http://docs.google.com/uc?export=open&id="+token


if __name__ == "__main__":
    app.run_server(debug=False)
