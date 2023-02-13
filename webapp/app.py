from flask import Flask, request, render_template, send_from_directory, redirect
from flask_socketio import SocketIO
import os
from PIL import Image
import uuid
from pathlib import Path

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

APP_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

image_processing_width = 560

(APP_ROOT / "static" / "job_uploads").mkdir(parents=True, exist_ok=True)


# default access page
@app.route("/")
def main():
    return render_template("index.html")


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = APP_ROOT / "static/images/"
    # create image directory if not found
    target.mkdir(parents=True, exist_ok=True)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = str.lower(os.path.splitext(filename)[1])
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return (
            render_template("error.html", message="The selected file is not supported"),
            400,
        )

    filename = "uploaded" + ext

    # save file
    destination = target / filename
    print("File saved to to:", destination)
    upload.save(destination)

    # filename = str(uuid.uuid4())[:8]
    # upload.save(APP_ROOT /  f'static/job_uploads/{filename}'))

    # forward to processing page
    return render_template(
        "processing.html", image_name=filename, max_width=image_processing_width
    )
    # return render_template("processing.html", image_name=filename)


@app.route("/upload-image", methods=["POST"])
def do_upload():
    return crop(skip_crop=True)


# crop filename from (x1,y1) to (x2,y2)
@app.route("/crop", methods=["POST"])
def crop(skip_crop=False):
    filename = request.form["image"]

    # open image
    target = APP_ROOT / "static/images"
    destination = target / filename
    img = Image.open(destination)

    if not skip_crop:
        # retrieve parameters from html form
        x1 = int(request.form["x1"])
        y1 = int(request.form["y1"])
        x2 = int(request.form["x2"])
        y2 = int(request.form["y2"])

        # check for valid crop parameters
        width = img.size[0]
        height = img.size[1]

        ################################################
        # scale to the actual image dimention
        x1 = (x1 / image_processing_width) * width
        y1 = (y1 / image_processing_width) * height
        x2 = (x2 / image_processing_width) * width
        y2 = (y2 / image_processing_width) * height
        ################################################

        crop_possible = True
        if not 0 <= x1 < width:
            crop_possible = False
        if not 0 < x2 <= width:
            crop_possible = False
        if not 0 <= y1 < height:
            crop_possible = False
        if not 0 < y2 <= height:
            crop_possible = False
        if not x1 < x2:
            crop_possible = False
        if not y1 < y2:
            crop_possible = False

        # crop image and show
        if not crop_possible:
            return (
                render_template("error.html", message="Crop dimensions not valid"),
                400,
            )

        img = img.crop((x1, y1, x2, y2))
        # # save and return image
        # destination = "/".join([target, 'temp.png'])
        # if os.path.isfile(destination):
        # 	os.remove(destination)
        # img.save(destination)
        # return send_image('temp.png')

    filename = f"{str(uuid.uuid4())[:8]}.png"
    import pathlib

    (APP_ROOT / "static/job_uploads").mkdir(exist_ok=True)
    img.save(APP_ROOT / f"static/job_uploads/{filename}")
    return redirect(f"job/{filename}/monitor", code=302)


# retrieve file from 'static/images' directory
@app.route("/static/images/<filename>")
def send_image(filename):
    # import threading

    # exporting_threads[thread_id] = ExportingThread()
    # exporting_threads[thread_id].start()

    return send_from_directory("static/images", filename)


import texture_backend

backend_bridge = texture_backend.BackendBridge(socketio)


@app.route("/job/stop")
def stop_process():
    backend_bridge.stop()
    return "ok"


@app.route("/job/<string:job_upload_path>/start", methods=["POST"])
def start_process(job_upload_path):
    # use output size as input size
    kwargs = dict()
    make_tileable = False
    if "make-tileable" in request.form:
        make_tileable = bool(request.form["make-tileable"])

    for key in ["gravel", "mud", "sand"]:
        kwargs[key] = float(request.form[key])

    # _out_size = (request.form[])
    _out_size = int(request.form["output-size"])
    job_id = backend_bridge.start(
        job_upload_path,
        input_size=_out_size,
        output_size=_out_size,
        make_tileable=make_tileable,
        **kwargs,
    )
    if job_id:
        return dict(status=True, job_id=job_id)
    return dict(status=False)


@app.route("/job/query")
def get_current_progress():
    return backend_bridge.get_data_dict()
    # _dict = dict(alive=backend_bridge.alive())
    # if _dict['alive']:
    # 	_dict['image_data'] = backend_bridge.get_data_dict()
    # return _dict


@app.route("/job/<string:filesrc>/monitor")
def monitor(filesrc):
    return render_template("monitor.html", filesrc=filesrc, active=True)


@app.route("/job/current_monitor")
def monitor_current_job():
    return render_template(
        "monitor.html",
        active=backend_bridge.job_path is not None,
        filesrc=backend_bridge.job_path,
    )


@app.route("/jobs")
def jobs_list():
    datapath = APP_ROOT / "static/jobs"
    jobs = []
    for folder in (d for d in datapath.iterdir() if d.is_dir()):
        with open(folder / "src_path.txt", "r") as f:
            src_name = f.read()
        jobs.append(
            dict(
                id=folder.name,
                src=f"/static/job_uploads/{src_name}",
                out=f"/static/jobs/{folder.name}/out.png",
            )
        )
    return render_template("jobs.html", jobs=jobs)


@app.route("/job-uploads")
def job_uploads_list():
    datapath = APP_ROOT / "static/job_uploads"
    uploads = []
    for file in (d for d in datapath.iterdir() if d.is_file()):
        uploads.append(dict(src=f"static/job_uploads/{file.name}", src_name=file.name))
    return render_template("job_uploads.html", uploads=uploads)


"""
Decorator for connect
"""


@socketio.on("connect")
def connect():
    global thread
    print("Client connected")


"""
Decorator for disconnect
"""


@socketio.on("disconnect")
def disconnect():
    print("Client disconnected", request.sid)


# run with host 0.0.0.0
# this is needed for exposing port outside docker container
if __name__ == "__main__":
    import sys

    kwargs = dict()
    if len(sys.argv) > 1:
        kwargs["port"] = int(sys.argv[1])
    app.run(host="0.0.0.0", debug=True, **kwargs)
