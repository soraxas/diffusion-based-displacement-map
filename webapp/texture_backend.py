import sys
from typing import Optional

import torchvision
import threading
from base64 import b64encode
from io import BytesIO
import uuid
import pathlib
import shutil

sys.path.insert(
    0, pathlib.Path(__file__).parent.resolve() / "../diffusion-based-displacement-map"
)

import diffusion_displacement_map.__main__ as dd
import diffusion_displacement_map.api

JOB_ROOT = pathlib.Path(__file__).parent.resolve() / "static" / "jobs"
UPLOAD_ROOT = pathlib.Path(__file__).parent.resolve() / "static" / "job_uploads"

JOB_ROOT.mkdir(exist_ok=True)


class ExportingThread(threading.Thread):
    def __init__(
        self,
        callback,
        callback_error,
        src,
        job_id,
        input_size,
        output_size,
        make_tileable,
        gravel,
        mud,
        sand,
        # **kwargs,
    ):
        self.job_id = job_id

        self.job_dir = JOB_ROOT / job_id
        self.job_dir.mkdir()

        # src = "ripples.displacement.png"

        with open(self.job_dir / "src_path.txt", "w") as f:
            f.write(src)

        _args = [
            "remix",
            f"{UPLOAD_ROOT / src}",
            "--input_size",
            f"{input_size}x{input_size}",
            "--output_size",
            f"{output_size}x{output_size}",
            "--output",
            f"{JOB_ROOT / job_id / 'out.png'}",
            "--gravel_mud_sand",
            f"{gravel},{mud},{sand}",
        ]
        if not make_tileable:
            _args.append("--ignore_tileability")

        cmd, args = dd.build_args(_args)

        self.callback_error = callback_error
        args.report = callback
        self.cmd = cmd
        self.args = args
        self.exception = None
        super().__init__()

    def run(self):
        self.exception = None
        try:
            diffusion_displacement_map.api.process_single_command(self.cmd, self.args)
        except Exception as e:
            if not "generator raised StopIteration" in str(e):
                self.exception = sys.exc_info()
                self.callback_error(sys.exc_info())
                raise e

        # # Your exporting stuff goes here ...
        # for _ in range(10):
        #     time.sleep(1)
        #     self.progress += 10


class BackendBridge:
    def __init__(self, socketio) -> None:
        self.background_process = None
        self.socketio = socketio
        self.image_io = None
        self.data_dict = None
        self._last_job_path = "null"

    def alive(self) -> bool:
        if self.background_process is None:
            return False
        return self.background_process.is_alive()

    @property
    def job_path(self) -> Optional[str]:
        if self.alive():
            return self._last_job_path
        return None

    def start(self, job_upload_path: str, **kwargs):
        if self.alive():
            return False
        job_id = str(uuid.uuid4())[:8]
        self._last_job_path = job_upload_path
        self.background_process = ExportingThread(
            self.callback,
            self.callback_error,
            job_id=job_id,
            src=job_upload_path,
            **kwargs,
        )
        self.background_process.start()
        return True

    def stop(self):
        if self.alive():
            self.background_process.args.should_stop = lambda *args: True
            self.background_process = None
        return True

    def callback(self, **kwargs):
        tensor = kwargs.pop("image")

        image_io = BytesIO()
        # convert the tensor to PIL image using above transform
        torchvision.utils.save_image(tensor, image_io, format="PNG")
        self.image_io = image_io
        self.data_dict = kwargs

        self.socketio.emit("updateCurrentProgress", self.get_data_dict())

    def callback_error(self, exception):
        self.socketio.emit("encounterException", dict(error=str(exception)))

    def get_data_dict(self):
        _dict = dict(alive=self.alive())
        if _dict["alive"]:
            if self.data_dict:
                _dict.update(self.data_dict)
            _dict["image_data"] = self.get_base64()
            _dict["job_id"] = self.background_process.job_id
        return _dict

    def get_base64(self):
        if self.image_io is None:
            return ""

        # pillowimg.save(image_io, 'PNG')
        dataurl = "data:image/png;base64," + b64encode(self.image_io.getvalue()).decode(
            "ascii"
        )
        return dataurl
