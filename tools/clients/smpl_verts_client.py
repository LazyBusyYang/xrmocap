# yapf: disable
import numpy as np
import socketio
import time
from xrprimer.utils.log_utils import logging, setup_logger
import datetime as datetime
from enum import Enum
from typing import Optional
# yapf: enable

class XRMocapSMPLClientActionsEnum(str, Enum):
    UPLOAD = 'upload'
    FORWARD = 'forward'
    GET_FACES = 'get_faces'

class XRMocapSMPLClient:
    def __init__(
        self,
        server_ip: str = '127.0.0.1',
        server_port: int = 8376,
        resp_type: str = 'bytes',
        log_path: Optional[str] = None
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.resp_type = resp_type

        # setup logger
        self.logger = setup_logger(
            logger_name=f'SMPLClient-{self.__get_time_stamp()}',
            logger_path=log_path,
            file_level=logging.WARN,
            console_level=logging.WARN
        )

        # setup websocket client
        self.socketio_client = socketio.Client()
        self.socketio_client.connect(f'http://{server_ip}:{server_port}')
        
        self.upload_success = False
        self.verts = None


    @classmethod
    def __get_time_stamp(cls) -> str:
        t = time.gmtime()
        t = time.strftime(r"%Y-%m-%d-%H-%M-%S", t)

        return t
    
    def __parse_upload_response(self, data):
        if data['status'] == 'success':
            num_frames = int(data['num_frames'])
        else:
            msg = data['msg']
            self.logger.error(f'Upload failed.\n{msg}')
            self.socketio_client.disconnect()
            raise RuntimeError
        
        return num_frames
        
    def upload_body_motion(self, body_motion: bytes) -> int:
        if body_motion is None:
            self.logger.error('Body motion is empty.\n')
            raise ValueError
        
        data = {'file_name': 'body_motion', 'file_data': body_motion}
        self.logger.info('Sending upload request...')
        resp_data = self.socketio_client.call(XRMocapSMPLClientActionsEnum.UPLOAD, data)
        num_frames = self.__parse_upload_response(resp_data)

        return num_frames

    def __parse_get_faces_response(self, data):
        success = False
        if self.resp_type == 'bytes':
            if not isinstance(data, dict):
                success = True
                faces = np.frombuffer(data, dtype=np.int32).reshape((-1, 3)).tolist()
            else:
                if data['status'] == 'success':
                    success = True
                    faces = data['faces']
        
        if not success:
            msg = data['msg']
            self.logger.error(f'Get faces failed.\n{msg}')
            self.close()
        
        self.logger.info('Get faces success.')

        return faces
            
    def get_faces(self):
        self.logger.info(f'Requesting faces...')
        resp_data = self.socketio_client.call(XRMocapSMPLClientActionsEnum.GET_FACES)
        faces = self.__parse_get_faces_response(resp_data)

        return faces

    def __parse_forward_response(self, data):
        success = False
        if self.resp_type == 'bytes':
            if not isinstance(data, dict):
                success = True
                verts = np.frombuffer(data, dtype=np.float16)
        else:
            if data['status'] == 'success':
                success = True
                verts = np.asarray(data['verts'])
        if success:
            verts = verts.reshape(-1, 3)
            assert verts.shape == (6890, 3)
            self.logger.info('Forward success.')
        else:
            msg = data['msg']
            self.logger.error(f'Forward failed.\n{msg}')

        return verts.tolist()

    def forward(self, frame_idx: int):
        self.logger.info(f'Requesting frame {frame_idx}...')

        resp_data = self.socketio_client.call(XRMocapSMPLClientActionsEnum.FORWARD, {'frame_idx': frame_idx})
        verts = self.__parse_forward_response(resp_data)

        return verts

    def close(self):
        self.socketio_client.disconnect()