type = 'SMPLVertsService'
name = 'smpl_verts_service'
work_dir = f'temp/{name}'
body_model_dir = 'xrmocap_data/body_models'
device = 'cuda:0'
enable_bytes = True
enable_cors = True
max_http_buffer_size = 128 * 1024 * 1024
