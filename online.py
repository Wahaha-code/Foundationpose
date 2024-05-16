from estimater import *
from datareader_online import *
import argparse
import pyrealsense2 as rs
import time
from ultralytics import YOLO
import numpy as np
import cv2
# 初始化 VideoWriter（移到循环外）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编解码器
# 获取当前时间并格式化为字符串（例如："20230513_123456"）
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 使用当前时间戳作为视频文件名
out = cv2.VideoWriter(f'output_{current_time}.mp4', fourcc, 20.0, (640, 480))


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/demo/mesh/textured_mesh.obj')
  # parser.add_argument('--mesh_file', type=str, default=f'/home/guo/project/FoundationPose/demo_data/taotou/mesh/tantou.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/demo')
  parser.add_argument('--rgb_dir', type=str, default=f'{code_dir}/demo_data/demo/rgb')
  parser.add_argument('--depth_dir', type=str, default=f'{code_dir}/demo_data/demo/depth')

  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  model = YOLO('best.pt')
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")
  os.makedirs(args.rgb_dir, exist_ok=True)
  os.makedirs(args.depth_dir, exist_ok=True)
  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

  pipeline.start(config)

  align_to = rs.stream.color
  align = rs.align(align_to)
  start_time = time.time()
  i = 0
  try:
    while i<500:
        frame_start_time = time.time()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()

        # Directly use the in-memory images for processing
        color_image = np.asanyarray(color_frame.get_data())[...,:3]
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Resize and normalize the depth image
        depth_image = depth_image / 1000.0  # Convert depth to meters
        depth_image = cv2.resize(depth_image, (reader.W, reader.H), interpolation=cv2.INTER_NEAREST)
        depth_image[(depth_image < 0.1) | (depth_image > reader.zfar)] = 0

        color = cv2.resize(color_image, (reader.W, reader.H), interpolation=cv2.INTER_NEAREST)

        # Process the data as needed
        if i == 0:
            results = model(color_image)
            for result in results:
                if result.masks is not None:
                    # Assuming result.masks[0] is a 3D mask with only one channel
                    mask_raw = result.masks[0].cpu().data.numpy().transpose(1, 2, 0)
                    for mask in result.masks[1:]:  # Start from second mask to avoid adding the first mask twice
                        mask_raw += mask.cpu().data.numpy().transpose(1, 2, 0)

                    # Normalize the mask data to 0-255 scale
                    mask_scaled = (mask_raw * 255).astype(np.uint8)
                    cv2.imwrite("result.png", mask_scaled)
            mask = reader.get_mask(mask_scaled).astype(bool)  # Assuming you still want to use masks
            pose = est.register(K=reader.K, rgb=color_image, depth=depth_image, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color_image, depth=depth_image, K=reader.K, iteration=args.track_refine_iter)

        i += 1
        if debug>=3:
          m = mesh.copy()
          m.apply_transform(pose)
          m.export(f'{debug_dir}/model_tf.obj')
          xyz_map = depth2xyzmap(depth_image, reader.K)
          valid = depth_image>=0.1
          pcd = toOpen3dCloud(xyz_map[valid], color[valid])
          o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
          pose = est.track_one(rgb=color, depth=depth_image, K=reader.K, iteration=args.track_refine_iter)

        os.makedirs(f'{reader.video_dir}/ob_in_cam', exist_ok=True)

        if debug>=1:
          center_pose = pose@np.linalg.inv(to_origin)
          vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
          vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
          out.write(vis)  # 注意：颜色转换BGR到RGB
          cv2.imshow('1', vis)
          cv2.waitKey(1)


        if debug>=2:
          os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
          imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)

        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time

        # 计算 FPS
        fps = 1.0 / frame_processing_time

        # 输出 FPS 到控制台
        print("FPS:", fps)
  finally:
      pipeline.stop()
      cv2.destroyAllWindows()