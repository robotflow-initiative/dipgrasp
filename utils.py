import sys
import math
import timeout_decorator
import numpy as np
import open3d as o3d
import os.path as osp

from typing import Optional, List, Dict
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv

def load_obj(obj_file: str, point_num = 2048) -> o3d.geometry.PointCloud:
    try:
        mesh = o3d.io.read_triangle_mesh(obj_file)
        mesh.compute_vertex_normals()
        point_cloud = mesh.sample_points_uniformly(number_of_points=point_num, use_triangle_normal=True)
        return point_cloud
    except:
        point_cloud = o3d.io.read_point_cloud(obj_file)
        point_cloud.estimate_normals()
        point_cloud.farthest_point_down_sample(point_num)
        return point_cloud


def export_meshes(mesh_list: List[Dict[str, np.ndarray]],
                  file_name: str):
    lines = ['# This is the intermediate obj file produced by DipGrasp\n']
    total_v = 1
    for idx, mesh in enumerate(mesh_list):
        vertices = mesh['vertices']
        faces = mesh['faces']
        for i in range(vertices.shape[0]):
            lines.append('v %.6f %.6f %.6f\n' % (vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        lines.append('\n')
        lines.append(f'o part{idx}\n')
        lines.append(f'g part{idx}\n')
        lines.append('\n')
        for i in range(faces.shape[0]):
            lines.append('f %d %d %d\n' % (faces[i, 0] + total_v, faces[i, 1] + total_v, faces[i, 2] + total_v))
        total_v += vertices.shape[0]
        lines.append('\n')
    with open(file_name, 'w') as f:
        f.writelines(lines)

def visualize_pcd(coordinate, rgb):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Vis')
    vis.get_render_option().point_size = 1
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinate)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def load_object_pointcloud(area:str, scene_idx:int, use_GT = False, compute_normal = True):
    root_path = 'mask3d'
    scene_idx = '%04d'% scene_idx
    input_scene = osp.join(root_path, f'{area}_scene_{scene_idx}_pcd.txt')
    output_scene = osp.join(root_path, f'{area}_scene_{scene_idx}.txt')
    full_pcd = None
    with open(input_scene) as f:
        full_pcd = np.loadtxt(f)
    objects_point = []
    objects_rgb = []
    objects_normal = []
    if use_GT:
        for i in range(5):
            gt_mask_path = osp.join(root_path, 'gt_mask', f'{area}_scene_{scene_idx}_{i}.txt')
            gt_mask = np.loadtxt(gt_mask_path)
            assert gt_mask.shape[0] == full_pcd.shape[0]
            gt_mask = gt_mask > 0
            gt_object = full_pcd[gt_mask]
            points = gt_object[:, :3]
            rgb = gt_object[:, 3:]
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(points)
            tmp.estimate_normals()
            normals = np.asarray(tmp.normals)
            objects_point.append(points)
            objects_rgb.append(rgb)
            objects_normal.append(normals)
    else:
        with open(output_scene) as f:
            for line in f:
                mask_path, pred, score = line.split(' ')
                mask_path = osp.join(root_path, mask_path)
                pred = int(pred)
                score = float(score)
                mask = np.loadtxt(mask_path, dtype=int)
                assert mask.shape[0] == full_pcd.shape[0]
                mask = mask > 0
                object_pcd = full_pcd[mask]
                points = object_pcd[:, :3]
                rgb = object_pcd[:, 3:]
                tmp = o3d.geometry.PointCloud()
                tmp.points = o3d.utility.Vector3dVector(points)
                tmp.estimate_normals()
                normals = np.asarray(tmp.normals)
                objects_point.append(points)
                objects_rgb.append(rgb)
                objects_normal.append(normals)
    if compute_normal:
        return full_pcd, objects_point, objects_rgb, objects_normal
    else:
        return full_pcd, objects_point, objects_rgb
@timeout_decorator.timeout(600)
def rfu_filter(poses: np.ndarray,
               joints: np.ndarray,
               gripper_name: str,
               datafile: str) -> [np.ndarray, np.ndarray]:

    success = None
    joint_state = None

    def ReceiveData(obj: list):
        nonlocal success
        nonlocal joint_state
        nonlocal done
        success = obj[0]
        joint_state = obj[1]
        done = True

    env_path = ''
    if sys.platform.startswith("linux"):
        env_path = "./assets/rfu/linux/GraspTest.x86_64"
    elif sys.platform.startswith("win"):
        env_path = "./assets/rfu/windows/GraspTest.x86_64"
    else:
        raise Exception("Unsupported platform")

    env = RFUniverseBaseEnv(env_path,
                            communication_backend="grpc",
                            graphics=False,
                            log_level=1)

    env.AddListenerObject('Result', ReceiveData)
    env.SetTimeScale(3)
    env.SetTimeStep(0.02)

    result_success = np.empty((poses.shape[0]), dtype=bool)
    result_joint_state = np.empty((joints.shape))

    chunk_count = poses.shape[0]
    chunk = poses.shape[0] / chunk_count
    chunk = math.ceil(chunk)

    for i in range(chunk):
        start = i * chunk_count
        end = (i + 1) * chunk_count
        if end > poses.shape[0]:
            end = poses.shape[0]
        pose_chunk = poses[start:end]
        joint_chunk = joints[start:end]

        env.SendObject(
            'Test',
            datafile,
            gripper_name,
            pose_chunk.reshape(-1).tolist(),
            joint_chunk.reshape(-1).tolist(),
            50,
            True,
            True,
        )
        done = False
        while not done:
            env.step()

        result_success[start:end] = np.array(success)
        result_joint_state[start:end] = np.array(joint_state)

    result_joint_state = result_joint_state[result_success]

    true_count = sum(result_success)
    total_count = len(result_success)

    if total_count == 0:
        total_count = 1

    print(f"{true_count}/{total_count} = {(true_count / total_count) * 100}%")

    env.close()

    poses = poses[result_success]
    joints = result_joint_state * np.pi / 180.0

    return poses, joints

def downsample_pcd(points, rgb, normal, sample_num = 2000):
    assert points.shape[0] == rgb.shape[0]
    assert points.shape[0] == normal.shape[0]
    n = points.shape[0]

    sample_idx = np.random.permutation(n)[:sample_num]
    return points[sample_idx], rgb[sample_idx], normal[sample_idx]

def visualize_point_cloud(cloud: np.ndarray, sample_point: Optional[np.ndarray] = None) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Vis')
    vis.get_render_option().point_size = 1
    opt = vis.get_render_option()
    opt.background_color = np.zeros((3,))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.paint_uniform_color(np.ones((3,)))
    vis.add_geometry(pcd)
    if sample_point is not None:
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(sample_point)
        pt.paint_uniform_color([0, 1, 0])
        vis.add_geometry(pt)
    vis.run()
    vis.destroy_window()

def link_points(SrcPC: np.ndarray, DstPC: np.ndarray, thread_point = 50) -> np.ndarray:
    # links the corresponding points between SrcPC and DstPC
    # the function returns points imitating the thread bewteen each point-pair

    n = SrcPC.shape[0]
    t = np.random.rand(n, thread_point)
    t = np.expand_dims(t, axis = 2)
    pts1 = np.expand_dims(SrcPC, axis = 1)
    pts1 = np.tile(pts1, (1, thread_point, 1))
    pts2 = np.expand_dims(DstPC, axis = 1)
    pts2 = np.tile(pts2, (1, thread_point, 1))
    pts1 = t * pts1
    pts2 = (np.ones_like(t) - t) * pts2
    pts1 = np.reshape(pts1, (-1, 3), order = 'C')
    pts2 = np.reshape(pts2, (-1, 3), order = 'C')
    return pts1 + pts2
