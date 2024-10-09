import torch
import urdfpy
import trimesh
import os.path as osp
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from xml.dom.minidom import parse
from typing import Union, Tuple, Dict

class Gripper:
    def __init__(self,
                 name: str = 'shadow',
                 sample_point_num: int = 400,
                 gripper_dir: str = osp.join('assets', 'gripper')):

        # Member Variables

        self.name = None
        self.bbox = None
        self.robot = None
        self.links = None
        self.joints = None
        self.base_link = None
        self.is_distal = None
        self.joint_num = None
        self.link_parent = None
        self.joint_parent = None

        # Initial Process

        self.name = name
        self.robot = urdfpy.URDF.load(osp.join(gripper_dir, f'{name}.urdf'))
        self.base_link = self.robot.links[2 if name == 'shadow' else 1]
        self.init_link_and_joint()
        self.sample_point_number = sample_point_num

    def init_link_and_joint(self) -> None:
        base_link = self.base_link
        robot = self.robot

        link_parent = dict()
        joint_parent = dict()
        joint2num = dict()

        links = list()
        joints = list()
        link_not_distal = set()
        link_not_distal.add(base_link)

        # select the links on subtree under the base_link
        for link in robot.links:
            if base_link in robot._paths_to_base[link]:
                links.append(link)

        # get the parent-child relation between links
        for link in links:
            if link == base_link:
                continue
            path = robot._paths_to_base[link]
            parent_link = path[1]
            joint = robot._G.get_edge_data(link, parent_link)['joint']
            link_parent[link] = parent_link
            link_not_distal.add(parent_link)
            joint_parent[link] = joint
            if joint.mimic is None and joint.joint_type != 'fixed':
                joints.append(joint)
        link_parent[base_link] = None

        # use the Topological Sorting to get a queue of links
        links_queue = list()
        # manual stack
        s = list()
        for link in links:
            if link not in links_queue:
                s.append(link)
                while len(s) > 0:
                    top = s[-1]
                    if not link_parent[top] or link_parent[top] in links_queue:
                        s.pop()
                        links_queue.append(top)
                    else:
                        s.append(link_parent[top])
        for i in range(len(joints)):
            joint2num[joints[i]] = i

        # judge if a link is distal
        is_distal = []
        for i in range(len(links_queue)):
            if links_queue[i] not in link_not_distal:
                is_distal.append(True)
            else:
                is_distal.append(False)

        self.link_parent = link_parent
        self.joint_parent = joint_parent
        self.joint2num = joint2num
        self.links = links_queue
        self.joints = joints
        self.is_distal = is_distal

    def get_link_pcd_from_xml(self, xml_file: str = None) -> Tuple[
        Dict[urdfpy.Link, np.ndarray], Dict[urdfpy.Link, np.ndarray]]:
        if xml_file is None:
            xml_file = osp.join('assets', 'gripper', f'{self.name}.xml')

        name2link = dict()
        points = dict()
        normals = dict()
        links = self.links

        DOMTree = parse(xml_file)
        collection = DOMTree.documentElement
        links_data = collection.getElementsByTagName("PointCloudLinkData")
        for link in links:
            name2link[link.name] = link
            points[link] = np.asarray([])
            normals[link] = np.asarray([])
        for link_data in links_data:
            name = link_data.getElementsByTagName('linkName')[0].childNodes[0].data
            if name not in name2link.keys(): continue
            link = name2link[name]
            point_tmp = list()
            normal_tmp = list()
            point_data = link_data.getElementsByTagName('points')[0].getElementsByTagName('Vector3')
            normal_data = link_data.getElementsByTagName('normal')[0].getElementsByTagName('Vector3')
            for point in point_data:
                x = point.getElementsByTagName('x')[0].childNodes[0].data
                y = point.getElementsByTagName('y')[0].childNodes[0].data
                z = point.getElementsByTagName('z')[0].childNodes[0].data
                point_tmp.append([float(z), -float(x), float(y)])
            for normal in normal_data:
                x = normal.getElementsByTagName('x')[0].childNodes[0].data
                y = normal.getElementsByTagName('y')[0].childNodes[0].data
                z = normal.getElementsByTagName('z')[0].childNodes[0].data
                normal_tmp.append([float(z), -float(x), float(y)])
            points[link] = np.asarray(point_tmp)
            normals[link] = np.asarray(normal_tmp)
        return points, normals

    def get_link_pcd_from_mesh(self) -> Tuple[Dict[urdfpy.Link, np.ndarray], Dict[urdfpy.Link, np.ndarray]]:
        scale = 1
        if self.name == 'shadow':
            scale = 0.001

        link_points = dict()
        link_normals = dict()
        links = self.links
        self.bbox = dict()

        for link in links:
            (link_points[link], link_normals[link]) = self.link_point_sample(link, self.sample_point_number)
            link_points[link] *= scale

            if link.name == "panda_rightfinger":
                link_points[link][:, 1] = -link_points[link][:, 1]
        return link_points, link_normals

    def link_point_sample(self, link: urdfpy.Link,
                          sample_point_number: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        scale = 1
        if self.name == 'shadow':
            scale = 0.001
        if len(link.visuals) == 0:
            print(link.name, "No collision mesh here!")
            return np.asarray([]), np.asarray([])
        geometry = link.visuals[0].geometry
        tf = link.visuals[0].origin
        if geometry.meshes:
            bbox_ = trimesh.bounds.oriented_bounds(geometry.meshes[0])
            transformation = bbox_[0]
            transformation[:3, 3] *= scale
            bound = bbox_[1] * scale
            self.bbox[link] = [transformation, bound]
        if geometry.meshes is not None:
            mesh = geometry.meshes[0]
            v_np = np.asarray(mesh.vertices)
            f_np = np.asarray(mesh.faces)
            # NOTE: apply the local origin transformation of the link visual element
            v_np = v_np @ tf[:3, :3].T + tf[:3, 3]
            v = o3d.utility.Vector3dVector(v_np)
            f = o3d.utility.Vector3iVector(f_np)
            mesh = o3d.geometry.TriangleMesh(v, f)
            mesh.compute_vertex_normals()
            PC = mesh.sample_points_uniformly(number_of_points=sample_point_number, use_triangle_normal=True)
            points = np.asarray(PC.points)
            normals = np.asarray(PC.normals)
            return points, normals
        return np.asarray([]), np.asarray([])

    def compute_pcd(self,
                    Tbase: torch.Tensor,
                    joint_para: torch.Tensor,
                    link_points: Dict[urdfpy.Link, np.ndarray],
                    link_normals: Dict[urdfpy.Link, np.ndarray],
                    compute_w: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:

        robot = self.robot
        links = self.links
        joints = self.joints
        joint2num = self.joint2num
        base_link = self.base_link
        link_parent = self.link_parent
        joint_parent = self.joint_parent

        T = dict()
        T[base_link] = Tbase
        device = joint_para.device
        batch_size = Tbase.shape[0]

        # compute transform matrix with respect to world coordinate
        for link in links[1:]:
            joint = joint_parent[link]
            parent = link_parent[link]
            cfg = None

            # judge if the joint is mimic
            if joint.mimic:
                mimic_joint = robot._joint_map[joint.mimic.joint]
                if mimic_joint in joints:
                    cfg = joint_para[:, joint2num[mimic_joint]]
                    cfg = joint.mimic.multiplier * cfg + joint.mimic.offset
            elif joint.joint_type != 'fixed':
                cfg = joint_para[:, joint2num[joint]]
            if isinstance(cfg, torch.Tensor):
                cfg = cfg.to(device)
            origin = torch.Tensor(joint.origin).type(torch.float).to(device)
            if cfg is None or joint.joint_type == 'fixed':
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
            elif joint.joint_type in ['revolute', 'continuous']:
                R = r_matrix(cfg, torch.Tensor(joint.axis).to(device))
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
                pose = torch.einsum('ijk,ikl->ijl', pose, R)
            elif joint.joint_type == 'prismatic':
                translation = torch.tile(torch.eye(4)[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
                tmp = torch.einsum('ij, i->ij', torch.tile(torch.Tensor(joint.axis[np.newaxis, :]).to(device), (batch_size, 1)), cfg)
                translation[:, :3, 3] = tmp
                pose = torch.tile(origin.unsqueeze(0), (batch_size, 1, 1))
                pose = torch.einsum('ijk,ikl->ijl', pose, translation)
            else:
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
            pose = torch.einsum('ijk,ikl->ijl', T[parent], pose)
            T[link] = pose
        fk = T

        # compute the point cloud of each link
        whole_points = torch.asarray([])
        whole_normals = torch.asarray([])
        whole_weights = torch.asarray([])
        for i in range(len(links)):
            link = links[i]
            fk_matrix = fk[link]
            if len(link_points[link]) == 0: continue

            rotation_matrix = fk_matrix[:, :3, :3]
            translation_matrix = fk_matrix[:, :3, 3].unsqueeze(1)
            points = torch.einsum('ijk, ilk->ijl',
                                  torch.tile(torch.Tensor(link_points[link][np.newaxis, :, :]), (batch_size, 1, 1)).to(
                                      device), rotation_matrix) + translation_matrix
            normals = torch.einsum('ijk, ilk->ijl', torch.tile(torch.Tensor(link_normals[link][np.newaxis, :, :]),
                                                               (batch_size, 1, 1)).to(device), rotation_matrix)
            weights = torch.ones((batch_size, points.shape[1])) * (1 if self.is_distal[i] else 0.2)
            weights = weights.to(device)
            if len(whole_points) == 0:
                whole_points = points
                whole_normals = normals
                whole_weights = weights
            else:
                whole_points = torch.cat((whole_points, points), dim=1)
                whole_normals = torch.cat((whole_normals, normals), dim=1)
                whole_weights = torch.cat((whole_weights, weights), dim=1)
        if compute_w:
            return whole_points, whole_normals, whole_weights
        else:
            return whole_points, whole_normals, None

    def compute_relative_matrix(self,
                                Tbase: torch.Tensor,
                                joint_para: torch.Tensor) -> Dict[urdfpy.Link, torch.Tensor]:
        # Compute forward kinematics in reverse topological order
        device = Tbase.device
        batch_size = Tbase.shape[0]
        robot = self.robot
        links = self.links
        joints = self.joints
        base_link = self.base_link
        joint2num = self.joint2num
        link_parent = self.link_parent
        joint_parent = self.joint_parent

        T = {}
        T[base_link] = Tbase
        for link in links[1:]:
            joint = joint_parent[link]
            parent = link_parent[link]
            cfg = None
            if joint.mimic:
                mimic_joint = robot._joint_map[joint.mimic.joint]
                if mimic_joint in joints:
                    cfg = joint_para[:, joint2num[mimic_joint]]
                    cfg = joint.mimic.multiplier * cfg + joint.mimic.offset
            elif joint.joint_type != 'fixed':
                cfg = joint_para[:, joint2num[joint]]
            if isinstance(cfg, torch.Tensor):
                cfg = cfg.to(device)
            origin = torch.Tensor(joint.origin).type(torch.float).to(device)
            if cfg is None or joint.joint_type == 'fixed':
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
            elif joint.joint_type in ['revolute', 'continuous']:
                R = r_matrix(cfg, torch.Tensor(joint.axis).to(device))
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
                pose = torch.einsum('ijk,ikl->ijl', pose, R)
            elif joint.joint_type == 'prismatic':
                translation = torch.tile(torch.eye(4)[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
                tmp = torch.einsum('ij, i->ij', torch.tile(torch.Tensor(joint.axis[np.newaxis, :]).to(device), (batch_size, 1)), cfg)
                translation[:, :3, 3] = tmp
                pose = torch.tile(origin.unsqueeze(0), (batch_size, 1, 1))
                pose = torch.einsum('ijk,ikl->ijl', pose, translation)
            else:
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
            pose = torch.einsum('ijk,ikl->ijl', T[parent], pose)
            T[link] = pose
        return T

def r_matrix(angle: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    device = angle.device
    sina = torch.sin(angle).to(device)
    cosa = torch.cos(angle).to(device)
    batch_size = cosa.shape[0]
    direction = direction[:3] / torch.linalg.norm(direction[:3])
    # rotation matrix around unit vector
    M = torch.einsum('ijk,i->ijk', torch.tile(torch.eye(4)[np.newaxis, :, :], (batch_size, 1, 1)).to(device), cosa)
    M[:, 3, 3] = torch.ones_like(cosa)
    M += F.pad(torch.einsum('ijk,i->ijk',
                            torch.tile(torch.outer(direction, direction)[np.newaxis, :, :], (batch_size, 1, 1)),
                            torch.ones_like(cosa) - cosa), (0, 1, 0, 1, 0, 0), 'constant', value=0)
    direction = torch.einsum('ij, i->ij', torch.tile(direction[np.newaxis, :], (batch_size, 1)), sina)
    tmp_matrix = torch.zeros((batch_size, 4, 4)).to(device)
    tmp_matrix[:, 0, 1] = -direction[:, 2]
    tmp_matrix[:, 0, 2] = direction[:, 1]
    tmp_matrix[:, 1, 0] = direction[:, 2]
    tmp_matrix[:, 1, 2] = -direction[:, 0]
    tmp_matrix[:, 2, 0] = -direction[:, 1]
    tmp_matrix[:, 2, 1] = direction[:, 0]
    M += tmp_matrix
    return M.to(angle.dtype)

