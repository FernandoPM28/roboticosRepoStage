import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import numpy as np

PATH_DEFINITION = [
    # Caminho para o primeiro objetivo principal (7, 7)
    {'type': 'sub',  'pos': np.array([-2.0, -5.0])}, 
    {'type': 'sub',  'pos': np.array([-2.0, 4.0])}, 
    {'type': 'sub',  'pos': np.array([2.5, 4.0])},   
    {'type': 'sub',  'pos': np.array([3.0, 7.5])},
    {'type': 'sub',  'pos': np.array([4.0, 7.5])},  
    {'type': 'main', 'pos': np.array([7.0, 7.0])},   
    # Caminho para o segundo objetivo principal (7, -3)
    {'type': 'sub',  'pos': np.array([7.0, 4.0])},
    {'type': 'sub',  'pos': np.array([3.5, 4.5])}, 
    {'type': 'sub',  'pos': np.array([-2.0, 4.0])},
    {'type': 'sub',  'pos': np.array([-2.0, -3.0])},        
    {'type': 'main', 'pos': np.array([7.0, -3.0])} 
]
WAIT_DURATION = 5.0 

# --- Constantes Gerais de Configuração ---
INITIAL_WORLD_POS = np.array([-7.0, -7.0])
INITIAL_WORLD_ORIENTATION = math.pi / 4.0
GOAL_THRESHOLD = 0.25 
MAX_LIDAR_RANGE = 10.0
ROBOT_WIDTH = 0.4
MIN_SCAN_RANGE = 0.1

# --- Parâmetros do Algoritmo Tangent Bug ---
GOAL_OBSTRUCTION_RANGE = 0.5
WALL_FOLLOW_DISTANCE = 0.6
LEAVE_PATH_CLEARANCE = ROBOT_WIDTH * 2.0
GOAL_PROXIMITY_THRESHOLD = 0.5
LEAVE_BOUNDARY_TOLERANCE = 0.2

# --- Parâmetros de Controle ---
LINEAR_SPEED_WALL = 0.15
ANGULAR_GAIN_WALL_DIST = 2.0
ANGULAR_GAIN_WALL_ALIGN = 1.5

# Parâmetros para o estado ACQUIRING_WALL
ACQUIRING_WALL_ANGULAR_SPEED = 0.5
ACQUIRE_WALL_RECEDE_DISTANCE = 0.1
ACQUIRE_WALL_RECEDE_SPEED = 0.1
ACQUIRE_WALL_ALIGN_GAIN_P = 2.5
ACQUIRE_WALL_ALIGN_TOLERANCE_ANGLE = 0.1
ACQUIRE_WALL_ALIGN_TOLERANCE_DIST = 0.05
ACQUIRE_WALL_FORWARD_CREEP_SPEED = 0.03

# --- Parâmetros de Navegação em Espaço Livre ---
MAX_LINEAR_SPEED_GOAL = 0.4
MIN_LINEAR_SPEED_GOAL = 0.05
PROXIMITY_DAMPING_RANGE = 1.8

class TangentBugNavigator(Node):
    def __init__(self):
        super().__init__('tangent_bug_navigator_final')
        self.state = 'IDLE'
        
        self.path = PATH_DEFINITION
        self.current_path_index = 0
        self.active_goal = self.path[self.current_path_index]['pos']
        self.wait_start_time = None
        
        self.distance_at_hit_point = float('inf')
        self.wall_follow_direction = 1
        self.m_line_start_point = None
        
        self.initial_odom_pos = None
        self.initial_odom_orientation = None
        self.current_world_pos = INITIAL_WORLD_POS.copy()
        self.current_orientation = INITIAL_WORLD_ORIENTATION
        self.laser_scan = None
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)
        self.laser_sub = self.create_subscription(LaserScan, '/base_scan', self.laser_callback, qos_profile)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navegador Tangent Bug (Revisado) Iniciado.")
        self.get_logger().info(f"Primeiro objetivo: {self.active_goal}")

    def odom_callback(self, msg):
        current_odom_pos_raw = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        _, _, current_odom_orientation_raw = self.euler_from_quaternion(msg.pose.pose.orientation)

        if self.initial_odom_pos is None:
            self.initial_odom_pos = current_odom_pos_raw.copy()
            self.initial_odom_orientation = current_odom_orientation_raw
            self.state = 'MOTION_TO_GOAL'
            self.get_logger().info("Odometria recebida. Iniciando navegacao.")
            return

        odom_pos_delta = current_odom_pos_raw - self.initial_odom_pos
        cos_initial = math.cos(INITIAL_WORLD_ORIENTATION)
        sin_initial = math.sin(INITIAL_WORLD_ORIENTATION)
        
        dx_world = odom_pos_delta[0] * cos_initial - odom_pos_delta[1] * sin_initial
        dy_world = odom_pos_delta[0] * sin_initial + odom_pos_delta[1] * cos_initial
        rotated_pos_delta = np.array([dx_world, dy_world])
        
        self.current_world_pos = INITIAL_WORLD_POS + rotated_pos_delta
        
        odom_orient_delta = self.normalize_angle(current_odom_orientation_raw - self.initial_odom_orientation)
        self.current_orientation = self.normalize_angle(INITIAL_WORLD_ORIENTATION + odom_orient_delta)

    def laser_callback(self, msg):
        proc_ranges = []
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r) or r < MIN_SCAN_RANGE:
                proc_ranges.append(MAX_LIDAR_RANGE)
            else:
                proc_ranges.append(r)
        msg.ranges = proc_ranges
        self.laser_scan = msg

    def control_loop(self):
        if self.state == 'IDLE' or self.laser_scan is None or self.initial_odom_pos is None:
            self.stop_robot()
            return

        dist_to_goal = np.linalg.norm(self.active_goal - self.current_world_pos)
        
        if (self.state == 'MOTION_TO_GOAL' or self.state == 'BOUNDARY_FOLLOWING') and dist_to_goal < GOAL_THRESHOLD:
            current_waypoint = self.path[self.current_path_index]
            self.get_logger().info(f"Waypoint do tipo '{current_waypoint['type']}' em {current_waypoint['pos']} alcancado.")

            if self.current_path_index >= len(self.path) - 1:
                self.state = 'MISSION_COMPLETE'
                self.stop_robot()
            else:
                if current_waypoint['type'] == 'main':
                    self.state = 'WAITING_AT_WAYPOINT'
                    self.wait_start_time = self.get_clock().now()
                    self.stop_robot()
                elif current_waypoint['type'] == 'sub':
                    self.current_path_index += 1
                    self.active_goal = self.path[self.current_path_index]['pos']
                    self.get_logger().info(f"Avancando para proximo waypoint: {self.active_goal}")
                    if self.state == 'BOUNDARY_FOLLOWING':
                         self.get_logger().info("Sub-goal reached while following boundary. Transitioning to MOTION_TO_GOAL.")
                         self.state = 'MOTION_TO_GOAL'
                    self.reset_bug_vars()

        if self.state == 'MOTION_TO_GOAL':
            self.motion_to_goal()
        elif self.state == 'ACQUIRING_WALL':
            self.acquiring_wall()
        elif self.state == 'BOUNDARY_FOLLOWING':
            self.boundary_following()
        elif self.state == 'WAITING_AT_WAYPOINT':
            self.waiting_at_waypoint()
        elif self.state == 'MISSION_COMPLETE':
            self.get_logger().info("MISSAO COMPLETA! Todos os waypoints foram alcancados.")
            self.stop_robot()
            self.control_timer.cancel()
            return

    def waiting_at_waypoint(self):
        if self.wait_start_time is None: return
        elapsed_time = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
        
        if elapsed_time > WAIT_DURATION:
            self.get_logger().info("Tempo de espera concluido.")
            self.current_path_index += 1
            self.active_goal = self.path[self.current_path_index]['pos']
            self.get_logger().info(f"Indo para o proximo objetivo: {self.active_goal}")
            self.state = 'MOTION_TO_GOAL'
            self.reset_bug_vars()

    def motion_to_goal(self):
        if self.laser_scan is None: return
        
        delta = self.active_goal - self.current_world_pos
        dist_to_goal = np.linalg.norm(delta)
        angle_to_goal_world = math.atan2(delta[1], delta[0])
        angle_error = self.normalize_angle(angle_to_goal_world - self.current_orientation)

        frontal_cone_angle = math.atan2(ROBOT_WIDTH / 2.0, GOAL_OBSTRUCTION_RANGE) * 2.5
        min_dist_front, angle_to_obstacle_relative = self.get_min_dist_and_angle_in_cone(
            -frontal_cone_angle / 2.0, frontal_cone_angle / 2.0
        )

        if min_dist_front < GOAL_OBSTRUCTION_RANGE and dist_to_goal > GOAL_PROXIMITY_THRESHOLD:
            self.get_logger().warn(f"Caminho BLOQUEADO a {min_dist_front:.2f}m. Iniciando Tangent Bug.")
            self.state = 'ACQUIRING_WALL'
            self.distance_at_hit_point = dist_to_goal
            self.m_line_start_point = self.current_world_pos.copy()
            
            if angle_to_obstacle_relative > 0:
                self.wall_follow_direction = 1
                self.get_logger().info(f"Obstaculo a esquerda ({angle_to_obstacle_relative:.2f} rad). Contornando pela ESQUERDA (robo vira a direita).")
            else:
                self.wall_follow_direction = -1
                self.get_logger().info(f"Obstaculo a direita ({angle_to_obstacle_relative:.2f} rad). Contornando pela DIREITA (robo vira a esquerda).")
            return

        target_linear_speed = MAX_LINEAR_SPEED_GOAL
        min_overall_front_dist = self.get_min_dist_in_cone(-math.pi/3.0, math.pi/3.0) 
        if min_overall_front_dist < PROXIMITY_DAMPING_RANGE:
            speed_ratio = max(0, (min_overall_front_dist - MIN_SCAN_RANGE) / (PROXIMITY_DAMPING_RANGE - MIN_SCAN_RANGE))
            target_linear_speed = MIN_LINEAR_SPEED_GOAL + (MAX_LINEAR_SPEED_GOAL - MIN_LINEAR_SPEED_GOAL) * speed_ratio
        
        twist = Twist()
        twist.linear.x = float(np.clip(target_linear_speed, MIN_LINEAR_SPEED_GOAL, MAX_LINEAR_SPEED_GOAL))
        twist.angular.z = float(np.clip(1.8 * angle_error, -1.2, 1.2))
        self.vel_pub.publish(twist)

    def acquiring_wall(self):
        if self.laser_scan is None: return
        twist = Twist()

        dist_to_wall_candidate, angle_to_wall_candidate_rad = self.find_wall_to_follow()

        if dist_to_wall_candidate is None:
            self.get_logger().warn("ACQUIRING_WALL: Nao foi possivel encontrar uma parede para seguir. Voltando para MOTION_TO_GOAL.")
            self.state = 'MOTION_TO_GOAL'
            self.reset_bug_vars()
            return

        if dist_to_wall_candidate < (WALL_FOLLOW_DISTANCE - ACQUIRE_WALL_RECEDE_DISTANCE):
            self.get_logger().info(f"ACQUIRING_WALL: Muito proximo da parede ({dist_to_wall_candidate:.2f}m). Recuando.")
            twist.linear.x = -ACQUIRE_WALL_RECEDE_SPEED
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            return

        target_angle_to_wall_point = (math.pi / 2.0) * self.wall_follow_direction
        angle_error_to_wall_orientation = self.normalize_angle(angle_to_wall_candidate_rad - target_angle_to_wall_point)

        is_aligned = abs(angle_error_to_wall_orientation) < ACQUIRE_WALL_ALIGN_TOLERANCE_ANGLE
        is_at_correct_distance = abs(dist_to_wall_candidate - WALL_FOLLOW_DISTANCE) < ACQUIRE_WALL_ALIGN_TOLERANCE_DIST
        
        if is_aligned and is_at_correct_distance:
            self.get_logger().info(f"ACQUIRING_WALL: Parede adquirida e alinhada. Dist: {dist_to_wall_candidate:.2f}m, AngleErr: {angle_error_to_wall_orientation:.2f}rad. Iniciando BOUNDARY_FOLLOWING.")
            self.state = 'BOUNDARY_FOLLOWING'
            self.stop_robot()
            return

        twist.angular.z = ACQUIRE_WALL_ALIGN_GAIN_P * angle_error_to_wall_orientation
        twist.angular.z = float(np.clip(twist.angular.z, -ACQUIRING_WALL_ANGULAR_SPEED, ACQUIRING_WALL_ANGULAR_SPEED))

        if dist_to_wall_candidate > WALL_FOLLOW_DISTANCE + ACQUIRE_WALL_ALIGN_TOLERANCE_DIST :
            twist.linear.x = ACQUIRE_WALL_FORWARD_CREEP_SPEED
        elif dist_to_wall_candidate < WALL_FOLLOW_DISTANCE - ACQUIRE_WALL_ALIGN_TOLERANCE_DIST:
             twist.linear.x = -ACQUIRE_WALL_FORWARD_CREEP_SPEED * 0.5
        else:
            twist.linear.x = ACQUIRE_WALL_FORWARD_CREEP_SPEED * 0.5

        if abs(twist.angular.z) > ACQUIRING_WALL_ANGULAR_SPEED * 0.7:
            twist.linear.x *= 0.5

        self.vel_pub.publish(twist)

    def boundary_following(self):
        if self.laser_scan is None: return

        min_dist_wall, angle_to_wall_obstacle_rad = self.find_wall_to_follow()
        
        if min_dist_wall is None or min_dist_wall > WALL_FOLLOW_DISTANCE * 2.5:
            self.get_logger().warn("BOUNDARY_FOLLOWING: Perdi a parede. Voltando para MOTION_TO_GOAL.")
            self.state = 'MOTION_TO_GOAL'
            self.reset_bug_vars()
            return

        dist_to_goal_current = np.linalg.norm(self.active_goal - self.current_world_pos)
        
        delta_goal = self.active_goal - self.current_world_pos
        angle_to_goal_world = math.atan2(delta_goal[1], delta_goal[0])
        angle_to_goal_relative = self.normalize_angle(angle_to_goal_world - self.current_orientation)

        is_closer_than_hit_point = dist_to_goal_current < (self.distance_at_hit_point - LEAVE_BOUNDARY_TOLERANCE)
        
        can_leave_boundary = False
        if is_closer_than_hit_point:
            if self.is_corridor_to_goal_clear(angle_to_goal_relative, clearance_dist=dist_to_goal_current):
                if (self.wall_follow_direction == 1 and angle_to_goal_relative > -0.1) or \
                   (self.wall_follow_direction == -1 and angle_to_goal_relative < 0.1):
                    self.get_logger().info(f"BOUNDARY_FOLLOWING: Condicao de saida alcancada. DistGoal: {dist_to_goal_current:.2f} < HitDist: {self.distance_at_hit_point:.2f}. Caminho para o alvo esta livre.")
                    can_leave_boundary = True
        
        if can_leave_boundary:
            self.state = 'MOTION_TO_GOAL'
            self.reset_bug_vars()
            return
        
        target_linear_speed = LINEAR_SPEED_WALL
        
        min_dist_dead_ahead = self.get_min_dist_in_cone(-math.pi/12, math.pi/12)
        if min_dist_dead_ahead < WALL_FOLLOW_DISTANCE * 0.8:
            target_linear_speed *= 0.3
        elif min_dist_dead_ahead < WALL_FOLLOW_DISTANCE * 1.2:
             target_linear_speed *= 0.7

        error_dist = min_dist_wall - WALL_FOLLOW_DISTANCE
        target_angle_to_wall_point_rad = (math.pi / 2.0) * self.wall_follow_direction
        error_angle_to_wall_orientation = self.normalize_angle(angle_to_wall_obstacle_rad - target_angle_to_wall_point_rad)
        
        twist = Twist()
        twist.linear.x = float(target_linear_speed)
        
        angular_vel_z = (ANGULAR_GAIN_WALL_DIST * -error_dist) + (ANGULAR_GAIN_WALL_ALIGN * error_angle_to_wall_orientation)
        twist.angular.z = float(np.clip(angular_vel_z, -1.2, 1.2))
        
        self.vel_pub.publish(twist)

    def reset_bug_vars(self):
        self.distance_at_hit_point = float('inf')
        self.m_line_start_point = None

    def is_corridor_to_goal_clear(self, rel_angle_to_goal, clearance_dist=LEAVE_PATH_CLEARANCE):
        if not self.laser_scan: return False
        
        corridor_angular_width = math.atan2(ROBOT_WIDTH, clearance_dist) * 2.0
        corridor_angular_width = max(corridor_angular_width, math.radians(15.0))

        start_angle = self.normalize_angle(rel_angle_to_goal - corridor_angular_width / 2.0)
        end_angle = self.normalize_angle(rel_angle_to_goal + corridor_angular_width / 2.0)
        
        min_dist_in_corridor = self.get_min_dist_in_cone(start_angle, end_angle)

        if min_dist_in_corridor < clearance_dist :
            return False
        
        return True
    
    def get_indices_in_range(self, start_idx, end_idx):
        indices = []
        num_ranges = len(self.laser_scan.ranges)
        start_idx = max(0, min(num_ranges - 1, start_idx))
        end_idx = max(0, min(num_ranges - 1, end_idx))
        if start_idx <= end_idx:
            indices.extend(range(start_idx, end_idx + 1))
        else:
            indices.extend(range(start_idx, num_ranges))
            indices.extend(range(0, end_idx + 1))
        return indices

    def get_min_dist_and_angle_in_cone(self, start_angle_rel, end_angle_rel):
        if not self.laser_scan: return MAX_LIDAR_RANGE, 0.0
        
        start_idx = self.get_index_from_angle(start_angle_rel)
        end_idx = self.get_index_from_angle(end_angle_rel)
        indices = self.get_indices_in_range(start_idx, end_idx)
        
        if not indices: return MAX_LIDAR_RANGE, 0.0
        
        min_dist = MAX_LIDAR_RANGE
        best_index = -1
        
        for i in indices:
            if self.laser_scan.ranges[i] < min_dist:
                min_dist = self.laser_scan.ranges[i]
                best_index = i
        
        if best_index != -1:
            return min_dist, self.get_angle_from_index(best_index)
            
        return MAX_LIDAR_RANGE, 0.0

    def get_min_dist_in_cone(self, start_angle_rel, end_angle_rel):
        min_dist, _ = self.get_min_dist_and_angle_in_cone(start_angle_rel, end_angle_rel)
        return min_dist

    def find_wall_to_follow(self):
        if not self.laser_scan: return None, None

        search_cone_width_rad = math.pi / 2.0
        center_angle_rad = (math.pi / 2.0) * self.wall_follow_direction
        
        start_angle_search_rad = self.normalize_angle(center_angle_rad - search_cone_width_rad / 2.0)
        end_angle_search_rad = self.normalize_angle(center_angle_rad + search_cone_width_rad / 2.0)
        
        min_dist_found, angle_at_min_dist_rad = self.get_min_dist_and_angle_in_cone(
            start_angle_search_rad, end_angle_search_rad
        )
        
        if min_dist_found >= MAX_LIDAR_RANGE - 0.1 :
            return None, None 
        
        return min_dist_found, angle_at_min_dist_rad

    def stop_robot(self):
        self.vel_pub.publish(Twist())

    def euler_from_quaternion(self, q_geom_msg):
        """
        Esta é a função corrigida que usa os índices da lista para os cálculos.
        """
        q = [q_geom_msg.x, q_geom_msg.y, q_geom_msg.z, q_geom_msg.w]
        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return 0.0, 0.0, yaw # Roll, Pitch, Yaw

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def get_index_from_angle(self, angle_rel_robot_rad):
        if not self.laser_scan or self.laser_scan.angle_increment == 0:
            return 0
            
        index = (angle_rel_robot_rad - self.laser_scan.angle_min) / self.laser_scan.angle_increment
        num_ranges = len(self.laser_scan.ranges)
        return int(round(max(0, min(num_ranges - 1, index))))

    def get_angle_from_index(self, index):
        if not self.laser_scan: return 0.0
        angle = self.laser_scan.angle_min + index * self.laser_scan.angle_increment
        return self.normalize_angle(angle)


def main(args=None):
    rclpy.init(args=args)
    navigator = TangentBugNavigator()
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info("KeyboardInterrupt recebido.")
    except Exception as e:
        navigator.get_logger().error(f"Erro nao tratado durante o spin: {e}")
        import traceback
        navigator.get_logger().error(traceback.format_exc())
    finally:
        navigator.get_logger().info("Parando o robo e encerrando o no.")
        navigator.stop_robot()
        navigator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()