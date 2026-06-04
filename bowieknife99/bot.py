import math
from typing import override

import rlbot_flatbuffers
from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

from vec import Vec3


class Orientation:
    yaw: float
    roll: float
    pitch: float

    forward: Vec3
    right: Vec3
    up: Vec3

    def __init__(self, rotation):
        self.pitch = float(rotation.pitch)
        self.yaw = float(rotation.yaw)
        self.roll = float(rotation.roll)

        cr = math.cos(self.roll)
        sr = math.sin(self.roll)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        self.forward = Vec3(cp * cy, cp * sy, sp)
        self.right = Vec3(cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr)
        self.up = Vec3(-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr)


def world_to_local(car_pos: Vec3, orientation: Orientation, target: Vec3) -> Vec3:
    relative = target - car_pos
    return Vec3(
        relative.dot(orientation.forward),
        relative.dot(orientation.right),
        relative.dot(orientation.up),
    )


class FlipState:
    def __init__(self):
        self.active = False
        self.timer = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def start(self, pitch: float, yaw: float):
        self.active = True
        self.timer = 0.0
        self.pitch = pitch
        self.yaw = yaw

    def update(self, dt: float, controller: ControllerState):
        self.timer += dt
        if self.timer < 0.12:
            controller.jump = True
            controller.throttle = 1.0
        elif self.timer < 0.16:
            controller.jump = False
            controller.throttle = 1.0
        elif self.timer < 0.35:
            controller.jump = True
            controller.pitch = self.pitch
            controller.yaw = self.yaw
            controller.throttle = 1.0
        elif self.timer < 0.55:
            controller.jump = False
            controller.pitch = self.pitch
            controller.yaw = self.yaw
            controller.throttle = 1.0
        elif self.timer < 0.75:
            controller.jump = False
            controller.pitch = self.pitch
            controller.yaw = self.yaw
            controller.throttle = 1.0
            controller.handbrake = True
        else:
            self.active = False


class MyBot(Bot):
    @override
    def initialize(self):
        self.target_id = None
        self.flip_state = FlipState()
        self.prev_time = None
        self.stuck_timer = 0.0
        self.aerial_timer = 0.0

    def clamp_to_field(self, pos: Vec3) -> Vec3:
        x = max(-4000.0, min(4000.0, pos.x))
        y = max(-5100.0, min(5100.0, pos.y))
        z = max(17.0, min(2000.0, pos.z))
        return Vec3(x, y, z)

    def find_closest_boost(self, my_pos: Vec3, packet: GamePacket) -> Vec3 | None:
        if not self.field_info or not self.field_info.boost_pads:
            return None

        closest_boost = None
        min_dist = float("inf")

        for i, pad_state in enumerate(packet.boost_pads):
            if pad_state.is_active:
                static_pad = self.field_info.boost_pads[i]
                if static_pad.is_full_boost:
                    pad_pos = Vec3(static_pad.location)
                    dist = my_pos.dist(pad_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_boost = pad_pos

        return closest_boost

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        controller = ControllerState()

        current_time = packet.match_info.seconds_elapsed
        dt = current_time - self.prev_time if self.prev_time is not None else 0.016
        self.prev_time = current_time

        if self.index >= len(packet.players):
            return controller

        my_car = packet.players[self.index]
        my_pos = Vec3(my_car.physics.location)
        my_vel = Vec3(my_car.physics.velocity)
        my_speed = my_vel.length()

        if self.flip_state.active:
            self.flip_state.update(dt, controller)
            return controller

        if my_speed < 50.0 and my_car.air_state == rlbot_flatbuffers.AirState.OnGround:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0

        if self.stuck_timer > 0.75:
            controller.throttle = -1.0
            controller.steer = -1.0
            controller.handbrake = False
            controller.boost = False
            if self.stuck_timer > 1.5:
                self.stuck_timer = 0.0
            return controller

        target_car = None
        enemies = []

        for i, player in enumerate(packet.players):
            if i != self.index and player.team != my_car.team:
                if player.demolished_timeout < 0:
                    enemies.append((i, player))

        if not enemies:
            controller.jump = (
                True
                if my_car.air_state == rlbot_flatbuffers.AirState.OnGround
                else False
            )
            return controller

        if self.target_id is not None:
            matching = [p for i, p in enemies if i == self.target_id]
            if matching:
                target_car = matching[0]

        if target_car is None:
            best_dist = float("inf")
            best_idx = None
            for idx, p in enemies:
                dist = my_pos.dist(Vec3(p.physics.location))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                self.target_id = best_idx
                target_car = packet.players[best_idx]

        if target_car is None:
            # Fallback
            self.target_id = None
            controller.throttle = 1.0
            return controller

        enemy_pos = Vec3(target_car.physics.location)
        enemy_vel = Vec3(target_car.physics.velocity)
        dist_to_enemy = my_pos.dist(enemy_pos)

        # Estimated Time Arrival
        travel_speed = max(1400.0, my_speed)
        t_intercept = dist_to_enemy / travel_speed

        predicted_pos = enemy_pos + enemy_vel * t_intercept
        predicted_pos = self.clamp_to_field(predicted_pos)

        if my_car.boost < 20.0 and dist_to_enemy > 1500.0:
            boost_target = self.find_closest_boost(my_pos, packet)
            if boost_target is not None:
                predicted_pos = boost_target

        orientation = Orientation(my_car.physics.rotation)
        target_local = world_to_local(my_pos, orientation, predicted_pos)

        steer_angle = math.atan2(target_local.y, target_local.x)

        if my_car.air_state == rlbot_flatbuffers.AirState.OnGround:
            controller.throttle = 1.0
            controller.steer = max(-1.0, min(1.0, steer_angle * 3.0))

            # Handbrake
            if abs(steer_angle) > 1.2:
                controller.handbrake = True

            if abs(steer_angle) < 0.4 and not my_car.is_supersonic and my_car.boost > 0:
                controller.boost = True

            if (
                dist_to_enemy > 2100.0
                and not my_car.is_supersonic
                and abs(steer_angle) < 0.08
            ):
                self.flip_state.start(pitch=-1.0, yaw=0.0)
                self.flip_state.update(dt, controller)
                return controller

            elif dist_to_enemy < 250.0 and abs(steer_angle) < 0.1:
                denom = math.sqrt(target_local.x**2 + target_local.y**2) + 1e-5
                dodge_x = target_local.x / denom
                dodge_y = target_local.y / denom
                self.flip_state.start(
                    pitch=max(-1.0, min(1.0, -dodge_x)),
                    yaw=max(-1.0, min(1.0, dodge_y)),
                )
                self.flip_state.update(dt, controller)
                return controller
            elif (
                dist_to_enemy < 450.0
                and abs(steer_angle) < 0.1
                and (enemy_pos.z > my_pos.z + 100.0)
            ):
                denom = math.sqrt(target_local.x**2 + target_local.y**2) + 1e-5
                dodge_x = target_local.x / denom
                dodge_y = target_local.y / denom
                self.flip_state.start(
                    pitch=max(-1.0, min(1.0, -dodge_x)),
                    yaw=max(-1.0, min(1.0, dodge_y)),
                )
                self.flip_state.update(dt, controller)
                return controller

        else:
            controller.throttle = 1.0
            controller.pitch = max(-1.0, min(1.0, target_local.z * 0.01))
            controller.yaw = max(-1.0, min(1.0, target_local.y * 0.01))

            if target_local.x > 0 and abs(steer_angle) < 0.4:
                if my_car.boost > 0:
                    controller.boost = True

        return controller


if __name__ == "__main__":
    MyBot("martico2432/bowieknife99").run()
