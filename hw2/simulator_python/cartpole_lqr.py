import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pygame
from scipy.linalg import solve_discrete_are


@dataclass
class CartPoleConfig:
    m_cart: float = 1.0
    m_pole: float = 0.1
    pole_half_length: float = 0.5
    gravity: float = 9.81

    dt: float = 0.02
    force_limit: float = 10.0

    x_threshold: float = 2.4
    theta_threshold_radians: float = 12.0 * math.pi / 180.0
    alive_reward: float = 1.0


class CartPoleSim:
    def __init__(self, config: Optional[CartPoleConfig] = None, seed: Optional[int] = None):
        self.cfg = config or CartPoleConfig()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float64)   # [x, x_dot, theta, theta_dot]
        self.time = 0.0

    def reset(self, noise_scale: float = 0.01) -> np.ndarray:
        self.state = self.rng.uniform(-noise_scale, noise_scale, size=4).astype(np.float64)
        self.time = 0.0
        return self.state.copy()

    def set_state(self, x: float, x_dot: float, theta: float, theta_dot: float) -> None:
        self.state[:] = [x, x_dot, theta, theta_dot]

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def _clip_force(self, u: float) -> float:
        return float(np.clip(u, -self.cfg.force_limit, self.cfg.force_limit))

    def dynamics(self, state: np.ndarray, action: float) -> np.ndarray:
        x, x_dot, theta, theta_dot = state
        force = self._clip_force(action)

        m_c = self.cfg.m_cart
        m_p = self.cfg.m_pole
        total_mass = m_c + m_p
        l = self.cfg.pole_half_length
        g = self.cfg.gravity

        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        temp = (force + m_p * l * theta_dot**2 * sin_theta) / total_mass
        theta_ddot = (g * sin_theta - cos_theta * temp) / (
            l * (4.0 / 3.0 - (m_p * cos_theta**2) / total_mass)
        )
        x_ddot = temp - (m_p * l * theta_ddot * cos_theta) / total_mass

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float64)

    def _euler_step(self, state: np.ndarray, action: float, dt: float) -> np.ndarray:
        return state + dt * self.dynamics(state, action)

    def is_terminated(self, state: Optional[np.ndarray] = None) -> bool:
        s = self.state if state is None else state
        x, _, theta, _ = s
        return (
            abs(x) > self.cfg.x_threshold
            or abs(theta) > self.cfg.theta_threshold_radians
        )

    def reward(self, state: Optional[np.ndarray] = None) -> float:
        return 0.0 if self.is_terminated(state) else self.cfg.alive_reward

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        self.state = self._euler_step(self.state, action, self.cfg.dt)
        self.time += self.cfg.dt
        done = self.is_terminated(self.state)
        reward = self.reward(self.state)
        info = {"time": self.time}
        return self.state.copy(), reward, done, info


class LQRController:
    """
    LQR around the upright equilibrium:
        x = 0, x_dot = 0, theta = 0, theta_dot = 0

    State order:
        [x, x_dot, theta, theta_dot]

    This controller is designed from the linearized model, but is applied
    to the nonlinear simulator.
    """
    def __init__(self, cfg: CartPoleConfig):
        self.cfg = cfg
        self.K = self._build_lqr_gain()

    def _linearized_matrices_continuous(self) -> Tuple[np.ndarray, np.ndarray]:
        m_c = self.cfg.m_cart
        m_p = self.cfg.m_pole
        l = self.cfg.pole_half_length
        g = self.cfg.gravity

        denom = 4.0 * m_c + m_p

        # Linearization of the given nonlinear equations around upright:
        #
        # x_ddot     = -(3 m_p g / denom) * theta + (4 / denom) * u
        # theta_ddot =  (3 (m_c + m_p) g / (l denom)) * theta - (3 / (l denom)) * u
        #
        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -(3.0 * m_p * g) / denom, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (3.0 * (m_c + m_p) * g) / (l * denom), 0.0],
        ], dtype=np.float64)

        B = np.array([
            [0.0],
            [4.0 / denom],
            [0.0],
            [-3.0 / (l * denom)],
        ], dtype=np.float64)

        return A, B

    def _build_lqr_gain(self) -> np.ndarray:
        A_c, B_c = self._linearized_matrices_continuous()

        # Simple discretization consistent with the Euler simulator
        dt = self.cfg.dt
        A_d = np.eye(4) + dt * A_c
        B_d = dt * B_c

        # Tune these as needed
        Q = np.diag([1.0, 1.0, 25.0, 3.0])
        R = np.array([[0.08]])

        P = solve_discrete_are(A_d, B_d, Q, R)
        K = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)
        return K

    def __call__(self, state: np.ndarray) -> float:
      u = -np.dot(self.K, state).item()
      return float(np.clip(u, -self.cfg.force_limit, self.cfg.force_limit))

    # def __call__(self, state: np.ndarray) -> float:
    #     u = -float(self.K @ state.reshape(-1, 1))
    #     return float(np.clip(u, -self.cfg.force_limit, self.cfg.force_limit))


class CartPoleViewer:
    def __init__(self, sim: CartPoleSim, controller: LQRController, width: int = 1000, height: int = 500):
        self.sim = sim
        self.controller = controller
        self.width = width
        self.height = height
        self.world_width = 2 * (sim.cfg.x_threshold + 0.8)

        pygame.init()
        pygame.display.set_caption("CartPole LQR Control")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 20)
        self.big_font = pygame.font.SysFont("Menlo", 28)

    def close(self):
        pygame.quit()

    def world_to_screen_x(self, x: float) -> int:
        return int((x / self.world_width + 0.5) * self.width)

    def handle_events(self, sim: CartPoleSim) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    sim.reset()
                    sim.set_state(x=0.0, x_dot=0.0, theta=0.08, theta_dot=0.0)
        return True

    def draw(self, action: float, reward: float, done: bool):
        screen = self.screen
        screen.fill((245, 247, 250))

        ground_y = int(self.height * 0.72)
        pygame.draw.line(screen, (60, 60, 60), (0, ground_y), (self.width, ground_y), 3)

        x, x_dot, theta, theta_dot = self.sim.get_state()

        cart_w = 100
        cart_h = 50
        cart_x = self.world_to_screen_x(x)
        cart_y = ground_y - cart_h // 2

        pole_len_px = int(2.0 * self.sim.cfg.pole_half_length / self.world_width * self.width)
        pivot_x = cart_x
        pivot_y = cart_y - cart_h // 3

        pole_end_x = pivot_x + int(pole_len_px * math.sin(theta))
        pole_end_y = pivot_y - int(pole_len_px * math.cos(theta))

        left_thr = self.world_to_screen_x(-self.sim.cfg.x_threshold)
        right_thr = self.world_to_screen_x(self.sim.cfg.x_threshold)
        pygame.draw.line(screen, (220, 80, 80), (left_thr, 40), (left_thr, ground_y + 30), 2)
        pygame.draw.line(screen, (220, 80, 80), (right_thr, 40), (right_thr, ground_y + 30), 2)

        cart_rect = pygame.Rect(0, 0, cart_w, cart_h)
        cart_rect.center = (cart_x, cart_y)
        pygame.draw.rect(screen, (60, 120, 220), cart_rect, border_radius=10)
        pygame.draw.rect(screen, (20, 40, 90), cart_rect, width=2, border_radius=10)

        pygame.draw.circle(screen, (40, 40, 40), (cart_x - 25, ground_y + 5), 10)
        pygame.draw.circle(screen, (40, 40, 40), (cart_x + 25, ground_y + 5), 10)

        pygame.draw.line(screen, (220, 70, 70), (pivot_x, pivot_y), (pole_end_x, pole_end_y), 10)
        pygame.draw.circle(screen, (30, 30, 30), (pivot_x, pivot_y), 8)
        pygame.draw.circle(screen, (220, 120, 50), (pole_end_x, pole_end_y), 12)

        arrow_len = int(np.clip(action * 8.0, -120, 120))
        if arrow_len != 0:
            start = (cart_x, cart_y + 80)
            end = (cart_x + arrow_len, cart_y + 80)
            pygame.draw.line(screen, (20, 160, 20), start, end, 5)
            pygame.draw.circle(screen, (20, 160, 20), end, 7)

        k_text = np.array2string(self.controller.K, precision=3, suppress_small=True)

        lines = [
            "Controls: R to reset, close window to quit",
            "Controller: LQR",
            f"time: {self.sim.time:6.2f}s",
            f"x: {x:7.3f}",
            f"x_dot: {x_dot:7.3f}",
            f"theta: {theta:7.3f} rad ({math.degrees(theta):6.2f} deg)",
            f"theta_dot: {theta_dot:7.3f}",
            f"action: {action:7.3f}",
            f"reward: {reward:5.2f}",
            f"K: {k_text}",
        ]
        for i, txt in enumerate(lines):
            surf = self.font.render(txt, True, (20, 20, 20))
            screen.blit(surf, (20, 20 + i * 28))

        status = "FAILED" if done else "RUNNING"
        color = (200, 40, 40) if done else (20, 140, 40)
        surf = self.big_font.render(status, True, color)
        screen.blit(surf, (self.width - 180, 20))

        pygame.display.flip()

    def tick(self):
        self.clock.tick(int(round(1.0 / self.sim.cfg.dt)))


if __name__ == "__main__":
    sim = CartPoleSim(seed=0)
    sim.reset()
    sim.set_state(x=0.0, x_dot=0.0, theta=0.08, theta_dot=0.0)

    controller = LQRController(sim.cfg)
    print("LQR gain K =", controller.K)

    viewer = CartPoleViewer(sim, controller)
    running = True
    total_reward = 0.0

    try:
        while running:
            running = viewer.handle_events(sim)

            if sim.is_terminated():
                action = 0.0
                reward = 0.0
                done = True
            else:
                state = sim.get_state()
                action = controller(state)
                _, reward, done, _ = sim.step(action)
                total_reward += reward

            viewer.draw(action=action, reward=reward, done=done)
            viewer.tick()
    finally:
        viewer.close()