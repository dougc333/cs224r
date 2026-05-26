from manim import *


class PolicyGradientIntro(Scene):
    def construct(self):
        title = Text("Policy Gradients", font_size=54)
        subtitle = Text("Learning by increasing the probability of rewarding actions", font_size=28)
        subtitle.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1.5)

        self.play(FadeOut(title), FadeOut(subtitle))

        # Agent / Environment loop
        agent = Circle(radius=0.7).set_fill(BLUE, opacity=0.35)
        agent_label = Text("Agent\nπθ(a|s)", font_size=26).move_to(agent)

        env = Square(side_length=1.5).set_fill(GREEN, opacity=0.35)
        env_label = Text("Environment", font_size=26).move_to(env)

        agent_group = VGroup(agent, agent_label).shift(LEFT * 3)
        env_group = VGroup(env, env_label).shift(RIGHT * 3)

        arrow_action = Arrow(agent_group.get_right(), env_group.get_left(), buff=0.25)
        arrow_obs = Arrow(env_group.get_left() + DOWN * 0.6, agent_group.get_right() + DOWN * 0.6, buff=0.25)

        action_text = Text("action aₜ", font_size=24).next_to(arrow_action, UP)
        obs_text = Text("state sₜ₊₁, reward rₜ", font_size=24).next_to(arrow_obs, DOWN)

        self.play(FadeIn(agent_group), FadeIn(env_group))
        self.play(GrowArrow(arrow_action), FadeIn(action_text))
        self.play(GrowArrow(arrow_obs), FadeIn(obs_text))
        self.wait(1.5)

        self.play(
            FadeOut(agent_group),
            FadeOut(env_group),
            FadeOut(arrow_action),
            FadeOut(arrow_obs),
            FadeOut(action_text),
            FadeOut(obs_text),
        )

        # Trajectory
        traj_title = Text("A rollout is a trajectory", font_size=40).to_edge(UP)
        traj = MathTex(
            r"\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)"
        )
        reward = MathTex(
            r"R(\tau) = \sum_{t=0}^{T} \gamma^t r_t"
        ).next_to(traj, DOWN, buff=0.7)

        self.play(Write(traj_title))
        self.play(Write(traj))
        self.play(Write(reward))
        self.wait(2)

        self.play(FadeOut(traj_title), FadeOut(traj), FadeOut(reward))

        # Main objective
        obj_title = Text("Goal: maximize expected reward", font_size=40).to_edge(UP)
        objective = MathTex(
            r"J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]"
        )

        self.play(Write(obj_title))
        self.play(Write(objective))
        self.wait(2)

        self.play(objective.animate.shift(UP * 1.0))

        grad = MathTex(
            r"\nabla_\theta J(\theta)",
            r"=",
            r"\mathbb{E}_{\tau \sim \pi_\theta}",
            r"\left[",
            r"R(\tau)",
            r"\nabla_\theta \log \pi_\theta(\tau)",
            r"\right]"
        ).scale(0.85).next_to(objective, DOWN, buff=0.8)

        self.play(Write(grad))
        self.wait(2)

        explanation = Text(
            "Good trajectories push their actions to become more likely.",
            font_size=28
        ).to_edge(DOWN)

        self.play(FadeIn(explanation))
        self.wait(2)

        self.play(FadeOut(obj_title), FadeOut(objective), FadeOut(grad), FadeOut(explanation))

        # Action probabilities
        prob_title = Text("Policy = probability distribution over actions", font_size=38).to_edge(UP)

        bars = VGroup()
        labels = VGroup()
        probs = [0.2, 0.5, 0.3]
        names = ["left", "right", "jump"]

        for i, (p, name) in enumerate(zip(probs, names)):
            bar = Rectangle(width=0.8, height=4 * p)
            bar.set_fill(ORANGE, opacity=0.7)
            bar.move_to(LEFT * 2 + RIGHT * i * 2 + DOWN * (2 - 2 * p))
            label = Text(name, font_size=24).next_to(bar, DOWN)
            prob_label = Text(f"{p:.1f}", font_size=22).next_to(bar, UP)
            bars.add(VGroup(bar, prob_label))
            labels.add(label)

        self.play(Write(prob_title))
        self.play(FadeIn(bars), FadeIn(labels))
        self.wait(1.5)

        # Update probabilities after reward
        new_probs = [0.1, 0.75, 0.15]
        new_bars = VGroup()
        for i, p in enumerate(new_probs):
            bar = Rectangle(width=0.8, height=4 * p)
            bar.set_fill(ORANGE, opacity=0.7)
            bar.move_to(LEFT * 2 + RIGHT * i * 2 + DOWN * (2 - 2 * p))
            prob_label = Text(f"{p:.2f}", font_size=22).next_to(bar, UP)
            new_bars.add(VGroup(bar, prob_label))

        reward_text = Text("Reward was high after action: right", font_size=30).to_edge(DOWN)
        self.play(FadeIn(reward_text))
        self.wait(1)

        self.play(Transform(bars, new_bars))
        self.wait(2)

        self.play(FadeOut(prob_title), FadeOut(bars), FadeOut(labels), FadeOut(reward_text))

        # REINFORCE update
        update_title = Text("REINFORCE update", font_size=42).to_edge(UP)

        update = MathTex(
            r"\theta \leftarrow \theta + \alpha",
            r"R(\tau)",
            r"\nabla_\theta \log \pi_\theta(a_t|s_t)"
        ).scale(0.9)

        note = Text(
            "Increase log-probability of actions from high-reward trajectories.",
            font_size=28
        ).next_to(update, DOWN, buff=0.8)

        self.play(Write(update_title))
        self.play(Write(update))
        self.play(FadeIn(note))
        self.wait(2)

        self.play(FadeOut(update_title), FadeOut(update), FadeOut(note))

        # Baseline
        baseline_title = Text("Variance reduction: subtract a baseline", font_size=38).to_edge(UP)

        baseline_formula = MathTex(
            r"\theta \leftarrow \theta + \alpha",
            r"(R(\tau) - b)",
            r"\nabla_\theta \log \pi_\theta(a_t|s_t)"
        ).scale(0.85)

        baseline_note = Text(
            "If reward is better than expected, reinforce the action.",
            font_size=28
        ).next_to(baseline_formula, DOWN, buff=0.8)

        self.play(Write(baseline_title))
        self.play(
          
          
          (baseline_formula))
        self.play(FadeIn(baseline_note))
        self.wait(2)

        self.play(FadeOut(baseline_title), FadeOut(baseline_formula), FadeOut(baseline_note))

        # Final summary
        summary_title = Text("Policy Gradient Summary", font_size=44).to_edge(UP)

        lines = VGroup(
            Text("1. Sample actions from policy πθ(a|s)", font_size=30),
            Text("2. Run a rollout and compute reward", font_size=30),
            Text("3. Increase probability of good actions", font_size=30),
            Text("4. Decrease probability of bad actions indirectly", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)

        self.play(Write(summary_title))
        self.play(FadeIn(lines))
        self.wait(3)