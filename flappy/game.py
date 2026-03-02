import pygame
import random
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
# All tunable values live here. For progressive difficulty, mutate these values
# as score increases (e.g. increase PIPE_SPEED, decrease PIPE_GAP) without
# touching any game logic.
CONFIG = {
    "SCREEN_WIDTH":   288,
    "SCREEN_HEIGHT":  512,
    "FPS":            60,
    "GRAVITY":        0.25,
    "FLAP_VELOCITY":  -7,
    "PIPE_SPEED":     3,
    "PIPE_GAP":       150,      # vertical gap between top and bottom pipe
    "PIPE_INTERVAL":  1500,     # ms between pipe spawns
    "GROUND_HEIGHT":  80,
    "BIRD_WIDTH":     34,
    "BIRD_HEIGHT":    24,
    "PIPE_WIDTH":     52,
}

# ── COLORS ────────────────────────────────────────────────────────────────────
SKY_BLUE    = (113, 197, 207)
GROUND_TAN  = (222, 216, 149)
PIPE_GREEN  = ( 83, 156,  77)
PIPE_CAP    = ( 60, 120,  55)
BIRD_YELLOW = (255, 213,   0)
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)
RED         = (200,  50,  50)

# ── GAME STATES ───────────────────────────────────────────────────────────────
WAITING   = "waiting"
PLAYING   = "playing"
GAME_OVER = "game_over"


class Bird:
    def __init__(self, x, y):
        self.x        = x
        self.y        = float(y)
        self.velocity = 0.0
        self.width    = CONFIG["BIRD_WIDTH"]
        self.height   = CONFIG["BIRD_HEIGHT"]

    def flap(self):
        self.velocity = CONFIG["FLAP_VELOCITY"]

    def update(self):
        self.velocity += CONFIG["GRAVITY"]
        self.y += self.velocity

    def get_rect(self):
        return pygame.Rect(self.x, int(self.y), self.width, self.height)

    def draw(self, surface):
        rect = self.get_rect()
        pygame.draw.rect(surface, BIRD_YELLOW, rect)
        # simple eye detail
        eye_x = rect.right - 8
        eye_y = rect.top + 6
        pygame.draw.circle(surface, BLACK, (eye_x, eye_y), 3)
        pygame.draw.circle(surface, WHITE, (eye_x - 1, eye_y - 1), 1)


class PipePair:
    def __init__(self, x, screen_height, ground_height):
        self.x     = float(x)
        self.width = CONFIG["PIPE_WIDTH"]
        self.gap   = CONFIG["PIPE_GAP"]
        self.scored = False

        play_area_height = screen_height - ground_height
        min_top = 50
        max_top = play_area_height - self.gap - 50
        top_height = random.randint(min_top, max_top)

        self.top_rect = pygame.Rect(int(self.x), 0, self.width, top_height)

        bottom_y = top_height + self.gap
        self.bottom_rect = pygame.Rect(
            int(self.x),
            bottom_y,
            self.width,
            play_area_height - bottom_y,
        )

    def update(self):
        self.x -= CONFIG["PIPE_SPEED"]
        self.top_rect.x    = int(self.x)
        self.bottom_rect.x = int(self.x)

    def is_off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, bird_rect):
        return (self.top_rect.colliderect(bird_rect) or
                self.bottom_rect.colliderect(bird_rect))

    def draw(self, surface):
        pygame.draw.rect(surface, PIPE_GREEN, self.top_rect)
        pygame.draw.rect(surface, PIPE_GREEN, self.bottom_rect)

        # pipe caps (slightly darker, slightly wider)
        cap_h   = 12
        cap_ext = 3
        top_cap = pygame.Rect(
            self.top_rect.x - cap_ext,
            self.top_rect.bottom - cap_h,
            self.width + cap_ext * 2,
            cap_h,
        )
        bot_cap = pygame.Rect(
            self.bottom_rect.x - cap_ext,
            self.bottom_rect.top,
            self.width + cap_ext * 2,
            cap_h,
        )
        pygame.draw.rect(surface, PIPE_CAP, top_cap)
        pygame.draw.rect(surface, PIPE_CAP, bot_cap)


class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.width        = CONFIG["SCREEN_WIDTH"]
        self.height       = CONFIG["SCREEN_HEIGHT"]
        self.ground_height = CONFIG["GROUND_HEIGHT"]

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()

        self.font_big   = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 22)

        self.reset()

    # ── Public API (designed for easy gymnasium wrapping later) ───────────────

    def reset(self):
        """Reset game to initial state. Returns first observation."""
        self.bird       = Bird(self.width // 4, self.height // 2)
        self.pipes      = []
        self.score      = 0
        self.state      = WAITING
        self.pipe_timer = 0

    def step(self, action: bool, dt: int):
        """
        Advance game by one frame.
        action: True = flap, False = do nothing
        dt:     milliseconds since last frame
        Returns: (game_state, score, terminated)
        """
        if action and self.state == PLAYING:
            self.bird.flap()
        self._update(dt)
        terminated = self.state == GAME_OVER
        return self.state, self.score, terminated

    def get_observation(self):
        """
        Returns a compact state dict — useful for RL feature engineering later.
        """
        next_pipe = self._get_next_pipe()
        if next_pipe:
            pipe_x        = next_pipe.x
            pipe_top_y    = next_pipe.top_rect.bottom
            pipe_bottom_y = next_pipe.bottom_rect.top
        else:
            pipe_x        = self.width
            pipe_top_y    = 0
            pipe_bottom_y = self.height - self.ground_height

        return {
            "bird_y":        self.bird.y,
            "bird_velocity": self.bird.velocity,
            "pipe_x":        pipe_x,
            "pipe_top_y":    pipe_top_y,
            "pipe_bottom_y": pipe_bottom_y,
            "score":         self.score,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_next_pipe(self):
        """Return the next pipe the bird hasn't passed yet."""
        bird_x = self.bird.x
        ahead = [p for p in self.pipes if p.x + p.width > bird_x]
        return min(ahead, key=lambda p: p.x) if ahead else None

    def _spawn_pipe(self):
        self.pipes.append(PipePair(self.width + 10, self.height, self.ground_height))

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    if self.state == WAITING:
                        self.state = PLAYING
                        self.bird.flap()
                    elif self.state == PLAYING:
                        self.bird.flap()
                    elif self.state == GAME_OVER:
                        self.reset()

    def _update(self, dt):
        if self.state != PLAYING:
            return

        self.bird.update()

        # boundary collisions
        ground_y = self.height - self.ground_height
        if self.bird.y + self.bird.height >= ground_y or self.bird.y <= 0:
            self.state = GAME_OVER
            return

        # pipe spawning
        self.pipe_timer += dt
        if self.pipe_timer >= CONFIG["PIPE_INTERVAL"]:
            self._spawn_pipe()
            self.pipe_timer = 0

        # pipe update, collision, scoring
        bird_rect = self.bird.get_rect()
        for pipe in self.pipes:
            pipe.update()
            if pipe.collides_with(bird_rect):
                self.state = GAME_OVER
                return
            pipe_center_x = pipe.x + pipe.width / 2
            if not pipe.scored and pipe_center_x < self.bird.x:
                pipe.scored = True
                self.score += 1

        self.pipes = [p for p in self.pipes if not p.is_off_screen()]

    def _draw(self):
        # sky
        self.screen.fill(SKY_BLUE)

        # pipes (behind bird)
        for pipe in self.pipes:
            pipe.draw(self.screen)

        # ground
        ground_rect = pygame.Rect(
            0, self.height - self.ground_height,
            self.width, self.ground_height,
        )
        pygame.draw.rect(self.screen, GROUND_TAN, ground_rect)

        # bird
        self.bird.draw(self.screen)

        # score
        score_surf = self.font_big.render(str(self.score), True, WHITE)
        self.screen.blit(score_surf, score_surf.get_rect(centerx=self.width // 2, top=20))

        if self.state == WAITING:
            hint = self.font_small.render("Press SPACE to start", True, WHITE)
            self.screen.blit(hint, hint.get_rect(centerx=self.width // 2, centery=self.height // 2 + 60))

        elif self.state == GAME_OVER:
            self._draw_game_over()

        pygame.display.flip()

    def _draw_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        panel = pygame.Rect(self.width // 2 - 105, self.height // 2 - 85, 210, 170)
        pygame.draw.rect(self.screen, (240, 230, 200), panel, border_radius=10)
        pygame.draw.rect(self.screen, BLACK,           panel, 2, border_radius=10)

        over_surf = self.font_big.render("GAME OVER", True, RED)
        self.screen.blit(over_surf, over_surf.get_rect(centerx=self.width // 2, centery=panel.top + 38))

        score_surf = self.font_small.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_surf, score_surf.get_rect(centerx=self.width // 2, centery=panel.top + 85))

        restart_surf = self.font_small.render("Press SPACE to restart", True, BLACK)
        self.screen.blit(restart_surf, restart_surf.get_rect(centerx=self.width // 2, centery=panel.top + 130))

    def run(self):
        while True:
            dt = self.clock.tick(CONFIG["FPS"])
            self._handle_events()
            if self.state == WAITING:
                self._draw()
                continue
            self._update(dt)
            self._draw()


if __name__ == "__main__":
    game = FlappyBirdGame()
    game.run()
