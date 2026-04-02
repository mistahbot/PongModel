import logic
import pygame
import NeuralNetwork
import numpy as np



def normalize_state(game:logic.Game):
    state_dict = game.get_state()
    state = np.array([
        state_dict["pos_y_p1"] / 600,
        state_dict["pos_y_p2"] / 600,
        state_dict["pos_ball_x"] / 600,
        state_dict["pos_ball_y"] / 600,
        state_dict["velocity_ball_x"] / 4,
        state_dict["velocity_ball_y"] / 4
    ]).reshape(6, 1)
    return state


def normalize_state_mirror(game:logic.Game):
    state_dict = game.get_state()
    state = np.array([
        state_dict["pos_y_p2"] / 600,
        state_dict["pos_y_p1"] / 600,
        (logic.SCREEN_SIZE[1] - state_dict["pos_ball_x"]) / 600,
        state_dict["pos_ball_y"] / 600,
        -state_dict["velocity_ball_x"] / 4,
        state_dict["velocity_ball_y"] / 4
    ]).reshape(6, 1)
    return state



pygame.init()
screen = pygame.display.set_mode(logic.SCREEN_SIZE)
clock = pygame.time.Clock()
running = True
font = pygame.font.Font(None, 50)

nn1 = NeuralNetwork.NeuralNetwork(6, 64, 64, 3)
nn1.load_state("model_10k_episodes.json")

nn2 = NeuralNetwork.NeuralNetwork(6, 64, 64, 3)
nn2.load_state("model_10k_episodes.json")

game = logic.Game()

while running:
    game.clock += 1
    if game.clock // 60 == game.time_limit:
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # moving the players

    state = normalize_state(game)
    action = np.argmax(nn2.Predict(state))
    move_p2 = 0
    if action == 1:
        move_p2 = -1
    elif action == 2:
        move_p2 = 1
    game.pos_y_p2 += logic.MOVE * move_p2


    state = normalize_state_mirror(game)
    action = np.argmax(nn1.Predict(state))
    move_p1 = 0
    if action == 1:
        move_p1 = -1
    elif action == 2:
        move_p1 = 1

    game.pos_y_p1 += logic.MOVE * move_p1

#    keys = pygame.key.get_pressed()
#    if keys[pygame.K_w]:
#        game.pos_y_p1 -= logic.MOVE
#    if keys[pygame.K_s]:
#        game.pos_y_p1 += logic.MOVE
#   if keys[pygame.K_i]:
#        game.pos_y_p2 -= logic.MOVE
#    if keys[pygame.K_k]:
#        game.pos_y_p2 += logic.MOVE
    game.snap_back_to_screen()

    # moving the ball
    game.pos_x_ball += game.vx_ball
    game.pos_y_ball += game.vy_ball

    # checking collisions

    # ceiling collsion
    if game.pos_y_ball < 0 or game.pos_y_ball + game.dim > logic.SCREEN_SIZE[1]:
        game.change_dir_ball_env_collision()

    # wall collision
    if game.pos_x_ball < 0:
        game.update_score_p2()
        game.reset_ball()
        
    if game.pos_x_ball + game.dim > logic.SCREEN_SIZE[0]:
        game.update_score_p1()
        game.reset_ball()

    game.check_player_collision()
        
    # drawing
    screen.fill("black")

    text_surface = font.render(game.get_score(), True, (255, 255, 255))
    screen.blit(text_surface, (logic.SCREEN_SIZE[0] // 2 - 30, 0))

    pygame.draw.rect(screen, (255, 255, 255), [game.pos_x_p1, game.pos_y_p1, game.dim_x, game.dim_y])
    pygame.draw.rect(screen, (255, 255, 255), [game.pos_x_p2, game.pos_y_p2, game.dim_x, game.dim_y])
    pygame.draw.rect(screen, (255, 255, 255), [game.pos_x_ball, game.pos_y_ball, game.dim, game.dim])

    pygame.display.flip()

    clock.tick(60)

pygame.quit()

