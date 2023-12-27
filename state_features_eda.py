import gymnasium as gym
# from gym.utils import play
import pygame
import numpy as np

def main():

    env = gym.make("LunarLander-v2", render_mode="human")    
    quit, restart = False, False
    
    def register_input(a):
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # print(event.key)
                if event.key == pygame.K_q:
                    # print("Press Q")
                    a = 1
                if event.key == pygame.K_w:
                    a = 2
                if event.key == pygame.K_e:
                    a = 3
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_q:
                    a = 0
                if event.key == pygame.K_w:
                    a = 0
                if event.key == pygame.K_e:
                    a = 0

            if event.type == pygame.QUIT:
                quit = True
        return a
    
    # env = CarRacing(render_mode="human")
    a = 0
    quit = False
    
    with open('state_vectors.txt', 'a') as f:
        while not quit:
            env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                a = register_input(a)
                # print("a is " + str(a))
                s, r, terminated, truncated, info = env.step(a)
                # print(s,r,terminated,truncated,info)
                total_reward += r
                # if steps % 200 == 0 or terminated or truncated:
                    # print("\naction " + str([f"{x:+0.2f}" for x in a]))
                    # print(f"step {steps} total_reward {total_reward:+0.2f}")
                    # np.array2string
                to_write = ' '.join(map(str, s.flatten()))
                f.write(str(to_write))
                f.write('\n')
                steps += 1
                if terminated or truncated or restart or quit:
                    break
            print(total_reward)
    env.close()
    
if __name__ == "__main__":
    main()
            
