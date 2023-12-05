# contoh simulasi
import pygame
import sys
pygame.init()
width, height = 400, 300
screen = pygame.display.set_mode((width, height))
bg_color = (0, 162, 92)
x, y = width // 2, height // 2
object_radius = 20
x_speed, y_speed = 1, 1
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    x += x_speed
    y += y_speed
    screen.fill(bg_color)
    pygame.draw.circle(screen, (0, 0, 0), (x, y), object_radius)
    pygame.display.flip()
    pygame.time.delay(10)
